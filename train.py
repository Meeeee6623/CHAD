import torch
import yaml
import os
import time
import numpy as np
from tqdm.auto import tqdm
from transformers import get_scheduler
import math
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
import warnings
from accelerate import Accelerator, DistributedType
from accelerate.utils import set_seed as accelerate_set_seed # Use accelerate's seed setting
from peft import LoraConfig, get_peft_model, TaskType
import wandb  # Add wandb import

# Local imports
from model import HopfieldLlama3Model, create_model_and_load_weights # Function to handle creation+loading
from tokenizer_utils import get_tokenizer
from data_loader import get_dataloader
from utils import save_checkpoint, compute_metrics, init_metrics # Removed set_seed from utils

# Suppress warnings
warnings.filterwarnings("ignore", message=".*Torch was not compiled with flash attention.*")
warnings.filterwarnings("ignore", message=".*does not support gradient checkpointing*")
warnings.filterwarnings("ignore", message=".*Passing IntelMPI environment variable.*") # Common Accelerate info msg

def evaluate(model, eval_dataloader, tokenizer, device, config, accelerator):
    """Evaluates the model on one epoch of eval data."""
    if accelerator.is_local_main_process:
        print("Starting evaluation...")
        init_metrics() # Ensure metrics loaded on main process

    model.eval()
    all_preds_ids_list = []
    all_labels_ids_list = []
    last_eval_doc_id = None # For memory reset during eval

    # --- Max Eval Examples Logic ---
    max_eval_examples = config.get("max_eval_examples", None)
    max_steps = float('inf') # Default to no limit
    if isinstance(max_eval_examples, int) and max_eval_examples > 0:
        eval_batch_size_per_device = config.get('per_device_eval_batch_size', 1) # Get eval batch size
        # Calculate total effective batch size across devices
        total_eval_batch_size = eval_batch_size_per_device * accelerator.num_processes
        if total_eval_batch_size > 0:
             max_steps = math.ceil(max_eval_examples / total_eval_batch_size)
             if accelerator.is_local_main_process:
                 print(f"Evaluation limited to approx {max_eval_examples} examples ({max_steps} steps across {accelerator.num_processes} devices).")
        else:
             if accelerator.is_local_main_process:
                 print("Warning: Could not determine total eval batch size, evaluating all examples.")
             max_eval_examples = None # Disable limit if batch size is zero
    # --- End Max Eval Examples Logic ---


    progress_bar = tqdm(eval_dataloader, desc="Evaluating", leave=False, disable=not accelerator.is_local_main_process)
    for step, batch in enumerate(progress_bar):
        # --- Check Step Limit ---
        if step >= max_steps:
             if accelerator.is_local_main_process: print(f"Reached max evaluation steps ({max_steps}). Stopping evaluation.")
             break
        # --- End Check Step Limit ---

        # Batch is already on the correct device due to accelerator.prepare(eval_dataloader)
        input_ids = batch['input_ids']
        labels = batch['labels']
        doc_ids = batch['doc_ids'] # Available if needed for state management

        # --- Memory Reset Logic for Evaluation (Simpler: Reset per batch) ---
        # A more precise implementation would track doc_id changes *within* the batch if possible/needed.
        # Resetting per batch is safer if batches might contain mixed docs.
        unwrapped_model = accelerator.unwrap_model(model)
        if hasattr(unwrapped_model, 'hopfield_memory') and hasattr(unwrapped_model.hopfield_memory, 'reset_memory'):
             # print("Resetting Hopfield memory for new evaluation batch.")
             unwrapped_model.hopfield_memory.reset_memory()
        # No need to wait_for_everyone here as eval is usually single-process or syncs later

        # --- Prepare Prompts for Generation ---
        # Identify prompt part for generation - run on CPU to avoid OOM during generation
        prompt_end_indices = []
        for i in range(batch['input_ids'].shape[0]):
             first_label_idx = (batch['labels'][i].cpu() != -100).nonzero(as_tuple=True)[0]
             if len(first_label_idx) > 0:
                 prompt_end_indices.append(first_label_idx[0].item())
             else:
                 prompt_end_indices.append(batch['input_ids'].shape[1])

        max_prompt_len = min(batch['input_ids'].shape[1], config['context_length'] - config['max_answer_length'])
        prompts_list = [ids[:min(end_idx, max_prompt_len)] for ids, end_idx in zip(batch['input_ids'], prompt_end_indices)]

        # Generate predictions for the batch
        # Use a simple greedy loop for custom model, handle padding within the loop
        generated_ids_batch = []
        max_new = config['max_answer_length']
        eos_token_id = tokenizer.eos_token_id
        pad_token_id = tokenizer.pad_token_id

        for i in range(len(prompts_list)):
            current_prompt = prompts_list[i].unsqueeze(0) # Add batch dim [1, T_prompt]
            # Get unwrapped model for generation if needed
            model_to_generate = accelerator.unwrap_model(model)
            # Memory state reset logic would go here if implemented based on doc_ids[i]
            # if hasattr(model_to_generate.hopfield_memory, "reset_memory"): model_to_generate.hopfield_memory.reset_memory()

            with torch.no_grad():
                for _ in range(max_new):
                    # Prepare input, handle context length limit
                    input_gen = current_prompt[:, -config['context_length']:]
                    # Generate position IDs for the current input length
                    current_len = input_gen.shape[1]
                    position_ids_gen = torch.arange(current_len, device=accelerator.device).unsqueeze(0)

                    outputs = model_to_generate(input_ids=input_gen, position_ids=position_ids_gen)
                    # Model returns dict with 'logits'
                    next_token_logits = outputs["logits"][:, -1, :]
                    next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)

                    # Append token
                    current_prompt = torch.cat([current_prompt, next_token], dim=1)

                    # Check for EOS
                    if eos_token_id is not None and next_token.item() == eos_token_id:
                        break

            # Extract generated sequence (excluding prompt)
            gen_only_ids = current_prompt[0, prompts_list[i].shape[0]:]
            generated_ids_batch.append(gen_only_ids)


        # Gather predictions and labels across all processes
        # Ensure labels are also gathered correctly (move relevant part to CPU first)
        gathered_preds = accelerator.gather_for_metrics(generated_ids_batch) # Gathers lists of tensors
        gathered_labels = accelerator.gather_for_metrics([l.cpu() for l in labels]) # Gather labels from CPU copies

        all_preds_ids_list.extend(gathered_preds)
        all_labels_ids_list.extend(gathered_labels)

    # Compute metrics on the main process after gathering
    metrics = {}
    if accelerator.is_local_main_process:
        if not all_preds_ids_list or not all_labels_ids_list:
            print("Warning: No predictions or labels gathered for metric computation.")
        else:
            # Pad gathered predictions for metric computation if lengths vary
            # Find max length among gathered predictions
            # Guard against empty list in case max_steps=0 or data issue
            if all_preds_ids_list:
                 max_len_pred = max(len(p) for p in all_preds_ids_list)
                 padded_preds = [
                      # Move prediction tensor p to CPU before concatenating
                      torch.cat([p.cpu(), torch.full((max_len_pred - len(p),), tokenizer.pad_token_id, dtype=p.dtype, device='cpu')])
                      for p in all_preds_ids_list
                 ]
                 # Ensure labels are also padded/truncated consistently if needed, though compute_metrics handles -100
                 # Convert lists of tensors to lists of lists/numpy for compute_metrics
                 preds_np = [p.tolist() for p in padded_preds]
                 labels_np = [l.tolist() for l in all_labels_ids_list]

                 print(f"Computing metrics for {len(preds_np)} examples...")
                 metrics = compute_metrics(preds_np, labels_np, tokenizer)
            else:
                 print("Warning: Prediction list is empty, skipping metric computation.")
                 metrics = {} # Return empty metrics

    # Synchronize model state back to train
    model.train()
    return metrics


def train(config_path, resume_from_checkpoint=None):
    """Main training function."""
    # --- Load Config ---
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # --- Initialize Accelerator ---
    mixed_precision = "bf16" if config.get("model_dtype") == "bfloat16" else "fp16" if config.get("model_dtype") == "float16" else "no"
    # Gradient accumulation plugin handled by accelerator
    accelerator = Accelerator(
        mixed_precision=mixed_precision,
        gradient_accumulation_steps=config['gradient_accumulation_steps'],
        log_with="wandb" if config.get('use_wandb', False) else "tensorboard",
        project_dir=config['output_dir']
    )

    # Initialize wandb through accelerator if enabled
    if config.get('use_wandb', False):
        accelerator.init_trackers(
            project_name=config.get('wandb_project', 'ECE 661'),
            config=config,
            init_kwargs={
                "wandb": {
                    "entity": config.get('wandb_entity', 'benjamin-chauhan-usyd'),
                    "name": config.get('wandb_run_name')
                }
            }
        )

    # Override resume_from_checkpoint if provided as argument
    if resume_from_checkpoint:
        print(f"Command line resume_from_checkpoint provided: {resume_from_checkpoint}")
        print(f"Overriding config value: {config.get('resume_from_checkpoint')}")
        config['resume_from_checkpoint'] = resume_from_checkpoint

    # --- Validate critical config types --- # MODIFIED VALIDATION
    try:
        # Attempt conversion to float, catches strings like '2e-5' and actual numbers
        config['learning_rate'] = float(config.get('learning_rate'))
    except (ValueError, TypeError, AttributeError):
        # AttributeError included in case config.get returns None or other non-convertible type
        raise TypeError(f"Configuration error: 'learning_rate' ('{config.get('learning_rate')}') must be convertible to a number (float).")

    try:
        # Validate hopfield_lr_multiplier
        config['hopfield_lr_multiplier'] = float(config.get('hopfield_lr_multiplier', 1.0))
    except (ValueError, TypeError, AttributeError):
        raise TypeError(f"Configuration error: 'hopfield_lr_multiplier' ('{config.get('hopfield_lr_multiplier')}') must be convertible to a number (float).")

    try:
        # Validate adam_epsilon
        config['adam_epsilon'] = float(config.get('adam_epsilon', 1e-8))
    except (ValueError, TypeError, AttributeError):
        raise TypeError(f"Configuration error: 'adam_epsilon' ('{config.get('adam_epsilon')}') must be convertible to a number (float).")

    try:
        # Validate weight_decay
        config['weight_decay'] = float(config.get('weight_decay', 0.0))
    except (ValueError, TypeError, AttributeError):
        raise TypeError(f"Configuration error: 'weight_decay' ('{config.get('weight_decay')}') must be convertible to a number (float).")

    try:
        # Validate rms_norm_eps
        config['rms_norm_eps'] = float(config.get('rms_norm_eps', 1e-5))
    except (ValueError, TypeError, AttributeError):
        raise TypeError(f"Configuration error: 'rms_norm_eps' ('{config.get('rms_norm_eps')}') must be convertible to a number (float).")


    # --- Initialize Accelerator ---
    accelerator.print(f"--- Configuration ---")
    # Handle resuming from checkpoint
    if config.get('resume_from_checkpoint'):
        accelerator.print(f"Resuming from checkpoint: {config.get('resume_from_checkpoint')}")
    for k, v in config.items(): accelerator.print(f"{k}: {v}")
    accelerator.print("---------------------")

    accelerate_set_seed(config['seed']) # Use Accelerate's seed setting

    # --- Output Dir & Logging (Main process only) ---
    output_dir = config['output_dir']
    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)

        # --- Prepare config for TensorBoard hparams logging ---
        hparams_config = {}
        for k, v in config.items():
            if isinstance(v, (int, float, str, bool, torch.Tensor)):
                hparams_config[k] = v
            elif isinstance(v, (list, dict, tuple)):
                # Convert lists/dicts/tuples to string representation
                hparams_config[k] = str(v)
            elif v is None:
                hparams_config[k] = "None" # Represent None as a string
            # Else: Silently skip other unsupported types for hparam logging

        # Init trackers only on main process, Accelerator handles logging sync
        try:
             accelerator.init_trackers("narrativeqa_hat", config=hparams_config) # Use sanitized config
             # Find the TensorBoard tracker and print its specific log directory
             for tracker in accelerator.trackers:
                 # Check if it's the TensorBoard tracker (adjust class name if using different logger)
                 if tracker.__class__.__name__ == "TensorBoardTracker":
                     # Access the underlying SummaryWriter's log_dir
                     if hasattr(tracker, 'writer') and hasattr(tracker.writer, 'log_dir'):
                         tb_log_dir = tracker.writer.log_dir
                         accelerator.print(f"---> TensorBoard logging initialized. Log directory for this run: {tb_log_dir}")
                         break # Found it, no need to check others
             else: # If loop finishes without break
                 accelerator.print(f"---> Could not automatically determine TensorBoard log directory. Check base directory: {accelerator.logging_dir}")
        except Exception as e:
            accelerator.print(f"Warning: Failed to initialize trackers: {e}")
            # Optionally, initialize without config if it fails:
            # accelerator.init_trackers("narrativeqa_hat")

    # --- Load Tokenizer (Load once, use everywhere) ---
    accelerator.print("Loading tokenizer...")
    tokenizer = get_tokenizer(config['model_name'], cache_dir=config.get("data_cache_dir"))
    # Resize model embeddings later if needed after model loading

    # --- Load Model & Weights ---
    accelerator.print("Creating/Loading model and weights...")
    # Use LoRA settings from config
    lora_config_dict = None
    if config.get("use_lora"):
        lora_config_dict = {
            "r": config["lora_r"],
            "lora_alpha": config["lora_alpha"],
            "lora_dropout": config["lora_dropout"],
            "target_modules": config["lora_target_modules"],
            "bias": "none",
            "task_type": TaskType.CAUSAL_LM,
        }

    # Create/load model on CPU first before PEFT/prepare
    # create_model_and_load_weights assumes CPU loading internally
    model, config = create_model_and_load_weights(config, use_lora=config.get("use_lora"), lora_config_dict=lora_config_dict)
    accelerator.print("Model structure created and weights loaded.")

    # Resize embeddings if tokenizer vocab changed (AFTER loading weights)
    if len(tokenizer) != model.cfg['vocab_size']:
         accelerator.print(f"Resizing model embeddings from {model.cfg['vocab_size']} to {len(tokenizer)}")
         model.resize_token_embeddings(len(tokenizer))
         # Update config vocab size if it changed
         config['vocab_size'] = len(tokenizer)
         model.cfg['vocab_size'] = len(tokenizer) # Update model's internal config too
         # Tie weights again if resized
         model.out_head.weight = model.tok_emb.weight
         accelerator.print("Re-tied output head weights after resize.")

    # Enable gradient checkpointing if configured
    if config.get("gradient_checkpointing"):
        model.gradient_checkpointing_enable()


    # --- Optimizer ---
    accelerator.print("Setting up optimizer...")
    hopfield_params = []
    other_params = []
    base_lr = config['learning_rate']
    hopfield_lr = base_lr * config.get('hopfield_lr_multiplier', 1.0)

    model_to_inspect = model.module if hasattr(model, "module") else model # Handle potential DDP wrap later if not using Accelerator
    if hasattr(model, "base_model"): model_to_inspect=model.base_model # Handle PEFT model

    for name, param in model.named_parameters():
        if param.requires_grad:
            is_hopfield = False
            if hasattr(model_to_inspect, 'hopfield_memory'):
                 for h_name, h_param in model_to_inspect.hopfield_memory.named_parameters():
                     if param is h_param:
                         hopfield_params.append(param); is_hopfield = True; break
            if not is_hopfield:
                 other_params.append(param)

    optimizer_grouped_parameters = [
        {'params': other_params, 'lr': base_lr, 'name': 'base_or_lora'},
        {'params': hopfield_params, 'lr': hopfield_lr, 'name': 'hopfield'}
    ]
    accelerator.print(f"Optimizer: Base/LoRA LR={base_lr}, Hopfield LR={hopfield_lr}")
    accelerator.print(f"Found {len(other_params)} trainable base/LoRA params, {len(hopfield_params)} trainable Hopfield params.")

    # Use standard AdamW
    optimizer = AdamW(optimizer_grouped_parameters, eps=config['adam_epsilon'], weight_decay=config['weight_decay'])
    # # Use 8-bit AdamW
    # optimizer = bnb.optim.AdamW8bit(
    #     optimizer_grouped_parameters,
    #     eps=config['adam_epsilon'],
    #     weight_decay=config['weight_decay']
    #     # AdamW8bit might have slightly different default args, but eps/weight_decay are key
    # )
    # accelerator.print("Using 8-bit AdamW optimizer (bitsandbytes).")

    # --- Load Data ---
    accelerator.print("Loading datasets...")
    # Let dataloader handle device placement via accelerator.prepare
    train_dataloader = get_dataloader(config, tokenizer, split="train")
    eval_dataloader = get_dataloader(config, tokenizer, split="validation")
    if train_dataloader is None or eval_dataloader is None:
        raise ValueError("Failed to load data.")

    # --- Scheduler ---
    # Calculate steps based on distributed training
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / config['gradient_accumulation_steps'])
    max_train_steps = config['num_train_epochs'] * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        name=config['lr_scheduler_type'],
        optimizer=optimizer,
        num_warmup_steps=int(max_train_steps * config['warmup_ratio'] * accelerator.num_processes), # Scale warmup steps by num processes
        num_training_steps=max_train_steps * accelerator.num_processes, # Scale total steps
    )
    accelerator.print(f"Scheduler: Type={config['lr_scheduler_type']}, Total Steps={max_train_steps}, Num Processes={accelerator.num_processes}")


    # --- Prepare with Accelerator ---
    # Order matters: model, optimizer, dataloaders, scheduler
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )
    accelerator.print("Model, optimizer, dataloaders, scheduler prepared with Accelerator.")
    
    # --- Load checkpoint if specified ---
    if config.get('resume_from_checkpoint'):
        checkpoint_path = config.get('resume_from_checkpoint')
        accelerator.print(f"Loading checkpoint from {checkpoint_path}")
        
        # Get the global step from the checkpoint
        # Try to load checkpoint state if available (contains epoch, step)
        trainer_state_path = os.path.join(checkpoint_path, "trainer_state.pt")
        if os.path.exists(trainer_state_path):
            trainer_state = torch.load(trainer_state_path, map_location="cpu")
            global_step = trainer_state.get("step", 0)
            start_epoch = trainer_state.get("epoch", 0)
            accelerator.print(f"Found trainer state, resuming from epoch {start_epoch}, global step {global_step}")
        else:
            # If no trainer state, try to infer step from checkpoint dir name
            try:
                global_step = int(os.path.basename(checkpoint_path).split("-")[1])
                accelerator.print(f"Inferred global step from checkpoint dir: {global_step}")
            except (IndexError, ValueError):
                accelerator.print("Could not determine global step from checkpoint. Starting from step 0.")
                global_step = 0
            start_epoch = 0
        
        try:
            # Try standard accelerator state loading first
            accelerator.print("Attempting to load state with accelerator.load_state...")
            accelerator.load_state(checkpoint_path)
            accelerator.print("Successfully loaded checkpoint with accelerator.load_state")
        except Exception as e:
            accelerator.print(f"Standard accelerator loading failed: {e}")
            accelerator.print("Attempting alternative loading method for LoRA and optimizer...")
            
            # Load optimizer and scheduler states
            try:
                optimizer_path = os.path.join(checkpoint_path, "optimizer.pt")
                scheduler_path = os.path.join(checkpoint_path, "scheduler.pt")
                
                if os.path.exists(optimizer_path):
                    accelerator.print("Loading optimizer state...")
                    optimizer_state = torch.load(optimizer_path, map_location="cpu")
                    optimizer.load_state_dict(optimizer_state)
                    accelerator.print("Optimizer state loaded.")
                
                if os.path.exists(scheduler_path):
                    accelerator.print("Loading scheduler state...")
                    scheduler_state = torch.load(scheduler_path, map_location="cpu")
                    lr_scheduler.load_state_dict(scheduler_state)
                    accelerator.print("Scheduler state loaded.")
                    
                    # Verify the scheduler loaded correctly by checking learning rate
                    current_lr = lr_scheduler.get_last_lr()[0]  # Get first group's LR
                    accelerator.print(f"Current learning rate after loading scheduler: {current_lr:.6f}")
                    
                    # Calculate expected LR based on scheduler type and steps
                    if config['lr_scheduler_type'] == 'cosine':
                        warmup_steps = int(max_train_steps * config['warmup_ratio'])
                        base_lr = config['learning_rate']
                        if global_step < warmup_steps:
                            expected_lr = base_lr * (global_step / warmup_steps)
                        else:
                            progress = (global_step - warmup_steps) / (max_train_steps - warmup_steps)
                            expected_lr = base_lr * 0.5 * (1 + math.cos(math.pi * progress))
                        accelerator.print(f"Expected learning rate at step {global_step}: {expected_lr:.6f}")
                        
                        # Warn if very different 
                        if abs(current_lr - expected_lr) > expected_lr * 0.25:  # If more than 25% different
                            accelerator.print(f"WARNING: Current LR ({current_lr:.6f}) differs significantly from expected LR ({expected_lr:.6f}) for step {global_step}")
            except Exception as load_err:
                accelerator.print(f"Error loading optimizer/scheduler states: {load_err}")
            
            # Load LoRA adapter if using LoRA
            if config.get('use_lora', False):
                try:
                    # Check if adapter config and model exist
                    adapter_config_path = os.path.join(checkpoint_path, "adapter_config.json")
                    adapter_model_path = os.path.join(checkpoint_path, "adapter_model.safetensors")
                    
                    if os.path.exists(adapter_config_path) and os.path.exists(adapter_model_path):
                        accelerator.print("Found LoRA adapter files, loading into model...")
                        
                        # Unwrap model to get the base model
                        unwrapped_model = accelerator.unwrap_model(model)
                        
                        # For PEFT/LoRA models, we need to load the adapter weights
                        from peft import PeftModel
                        
                        # If model is already a PeftModel, we just load the adapter weights
                        if isinstance(unwrapped_model, PeftModel):
                            accelerator.print("Model is already a PeftModel, loading adapter weights...")
                            from safetensors.torch import load_file
                            
                            # Load adapter weights
                            adapter_state_dict = load_file(adapter_model_path)
                            
                            # Make sure we're on CPU first
                            for param_name, param in adapter_state_dict.items():
                                adapter_state_dict[param_name] = param.to("cpu")
                            
                            # Load adapter weights into the model
                            # IMPORTANT: log which keys are in the state dict and explicitly set strict=False
                            accelerator.print(f"LoRA adapter keys: {list(adapter_state_dict.keys())[:5]}... (total: {len(adapter_state_dict)} keys)")
                            # Ensure model's LoRA weights are marked trainable before loading
                            for n, p in unwrapped_model.named_parameters():
                                if 'lora_' in n:
                                    p.requires_grad = True
                                    
                            load_result = unwrapped_model.load_state_dict(adapter_state_dict, strict=False)
                            accelerator.print(f"LoRA adapter load result - missing keys: {len(load_result.missing_keys)}, unexpected keys: {len(load_result.unexpected_keys)}")
                            accelerator.print("LoRA adapter weights loaded successfully.")
                            
                            # Add validation to ensure LoRA weights are properly loaded
                            # Check a few LoRA parameters to make sure they're not all zeros
                            lora_param_count = 0
                            nonzero_lora_params = 0
                            for n, p in unwrapped_model.named_parameters():
                                if 'lora_' in n:
                                    lora_param_count += 1
                                    if torch.abs(p).sum() > 0:
                                        nonzero_lora_params += 1
                                        
                            accelerator.print(f"LoRA parameter check: {nonzero_lora_params}/{lora_param_count} LoRA parameters have non-zero values")
                            if nonzero_lora_params < lora_param_count * 0.9:  # If less than 90% have values
                                accelerator.print("WARNING: Many LoRA parameters appear to be zeros - checkpoint may not be properly loaded!")
                        else:
                            accelerator.print("Model is not a PeftModel, cannot load adapter directly.")
                            
                    else:
                        accelerator.print("LoRA adapter files not found in checkpoint directory.")
                except Exception as lora_err:
                    accelerator.print(f"Error loading LoRA adapter: {lora_err}")
            
            # Ensure Hopfield memory state is properly initialized (doesn't need loading)
            try:
                accelerator.print("Checking Hopfield memory initialization...")
                unwrapped_model = accelerator.unwrap_model(model)
                base_model = unwrapped_model.base_model if hasattr(unwrapped_model, 'base_model') else unwrapped_model
                
                # IMPORTANT CHANGE: Don't reinitialize Hopfield memory when loading a checkpoint
                # This was causing the model to lose its learned state
                if hasattr(base_model, 'hopfield_memory'):
                    accelerator.print("Preserving existing Hopfield memory state during resume")
                    # We don't call initialize_memory() here, just let it keep the current state
            except Exception as hop_err:
                accelerator.print(f"Error checking Hopfield memory: {hop_err}")
    else:
        global_step = 0
        start_epoch = 0

    # --- Training Loop ---
    accelerator.print("Starting training...")
    total_batch_size = config['per_device_train_batch_size'] * accelerator.num_processes * config['gradient_accumulation_steps']
    accelerator.print(f"Total train batch size (accumulated, distributed): {total_batch_size}")
    accelerator.print(f"Total optimization steps: {max_train_steps}")

    best_eval_metric = -float('inf')
    completed_steps = 0
    last_processed_doc_id = None # For memory reset tracking

    for epoch in range(start_epoch, config['num_train_epochs']):
        model.train() # Set model to training mode
        total_loss = 0

        # Use Accelerator's dataloader iteration
        progress_bar = tqdm(range(max_train_steps // config['num_train_epochs']), desc=f"Epoch {epoch+1}",
                             disable=not accelerator.is_local_main_process)
        
        # IMPORTANT: When resuming, update the progress bar to the current step 
        # This makes it clear we're continuing from a checkpoint
        if resume_from_checkpoint and completed_steps == 0:
            # Update progress bar to show we're starting from global_step, not 0
            # Only do this if we just resumed and haven't completed any steps yet
            for _ in range(min(global_step % (max_train_steps // config['num_train_epochs']), len(progress_bar))):
                progress_bar.update(1)
                
        for step, batch in enumerate(train_dataloader):

            # --- Memory Reset Logic (Approximate) ---
            if accelerator.is_main_process: # Check/reset only on main process
                # Check if any document ID in the batch is different from the last processed ID
                current_batch_doc_ids = batch['doc_ids'] if 'doc_ids' in batch else []
                should_reset = False
                
                # If this is the first batch we've seen, no need to reset
                if last_processed_doc_id is not None and current_batch_doc_ids:
                    # Check if ANY document ID in the batch is different from the last one
                    # This is a more conservative approach that ensures memory is reset
                    # when moving between documents, even within a batch
                    current_batch_first_doc_id = current_batch_doc_ids[0]
                    
                    # First check if the first doc ID changed
                    if current_batch_first_doc_id != last_processed_doc_id:
                        should_reset = True
                    # Additionally check if there are mixed document IDs within the batch
                    elif len(set(current_batch_doc_ids)) > 1:
                        should_reset = True
                        if accelerator.is_local_main_process:
                            print(f"Batch contains mixed document IDs: {set(current_batch_doc_ids)}, resetting memory")
                
                if should_reset:
                    unwrapped_model = accelerator.unwrap_model(model)
                    if hasattr(unwrapped_model, 'hopfield_memory') and hasattr(unwrapped_model.hopfield_memory, 'reset_memory'):
                        # print(f"Doc ID changed ({last_processed_doc_id} -> {current_batch_first_doc_id}), resetting Hopfield memory.")
                        unwrapped_model.hopfield_memory.reset_memory()
                
                # Update last processed doc_id to the last one in this batch
                # This ensures we'll reset on the next batch if needed
                if current_batch_doc_ids:
                    last_processed_doc_id = current_batch_doc_ids[-1]
                    
            accelerator.wait_for_everyone() # Ensure reset happens across processes if needed (though called on unwrapped)

            # --- Two-Stage Forward Pass & Accumulation ---
            with accelerator.accumulate(model):

                # --- Stage 1: Context Processing (Populates Hopfield internal state) ---
                # Slice context inputs for each item in the batch
                context_outputs_list = []
                # Iterate through batch items to handle variable context lengths before padding
                max_len_batch = batch['input_ids'].shape[1]
                for i in range(batch['input_ids'].shape[0]):
                     ctx_end = batch['context_end_idx'][i].item()
                     if ctx_end > 0:
                          ctx_ids = batch['input_ids'][i, :ctx_end].unsqueeze(0)
                          ctx_pos_ids = torch.arange(ctx_end, device=accelerator.device).unsqueeze(0)
                          # Forward pass just for context - result not directly used, but populates internal state
                          with torch.set_grad_enabled(accelerator.optimizer_step_was_skipped): # Keep grads if not skipped
                              _ = model(input_ids=ctx_ids, position_ids=ctx_pos_ids)
                     # Else: No context, skip stage 1 for this item

                # --- Stage 1.5: Update Hopfield Memory (using internal state from Stage 1) ---
                unwrapped_model = accelerator.unwrap_model(model)
                if hasattr(unwrapped_model, 'hopfield_memory') and hasattr(unwrapped_model.hopfield_memory, 'update_memory'):
                     # Update happens based on the state stored during the context forward passes
                     if unwrapped_model.hopfield_memory.update_strategy != "none":
                         unwrapped_model.hopfield_memory.update_memory() # Update based on stored state

                # --- Stage 2: QA Processing & Loss Calculation (Uses updated Hopfield state) ---
                # MODIFIED: Use only question portion to force memory usage
                # Extract question-only inputs and corresponding labels
                question_inputs = []
                question_labels = []
                question_position_ids = []  # Track position IDs for each question
                
                for i in range(batch['input_ids'].shape[0]):
                    ctx_end = batch['context_end_idx'][i].item()
                    # Get question-only portion (everything after context_end_idx)
                    q_inputs = batch['input_ids'][i, ctx_end:].unsqueeze(0)
                    q_labels = batch['labels'][i, ctx_end:].unsqueeze(0)
                    
                    # Verify we have question tokens
                    if q_inputs.shape[1] == 0:
                        # Skip examples without valid question tokens
                        continue
                    
                    # Create position IDs that start at the original position (ctx_end)
                    # This maintains the positional context from the full sequence
                    q_pos_ids = torch.arange(ctx_end, ctx_end + q_inputs.shape[1], 
                                           device=accelerator.device).unsqueeze(0)
                    
                    question_inputs.append(q_inputs)
                    question_labels.append(q_labels)
                    question_position_ids.append(q_pos_ids)
                
                # Skip batch if no valid questions but don't terminate training
                if not question_inputs:
                    warnings.warn(f"No valid question inputs in batch. Skipping batch.")
                    # Use continue to proceed to next batch instead of breaking or raising error
                    optimizer.zero_grad()  # Ensure optimizer state is clean
                    continue
                
                # Stack questions into batch tensor
                question_inputs_batch = torch.cat(question_inputs, dim=0)
                question_labels_batch = torch.cat(question_labels, dim=0)
                question_position_ids_batch = torch.cat(question_position_ids, dim=0)
                
                # Create attention mask for question-only inputs
                question_attention_mask = torch.ones_like(question_inputs_batch)
                
                # Forward pass with ONLY the question sequence (no context)
                # Pass the position_ids to maintain positional context
                outputs = model(
                    input_ids=question_inputs_batch,
                    attention_mask=question_attention_mask,
                    position_ids=question_position_ids_batch,
                    labels=question_labels_batch
                )
                loss = outputs["loss"]

                if loss is None:
                    # Try to get batch size for logging
                    bs = question_inputs_batch.shape[0] if hasattr(question_inputs_batch, 'shape') else 'unknown'
                    warnings.warn(f"Loss is None for batch (batch size: {bs}). Skipping step.")
                    continue

                # Gather loss across devices for logging (average)
                avg_loss = accelerator.gather(loss).mean()

                accelerator.backward(loss)

                # Optimizer step happens only when accumulation count is reached
                if accelerator.sync_gradients:
                    # Clip gradients
                    accelerator.clip_grad_norm_(model.parameters(), config['max_grad_norm'])

                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                    # Update tracking variables
                    progress_bar.update(1)
                    completed_steps += 1
                    global_step += 1 # Increment global step only on optimizer step

                    # Accumulate total loss for logging period
                    total_loss += avg_loss.item() # Add the gathered loss from this optimizer step

                    # Logging
                    if global_step % config['logging_steps'] == 0:
                        # Average loss over the logging period
                        avg_step_loss = total_loss / config['logging_steps']
                        current_lr = lr_scheduler.get_last_lr()[0] # Get first LR group
                        accelerator.print(f" Step: {global_step}, Avg Loss: {avg_step_loss:.4f}, LR: {current_lr:.2e}")
                        log_data = {"Loss/train": avg_step_loss, "LR": current_lr, "epoch": epoch + (completed_steps / (max_train_steps // config['num_train_epochs']))}
                        try: 
                            accelerator.log(log_data, step=global_step)
                        except Exception as e: accelerator.print(f"Logging error: {e}")
                        total_loss = 0.0 # Reset loss accumulator for next period

                    # Evaluation & Checkpointing (Only on effective optimizer steps)
                    if global_step % config['eval_steps'] == 0:
                        accelerator.print(f"\n--- Evaluating at Step {global_step} ---")
                        eval_metrics = evaluate(model, eval_dataloader, tokenizer, accelerator.device, config, accelerator)

                        if accelerator.is_main_process: # Only log/save on main process
                             accelerator.print(f"Step {global_step} Eval Metrics: {eval_metrics}")
                             log_eval = {f"Eval/{k}": v for k,v in eval_metrics.items()}
                             try: 
                                 accelerator.log(log_eval, step=global_step)
                             except Exception as e: accelerator.print(f"Logging error: {e}")

                             # Update best model tracking based on eval metric
                             current_metric_key = config.get("metric_for_best_model", "rougeL") # Allow config override
                             current_metric = eval_metrics.get(current_metric_key, -float('inf'))
                             is_best = current_metric > best_eval_metric
                             if is_best:
                                 accelerator.print(f"New best model found! Metric ({current_metric_key}): {current_metric:.4f}")
                                 best_eval_metric = current_metric

                        accelerator.wait_for_everyone() # Ensure all processes sync after eval
                        model.train() # Ensure model is back in training mode after eval

                    # --- Independent Checkpoint Saving Logic ---
                    # Add a print *before* the main process check
                    if global_step % config['save_steps'] == 0:
                         # Use accelerator.print which handles distributed printing and add flush=True
                         accelerator.print(f"[Debug Checkpoint] Step {global_step}: Checking save condition (save_steps={config['save_steps']}).", flush=True)

                    if accelerator.is_main_process:
                        # Check if it's time to save based on save_steps
                        if global_step % config['save_steps'] == 0:
                            # Add a print *inside* the main process check, before other logic
                            accelerator.print(f"[Debug Checkpoint] Step {global_step}: Inside main process save block.", flush=True)

                            # Current approach: Save every `save_steps`, and mark as best *if* this step
                            # also happened to be an evaluation step where it was the best score.
                            # This relies on best_eval_metric being updated correctly during the eval step.
                            is_best_this_step = False
                            if global_step % config['eval_steps'] == 0: # Check if eval just happened
                                # Re-get metric from the eval results if possible, otherwise rely on tracker
                                current_metric_key = config.get("metric_for_best_model", "rougeL")
                                # We don't have eval_metrics here unless eval just ran. Use the tracked best_eval_metric.
                                # Check if eval_metrics exists in local scope from the last eval run
                                if 'eval_metrics' in locals() and isinstance(eval_metrics, dict) and current_metric_key in eval_metrics:
                                     is_best_this_step = eval_metrics.get(current_metric_key, -float('inf')) == best_eval_metric

                            # Use accelerator.print with flush=True for the message in utils.py
                            # Note: The print inside save_checkpoint itself doesn't need flush=True here,
                            #       but the message *triggering* the save does.
                            # accelerator.print(f"Saving checkpoint at step {global_step}...", flush=True) # Redundant now print is in save_checkpoint
                            accelerator.wait_for_everyone() # Wait before saving on main process
                            unwrapped_model = accelerator.unwrap_model(model)
                            # --- DEBUG --- Check model type and attributes before saving ---
                            accelerator.print(f"[Debug Save] Type of unwrapped_model: {type(unwrapped_model)}", flush=True)
                            accelerator.print(f"[Debug Save] Has peft_config attribute: {hasattr(unwrapped_model, 'peft_config')}", flush=True)
                            # --- END DEBUG ---
                            is_lora_active = hasattr(unwrapped_model, 'peft_config')
                            # Pass the potentially updated is_best_this_step flag
                            # The print statement is now inside save_checkpoint in utils.py
                            save_checkpoint(unwrapped_model, optimizer, lr_scheduler, epoch, global_step, output_dir, is_best=is_best_this_step, is_lora=is_lora_active)
                            accelerator.wait_for_everyone() # Sync after main process saves

                    # --- End Independent Checkpoint Saving Logic ---

            # End of accumulation loop for one batch
            # Update progress bar postfix outside sync_gradients block if needed, using latest calculated loss
            if accelerator.is_main_process and 'loss' in locals(): # Check if loss was calculated
                 progress_bar.set_postfix({"Loss": f"{avg_loss:.3f}", "LR": f"{lr_scheduler.get_last_lr()[0]:.2e}"})

            if completed_steps >= max_train_steps: break # Exit epoch early if max steps reached

        # End of Epoch Logic
        progress_bar.close()
        if accelerator.is_main_process:
            accelerator.print(f"--- End of Epoch {epoch+1} ---")
            # Check if eval is needed: if eval_steps <= 0 OR if last step wasn't an eval step
            run_final_eval = config['eval_steps'] <= 0 or (completed_steps % config['eval_steps'] != 0) # Use completed_steps here
            if run_final_eval and eval_dataloader is not None:
                accelerator.print(f"Running final evaluation for Epoch {epoch+1}...")
                eval_metrics = evaluate(model, eval_dataloader, tokenizer, accelerator.device, config, accelerator)
                accelerator.print(f"End of Epoch {epoch+1} Eval Metrics: {eval_metrics}")
                log_epoch_eval = {f"Eval_Epoch/{k}": v for k,v in eval_metrics.items()}
                try: 
                    accelerator.log(log_epoch_eval, step=global_step)
                except Exception as e: accelerator.print(f"Logging error: {e}")

                current_metric_key = config.get("metric_for_best_model", "rougeL")
                current_metric = eval_metrics.get(current_metric_key, -float('inf'))
                # Determine if the *final* model is the best overall
                is_final_best = current_metric > best_eval_metric
                if is_final_best: best_eval_metric = current_metric
            else:
                 # If no final eval ran, we can't determine if the final model is the best
                 # based on the standard metric. is_final_best remains False.
                 is_final_best = False

            # Save final checkpoint
            # accelerator.print("Saving final checkpoint...", flush=True) # Redundant now print is in save_checkpoint
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            # --- DEBUG --- Check model type and attributes before final saving ---
            accelerator.print(f"[Debug Final Save] Type of unwrapped_model: {type(unwrapped_model)}", flush=True)
            accelerator.print(f"[Debug Final Save] Has peft_config attribute: {hasattr(unwrapped_model, 'peft_config')}", flush=True)
            # --- END DEBUG ---
            is_lora_active = hasattr(unwrapped_model, 'peft_config')
            # Pass the is_final_best status to the save function for the final save
            save_checkpoint(unwrapped_model, optimizer, lr_scheduler, epoch, global_step, output_dir, is_best=is_final_best, is_lora=is_lora_active)

        accelerator.wait_for_everyone()
        if completed_steps >= max_train_steps:
             accelerator.print("Max training steps reached. Exiting training.")
             break


    accelerator.print("Training finished.")
    # Clean up trackers
    accelerator.end_training()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train a model with HAT (Hopfield Associative Transformer)")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the configuration file")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, 
                        help="Path to a checkpoint directory to resume training from")
    
    args = parser.parse_args()
    
    config_file = args.config
    if not os.path.exists(config_file):
         raise FileNotFoundError(f"{config_file} not found. Please create it.")

    # Login to HF Hub if needed, outside the main function for clarity
    # from huggingface_hub import login
    # login()

    train(config_file, args.resume_from_checkpoint)