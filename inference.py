import torch
import yaml
import os
import argparse
from tqdm.auto import tqdm
from transformers import GenerationConfig, AutoModelForCausalLM, AutoConfig
from accelerate import Accelerator

# Local imports
from model import (
    HopfieldLlama3Model,
    create_model_and_load_weights,
)  # Re-use model creation logic
from tokenizer_utils import get_tokenizer
from data_loader import get_dataloader  # Use the updated data loader
from peft import PeftModel  # For loading LoRA adapters

# Suppress warnings
import warnings

warnings.filterwarnings(
    "ignore", message=".*Torch was not compiled with flash attention.*"
)
warnings.filterwarnings("ignore", message=".*does not support gradient checkpointing*")


def run_inference(config_path, checkpoint_dir=None, num_examples=5, memory_only_test=False):
    """Runs inference on the test set using a saved checkpoint or the base model.

    Args:
        config_path: Path to config file
        checkpoint_dir: Path to model checkpoint directory (optional, loads base model if None)
        num_examples: Number of examples to process
        memory_only_test: If True and checkpoint_dir is provided, tests memory by providing
                          only question without context in Stage 2
    """
    # --- Load Config --- #
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # --- Initialize Accelerator --- #
    accelerator = Accelerator(
        mixed_precision=config.get("mixed_precision", "no") # Use config or default
    )
    print(f"Accelerator initialized with state: {accelerator.state}")
    # --- End Accelerator Init --- #


    # --- Device Setup --- #
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Replaced by accelerator.device
    # print(f"Using device: {device}")
    print(f"Using accelerator device: {accelerator.device}")
    # Determine dtype from config for consistency
    dtype_str = config.get("model_dtype", "float32")
    if dtype_str == "bfloat16" and torch.cuda.is_bf16_supported():
        model_dtype = torch.bfloat16
    elif dtype_str == "float16":
        model_dtype = torch.float16
    else:
        model_dtype = torch.float32
    print(f"Using model dtype: {model_dtype}")

    # --- Load Tokenizer (needed for both paths) --- #
    print("Loading tokenizer...")
    tokenizer = get_tokenizer(
        config["model_name"], cache_dir=config.get("data_cache_dir")
    )
    # Ensure pad token is set
    if tokenizer.pad_token_id is None:
        print("No PAD token found. Adding and using EOS token as PAD.")
        tokenizer.pad_token_id = tokenizer.eos_token_id
    print(f"Tokenizer loaded. Vocab size: {tokenizer.vocab_size}")
    print(f"BOS: '{tokenizer.bos_token}' (ID: {tokenizer.bos_token_id}), EOS: '{tokenizer.eos_token}' (ID: {tokenizer.eos_token_id}), PAD: '{tokenizer.pad_token}' (ID: {tokenizer.pad_token_id})")

    # --- Load Standard HF Config (Needed for generate compatibility) ---
    hf_config = AutoConfig.from_pretrained(
        config["model_name"], cache_dir=config.get("data_cache_dir"), trust_remote_code=True
    )
    print("Standard Hugging Face config loaded.")
    # --- End HF Config Load ---

    # --- Load Model --- #
    is_custom_hopfield_model = False # Flag to track if we loaded our custom model
    if checkpoint_dir:
        # --- Load Fine-tuned Model (using custom logic) --- #
        is_custom_hopfield_model = True # Assume custom if checkpoint provided
        print(f"Checkpoint directory provided: {checkpoint_dir}")
        if not os.path.exists(checkpoint_dir):
            raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}.")

        print("Loading base model structure and weights (for checkpoint adaptation)...")
        base_model, loaded_config = create_model_and_load_weights(config, use_lora=False)
        # --- MOVE BASE MODEL TO DEVICE **IMMEDIATELY** ---
        # base_model.to(dtype=model_dtype, device=accelerator.device) # <<< REMOVED: Defer to accelerator.prepare
        # print(f"Base model explicitly moved to {accelerator.device} with dtype {model_dtype}.") # <<< REMOVED
        # --- End Immediate Move ---
        print("Base model loaded (pre-adaptation)...")

        # Check if it's a LoRA checkpoint
        is_lora_checkpoint = os.path.exists(
            os.path.join(checkpoint_dir, "adapter_config.json")
        )

        if is_lora_checkpoint:
            print("LoRA checkpoint detected. Loading adapter onto base model...")
            # Base model is already on device from above
            # --- NOW WRAP WITH PEFT ---
            model = PeftModel.from_pretrained(
                base_model, checkpoint_dir, is_trainable=False
            )
            print("LoRA adapter loaded.")
            # model should inherit device from base_model, but we'll ensure below

            # --- Debugging: Print Model Parameter Names ---
            print("\n--- DEBUG: Model Keys (before loading Hopfield state) ---")
            model_param_keys = {name for name, _ in model.named_parameters()} # Use a set for faster lookup
            # Print only a few keys for brevity, especially Hopfield-related if possible
            hopfield_keys_in_model = [k for k in model_param_keys if 'hopfield' in k]
            other_keys_sample = list(model_param_keys - set(hopfield_keys_in_model))[:20]
            print("Sample Hopfield Keys in Model:", hopfield_keys_in_model)
            print("Sample Other Keys in Model:", other_keys_sample)
            print(f"Total parameters in model: {len(model_param_keys)}")
            print("--- END DEBUG: Model Keys ---\n")
            # --- End Debugging ---

            # --- Load Hopfield state into the *base model* --- #
            hopfield_state_path = os.path.join(
                checkpoint_dir, "hopfield_memory_state_dict.pt"
            )
            if os.path.exists(hopfield_state_path):
                print(f"Loading trained Hopfield state from: {hopfield_state_path}")
                # Load to CPU first
                raw_hopfield_state_dict = torch.load(
                    hopfield_state_path, map_location="cpu", weights_only=True
                )

                # --- Debugging: Print Raw State Dict Keys ---
                print("\n--- DEBUG: Raw Hopfield State Dict Keys (from file) ---")
                raw_keys = list(raw_hopfield_state_dict.keys())
                print(raw_keys)
                print(f"Total keys in raw state dict: {len(raw_keys)}")
                print("--- END DEBUG: Raw Hopfield State Dict Keys ---\n")
                # --- End Debugging ---

                # Get the actual base model to load Hopfield state into
                # The PEFT model wraps the custom model, which might be under 'model'
                if hasattr(model, 'base_model') and hasattr(model.base_model, 'model'):
                    target_model_for_hopfield = model.base_model.model
                elif hasattr(model, 'base_model'): # If PEFT wraps the custom model directly
                    target_model_for_hopfield = model.base_model
                else: # Should not happen if PEFT loaded correctly
                    target_model_for_hopfield = model
                    print("Warning: Could not find base_model structure, loading Hopfield into main model object.")

                # Adapt keys - NO prefix needed if loading directly into the base model structure
                adapted_hopfield_state_dict = {}
                keys_to_remove_prefix = ["pre_hopfield_memory.", "middle_hopfield_memory.", "post_hopfield_memory."]
                prefix_found = False
                for key, value in raw_hopfield_state_dict.items():
                    new_key = key
                    # Check if any known prefixes need removal (e.g., if saved with base_model.model prefix)
                    for prefix in keys_to_remove_prefix:
                        if key.startswith(prefix):
                            prefix_found = True
                            break # Found prefix, no need to check others
                    # --- MOVE TENSOR TO DEVICE --- #
                    # adapted_hopfield_state_dict[new_key] = value.to(accelerator.device) # <<< REMOVED: Load to CPU
                    adapted_hopfield_state_dict[new_key] = value # <<< Keep on CPU

                if prefix_found:
                    print(f"  Removed unnecessary prefixes. Adapted {len(adapted_hopfield_state_dict)} Hopfield keys (on CPU).") # Updated message
                else:
                    print(f"  Adapted {len(adapted_hopfield_state_dict)} Hopfield keys (on CPU).") # Updated message

                # --- Debugging: Print Adapted Keys and Check Existence in Target Model ---
                print("\n--- DEBUG: Adapted Hopfield State Dict Keys (before loading into target) ---")
                adapted_keys = list(adapted_hopfield_state_dict.keys())
                print(adapted_keys)
                print(f"Checking if adapted keys exist in target model structure ({type(target_model_for_hopfield).__name__}):")
                target_param_keys = {name for name, _ in target_model_for_hopfield.named_parameters()}
                keys_not_in_target = []
                for k in adapted_keys:
                    if k not in target_param_keys:
                        keys_not_in_target.append(k)
                if keys_not_in_target:
                    print(f"  WARNING: {len(keys_not_in_target)} adapted keys NOT found in target model parameters:")
                    print(f"    {keys_not_in_target}")
                else:
                    print("  All adapted keys found in target model parameters.")
                print("--- END DEBUG: Adapted Hopfield State Dict Keys ---\n")
                # --- End Debugging ---

                # Load the adapted state dict into the target base model
                load_result = target_model_for_hopfield.load_state_dict(adapted_hopfield_state_dict, strict=False)
                print(
                    f"  Hopfield state loaded into base model ({type(target_model_for_hopfield).__name__}). Load result: {load_result}"
                )
                # Filter missing/unexpected keys if necessary
                if load_result.missing_keys or load_result.unexpected_keys:
                    print(f"    Missing keys in base model load: {load_result.missing_keys}")
                    print(f"    Unexpected keys in base model load: {load_result.unexpected_keys}") # Should be empty now for combine_gate
            else:
                print("  No Hopfield state dict found in LoRA checkpoint dir.")

        else:
            # Attempt to load as a full model checkpoint
            print(
                f"No adapter_config.json found. Attempting to load {checkpoint_dir} as a full model checkpoint..."
            )
            try:
                state_dict_path = os.path.join(checkpoint_dir, "pytorch_model.bin")
                if not os.path.exists(state_dict_path):
                    state_dict_path_safe = os.path.join(checkpoint_dir, "model.safetensors")
                    if os.path.exists(state_dict_path_safe):
                        state_dict_path = state_dict_path_safe
                        print("  Found model.safetensors.")
                        from safetensors.torch import load_file
                        state_dict = load_file(state_dict_path, device="cpu")
                    else:
                        raise FileNotFoundError(
                            f"Full model state dict not found: {state_dict_path} or {state_dict_path_safe}"
                        )
                else:
                    state_dict = torch.load(
                        state_dict_path, map_location="cpu", weights_only=True
                    )

                # Handle DDP/FSDP prefixes
                corrected_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith("module."):
                        corrected_state_dict[k[len("module.") :]] = v
                    else:
                        corrected_state_dict[k] = v

                load_result = base_model.load_state_dict(corrected_state_dict, strict=True)
                print(
                    f"Successfully loaded full model state_dict into base model structure: {load_result}"
                )
                model = base_model # Use the base model loaded with checkpoint weights
                # Ensure this path also moves model to device
                # model.to(accelerator.device) # <<< REMOVED: Defer to accelerator.prepare
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load checkpoint from {checkpoint_dir} as either LoRA or full state_dict: {e}"
                )
        print("Fine-tuned model loading complete.")

    else:
        # --- Load Base Model (using standard HF AutoModel) --- #
        is_custom_hopfield_model = False # Base model is not our custom one
        print("No checkpoint directory provided. Loading base model using AutoModelForCausalLM...")
        attn_implementation = config.get("attn_implementation")
        model = AutoModelForCausalLM.from_pretrained(
            config["model_name"],
            torch_dtype=model_dtype,
            cache_dir=config.get("data_cache_dir", "./model_cache"),
            attn_implementation=attn_implementation if attn_implementation else None,
            trust_remote_code=True
        )
        print("Base model loaded via AutoModelForCausalLM.")
        # Ensure this path also moves model to device
        # model.to(accelerator.device) # <<< REMOVED: Defer to accelerator.prepare

    # Ensure model is on the correct device and in eval mode
    # model.to(accelerator.device) # <<< REMOVED: Defer to accelerator.prepare
    model.eval()
    # print("Model moved to device and set to eval mode.") # <<< REMOVED: Happens later

    # --- Prepare Model with Accelerator (AFTER assembly on CPU) --- #
    print("Preparing model with Accelerator...")
    model = accelerator.prepare(model)
    print(f"Model prepared and placed on {accelerator.device}.")
    # --- End Accelerator Prepare Model --- #

    # --- Define Generation Config --- #
    eos_token_id_val = tokenizer.eos_token_id
    if isinstance(eos_token_id_val, list):
        eos_token_id_val = eos_token_id_val[0]

    generation_config_obj = GenerationConfig(
        max_new_tokens=config["max_answer_length"],
        eos_token_id=eos_token_id_val,
        pad_token_id=tokenizer.pad_token_id,
        do_sample=False,
        num_beams=config.get("num_beams", 1),
    )

    # --- Attach Generation Config to Model --- #
    try:
        model.generation_config = generation_config_obj
        print("Generation config attached to model.")
    except AttributeError:
        try:
            model.config.generation_config = generation_config_obj
            print("Generation config attached to model.config")
        except Exception as e_cfg:
            print(f"Warning: Could not attach generation_config to model or model.config: {e_cfg}")
    except Exception as e:
        print(f"Warning: Could not attach generation_config to model: {e}")


    # --- Load Test Data --- #
    print("Loading test dataset (summaries)...")
    test_dataloader = get_dataloader(config, tokenizer, split="test")
    if test_dataloader is None:
        raise ValueError(
            "Failed to load test data. Check dataset availability and data_loader.py."
        )
    print(f"Test dataset loaded with {len(test_dataloader.dataset)} examples.")

    # --- Prepare Dataloader with Accelerator --- #
    print("Preparing dataloader with Accelerator...")
    test_dataloader = accelerator.prepare(test_dataloader)
    print("Dataloader prepared.")
    # --- End Accelerator Prepare Dataloader --- #


    # --- Inference Loop --- #
    results = []
    # Adjust print message based on mode
    if is_custom_hopfield_model and memory_only_test:
        print(f"Running MEMORY-ONLY inference test on first {num_examples} examples...")
        print("NOTICE: Stage 2 will ONLY see the question without context to test memory retention")
    else:
        print(f"Running standard inference on first {num_examples} examples...")

    count = 0
    effective_batch_size = config.get("per_device_eval_batch_size", 1)
    total_iterations = (
                               min(num_examples, len(test_dataloader.dataset)) + effective_batch_size - 1
                       ) // effective_batch_size

    for batch in tqdm(test_dataloader, desc="Inference", total=total_iterations):
        if count >= num_examples:
            break

        # Data is already on the correct device from dataloader
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"] # Keep for reference answer extraction
        context_end_idx_tensor = batch["context_end_idx"]
        doc_ids = batch["doc_ids"]
        answer_texts = batch["answer_texts"]

        batch_size = input_ids.shape[0]
        generated_texts_batch = [None] * batch_size # Initialize list for results

        # --- Determine Inference Path --- #
        run_two_stage = is_custom_hopfield_model and memory_only_test

        with torch.no_grad():
            # Process batch example by example for memory reset and stage handling
            for i in range(batch_size):
                current_input_ids = input_ids[i : i + 1]          # Shape [1, seq_len]
                current_attention_mask = attention_mask[i : i + 1]  # Shape [1, seq_len]
                current_context_end_idx = context_end_idx_tensor[i].item()

                if run_two_stage:
                    # --- Two-Stage Inference (Memory-Only Test) --- #
                    # print(f"\nExample {count + i + 1}: Running Two-Stage Memory Test") # Less verbose

                    # Reset Hopfield memory state before processing each example
                    reset_done = False
                    model_to_reset = model.base_model if hasattr(model, 'base_model') else model
                    if hasattr(model_to_reset, 'reset_all_memories') and callable(model_to_reset.reset_all_memories):
                        model_to_reset.reset_all_memories()
                        reset_done = True
                    elif hasattr(model_to_reset, 'reset_hopfield_memory') and callable(model_to_reset.reset_hopfield_memory):
                        model_to_reset.reset_hopfield_memory()
                        reset_done = True
                    # if not reset_done: print("  Warning: Could not find memory reset method.") # Less verbose

                    # **Stage 1: Process Context (if any) to populate memory**
                    if current_context_end_idx > 0:
                        context_tokens = current_input_ids[:, :current_context_end_idx]
                        context_pos_ids = torch.arange(context_tokens.shape[1], device=accelerator.device).unsqueeze(0)
                        context_attn_mask = current_attention_mask[:, :current_context_end_idx]
                        # Ensure these are on the accelerator device (prepare should handle batch tensors, but explicit is safer for created tensors)
                        # context_tokens = context_tokens.to(accelerator.device) # Already handled by accelerator.prepare(dataloader)
                        # context_attn_mask = context_attn_mask.to(accelerator.device) # Already handled by accelerator.prepare(dataloader)
                        context_pos_ids = context_pos_ids.to(accelerator.device)
                        # model = model.to(accelerator.device) # Model already prepared
                        # print(f"  Stage 1: Processing context (shape {context_tokens.shape})") # Less verbose
                        _ = model(input_ids=context_tokens, attention_mask=context_attn_mask, position_ids=context_pos_ids)
                        # print("  Stage 1: Context processing finished.") # Less verbose
                    # else:
                    #      print("  Stage 1: No context tokens to process.") # Less verbose

                    # **Stage 2: Generate Answer using the *original full prompt* from dataloader**
                    # The prompt now contains the correct chat template structure.
                    # Generation should leverage the memory populated in Stage 1.
                    # print(f"  Stage 2: Generating from full prompt (shape {current_input_ids.shape})") # Less verbose
                    generate_input_ids = current_input_ids
                    generate_attention_mask = current_attention_mask

                else:
                    # --- Standard Single-Stage Inference --- #
                    # print(f"\nExample {count + i + 1}: Running Standard Single-Stage Inference") # Less verbose
                    # Use the full input sequence directly from the dataloader
                    generate_input_ids = current_input_ids
                    generate_attention_mask = current_attention_mask
                    # Position IDs are not needed for standard HF generate if model doesn't require them explicitly
                    # print(f"  Input shape {generate_input_ids.shape}") # Less verbose

                # --- Generate prediction for the current example ---
                # print(f"  Generating answer for example {count + i + 1} using model.generate()...") # Less verbose

                try:
                    # Make sure generation_config is correctly set (it should be from earlier code)
                    if not hasattr(model, 'generation_config'):
                        print("WARNING: model.generation_config not found, using default settings!")
                        gen_outputs = model.generate(
                            input_ids=generate_input_ids,
                            attention_mask=generate_attention_mask,
                            max_new_tokens=config["max_answer_length"],
                            eos_token_id=tokenizer.eos_token_id,
                            pad_token_id=tokenizer.pad_token_id,
                            do_sample=False, # Ensure greedy search
                            num_beams=4,                 # Use 4 beams
                            no_repeat_ngram_size=2,    # Prevent repeating 2-grams
                            early_stopping=True          # Stop when beams finalize
                        )
                    else:
                        # Use the generation config attached earlier
                        gen_outputs = model.generate(
                            input_ids=generate_input_ids,       # Already on correct device via prepare
                            attention_mask=generate_attention_mask, # Already on correct device via prepare
                            max_new_tokens=config["max_answer_length"],
                            pad_token_id=tokenizer.pad_token_id,
                            eos_token_id=tokenizer.eos_token_id,
                            num_beams=4,                 # Use 4 beams
                            do_sample=False,             # Disable sampling for beam search
                            no_repeat_ngram_size=2,    # Prevent repeating 2-grams
                            early_stopping=True          # Stop when beams finalize
                        )

                    # Decode the generated part (excluding the input prompt)
                    # IMPORTANT: Slice output based on the *actual input* to generate
                    prompt_len_for_decode = generate_input_ids.shape[1]
                    generated_ids = gen_outputs[0, prompt_len_for_decode:]
                    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
                    # print(f"  Generation finished for example {count + i + 1}") # Less verbose
                except Exception as gen_error:
                    print(f"  model.generate() failed for example {count + i + 1}: {gen_error}")
                    import traceback
                    traceback.print_exc()
                    generated_text = "[Generation failed due to an error in model.generate()]"

                generated_texts_batch[i] = generated_text

        # --- Store results for the whole batch --- #
        for i in range(batch_size):
            if count >= num_examples:
                break

            current_doc_id = doc_ids[i]
            current_answer_text = answer_texts[i]
            current_context_end_idx = context_end_idx_tensor[i].item()
            original_input_ids = input_ids[i:i+1] # Get the original input for this example

            # Decode original context and question parts for logging
            context_text = tokenizer.decode(original_input_ids[0, :current_context_end_idx], skip_special_tokens=True)
            # Decode the full original prompt (context + QA template) used as input for generation
            full_prompt_text = tokenizer.decode(original_input_ids[0], skip_special_tokens=True)
            # Extract only the QA part (after context) for cleaner display if needed
            qa_part_text = tokenizer.decode(original_input_ids[0, current_context_end_idx:], skip_special_tokens=True).strip()

            results.append(
                {
                    "doc_id": current_doc_id,
                    "summary_context": context_text, # Log the context part
                    "prompt": full_prompt_text,      # Log the full prompt fed to generate
                    "prompt_without_context": qa_part_text, # Log only the QA part
                    "reference_answer": current_answer_text,
                    "generated_answer": generated_texts_batch[i] if i < len(generated_texts_batch) else "[Error: No generation result]",
                    "memory_only_test": run_two_stage
                }
            )
            count += 1
            if count >= num_examples:
                break # Break outer loop too if limit reached

    # --- Print Final Results Summary --- #
    print("--- Inference Results ---")

    for i, res in enumerate(results):
        is_mem_test = res.get('memory_only_test', False)
        print(f"--- Example {i+1} {'(MEMORY-ONLY TEST)' if is_mem_test else ''} ---")
        print(f"Doc ID: {res['doc_id']}")
        print(f"Summary Context Used {'(Stage 1 only)' if is_mem_test else ''}: {res['summary_context']}")
        print(f"Question/Instruction {'(Stage 2 Input)' if is_mem_test else ''}:")

        cleaned_question = res['prompt_without_context']
        # markers_to_remove = [
        #     "<|begin_of_text|>",
        #     "<|start_header_id|>",
        #     "<|end_header_id|>",
        #     "<|eot_id|>",
        #     "user",
        #     "assistant",
        #     "user",
        #     "assistant",
        #     "",
        #     "",
        # ]
        # for marker in markers_to_remove:
        #     cleaned_question = cleaned_question.replace(marker, "")
        print(cleaned_question.strip())

        print(f"Reference Answer (Ground Truth from Dataset): {res['reference_answer']}")
        print(f"Model Generated Answer: {res['generated_answer']}")
        print("-" * (len(f"--- Example {i+1} ---") + (19 if is_mem_test else 0)))
        print("")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run inference with a trained model or base model."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the configuration file.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=False,
        default=None,
        help="Path to the checkpoint directory (e.g., ./output/checkpoint-500). If None, uses base model.",
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=5,
        help="Number of examples from the test set to run inference on.",
    )
    parser.add_argument(
        "--memory_only",
        action="store_true",
        help="Run memory-only test (requires checkpoint)"
    )

    args = parser.parse_args()

    if args.memory_only and not args.checkpoint:
        parser.error("--memory_only requires a --checkpoint to be specified.")

    # Pass all args
    run_inference(args.config, args.checkpoint, args.num_examples, args.memory_only)
