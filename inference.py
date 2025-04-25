import torch
import yaml
import os
import argparse
from tqdm.auto import tqdm
from transformers import GenerationConfig
from collections import defaultdict
import wandb  # Add wandb import

# Local imports
from model import HopfieldLlama3Model, create_model_and_load_weights # Re-use model creation logic
from tokenizer_utils import get_tokenizer
from data_loader import get_dataloader # Use the updated data loader
from peft import PeftModel # For loading LoRA adapters
from utils import compute_metrics, init_metrics  # Import metrics functions

# Suppress warnings
import warnings
warnings.filterwarnings("ignore", message=".*Torch was not compiled with flash attention.*")
warnings.filterwarnings("ignore", message=".*does not support gradient checkpointing*")

def run_inference(config_path, checkpoint_dir, num_stories=3, examples_per_story=3, memory_only_test=False, use_wandb=False, wandb_run_name=None):
    """Runs inference on the test set using a saved checkpoint.
    
    Args:
        config_path: Path to config file
        checkpoint_dir: Path to model checkpoint directory
        num_stories: Number of unique stories/documents to process
        examples_per_story: Number of examples per story to process
        memory_only_test: If True, tests memory by providing only question without context in Stage 2
        use_wandb: Whether to log results to Weights & Biases
        wandb_run_name: Optional custom name for the wandb run
    """
    # --- Load Config --- #
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    # --- Initialize WandB if requested --- #
    if use_wandb:
        print("Initializing Weights & Biases for inference logging...")
        wandb.init(
            project=config.get('wandb_project', 'ECE 661'),
            entity=config.get('wandb_entity', 'benjamin-chauhan-usyd'),
            name=wandb_run_name or f"inference_{os.path.basename(checkpoint_dir)}",
            config={
                "checkpoint": checkpoint_dir,
                "num_stories": num_stories,
                "examples_per_story": examples_per_story,
                "memory_only_test": memory_only_test,
                **config  # Include all config parameters
            }
        )
        print(f"WandB initialized with run name: {wandb.run.name}")

    # --- Device Setup --- #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # Determine dtype from config for consistency
    dtype_str = config.get("model_dtype", "float32")
    if dtype_str == "bfloat16" and torch.cuda.is_bf16_supported(): model_dtype = torch.bfloat16
    elif dtype_str == "float16": model_dtype = torch.float16
    else: model_dtype = torch.float32
    print(f"Using model dtype: {model_dtype}")

    # --- Load Tokenizer --- #
    print("Loading tokenizer...")
    tokenizer = get_tokenizer(config['model_name'], cache_dir=config.get("data_cache_dir"))
    # Ensure pad token is set (should be handled by get_tokenizer, but double-check)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # --- Load Base Model (Structure only, no pretrained weights initially) --- #
    print("Creating base model structure...")
    # We create the base model structure first, then load pretrained weights,
    # and finally apply the LoRA adapter from the checkpoint.
    # Create a temporary config without LoRA flags for initial structure creation
    temp_config_for_structure = config.copy()
    temp_config_for_structure['use_lora'] = False
    # Ensure correct dtype is set for initial model creation
    temp_config_for_structure['dtype'] = model_dtype
    base_model = HopfieldLlama3Model(temp_config_for_structure)
    print("Base model structure created.")

    # --- Load Pretrained Weights into Base Model --- #
    print(f"Loading pretrained weights for {config['model_name']} into base structure...")
    # Use the loading function, but tell it NOT to apply LoRA at this stage
    # This ensures the base weights are loaded correctly before adapter merging.
    base_model, _ = create_model_and_load_weights(config, use_lora=False)
    print("Pretrained weights loaded into base model.")

    # --- Load LoRA Adapter from Checkpoint --- #
    print(f"Loading LoRA adapter from checkpoint: {checkpoint_dir}")
    if not os.path.exists(checkpoint_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}. Make sure the path is correct relative to your execution location.")
    if not os.path.exists(os.path.join(checkpoint_dir, "adapter_config.json")):
         print(f"Warning: Did not find 'adapter_config.json' in {checkpoint_dir}. This might indicate an issue with the checkpoint saving or that it's not a LoRA checkpoint.")
         print(f"Attempting to load {checkpoint_dir} as a full model checkpoint instead...")
         try:
             # Use weights_only=True for security if loading untrusted checkpoints
             state_dict = torch.load(os.path.join(checkpoint_dir, "pytorch_model.bin"), map_location="cpu", weights_only=True)
             # Handle potential DDP/FSDP prefixes if saved directly from a wrapped model
             corrected_state_dict = {}
             for k, v in state_dict.items():
                 if k.startswith("module."):
                      corrected_state_dict[k[len("module."):]] = v
                 else:
                      corrected_state_dict[k] = v
             load_result = base_model.load_state_dict(corrected_state_dict, strict=True) # Use strict=True if loading full model
             print("Successfully loaded full model state_dict from checkpoint.")
             model = base_model # Use the base model loaded with checkpoint weights
         except Exception as e:
             raise RuntimeError(f"Failed to load checkpoint from {checkpoint_dir} as either LoRA or full state_dict: {e}")
    else:
        # Load the PEFT model
        model = PeftModel.from_pretrained(base_model, checkpoint_dir)
        print("LoRA adapter loaded.")
        # Optionally merge adapter weights into the base model for potentially faster inference
        # model = model.merge_and_unload()
        # print("LoRA adapter merged and unloaded.")

    # Ensure model is on the correct device and in eval mode
    model.to(device)
    model.eval()
    print("Model moved to device and set to eval mode.")

    # --- Define Generation Config --- #
    eos_token_id_val = tokenizer.eos_token_id
    if isinstance(eos_token_id_val, list): eos_token_id_val = eos_token_id_val[0]

    generation_config_obj = GenerationConfig(
        max_new_tokens=config['max_answer_length'],
        eos_token_id=eos_token_id_val,
        pad_token_id=tokenizer.pad_token_id,
        do_sample=False, # Use greedy decoding for simplicity
        num_beams=config.get('num_beams', 1)
    )

    # --- Attach Generation Config to Model --- #
    try:
        model.generation_config = generation_config_obj
        print("Generation config attached to model.")
    except Exception as e:
        print(f"Warning: Could not attach generation_config to model: {e}")
    # --- End Attachment ---

    # --- Load Test Data (Using Summaries) --- #
    print("Loading test dataset (summaries)...")
    # Use the same config, but specify the 'test' split
    test_dataloader = get_dataloader(config, tokenizer, split="test")
    if test_dataloader is None:
        raise ValueError("Failed to load test data. Check dataset availability and data_loader.py.")
    print(f"Test dataset loaded with {len(test_dataloader.dataset)} examples.")

    # --- Prepare for Story-based Sampling --- #
    print(f"Will process {num_stories} stories with {examples_per_story} examples per story")
    if memory_only_test:
        print("NOTICE: Stage 2 will ONLY see the question without context to test memory retention")
    
    # Track stories and examples per story
    story_counts = defaultdict(int)  # Counts examples per document/story
    processed_stories = 0  # Number of unique stories processed
    results = []  # To store all results
    
    # --- Inference Loop --- #
    print("Starting inference loop...")
    for batch in tqdm(test_dataloader, desc="Inference"):
        if processed_stories >= num_stories:
            break  # We've processed the requested number of stories

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels']
        context_end_idx_tensor = batch['context_end_idx'].to(device)
        doc_ids = batch['doc_ids']
        answer_texts = batch['answer_texts']

        batch_size = input_ids.shape[0]

        for i in range(batch_size):
            current_doc_id = doc_ids[i]
            
            # Skip if we already have enough examples for this story
            if story_counts[current_doc_id] >= examples_per_story:
                continue
                
            # Skip if we've already processed enough unique stories
            if processed_stories >= num_stories and current_doc_id not in story_counts:
                continue
                
            # We're processing a new story
            if current_doc_id not in story_counts:
                processed_stories += 1
            
            # Process this example
            current_input_ids = input_ids[i:i+1]  # Keep batch dim
            current_attention_mask = attention_mask[i:i+1]
            current_labels = labels[i]
            current_context_end_idx = context_end_idx_tensor[i].item()
            current_answer_text = answer_texts[i]

            # --- Debugging Labels Tensor in Inference ---
            if i == 0 and len(results) == 0:  # Print only for the very first example
                print(f"\n--- INFERENCE LABEL DEBUG (First Example) ---")
                print(f"Raw labels shape for this item: {current_labels.shape}")
                # Get the part of labels that should contain the answer
                # Move labels to the same device as the indices for indexing
                current_labels_cpu = current_labels.cpu()
                label_indices_after_context = torch.arange(current_context_end_idx, current_labels.shape[0], device='cpu')
                if len(label_indices_after_context) > 0:
                    answer_part_labels = current_labels_cpu[label_indices_after_context]
                    # Filter out -100 padding
                    valid_answer_label_tokens = answer_part_labels[answer_part_labels != -100]
                    print(f"Valid answer label token IDs: {valid_answer_label_tokens.tolist()}")
                    decoded_inference_label_answer = tokenizer.decode(valid_answer_label_tokens, skip_special_tokens=True)
                    print(f"Decoded Answer directly from Inference Labels: '{decoded_inference_label_answer}'")
                else:
                    print("No label indices found after context_end_idx.")
                print("------------------------------------------\n")
            # --- End Debugging ---

            # --- Two-Stage Inference Simulation --- #
            with torch.no_grad():
                # **Stage 1: Process Context**
                # Extract context tokens (summary)
                context_tokens = current_input_ids[:, :current_context_end_idx]
                
                # Extract question tokens for memory-only test
                question_tokens = None
                if memory_only_test:
                    question_tokens = current_input_ids[:, current_context_end_idx:]
                    # Verify we have question tokens
                    if question_tokens.shape[1] == 0:
                        print(f"WARNING: No question tokens found for example {current_doc_id}-{story_counts[current_doc_id]+1}, skipping")
                        continue
                
                # Reset Hopfield memory before processing context
                if hasattr(model, 'base_model') and hasattr(model.base_model, 'hopfield_memory'):
                    if hasattr(model.base_model.hopfield_memory, 'reset_memory'):
                        model.base_model.hopfield_memory.reset_memory()
                elif hasattr(model, 'hopfield_memory'):
                    if hasattr(model.hopfield_memory, 'reset_memory'):
                        model.hopfield_memory.reset_memory()

                if context_tokens.shape[1] > 0:
                    # Forward pass just for context - populates internal state (hidden states for Hopfield)
                    context_pos_ids = torch.arange(context_tokens.shape[1], device=device).unsqueeze(0)
                    _ = model(input_ids=context_tokens, position_ids=context_pos_ids)

                    # **Stage 1.5: Update Hopfield Memory (if applicable)**
                    # Manually trigger update based on internal state from context pass
                    update_occurred = False
                    if hasattr(model, 'base_model') and hasattr(model.base_model, 'hopfield_memory'):
                        hopfield_layer = model.base_model.hopfield_memory
                    elif hasattr(model, 'hopfield_memory'):
                        hopfield_layer = model.hopfield_memory
                    else:
                        hopfield_layer = None

                    if hopfield_layer and hasattr(hopfield_layer, 'update_memory') and hopfield_layer.update_strategy != "none":
                        hopfield_layer.update_memory()
                        update_occurred = True

                # **Stage 2: Generate Answer**
                # For memory-only test, use ONLY question tokens without context
                if memory_only_test and question_tokens is not None:
                    prompt_tokens = question_tokens
                    # Create appropriate attention mask for question-only
                    question_attention_mask = torch.ones_like(question_tokens)
                    
                    # Preserve original position IDs even in memory-only test
                    # This is crucial for Hopfield models which rely on position information
                    context_length = current_context_end_idx
                    
                    # The key is to create position IDs starting from context_end_idx
                    # So the model sees these positions as a continuation
                    print(f"\nExample {current_doc_id}-{story_counts[current_doc_id]+1}: Testing MEMORY ONLY with question (no context)")
                    print(f"  Question tokens shape: {question_tokens.shape}, Context length: {context_length}")
                else:
                    # Standard mode: use full prompt including context
                    prompt_tokens = current_input_ids
                    question_attention_mask = current_attention_mask

                # Generate answer with proper error handling
                # MODIFICATION: Directly use fallback for memory_only_test due to persistent issues
                if memory_only_test:
                    try:
                        print(f"Using fallback generation method directly for memory-only example {current_doc_id}-{story_counts[current_doc_id]+1}")
                        # Create a copy of input to avoid modifying the original
                        current_tokens = prompt_tokens.clone()
                        generated_text_pieces = []
                        
                        # Use max_new_tokens from generation_config_obj
                        max_new_tokens = generation_config_obj.max_new_tokens
                        
                        for _ in range(max_new_tokens):
                            # Create position IDs matching the current sequence length
                            # For memory-only tests, start from context_end_idx
                            if question_tokens is not None: # Check if question_tokens exist (implies memory_only)
                                current_pos_ids = torch.arange(
                                    current_context_end_idx,
                                    current_context_end_idx + current_tokens.shape[1],
                                    device=device
                                ).unsqueeze(0)
                            else:
                                # Standard case: position IDs start from 0
                                current_pos_ids = torch.arange(current_tokens.shape[1], device=device).unsqueeze(0)
                            
                            # Verify position IDs are correct length
                            if current_pos_ids.shape[1] != current_tokens.shape[1]:
                                print(f"WARNING: Position IDs length {current_pos_ids.shape[1]} doesn't match tokens length {current_tokens.shape[1]}")
                                if current_pos_ids.shape[1] > current_tokens.shape[1]:
                                    current_pos_ids = current_pos_ids[:, :current_tokens.shape[1]]
                                else:
                                    print("Critical error with position IDs length")
                                    raise ValueError("Position IDs too short")
                            
                            # Get next token prediction
                            with torch.no_grad():
                                outputs = model(
                                    input_ids=current_tokens,
                                    attention_mask=torch.ones_like(current_tokens),
                                    position_ids=current_pos_ids
                                )
                                next_token_logits = outputs["logits"][:, -1, :]
                                next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
                                
                            # Add token to sequence
                            current_tokens = torch.cat([current_tokens, next_token], dim=1)
                            
                            # Add to generated text
                            token_text = tokenizer.decode(next_token[0])
                            generated_text_pieces.append(token_text)
                            
                            # Stop if EOS token
                            if next_token.item() == tokenizer.eos_token_id:
                                break
                                
                        # Combine the generated pieces
                        generated_text = ''.join(generated_text_pieces).strip()
                        print(f"Fallback generation succeeded for example {current_doc_id}-{story_counts[current_doc_id]+1}")
                    except Exception as fallback_error:
                        print(f"Fallback generation failed: {fallback_error}")
                        generated_text = "[Generation failed due to an error]"
                else:
                    # Standard inference using model.generate()
                    try:
                        # Normal case: position IDs start from 0
                        seq_len = prompt_tokens.shape[1]
                        prompt_position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
                        
                        # Ensure position_ids have the exact same shape as input_ids for first dim
                        assert prompt_position_ids.shape[1] == prompt_tokens.shape[1], \
                            f"Position IDs shape {prompt_position_ids.shape} doesn't match input_ids shape {prompt_tokens.shape}"
                        
                        # Generate with better error handling
                        outputs = model.generate(
                            input_ids=prompt_tokens,
                            generation_config=generation_config_obj,
                            attention_mask=question_attention_mask,
                            position_ids=prompt_position_ids,
                            use_cache=False  # Explicitly disable cache for consistency
                        )

                        # Decode generated sequence (excluding the input prompt part)
                        generated_ids = outputs[0, prompt_tokens.shape[1]:]
                        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
                    except Exception as e:
                        print(f"Error during standard generation for example {current_doc_id}-{story_counts[current_doc_id]+1}: {e}")
                        # Provide a placeholder response if generation fails
                        generated_text = "[Generation failed due to an error]"

            # --- Store and Print Results --- #
            full_input_text = tokenizer.decode(current_input_ids[0], skip_special_tokens=False)
            context_text = tokenizer.decode(context_tokens[0], skip_special_tokens=True)
            
            # Find where the question starts in the prompt
            prompt_text_full = tokenizer.decode(prompt_tokens[0], skip_special_tokens=True)
            
            # For standard mode, extract question by removing context
            if not memory_only_test:
                conversation_format = prompt_text_full.replace(context_text, "", 1).strip()
            else:
                # For memory-only test, the prompt is already just the question
                conversation_format = prompt_text_full.strip()

            # Get the reference answer
            # Use the stored answer_text directly instead of decoding labels
            reference_text = current_answer_text
            
            # Debug print for the first example to check fix
            if len(results) == 0:
                print("\n--- FIXED REFERENCE ANSWER EXTRACTION (First Example) ---")
                print(f"Decoded reference answer (from stored text): '{reference_text}'")
                # Move to CPU for consistent handling
                # Ensure current_labels_device exists and is on device before indexing
                current_labels_device = current_labels.to(device)
                orig_labels = current_labels_device[current_labels_device != -100]
                orig_text = tokenizer.decode(orig_labels.cpu(), skip_special_tokens=True)
                print("-------------------------------------------------------\n")

            # Create a result dictionary with all relevant information
            result_dict = {
                "doc_id": current_doc_id,
                "example_num": story_counts[current_doc_id] + 1,  # 1-indexed for display
                "summary_context": context_text,
                "prompt": prompt_text_full,
                "prompt_without_context": conversation_format,
                "reference_answer": reference_text,
                "generated_answer": generated_text,
                "memory_only_test": memory_only_test
            }
            
            results.append(result_dict)
            
            # Log individual result to wandb if enabled
            if use_wandb:
                # Compute metrics for this single example
                init_metrics()  # Ensure metrics are initialized
                single_prediction = [[ord(c) for c in generated_text]]  # Convert to token IDs format expected by compute_metrics
                single_reference = [[ord(c) for c in reference_text]]
                try:
                    metrics = compute_metrics(single_prediction, single_reference, tokenizer)
                    
                    # Log to wandb with metrics
                    wandb.log({
                        f"example_{current_doc_id}_{story_counts[current_doc_id]}": {
                            "prompt": conversation_format,
                            "reference": reference_text,
                            "prediction": generated_text,
                            "rouge1": metrics.get("rouge1", 0),
                            "rouge2": metrics.get("rouge2", 0),
                            "rougeL": metrics.get("rougeL", 0),
                            "bleu": metrics.get("bleu", 0),
                            "exact_match": metrics.get("exact_match", 0),
                            "memory_only": memory_only_test
                        }
                    })
                except Exception as e:
                    print(f"Error computing metrics for wandb logging: {e}")
            
            # Increment the count for this story
            story_counts[current_doc_id] += 1
            
            # Break if we have enough examples for this story
            if story_counts[current_doc_id] >= examples_per_story:
                print(f"Completed all {examples_per_story} examples for story {current_doc_id}")

    # --- Print Final Results Summary --- #
    print("\n--- Inference Results ---")
    print(f"Processed {len(story_counts)} unique stories with up to {examples_per_story} examples each")
    if memory_only_test:
        print("MEMORY-ONLY TEST: Model was given ONLY the question without context in Stage 2")
    
    # Group results by doc_id for better organization
    for doc_id in sorted(story_counts.keys()):
        doc_results = [r for r in results if r['doc_id'] == doc_id]
        print(f"\n=== Story: {doc_id} (Processed {len(doc_results)} examples) ===")
        
        # Print context only once per story (assuming it's the same for all examples)
        if doc_results:
            print(f"Summary Context Used (Stage 1 only):\n{doc_results[0]['summary_context']}")
        
        # Print each example for this story
        for res in sorted(doc_results, key=lambda x: x['example_num']):
            print(f"\n--- Example {res['example_num']} ---")
            print(f"Question (given to model {'WITHOUT context' if res.get('memory_only_test') else 'WITH context'}):")
            
            # Question extraction logic remains the same...
            question_found = False
            
            # Method 1: Try to extract using conversation format markers
            for marker_set in [
                ("<|start_header_id|>user<|end_header_id|>\n\n", "<|eot_id|>"), 
                ("<user>", "</user>"),
                ("[INST]", "[/INST]")
            ]:
                start_marker, end_marker = marker_set
                if start_marker in res['prompt']:
                    q_parts = res['prompt'].split(start_marker)
                    if len(q_parts) > 1:
                        if end_marker in q_parts[1]:
                            q_text = q_parts[1].split(end_marker)[0].strip()
                            print(q_text)
                            question_found = True
                            break
            
            # Method 2: If the prompt_without_context was extracted successfully
            if not question_found and res['prompt_without_context']:
                # Try to clean up any remaining markers
                q_text = res['prompt_without_context']
                for marker in ["<|start_header_id|>user<|end_header_id|>", "<|eot_id|>", 
                              "<user>", "</user>", "[INST]", "[/INST]"]:
                    q_text = q_text.replace(marker, "")
                print(q_text.strip())
                question_found = True
            
            # Fallback message if extraction failed
            if not question_found:
                print("[Could not auto-extract question from prompt]")
            
            # Print the reference answer and generated answer with clear labels
            print(f"\nReference Answer (Ground Truth):\n{res['reference_answer']}")
            print(f"\nModel Generated Answer:\n{res['generated_answer']}")
            print("-"*50)
    
    # Compute aggregate metrics if wandb is enabled
    if use_wandb and results:
        print("\nComputing aggregate metrics for all examples...")
        init_metrics()  # Ensure metrics are initialized
        
        # Prepare data for metrics computation
        all_predictions = []
        all_references = []
        
        for res in results:
            # Convert to token IDs format expected by compute_metrics
            # This is simplistic - in a real implementation, you might use the actual token IDs
            pred_ids = [ord(c) for c in res['generated_answer']]
            ref_ids = [ord(c) for c in res['reference_answer']]
            all_predictions.append(pred_ids)
            all_references.append(ref_ids)
        
        try:
            # Compute metrics across all examples
            aggregate_metrics = compute_metrics(all_predictions, all_references, tokenizer)
            
            # Add memory test flag to metrics
            aggregate_metrics['memory_only_test'] = memory_only_test
            aggregate_metrics['num_stories'] = len(story_counts)
            aggregate_metrics['total_examples'] = len(results)
            
            # Log aggregate metrics to wandb
            wandb.log({
                "aggregate_metrics": aggregate_metrics
            })
            
            print("\nAggregate Metrics:")
            for metric_name, metric_value in aggregate_metrics.items():
                print(f"{metric_name}: {metric_value}")
                
            # Create and log a table of all results
            table_data = []
            for res in results:
                table_data.append([
                    res['doc_id'],
                    res['example_num'],
                    res['prompt_without_context'],
                    res['reference_answer'],
                    res['generated_answer'],
                    res.get('memory_only_test', False)
                ])
            
            result_table = wandb.Table(
                columns=["doc_id", "example_num", "question", "reference", "prediction", "memory_only"],
                data=table_data
            )
            wandb.log({"results_table": result_table})
            
            # Finish the wandb run
            wandb.finish()
            
        except Exception as e:
            print(f"Error computing aggregate metrics: {e}")
            if use_wandb:
                wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with a trained HAT model.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the configuration file.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the checkpoint directory (e.g., ./output/checkpoint-500).")
    parser.add_argument("--num_stories", type=int, default=3, help="Number of unique stories/documents to process.")
    parser.add_argument("--examples_per_story", type=int, default=3, help="Number of examples to process per story.")
    parser.add_argument("--memory_only", action="store_true", help="Run memory-only test (question without context in Stage 2)")
    parser.add_argument("--use_wandb", action="store_true", help="Log inference results to Weights & Biases")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="Optional custom name for the wandb run")

    args = parser.parse_args()

    run_inference(args.config, args.checkpoint, args.num_stories, args.examples_per_story, args.memory_only, args.use_wandb, args.wandb_run_name) 