# Use the NarrativeQADataset and get_dataloader functions from the previous response.
# Key things to ensure:
# - Correct handling of tokenizer's chat template for Llama 3.2 Instruct.
# - Accurate calculation of label start/end indices, ignoring context tokens (-100).
# - Robust chunking and instance creation logic.

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset # Import only load_dataset
from transformers import PreTrainedTokenizerBase
import random
import numpy as np
import warnings
from tqdm.auto import tqdm # Keep for overall progress if needed
import os # <<< ADDED OS IMPORT
import hashlib # <<< ADDED HASHLIB IMPORT

class NarrativeQADataset(Dataset):
    def __init__(self, config, tokenizer: PreTrainedTokenizerBase, split="train"):
        self.config = config
        self.tokenizer = tokenizer
        self.split = split

        # --- Cache Logic ---
        cache_dir = config.get("data_cache_dir", "./data_cache")
        processed_cache_dir = os.path.join(cache_dir, "processed_narrativeqa_summary") # Changed cache subfolder name
        os.makedirs(processed_cache_dir, exist_ok=True)

        # Create a unique hash for the cache filename based on relevant config
        # Removed chunk size/overlap from hash as they are no longer used
        config_hash_str = f"{config['model_name']}-{split}-{config['context_length']}"
        cache_hash = hashlib.sha256(config_hash_str.encode()).hexdigest()[:16]
        self.cache_path = os.path.join(processed_cache_dir, f"processed_summary_{split}_{cache_hash}.pt") # Changed filename prefix

        # Check if cached file exists
        if os.path.exists(self.cache_path):
            print(f"Loading processed summary data for split '{split}' from cache: {self.cache_path}")
            try:
                # Load with weights_only=True for security unless strictly necessary otherwise
                self._processed_data = torch.load(self.cache_path, weights_only=True) # <<< Added weights_only=True
                if not self._processed_data:
                    warnings.warn(f"Cache file {self.cache_path} loaded empty data. Reprocessing.")
                    self._processed_data = None # Mark as None to trigger reprocessing
                else:
                    print(f"Successfully loaded {len(self._processed_data)} items from cache.")
            except Exception as e:
                warnings.warn(f"Failed to load cache file {self.cache_path}: {e}. Reprocessing.")
                self._processed_data = None # Mark as None to trigger reprocessing
        else:
            print(f"Processed summary data cache not found for split '{split}': {self.cache_path}. Processing...")
            self._processed_data = None
        # --- End Cache Logic ---

        # Load raw dataset if needed
        if self._processed_data is None:
            try:
                self.dataset = load_dataset(config['dataset_name'], split=split, cache_dir=cache_dir)
            except Exception as e:
                print(f"Failed to load dataset '{config['dataset_name']}' split '{split}': {e}")
                raise

            # Removed chunk_size and overlap attributes
            self.max_seq_length = config['context_length']

            # Compute processed data
            self._processed_data = self._create_training_instances()

            # Save to cache if computed
            if self._processed_data:
                print(f"Saving processed summary data for split '{split}' to cache: {self.cache_path}")
                try:
                    torch.save(self._processed_data, self.cache_path)
                except Exception as e:
                    warnings.warn(f"Failed to save processed data to cache {self.cache_path}: {e}")
            else:
                 warnings.warn(f"No processed summary data was generated for split '{split}', cache not saved.")


        # Final check if data exists
        if not self._processed_data:
             raise RuntimeError(f"Failed to load or process summary data for split '{split}'. Ensure dataset exists and processing logic is correct.")

    def _create_training_instances(self):
        processed = []
        num_skipped = 0
        printed_example_count = 0
        print(f"Processing {self.split} split using summaries...")
        # Use standard tqdm here for overall progress
        for idx, example in enumerate(tqdm(self.dataset, desc=f"Processing {self.split} (Summaries)")):
            try:
                doc_id = example['document']['id']
                # --- Use Summary Text ---
                # Handle potential missing keys gracefully
                summary_text = example.get('document', {}).get('summary', {}).get('text', "")
                question_text = example.get('question', {}).get('text', "")
                answers_list = example.get('answers', [])

                # Print raw answers structure for the first example to debug
                if idx == 0:
                    print(f"\n--- RAW ANSWERS STRUCTURE ---")
                    print(f"Type: {type(answers_list)}, Length: {len(answers_list)}")
                    if len(answers_list) > 0:
                        print(f"First answer object: {answers_list[0]}")
                        if isinstance(answers_list[0], dict):
                            print(f"Keys in first answer: {list(answers_list[0].keys())}")
                            print(f"Text field: {answers_list[0].get('text', 'NOT FOUND')}")
                            print(f"Tokens field: {answers_list[0].get('tokens', 'NOT FOUND')}")
                    print("---------------------------\n")

                # FIXED: Properly extract full answer text from the answer objects
                # Make sure we're accessing the 'text' field from each answer object
                answer_texts = []
                for ans in answers_list:
                    # Ensure ans is a dictionary and has a 'text' field
                    if isinstance(ans, dict) and 'text' in ans:
                        # Get the complete text field
                        full_answer_text = ans['text']
                        answer_texts.append(full_answer_text)
                        # Debug print the first answer we extract
                        if idx == 0 and len(answer_texts) == 1:
                            print(f"First answer text extracted: '{full_answer_text}'")

                # Skip if essential components are missing
                if not summary_text or not question_text or not answer_texts:
                    # print(f"Skipping example {doc_id}: Missing summary, question, or answer.")
                    num_skipped += 1; continue
                answer_text = answer_texts[0] # Use first answer

                # Debug print to verify the answer text is complete
                if idx == 0:
                    print(f"DEBUG - First answer being used: '{answer_text}'")
                    print(f"Complete answer length: {len(answer_text)}")

                # --- Encode Summary directly ---
                # Encode summary without special tokens initially
                summary_tokens = self.tokenizer.encode(summary_text, add_special_tokens=False)
                if not summary_tokens: # Skip if summary encoding is empty
                    # print(f"Skipping example {doc_id}: Empty summary encoding.")
                    num_skipped += 1; continue

                # Prepare QA using chat template
                conversation = [{"role": "user", "content": question_text}]
                # Ensure template application doesn't add BOS if tokenizer already does implicitly
                # Some tokenizers might add BOS automatically during encode, others need it in template
                # Check tokenizer.chat_template and tokenizer behaviour
                try:
                    qa_formatted_text = self.tokenizer.apply_chat_template(
                        conversation,
                        add_generation_prompt=True, # Adds assistant role and signals turn
                        tokenize=False
                    )
                except (AttributeError, ValueError, TypeError) as e:
                    # Fallback for tokenizers without chat template
                    warnings.warn(f"Chat template application failed: {e}. Using basic format.")
                    qa_formatted_text = f"User: {question_text}\nAssistant: "
                    
                qa_prompt_tokens = self.tokenizer.encode(qa_formatted_text, add_special_tokens=False) # No BOS/EOS here
                answer_tokens = self.tokenizer.encode(answer_text, add_special_tokens=False) # No BOS/EOS here

                # Add EOS token to answer if tokenizer has one and it's not already there
                if self.tokenizer.eos_token_id is not None and (not answer_tokens or answer_tokens[-1] != self.tokenizer.eos_token_id):
                     answer_tokens += [self.tokenizer.eos_token_id]

                # Calculate available length for the summary context
                # Reserve space for the QA prompt tokens
                available_len_for_context = self.max_seq_length - len(qa_prompt_tokens)

                if available_len_for_context < 0: # QA prompt alone is too long
                     # print(f"Warning: QA prompt ({len(qa_prompt_tokens)}) exceeds max_seq_length ({self.max_seq_length}). Skipping.")
                     num_skipped += 1
                     continue

                # --- Select Context Tokens from Summary ---
                tokens_to_add_count = min(len(summary_tokens), available_len_for_context)
                if tokens_to_add_count > 0:
                    context_to_include_tokens = summary_tokens[-tokens_to_add_count:]
                else:
                    context_to_include_tokens = [] # No space for context

                # Final input sequence (Context + QA Prompt)
                input_tokens = context_to_include_tokens + qa_prompt_tokens
                # Truncate explicitly *before* creating labels
                input_tokens = input_tokens[:self.max_seq_length]
                
                # Important: Recalculate context_end_idx after possible truncation
                # This ensures context_end_idx is always accurate even if truncation happened
                context_end_idx = min(len(context_to_include_tokens), len(input_tokens))

                # Target sequence includes the answer tokens
                target_tokens = input_tokens + answer_tokens

                # Create labels, same length as input_tokens
                labels = torch.full((len(input_tokens),), -100, dtype=torch.long)

                # Assign labels: labels[i] should be target_tokens[i+1]
                num_valid_labels = 0
                for i in range(len(input_tokens)):
                    target_idx = i + 1
                    if target_idx < len(target_tokens):
                        labels[i] = target_tokens[target_idx]
                        num_valid_labels += 1 # Count how many labels we set
                    # else: labels[i] remains -100

                # Mask the context part of the labels
                context_end_idx = len(context_to_include_tokens)
                # Ensure context_end_idx doesn't go out of bounds for labels
                context_end_idx = min(context_end_idx, len(labels))
                labels[:context_end_idx] = -100
                num_valid_labels -= context_end_idx # Subtract masked context labels

                # More debugging for the first example
                if idx == 0:
                    print(f"\n--- LABEL DEBUG --- First Example ---")
                    print(f"Input Tokens Len: {len(input_tokens)}")
                    print(f"Answer Tokens Len: {len(answer_tokens)}")
                    print(f"Target Tokens Len: {len(target_tokens)}")
                    print(f"Labels Tensor Len: {len(labels)}")
                    print(f"Context End Idx: {context_end_idx}")
                    print(f"Number of non -100 labels BEFORE context mask: {torch.sum(labels != -100).item()}")
                    print(f"Number of non -100 labels AFTER context mask (should be >0): {num_valid_labels}")
                    # Decode the part of labels that should contain the answer
                    label_answer_tokens = labels[context_end_idx:]
                    decoded_label_answer = self.tokenizer.decode(label_answer_tokens[label_answer_tokens != -100], skip_special_tokens=True)
                    print(f"Decoded Answer from Labels: '{decoded_label_answer}'")
                    print(f"Original Answer Text:       '{answer_text}'")
                    print("----------------------------------\n")

                # Check if any valid labels remain after masking context
                if torch.sum(labels != -100) > 0:
                     instance_data = {
                         "doc_id": doc_id,
                         "input_ids": input_tokens,
                         "labels": labels.tolist(),
                         "context_end_idx": context_end_idx,
                         "is_qa_instance": True,
                         "answer_text": answer_text # <<< Store original answer text
                     }
                     processed.append(instance_data)

                     # Optional: Print first processed instance
                     if printed_example_count == 0:
                         print("\n--- First Processed Example (Summary Based) --- ")
                         decoded_context = self.tokenizer.decode(context_to_include_tokens, skip_special_tokens=True)
                         # Answer tokens might include EOS, decode without it for readability if present
                         ans_tokens_to_decode = answer_tokens
                         if ans_tokens_to_decode and ans_tokens_to_decode[-1] == self.tokenizer.eos_token_id:
                             ans_tokens_to_decode = ans_tokens_to_decode[:-1]
                         # Use stored answer_text for printing here too for consistency
                         # decoded_answer = self.tokenizer.decode(ans_tokens_to_decode, skip_special_tokens=True)
                         decoded_answer = answer_text 

                         print(f"Document ID: {doc_id}")
                         print(f"\nSummary Context Tokens ({len(context_to_include_tokens)}):\n{decoded_context}") # Changed label
                         print(f"\nOriginal Question:\n{question_text}")
                         # Optional: Print formatted prompt to see template effect
                         # print(f"\nFormatted QA Prompt Tokens ({len(qa_prompt_tokens)}):\n{self.tokenizer.decode(qa_prompt_tokens)}")
                         # Print the stored answer_text
                         print(f"\nOriginal Answer Text:\n{decoded_answer}") 
                         print(f"\nContext End Index: {context_end_idx}")
                         print("-----------------------------\n")
                         printed_example_count += 1
                else:
                     # print(f"Warning: No valid labels after masking context for QA pair in doc {doc_id}, question: {question_text[:50]}...")
                     num_skipped += 1

            except Exception as e:
                 # Added (summary) to error message
                 print(f"Error processing example {idx} (summary) in split {self.split}: {e}")
                 num_skipped += 1

        print(f"Finished processing {self.split} (summaries). Created {len(processed)} instances. Skipped {num_skipped}.")
        return [item for item in processed if item is not None]


    def __len__(self):
        return len(self._processed_data)

    def __getitem__(self, idx):
        item = self._processed_data[idx]
        # Convert to tensors here - explicitly on CPU to avoid device mismatches
        input_ids = torch.tensor(item['input_ids'], dtype=torch.long, device='cpu')
        labels = torch.tensor(item['labels'], dtype=torch.long, device='cpu')
        context_end_idx = torch.tensor(item['context_end_idx'], dtype=torch.long, device='cpu')

        # Attention mask (all ones before padding)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long, device='cpu')

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "context_end_idx": context_end_idx,
            "doc_id": item['doc_id'], # Keep doc_id for state management
            "answer_text": item['answer_text'] # <<< Return answer_text
        }


def collate_fn(batch, tokenizer, max_length):
    # Extract items - handle potential None if __getitem__ fails (shouldn't happen with current checks)
    batch = [item for item in batch if item is not None]
    if not batch: return {} # Return empty dict if batch is empty

    # Ensure all tensors are on CPU before collation to avoid device mismatch issues
    input_ids = [item['input_ids'].cpu() for item in batch]
    labels = [item['labels'].cpu() for item in batch]
    context_end_indices = [item['context_end_idx'].cpu() for item in batch]
    doc_ids = [item['doc_id'] for item in batch]
    answer_texts = [item['answer_text'] for item in batch] # <<< Extract answer texts

    tokenizer.padding_side = "right" # Important for Causal LM
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            # Add a pad token if EOS is also missing (highly unlikely for models like Llama)
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            # Warning: Model embeddings need resizing if a token was added here
            # This resize should happen in the main train script after tokenizer is loaded
            warnings.warn("Tokenizer missing EOS and PAD token. Added '[PAD]'. Ensure model embeddings are resized.")


    # Pad inputs using tokenizer.pad
    padded_inputs = tokenizer.pad(
        {"input_ids": input_ids},
        padding="longest", # Pad to longest in batch
        max_length=max_length, # Ensure not exceeding max_length
        return_tensors="pt",
        return_attention_mask=True, # Ensure attention mask is returned
    )

    # Pad labels manually to match the padded input length
    batch_max_len = padded_inputs["input_ids"].shape[1]
    padded_labels = []
    for label_seq in labels:
        # Convert label_seq to tensor if it's not already (it should be from __getitem__)
        if not isinstance(label_seq, torch.Tensor):
            label_seq = torch.tensor(label_seq, dtype=torch.long, device='cpu')
        else:
            # Ensure it's on CPU
            label_seq = label_seq.cpu()
            
        # How much padding is needed for this sequence?
        padding_length = batch_max_len - len(label_seq)
        # Pad with -100
        if padding_length >= 0:
            padded_seq = torch.cat([
                label_seq,
                torch.full((padding_length,), -100, dtype=torch.long, device='cpu')
            ])
        else:
            # If label_seq was longer than batch_max_len (shouldn't happen with truncation), truncate
            padded_seq = label_seq[:batch_max_len]
        padded_labels.append(padded_seq)

    # Stack padded labels
    final_labels = torch.stack(padded_labels)

    # Batch context_end_idx (it's just a list of ints/tensors)
    # Convert to tensor for consistency
    final_context_end_indices = torch.tensor(context_end_indices, dtype=torch.long) # <<< ADDED

    return {
        "input_ids": padded_inputs["input_ids"],
        "attention_mask": padded_inputs["attention_mask"],
        "labels": final_labels,
        "context_end_idx": final_context_end_indices, # <<< ADDED
        "doc_ids": doc_ids, # Pass doc_ids through
        "answer_texts": answer_texts # <<< Pass answer texts through
    }

def get_dataloader(config, tokenizer, split="train"):
    try:
        # Use NarrativeQADataset which now uses summaries
        dataset = NarrativeQADataset(config, tokenizer, split=split)
    except Exception as e:
        warnings.warn(f"Failed to create NarrativeQADataset (summary) for split '{split}': {e}")
        return None # Return None if dataset creation fails

    if len(dataset) == 0: # Handle empty dataset
        warnings.warn(f"NarrativeQADataset (summary) for split '{split}' is empty.")
        return None

    batch_size = config['per_device_train_batch_size'] if split == "train" else config['per_device_eval_batch_size']
    from functools import partial
    # Ensure max_length passed to collate_fn matches config's context_length
    collate_wrapper = partial(collate_fn, tokenizer=tokenizer, max_length=config['context_length'])

    num_workers = config.get('num_dataloader_workers', 4) # Get from config or default

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        collate_fn=collate_wrapper,
        num_workers=num_workers, # Use config value
        pin_memory=True,
        # Consider prefetch_factor based on num_workers
        prefetch_factor=2 if num_workers > 0 else None
    )