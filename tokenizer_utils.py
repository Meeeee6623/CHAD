import os
from transformers import AutoTokenizer
import warnings

def get_tokenizer(model_name_or_path, cache_dir=None):
    """Loads the Llama 3 tokenizer using AutoTokenizer."""
    try:
        # trust_remote_code might be needed depending on HF model repo setup
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            cache_dir=cache_dir,
            trust_remote_code=False, # Set to True if prompted/required
            use_fast=True
        )

        # Llama 3.2 Instruct Specific Tokens (Verify IDs from official source)
        special_tokens_map = {
            # BOS is usually handled by template/added automatically
            # "bos_token": {"content": "<|begin_of_text|>", "id": 128000},
            "eos_token": {"content": "<|eot_id|>", "id": 128009, "single_word": False}, # End of Turn primarily used
            # "<|end_of_text|>" (128001) is also available but eot_id often signals generation stop
        }

        # Add PAD token if missing, crucial for batching
        if tokenizer.pad_token is None:
            print("No PAD token found. Adding and using EOS token as PAD.")
            # Using EOS as PAD is common practice, but check if Llama 3.2 has specific recs
            num_added = tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})
            if num_added > 0: print(f"Added PAD token: {tokenizer.pad_token}")
            # No need to resize embeddings here, will handle in train script after loading model

        # Check other special tokens used by the chat template - AutoTokenizer *should* load these
        # e.g., <|start_header_id|> (128006), <|end_header_id|> (128007)
        # If issues arise, might need manual add_special_tokens here too

        # Ensure padding side is correct for Causal LM training (usually right)
        tokenizer.padding_side = "right"

        print(f"Tokenizer loaded. Vocab size: {tokenizer.vocab_size}")
        print(f"BOS: '{tokenizer.bos_token}' (ID: {tokenizer.bos_token_id}), EOS: '{tokenizer.eos_token}' (ID: {tokenizer.eos_token_id}), PAD: '{tokenizer.pad_token}' (ID: {tokenizer.pad_token_id})")
        # print(f"Chat Template:\n{getattr(tokenizer, 'chat_template', 'Not Set')}")

        return tokenizer

    except Exception as e:
        print(f"Error loading tokenizer for {model_name_or_path}: {e}")
        raise