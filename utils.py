import torch
import random
import numpy as np
import os
import evaluate
import nltk
from nltk.tokenize import sent_tokenize
import warnings
import traceback

# --- set_seed function ---
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"Set random seed to {seed}")

# --- save_checkpoint function ---
def save_checkpoint(model, optimizer, scheduler, epoch, step, output_dir, is_best=False, is_lora=False):
    """Saves model, optimizer, scheduler state, handling PEFT."""
    checkpoint_dir = os.path.join(output_dir, f"checkpoint-{step}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Saving checkpoint to {checkpoint_dir}...")

    # Save model state (PEFT adapter or full model)
    # Use model.save_pretrained for PEFT models to save adapter correctly
    # Need to handle the case where model might be wrapped by Accelerator/DDP
    unwrapped_model = model # Assume model is already unwrapped if needed

    if is_lora and hasattr(unwrapped_model, "save_pretrained") and callable(getattr(unwrapped_model, "save_pretrained", None)):
         try:
             unwrapped_model.save_pretrained(checkpoint_dir)
             print(f"Saved PEFT adapter checkpoint.")
         except Exception as e:
             print(f"!!! ERROR during unwrapped_model.save_pretrained: {e}")
             traceback.print_exc()
             print(f"Falling back to saving full PeftModel state_dict instead.")
             try:
                 torch.save(unwrapped_model.state_dict(), os.path.join(checkpoint_dir, "pytorch_model.bin"))
             except Exception as save_err:
                 print(f"!!! ERROR during fallback save_state_dict: {save_err}")
                 traceback.print_exc()
    else:
        try:
             torch.save(unwrapped_model.state_dict(), os.path.join(checkpoint_dir, "pytorch_model.bin"))
             print(f"Saved full model state_dict (non-LoRA).")
        except Exception as save_err:
             print(f"!!! ERROR during non-LoRA save_state_dict: {save_err}")
             traceback.print_exc()

    # Save optimizer and scheduler state
    try:
         torch.save(optimizer.state_dict(), os.path.join(checkpoint_dir, "optimizer.pt"))
         if scheduler is not None:
             torch.save(scheduler.state_dict(), os.path.join(checkpoint_dir, "scheduler.pt"))
         state = {"epoch": epoch, "step": step}
         torch.save(state, os.path.join(checkpoint_dir, "trainer_state.pt"))
    except Exception as state_save_err:
         print(f"!!! ERROR saving optimizer/scheduler/trainer state: {state_save_err}")
         traceback.print_exc()

    if is_best:
        best_dir = os.path.join(output_dir, "best_model")
        os.makedirs(best_dir, exist_ok=True)
        print(f"Saving best model checkpoint to {best_dir}...")
        if is_lora and hasattr(unwrapped_model, "save_pretrained") and callable(getattr(unwrapped_model, "save_pretrained", None)):
             try:
                  unwrapped_model.save_pretrained(best_dir)
                  print(f"Saved best PEFT adapter checkpoint.")
             except Exception as e:
                  print(f"!!! ERROR during best model unwrapped_model.save_pretrained: {e}")
                  traceback.print_exc()
                  print(f"Falling back to saving best model as full PeftModel state_dict instead.")
                  try:
                      torch.save(unwrapped_model.state_dict(), os.path.join(best_dir, "pytorch_model.bin"))
                  except Exception as save_err:
                      print(f"!!! ERROR during best model fallback save_state_dict: {save_err}")
                      traceback.print_exc()
        else:
             import shutil
             try:
                 shutil.copyfile(os.path.join(checkpoint_dir, "pytorch_model.bin"), os.path.join(best_dir, "pytorch_model.bin"))
                 print(f"Copied best model state_dict (non-LoRA).")
             except FileNotFoundError:
                 print(f"!!! ERROR copying best model: Source file not found at {os.path.join(checkpoint_dir, 'pytorch_model.bin')}")
             except Exception as e:
                 print(f"!!! ERROR copying best model state_dict: {e}")
                 traceback.print_exc()


# --- Evaluation Metrics ---
rouge_metric = None
bleu_metric = None

def init_metrics():
    global rouge_metric, bleu_metric
    if rouge_metric is not None: return # Already initialized
    
    # Robustly check and download NLTK data
    try:
        nltk.data.find('tokenizers/punkt')
    except (nltk.downloader.DownloadError, LookupError):
        print("NLTK punkt resource not found. Downloading...")
        nltk.download('punkt', quiet=True)
    
    # Also download punkt_tab specifically (mentioned in the error)
    try:
        print("Attempting to download punkt_tab resource specifically...")
        nltk.download('punkt_tab', quiet=True)
    except Exception as e:
        print(f"Note: punkt_tab download attempt resulted in: {e}")
        print("This is normal if punkt_tab is part of another package.")
        
    # Verify if punkt is available after download attempts
    try:
        nltk.data.find('tokenizers/punkt')
        print("NLTK punkt resource found.")
    except LookupError:
        warnings.warn("Failed to find or download NLTK punkt resource. ROUGE scores might be inaccurate.", UserWarning)

    # Alternative: Use a try-except in the compute_metrics function
    # Let's modify our approach to make sent_tokenize more robust
    try:
        rouge_metric = evaluate.load("rouge")
        bleu_metric = evaluate.load("sacrebleu")
        print("Evaluation metrics (ROUGE, BLEU) initialized.")
    except Exception as e:
        print(f"Failed to load evaluation metrics: {e}")
        raise

def compute_metrics(predictions_ids, references_ids, tokenizer):
    """Computes ROUGE, BLEU, EM metrics from token IDs."""
    if rouge_metric is None or bleu_metric is None: init_metrics()

    # Decode predictions and labels
    # Need to handle potential padding tokens if generate doesn't strip them
    decoded_preds = [tokenizer.decode(p, skip_special_tokens=True) for p in predictions_ids]
    # Ignore -100 in labels - CORRECTED FILTERING
    decoded_labels = [
        tokenizer.decode([token_id for token_id in l if token_id != -100], skip_special_tokens=True)
        for l in references_ids
    ]

    # Clean empty strings
    decoded_preds = [p.strip() if p else "<empty>" for p in decoded_preds]
    decoded_labels = [l.strip() if l else "<empty>" for l in decoded_labels]

    # ROUGE
    try:
        # First try with sentence tokenization
        try:
            preds_rouge = ["\n".join(sent_tokenize(p)) for p in decoded_preds]
            labels_rouge = ["\n".join(sent_tokenize(l)) for l in decoded_labels]
        except (LookupError, NameError) as e:
            # If sent_tokenize fails, use simple line splitting as fallback
            print(f"Sentence tokenization failed ({e}), using text as-is for ROUGE calculation.")
            preds_rouge = decoded_preds
            labels_rouge = decoded_labels
            
        rouge_result = rouge_metric.compute(predictions=preds_rouge, references=labels_rouge, use_stemmer=True)
        rouge_scores = {k: v * 100 for k, v in rouge_result.items()} # Use default rouge types
    except Exception as e:
        print(f"Error computing ROUGE: {e}")
        rouge_scores = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0, "rougeLsum": 0.0}

    # BLEU
    try:
        labels_bleu = [[l] for l in decoded_labels] # sacrebleu expects list of references
        bleu_result = bleu_metric.compute(predictions=decoded_preds, references=labels_bleu)
        bleu_score = bleu_result["score"]
    except Exception as e:
        print(f"Error computing BLEU: {e}")
        bleu_score = 0.0

    # Exact Match (EM)
    exact_match = sum(1 for pred, label in zip(decoded_preds, decoded_labels) if pred == label)
    em_score = (exact_match / len(predictions_ids)) * 100 if predictions_ids else 0.0

    results = {
        **rouge_scores,
        "bleu": bleu_score,
        "exact_match": em_score,
    }
    return {k: round(v, 4) for k, v in results.items()}