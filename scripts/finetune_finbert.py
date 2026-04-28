"""
EarningsEdge AI — Phase 2.1 & 2.2
FinBERT Fine-Tuning & Baseline Testing

Loads sentences from processed dataset.
Labels them using LM Lexicon (Weak Supervision) if labels missing.
Fine-tunes ProsusAI/finbert to improve domain-specific performance.
Uses PEFT (LoRA) to allow fast CPU fine-tuning.

Usage:
    python scripts/finetune_finbert.py --baseline
    python scripts/finetune_finbert.py --train --samples 1000
"""

import argparse
import json
import logging
import sys
import os
from pathlib import Path

import evaluate
import numpy as np
import torch
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from peft import get_peft_model, LoraConfig, TaskType

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    PROCESSED_DIR, MODELS_DIR, FINBERT_BASE, FINBERT_LABEL2ID, FINBERT_ID2LABEL,
    FINBERT_MAX_LEN, FINBERT_BATCH_SIZE, FINBERT_LR, FINBERT_EPOCHS,
    RANDOM_SEED
)
from lm_lexicon import compute_lexicon_scores

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("finetune")

MODELS_DIR.mkdir(parents=True, exist_ok=True)
FINBERT_SAVE_DIR = MODELS_DIR / "finbert"

# ─── Data Prep ────────────────────────────────────────────────────────────────
def load_and_label_data(max_samples: int = 2000) -> list[dict]:
    """
    Load sentences from processed data and auto-label using LM lexicon.
    Weak supervision: sentiment_net > 0 -> positive, < 0 -> negative, else neutral.
    """
    log.info(f"Loading and auto-labeling up to {max_samples} sentences...")
    sent_list = []
    
    # Balance classes loosely
    pos_count = neg_count = neu_count = 0
    limit_per_class = max_samples // 3

    for fpath in PROCESSED_DIR.glob("*_processed.json"):
        with open(fpath, encoding="utf-8") as f:
            records = json.load(f)
            
        for rec in records:
            text = rec["text"]
            if len(text.split()) < 5:
                continue
            
            scores = compute_lexicon_scores(text)
            net = scores["sentiment_net"]
            
            if net > 0.01:
                label = "positive"
                if pos_count >= limit_per_class: continue
                pos_count += 1
            elif net < -0.01:
                label = "negative"
                if neg_count >= limit_per_class: continue
                neg_count += 1
            else:
                label = "neutral"
                if neu_count >= limit_per_class: continue
                neu_count += 1
                
            sent_list.append({"text": text, "label": FINBERT_LABEL2ID[label]})
            
            if len(sent_list) >= max_samples:
                break
        if len(sent_list) >= max_samples:
            break
            
    log.info(f"Loaded {len(sent_list)} records. Distributed: Pos={pos_count}, Neg={neg_count}, Neu={neu_count}")
    return sent_list


def create_hf_dataset(data: list[dict], tokenizer) -> tuple[Dataset, Dataset, Dataset]:
    """Train/Val/Test split and tokenize."""
    texts = [x["text"] for x in data]
    labels = [x["label"] for x in data]
    
    # 70/15/15 Split
    X_train, X_temp, y_train, y_temp = train_test_split(
        texts, labels, test_size=0.3, random_state=RANDOM_SEED
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=RANDOM_SEED
    )
    
    def to_dataset(X, y):
        encodings = tokenizer(X, truncation=True, padding="max_length", max_length=FINBERT_MAX_LEN)
        return Dataset.from_dict({
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "labels": y
        })

    return to_dataset(X_train, y_train), to_dataset(X_val, y_val), to_dataset(X_test, y_test)


# ─── Baseline Assessment ──────────────────────────────────────────────────────
def run_baseline(test_dataset, tokenizer):
    """Run off-the-shelf FinBERT on test data."""
    log.info("Running baseline evaluation on ProsusAI/finbert...")
    model = AutoModelForSequenceClassification.from_pretrained(
        FINBERT_BASE, num_labels=3, id2label=FINBERT_ID2LABEL, label2id=FINBERT_LABEL2ID
    )
    
    # Determine device safely
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    trainer = Trainer(model=model)
    results = trainer.predict(test_dataset)
    
    preds = np.argmax(results.predictions, axis=-1)
    
    metric = evaluate.load("f1")
    f1_res = metric.compute(predictions=preds, references=results.label_ids, average="macro")
    log.info(f"Baseline Macro F1: {f1_res['f1']:.4f}")
    return f1_res["f1"]


# ─── Fine-tuning ──────────────────────────────────────────────────────────────
def compute_metrics(eval_pred):
    metric_f1 = evaluate.load("f1")
    metric_acc = evaluate.load("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    f1 = metric_f1.compute(predictions=predictions, references=labels, average="macro")["f1"]
    acc = metric_acc.compute(predictions=predictions, references=labels)["accuracy"]
    return {"f1": f1, "accuracy": acc}


def fine_tune_model(train_dataset, val_dataset, test_dataset):
    """Fine-tune the model with PEFT (LoRA) for quick inference on CPU/Edge."""
    log.info("Preparing model for Fine-Tuning (using LoRA PEFT for speed)...")
    model = AutoModelForSequenceClassification.from_pretrained(
        FINBERT_BASE, num_labels=3, id2label=FINBERT_ID2LABEL, label2id=FINBERT_LABEL2ID
    )
    
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS, 
        inference_mode=False, 
        r=8, 
        lora_alpha=16, 
        lora_dropout=0.1
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    training_args = TrainingArguments(
        output_dir=str(FINBERT_SAVE_DIR / "checkpoints"),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=FINBERT_LR,
        per_device_train_batch_size=FINBERT_BATCH_SIZE,
        per_device_eval_batch_size=FINBERT_BATCH_SIZE,
        num_train_epochs=FINBERT_EPOCHS,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        use_cpu=not torch.cuda.is_available(), # Fallback securely
        logging_dir=str(FINBERT_SAVE_DIR / "logs"), 
        report_to="none" # Keep it local
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    
    log.info("Starting Fine-Tuning...")
    trainer.train()
    
    log.info("Evaluating on Test Set...")
    test_results = trainer.evaluate(test_dataset)
    f1_score = test_results.get("eval_f1", 0.0)
    log.info(f"Fine-Tuned Macro F1 on Test Set: {f1_score:.4f}")
    
    if f1_score > 0.78:
        log.info("✓ Success: Reached target > 0.78 F1 score!")
    else:
        log.warning("⚠ Warning: F1 score target of 0.78 was not reached.")
        
    model.save_pretrained(str(FINBERT_SAVE_DIR / "best_model"))
    log.info(f"Model saved to {FINBERT_SAVE_DIR / 'best_model'}")
    return f1_score


def main():
    parser = argparse.ArgumentParser("FinBERT FineTuning")
    parser.add_argument("--baseline", action="store_true", help="Run baseline evaluate only")
    parser.add_argument("--train", action="store_true", help="Run fine-tuning")
    parser.add_argument("--samples", type=int, default=150, help="Number of samples to use (low default for fast dev/CPU)")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(FINBERT_BASE)
    
    data = load_and_label_data(max_samples=args.samples)
    if not data:
        log.error("No processed data found. Run preprocess.py first")
        sys.exit(1)
        
    train_ds, val_ds, test_ds = create_hf_dataset(data, tokenizer)
    
    if args.baseline:
        run_baseline(test_ds, tokenizer)
    
    if args.train:
        fine_tune_model(train_ds, val_ds, test_ds)


if __name__ == "__main__":
    main()
