"""
EarningsEdge AI — Phase 2.3 & 2.5
Section-Aware Sentiment Aggregation & Management Confidence Score

Loads FinBERT & Hedge NER to process sentence records for each transcript.
Aggregates section sentiments and computes composite Alpha signals.

Output:
    signals/transcript_signals.parquet
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import spacy
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    PROCESSED_DIR, MODELS_DIR, SIGNALS_DIR, FINBERT_BASE, FINBERT_LABEL2ID,
    CONF_WEIGHT_CERTAINTY, CONF_WEIGHT_SPECIFICITY, CONF_WEIGHT_HEDGE_INV, CONF_WEIGHT_QA_DELTA
)
from lm_lexicon import compute_lexicon_scores

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("compute_signals")

SIGNALS_DIR.mkdir(parents=True, exist_ok=True)
HEDGE_NER_DIR = MODELS_DIR / "hedge_ner"
FINBERT_SAVE_DIR = MODELS_DIR / "finbert" / "best_model"


def load_models():
    """Load the FinBERT classifier and Hedge NER model."""
    log.info("Loading models...")
    device = 0 if torch.cuda.is_available() else -1
    
    # 1. Load Custom Hedge NER
    try:
        nlp_ner = spacy.load(str(HEDGE_NER_DIR))
    except Exception as e:
        log.error(f"Failed to load Hedge NER. Did you run hedge_ner.py? {e}")
        sys.exit(1)
        
    # 2. Load FinBERT (fine-tuned if available, else baseline)
    model_path = str(FINBERT_SAVE_DIR) if FINBERT_SAVE_DIR.exists() else FINBERT_BASE
    log.info(f"Loading FinBERT from {model_path}...")
    
    tokenizer = AutoTokenizer.from_pretrained(FINBERT_BASE)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    finbert_clf = pipeline("text-classification", model=model, tokenizer=tokenizer, device=device, truncation=True)
    
    return finbert_clf, nlp_ner


def process_transcript(records: list[dict], finbert_clf, nlp_ner) -> dict | None:
    """Computes aggregate signals for a single transcript."""
    if not records:
        return None
        
    ticker = records[0].get("ticker", "UNKNOWN")
    date = records[0].get("date", "19000101")
    
    texts = [rec["text"] for rec in records]
    
    # Batch predict sentiment
    # Convert labels to numeric score: pos=1, neu=0, neg=-1
    try:
        sentiments = finbert_clf(texts, batch_size=16)
        scores = []
        for s in sentiments:
            label = s["label"].lower()
            if label == "positive": scores.append(1.0)
            elif label == "negative": scores.append(-1.0)
            else: scores.append(0.0)
    except Exception as e:
        log.warning(f"FinBERT inference failed for {ticker}_{date}: {e}")
        return None

    # Track metrics
    metrics = {
        "remarks_ceo": [],
        "remarks_cfo": [],
        "qa_mgmt": [],
        "qa_analyst": [],
        "all_hedges": 0,
        "all_words": 0,
        "certainty_ratios": []
    }
    
    # Aggregate by section
    for i, rec in enumerate(records):
        score = scores[i]
        role = rec.get("speaker_role", "other")
        section = rec.get("section", "unknown")
        
        if section == "remarks":
            if role == "ceo": metrics["remarks_ceo"].append(score)
            elif role == "cfo": metrics["remarks_cfo"].append(score)
        elif section == "qa":
            if role in ("ceo", "cfo"): metrics["qa_mgmt"].append(score)
            elif role == "analyst": metrics["qa_analyst"].append(score)
            
        # Hedge density & Certainty via NER + Lexicon
        text = rec["text"]
        doc = nlp_ner(text)
        metrics["all_hedges"] += len([e for e in doc.ents if e.label_ == "HEDGE"])
        metrics["all_words"] += rec["word_count"]
        
        lex_scores = compute_lexicon_scores(text)
        metrics["certainty_ratios"].append(lex_scores.get("strong_modal", 0.0))

    # Calculate Aggregates
    def safe_mean(lst): return float(np.mean(lst)) if lst else 0.0
    
    remarks_sentiment = safe_mean(metrics["remarks_ceo"] + metrics["remarks_cfo"])
    qa_sentiment = safe_mean(metrics["qa_mgmt"])
    qa_analyst_sentiment = safe_mean(metrics["qa_analyst"])
    
    # Divergences
    ceo_cfo_divergence = 0.0
    if metrics["remarks_ceo"] and metrics["remarks_cfo"]:
        ceo_cfo_divergence = safe_mean(metrics["remarks_ceo"]) - safe_mean(metrics["remarks_cfo"])
        
    qa_analyst_tone_delta = qa_sentiment - qa_analyst_sentiment 
    
    # Normalized components for confidence score (scaled 0-1)
    hedge_density = (metrics["all_hedges"] / metrics["all_words"]) if metrics["all_words"] > 0 else 0.0
    hedge_density_norm = min(1.0, hedge_density * 10) # scale up for visibility
    
    certainty_ratio = safe_mean(metrics["certainty_ratios"])
    certainty_norm = min(1.0, certainty_ratio * 5)
    
    # Mocking specific guidance for MVP as 0.5 default if numeric digits exist
    numeric_count = sum(1 for rec in records if any(char.isdigit() for char in rec["text"]))
    guidance_specificity = min(1.0, numeric_count / max(1, len(records)) * 2)

    # 2.5 Management Confidence Score Computation
    # Base confidence from sentiment (positive sentiment should increase confidence)
    sentiment_confidence = (remarks_sentiment + 1) / 2  # Normalize -1 to 1 -> 0 to 1
    
    confidence_score = (
        0.30 * sentiment_confidence +  # 30% weight on actual sentiment
        CONF_WEIGHT_CERTAINTY * certainty_norm + 
        CONF_WEIGHT_SPECIFICITY * guidance_specificity +
        CONF_WEIGHT_HEDGE_INV * (1.0 - hedge_density_norm) +
        CONF_WEIGHT_QA_DELTA * max(0.0, min(1.0, (qa_analyst_tone_delta + 1) / 2))
    )
    
    # Ensure -1 to 1 theoretical range by shifting and scaling
    confidence_score_scaled = (confidence_score * 2) - 1.0

    return {
        "ticker": ticker,
        "call_date": date,
        "remarks_sentiment": remarks_sentiment,
        "qa_sentiment": qa_sentiment,
        "confidence_score": confidence_score_scaled,
        "hedge_density": hedge_density,
        "ceo_cfo_divergence": ceo_cfo_divergence,
        "qa_pushback_signal": qa_analyst_tone_delta
    }


def compute_all_signals():
    """Iterate through all transcripts and generate the final output parquet."""
    finbert_clf, nlp_ner = load_models()
    
    processed_files = list(PROCESSED_DIR.glob("*_processed.json"))
    log.info(f"Found {len(processed_files)} transcripts to score.")
    
    results = []
    
    for i, fpath in enumerate(processed_files):
        with open(fpath, encoding="utf-8") as f:
            records = json.load(f)
            
        signal_data = process_transcript(records, finbert_clf, nlp_ner)
        if signal_data:
            results.append(signal_data)
            
        if (i + 1) % 50 == 0:
            log.info(f"  Processed {i+1}/{len(processed_files)} transcripts...")
            
    if not results:
        log.error("No signals generated.")
        return
        
    df = pd.DataFrame(results)
    out_path = SIGNALS_DIR / "transcript_signals.parquet"
    df.to_parquet(out_path, index=False)
    
    log.info("=" * 60)
    log.info("SIGNALS COMPUTATION SUMMARY")
    log.info("=" * 60)
    log.info(f"  Transcripts scored: {len(df)}")
    log.info(f"  Outputs saved to:   {out_path}")
    log.info(f"  Sentiment Range:    [{df['remarks_sentiment'].min():.2f}, {df['remarks_sentiment'].max():.2f}]")
    log.info(f"  Confidence Range:   [{df['confidence_score'].min():.2f}, {df['confidence_score'].max():.2f}]")
    log.info("=" * 60)
    
    if len(df) >= 450:
        log.info("✓ Phase 2 checkpoint: Signals matrix generated for backtesting.")
    else:
        log.warning(f"⚠ Only generated {len(df)} signals. Target was 450+.")


def main():
    parser = argparse.ArgumentParser("Compute Signals")
    parser.add_argument("--process", action="store_true", help="Process all processed transcripts into signals")
    args = parser.parse_args()
    
    if args.process:
        compute_all_signals()


if __name__ == "__main__":
    main()
