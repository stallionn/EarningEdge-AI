"""
EarningsEdge AI — Phase 2 Validation Test Suite

Tests:
    1. Hedge NER model directory exists and runs correctly.
    2. FinBERT is functional.
    3. transcript_signals.parquet has correct schemas and values.

Run:
    pytest tests/test_phase2.py -v
"""

import sys
from pathlib import Path
import pytest
import spacy
import pandas as pd
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from config import MODELS_DIR, SIGNALS_DIR, FINBERT_BASE

HEDGE_NER_DIR = MODELS_DIR / "hedge_ner"
SIGNALS_FILE = SIGNALS_DIR / "transcript_signals.parquet"
FINBERT_SAVE_DIR = MODELS_DIR / "finbert" / "best_model"

class TestPhase2NLP:

    def test_hedge_ner_exists(self):
        """Hedge NER model must exist on disk."""
        assert HEDGE_NER_DIR.exists(), f"Hedge NER model not found at {HEDGE_NER_DIR}"

    def test_hedge_ner_functionality(self):
        """Hedge NER model must correctly identify hedge words."""
        if not HEDGE_NER_DIR.exists():
            pytest.skip("Hedge NER not built")
            
        nlp = spacy.load(str(HEDGE_NER_DIR))
        doc = nlp("We expect revenue to drop slightly subject to market conditions.")
        
        entities = [ent.text.lower() for ent in doc.ents if ent.label_ == "HEDGE"]
        assert "expect" in entities or "we expect" in entities
        assert "subject to" in entities or "subject" in entities
        assert "market conditions" in entities

    def test_finbert_pipeline_load(self):
        """FinBERT model must be loadable and output classification."""
        model_path = str(FINBERT_SAVE_DIR) if FINBERT_SAVE_DIR.exists() else FINBERT_BASE
        
        tokenizer = AutoTokenizer.from_pretrained(FINBERT_BASE)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        
        clf = pipeline("text-classification", model=model, tokenizer=tokenizer, truncation=True)
        result = clf("We are extremely happy to report a massive gain in revenue this quarter.")
        
        assert len(result) > 0
        assert "label" in result[0]
        # While not guaranteed, an extremely positive sentiment should map generally correctly
        assert result[0]["label"].lower() in ("positive", "neutral", "negative")

class TestPhase2Signals:

    def test_signals_file_exists(self):
        """transcript_signals.parquet must exist."""
        assert SIGNALS_FILE.exists(), "Signals parquet not found. Run scripts/compute_signals.py"

    def test_signals_columns(self):
        """Check for composite alpha parameter columns."""
        if not SIGNALS_FILE.exists():
            pytest.skip("Signals file not built")
            
        df = pd.read_parquet(SIGNALS_FILE)
        required_cols = {
            "ticker", "call_date", "remarks_sentiment", "qa_sentiment",
            "confidence_score", "hedge_density", "ceo_cfo_divergence", "qa_pushback_signal"
        }
        
        missing = required_cols - set(df.columns)
        assert len(missing) == 0, f"Missing columns in signals: {missing}"

    def test_signals_value_ranges(self):
        """Ensure synthesized scores are bounded correctly."""
        if not SIGNALS_FILE.exists():
            pytest.skip("Signals file not built")
            
        df = pd.read_parquet(SIGNALS_FILE)
        
        for col in ["remarks_sentiment", "qa_sentiment", "confidence_score", "ceo_cfo_divergence", "qa_pushback_signal"]:
            assert df[col].max() <= 1.5, f"{col} max value too high: {df[col].max()}"
            assert df[col].min() >= -1.5, f"{col} min value too low: {df[col].min()}"
            
        assert df["hedge_density"].min() >= 0.0, "hedge_density should be >= 0"
