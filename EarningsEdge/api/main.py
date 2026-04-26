"""
EarningsEdge AI — Phase 4 API Backend

Core FastAPI server providing real-time FinBERT and Hedge NLP analysis capabilities
along with REST endpoints to query historical alpha signals.

Run via:
    uvicorn api.main:app --reload
"""

import sys
from pathlib import Path
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel, Field
import spacy
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from config import MODELS_DIR, SIGNALS_DIR, FINBERT_BASE
from preprocess import segment_sentences, clean_text
from compute_signals import process_transcript

log = logging.getLogger("api")
logging.basicConfig(level=logging.INFO)

HEDGE_NER_DIR = MODELS_DIR / "hedge_ner"
FINBERT_SAVE_DIR = MODELS_DIR / "finbert" / "best_model"
SIGNALS_FILE = SIGNALS_DIR / "transcript_signals.parquet"

# Models stored globally for active caching
models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle event. Load expensive models strictly on startup to prevent latency."""
    log.info("Booting ML Models into memory...")
    
    # 1. spaCy sentence segmentation wrapper
    try:
        nlp_split = spacy.load("en_core_web_sm", disable=["ner", "tagger", "lemmatizer"])
        models["nlp_split"] = nlp_split
    except Exception as e:
        log.warning(f"Failed to load en_core_web_sm: {e}")
        
    # 2. Hedge NER
    try:
        nlp_ner = spacy.load(str(HEDGE_NER_DIR))
        models["nlp_ner"] = nlp_ner
    except Exception as e:
        log.warning(f"Failed to load Hedge NER: {e}")
        
    # 3. FinBERT
    model_path = str(FINBERT_SAVE_DIR) if FINBERT_SAVE_DIR.exists() else FINBERT_BASE
    try:
        tokenizer = AutoTokenizer.from_pretrained(FINBERT_BASE)
        clf_model = AutoModelForSequenceClassification.from_pretrained(model_path)
        # Assuming CPU deployment for the API
        finbert_clf = pipeline("text-classification", model=clf_model, tokenizer=tokenizer, truncation=True)
        models["finbert_clf"] = finbert_clf
    except Exception as e:
        log.error(f"Failed to load FinBERT: {e}")
        
    # 4. In-Memory Signals Parquet Lookups
    try:
        df = pd.read_parquet(SIGNALS_FILE)
        models["signals_db"] = df
    except Exception as e:
        log.warning(f"Failed to load signals DB. Missing Phase 2 run? {e}")

    log.info("API Boot sequence completed.")
    yield
    
    log.info("API Shutting down, cleaning up globals.")
    models.clear()


app = FastAPI(
    title="EarningsEdge AI Alpha API",
    description="Realtime NLP analytics for Earnings Call processing.",
    version="1.0.0",
    lifespan=lifespan
)

# ─── Schemas ──────────────────────────────────────────────────────────────────
class AnalysisRequest(BaseModel):
    text: str = Field(..., description="Raw text of the earnings call transcript to analyze.")

class SignalResponse(BaseModel):
    confidence_score: float
    remarks_sentiment: float
    qa_sentiment: float
    hedge_density: float
    ceo_cfo_divergence: float
    qa_pushback_signal: float
    total_words: int
    sentences_parsed: int


# ─── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/health")
def read_health():
    """Confirms Server Status and ML Model availability."""
    return {
        "status": "healthy",
        "models_loaded": {
            "sentence_segmenter": "nlp_split" in models,
            "hedge_ner": "nlp_ner" in models,
            "finbert_tone": "finbert_clf" in models,
            "database_ready": "signals_db" in models
        }
    }

@app.post("/analyze", response_model=SignalResponse)
def run_realtime_analysis(payload: AnalysisRequest):
    """
    Takes pure raw string text of a new transcript and pushes it entirely through
    the core NLP pipeline: Sentence Split -> Regex Preprocess -> FinBERT -> 
    Hedge NER -> Mathematical Output Matrix.
    """
    if "finbert_clf" not in models or "nlp_ner" not in models:
        raise HTTPException(status_code=503, detail="NLP models failed to initialize. Cannot process.")
        
    text = payload.text.strip()
    if len(text) < 50:
        raise HTTPException(status_code=400, detail="Text too short. Provide valid earnings record.")
        
    # Note: Realtime endpoint doesn't dynamically resolve exact speaker roles 
    # natively without specific formatting headers, so we default segments to 'remarks' & 'ceo'
    clean_text_payload = clean_text(text)
    doc = models["nlp_split"](clean_text_payload)
    
    records = []
    for idx, sent in enumerate(doc.sents):
        cleaned = sent.text.strip()
        if len(cleaned.split()) > 3:
            records.append({
                "sentence_id": idx,
                "text": cleaned,
                "speaker_role": "ceo", # Assumed for instant eval
                "section": "remarks",   # Assumed for instant eval
                "word_count": len(cleaned.split())
            })
            
    if not records:
        raise HTTPException(status_code=400, detail="Parsing failed to extract valid sentences.")
        
    # Execute heavy inference
    try:
        outputs = process_transcript(records, models["finbert_clf"], models["nlp_ner"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
        
    if not outputs:
        raise HTTPException(status_code=500, detail="Signal formulation yielded null.")

    # Convert to expected JSON format
    return SignalResponse(
        confidence_score=outputs.get("confidence_score", 0.0),
        remarks_sentiment=outputs.get("remarks_sentiment", 0.0),
        qa_sentiment=outputs.get("qa_sentiment", 0.0),
        hedge_density=outputs.get("hedge_density", 0.0),
        ceo_cfo_divergence=outputs.get("ceo_cfo_divergence", 0.0),
        qa_pushback_signal=outputs.get("qa_pushback_signal", 0.0),
        total_words=sum(r["word_count"] for r in records),
        sentences_parsed=len(records)
    )

@app.get("/signals/{ticker}")
def get_historical_signals(ticker: str):
    """Retrieves computed backtest matrix records for a specific symbol."""
    if "signals_db" not in models:
        raise HTTPException(status_code=503, detail="Database parity offline.")
        
    df = models["signals_db"]
    subset = df[df["ticker"].str.upper() == ticker.upper()]
    
    if subset.empty:
        raise HTTPException(status_code=404, detail=f"No call records found for {ticker}")
        
    records = subset.to_dict(orient="records")
    return {"ticker": ticker.upper(), "calls_indexed": len(records), "data": records}
