"""
EarningsEdge AI - Configuration
All constants, paths, API keys, and hyperparameters in one place.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ─── Project Root ─────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
PRICES_DIR = DATA_DIR / "prices"
LABELS_DIR = DATA_DIR / "labels"
LEXICON_DIR = DATA_DIR / "lexicon"
MODELS_DIR = ROOT_DIR / "models"
SIGNALS_DIR = ROOT_DIR / "signals"

# ─── API Keys (set in .env) ───────────────────────────────────────────────────
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")
ALPACA_BASE_URL = "https://paper-api.alpaca.markets"  # paper for free tier

# ─── Data Sources ─────────────────────────────────────────────────────────────
HF_EARNINGS_DATASET = "lamini/earnings-calls-qa"      # Primary HF dataset
HF_EARNINGS_DATASET_ALT = "gtfintechlab/EarningsCall" # Alternative

EDGAR_SEARCH_URL = "https://efts.sec.gov/LATEST/search-index"
EDGAR_RATE_LIMIT = 8  # requests per second (EDGAR allows 10/s; stay safe)

LM_WORDLIST_URL = (
    "https://sraf.nd.edu/files/LoughranMcDonald_MasterDictionary_3.csv"
)

# ─── Model Configs ──────────────────────────────────────────────────────────
FINBERT_BASE = "ProsusAI/finbert"
FINBERT_LABELS = ["positive", "negative", "neutral"]
FINBERT_LABEL2ID = {"positive": 0, "negative": 1, "neutral": 2}
FINBERT_ID2LABEL = {0: "positive", 1: "negative", 2: "neutral"}

FINBERT_MAX_LEN = 128          # sentence-level tokens
FINBERT_BATCH_SIZE = 16
FINBERT_LR = 2e-5
FINBERT_EPOCHS = 5
FINBERT_WARMUP_RATIO = 0.1
FINBERT_WEIGHT_DECAY = 0.01

# ─── Preprocessing ────────────────────────────────────────────────────────────
SPACY_MODEL = "en_core_web_sm"

# Speaker role keywords used in transcript diarization
OPERATOR_KEYWORDS = [
    "operator", "conference operator", "please go ahead",
    "your line is open", "thank you for joining"
]
CEO_KEYWORDS = ["chief executive officer", "ceo", "president and ceo"]
CFO_KEYWORDS = ["chief financial officer", "cfo", "chief financial"]
QA_SECTION_MARKERS = [
    "question-and-answer", "q&a", "q & a",
    "we will now begin the question-and-answer",
    "we will now take questions"
]
REMARKS_SECTION_MARKERS = [
    "prepared remarks", "management discussion",
    "opening remarks", "good morning", "good afternoon", "good evening"
]

# ─── Signal Weights ────────────────────────────────────────────────────────────
# Alpha signal composite weights (must sum to 1.0)
SIGNAL_WEIGHT_FINBERT = 0.40
SIGNAL_WEIGHT_CONFIDENCE = 0.30
SIGNAL_WEIGHT_HEDGE = 0.20
SIGNAL_WEIGHT_QA = 0.10

# Management confidence score weights (must sum to 1.0)
CONF_WEIGHT_CERTAINTY = 0.35
CONF_WEIGHT_SPECIFICITY = 0.25
CONF_WEIGHT_HEDGE_INV = 0.25
CONF_WEIGHT_QA_DELTA = 0.15

# ─── Backtesting ───────────────────────────────────────────────────────────────
BACKTEST_TRAIN_START = "2019-01-01"
BACKTEST_TRAIN_END = "2023-12-31"
BACKTEST_TEST_START = "2024-01-01"
BACKTEST_TEST_END = "2024-12-31"
RETURN_WINDOWS = [1, 3, 5]   # days post-call
QUINTILE_COUNT = 5

# ─── API ───────────────────────────────────────────────────────────────────────
API_HOST = "0.0.0.0"
API_PORT = 8000
API_KEY_HEADER = "X-API-Key"
API_MAX_LATENCY_MS = 5000  # target < 5 seconds

# ─── Reproducibility ───────────────────────────────────────────────────────────
RANDOM_SEED = 42
