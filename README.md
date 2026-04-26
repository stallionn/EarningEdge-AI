EarningsEdge AI — Complete Project Workflow
Capstone BS-AI | NLP Pipeline for Earnings Call Sentiment & Alpha Signal Generation No shortcuts. No placeholders. This is the exact execution roadmap to deliver best-in-class results.

Overview
EarningsEdge is a 3-layer NLP pipeline:

Data Ingestion Layer — Collect & preprocess earnings call transcripts + price data
NLP Model Layer — Fine-tuned FinBERT + hedge NER + management confidence scoring
Alpha Signal Layer — Cross-reference sentiment scores with price returns, backtest, REST API
Target Metrics (must hit for full marks):

Metric	MVP Target	Stretch Goal
F1 Score (Sentiment Model)	> 0.78	> 0.85
Hedge Phrase Recall	> 0.80	> 0.88
Signal-Return Correlation (Pearson r)	> 0.20	> 0.35
Backtest Sharpe Ratio	> 0.8	> 1.2
API Latency per transcript	< 5 sec	< 2 sec
Transcripts processed	500+	2,000+
False positive rate (signals)	< 20%	< 12%
Phase 1: Data Collection & Preprocessing
Goal: Have clean, speaker-diarized, tokenized transcript data + aligned price data ready for modeling.

1.1 Environment Setup
 Create Python virtual environment (venv or conda)
 Install all dependencies (pinned versions):
transformers, datasets, torch, spacy, fastapi, uvicorn
pandas, numpy, scikit-learn, scipy
alpaca-trade-api or alpaca-py
requests, beautifulsoup4, lxml, tqdm
matplotlib, seaborn, plotly
evaluate, accelerate, peft (for fine-tuning)
 Create requirements.txt with pinned versions
 Create project folder structure:
EarningsEdge/
├── data/
│   ├── raw/          # raw transcripts (JSON/txt)
│   ├── processed/    # speaker-diarized, cleaned
│   ├── prices/       # OHLCV from Alpaca
│   └── labels/       # sentiment labels
├── models/
│   ├── finbert/      # fine-tuned checkpoint
│   └── hedge_ner/    # custom NER model
├── signals/          # backtest results, signal scores
├── api/              # FastAPI app
├── notebooks/        # EDA, experiments
├── tests/            # unit tests
└── scripts/          # data collection scripts
1.2 Transcript Data Collection (Primary: HuggingFace EarningsCall Dataset)
 Load EarningsCall dataset from Hugging Face (datasets library)
Dataset: lamini/earnings-calls-qa or the EarningsCall corpus (2,700+ transcripts)
Validate columns: ticker, date, transcript_text, section labels
 Fallback: SEC EDGAR 8-K full-text search API
Endpoint: https://efts.sec.gov/LATEST/search-index?q=%22earnings+call%22&dateRange=custom&startdt=2019-01-01&enddt=2024-12-31&forms=8-K
Write scripts/edgar_scraper.py — paginated HTTP pull, rate-limit-safe (10 req/sec max)
Parse filing text: strip boilerplate, extract transcript section
 Save raw transcripts as data/raw/{ticker}_{YYYYMMDD}.json
Fields: ticker, date, raw_text, source
1.3 Price Data Collection (Alpaca Markets API)
 Register for Alpaca free API key (paper/live)
 Write scripts/price_fetcher.py:
Pull daily OHLCV for each ticker, ±10 days around each earnings call date
Compute: 1-day, 3-day, 5-day post-call price return
Store in data/prices/{ticker}.parquet
 Handle missing data: delisted tickers, trading halts, weekends/holidays
1.4 Speaker Diarization & Preprocessing
 Write scripts/preprocess.py:
Section splitting: separate Prepared Remarks vs. Q&A sections via regex patterns
Speaker tagging: identify CEO/CFO/Analyst lines using regex on names + role keywords
Sentence segmentation: use spacy (en_core_web_sm) for sentence boundary detection
Domain tokenization: preserve financial terms (e.g., "Q3", "$2.3B", "EBITDA")
Text cleaning: remove operator/boilerplate lines ("Thank you. Please go ahead.")
 Output: data/processed/{ticker}_{date}_processed.json
Fields per record: sentence_id, speaker_role (CEO/CFO/analyst/other), section (remarks/qa), text, ticker, date
1.5 Loughran-McDonald Lexicon Integration
 Download LM Financial Wordlist (CSV from Notre Dame site)
 Parse into Python dicts: positive, negative, uncertainty, litigious, modal_strong, modal_weak
 Write scripts/lm_lexicon.py with functions:
get_hedge_phrases(text) → list of matched hedge terms
compute_lexicon_scores(text) → dict of category counts normalized by word count
 Save as data/lm_lexicon.pkl
1.6 Validation Checkpoint (Phase 1 Done When)
 ≥ 500 transcripts loaded, cleaned, and diarized
 Parquet price files aligned to each transcript by ticker+date
 LM lexicon loaded and get_hedge_phrases() returns correct output on 5 test sentences
 All data passes schema validation (no null speaker roles, no empty sentences)
 Write tests/test_phase1.py with assertions for all the above
Phase 2: NLP Model Development
Goal: Fine-tuned FinBERT with > 0.78 F1 + hedge NER with > 0.80 recall.

2.1 Baseline Sentiment Analysis (Off-the-Shelf FinBERT)
 Load ProsusAI/finbert from HuggingFace
 Run inference on 50 transcripts (sentence-level classification: positive/neutral/negative)
 Sanity check: manually inspect 20 sentences to verify predictions make sense
 Record baseline F1 (this is your "before fine-tuning" benchmark)
2.2 FinBERT Fine-Tuning
 Prepare labeled dataset:
Use HuggingFace EarningsCall dataset labels if available
If not pre-labeled: use LM lexicon as weak supervision to auto-label, then manual review of 200 samples
Split: 70% train / 15% val / 15% test (stratified by label)
 Implement scripts/finetune_finbert.py:
Base model: ProsusAI/finbert
Tokenizer: BertTokenizer with max_length=128 (sentence-level)
Training: AdamW optimizer, lr=2e-5, batch=16, epochs=3-5
Use Trainer API from HuggingFace (easiest for reproducibility)
Log metrics per epoch: loss, F1, precision, recall
Save best checkpoint by val F1 → models/finbert/best_checkpoint/
 Evaluate on held-out test set → must hit F1 > 0.78
Confusion matrix, per-class metrics, error analysis on failures
2.3 Section-Aware Sentiment Aggregation
 For each transcript compute:
prepared_remarks_sentiment: avg sentence-level FinBERT score for CEO+CFO remarks section
qa_sentiment: avg score for Q&A section
analyst_pushback_signal: if analyst Q sentiment < management answer sentiment → divergence score
ceo_vs_cfo_divergence: delta between CEO and CFO avg sentence scores
2.4 Hedge Language NER Model
 Build custom spaCy NER model for hedge phrases:
Entity label: HEDGE
Training data: sentences annotated with LM uncertainty/modal_weak words as HEDGE spans (use spacy-annotator or manual JSON format)
At minimum: 500 annotated sentences
Alternatively: pattern-based EntityRuler using compiled LM wordlist (faster, interpretable)
 Recommended hybrid approach:
Start with EntityRuler (rule-based, instant, high precision)
Train statistical NER on top for novel phrase generalization
 Evaluate: recall > 0.80 on held-out hedge-labeled sentences
 Save model to models/hedge_ner/
2.5 Management Confidence Score
 Compute per-transcript composite score (all normalized 0-1):
confidence_score = (
    0.35 × certainty_language_ratio         # strong modal / total words (LM)
    + 0.25 × forward_guidance_specificity    # numeric mentions in guidance section
    + 0.25 × (1 - hedge_density)             # inverse of hedge word density
    + 0.15 × qa_analyst_tone_delta           # how well management held up under Q&A
)
 Normalize to -1 to +1 scale via z-score relative to historical baseline
 QoQ comparison: load prior quarter's confidence score for same ticker, compute delta
2.6 Validation Checkpoint (Phase 2 Done When)
 Fine-tuned FinBERT checkpoint saved, val F1 ≥ 0.78 confirmed
 Hedge NER recall ≥ 0.80 on test set
 Management confidence score computed for all 500+ transcripts
 Write tests/test_phase2.py: unit tests for scoring functions
Phase 3: Alpha Signal Generation & Backtesting
Goal: Cross-reference sentiment with price returns. Achieve Pearson r > 0.20, Sharpe > 0.8.

3.1 Signal Construction
 For each transcript, build composite alpha signal:
alpha_signal = (
    0.40 × finbert_composite_score
    + 0.30 × management_confidence_score
    + 0.20 × (−hedge_density_normalized)
    + 0.10 × qa_pushback_signal
)
Normalized to [-1, +1]
 Store in signals/transcript_signals.parquet
Columns: ticker, call_date, alpha_signal, prepared_remarks_score, qa_score, confidence_score, hedge_density
3.2 Price Return Alignment
 Join transcript_signals with Alpaca price data on ticker + call_date
 Compute targets: ret_1d, ret_3d, ret_5d (close-to-close returns)
 Handle earnings-day price jumps: use post-market or next-day open
3.3 Signal-Return Analysis
 Pearson correlation: alpha_signal vs. ret_1d, ret_3d, ret_5d
 Must exceed r > 0.20 (stretch: r > 0.35)
 Plot: scatter plots with regression line, binned return analysis (quintile bucketing)
 Segment by sector (GICS codes) — tech sector expected to show stronger signal
3.4 Backtesting (Train: 2019-2023, Test: 2024)
 Long/short strategy: long top quintile alpha signals, short bottom quintile
 Position sizing: equal-weight within quintile
 Hold period: test 1-day, 3-day, 5-day separately
 Compute:
Portfolio returns per period
Sharpe ratio (annualized) — must exceed 0.8
Max drawdown
Hit rate (% trades profitable)
Information coefficient (IC) by quarter
 Use vectorbt or manual NumPy implementation — no black boxes
 Out-of-sample test on 2024 data — report separately
3.5 Validation Checkpoint (Phase 3 Done When)
 Pearson r ≥ 0.20 documented with p-value
 Sharpe ≥ 0.8 documented for backtest period
 Backtest equity curve plot saved
 OOS 2024 performance reported
Phase 4: REST API Development
Goal: FastAPI endpoint returning JSON signal per ticker + call date. Latency < 5 seconds.

4.1 FastAPI App
 api/main.py — FastAPI app with:
POST /analyze — accepts {ticker, transcript_text}, returns full signal JSON
GET /signal/{ticker}/{date} — returns cached signal for historical call
GET /health — liveness probe
 Pipeline on request:
Preprocess text (speaker diarization, sentence segmentation)
Run fine-tuned FinBERT inference (batch sentences)
Run hedge NER
Compute confidence score
Compute alpha signal composite
Return structured JSON response
 Response schema:
{
  "ticker": "AAPL",
  "call_date": "2024-10-31",
  "alpha_signal": 0.62,
  "sentiment": {"prepared_remarks": 0.71, "qa_section": 0.48},
  "confidence_score": 0.64,
  "hedge_density": 0.03,
  "ceo_cfo_divergence": 0.12,
  "key_hedge_phrases": ["subject to market conditions", "if demand permits"],
  "latency_ms": 1840
}
 Add request validation with Pydantic
 Add basic auth (API key header)
4.2 Performance Optimization
 Pre-load model on startup (not per-request)
 Batch inference for sentence-level predictions
 Async endpoint handler
 Target: < 5 second end-to-end latency
4.3 Validation Checkpoint (Phase 4 Done When)
 curl test of /analyze returns valid JSON with all fields
 Latency < 5 sec measured with time on 10 consecutive calls
 Automated test in tests/test_api.py
Phase 5: Evaluation, Analysis & Reporting
Goal: Professional-grade results section for submission. Hit all targets. Document everything.

5.1 Comprehensive Evaluation Report
 Model Performance:
FinBERT: precision, recall, F1 per class (confusion matrix)
Hedge NER: precision, recall, F1 for HEDGE entity
Compare baseline (off-shelf) vs fine-tuned — show improvement
 Signal Quality:
Pearson correlation table (1d/3d/5d returns)
Quintile return analysis: bar chart of avg returns by signal quintile
Sharpe ratio breakdown by sector, hold period, year
 System:
API latency benchmarks (percentiles: p50, p95, p99)
Throughput: transcripts/minute
5.2 Error Analysis
 Identify failure modes: which transcript types cause FinBERT errors?
 Sectors with low signal correlation — explain why (e.g., utilities, regulated industries)
 Hedge phrase false negatives: novel language not in LM wordlist
5.3 Ablation Study (This Is What Impresses Professors)
 Compare signal quality with/without each component:
Signal with only FinBERT scores (no hedge/confidence features)
Signal with only LM lexicon (no FinBERT)
Full composite signal
 Show that each component adds value → justifies architectural choices
5.4 Visualization Suite
 Sentiment score distribution histograms
 Management confidence score time-series for major tickers (Apple, Microsoft, Nvidia)
 Backtest equity curve (long/short portfolio vs. S&P 500)
 Signal vs. return scatter plot by sector
 Confusion matrix heatmap
 Example transcript walkthrough: annotated with FinBERT scores + hedge highlights
Tech Stack (Exact Libraries & Versions)
Library	Purpose	Version
transformers	FinBERT fine-tuning & inference	≥4.40
datasets	Load HuggingFace EarningsCall data	≥2.18
torch	Deep learning backend	≥2.2
spacy	NER, sentence segmentation	≥3.7
fastapi	REST API	≥0.111
uvicorn	ASGI server	≥0.29
alpaca-py	Price data	≥0.20
pandas	Data manipulation	≥2.2
scikit-learn	Metrics, train/test split	≥1.4
scipy	Pearson correlation	≥1.13
evaluate	HuggingFace metrics	≥0.4
accelerate	Training hardware optimization	≥0.29
plotly	Interactive visualizations	≥5.20
pydantic	API schema validation	≥2.7
Execution Order Summary
Phase 1 (Days 1-2): Data + Preprocessing
    → Collect 500+ transcripts (HuggingFace first, EDGAR fallback)
    → Pull price data from Alpaca for all tickers
    → Speaker diarization + sentence segmentation
    → LM lexicon integration

Phase 2 (Days 3-4): NLP Models
    → Baseline FinBERT on 50 samples
    → Fine-tune FinBERT (hits F1 > 0.78)
    → Hedge NER (hits recall > 0.80)
    → Management confidence score

Phase 3 (Day 5): Alpha Signals & Backtest
    → Composite signal construction
    → Price return alignment
    → Pearson correlation analysis
    → Long/short backtest (Sharpe > 0.8)

Phase 4 (Day 6): REST API
    → FastAPI endpoints
    → Latency optimization
    → API testing

Phase 5 (Day 7): Reporting
    → Full evaluation metrics
    → Ablation study
    → Visualizations
    → Final report
Critical Rule: Every script must be deterministic — set all random seeds. Every result must be reproducible from a single run_all.sh script. Every metric must be computed on a held-out test set, not training data.
