# EarningsEdge AI 📈

> **Capstone BS-AI** — NLP Pipeline for Earnings Call Sentiment & Alpha Signal Generation

A 3-layer NLP pipeline that extracts financial sentiment, detects hedge language, and generates alpha signals from earnings call transcripts.

---

## Architecture

```
Data Ingestion Layer   →   NLP Model Layer   →   Alpha Signal Layer
Transcripts + Prices       FinBERT + NER          Backtest + REST API
```

---

## Target Metrics

| Metric | MVP Target | Stretch Goal |
|--------|-----------|--------------|
| F1 Score (Sentiment Model) | > 0.78 | > 0.85 |
| Hedge Phrase Recall | > 0.80 | > 0.88 |
| Signal-Return Correlation (Pearson r) | > 0.20 | > 0.35 |
| Backtest Sharpe Ratio | > 0.8 | > 1.2 |
| API Latency per transcript | < 5 sec | < 2 sec |
| Transcripts Processed | 500+ | 2,000+ |
| False Positive Rate (signals) | < 20% | < 12% |

---

## Project Structure

```
EarningsEdge/
├── data/
│   ├── raw/          # Raw transcripts (JSON/txt)
│   ├── processed/    # Speaker-diarized, cleaned
│   ├── prices/       # OHLCV from Alpaca
│   └── labels/       # Sentiment labels
├── models/
│   ├── finbert/      # Fine-tuned checkpoint
│   └── hedge_ner/    # Custom NER model
├── signals/          # Backtest results, signal scores
├── api/              # FastAPI app
├── notebooks/        # EDA, experiments
├── tests/            # Unit tests
└── scripts/          # Data collection scripts
```

---

## Pipeline Phases

### Phase 1 — Data Collection & Preprocessing

**Goal:** Clean, speaker-diarized, tokenized transcript data + aligned price data.

- **Transcripts** — HuggingFace `EarningsCall` corpus (2,700+ transcripts); SEC EDGAR 8-K as fallback
- **Price Data** — Alpaca Markets API: daily OHLCV ±10 days around each call; computes 1d/3d/5d post-call returns
- **Preprocessing** — Section splitting (Prepared Remarks vs Q&A), speaker tagging (CEO/CFO/Analyst), spaCy sentence segmentation, domain tokenization
- **Loughran-McDonald Lexicon** — Parsed into positive/negative/uncertainty/litigious/modal categories for hedge phrase detection

**Done when:** ≥500 transcripts cleaned & diarized, price parquets aligned, LM lexicon validated, schema tests passing.

---

### Phase 2 — NLP Model Development

**Goal:** Fine-tuned FinBERT (F1 > 0.78) + hedge NER (recall > 0.80).

- **Baseline** — Off-the-shelf `ProsusAI/finbert` inference on 50 transcripts
- **Fine-tuning** — AdamW, lr=2e-5, batch=16, epochs=3–5; 70/15/15 stratified split; best checkpoint saved by val F1
- **Section-Aware Aggregation** — Separate sentiment scores for prepared remarks vs Q&A; CEO/CFO divergence signal; analyst pushback detection
- **Hedge NER** — Hybrid EntityRuler (rule-based, high precision) + statistical spaCy NER for novel phrase generalization
- **Management Confidence Score:**

```
confidence_score = (
    0.35 × certainty_language_ratio
  + 0.25 × forward_guidance_specificity
  + 0.25 × (1 − hedge_density)
  + 0.15 × qa_analyst_tone_delta
)
```

**Done when:** FinBERT val F1 ≥ 0.78, hedge NER recall ≥ 0.80, confidence scores computed for all 500+ transcripts.

---

### Phase 3 — Alpha Signal Generation & Backtesting

**Goal:** Pearson r > 0.20, Sharpe ratio > 0.8.

- **Composite Alpha Signal:**

```
alpha_signal = (
    0.40 × finbert_composite_score
  + 0.30 × management_confidence_score
  + 0.20 × (−hedge_density_normalized)
  + 0.10 × qa_pushback_signal
)
```

- **Backtest** — Long/short strategy (top vs bottom quintile); train 2019–2023, OOS test 2024; equal-weight position sizing; 1d/3d/5d hold periods
- **Metrics** — Sharpe ratio (annualized), max drawdown, hit rate, information coefficient by quarter

**Done when:** Pearson r ≥ 0.20 with p-value documented, Sharpe ≥ 0.8, equity curve plotted, OOS 2024 results reported.

---

### Phase 4 — REST API

**Goal:** FastAPI endpoint < 5 second latency.

**Endpoints:**

| Method | Route | Description |
|--------|-------|-------------|
| `POST` | `/analyze` | Accepts `{ticker, transcript_text}`, returns full signal JSON |
| `GET` | `/signal/{ticker}/{date}` | Returns cached signal for historical call |
| `GET` | `/health` | Liveness probe |

**Sample Response:**

```json
{
  "ticker": "AAPL",
  "call_date": "2024-10-31",
  "alpha_signal": 0.62,
  "sentiment": { "prepared_remarks": 0.71, "qa_section": 0.48 },
  "confidence_score": 0.64,
  "hedge_density": 0.03,
  "ceo_cfo_divergence": 0.12,
  "key_hedge_phrases": ["subject to market conditions", "if demand permits"],
  "latency_ms": 1840
}
```

**Optimizations:** Model pre-loaded on startup, batched sentence inference, async handlers.

---

### Phase 5 — Evaluation & Reporting

- **Model Performance** — FinBERT precision/recall/F1 per class; baseline vs fine-tuned comparison; hedge NER metrics
- **Signal Quality** — Pearson correlation table (1d/3d/5d), quintile return analysis, Sharpe breakdown by sector/year
- **Ablation Study** — FinBERT only vs LM lexicon only vs full composite signal
- **Visualizations** — Sentiment distributions, confidence time-series (AAPL/MSFT/NVDA), backtest equity curve vs S&P 500, scatter plots by sector, confusion matrix heatmap

---

## Tech Stack

| Library | Purpose | Version |
|---------|---------|---------|
| `transformers` | FinBERT fine-tuning & inference | ≥ 4.40 |
| `datasets` | HuggingFace EarningsCall data | ≥ 2.18 |
| `torch` | Deep learning backend | ≥ 2.2 |
| `spacy` | NER, sentence segmentation | ≥ 3.7 |
| `fastapi` | REST API | ≥ 0.111 |
| `uvicorn` | ASGI server | ≥ 0.29 |
| `alpaca-py` | Price data | ≥ 0.20 |
| `pandas` | Data manipulation | ≥ 2.2 |
| `scikit-learn` | Metrics, train/test split | ≥ 1.4 |
| `scipy` | Pearson correlation | ≥ 1.13 |
| `evaluate` | HuggingFace metrics | ≥ 0.4 |
| `accelerate` | Training hardware optimization | ≥ 0.29 |
| `plotly` | Interactive visualizations | ≥ 5.20 |
| `pydantic` | API schema validation | ≥ 2.7 |

---

## Execution Timeline

```
Day 1–2  │ Phase 1 │ Data collection, preprocessing, LM lexicon
Day 3–4  │ Phase 2 │ FinBERT fine-tuning, hedge NER, confidence scoring
Day 5    │ Phase 3 │ Alpha signal construction, backtesting
Day 6    │ Phase 4 │ FastAPI endpoints, latency optimization
Day 7    │ Phase 5 │ Evaluation, ablation study, final report
```

---

## Reproducibility

> Every script is deterministic (fixed random seeds). All results are reproducible via a single entry point:

```bash
bash run_all.sh
```

> **All metrics are computed on held-out test sets only — never on training data.**
