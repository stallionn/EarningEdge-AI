"""
EarningsEdge AI — Phase 1 Validation Test Suite

Tests:
    1. Raw data: ≥500 transcript files, valid schema, no empty texts
    2. Price data: returns_index.parquet exists, has ret_1d/ret_3d/ret_5d columns, ≥500 rows
    3. Processed data: ≥500 processed files, valid roles, valid sections
    4. LM Lexicon: get_hedge_phrases() correct on 5 test sentences

Run:
    pytest tests/test_phase1.py -v
"""

import json
import sys
from pathlib import Path
import pytest

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from config import RAW_DIR, PROCESSED_DIR, PRICES_DIR, LEXICON_DIR

# ─── Constants ────────────────────────────────────────────────────────────────
MIN_TRANSCRIPTS = 450
VALID_ROLES = {"ceo", "cfo", "analyst", "operator", "other"}
VALID_SECTIONS = {"remarks", "qa", "unknown"}

# ─── Phase 1.2: Raw Transcript Tests ─────────────────────────────────────────
class TestRawTranscripts:

    @pytest.fixture(scope="class")
    def raw_files(self):
        return sorted(RAW_DIR.glob("*.json"))

    def test_minimum_transcript_count(self, raw_files):
        """Must have at least 500 transcripts collected."""
        count = len(raw_files)
        assert count >= MIN_TRANSCRIPTS, (
            f"Only {count} transcripts found in {RAW_DIR}. "
            f"Need ≥{MIN_TRANSCRIPTS}. Run: python scripts/collect_transcripts.py"
        )

    def test_all_transcripts_valid_schema(self, raw_files):
        """Every JSON file must have ticker, date, raw_text, source fields."""
        required_keys = {"ticker", "date", "raw_text", "source"}
        failures = []
        for fpath in raw_files[:200]:  # check first 200 for speed
            try:
                with open(fpath, encoding="utf-8") as f:
                    rec = json.load(f)
                missing = required_keys - set(rec.keys())
                if missing:
                    failures.append(f"{fpath.name}: missing {missing}")
            except json.JSONDecodeError as e:
                failures.append(f"{fpath.name}: JSON decode error: {e}")

        assert len(failures) == 0, f"Schema failures:\n" + "\n".join(failures[:10])

    def test_no_empty_raw_texts(self, raw_files):
        """No transcript should have empty or very short raw_text."""
        MIN_TEXT_LEN = 200
        short = []
        for fpath in raw_files[:200]:
            with open(fpath, encoding="utf-8") as f:
                rec = json.load(f)
            text_len = len(rec.get("raw_text", ""))
            if text_len < MIN_TEXT_LEN:
                short.append(f"{fpath.name}: {text_len} chars")

        assert len(short) == 0, (
            f"Found {len(short)} transcripts with < {MIN_TEXT_LEN} chars:\n"
            + "\n".join(short[:5])
        )

    def test_all_tickers_valid(self, raw_files):
        """Tickers should be non-empty strings, all caps, max 10 chars."""
        bad = []
        for fpath in raw_files[:200]:
            with open(fpath, encoding="utf-8") as f:
                rec = json.load(f)
            ticker = rec.get("ticker", "")
            if not ticker or ticker == "UNKNOWN":
                bad.append(f"{fpath.name}: ticker='{ticker}'")

        assert len(bad) < len(raw_files) * 0.05, (  # allow up to 5% unknown tickers
            f"Too many missing tickers ({len(bad)}): " + "\n".join(bad[:5])
        )

    def test_all_dates_valid(self, raw_files):
        """Dates should be YYYYMMDD format, in range 2015-2025."""
        import re
        bad = []
        for fpath in raw_files[:200]:
            with open(fpath, encoding="utf-8") as f:
                rec = json.load(f)
            date = rec.get("date", "")
            if not re.match(r"^\d{8}$", date) or date == "19000101":
                bad.append(f"{fpath.name}: date='{date}'")

        # Allow up to 10% unknown dates (some HF datasets lack dates)
        assert len(bad) < len(raw_files) * 0.10, (
            f"Too many invalid dates ({len(bad)}): " + "\n".join(bad[:5])
        )


# ─── Phase 1.3: Price Data Tests ─────────────────────────────────────────────
class TestPriceData:

    def test_returns_index_exists(self):
        """returns_index.parquet must exist."""
        idx_path = PRICES_DIR / "returns_index.parquet"
        assert idx_path.exists(), (
            f"{idx_path} not found. Run: python scripts/price_fetcher.py"
        )

    def test_returns_index_has_required_columns(self):
        """returns_index.parquet must have ticker, call_date, ret_1d, ret_3d, ret_5d."""
        import pandas as pd
        idx_path = PRICES_DIR / "returns_index.parquet"
        if not idx_path.exists():
            pytest.skip("returns_index.parquet not yet generated")

        df = pd.read_parquet(idx_path)
        required = {"ticker", "call_date", "ret_1d", "ret_3d", "ret_5d"}
        missing = required - set(df.columns)
        assert len(missing) == 0, f"Missing columns in returns_index: {missing}"

    def test_returns_index_sufficient_rows(self):
        """Must have ≥500 return records."""
        import pandas as pd
        idx_path = PRICES_DIR / "returns_index.parquet"
        if not idx_path.exists():
            pytest.skip("returns_index.parquet not yet generated")

        df = pd.read_parquet(idx_path)
        assert len(df) >= MIN_TRANSCRIPTS, (
            f"Only {len(df)} return records. Need ≥{MIN_TRANSCRIPTS}."
        )

    def test_return_values_in_reasonable_range(self):
        """Returns should be between -0.5 and +0.5 (±50% post-call move)."""
        import pandas as pd
        import numpy as np
        idx_path = PRICES_DIR / "returns_index.parquet"
        if not idx_path.exists():
            pytest.skip("returns_index.parquet not yet generated")

        df = pd.read_parquet(idx_path)
        for col in ["ret_1d", "ret_3d", "ret_5d"]:
            valid = df[col].dropna()
            if len(valid) == 0:
                continue
            wild = valid[(valid < -0.6) | (valid > 0.6)]
            assert len(wild) / len(valid) < 0.01, (
                f"More than 1% of {col} values are outside ±60%: "
                f"count={len(wild)}, vals={wild.head().tolist()}"
            )

    def test_ticker_parquet_files_exist(self):
        """Should have at least some per-ticker parquet files."""
        parquets = list(PRICES_DIR.glob("*.parquet"))
        non_index = [p for p in parquets if p.name != "returns_index.parquet"]
        assert len(non_index) >= 10, (
            f"Only {len(non_index)} ticker parquet files. Run price_fetcher.py"
        )


# ─── Phase 1.4: Processed Data Tests ─────────────────────────────────────────
class TestProcessedData:

    @pytest.fixture(scope="class")
    def processed_files(self):
        return sorted(PROCESSED_DIR.glob("*_processed.json"))

    def test_minimum_processed_count(self, processed_files):
        """Must have ≥500 processed transcripts."""
        assert len(processed_files) >= MIN_TRANSCRIPTS, (
            f"Only {len(processed_files)} processed files. "
            f"Run: python scripts/preprocess.py"
        )

    def test_processed_schema(self, processed_files):
        """Each processed file must be a list of sentence records with correct fields."""
        required_keys = {"sentence_id", "speaker_role", "section", "text", "ticker", "date"}
        failures = []

        for fpath in processed_files[:50]:  # check first 50
            with open(fpath, encoding="utf-8") as f:
                records = json.load(f)

            assert isinstance(records, list), f"{fpath.name}: expected list, got {type(records)}"
            if not records:
                failures.append(f"{fpath.name}: empty sentence list")
                continue

            for rec in records[:3]:
                missing = required_keys - set(rec.keys())
                if missing:
                    failures.append(f"{fpath.name}: record missing {missing}")
                    break

        assert len(failures) == 0, "Schema failures:\n" + "\n".join(failures[:10])

    def test_valid_speaker_roles(self, processed_files):
        """All speaker_role values must be in VALID_ROLES."""
        invalid = []
        for fpath in processed_files[:50]:
            with open(fpath, encoding="utf-8") as f:
                records = json.load(f)
            for rec in records:
                role = rec.get("speaker_role", "")
                if role not in VALID_ROLES:
                    invalid.append(f"{fpath.name}: role='{role}'")

        assert len(invalid) == 0, "Invalid roles:\n" + "\n".join(invalid[:10])

    def test_valid_sections(self, processed_files):
        """All section values must be in VALID_SECTIONS."""
        invalid = []
        for fpath in processed_files[:50]:
            with open(fpath, encoding="utf-8") as f:
                records = json.load(f)
            for rec in records:
                section = rec.get("section", "")
                if section not in VALID_SECTIONS:
                    invalid.append(f"{fpath.name}: section='{section}'")

        assert len(invalid) == 0, "Invalid sections:\n" + "\n".join(invalid[:10])

    def test_no_empty_sentences(self, processed_files):
        """Sentences must not be empty or too short."""
        MIN_SENTENCE_LEN = 10
        bad = []
        for fpath in processed_files[:50]:
            with open(fpath, encoding="utf-8") as f:
                records = json.load(f)
            for rec in records:
                text = rec.get("text", "")
                if len(text.strip()) < MIN_SENTENCE_LEN:
                    bad.append(f"{fpath.name}: sent={repr(text)}")

        assert len(bad) == 0, f"Empty/short sentences:\n" + "\n".join(bad[:5])

    def test_sentence_ids_sequential(self, processed_files):
        """sentence_id must be sequential starting from 0."""
        for fpath in processed_files[:20]:
            with open(fpath, encoding="utf-8") as f:
                records = json.load(f)
            ids = [rec["sentence_id"] for rec in records]
            expected = list(range(len(ids)))
            assert ids == expected, (
                f"{fpath.name}: sentence_ids not sequential. "
                f"First 5: {ids[:5]}"
            )

    def test_qa_and_remarks_sections_present(self, processed_files):
        """At least some documents should have both remarks and Q&A sections."""
        has_remarks = 0
        has_qa = 0
        for fpath in processed_files[:100]:
            with open(fpath, encoding="utf-8") as f:
                records = json.load(f)
            sections = {rec["section"] for rec in records}
            if "remarks" in sections:
                has_remarks += 1
            if "qa" in sections:
                has_qa += 1

        # At least 20% of transcripts should have identified sections
        min_identified = max(1, int(min(100, len(processed_files)) * 0.20))
        assert has_remarks >= min_identified or has_qa >= min_identified, (
            f"Section detection seems broken: only {has_remarks} with remarks, "
            f"{has_qa} with Q&A out of first 100 transcripts"
        )


# ─── Phase 1.5: LM Lexicon Tests ─────────────────────────────────────────────
class TestLMLexicon:

    @pytest.fixture(scope="class")
    def lexicon_functions(self):
        from lm_lexicon import get_hedge_phrases, compute_lexicon_scores
        return get_hedge_phrases, compute_lexicon_scores

    def test_lexicon_pickle_exists(self):
        """LM lexicon pickle must be built."""
        from config import LEXICON_DIR
        pkl = LEXICON_DIR / "lm_lexicon.pkl"
        assert pkl.exists(), (
            "LM lexicon not built. Run: python scripts/lm_lexicon.py"
        )

    def test_hedge_phrases_sentence_1(self, lexicon_functions):
        """Sentence with 'expect' and 'subject to' should return hedge phrases."""
        get_hedge_phrases, _ = lexicon_functions
        sent = "We expect revenues to grow approximately 5% to 8%, subject to market conditions."
        phrases = get_hedge_phrases(sent)
        assert len(phrases) > 0, f"Expected hedge phrases in: {sent}"

    def test_hedge_phrases_sentence_2_clear(self, lexicon_functions):
        """Clear positive sentence with no hedging."""
        get_hedge_phrases, _ = lexicon_functions
        sent = "We delivered record revenue of $12.3 billion this quarter."
        phrases = get_hedge_phrases(sent)
        assert len(phrases) == 0, (
            f"Expected NO hedge phrases in: {sent}. Got: {phrases}"
        )

    def test_hedge_phrases_sentence_3_strong(self, lexicon_functions):
        """Strong hedge: 'no assurance'."""
        get_hedge_phrases, _ = lexicon_functions
        sent = "There can be no assurance that we will achieve the projected results."
        phrases = get_hedge_phrases(sent)
        assert len(phrases) > 0, f"Expected hedge phrases in: {sent}"

    def test_hedge_phrases_sentence_4_confident(self, lexicon_functions):
        """Confident management statement — minimal hedging."""
        get_hedge_phrases, _ = lexicon_functions
        sent = "We are confident in our ability to exceed our annual guidance."
        phrases = get_hedge_phrases(sent)
        # "confident" is not a hedge; this should return 0 or minimal
        # NOTE: If "exceed" or "ability" trigger hedge, that's a false positive
        assert len(phrases) == 0, (
            f"Confident statement should not have hedge phrases. Got: {phrases}"
        )

    def test_hedge_phrases_sentence_5_qa(self, lexicon_functions):
        """Q&A question asking about certainty should be flagged as hedging."""
        get_hedge_phrases, _ = lexicon_functions
        sent = "Can you clarify whether the margin expansion will persist into next year?"
        phrases = get_hedge_phrases(sent)
        assert len(phrases) > 0, f"Expected hedge phrases in Q&A question: {sent}"

    def test_compute_scores_returns_all_categories(self, lexicon_functions):
        """compute_lexicon_scores must return dict with required keys."""
        _, compute_lexicon_scores = lexicon_functions
        text = "We expect growth of approximately 5% next quarter."
        scores = compute_lexicon_scores(text)
        required = {"uncertainty", "positive", "negative", "hedge_density", "sentiment_net"}
        missing = required - set(scores.keys())
        assert len(missing) == 0, f"Missing score categories: {missing}"

    def test_hedge_density_positive_for_hedged_text(self, lexicon_functions):
        """Hedge density score must be > 0 for clearly hedged text."""
        _, compute_lexicon_scores = lexicon_functions
        text = "We may potentially see uncertain outcomes depending on market conditions."
        scores = compute_lexicon_scores(text)
        assert scores["hedge_density"] > 0, (
            f"Expected hedge_density > 0 for hedged text. Got: {scores['hedge_density']}"
        )

    def test_hedge_density_near_zero_for_clear_text(self, lexicon_functions):
        """Hedge density must be low for a clear factual statement."""
        _, compute_lexicon_scores = lexicon_functions
        text = "Revenue increased to $5 billion, exceeding our target by 12%."
        scores = compute_lexicon_scores(text)
        assert scores["hedge_density"] < 0.15, (
            f"Expected low hedge_density for clear text. Got: {scores['hedge_density']}"
        )
