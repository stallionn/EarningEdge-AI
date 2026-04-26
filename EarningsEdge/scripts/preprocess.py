"""
EarningsEdge AI — Phase 1.4
Preprocessing Pipeline: Speaker Diarization, Sentence Segmentation,
Domain Tokenization, Text Cleaning

Input:  data/raw/{ticker}_{YYYYMMDD}.json
Output: data/processed/{ticker}_{YYYYMMDD}_processed.json

Each output file is a list of sentence records:
{
  "sentence_id": int,
  "ticker": str,
  "date": str,
  "speaker_role": "ceo" | "cfo" | "analyst" | "operator" | "other",
  "speaker_name": str,
  "section": "remarks" | "qa" | "unknown",
  "text": str,
  "word_count": int
}

Usage:
    python scripts/preprocess.py
    python scripts/preprocess.py --input data/raw/AAPL_20240101.json
"""

import json
import logging
import re
import sys
from pathlib import Path
import argparse
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    RAW_DIR, PROCESSED_DIR, SPACY_MODEL,
    CEO_KEYWORDS, CFO_KEYWORDS, OPERATOR_KEYWORDS,
    QA_SECTION_MARKERS, REMARKS_SECTION_MARKERS,
    RANDOM_SEED
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("preprocess")

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ─── Boilerplate lines to strip ───────────────────────────────────────────────
BOILERPLATE_PATTERNS = [
    r"^\s*$",                          # empty lines
    r"forward.looking statements",     # safe harbor
    r"this transcript has been",       # disclaimer
    r"copyright \d{4}",
    r"all rights reserved",
    r"the information in this",
    r"^\s*\[?\s*applause\s*\]?\s*$",
    r"^\s*\[?\s*laughter\s*\]?\s*$",
    r"operator\s+instructions",
    r"please\s+stand\s+by",
    r"^page \d+ of \d+$",
]
BOILERPLATE_RE = [re.compile(p, re.IGNORECASE) for p in BOILERPLATE_PATTERNS]


# ─── Load spaCy ───────────────────────────────────────────────────────────────
def load_spacy_model():
    """Load spaCy model for sentence segmentation."""
    try:
        import spacy
        try:
            nlp = spacy.load(SPACY_MODEL)
            log.info(f"Loaded spaCy model: {SPACY_MODEL}")
            return nlp
        except OSError:
            log.info(f"Downloading spaCy model {SPACY_MODEL}...")
            import subprocess
            subprocess.run(
                [sys.executable, "-m", "spacy", "download", SPACY_MODEL],
                check=True, capture_output=True
            )
            return spacy.load(SPACY_MODEL)
    except Exception as e:
        log.warning(f"spaCy unavailable ({e}). Using regex-based sentence splitting.")
        return None


# ─── Section Detection ────────────────────────────────────────────────────────
def detect_section(line: str, current_section: str) -> str:
    """
    Determine the transcript section from a line.
    Returns: "remarks" | "qa" | current_section (unchanged)
    """
    line_lower = line.lower()

    for marker in QA_SECTION_MARKERS:
        if marker in line_lower:
            return "qa"

    for marker in REMARKS_SECTION_MARKERS:
        if marker in line_lower:
            return "remarks"

    return current_section


# ─── Speaker Detection ────────────────────────────────────────────────────────
# Pattern: "Firstname Lastname - Title - Company" OR just "Firstname Lastname:"
SPEAKER_LINE_RE = re.compile(
    r"^(?P<name>[A-Z][a-zA-Z\s\-\.\']{2,50})"
    r"(?:\s*[-–—:]\s*(?P<role_hint>[^-\n]{0,80}))?"
    r"\s*[:-]?\s*$"
)

def classify_speaker_role(name: str, role_hint: str) -> str:
    """
    Classify speaker as ceo, cfo, analyst, operator, or other.
    """
    combined = (name + " " + (role_hint or "")).lower()

    for kw in OPERATOR_KEYWORDS:
        if kw.lower() in combined:
            return "operator"

    for kw in CEO_KEYWORDS:
        if kw.lower() in combined:
            return "ceo"

    for kw in CFO_KEYWORDS:
        if kw.lower() in combined:
            return "cfo"

    analyst_keywords = ["analyst", "research", "bank", "capital", "morgan", "goldman",
                        "jpmorgan", "barclays", "ubs", "citi", "deutsche", "credit suisse",
                        "raymond james", "piper", "needham", "baird"]
    for kw in analyst_keywords:
        if kw.lower() in combined:
            return "analyst"

    return "other"


def parse_speaker_line(line: str) -> tuple[str, str, str] | None:
    """
    Try to parse a speaker-header line.
    Returns (speaker_name, role_hint, speaker_role) or None.
    """
    line = line.strip()
    m = SPEAKER_LINE_RE.match(line)
    if not m:
        return None
    name = m.group("name").strip()
    role_hint = (m.group("role_hint") or "").strip()
    role = classify_speaker_role(name, role_hint)
    return name, role_hint, role


# ─── Sentence Segmentation ────────────────────────────────────────────────────
def segment_sentences(text: str, nlp) -> list[str]:
    """
    Split text into sentences using spaCy (preferred) or regex fallback.
    Preserves financial number patterns like "$2.3B", "Q3 2024", "3.5%".
    """
    if not text or len(text.strip()) < 10:
        return []

    if nlp is not None:
        # spaCy handles up to 1M chars; chunk if needed
        max_chars = 100_000
        sentences = []
        for i in range(0, len(text), max_chars):
            chunk = text[i:i + max_chars]
            doc = nlp(chunk)
            sentences.extend([sent.text.strip() for sent in doc.sents if sent.text.strip()])
        return sentences
    else:
        # Regex fallback: split on sentence-ending punctuation not inside numbers
        parts = re.split(r"(?<!\d)(?<![A-Z])(?<!\.\w)[\.\!\?]+\s+(?=[A-Z])", text)
        return [p.strip() for p in parts if p.strip()]


# ─── Text Cleaning ─────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    """
    Clean raw transcript text:
    - Normalize whitespace
    - Fix encoding artifacts
    - Remove control characters
    - Preserve financial tokens
    """
    # Fix common encoding artifacts
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    text = text.replace("\u00a0", " ")  # non-breaking space

    # Remove control characters (keep newlines)
    text = re.sub(r"[^\S\n]", " ", text)  # normalize all whitespace except newlines to single space
    text = re.sub(r"\n{3,}", "\n\n", text)  # max 2 consecutive newlines
    text = re.sub(r" {2,}", " ", text)      # max 1 consecutive space

    return text.strip()


def is_boilerplate(line: str) -> bool:
    """Return True if this line is boilerplate that should be stripped."""
    line_stripped = line.strip()
    if len(line_stripped) < 3:
        return True
    for pattern in BOILERPLATE_RE:
        if pattern.search(line_stripped):
            return True
    return False


# ─── Main Preprocessing ───────────────────────────────────────────────────────
def preprocess_transcript(raw_record: dict, nlp) -> list[dict]:
    """
    Full preprocessing pipeline for one transcript.
    Returns list of sentence-level records.
    """
    ticker = raw_record.get("ticker", "UNKNOWN")
    date = raw_record.get("date", "19000101")
    raw_text = raw_record.get("raw_text", "")

    if not raw_text:
        return []

    raw_text = clean_text(raw_text)
    lines = raw_text.split("\n")

    sentence_records = []
    sentence_id = 0
    current_section = "unknown"
    current_speaker_name = "unknown"
    current_speaker_role = "other"
    current_block_lines = []

    def flush_block():
        """Process accumulated lines from one speaker block."""
        nonlocal sentence_id
        if not current_block_lines:
            return

        block_text = " ".join(current_block_lines).strip()
        if not block_text:
            return

        sentences = segment_sentences(block_text, nlp)
        for sent_text in sentences:
            sent_text = sent_text.strip()
            if not sent_text or len(sent_text) < 15:  # skip very short sentences
                continue
            word_count = len(sent_text.split())
            if word_count < 3:  # skip very short fragments
                continue

            sentence_records.append({
                "sentence_id": sentence_id,
                "ticker": ticker,
                "date": date,
                "speaker_role": current_speaker_role,
                "speaker_name": current_speaker_name,
                "section": current_section,
                "text": sent_text,
                "word_count": word_count,
            })
            sentence_id += 1

    for line in lines:
        # Skip boilerplate
        if is_boilerplate(line):
            continue

        # Update section
        current_section = detect_section(line, current_section)

        # Check if this is a speaker header
        speaker_parse = parse_speaker_line(line)
        if speaker_parse:
            # Flush previous block first
            flush_block()
            current_block_lines = []
            current_speaker_name, _, current_speaker_role = speaker_parse
            # If section still unknown, use speaker cues
            if current_section == "unknown" and current_speaker_role == "analyst":
                current_section = "qa"
            elif current_section == "unknown" and current_speaker_role in ("ceo", "cfo"):
                current_section = "remarks"
        else:
            current_block_lines.append(line.strip())

    # Flush final block
    flush_block()

    return sentence_records


# ─── Pipeline Runner ──────────────────────────────────────────────────────────
def run_preprocessing(input_path: Path | None = None):
    """
    Run preprocessing on all raw transcripts (or single file if specified).
    """
    nlp = load_spacy_model()

    if input_path:
        raw_files = [input_path]
    else:
        raw_files = sorted(RAW_DIR.glob("*.json"))

    log.info(f"Preprocessing {len(raw_files)} transcript(s)...")

    stats = {
        "processed": 0,
        "failed": 0,
        "total_sentences": 0,
        "skipped_short": 0,
    }

    role_counts = {"ceo": 0, "cfo": 0, "analyst": 0, "operator": 0, "other": 0}
    section_counts = {"remarks": 0, "qa": 0, "unknown": 0}

    for fpath in tqdm(raw_files, desc="Preprocessing", unit="transcript"):
        try:
            with open(fpath, encoding="utf-8") as f:
                raw_record = json.load(f)

            ticker = raw_record.get("ticker", "UNKNOWN")
            date = raw_record.get("date", "19000101")

            # Skip very short texts
            if len(raw_record.get("raw_text", "")) < 200:
                stats["skipped_short"] += 1
                continue

            sentence_records = preprocess_transcript(raw_record, nlp)

            if not sentence_records:
                stats["failed"] += 1
                continue

            # Save processed output
            out_name = f"{ticker}_{date}_processed.json"
            out_path = PROCESSED_DIR / out_name
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(sentence_records, f, ensure_ascii=False, indent=2)

            # Accumulate stats
            stats["processed"] += 1
            stats["total_sentences"] += len(sentence_records)
            for rec in sentence_records:
                role_counts[rec["speaker_role"]] = role_counts.get(rec["speaker_role"], 0) + 1
                section_counts[rec["section"]] = section_counts.get(rec["section"], 0) + 1

        except Exception as e:
            log.error(f"Failed to preprocess {fpath.name}: {e}")
            stats["failed"] += 1
            continue

    log.info("=" * 60)
    log.info("PREPROCESSING SUMMARY")
    log.info("=" * 60)
    log.info(f"  Transcripts processed: {stats['processed']}")
    log.info(f"  Transcripts failed:    {stats['failed']}")
    log.info(f"  Skipped (too short):   {stats['skipped_short']}")
    log.info(f"  Total sentences:       {stats['total_sentences']:,}")
    log.info(f"  Avg sentences/doc:     {stats['total_sentences'] / max(1, stats['processed']):.0f}")
    log.info(f"  Speaker role dist:     {role_counts}")
    log.info(f"  Section dist:          {section_counts}")
    log.info("=" * 60)

    if stats["processed"] >= 500:
        log.info("✓ Phase 1.4 checkpoint: ≥500 transcripts preprocessed.")
    else:
        log.warning(f"⚠ Only {stats['processed']} preprocessed. Need ≥500.")

    return stats


# ─── CLI ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="EarningsEdge Preprocessing Pipeline")
    parser.add_argument(
        "--input", type=Path, default=None,
        help="Path to a single raw JSON file (default: all files in data/raw/)"
    )
    args = parser.parse_args()
    run_preprocessing(input_path=args.input)


if __name__ == "__main__":
    main()
