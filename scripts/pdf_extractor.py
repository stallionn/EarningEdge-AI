"""
EarningsEdge AI - Phase 0.4
PDF Transcript Extraction

Extracts text from earnings-call PDF transcripts and parses speaker/section structure.

Usage:
    python scripts/pdf_extractor.py --pdf path/to/transcript.pdf
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent))
from config import PROCESSED_DIR, QA_SECTION_MARKERS, REMARKS_SECTION_MARKERS

log = logging.getLogger("pdf_extractor")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

SPEAKER_LINE_RE = re.compile(
    r"^(?P<speaker>[A-Z][A-Za-z\s\.'\(\)]{2,80}?)(?:\s*[-:]\s*(?P<title>[^\n]{1,120}))?$"
)

ANALYST_HINTS = [
    "analyst", "research", "capital", "morgan", "goldman", "barclays",
    "ubs", "citi", "deutsche", "baird", "raymond james", "needham",
]


def extract_text_with_pdfplumber(pdf_path: str) -> str:
    """Extract text using pdfplumber with layout-aware parsing."""
    try:
        import pdfplumber
    except ImportError as exc:
        raise ImportError("pdfplumber is required. Install with: pip install pdfplumber") from exc

    pages: List[str] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            pages.append(page_text)
    return "\n".join(pages)


def extract_text_with_pypdf(pdf_path: str) -> str:
    """Extract text using pypdf as fallback parser."""
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise ImportError("pypdf is required. Install with: pip install pypdf") from exc

    reader = PdfReader(pdf_path)
    pages: List[str] = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n".join(pages)


def extract_pdf_text(pdf_path: str, prefer: str = "pdfplumber") -> str:
    """
    Extract PDF text using preferred engine with fallback.
    prefer: "pdfplumber" or "pypdf"
    """
    if prefer not in {"pdfplumber", "pypdf"}:
        raise ValueError("prefer must be 'pdfplumber' or 'pypdf'")

    if prefer == "pdfplumber":
        try:
            return extract_text_with_pdfplumber(pdf_path)
        except Exception as exc:
            log.warning("pdfplumber extraction failed (%s). Falling back to pypdf.", exc)
            return extract_text_with_pypdf(pdf_path)

    try:
        return extract_text_with_pypdf(pdf_path)
    except Exception as exc:
        log.warning("pypdf extraction failed (%s). Falling back to pdfplumber.", exc)
        return extract_text_with_pdfplumber(pdf_path)


def _normalize_line(line: str) -> str:
    line = line.replace("\u2019", "'").replace("\u2018", "'")
    line = line.replace("\u201c", '"').replace("\u201d", '"')
    line = line.replace("\u2013", "-").replace("\u2014", "-")
    line = re.sub(r"\s+", " ", line)
    return line.strip()


def _classify_speaker(speaker: str, title: str) -> str:
    text = f"{speaker} {title}".lower()
    if "operator" in text:
        return "operator"
    if "chief executive" in text or " ceo" in text or "president" in text:
        return "ceo"
    if "chief financial" in text or " cfo" in text:
        return "cfo"
    for hint in ANALYST_HINTS:
        if hint in text:
            return "analyst"
    return "other"


def detect_section(line: str, current_section: str) -> str:
    """Detect transcript section using configured markers."""
    line_lower = line.lower()
    for marker in QA_SECTION_MARKERS:
        if marker in line_lower:
            return "qa"
    for marker in REMARKS_SECTION_MARKERS:
        if marker in line_lower:
            return "remarks"
    return current_section


def _is_section_header_line(line: str) -> bool:
    """Return True when a line is likely a section marker/header."""
    line_lower = line.lower()
    for marker in QA_SECTION_MARKERS + REMARKS_SECTION_MARKERS:
        if marker in line_lower:
            return True
    return False


def _looks_like_speaker_header(line: str) -> bool:
    """Heuristic to avoid classifying normal prose as a speaker header."""
    stripped = line.strip()
    if not stripped:
        return False

    # Most transcript speaker lines include a delimiter before title/role.
    if " - " in stripped or ":" in stripped:
        return True

    # Reject regular sentence-like lines.
    if any(ch in stripped for ch in ".?!"):
        return False

    # Accept short all-caps names like "JOHN DOE".
    letters = [ch for ch in stripped if ch.isalpha()]
    if not letters:
        return False
    uppercase_ratio = sum(1 for ch in letters if ch.isupper()) / len(letters)
    return uppercase_ratio > 0.8 and len(stripped.split()) <= 6


def parse_speaker_blocks(raw_text: str) -> List[Dict[str, Any]]:
    """
    Parse extracted text into speaker blocks.

    Returns list of blocks:
    {
      "speaker": str,
      "speaker_role": str,
      "section": "remarks|qa|unknown",
      "text": str
    }
    """
    lines = [_normalize_line(x) for x in raw_text.splitlines()]
    lines = [x for x in lines if x]

    blocks: List[Dict[str, Any]] = []
    current_section = "unknown"
    current_speaker = "unknown"
    current_role = "other"
    current_lines: List[str] = []

    def flush_block() -> None:
        if not current_lines:
            return
        text = " ".join(current_lines).strip()
        if not text:
            return
        blocks.append(
            {
                "speaker": current_speaker,
                "speaker_role": current_role,
                "section": current_section,
                "text": text,
            }
        )

    for line in lines:
        new_section = detect_section(line, current_section)
        section_changed = new_section != current_section
        current_section = new_section

        if section_changed and _is_section_header_line(line):
            continue

        match = SPEAKER_LINE_RE.match(line)
        if match and len(line.split()) <= 16 and _looks_like_speaker_header(line):
            flush_block()
            current_lines = []
            current_speaker = match.group("speaker").strip()
            title = (match.group("title") or "").strip()
            current_role = _classify_speaker(current_speaker, title)
            continue

        current_lines.append(line)

    flush_block()
    return blocks


def extract_transcript_from_pdf(
    pdf_path: str,
    ticker: Optional[str] = None,
    date: Optional[str] = None,
    prefer_engine: str = "pdfplumber",
) -> Dict[str, Any]:
    """
    Full PDF transcript extraction pipeline.
    """
    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_file}")

    text = extract_pdf_text(str(pdf_file), prefer=prefer_engine)
    speaker_blocks = parse_speaker_blocks(text)

    return {
        "source_pdf": str(pdf_file),
        "ticker": ticker,
        "date": date,
        "char_count": len(text),
        "speaker_block_count": len(speaker_blocks),
        "text": text,
        "speaker_blocks": speaker_blocks,
    }


def save_pdf_extraction_output(output_path: str, payload: Dict[str, Any]) -> None:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _build_output_path(pdf_path: str) -> Path:
    stem = Path(pdf_path).stem
    return PROCESSED_DIR / f"{stem}_pdf_extracted.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract transcript text from earnings-call PDF")
    parser.add_argument("--pdf", required=True, help="Path to PDF transcript")
    parser.add_argument("--ticker", default=None, help="Ticker symbol")
    parser.add_argument("--date", default=None, help="Call date (YYYYMMDD)")
    parser.add_argument("--engine", default="pdfplumber", choices=["pdfplumber", "pypdf"])
    parser.add_argument("--out", default=None, help="Output JSON path")
    args = parser.parse_args()

    payload = extract_transcript_from_pdf(
        pdf_path=args.pdf,
        ticker=args.ticker,
        date=args.date,
        prefer_engine=args.engine,
    )

    out_path = Path(args.out) if args.out else _build_output_path(args.pdf)
    save_pdf_extraction_output(str(out_path), payload)
    log.info("Saved PDF extraction output to %s", out_path)


if __name__ == "__main__":
    main()
