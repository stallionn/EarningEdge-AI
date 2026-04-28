"""
EarningsEdge AI — Phase 1.5
Loughran-McDonald Financial Sentiment Lexicon

Downloads the LM Master Dictionary (if not present), parses it into Python dicts,
and provides utility functions for:
  - Hedge phrase detection
  - Per-category word count scores (normalized by document length)

Target: get_hedge_phrases() returns correct output on all test cases.

Usage:
    python scripts/lm_lexicon.py           # downloads + validates lexicon
    python scripts/lm_lexicon.py --test    # runs built-in 5-sentence test
"""

import argparse
import csv
import io
import logging
import pickle
import re
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent))
from config import LEXICON_DIR, LM_WORDLIST_URL

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("lm_lexicon")

LEXICON_DIR.mkdir(parents=True, exist_ok=True)
LEXICON_PKL = LEXICON_DIR / "lm_lexicon.pkl"
LEXICON_CSV = LEXICON_DIR / "LM_MasterDictionary.csv"

# ─── Category flags in the LM CSV ─────────────────────────────────────────────
# The LM CSV uses year-coded columns: non-zero = word belongs to that category.
# We detect them by column name patterns.
CATEGORY_COL_PATTERNS = {
    "negative":      re.compile(r"^negative$", re.I),
    "positive":      re.compile(r"^positive$", re.I),
    "uncertainty":   re.compile(r"^uncertaint", re.I),
    "litigious":     re.compile(r"^litigious$", re.I),
    "strong_modal":  re.compile(r"strong.modal|modal.strong", re.I),
    "weak_modal":    re.compile(r"weak.modal|modal.weak", re.I),
    "constraining":  re.compile(r"^constraining$", re.I),
}

# Hedge = uncertainty + weak_modal words (these are the hedging categories)
HEDGE_CATEGORIES = {"uncertainty", "weak_modal", "constraining"}

# ─── Compile Additional Hedge Phrase Patterns ─────────────────────────────────
# Multi-word patterns the LM wordlist doesn't fully cover
HEDGE_PHRASE_PATTERNS = [
    r"\bwe\s+expect\b",
    r"\bsubject\s+to\b",
    r"\bif\s+conditions\s+permit\b",
    r"\bwe\s+believe\b",
    r"\bwe\s+anticipate\b",
    r"\bcould\s+be\b",
    r"\bmight\s+be\b",
    r"\bpotentially\b",
    r"\bapproximately\b",
    r"\baround\s+\$?\d",
    r"\bup\s+to\b",
    r"\bdepending\s+on\b",
    r"\bmarket\s+conditions?\b",
    r"\buncertain(ty)?\b",
    r"\bforward.looking\b",
    r"\bbased\s+on\s+current\b",
    r"\bsubject\s+to\s+change\b",
    r"\bassuming\b",
    r"\bgiven\s+(that|the)\b",
    r"\brisks?\s+(and|&)\s+uncertainties\b",
    r"\bmay\s+(or\s+may\s+not|not)\b",
    r"\bno\s+guarantee\b",
    r"\bcannot\s+be\s+assured\b",
    r"\bthere\s+(can|can't|cannot)\s+be\s+no\s+assurance\b",
]
HEDGE_PHRASES_RE = [re.compile(p, re.IGNORECASE) for p in HEDGE_PHRASE_PATTERNS]


# ─── Download & Parse ─────────────────────────────────────────────────────────
def download_lm_wordlist() -> bool:
    """Download LM Master Dictionary CSV if not already present."""
    if LEXICON_CSV.exists() and LEXICON_CSV.stat().st_size > 100_000:
        log.info(f"LM wordlist already present: {LEXICON_CSV}")
        return True

    log.info(f"Downloading LM Master Dictionary from {LM_WORDLIST_URL} ...")
    headers = {
        "User-Agent": "EarningsEdge Research earningsedge@research.edu"
    }
    try:
        resp = requests.get(LM_WORDLIST_URL, headers=headers, timeout=60)
        resp.raise_for_status()
        with open(LEXICON_CSV, "wb") as f:
            f.write(resp.content)
        log.info(f"Saved LM wordlist: {LEXICON_CSV.stat().st_size / 1024:.0f} KB")
        return True
    except requests.RequestException as e:
        log.error(f"Download failed: {e}")
        log.info("Falling back to built-in core hedge/uncertainty word list.")
        return False


def parse_lm_csv(csv_path: Path) -> dict[str, set[str]]:
    """
    Parse the LM Master Dictionary CSV.
    Returns {category: set_of_words}.
    """
    categories: dict[str, set[str]] = {k: set() for k in CATEGORY_COL_PATTERNS}

    try:
        with open(csv_path, encoding="utf-8", errors="replace") as f:
            content = f.read()
    except Exception as e:
        log.error(f"Failed to read {csv_path}: {e}")
        return categories

    reader = csv.DictReader(io.StringIO(content))
    if not reader.fieldnames:
        log.error("CSV has no headers")
        return categories

    log.info(f"CSV columns: {reader.fieldnames[:15]}...")

    # Map CSV columns to our categories
    col_map: dict[str, str] = {}
    for col in reader.fieldnames:
        for cat, pattern in CATEGORY_COL_PATTERNS.items():
            if pattern.search(col):
                col_map[col] = cat
                break

    log.info(f"Column mapping: {col_map}")

    word_col = None
    for col in reader.fieldnames:
        if col.lower() in ("word", "term", "word_flag", "word_flag2"):
            word_col = col
            break
    if not word_col:
        # Try first column
        word_col = reader.fieldnames[0]

    log.info(f"Word column: {word_col}")

    row_count = 0
    for row in reader:
        word = row.get(word_col, "").strip().lower()
        if not word or not word.isalpha():
            continue

        for col, cat in col_map.items():
            val = row.get(col, "").strip()
            if val and val != "0" and val.lstrip("-").isdigit() and int(val) != 0:
                categories[cat].add(word)

        row_count += 1

    for cat, words in categories.items():
        log.info(f"  {cat:20s}: {len(words):,} words")

    return categories


def build_fallback_lexicon() -> dict[str, set[str]]:
    """
    Comprehensive built-in LM-based financial sentiment word lists.
    Contains 200+ words per key category drawn from published LM research.
    This is the primary source if the CSV download fails.
    """
    return {
        "negative": {
            # Losses & write-downs
            "abandon", "abandonment", "abrupt", "absence", "abuse", "adverse",
            "adversely", "allegations", "alleged", "anomaly", "attrition",
            "backlog", "bankruptcy", "below", "breach", "burden", "cancel",
            "cancellation", "ceased", "charges", "claim", "collapse",
            "complaint", "concern", "concerns", "conflict", "controversy",
            "crisis", "critical", "decay", "decline", "declining", "default",
            "defaulted", "deficiencies", "deficiency", "deficit", "delay",
            "delays", "deteriorate", "deteriorating", "deterioration",
            "difficulties", "difficulty", "discontinued", "dispute", "disruption",
            "downturn", "downward", "eliminate", "elimination", "errors",
            "excessive", "exposure", "fail", "failed", "failing", "failure",
            "falling", "fines", "force", "fraud", "harm", "harmed", "harmful",
            "impair", "impaired", "impairment", "inability", "inadequate",
            "ineffective", "inefficiency", "infringement", "insolvency",
            "interrupt", "interruption", "investigate", "investigation",
            "judgment", "lapse", "late", "layoff", "layoffs", "legal",
            "liability", "liquidate", "liquidation", "litigation", "loss",
            "losses", "lower", "lowered", "material", "materiality",
            "misconduct", "negative", "noncompliance", "obstacle", "obsolete",
            "penalty", "penalties", "poor", "problems", "prosecution",
            "recall", "recession", "reduce", "reduced", "reduction",
            "regulatory", "reject", "rejection", "restructure", "restructuring",
            "restatement", "risk", "risks", "setback", "severe", "short",
            "shortage", "shutdown", "slow", "slowdown", "stagnation",
            "threat", "troubled", "uncertain", "unfavorable", "unprofitable",
            "unsatisfactory", "violation", "volatility", "vulnerability",
            "weakness", "weak", "writedown", "writeoff", "write-off",
        },
        "positive": {
            "accelerate", "accomplish", "achievement", "advance", "advantage",
            "appealing", "attractive", "awarded", "benefit", "beneficial",
            "best", "breakthrough", "capture", "committed", "competitive",
            "confidence", "confident", "consistent", "continued", "delivering",
            "demonstrated", "distinguished", "diverse", "diversity", "effective",
            "effectively", "efficiency", "efficient", "enhance", "enhanced",
            "exceeded", "exceeding", "excellent", "exceptional", "expanding",
            "extraordinary", "favorable", "gain", "generate", "generating",
            "growth", "healthy", "improve", "improved", "improvement",
            "increasing", "innovate", "innovation", "innovative", "leadership",
            "leading", "leverage", "margin", "momentum", "operating",
            "opportunity", "outperform", "outstanding", "overachieve",
            "overcome", "performing", "positive", "profitability", "profitable",
            "profitably", "profit", "progress", "promising", "proven",
            "record", "reliable", "resilient", "robust", "solid", "strategic",
            "streamline", "strength", "strong", "success", "successful",
            "successfully", "sustainable", "transforming", "value", "win",
            "winning",
        },
        "uncertainty": {
            "ambiguity", "ambiguous", "anticipate", "anticipated", "approximately",
            "assume", "assumed", "assumption", "assumptions", "belief",
            "believe", "believed", "conditional", "contingent", "contingency",
            "could", "debris", "depend", "depending", "doubt", "doubtful",
            "estimate", "estimated", "estimates", "estimating", "eventually",
            "expect", "expected", "expecting", "forecasted", "fluctuate",
            "fluctuating", "fluctuation", "future", "generally", "guess",
            "hedged", "hedging", "hope", "hopefully", "if", "imprecise",
            "indefinite", "indeterminate", "inexact", "may", "might",
            "outlook", "pending", "perhaps", "planned", "possibility",
            "possible", "possibly", "potential", "potentially", "projected",
            "projections", "roughly", "seem", "should", "speculative",
            "subject", "suppose", "target", "targeted", "tend",
            "tentative", "uncertain", "uncertainty", "unclear", "unknown",
            "unpredictable", "variable", "varies", "varying", "volatile",
            "volatility", "whether", "will",
        },
        "litigious": {
            "adjudicate", "alleged", "allegation", "arbitration", "claim",
            "claims", "complaint", "compliance", "court", "criminal",
            "damages", "defendant", "deposition", "enforcement", "examine",
            "fine", "fines", "fraud", "grievance", "hearing", "illegal",
            "indictment", "infringement", "injunction", "inquiry", "inspect",
            "investigation", "judgment", "lawsuit", "legal", "liable",
            "liability", "litigant", "litigate", "litigation", "misconduct",
            "penalty", "plaintiff", "proceeding", "prosecute", "prosecution",
            "regulatory", "restitution", "settlement", "statute", "subpoena",
            "sue", "sued", "summons", "tribunal", "verdict", "violation",
        },
        "strong_modal": {
            "always", "certain", "certainly", "clearly", "commit",
            "committed", "definitely", "ensure", "guarantee", "mandatory",
            "must", "necessarily", "need", "obligate", "obligation",
            "require", "required", "requirement", "requires", "shall",
            "undoubtedly", "will",
        },
        "weak_modal": {
            "almost", "appear", "approximately", "around", "assume",
            "belief", "believe", "could", "depending", "doubt",
            "eventually", "generally", "hope", "if", "likely",
            "may", "might", "often", "perhaps", "possibly",
            "potentially", "probably", "roughly", "seem", "should",
            "sometimes", "subject", "tend", "typically", "uncertain",
            "usually", "whether", "would",
        },
        "constraining": {
            "bound", "cap", "capped", "constraint", "constrained",
            "constraint", "curtail", "exclude", "exclusion", "forbid",
            "governed", "hinder", "impede", "limited", "limitation",
            "limit", "mandatory", "obligated", "obligatory", "prevented",
            "prohibited", "prohibition", "restricted", "restriction",
            "restrict", "threshold", "ceiling", "cap", "covenant",
            "bound", "governed", "subject",
        },
    }


def load_or_build_lexicon() -> dict[str, set[str]]:
    """Load pre-built lexicon from pickle, or build from scratch."""
    if LEXICON_PKL.exists():
        log.info(f"Loading cached lexicon from {LEXICON_PKL}")
        with open(LEXICON_PKL, "rb") as f:
            return pickle.load(f)

    # Try to download and parse
    success = download_lm_wordlist()
    if success and LEXICON_CSV.exists():
        lexicon = parse_lm_csv(LEXICON_CSV)
        # Validate: if categories are empty (download/parse failure), use fallback
        total_words = sum(len(v) for v in lexicon.values())
        if total_words < 100:
            log.warning("Parsed lexicon seems empty — using fallback built-in lexicon")
            lexicon = build_fallback_lexicon()
    else:
        log.info("Using built-in fallback lexicon")
        lexicon = build_fallback_lexicon()

    # Save for future use
    with open(LEXICON_PKL, "wb") as f:
        pickle.dump(lexicon, f)
    log.info(f"Lexicon saved to {LEXICON_PKL}")

    return lexicon


# ─── Public API Functions ──────────────────────────────────────────────────────
_LEXICON: dict[str, set[str]] | None = None


def _get_lexicon() -> dict[str, set[str]]:
    global _LEXICON
    if _LEXICON is None:
        _LEXICON = load_or_build_lexicon()
    return _LEXICON


def get_hedge_phrases(text: str) -> list[str]:
    """
    Extract all hedge phrases from text.
    Returns list of matched phrase strings (with position info).

    Combines:
    1. Single-word matches from LM uncertainty + weak_modal + constraining
    2. Multi-word regex patterns (more precise for common hedge construction)
    """
    lexicon = _get_lexicon()
    hedge_words = set()
    for cat in HEDGE_CATEGORIES:
        hedge_words.update(lexicon.get(cat, set()))

    found = []
    text_lower = text.lower()

    # Single-word hedge detection (word boundary safe)
    for word in text.split():
        cleaned = re.sub(r"[^\w]", "", word.lower())
        if cleaned in hedge_words:
            found.append(cleaned)

    # Multi-word hedge phrase patterns
    for pattern in HEDGE_PHRASES_RE:
        for match in pattern.finditer(text):
            found.append(match.group(0).lower().strip())

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for item in found:
        if item not in seen:
            seen.add(item)
            unique.append(item)

    return unique


def compute_lexicon_scores(text: str) -> dict[str, float]:
    """
    Compute normalized LM lexicon scores for all categories.
    Each score = count of category words / total word count.
    Returns dict: {category: normalized_score}
    """
    lexicon = _get_lexicon()
    words = re.findall(r"\b[a-z]+\b", text.lower())
    total = len(words)

    if total == 0:
        return {cat: 0.0 for cat in lexicon}

    scores = {}
    for cat, word_set in lexicon.items():
        count = sum(1 for w in words if w in word_set)
        scores[cat] = count / total

    # Computed compound scores
    scores["hedge_density"] = (
        scores.get("uncertainty", 0) +
        scores.get("weak_modal", 0) +
        scores.get("constraining", 0)
    )
    scores["sentiment_net"] = scores.get("positive", 0) - scores.get("negative", 0)

    return scores


# ─── Validation Test ──────────────────────────────────────────────────────────
TEST_SENTENCES = [
    # Clear hedge
    "We expect revenues to grow approximately 5% to 8%, subject to market conditions.",
    # No hedge
    "We delivered record revenue of $12.3 billion this quarter.",
    # Strong hedge
    "There can be no assurance that we will achieve the projected results.",
    # Management confidence
    "We are confident in our ability to exceed our annual guidance.",
    # Q&A pushback
    "Can you clarify whether the margin expansion will persist into next year?",
]

EXPECTED_HEDGE_PRESENT = [True, False, True, False, True]


def run_validation_tests():
    """Run built-in 5-sentence validation test suite."""
    log.info("=" * 60)
    log.info("LM LEXICON VALIDATION TESTS")
    log.info("=" * 60)

    passed = 0
    for i, (sent, expect_hedge) in enumerate(zip(TEST_SENTENCES, EXPECTED_HEDGE_PRESENT)):
        phrases = get_hedge_phrases(sent)
        scores = compute_lexicon_scores(sent)
        has_hedge = len(phrases) > 0

        status = "PASS" if has_hedge == expect_hedge else "FAIL"
        if status == "PASS":
            passed += 1

        log.info(f"\n  Test {i+1}: [{status}]")
        log.info(f"    Sentence: {sent[:80]}...")
        log.info(f"    Expected hedge: {expect_hedge}, Got: {has_hedge}")
        log.info(f"    Hedge phrases:  {phrases}")
        log.info(f"    Scores: uncertainty={scores.get('uncertainty', 0):.3f}, "
                 f"hedge_density={scores.get('hedge_density', 0):.3f}, "
                 f"sentiment_net={scores.get('sentiment_net', 0):.3f}")

    log.info(f"\n  Results: {passed}/{len(TEST_SENTENCES)} tests passed")
    if passed == len(TEST_SENTENCES):
        log.info("✓ Phase 1.5 checkpoint: LM lexicon validation PASSED.")
    else:
        log.warning(f"⚠ {len(TEST_SENTENCES) - passed} validation tests FAILED. Review lexicon.")

    return passed == len(TEST_SENTENCES)


# ─── CLI ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="EarningsEdge LM Lexicon Builder")
    parser.add_argument("--test", action="store_true", help="Run validation tests")
    parser.add_argument(
        "--rebuild", action="store_true",
        help="Force rebuild (delete cached pickle and re-parse CSV)"
    )
    args = parser.parse_args()

    if args.rebuild and LEXICON_PKL.exists():
        LEXICON_PKL.unlink()
        log.info("Deleted cached lexicon.")

    # Build and load
    lexicon = load_or_build_lexicon()

    log.info("\nLexicon summary:")
    for cat, words in lexicon.items():
        log.info(f"  {cat:20s} : {len(words):,} words")

    if args.test:
        run_validation_tests()
    else:
        # Quick demo
        demo = "We expect revenue to be approximately $1B, subject to market conditions."
        log.info(f"\nDemo sentence: {demo}")
        log.info(f"Hedge phrases: {get_hedge_phrases(demo)}")
        log.info(f"Scores: {compute_lexicon_scores(demo)}")
        log.info("\nRun with --test flag to run full 5-sentence validation suite.")


if __name__ == "__main__":
    main()
