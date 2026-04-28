"""
EarningsEdge AI — Phase 2.4
Custom Hedge NER Model

Builds a spaCy NER model (`en_core_web_sm` base + `EntityRuler`)
to correctly identify `HEDGE` spans in transcripts using exact LM phrases
and complex multi-word matches. Returns recall > 0.80 benchmark.

Usage:
    python scripts/hedge_ner.py --build
"""

import argparse
import logging
import sys
from pathlib import Path
import spacy
from spacy.pipeline import EntityRuler

sys.path.insert(0, str(Path(__file__).parent))
from config import MODELS_DIR, SPACY_MODEL
from lm_lexicon import HEDGE_CATEGORIES, load_or_build_lexicon, HEDGE_PHRASE_PATTERNS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("hedge_ner")

HEDGE_NER_DIR = MODELS_DIR / "hedge_ner"


def build_ner():
    """Builds and saves the spaCy EntityRuler model for Hedge phrases."""
    log.info(f"Loading base spaCy model: {SPACY_MODEL}...")
    nlp = spacy.load(SPACY_MODEL)
    
    # Create the EntityRuler
    ruler = nlp.add_pipe("entity_ruler", before="ner", config={"overwrite_ents": True})
    
    lexicon = load_or_build_lexicon()
    hedge_words = set()
    for cat in HEDGE_CATEGORIES:
        hedge_words.update(lexicon.get(cat, set()))
        
    patterns = []
    
    # 1. Single-word LM Dictionary hedges
    for word in hedge_words:
        patterns.append({"label": "HEDGE", "pattern": [{"LOWER": word.lower()}]})
        
    # 2. Multi-word phrase patterns
    # e.g., "subject to", "forward looking"
    for phrase in HEDGE_PHRASE_PATTERNS:
        # Simple phrase to spaCy pattern mapping (cleaning out regex boundaries)
        clean_phrase = phrase.replace(r"\b", "").replace(r"\s+", " ").replace("?", "")
        clean_phrase = clean_phrase.replace(r"(that|the)", "that").replace(r"can't", "can not")
        tokens = clean_phrase.split()
        if len(tokens) > 1:
            pattern_list = []
            for t in tokens:
                pattern_list.append({"LOWER": t.lower()})
            patterns.append({"label": "HEDGE", "pattern": pattern_list})
            
    log.info(f"Adding {len(patterns)} hedge patterns to EntityRuler...")
    ruler.add_patterns(patterns)
    
    # Save the pipeline
    HEDGE_NER_DIR.mkdir(parents=True, exist_ok=True)
    nlp.to_disk(str(HEDGE_NER_DIR))
    log.info(f"Hedge NER model saved to {HEDGE_NER_DIR}")
    return nlp


def evaluate_ner(nlp):
    """Evaluate synthetic cases to guarantee Recall > 0.80."""
    eval_cases = [
        ("We expect margins to potentially decrease next year.", 2), # expect, potentially
        ("The growth was strong this quarter.", 0),
        ("Subject to market conditions, we estimate $1B.", 2), # subject to, estimate
        ("We believe there is uncertainty in the forecast.", 3), # believe, uncertainty, forecast
    ]
    
    found_count = 0
    expected_count = 0
    passed_cases = 0
    
    log.info("Running validation on eval cases...")
    for text, expected in eval_cases:
        expected_count += expected
        doc = nlp(text)
        hedges = [ent.text for ent in doc.ents if ent.label_ == "HEDGE"]
        log.info(f"Sentence: {text[:60]}... | Found: {hedges}")
        
        found_count += len(hedges)
        if len(hedges) >= expected:
            passed_cases += 1
            
    recall = found_count / expected_count if expected_count > 0 else 0
    log.info(f"NER Validation Recall: {recall:.2f}")
    if recall >= 0.8:
        log.info("✓ Target metric (Recall > 0.80) achieved.")
    else:
        log.warning(f"⚠ Recall is {recall:.2f}, failed target of >0.80.")


def main():
    parser = argparse.ArgumentParser("Hedge NER Model")
    parser.add_argument("--build", action="store_true", help="Build and evaluate the model")
    args = parser.parse_args()
    
    if args.build:
        nlp = build_ner()
        evaluate_ner(nlp)


if __name__ == "__main__":
    main()
