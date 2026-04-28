"""
EarningsEdge AI - Phase 0.5
Audio-Text Alignment

Align transcript sentences to Whisper timestamped words/segments.

Usage:
    python scripts/audio_text_alignment.py --transcription data/processed/call_transcription.json
"""

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List

log = logging.getLogger("audio_text_alignment")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

_TOKEN_RE = re.compile(r"[A-Za-z0-9%$\.\-']+")


def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


def build_word_timeline(transcription_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Flatten Whisper segments into a word-level timeline.
    """
    timeline: List[Dict[str, Any]] = []

    for seg in transcription_result.get("segments", []):
        seg_start = float(seg.get("start", 0.0))
        seg_end = float(seg.get("end", seg_start))
        seg_words = seg.get("words", [])

        if seg_words:
            for word in seg_words:
                raw_word = (word.get("word") or "").strip()
                if not raw_word:
                    continue
                timeline.append(
                    {
                        "word": raw_word,
                        "token": raw_word.lower(),
                        "start": float(word.get("start", seg_start)),
                        "end": float(word.get("end", seg_end)),
                    }
                )
        else:
            # Fallback if model output has no word-level timestamps.
            text_tokens = _tokenize(seg.get("text", ""))
            if not text_tokens:
                continue
            dur = max(0.001, seg_end - seg_start)
            step = dur / len(text_tokens)
            for i, token in enumerate(text_tokens):
                w_start = seg_start + i * step
                w_end = min(seg_end, w_start + step)
                timeline.append(
                    {
                        "word": token,
                        "token": token,
                        "start": w_start,
                        "end": w_end,
                    }
                )

    return timeline


def align_sentences_to_audio(
    sentences: List[Dict[str, Any]],
    word_timeline: List[Dict[str, Any]],
    max_search_window: int = 120,
) -> List[Dict[str, Any]]:
    """
    Greedy monotonic alignment from sentence tokens to timeline tokens.

    Input sentence dict fields expected:
    - sentence_id (optional)
    - text

    Output adds:
    - start_sec
    - end_sec
    - aligned_token_count
    - alignment_confidence
    """
    aligned: List[Dict[str, Any]] = []
    if not sentences:
        return aligned

    cursor = 0
    timeline_tokens = [x.get("token", "").lower() for x in word_timeline]

    for idx, sentence in enumerate(sentences):
        sent_text = sentence.get("text", "")
        sent_tokens = _tokenize(sent_text)

        if not sent_tokens or cursor >= len(word_timeline):
            out = dict(sentence)
            out.update(
                {
                    "start_sec": None,
                    "end_sec": None,
                    "aligned_token_count": 0,
                    "alignment_confidence": 0.0,
                }
            )
            aligned.append(out)
            continue

        window_end = min(len(word_timeline), cursor + max_search_window)
        token_positions: List[int] = []

        seek = cursor
        for token in sent_tokens:
            found = -1
            for pos in range(seek, window_end):
                if timeline_tokens[pos] == token:
                    found = pos
                    break
            if found == -1:
                continue
            token_positions.append(found)
            seek = found + 1

        out = dict(sentence)
        if token_positions:
            start_pos = token_positions[0]
            end_pos = token_positions[-1]
            out["start_sec"] = float(word_timeline[start_pos]["start"])
            out["end_sec"] = float(word_timeline[end_pos]["end"])
            out["aligned_token_count"] = len(token_positions)
            out["alignment_confidence"] = len(token_positions) / max(1, len(sent_tokens))
            cursor = end_pos + 1
        else:
            out["start_sec"] = None
            out["end_sec"] = None
            out["aligned_token_count"] = 0
            out["alignment_confidence"] = 0.0

        if "sentence_id" not in out:
            out["sentence_id"] = idx
        aligned.append(out)

    return aligned


def align_transcript_to_audio(
    transcription_result: Dict[str, Any],
    sentence_records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    End-to-end alignment helper for processed transcript sentence records.
    """
    word_timeline = build_word_timeline(transcription_result)
    aligned_sentences = align_sentences_to_audio(sentence_records, word_timeline)

    aligned_count = sum(1 for s in aligned_sentences if s.get("start_sec") is not None)
    avg_conf = 0.0
    if aligned_sentences:
        avg_conf = sum(float(s.get("alignment_confidence", 0.0)) for s in aligned_sentences) / len(aligned_sentences)

    return {
        "audio_path": transcription_result.get("audio_path"),
        "sentence_count": len(sentence_records),
        "aligned_sentence_count": aligned_count,
        "alignment_rate": aligned_count / max(1, len(sentence_records)),
        "average_alignment_confidence": avg_conf,
        "word_timeline": word_timeline,
        "aligned_sentences": aligned_sentences,
    }


def save_alignment_output(output_path: str, payload: Dict[str, Any]) -> None:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_json(path: str) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Align transcript sentences to audio timestamps")
    parser.add_argument("--transcription", required=True, help="Whisper transcription JSON path")
    parser.add_argument("--sentences", required=False, help="Processed sentence records JSON path")
    parser.add_argument("--out", required=True, help="Output alignment JSON path")
    args = parser.parse_args()

    transcription = load_json(args.transcription)

    if args.sentences:
        payload = load_json(args.sentences)
        if isinstance(payload, list):
            sentence_records = payload
        else:
            sentence_records = payload.get("sentences", [])
    else:
        # Fallback: treat Whisper segments as sentence records.
        sentence_records = [
            {"sentence_id": i, "text": seg.get("text", "")} for i, seg in enumerate(transcription.get("segments", []))
        ]

    aligned = align_transcript_to_audio(transcription, sentence_records)
    save_alignment_output(args.out, aligned)
    log.info("Saved alignment output to %s", args.out)


if __name__ == "__main__":
    main()
