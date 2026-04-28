"""
EarningsEdge AI - Phase 0.3
Audio Transcription and Speaker Diarization

Provides:
- Whisper transcription with timestamps
- Optional pyannote speaker diarization
- Transcript-speaker alignment by time overlap

Usage:
    python scripts/transcription.py --audio path/to/call.wav
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent))
from config import PROCESSED_DIR

log = logging.getLogger("transcription")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def transcribe_audio_with_timestamps(
    audio_path: str,
    model_size: str = "base",
    language: str = "en",
    device: Optional[str] = None,
    task: str = "transcribe",
) -> Dict[str, Any]:
    """
    Transcribe audio using OpenAI Whisper with segment and word timestamps.

    Returns a dict with fields:
    - text: full transcript
    - language: detected language
    - segments: list of segment dicts (start/end/text/words)
    """
    audio_file = Path(audio_path)
    if not audio_file.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_file}")

    try:
        import whisper
    except ImportError as exc:
        raise ImportError(
            "openai-whisper is required for transcription. Install with: pip install openai-whisper"
        ) from exc

    log.info("Loading Whisper model: %s", model_size)
    model = whisper.load_model(model_size, device=device)

    log.info("Transcribing audio: %s", audio_file.name)
    result = model.transcribe(
        str(audio_file),
        task=task,
        language=language,
        word_timestamps=True,
        verbose=False,
    )

    segments = []
    for seg in result.get("segments", []):
        words = []
        for word in seg.get("words", []):
            words.append(
                {
                    "word": word.get("word", "").strip(),
                    "start": float(word.get("start", seg.get("start", 0.0))),
                    "end": float(word.get("end", seg.get("end", 0.0))),
                    "probability": float(word.get("probability", 0.0)),
                }
            )
        segments.append(
            {
                "id": int(seg.get("id", len(segments))),
                "start": float(seg.get("start", 0.0)),
                "end": float(seg.get("end", 0.0)),
                "text": seg.get("text", "").strip(),
                "avg_logprob": float(seg.get("avg_logprob", 0.0)),
                "no_speech_prob": float(seg.get("no_speech_prob", 0.0)),
                "words": words,
            }
        )

    return {
        "audio_path": str(audio_file),
        "text": result.get("text", "").strip(),
        "language": result.get("language", language),
        "segments": segments,
        "segment_count": len(segments),
    }


def perform_speaker_diarization(
    audio_path: str,
    hf_token: Optional[str] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Run speaker diarization with pyannote.audio.

    Returns a list of diarization segments:
    [{"start": float, "end": float, "speaker": str}, ...]

    If pyannote is unavailable, returns an empty list.
    """
    audio_file = Path(audio_path)
    if not audio_file.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_file}")

    try:
        from pyannote.audio import Pipeline
    except ImportError:
        log.warning("pyannote.audio not installed. Skipping diarization.")
        return []

    token = hf_token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    if not token:
        log.warning("No HF token found. Set HF_TOKEN to enable diarization.")
        return []

    log.info("Loading pyannote speaker diarization pipeline")
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=token)

    diarization = pipeline(
        str(audio_file),
        min_speakers=min_speakers,
        max_speakers=max_speakers,
    )

    segments: List[Dict[str, Any]] = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append(
            {
                "start": float(turn.start),
                "end": float(turn.end),
                "speaker": str(speaker),
            }
        )

    segments.sort(key=lambda x: x["start"])
    return segments


def _overlap_seconds(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def align_transcript_to_speakers(
    transcription_result: Dict[str, Any],
    speaker_segments: List[Dict[str, Any]],
    unknown_label: str = "unknown",
) -> List[Dict[str, Any]]:
    """
    Assign a speaker label to each transcript segment by maximum time overlap.
    """
    aligned: List[Dict[str, Any]] = []
    for seg in transcription_result.get("segments", []):
        seg_start = float(seg.get("start", 0.0))
        seg_end = float(seg.get("end", seg_start))

        best_speaker = unknown_label
        best_overlap = 0.0
        for spk in speaker_segments:
            overlap = _overlap_seconds(seg_start, seg_end, float(spk["start"]), float(spk["end"]))
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = str(spk.get("speaker", unknown_label))

        aligned_seg = dict(seg)
        aligned_seg["speaker"] = best_speaker
        aligned_seg["speaker_overlap_sec"] = best_overlap
        aligned.append(aligned_seg)

    return aligned


def save_transcription_output(
    output_path: str,
    transcription_result: Dict[str, Any],
    aligned_segments: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """Save transcription outputs as JSON."""
    payload = dict(transcription_result)
    if aligned_segments is not None:
        payload["aligned_segments"] = aligned_segments

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _build_output_path(audio_path: str) -> Path:
    stem = Path(audio_path).stem
    return PROCESSED_DIR / f"{stem}_transcription.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Transcribe earnings call audio with timestamps")
    parser.add_argument("--audio", required=True, help="Path to audio file")
    parser.add_argument("--model-size", default="base", help="Whisper model size")
    parser.add_argument("--language", default="en", help="Language code")
    parser.add_argument("--device", default=None, help="Device for Whisper (cpu/cuda)")
    parser.add_argument("--diarize", action="store_true", help="Run pyannote diarization")
    parser.add_argument("--min-speakers", type=int, default=None, help="Minimum speakers for diarization")
    parser.add_argument("--max-speakers", type=int, default=None, help="Maximum speakers for diarization")
    parser.add_argument("--out", default=None, help="Output JSON path")
    args = parser.parse_args()

    transcription = transcribe_audio_with_timestamps(
        audio_path=args.audio,
        model_size=args.model_size,
        language=args.language,
        device=args.device,
    )

    aligned = None
    if args.diarize:
        speaker_segments = perform_speaker_diarization(
            audio_path=args.audio,
            min_speakers=args.min_speakers,
            max_speakers=args.max_speakers,
        )
        aligned = align_transcript_to_speakers(transcription, speaker_segments)

    out_path = Path(args.out) if args.out else _build_output_path(args.audio)
    save_transcription_output(str(out_path), transcription, aligned)
    log.info("Saved transcription output to %s", out_path)


if __name__ == "__main__":
    main()
