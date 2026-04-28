"""
Tests for Phase 0.3-0.5 modules:
- transcription.py
- pdf_extractor.py
- audio_text_alignment.py
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from transcription import align_transcript_to_speakers
from pdf_extractor import parse_speaker_blocks
from audio_text_alignment import build_word_timeline, align_sentences_to_audio


def test_align_transcript_to_speakers_overlap():
    transcription_result = {
        "segments": [
            {"id": 0, "start": 0.0, "end": 2.0, "text": "Good morning everyone"},
            {"id": 1, "start": 2.0, "end": 5.0, "text": "Revenue increased this quarter"},
        ]
    }
    speakers = [
        {"start": 0.0, "end": 2.5, "speaker": "SPEAKER_00"},
        {"start": 2.5, "end": 5.5, "speaker": "SPEAKER_01"},
    ]

    aligned = align_transcript_to_speakers(transcription_result, speakers)
    assert aligned[0]["speaker"] == "SPEAKER_00"
    assert aligned[1]["speaker"] == "SPEAKER_01"


def test_parse_speaker_blocks_basic():
    text = """
Prepared Remarks
John Doe - Chief Executive Officer
Thank you for joining us today.
Our results were strong.
Jane Roe - Chief Financial Officer
Gross margin expanded in Q4.
Question-and-Answer Session
Analyst - Big Bank Research
Can you comment on guidance?
""".strip()

    blocks = parse_speaker_blocks(text)
    assert len(blocks) >= 3
    assert blocks[0]["speaker_role"] == "ceo"
    assert blocks[1]["speaker_role"] == "cfo"
    assert blocks[-1]["section"] == "qa"


def test_sentence_alignment_with_word_timeline():
    transcription = {
        "segments": [
            {
                "id": 0,
                "start": 0.0,
                "end": 4.0,
                "text": "We delivered strong growth",
                "words": [
                    {"word": "We", "start": 0.0, "end": 0.4},
                    {"word": "delivered", "start": 0.4, "end": 1.2},
                    {"word": "strong", "start": 1.2, "end": 1.8},
                    {"word": "growth", "start": 1.8, "end": 2.4},
                ],
            },
            {
                "id": 1,
                "start": 4.0,
                "end": 8.0,
                "text": "Operating margin improved",
                "words": [
                    {"word": "Operating", "start": 4.0, "end": 4.8},
                    {"word": "margin", "start": 4.8, "end": 5.4},
                    {"word": "improved", "start": 5.4, "end": 6.1},
                ],
            },
        ]
    }

    sentences = [
        {"sentence_id": 0, "text": "We delivered strong growth"},
        {"sentence_id": 1, "text": "Operating margin improved"},
    ]

    timeline = build_word_timeline(transcription)
    aligned = align_sentences_to_audio(sentences, timeline)

    assert aligned[0]["start_sec"] == 0.0
    assert aligned[0]["end_sec"] == 2.4
    assert aligned[0]["alignment_confidence"] >= 0.99

    assert aligned[1]["start_sec"] == 4.0
    assert aligned[1]["end_sec"] == 6.1
    assert aligned[1]["alignment_confidence"] >= 0.99
