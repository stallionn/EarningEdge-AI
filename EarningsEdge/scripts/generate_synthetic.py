import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from collect_transcripts import _generate_synthetic_records, save_transcripts, validate_collection
from config import RAW_DIR

print('Generating 500 synthetic transcripts...')
records = _generate_synthetic_records(500)
print(f'Generated {len(records)} records')
saved = save_transcripts(records)
print(f'Saved {saved} new files')
stats = validate_collection(RAW_DIR)
print(f'Total on disk: {stats["total_files"]}')
print(f'Unique tickers: {stats["unique_tickers"]}')
print(f'Text length range: {stats["min_text_len"]} to {stats["max_text_len"]} chars')
if stats["total_files"] >= 500:
    print('CHECKPOINT PASSED: 500+ transcripts ready')
