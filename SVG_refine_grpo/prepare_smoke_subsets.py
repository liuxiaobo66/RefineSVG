#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_ROOT = Path('${PROJECT_DIR}')
TRAIN_SRC = PROJECT_ROOT / 'train.parquet'
VAL_SRC = PROJECT_ROOT / 'val.parquet'
OUT_DIR = PROJECT_ROOT / 'data'
TRAIN_OUT = OUT_DIR / 'smoke_train_32.parquet'
VAL_OUT = OUT_DIR / 'smoke_val_16.parquet'
SUMMARY_OUT = OUT_DIR / 'smoke_dataset_summary.json'

TRAIN_SIZE = 32
VAL_SIZE = 16
MIN_TOKEN_LEN = 100
MAX_TOKEN_LEN = 700


def select_rows(src: Path, out: Path, limit: int) -> dict:
    table = pq.read_table(src)
    rows = table.to_pylist()
    filtered = [row for row in rows if (row.get('extra_info') or {}).get('gt_svg_token_length') is not None]
    filtered = [
        row
        for row in filtered
        if MIN_TOKEN_LEN <= row['extra_info']['gt_svg_token_length'] <= MAX_TOKEN_LEN
    ]
    filtered.sort(key=lambda row: (row['extra_info']['gt_svg_token_length'], row['extra_info'].get('sample_id', '')))
    picked = filtered[:limit]
    if len(picked) < limit:
        raise RuntimeError(f'{src} only has {len(picked)} rows <= {MAX_TOKEN_LEN} tokens, need {limit}')

    out.parent.mkdir(parents=True, exist_ok=True)
    out_table = pa.Table.from_pylist(picked, schema=table.schema)
    pq.write_table(out_table, out)

    lengths = [row['extra_info']['gt_svg_token_length'] for row in picked]
    return {
        'source': str(src),
        'output': str(out),
        'num_rows': len(picked),
        'min_token_len_filter': MIN_TOKEN_LEN,
        'max_token_len_filter': MAX_TOKEN_LEN,
        'min_gt_svg_token_length': min(lengths),
        'max_gt_svg_token_length': max(lengths),
        'avg_gt_svg_token_length': sum(lengths) / len(lengths),
        'sample_ids': [row['extra_info'].get('sample_id') for row in picked[:8]],
    }


def main() -> None:
    summary = {
        'train': select_rows(TRAIN_SRC, TRAIN_OUT, TRAIN_SIZE),
        'val': select_rows(VAL_SRC, VAL_OUT, VAL_SIZE),
    }
    SUMMARY_OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
