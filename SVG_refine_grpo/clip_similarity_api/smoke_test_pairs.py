#!/usr/bin/env python3
"""用 source/target 对应图片做一次批量调用 smoke test。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import requests


def collect_pairs(source_dir: str, target_dir: str, limit: int) -> list[dict]:
    s = Path(source_dir)
    t = Path(target_dir)
    target_names = {x.name for x in t.iterdir() if x.is_file()}

    pairs = []
    for fp in sorted([x for x in s.iterdir() if x.is_file()]):
        if fp.name in target_names:
            pairs.append(
                {
                    "image_a_path": str(fp),
                    "image_b_path": str(t / fp.name),
                }
            )
        if len(pairs) >= limit:
            break
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:18080/v1/similarity")
    parser.add_argument(
        "--source-dir",
        default="/path/to/test/source_images",
    )
    parser.add_argument(
        "--target-dir",
        default="/path/to/test/target_images",
    )
    parser.add_argument("--limit", type=int, default=16)
    args = parser.parse_args()

    pairs = collect_pairs(args.source_dir, args.target_dir, args.limit)
    payload = {"pairs": pairs, "normalize_to_01": False}

    resp = requests.post(args.url, json=payload, timeout=60)
    print("status:", resp.status_code)
    data = resp.json()

    results = data.get("results", [])
    succ = sum(1 for x in results if x.get("error") is None)
    fail = len(results) - succ

    print(f"pairs={len(results)} success={succ} fail={fail}")
    for i, r in enumerate(results[:10]):
        print(f"  [{i}] similarity={r.get('similarity')} error={r.get('error')}")

    print("meta:")
    meta = {k: v for k, v in data.items() if k != "results"}
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
