#!/usr/bin/env python3
"""CLIP 相似度 API 调用示例（路径输入 + base64 输入）。"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import requests


def img_to_b64(path: str) -> str:
    data = Path(path).read_bytes()
    return base64.b64encode(data).decode("utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:18080/v1/similarity")
    parser.add_argument(
        "--image-a",
        default="/path/to/test/source_images/00000012.png",
    )
    parser.add_argument(
        "--image-b",
        default="/path/to/test/target_images/00000012.png",
    )
    args = parser.parse_args()

    # 1) 路径模式
    payload_path = {
        "pairs": [
            {
                "image_a_path": args.image_a,
                "image_b_path": args.image_b,
            }
        ],
        "normalize_to_01": False,
    }
    resp1 = requests.post(args.url, json=payload_path, timeout=30)
    print("[PATH MODE] status=", resp1.status_code)
    print(json.dumps(resp1.json(), ensure_ascii=False, indent=2))

    # 2) base64 模式
    payload_b64 = {
        "pairs": [
            {
                "image_a_b64": img_to_b64(args.image_a),
                "image_b_b64": img_to_b64(args.image_b),
            }
        ],
        "normalize_to_01": False,
    }
    resp2 = requests.post(args.url, json=payload_b64, timeout=30)
    print("\n[B64 MODE] status=", resp2.status_code)
    print(json.dumps(resp2.json(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
