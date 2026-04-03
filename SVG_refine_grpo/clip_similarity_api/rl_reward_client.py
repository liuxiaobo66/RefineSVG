#!/usr/bin/env python3
"""给强化学习奖励计算用的简易客户端。"""

from __future__ import annotations

from typing import Iterable

import requests


class ClipSimilarityClient:
    def __init__(self, base_url: str = "http://127.0.0.1:18080", timeout: float = 30.0):
        self.url = base_url.rstrip("/") + "/v1/similarity"
        self.timeout = timeout

    def score_pair(self, image_a_path: str, image_b_path: str) -> float:
        payload = {
            "pairs": [{"image_a_path": image_a_path, "image_b_path": image_b_path}],
            "normalize_to_01": True,
        }
        r = requests.post(self.url, json=payload, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        item = data["results"][0]
        if item.get("error") is not None:
            raise RuntimeError(item["error"])
        return float(item["similarity"])

    def score_pairs(self, pairs: Iterable[tuple[str, str]]) -> list[float]:
        payload = {
            "pairs": [
                {"image_a_path": a, "image_b_path": b}
                for a, b in pairs
            ],
            "normalize_to_01": True,
        }
        r = requests.post(self.url, json=payload, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()

        out: list[float] = []
        for idx, item in enumerate(data["results"]):
            if item.get("error") is not None:
                raise RuntimeError(f"pair#{idx} failed: {item['error']}")
            out.append(float(item["similarity"]))
        return out


if __name__ == "__main__":
    # demo
    client = ClipSimilarityClient("http://127.0.0.1:18080")
    s = client.score_pair(
        "/path/to/test/source_images/00000012.png",
        "/path/to/test/target_images/00000012.png",
    )
    print("reward(similarity in [0,1]) =", s)
