#!/usr/bin/env python3
"""
并发压测脚本：
- 从 source_image / target_image 中按同名文件配对
- 发起并发请求到 /v1/similarity
- 统计吞吐和延迟
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import time
from pathlib import Path

import httpx


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = min(int(len(values) * p), len(values) - 1)
    return values[idx]


def build_pairs(source_dir: str, target_dir: str, limit: int) -> list[tuple[str, str]]:
    s = Path(source_dir)
    t = Path(target_dir)

    source_files = sorted([x for x in s.iterdir() if x.is_file()])
    target_names = {x.name for x in t.iterdir() if x.is_file()}

    pairs = []
    for fp in source_files:
        if fp.name in target_names:
            pairs.append((str(fp), str(t / fp.name)))
        if len(pairs) >= limit:
            break
    return pairs


async def run_bench(
    url: str,
    pairs: list[tuple[str, str]],
    concurrency: int,
    requests_total: int,
    pairs_per_request: int,
    timeout: float,
) -> None:
    sem = asyncio.Semaphore(concurrency)
    latencies_ms: list[float] = []
    ok, fail = 0, 0

    async with httpx.AsyncClient(timeout=timeout) as client:

        async def one_request(i: int) -> None:
            nonlocal ok, fail
            req_pairs = []
            base = i * pairs_per_request
            for j in range(pairs_per_request):
                a, b = pairs[(base + j) % len(pairs)]
                req_pairs.append({"image_a_path": a, "image_b_path": b})
            payload = {
                "pairs": req_pairs,
                "normalize_to_01": False,
            }
            async with sem:
                t0 = time.perf_counter()
                try:
                    r = await client.post(url, json=payload)
                    dt = (time.perf_counter() - t0) * 1000
                    latencies_ms.append(dt)
                    if r.status_code == 200:
                        js = r.json()
                        res_list = js.get("results", [])
                        if len(res_list) == pairs_per_request and all(
                            x.get("error") is None and isinstance(x.get("similarity"), (float, int))
                            for x in res_list
                        ):
                            ok += 1
                        else:
                            fail += 1
                    else:
                        fail += 1
                except Exception:
                    latencies_ms.append((time.perf_counter() - t0) * 1000)
                    fail += 1

        t_start = time.perf_counter()
        await asyncio.gather(*(one_request(i) for i in range(requests_total)))
        total_sec = time.perf_counter() - t_start

    qps = requests_total / total_sec if total_sec > 0 else 0.0
    pair_qps = (requests_total * pairs_per_request) / total_sec if total_sec > 0 else 0.0

    print("=" * 80)
    print("Benchmark Result")
    print("=" * 80)
    print(f"URL                 : {url}")
    print(f"Requests Total      : {requests_total}")
    print(f"Pairs / Request     : {pairs_per_request}")
    print(f"Concurrency         : {concurrency}")
    print(f"Success             : {ok}")
    print(f"Failed              : {fail}")
    print(f"Total Time (s)      : {total_sec:.3f}")
    print(f"Throughput (req/s)  : {qps:.2f}")
    print(f"Throughput (pair/s) : {pair_qps:.2f}")
    if latencies_ms:
        print(f"Latency p50 (ms)    : {percentile(latencies_ms, 0.50):.2f}")
        print(f"Latency p90 (ms)    : {percentile(latencies_ms, 0.90):.2f}")
        print(f"Latency p99 (ms)    : {percentile(latencies_ms, 0.99):.2f}")
        print(f"Latency mean (ms)   : {statistics.mean(latencies_ms):.2f}")


async def main_async(args: argparse.Namespace) -> None:
    pairs = build_pairs(args.source_dir, args.target_dir, args.pairs_limit)
    if not pairs:
        raise RuntimeError("No matched image pairs found.")
    await run_bench(
        url=args.url,
        pairs=pairs,
        concurrency=args.concurrency,
        requests_total=args.requests,
        pairs_per_request=args.pairs_per_request,
        timeout=args.timeout,
    )


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
    parser.add_argument("--pairs-limit", type=int, default=1024)
    parser.add_argument("--requests", type=int, default=1000)
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--pairs-per-request", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
