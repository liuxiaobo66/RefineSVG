#!/usr/bin/env python3
"""Analyze trajectory dump JSONL files from SVGReactAgentLoop.

Usage:
    python analyze_trajectories.py [trajectory_dump_dir]

Reads all traj_pid*.jsonl files in the directory and prints a comprehensive
diagnostic summary.
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path


def load_trajectories(dump_dir: str) -> list[dict]:
    records = []
    for p in sorted(Path(dump_dir).glob("traj_pid*.jsonl")):
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def analyze(records: list[dict]) -> None:
    n = len(records)
    if n == 0:
        print("No trajectory records found.")
        return

    print(f"{'='*70}")
    print(f"  Trajectory Analysis — {n} samples")
    print(f"{'='*70}\n")

    # ---- Outcome distribution ----
    outcomes = Counter(r.get("outcome", "unknown") for r in records)
    print("## Outcome Distribution")
    for outcome, count in outcomes.most_common():
        print(f"  {outcome:30s}  {count:5d}  ({100*count/n:.1f}%)")
    print()

    # ---- Draft Stage Analysis ----
    drafts = [r for r in records if "draft" in r]
    print(f"## Draft Stage ({len(drafts)} samples)")

    draft_has_open = sum(1 for r in drafts if r["draft"].get("has_open_tag"))
    draft_has_close = sum(1 for r in drafts if r["draft"].get("has_close_tag"))
    draft_truncated = sum(1 for r in drafts if r["draft"].get("was_truncated"))
    draft_svg_ok = sum(1 for r in drafts if r["draft"].get("svg_extracted"))

    print(f"  has <SVG_DRAFT> open tag:  {draft_has_open:5d}  ({100*draft_has_open/len(drafts):.1f}%)")
    print(f"  has </SVG_DRAFT> close tag: {draft_has_close:5d}  ({100*draft_has_close/len(drafts):.1f}%)")
    print(f"  was_truncated (stop found): {draft_truncated:5d}  ({100*draft_truncated/len(drafts):.1f}%)")
    print(f"  SVG extracted successfully: {draft_svg_ok:5d}  ({100*draft_svg_ok/len(drafts):.1f}%)")
    print()

    # Token counts
    raw_tokens = [r["draft"]["raw_token_count"] for r in drafts]
    trunc_tokens = [r["draft"]["token_count"] for r in drafts]
    times = [r["draft"]["elapsed_sec"] for r in drafts]
    print(f"  raw tokens:   min={min(raw_tokens):6d}  median={sorted(raw_tokens)[len(raw_tokens)//2]:6d}  "
          f"max={max(raw_tokens):6d}  mean={sum(raw_tokens)/len(raw_tokens):.0f}")
    print(f"  final tokens: min={min(trunc_tokens):6d}  median={sorted(trunc_tokens)[len(trunc_tokens)//2]:6d}  "
          f"max={max(trunc_tokens):6d}  mean={sum(trunc_tokens)/len(trunc_tokens):.0f}")
    print(f"  time (sec):   min={min(times):6.1f}  median={sorted(times)[len(times)//2]:6.1f}  "
          f"max={max(times):6.1f}  mean={sum(times)/len(times):.1f}")
    print()

    # ---- Failure categorization ----
    failed_drafts = [r for r in drafts if r.get("outcome") in ("draft_render_failed", "draft_extract_failed")]
    if failed_drafts:
        print(f"## Draft Failure Categories ({len(failed_drafts)} failed samples)")

        # Sub-categorize failures
        categories = Counter()
        for r in failed_drafts:
            d = r["draft"]
            if not d.get("has_open_tag"):
                categories["no_open_tag"] += 1
            elif not d.get("has_close_tag") and not d.get("svg_extracted"):
                categories["no_close_tag_no_svg"] += 1
            elif not d.get("has_close_tag") and d.get("svg_extracted"):
                categories["no_close_tag_but_svg_extracted"] += 1
            elif d.get("has_close_tag") and d.get("svg_extracted"):
                categories["tag_ok_but_render_failed"] += 1
            else:
                categories["other"] += 1

        for cat, count in categories.most_common():
            print(f"  {cat:40s}  {count:5d}  ({100*count/len(failed_drafts):.1f}%)")
        print()

        # Show render error snippets
        render_errors = Counter()
        for r in failed_drafts:
            tool = r.get("tool", {})
            err_text = tool.get("tool_text", "")
            if err_text:
                # Extract error type (first line or first 80 chars)
                err_key = err_text.split("\n")[0][:100]
                render_errors[err_key] += 1
        if render_errors:
            print("  Top render errors:")
            for err, count in render_errors.most_common(10):
                print(f"    [{count:4d}x] {err}")
            print()

    # ---- Successful samples: final stage ----
    completed = [r for r in records if r.get("outcome") == "completed" and "final" in r]
    if completed:
        print(f"## Final Stage ({len(completed)} completed samples)")
        final_has_close = sum(1 for r in completed if r["final"].get("has_close_tag"))
        final_truncated = sum(1 for r in completed if r["final"].get("was_truncated"))
        final_svg_ok = sum(1 for r in completed if r["final"].get("svg_extracted"))
        print(f"  has </SVG_FINAL> close tag: {final_has_close:5d}  ({100*final_has_close/len(completed):.1f}%)")
        print(f"  was_truncated (stop found): {final_truncated:5d}  ({100*final_truncated/len(completed):.1f}%)")
        print(f"  SVG extracted successfully: {final_svg_ok:5d}  ({100*final_svg_ok/len(completed):.1f}%)")

        raw_tokens = [r["final"]["raw_token_count"] for r in completed]
        trunc_tokens = [r["final"]["token_count"] for r in completed]
        times = [r["final"]["elapsed_sec"] for r in completed]
        print(f"  raw tokens:   min={min(raw_tokens):6d}  median={sorted(raw_tokens)[len(raw_tokens)//2]:6d}  "
              f"max={max(raw_tokens):6d}  mean={sum(raw_tokens)/len(raw_tokens):.0f}")
        print(f"  final tokens: min={min(trunc_tokens):6d}  median={sorted(trunc_tokens)[len(trunc_tokens)//2]:6d}  "
              f"max={max(trunc_tokens):6d}  mean={sum(trunc_tokens)/len(trunc_tokens):.0f}")
        print(f"  time (sec):   min={min(times):6.1f}  median={sorted(times)[len(times)//2]:6.1f}  "
              f"max={max(times):6.1f}  mean={sum(times)/len(times):.1f}")
        print()

    # ---- Total timing ----
    total_times = [r.get("total_sec", 0) for r in records]
    print(f"## Overall Timing ({n} samples)")
    print(f"  total_sec: min={min(total_times):.1f}  median={sorted(total_times)[n//2]:.1f}  "
          f"max={max(total_times):.1f}  mean={sum(total_times)/n:.1f}")
    print()

    # ---- Show a few example failures ----
    print(f"## Example Failed Drafts (first 5)")
    shown = 0
    for r in records:
        if r.get("outcome") in ("draft_render_failed", "draft_extract_failed") and shown < 5:
            d = r["draft"]
            print(f"\n  --- request={r['request_id'][:12]} outcome={r['outcome']} ---")
            print(f"  tokens={d['raw_token_count']} truncated={d['was_truncated']} "
                  f"open_tag={d['has_open_tag']} close_tag={d['has_close_tag']} "
                  f"svg_extracted={d['svg_extracted']} svg_chars={d['svg_chars']}")
            preview = d.get("raw_text_preview", "")[:300]
            tail = d.get("raw_text_tail", "")[:200]
            print(f"  text_start: {repr(preview)}")
            print(f"  text_tail:  {repr(tail)}")
            if "tool" in r:
                print(f"  render_error: {r['tool'].get('tool_text', '')[:200]}")
            shown += 1

    # ---- Show a few example successes ----
    print(f"\n## Example Completed Samples (first 3)")
    shown = 0
    for r in records:
        if r.get("outcome") == "completed" and shown < 3:
            d = r["draft"]
            f_info = r.get("final", {})
            print(f"\n  --- request={r['request_id'][:12]} ---")
            print(f"  draft: tokens={d['raw_token_count']} truncated={d['was_truncated']} "
                  f"svg_chars={d['svg_chars']} time={d['elapsed_sec']}s")
            print(f"  final: tokens={f_info.get('raw_token_count',0)} truncated={f_info.get('was_truncated',False)} "
                  f"svg_chars={f_info.get('svg_chars',0)} time={f_info.get('elapsed_sec',0)}s")
            print(f"  total: {r.get('total_sec', 0)}s")
            shown += 1

    print(f"\n{'='*70}")


if __name__ == "__main__":
    dump_dir = sys.argv[1] if len(sys.argv) > 1 else "${PROJECT_DIR}/trajectory_dump"
    if not os.path.isdir(dump_dir):
        print(f"Directory not found: {dump_dir}")
        sys.exit(1)
    records = load_trajectories(dump_dir)
    analyze(records)
