#!/usr/bin/env python3
from __future__ import annotations

import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from datasets import Dataset
from PIL import Image


TRAIN_ROOT = Path("${DATA_DIR}/train")
VAL_ROOT = Path("${DATA_DIR}/val")
OUT_ROOT = Path("${PROJECT_DIR}")

PROMPT_TEXT = (
    "<image>You are an expert SVG vector graphics designer and engineer. Generate the best possible complete SVG "
    "draft for this image. Output only a self-contained SVG wrapped\n"
    "  in <SVG_DRAFT>...</SVG_DRAFT>. "
)

DATA_SOURCE = "joyxbo/svg_refine_grpo"
AGENT_NAME = "svg_react_agent"
ABILITY = "svg_generation"


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def alpha_to_white_rgb(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as im:
        rgba = im.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        rgb = Image.alpha_composite(bg, rgba).convert("RGB")
        rgb.save(dst, format="PNG")


def process_image_dir(src_dir: Path, dst_dir: Path, max_workers: int = 32) -> None:
    image_paths = sorted(src_dir.glob("*.png"))
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(alpha_to_white_rgb, p, dst_dir / p.name) for p in image_paths]
        for i, fut in enumerate(futures, start=1):
            fut.result()
            if i % 2000 == 0:
                print(f"[image-process] {src_dir.name}: {i}/{len(image_paths)} done")


def build_rows(split: str, root: Path, processed_image_dir: Path) -> list[dict]:
    token_meta_path = root / "meta" / "token_lengths.jsonl"
    token_meta_rows = load_jsonl(token_meta_path)
    token_meta = {Path(row["image_filename"]).stem: row for row in token_meta_rows}

    image_paths = sorted((root / "image").glob("*.png"))
    rows = []
    missing_meta = 0

    for idx, image_path in enumerate(image_paths):
        stem = image_path.stem
        svg_path = root / "svg" / f"{stem}.svg"
        if not svg_path.exists():
            raise FileNotFoundError(f"Missing SVG for image {image_path}")

        meta = token_meta.get(stem, None)
        if meta is None:
            missing_meta += 1
            meta = {}

        gt_svg = svg_path.read_text(encoding="utf-8")
        token_length = meta.get("token_length")
        char_length = meta.get("char_length", len(gt_svg))
        length_band = meta.get("length_band")
        subset_type = meta.get("subset_type")

        processed_image_path = processed_image_dir / image_path.name

        row = {
            "data_source": DATA_SOURCE,
            "agent_name": AGENT_NAME,
            "prompt": [
                {
                    "role": "user",
                    "content": PROMPT_TEXT,
                }
            ],
            "images": [{"image": f"file://{processed_image_path}"}],
            "ability": ABILITY,
            "reward_model": {
                "style": "svg_refine",
                "ground_truth": gt_svg,
            },
            "extra_info": {
                "split": split,
                "index": idx,
                "sample_id": stem,
                "original_image_path": str(image_path),
                "processed_image_path": str(processed_image_path),
                "svg_path": str(svg_path),
                "gt_svg_token_length": token_length,
                "gt_svg_char_length": char_length,
                "length_band": length_band,
                "subset_type": subset_type,
            },
        }
        rows.append(row)

    print(
        f"[build_rows] split={split} rows={len(rows)} missing_meta={missing_meta} "
        f"root={root}"
    )
    return rows


def write_preview_json(rows: list[dict], path: Path, n: int = 3, seed: int = 2026) -> None:
    rng = random.Random(seed)
    sampled = rng.sample(rows, min(n, len(rows)))
    path.write_text(json.dumps(sampled, ensure_ascii=False, indent=2), encoding="utf-8")


def write_parquet(rows: list[dict], path: Path) -> None:
    dataset = Dataset.from_list(rows)
    dataset.to_parquet(str(path))


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    train_processed_dir = OUT_ROOT / "images_whitebg_train"
    val_processed_dir = OUT_ROOT / "images_whitebg_val"

    print("[stage] processing white background train images")
    process_image_dir(TRAIN_ROOT / "image", train_processed_dir)

    print("[stage] processing white background val images")
    process_image_dir(VAL_ROOT / "image", val_processed_dir)

    print("[stage] building train rows")
    train_rows = build_rows("train", TRAIN_ROOT, train_processed_dir)

    print("[stage] building val rows")
    val_rows = build_rows("val", VAL_ROOT, val_processed_dir)

    train_parquet = OUT_ROOT / "train.parquet"
    val_parquet = OUT_ROOT / "val.parquet"

    print("[stage] writing parquet")
    write_parquet(train_rows, train_parquet)
    write_parquet(val_rows, val_parquet)

    print("[stage] writing preview json")
    write_preview_json(train_rows, OUT_ROOT / "train_preview_samples.json", n=3, seed=2026)
    write_preview_json(val_rows, OUT_ROOT / "val_preview_samples.json", n=3, seed=2027)

    summary = {
        "train_count": len(train_rows),
        "val_count": len(val_rows),
        "train_parquet": str(train_parquet),
        "val_parquet": str(val_parquet),
        "train_processed_dir": str(train_processed_dir),
        "val_processed_dir": str(val_processed_dir),
    }
    (OUT_ROOT / "data_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
