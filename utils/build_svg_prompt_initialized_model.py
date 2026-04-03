#!/usr/bin/env python3
"""Build an SVG-extended model with prompt-based token embedding initialization.

This script follows the InternSVG-style workflow:
1) add SVG tokens to tokenizer as normal tokens;
2) resize model token embeddings;
3) initialize each new token from its semantic prompt text (e.g. en_prompt);
4) save tokenizer + model + report.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from transformers import AddedToken, AutoModelForImageTextToText, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src-model", required=True, help="Base model directory.")
    parser.add_argument("--dst-model", required=True, help="Output model directory.")
    parser.add_argument("--vocab-tsv", required=True, help="TSV with group/token/prompts.")
    parser.add_argument(
        "--prompt-field",
        default="en_prompt",
        help="Prompt column to initialize token embedding from.",
    )
    parser.add_argument(
        "--fallback-prompt-fields",
        default="en_prompt,zh_prompt",
        help="Fallback prompt columns when prompt-field is empty.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Enable trust_remote_code when loading model/tokenizer.",
    )
    parser.add_argument(
        "--torch-dtype",
        default="auto",
        choices=["auto", "float32", "float16", "bfloat16"],
        help="Model loading dtype.",
    )
    parser.add_argument(
        "--device-map",
        default="auto",
        help='device_map for from_pretrained. Use "none" to disable.',
    )
    parser.add_argument(
        "--low-cpu-mem-usage",
        action="store_true",
        help="Enable low_cpu_mem_usage when loading model.",
    )
    parser.add_argument(
        "--skip-model",
        action="store_true",
        help="Only save tokenizer + report; do not load/resize/save model.",
    )
    return parser.parse_args()


def _resolve_dtype(name: str):
    if name == "auto":
        return "auto"
    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {name}")


def _resolve_device_map(name: str):
    if name.lower() == "none":
        return None
    return name


def read_vocab_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = list(reader)
    required = {"group", "token"}
    missing = [k for k in required if k not in (rows[0].keys() if rows else set())]
    if missing:
        raise ValueError(f"TSV missing required columns: {missing}")
    return rows


def add_tokens_unique(tokenizer, rows: List[Dict[str, str]]) -> Tuple[int, List[Dict[str, str]], int]:
    vocab = tokenizer.get_vocab()
    new_tokens: List[AddedToken] = []
    added_rows: List[Dict[str, str]] = []
    skipped_existing = 0

    seen_in_input = set()
    for row in rows:
        token = row.get("token", "")
        if not token:
            continue
        if token in seen_in_input:
            continue
        seen_in_input.add(token)
        if token in vocab:
            skipped_existing += 1
            continue
        new_tokens.append(AddedToken(token, lstrip=False, rstrip=False, normalized=False))
        added_rows.append(row)

    n_new = tokenizer.add_tokens(new_tokens, special_tokens=False) if new_tokens else 0
    if n_new < len(added_rows):
        added_rows = added_rows[:n_new]
    return n_new, added_rows, skipped_existing


def _get_out_layer_and_weight(model):
    out_layer, out_weight = None, None
    try:
        if hasattr(model, "get_output_embeddings") and callable(model.get_output_embeddings):
            layer = model.get_output_embeddings()
            if layer is not None and hasattr(layer, "weight"):
                out_layer, out_weight = layer, layer.weight
    except Exception:
        pass

    if out_layer is None and hasattr(model, "language_model"):
        lm = model.language_model
        try:
            if hasattr(lm, "get_output_embeddings") and callable(lm.get_output_embeddings):
                layer = lm.get_output_embeddings()
                if layer is not None and hasattr(layer, "weight"):
                    out_layer, out_weight = layer, layer.weight
        except Exception:
            pass
    return out_layer, out_weight


def _is_tied(in_w: torch.Tensor, out_w: torch.Tensor) -> bool:
    try:
        return (in_w is out_w) or (in_w.data_ptr() == out_w.data_ptr())
    except Exception:
        return False


def _update_vocab_size_fields(model, new_vocab_size: int) -> None:
    for obj in (
        getattr(model, "config", None),
        getattr(getattr(model, "config", None), "text_config", None),
        getattr(model, "language_model", None),
        getattr(getattr(model, "language_model", None), "config", None),
        getattr(getattr(getattr(model, "language_model", None), "config", None), "text_config", None),
    ):
        if obj is not None and hasattr(obj, "vocab_size"):
            try:
                obj.vocab_size = new_vocab_size
            except Exception:
                pass
            try:
                obj.__dict__["vocab_size"] = new_vocab_size
            except Exception:
                pass


def _choose_prompt(
    row: Dict[str, str], primary: str, fallback_fields: List[str], token_fallback: str
) -> Tuple[str, str]:
    for field in [primary] + fallback_fields:
        text = (row.get(field) or "").strip()
        if text:
            return text, field
    return token_fallback, "token"


def semantic_initialize_new_tokens_from_prompt(
    model,
    tokenizer_new,
    tokenizer_ref,
    added_rows: List[Dict[str, str]],
    prompt_field: str,
    fallback_prompt_fields: List[str],
) -> Dict:
    model.eval()
    in_embed = model.get_input_embeddings()
    old_in_w = in_embed.weight.detach().clone()
    old_vocab = old_in_w.size(0)

    _, out_w_cur = _get_out_layer_and_weight(model)
    old_out_w = None
    if out_w_cur is not None:
        try:
            old_out_w = out_w_cur.detach().clone()
        except Exception:
            old_out_w = None

    model.resize_token_embeddings(len(tokenizer_new))
    _update_vocab_size_fields(model, len(tokenizer_new))

    in_w = model.get_input_embeddings().weight
    _, out_w = _get_out_layer_and_weight(model)
    tied = bool(out_w is not None and _is_tied(in_w, out_w))

    global_in_mean = old_in_w.mean(dim=0)
    global_out_mean = old_out_w.mean(dim=0) if old_out_w is not None else None
    unk_id = getattr(tokenizer_ref, "unk_token_id", None)

    stats = {
        "total_new_tokens": len(added_rows),
        "semantic_inited": 0,
        "fallback_global": 0,
        "missing_token_id": 0,
        "missing_in_output_head": 0,
        "prompt_field_primary": prompt_field,
        "prompt_field_fallback": fallback_prompt_fields,
        "tied_embeddings": tied,
        "examples": [],
        "prompt_source_counter": {},
    }
    prompt_source_counter = Counter()

    def _log_example(token: str, prompt: str, source: str, ids: List[int], fallback: bool, vec_sum: float):
        if len(stats["examples"]) < 8:
            stats["examples"].append(
                {
                    "token": token,
                    "prompt_source": source,
                    "prompt": prompt,
                    "piece_ids": ids[:32],
                    "used_fallback": fallback,
                    "in_vec_sum": vec_sum,
                }
            )

    for row in added_rows:
        token = row["token"]
        new_id = tokenizer_new.convert_tokens_to_ids(token)
        if not isinstance(new_id, int) or new_id < 0 or new_id >= len(tokenizer_new):
            stats["missing_token_id"] += 1
            continue

        prompt, source = _choose_prompt(
            row=row,
            primary=prompt_field,
            fallback_fields=fallback_prompt_fields,
            token_fallback=token,
        )
        prompt_source_counter[source] += 1

        piece_ids = tokenizer_ref.encode(prompt, add_special_tokens=False)
        valid_ids: List[int] = []
        for pid in piece_ids:
            if not isinstance(pid, int):
                continue
            if pid < 0 or pid >= old_vocab:
                continue
            if unk_id is not None and pid == unk_id:
                continue
            valid_ids.append(pid)

        with torch.no_grad():
            if valid_ids:
                vec_in = old_in_w[valid_ids].mean(dim=0)
                in_w[new_id].copy_(vec_in.to(in_w.device, dtype=in_w.dtype))
                if out_w is not None and not tied:
                    try:
                        if old_out_w is not None:
                            vec_out = old_out_w[valid_ids].mean(dim=0)
                            out_w[new_id].copy_(vec_out.to(out_w.device, dtype=out_w.dtype))
                        else:
                            out_w[new_id].copy_(vec_in.to(out_w.device, dtype=out_w.dtype))
                    except Exception:
                        stats["missing_in_output_head"] += 1
                stats["semantic_inited"] += 1
                _log_example(token, prompt, source, valid_ids, False, float(vec_in.sum().item()))
            else:
                in_w[new_id].copy_(global_in_mean.to(in_w.device, dtype=in_w.dtype))
                if out_w is not None and not tied:
                    try:
                        base = global_out_mean if global_out_mean is not None else global_in_mean
                        out_w[new_id].copy_(base.to(out_w.device, dtype=out_w.dtype))
                    except Exception:
                        stats["missing_in_output_head"] += 1
                stats["fallback_global"] += 1
                _log_example(token, prompt, source, piece_ids, True, float(in_w[new_id].sum().item()))

    stats["prompt_source_counter"] = dict(prompt_source_counter)

    try:
        if getattr(model.config, "tie_word_embeddings", None):
            model.tie_weights()
    except Exception:
        pass

    return stats


def write_report(dst: Path, report: Dict) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    json_path = dst / "added_tokens_report.json"
    txt_path = dst / "added_tokens_report.txt"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines: List[str] = []
    lines.append("========== SVG Prompt Init Report ==========")
    lines.append(f"Source model: {report['paths']['src_model']}")
    lines.append(f"Output model: {report['paths']['dst_model']}")
    lines.append(f"Vocab tsv   : {report['paths']['vocab_tsv']}")
    lines.append(f"Prompt field: {report['prompt']['primary']} | fallback={report['prompt']['fallback']}")
    lines.append("")
    lines.append(f"Original vocab size: {report['sizes']['orig_vocab']}")
    lines.append(f"New vocab size     : {report['sizes']['new_vocab']}")
    lines.append(f"Requested tokens   : {report['counts']['requested_tokens']}")
    lines.append(f"Added tokens       : {report['counts']['added_tokens']}")
    lines.append(f"Skipped existing   : {report['counts']['skipped_existing']}")
    lines.append(f"Skip model         : {report['counts']['skip_model']}")
    lines.append("")
    lines.append("Added by group:")
    for g, n in sorted(report["counts"]["added_by_group"].items()):
        lines.append(f"  - {g}: {n}")
    lines.append("")
    model_info = report["model_init"]
    for key in [
        "status",
        "message",
        "total_new_tokens",
        "semantic_inited",
        "fallback_global",
        "missing_token_id",
        "missing_in_output_head",
        "tied_embeddings",
    ]:
        if key in model_info:
            lines.append(f"{key}: {model_info[key]}")
    if model_info.get("prompt_source_counter"):
        lines.append(f"prompt_source_counter: {model_info['prompt_source_counter']}")
    if model_info.get("examples"):
        lines.append("Examples:")
        for ex in model_info["examples"]:
            lines.append(
                f"  - token={ex['token']} source={ex['prompt_source']} "
                f"fallback={ex['used_fallback']} pieces={ex['piece_ids'][:8]}"
            )
    lines.append("===========================================")
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def copy_extra_files(src: Path, dst: Path) -> List[str]:
    extra = [
        "preprocessor_config.json",
        "processor_config.json",
        "image_processor.json",
        "vision_config.json",
        "chat_template.json",
        "generation_config.json",
        "README.md",
    ]
    copied = []
    for name in extra:
        s = src / name
        d = dst / name
        if s.exists() and not d.exists():
            shutil.copy2(s, d)
            copied.append(name)
    return copied


def main() -> None:
    args = parse_args()
    src = Path(args.src_model)
    dst = Path(args.dst_model)
    vocab_tsv = Path(args.vocab_tsv)
    fallback_fields = [x.strip() for x in args.fallback_prompt_fields.split(",") if x.strip()]

    rows = read_vocab_rows(vocab_tsv)
    tok_ref = AutoTokenizer.from_pretrained(str(src), trust_remote_code=args.trust_remote_code)
    tok_new = AutoTokenizer.from_pretrained(str(src), trust_remote_code=args.trust_remote_code)
    orig_vocab = len(tok_new)

    n_added, added_rows, skipped_existing = add_tokens_unique(tok_new, rows)
    tok_new.save_pretrained(str(dst))
    new_vocab = len(tok_new)
    added_by_group = dict(Counter(r.get("group", "unknown") for r in added_rows))

    model_info: Dict = {
        "status": "skip",
        "message": "skip-model enabled; model not resized/initialized.",
    }

    if not args.skip_model:
        dtype = _resolve_dtype(args.torch_dtype)
        device_map = _resolve_device_map(args.device_map)
        model = AutoModelForImageTextToText.from_pretrained(
            str(src),
            trust_remote_code=args.trust_remote_code,
            dtype=dtype,
            low_cpu_mem_usage=args.low_cpu_mem_usage,
            device_map=device_map,
        )
        sem_stats = semantic_initialize_new_tokens_from_prompt(
            model=model,
            tokenizer_new=tok_new,
            tokenizer_ref=tok_ref,
            added_rows=added_rows,
            prompt_field=args.prompt_field,
            fallback_prompt_fields=fallback_fields,
        )
        model.save_pretrained(str(dst))
        model_info = {
            "status": "ok",
            "message": "Model embeddings resized and prompt-initialized.",
            **sem_stats,
        }

    copied = copy_extra_files(src=src, dst=dst)

    report = {
        "paths": {
            "src_model": str(src),
            "dst_model": str(dst),
            "vocab_tsv": str(vocab_tsv),
        },
        "prompt": {
            "primary": args.prompt_field,
            "fallback": fallback_fields,
        },
        "sizes": {
            "orig_vocab": orig_vocab,
            "new_vocab": new_vocab,
        },
        "counts": {
            "requested_tokens": len(rows),
            "added_tokens": n_added,
            "skipped_existing": skipped_existing,
            "added_by_group": added_by_group,
            "skip_model": bool(args.skip_model),
        },
        "copied_extra_files": copied,
        "model_init": model_info,
    }
    write_report(dst=dst, report=report)


if __name__ == "__main__":
    main()
