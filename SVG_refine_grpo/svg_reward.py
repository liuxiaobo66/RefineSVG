#!/usr/bin/env python3
from __future__ import annotations

import base64
import io
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import cairosvg
import numpy as np
import requests
from PIL import Image
from tokenizers import Tokenizer as HFTokenizer


DEFAULT_CLIP_API_URL = os.getenv("SVG_REWARD_CLIP_API_URL", "http://127.0.0.1:18080")
DEFAULT_CLIP_API_TIMEOUT = float(os.getenv("SVG_REWARD_CLIP_API_TIMEOUT", "30"))
DEFAULT_DINO_API_URL = os.getenv("SVG_REWARD_DINO_API_URL", "")
DEFAULT_DINO_API_TIMEOUT = float(os.getenv("SVG_REWARD_DINO_API_TIMEOUT", "30"))
DEFAULT_LPIPS_API_URL = os.getenv("SVG_REWARD_LPIPS_API_URL", "")
DEFAULT_LPIPS_API_TIMEOUT = float(os.getenv("SVG_REWARD_LPIPS_API_TIMEOUT", "30"))
DEFAULT_SVG_TOKENIZER_JSON_PATH = os.getenv(
    "SVG_REWARD_TOKENIZER_JSON_PATH",
    "${TOKENIZER_PATH}",
)
SVG_BLOCK_PATTERN = r"(<svg\b[\s\S]*?</svg>)"


@dataclass
class SVGRewardConfig:
    alpha_t: float = 3.0
    clip_threshold: float = 0.3
    clip_api_url: str = DEFAULT_CLIP_API_URL
    clip_api_timeout: float = DEFAULT_CLIP_API_TIMEOUT
    canvas_size: int = 256
    tokenizer_json_path: Optional[str] = DEFAULT_SVG_TOKENIZER_JSON_PATH
    format_failure_penalty: float = -0.5
    svg_missing_penalty: float = -0.5
    render_failure_penalty: float = -0.5
    weight_l2: float = 0.0
    weight_dino: float = 1.0
    weight_efficiency: float = 0.0
    dino_api_url: str = DEFAULT_DINO_API_URL
    dino_api_timeout: float = DEFAULT_DINO_API_TIMEOUT
    lpips_api_url: str = DEFAULT_LPIPS_API_URL
    lpips_api_timeout: float = DEFAULT_LPIPS_API_TIMEOUT


class _SingletonState:
    length_tokenizer = None
    length_tokenizer_path = None
    clip_session = None
    dino_session = None
    lpips_session = None


def build_svg_reward_result(
    cfg: SVGRewardConfig,
    extra_info: Optional[dict[str, Any]] = None,
    **overrides: Any,
) -> dict[str, Any]:
    extra_info = extra_info or {}
    result = {
        "score": 0.0,
        "weighted_sum": 0.0,
        "format_penalty": 0.0,
        "svg_missing_penalty": 0.0,
        "render_failure_penalty": 0.0,
        "total_penalty": 0.0,
        "format_ok": False,
        "has_svg_draft_tag": False,
        "has_svg_final_tag": False,
        "mse": 0.0,
        "psnr": 0.0,
        "ssim": 0.0,
        "l2_score": 0.0,
        "clip_score": 0.0,
        "clip_score_thresholded": 0.0,
        "dino_score": 0.0,
        "dino_score_thresholded": 0.0,
        "efficiency_multiplier": 0.0,
        "efficiency_ratio": 0.0,
        "gen_svg_token_length": 0,
        "gt_svg_token_length": extra_info.get("gt_svg_token_length", 0) or 0,
        "render_success": False,
        "score_dino_only": 0.0,
        "score_dino_efficiency": 0.0,
        "score_all_product": 0.0,
        "lpips_score": 0.0,
        "lpips_distance": 0.0,
        "svg_extracted": False,
    }
    for key, value in overrides.items():
        if value is not None:
            result[key] = value
    return result


def _normalize_clip_api_url(base_url: Optional[str] = None) -> str:
    url = (base_url or DEFAULT_CLIP_API_URL).strip()
    if url.endswith("/v1/similarity"):
        return url
    return url.rstrip("/") + "/v1/similarity"


def _get_clip_session() -> requests.Session:
    if _SingletonState.clip_session is None:
        _SingletonState.clip_session = requests.Session()
    return _SingletonState.clip_session


def _get_dino_session() -> requests.Session:
    if _SingletonState.dino_session is None:
        _SingletonState.dino_session = requests.Session()
    return _SingletonState.dino_session


def _get_lpips_session() -> requests.Session:
    if _SingletonState.lpips_session is None:
        _SingletonState.lpips_session = requests.Session()
    return _SingletonState.lpips_session


def _resolve_tokenizer_json_path(tokenizer_json_path: Optional[str]) -> Optional[str]:
    if not tokenizer_json_path:
        return None
    path = Path(tokenizer_json_path)
    if path.is_dir():
        path = path / "tokenizer.json"
    return str(path)


def _load_length_tokenizer(tokenizer_json_path: Optional[str]):
    resolved_path = _resolve_tokenizer_json_path(tokenizer_json_path)
    if not resolved_path:
        return None, "char_fallback"
    if _SingletonState.length_tokenizer is not None and _SingletonState.length_tokenizer_path == resolved_path:
        return _SingletonState.length_tokenizer, "tokenizer_json"
    if not Path(resolved_path).exists():
        raise FileNotFoundError(f"tokenizer.json not found: {resolved_path}")
    tokenizer = HFTokenizer.from_file(resolved_path)
    _SingletonState.length_tokenizer = tokenizer
    _SingletonState.length_tokenizer_path = resolved_path
    return tokenizer, "tokenizer_json"


def _check_required_format(solution_str: str) -> tuple[bool, bool, bool]:
    has_draft = bool(
        re.search(r"<SVG_DRAFT>\s*.*?\s*</SVG_DRAFT>", solution_str, flags=re.IGNORECASE | re.DOTALL)
    )
    has_final = bool(
        re.search(r"<SVG_FINAL>\s*.*?\s*</SVG_FINAL>", solution_str, flags=re.IGNORECASE | re.DOTALL)
    )
    return has_draft and has_final, has_draft, has_final


def _extract_final_svg(solution_str: str) -> tuple[Optional[str], str]:
    final_match = re.search(r"<SVG_FINAL>\s*(.*?)\s*</SVG_FINAL>", solution_str, flags=re.IGNORECASE | re.DOTALL)
    if final_match:
        final_content = final_match.group(1).strip()
        final_svg_match = re.search(SVG_BLOCK_PATTERN, final_content, flags=re.IGNORECASE)
        if final_svg_match:
            return final_svg_match.group(1).strip(), "svg_final_tag"

    svg_matches = re.findall(SVG_BLOCK_PATTERN, solution_str, flags=re.IGNORECASE)
    if svg_matches:
        return svg_matches[-1].strip(), "last_svg_fallback"

    return None, "not_found"


def _to_white_rgb_pil(image: Any, resize_to: Optional[tuple[int, int]] = None) -> Image.Image:
    if isinstance(image, str):
        if image.startswith("file://"):
            image = image[7:]
        image = Image.open(image)
    elif isinstance(image, Path):
        image = Image.open(image)
    elif isinstance(image, dict) and "bytes" in image:
        image = Image.open(io.BytesIO(image["bytes"]))

    if not isinstance(image, Image.Image):
        raise TypeError(f"Unsupported image type: {type(image)}")

    rgba = image.convert("RGBA")
    bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    rgb = Image.alpha_composite(bg, rgba).convert("RGB")
    if resize_to is not None and rgb.size != resize_to:
        rgb = rgb.resize(resize_to, Image.Resampling.BILINEAR)
    return rgb


def _render_svg_to_rgb(svg: str, size: int) -> Image.Image:
    png_bytes = cairosvg.svg2png(
        bytestring=svg.encode("utf-8"),
        output_width=size,
        output_height=size,
        background_color="white",
    )
    image = Image.open(io.BytesIO(png_bytes))
    return _to_white_rgb_pil(image, resize_to=(size, size))


def _compute_mse(target_rgb: Image.Image, pred_rgb: Image.Image) -> float:
    target = np.asarray(target_rgb, dtype=np.float32) / 255.0
    pred = np.asarray(pred_rgb, dtype=np.float32) / 255.0
    return float(np.mean((target - pred) ** 2))


def _compute_l2_score(mse: float, alpha_t: float) -> float:
    return float(math.exp(-alpha_t * mse))


def _compute_psnr(mse: float) -> float:
    """PSNR in dB for images normalized to [0, 1].  MAX=1.0."""
    if mse <= 0:
        return 100.0
    return float(10.0 * math.log10(1.0 / mse))


def _compute_ssim(target_rgb: Image.Image, pred_rgb: Image.Image, win_size: int = 11) -> float:
    """Mean SSIM over RGB channels using sliding-window statistics."""
    from scipy.ndimage import uniform_filter

    target = np.asarray(target_rgb, dtype=np.float64) / 255.0
    pred = np.asarray(pred_rgb, dtype=np.float64) / 255.0

    C1 = 0.01 ** 2  # (K1 * L)^2, L=1.0
    C2 = 0.03 ** 2  # (K2 * L)^2

    ssim_per_channel = []
    for c in range(target.shape[2]):
        x = target[:, :, c]
        y = pred[:, :, c]

        mu_x = uniform_filter(x, size=win_size)
        mu_y = uniform_filter(y, size=win_size)

        sigma_x_sq = np.maximum(uniform_filter(x ** 2, size=win_size) - mu_x ** 2, 0.0)
        sigma_y_sq = np.maximum(uniform_filter(y ** 2, size=win_size) - mu_y ** 2, 0.0)
        sigma_xy = uniform_filter(x * y, size=win_size) - mu_x * mu_y

        num = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
        den = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x_sq + sigma_y_sq + C2)
        ssim_per_channel.append(float((num / den).mean()))

    return float(np.mean(ssim_per_channel))


def _pil_to_png_b64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _compute_clip_similarity_via_api(
    target_rgb: Image.Image,
    pred_rgb: Image.Image,
    clip_api_url: str,
    clip_api_timeout: float,
) -> tuple[float, dict[str, Any]]:
    payload = {
        "pairs": [
            {
                "image_a_b64": _pil_to_png_b64(target_rgb),
                "image_b_b64": _pil_to_png_b64(pred_rgb),
            }
        ],
        "normalize_to_01": True,
    }
    session = _get_clip_session()
    response = session.post(
        _normalize_clip_api_url(clip_api_url),
        json=payload,
        timeout=clip_api_timeout,
    )
    response.raise_for_status()
    data = response.json()
    item = data["results"][0]
    if item.get("error") is not None:
        raise RuntimeError(item["error"])
    return float(item["similarity"]), {
        "clip_backend": "api",
        "clip_api_url": clip_api_url,
        "clip_api_request_time_ms": data.get("request_time_ms"),
        "clip_api_batch_size_last": data.get("batch_size_last"),
        "clip_api_device": data.get("device"),
        "clip_api_queue_size": data.get("queue_size"),
        "clip_api_model_path": data.get("model_path"),
    }


def _normalize_dino_api_url(base_url: Optional[str] = None) -> str:
    url = (base_url or DEFAULT_DINO_API_URL).strip()
    if url.endswith("/v1/similarity"):
        return url
    return url.rstrip("/") + "/v1/similarity"


def _compute_dino_similarity_via_api(
    target_rgb: Image.Image,
    pred_rgb: Image.Image,
    dino_api_url: str,
    dino_api_timeout: float,
) -> float:
    payload = {
        "pairs": [
            {
                "image_a_b64": _pil_to_png_b64(target_rgb),
                "image_b_b64": _pil_to_png_b64(pred_rgb),
            }
        ],
        "normalize_to_01": True,
    }
    session = _get_dino_session()
    response = session.post(
        _normalize_dino_api_url(dino_api_url),
        json=payload,
        timeout=dino_api_timeout,
    )
    response.raise_for_status()
    data = response.json()
    item = data["results"][0]
    if item.get("error") is not None:
        raise RuntimeError(item["error"])
    return float(item["score"])


def _normalize_lpips_api_url(base_url: Optional[str] = None) -> str:
    url = (base_url or DEFAULT_LPIPS_API_URL).strip()
    if url.endswith("/v1/similarity"):
        return url
    return url.rstrip("/") + "/v1/similarity"


def _compute_lpips_via_api(
    target_rgb: Image.Image,
    pred_rgb: Image.Image,
    lpips_api_url: str,
    lpips_api_timeout: float,
) -> tuple[float, float]:
    """Returns (score, lpips_distance)."""
    payload = {
        "pairs": [
            {
                "image_a_b64": _pil_to_png_b64(target_rgb),
                "image_b_b64": _pil_to_png_b64(pred_rgb),
            }
        ],
        "normalize_to_01": True,
    }
    session = _get_lpips_session()
    response = session.post(
        _normalize_lpips_api_url(lpips_api_url),
        json=payload,
        timeout=lpips_api_timeout,
    )
    response.raise_for_status()
    data = response.json()
    item = data["results"][0]
    if item.get("error") is not None:
        raise RuntimeError(item["error"])
    return float(item["score"]), float(item.get("lpips_distance", 0.0))


def _threshold_clip_score(clip_score: float, threshold: float) -> float:
    if clip_score < threshold:
        return 0.0
    return float(clip_score)


def _compute_generated_length(svg: str, tokenizer_json_path: Optional[str]) -> tuple[int, str]:
    tokenizer, source = _load_length_tokenizer(tokenizer_json_path)
    if tokenizer is None:
        return len(svg), source
    return len(tokenizer.encode(svg).ids), source


def _compute_efficiency_multiplier(gen_len: int, gt_len: Optional[int]) -> tuple[float, Optional[float]]:
    if gt_len is None or gt_len <= 0:
        return 1.0, None
    r = float(gen_len) / float(gt_len)
    x = max(0.0, min((r - 1.5) / (3.0 - 1.5), 1.0))
    m_eff = 0.5 * (1.0 + math.cos(math.pi * x))
    return float(m_eff), float(r)


def compute_svg_refine_reward(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[dict[str, Any]] = None,
    reward_router_address: Optional[str] = None,
    reward_model_tokenizer=None,
    alpha_t: float = 3.0,
    clip_threshold: float = 0.3,
    clip_api_url: str = DEFAULT_CLIP_API_URL,
    clip_api_timeout: float = DEFAULT_CLIP_API_TIMEOUT,
    clip_model_path: Optional[str] = None,
    clip_device: Optional[str] = None,
    canvas_size: int = 256,
    tokenizer_json_path: Optional[str] = DEFAULT_SVG_TOKENIZER_JSON_PATH,
    tokenizer_path: Optional[str] = None,
    format_failure_penalty: float = -0.5,
    svg_missing_penalty: float = -0.5,
    render_failure_penalty: float = -0.5,
    weight_l2: float = 0.0,
    weight_dino: float = 1.0,
    weight_efficiency: float = 0.0,
    dino_api_url: str = DEFAULT_DINO_API_URL,
    dino_api_timeout: float = DEFAULT_DINO_API_TIMEOUT,
    lpips_api_url: str = DEFAULT_LPIPS_API_URL,
    lpips_api_timeout: float = DEFAULT_LPIPS_API_TIMEOUT,
) -> dict[str, Any]:
    del data_source, reward_router_address, reward_model_tokenizer, clip_model_path, clip_device

    extra_info = extra_info or {}
    cfg = SVGRewardConfig(
        alpha_t=alpha_t,
        clip_threshold=clip_threshold,
        clip_api_url=clip_api_url,
        clip_api_timeout=clip_api_timeout,
        canvas_size=canvas_size,
        tokenizer_json_path=tokenizer_json_path or tokenizer_path,
        format_failure_penalty=format_failure_penalty,
        svg_missing_penalty=svg_missing_penalty,
        render_failure_penalty=render_failure_penalty,
        weight_l2=weight_l2,
        weight_dino=weight_dino,
        weight_efficiency=weight_efficiency,
        dino_api_url=dino_api_url,
        dino_api_timeout=dino_api_timeout,
        lpips_api_url=lpips_api_url,
        lpips_api_timeout=lpips_api_timeout,
    )

    format_ok, has_draft_tag, has_final_tag = _check_required_format(solution_str)
    format_penalty = 0.0 if format_ok else cfg.format_failure_penalty

    svg, _extract_method = _extract_final_svg(solution_str)
    if not svg:
        weighted_sum = 0.0
        # Use the more severe penalty (not additive) to cap at -0.5
        total_penalty = min(format_penalty, cfg.svg_missing_penalty)
        return build_svg_reward_result(
            cfg,
            extra_info=extra_info,
            score=weighted_sum + total_penalty,
            weighted_sum=weighted_sum,
            format_penalty=format_penalty,
            svg_missing_penalty=cfg.svg_missing_penalty,
            total_penalty=total_penalty,
            format_ok=format_ok,
            has_svg_draft_tag=has_draft_tag,
            has_svg_final_tag=has_final_tag,
            svg_extracted=False,
        )

    target_image_path = extra_info.get("processed_image_path") or extra_info.get("original_image_path")
    if not target_image_path:
        weighted_sum = 0.0
        total_penalty = min(format_penalty, cfg.render_failure_penalty)
        return build_svg_reward_result(
            cfg,
            extra_info=extra_info,
            score=weighted_sum + total_penalty,
            weighted_sum=weighted_sum,
            format_penalty=format_penalty,
            render_failure_penalty=cfg.render_failure_penalty,
            total_penalty=total_penalty,
            format_ok=format_ok,
            has_svg_draft_tag=has_draft_tag,
            has_svg_final_tag=has_final_tag,
            svg_extracted=True,
        )

    try:
        target_rgb = _to_white_rgb_pil(target_image_path, resize_to=(cfg.canvas_size, cfg.canvas_size))
        pred_rgb = _render_svg_to_rgb(svg, size=cfg.canvas_size)
        render_success = True
    except Exception:
        weighted_sum = 0.0
        total_penalty = min(format_penalty, cfg.render_failure_penalty)
        return build_svg_reward_result(
            cfg,
            extra_info=extra_info,
            score=weighted_sum + total_penalty,
            weighted_sum=weighted_sum,
            format_penalty=format_penalty,
            render_failure_penalty=cfg.render_failure_penalty,
            total_penalty=total_penalty,
            format_ok=format_ok,
            has_svg_draft_tag=has_draft_tag,
            has_svg_final_tag=has_final_tag,
            svg_extracted=True,
        )

    mse = _compute_mse(target_rgb, pred_rgb)
    l2_score = _compute_l2_score(mse, cfg.alpha_t)
    psnr = _compute_psnr(mse)
    ssim = _compute_ssim(target_rgb, pred_rgb)

    # DINO score: used in reward with threshold
    dino_score = 0.0
    if cfg.dino_api_url:
        try:
            dino_score = _compute_dino_similarity_via_api(
                target_rgb=target_rgb,
                pred_rgb=pred_rgb,
                dino_api_url=cfg.dino_api_url,
                dino_api_timeout=cfg.dino_api_timeout,
            )
        except Exception:
            dino_score = 0.0
    dino_score_thresholded = _threshold_clip_score(dino_score, cfg.clip_threshold)

    # CLIP score: monitoring only (not used in reward), logged to wandb
    try:
        clip_score, _ = _compute_clip_similarity_via_api(
            target_rgb=target_rgb,
            pred_rgb=pred_rgb,
            clip_api_url=cfg.clip_api_url,
            clip_api_timeout=cfg.clip_api_timeout,
        )
    except Exception:
        clip_score = 0.0
    clip_score_thresholded = _threshold_clip_score(clip_score, cfg.clip_threshold)

    # LPIPS score: monitoring only (not used in reward), logged to wandb
    lpips_score = 0.0
    lpips_distance = 0.0
    if cfg.lpips_api_url:
        try:
            lpips_score, lpips_distance = _compute_lpips_via_api(
                target_rgb=target_rgb,
                pred_rgb=pred_rgb,
                lpips_api_url=cfg.lpips_api_url,
                lpips_api_timeout=cfg.lpips_api_timeout,
            )
        except Exception:
            lpips_score = 0.0
            lpips_distance = 0.0

    gt_svg_token_length = extra_info.get("gt_svg_token_length", 0) or 0
    gen_svg_token_length, _length_source = _compute_generated_length(
        svg,
        tokenizer_json_path=cfg.tokenizer_json_path,
    )
    efficiency_multiplier, efficiency_ratio = _compute_efficiency_multiplier(
        gen_svg_token_length,
        gt_svg_token_length,
    )

    score_dino_only = dino_score_thresholded
    score_dino_efficiency = dino_score_thresholded * efficiency_multiplier
    score_all_product = dino_score_thresholded * efficiency_multiplier * l2_score
    weighted_sum = (
        cfg.weight_l2 * l2_score
        + cfg.weight_dino * dino_score_thresholded
        + cfg.weight_efficiency * efficiency_multiplier
    )
    total_penalty = format_penalty

    return build_svg_reward_result(
        cfg,
        extra_info=extra_info,
        score=weighted_sum + total_penalty,
        weighted_sum=weighted_sum,
        format_penalty=format_penalty,
        total_penalty=total_penalty,
        format_ok=format_ok,
        has_svg_draft_tag=has_draft_tag,
        has_svg_final_tag=has_final_tag,
        mse=mse,
        psnr=psnr,
        ssim=ssim,
        l2_score=l2_score,
        clip_score=clip_score,
        clip_score_thresholded=clip_score_thresholded,
        dino_score=dino_score,
        dino_score_thresholded=dino_score_thresholded,
        efficiency_multiplier=efficiency_multiplier,
        efficiency_ratio=efficiency_ratio,
        gen_svg_token_length=gen_svg_token_length,
        gt_svg_token_length=gt_svg_token_length,
        render_success=render_success,
        score_dino_only=score_dino_only,
        score_dino_efficiency=score_dino_efficiency,
        score_all_product=score_all_product,
        lpips_score=lpips_score,
        lpips_distance=lpips_distance,
        svg_extracted=True,
    )


def compute_score(*args, **kwargs):
    return compute_svg_refine_reward(*args, **kwargs)


if __name__ == "__main__":
    sample_solution = (
        "<SVG_DRAFT><svg xmlns='http://www.w3.org/2000/svg' width='256' height='256'>"
        "<rect width='256' height='256' fill='white'/></svg></SVG_DRAFT>"
        "<SVG_FINAL><svg xmlns='http://www.w3.org/2000/svg' width='256' height='256'>"
        "<rect width='256' height='256' fill='white'/></svg></SVG_FINAL>"
    )
    print(
        compute_svg_refine_reward(
            data_source="debug",
            solution_str=sample_solution,
            ground_truth="",
            extra_info={},
        )
    )
