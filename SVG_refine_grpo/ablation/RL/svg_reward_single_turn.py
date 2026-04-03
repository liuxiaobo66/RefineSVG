"""
Reward function for single-turn SVG generation RL.

Adapts the existing svg_reward scoring pipeline for single-turn output:
- Format check: only requires a valid <svg>...</svg> block (no DRAFT/FINAL tags)
- SVG extraction: takes the last <svg>...</svg> block from the output
- Scoring: reuses all existing metrics (CLIP, DINO, L2, efficiency, LPIPS)
"""
from __future__ import annotations

import re
from typing import Any, Optional

from SVG_refine_grpo.svg_reward import (
    SVG_BLOCK_PATTERN,
    SVGRewardConfig,
    build_svg_reward_result,
    _to_white_rgb_pil,
    _render_svg_to_rgb,
    _compute_mse,
    _compute_l2_score,
    _compute_psnr,
    _compute_ssim,
    _compute_clip_similarity_via_api,
    _compute_dino_similarity_via_api,
    _compute_lpips_via_api,
    _compute_generated_length,
    _compute_efficiency_multiplier,
    _threshold_clip_score,
    DEFAULT_CLIP_API_URL,
    DEFAULT_CLIP_API_TIMEOUT,
    DEFAULT_DINO_API_URL,
    DEFAULT_DINO_API_TIMEOUT,
    DEFAULT_LPIPS_API_URL,
    DEFAULT_LPIPS_API_TIMEOUT,
    DEFAULT_SVG_TOKENIZER_JSON_PATH,
)


def _extract_svg_single_turn(solution_str: str) -> Optional[str]:
    """Extract the last <svg>...</svg> block from the output."""
    matches = re.findall(SVG_BLOCK_PATTERN, solution_str, flags=re.IGNORECASE)
    if matches:
        return matches[-1].strip()
    return None


def compute_score(
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

    # Single-turn: extract <svg>...</svg>
    svg = _extract_svg_single_turn(solution_str)

    if not svg:
        return build_svg_reward_result(
            cfg,
            extra_info=extra_info,
            score=cfg.svg_missing_penalty,
            svg_missing_penalty=cfg.svg_missing_penalty,
            total_penalty=cfg.svg_missing_penalty,
            format_ok=False,
            svg_extracted=False,
        )

    target_image_path = extra_info.get("processed_image_path") or extra_info.get("original_image_path")
    if not target_image_path:
        return build_svg_reward_result(
            cfg,
            extra_info=extra_info,
            score=cfg.render_failure_penalty,
            render_failure_penalty=cfg.render_failure_penalty,
            total_penalty=cfg.render_failure_penalty,
            format_ok=True,
            svg_extracted=True,
        )

    try:
        target_rgb = _to_white_rgb_pil(target_image_path, resize_to=(cfg.canvas_size, cfg.canvas_size))
        pred_rgb = _render_svg_to_rgb(svg, size=cfg.canvas_size)
        render_success = True
    except Exception:
        return build_svg_reward_result(
            cfg,
            extra_info=extra_info,
            score=cfg.render_failure_penalty,
            render_failure_penalty=cfg.render_failure_penalty,
            total_penalty=cfg.render_failure_penalty,
            format_ok=True,
            svg_extracted=True,
        )

    mse = _compute_mse(target_rgb, pred_rgb)
    l2_score = _compute_l2_score(mse, cfg.alpha_t)
    psnr = _compute_psnr(mse)
    ssim = _compute_ssim(target_rgb, pred_rgb)

    dino_score = 0.0
    if cfg.dino_api_url:
        try:
            dino_score = _compute_dino_similarity_via_api(
                target_rgb=target_rgb, pred_rgb=pred_rgb,
                dino_api_url=cfg.dino_api_url, dino_api_timeout=cfg.dino_api_timeout,
            )
        except Exception:
            dino_score = 0.0
    dino_score_thresholded = _threshold_clip_score(dino_score, cfg.clip_threshold)

    try:
        clip_score, _ = _compute_clip_similarity_via_api(
            target_rgb=target_rgb, pred_rgb=pred_rgb,
            clip_api_url=cfg.clip_api_url, clip_api_timeout=cfg.clip_api_timeout,
        )
    except Exception:
        clip_score = 0.0
    clip_score_thresholded = _threshold_clip_score(clip_score, cfg.clip_threshold)

    lpips_score = 0.0
    lpips_distance = 0.0
    if cfg.lpips_api_url:
        try:
            lpips_score, lpips_distance = _compute_lpips_via_api(
                target_rgb=target_rgb, pred_rgb=pred_rgb,
                lpips_api_url=cfg.lpips_api_url, lpips_api_timeout=cfg.lpips_api_timeout,
            )
        except Exception:
            pass

    gt_svg_token_length = extra_info.get("gt_svg_token_length", 0) or 0
    gen_svg_token_length, _ = _compute_generated_length(svg, tokenizer_json_path=cfg.tokenizer_json_path)
    efficiency_multiplier, efficiency_ratio = _compute_efficiency_multiplier(gen_svg_token_length, gt_svg_token_length)

    score_dino_only = dino_score_thresholded
    score_dino_efficiency = dino_score_thresholded * efficiency_multiplier
    score_all_product = dino_score_thresholded * efficiency_multiplier * l2_score
    weighted_sum = (
        cfg.weight_l2 * l2_score
        + cfg.weight_dino * dino_score_thresholded
        + cfg.weight_efficiency * efficiency_multiplier
    )

    return build_svg_reward_result(
        cfg,
        extra_info=extra_info,
        score=weighted_sum,
        weighted_sum=weighted_sum,
        format_ok=True,
        mse=mse, psnr=psnr, ssim=ssim,
        l2_score=l2_score,
        clip_score=clip_score, clip_score_thresholded=clip_score_thresholded,
        dino_score=dino_score, dino_score_thresholded=dino_score_thresholded,
        efficiency_multiplier=efficiency_multiplier, efficiency_ratio=efficiency_ratio,
        gen_svg_token_length=gen_svg_token_length, gt_svg_token_length=gt_svg_token_length,
        render_success=render_success,
        score_dino_only=score_dino_only,
        score_dino_efficiency=score_dino_efficiency,
        score_all_product=score_all_product,
        lpips_score=lpips_score, lpips_distance=lpips_distance,
        svg_extracted=True,
    )
