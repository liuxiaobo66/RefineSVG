# Copyright 2026
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import asyncio
import io
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

import cairosvg
import numpy as np
from PIL import Image, ImageDraw

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionParametersSchema, OpenAIFunctionPropertySchema, OpenAIFunctionSchema
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class SVGReactFeedbackTool(BaseTool):
    """
    Render SVG to image, compare against target image, and return a 3-panel triptych:

    left   : target image
    middle : current rendered SVG
    right  : diff heatmap

    CPU-bound rendering runs in a thread pool to avoid blocking the asyncio
    event loop, enabling other agent-loop coroutines (GPU inference requests)
    to proceed in parallel.
    """

    # Shared thread pool across all tool instances within the same worker process.
    # CairoSVG and numpy release the GIL, so threads provide real parallelism here.
    _executor: ThreadPoolExecutor | None = None

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self.canvas_size = int(config.get("canvas_size", 256))
        self.norm_percentile = float(config.get("norm_percentile", 99.0))
        self.fallback_bg = tuple(config.get("fallback_bg", [255, 255, 255]))
        self.error_line_color = tuple(config.get("error_line_color", [220, 20, 60]))
        # Lazily create a shared thread pool (32 threads covers typical concurrency)
        if SVGReactFeedbackTool._executor is None:
            SVGReactFeedbackTool._executor = ThreadPoolExecutor(
                max_workers=int(config.get("render_threads", 32)),
                thread_name_prefix="svg_render",
            )

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return OpenAIFunctionToolSchema(
            type="function",
            function=OpenAIFunctionSchema(
                name="svg_react_feedback",
                description=(
                    "Render a complete SVG draft, compare it against the target image, "
                    "and return a 3-panel feedback image: target | render | diff heatmap."
                ),
                parameters=OpenAIFunctionParametersSchema(
                    type="object",
                    properties={
                        "svg": OpenAIFunctionPropertySchema(
                            type="string",
                            description="A complete SVG string to render and evaluate.",
                        ),
                    },
                    required=["svg"],
                ),
            ),
        )

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        svg = parameters.get("svg")
        if not svg or not isinstance(svg, str):
            return ToolResponse(text="Missing required `svg` string.", image=[self._make_error_canvas()]), 0.0, {}

        target_image = kwargs.get("target_image")
        if target_image is None:
            return (
                ToolResponse(text="Missing target image for SVG feedback tool.", image=[self._make_error_canvas()]),
                0.0,
                {},
            )

        # Offload CPU-bound rendering + diff to thread pool so the asyncio
        # event loop stays free for other coroutines (GPU inference requests).
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor, self._execute_sync, svg, target_image
        )

    def _execute_sync(self, svg: str, target_image: Any) -> tuple[ToolResponse, float, dict]:
        """Synchronous rendering + diff, runs in thread pool."""
        target_rgb = self._to_white_rgb_pil(target_image, resize_to=(self.canvas_size, self.canvas_size))

        render_error = None
        try:
            rendered_rgb = self._render_svg_to_rgb(svg, size=self.canvas_size)
        except Exception as e:
            render_error = str(e)
            logger.warning("Failed to render SVG with CairoSVG: %s", e)
            rendered_rgb = self._make_error_canvas()

        target_arr = np.asarray(target_rgb, dtype=np.float32) / 255.0
        rendered_arr = np.asarray(rendered_rgb, dtype=np.float32) / 255.0
        diff = self._build_diff_map(target_arr, rendered_arr, self.norm_percentile)
        diff_rgb = Image.fromarray(self._colorize_heatmap(diff), mode="RGB")
        triptych = self._make_triptych(target_rgb, rendered_rgb, diff_rgb)

        mean_diff = float(diff.mean())
        max_diff = float(diff.max())
        score = max(0.0, 1.0 - mean_diff)

        if render_error is None:
            text = (
                f"SVG rendered successfully. mean_diff={mean_diff:.6f}, "
                f"max_diff={max_diff:.6f}, similarity={score:.6f}"
            )
        else:
            text = (
                f"SVG rendering failed and fallback image was used. "
                f"error={render_error}; mean_diff={mean_diff:.6f}, max_diff={max_diff:.6f}"
            )

        metrics = {
            "mean_diff": mean_diff,
            "max_diff": max_diff,
            "similarity": score,
            "render_success": render_error is None,
        }
        return ToolResponse(text=text, image=[triptych]), 0.0, metrics

    def _render_svg_to_rgb(self, svg: str, size: int) -> Image.Image:
        png_bytes = cairosvg.svg2png(
            bytestring=svg.encode("utf-8"),
            output_width=size,
            output_height=size,
            background_color="white",
        )
        image = Image.open(io.BytesIO(png_bytes))
        return self._to_white_rgb_pil(image, resize_to=(size, size))

    def _to_white_rgb_pil(self, image: Any, resize_to: Optional[tuple[int, int]] = None) -> Image.Image:
        if not isinstance(image, Image.Image):
            if isinstance(image, dict) and "bytes" in image:
                image = Image.open(io.BytesIO(image["bytes"]))
            else:
                raise TypeError(f"Unsupported image type: {type(image)}")

        image = image.convert("RGBA")
        white_bg = Image.new("RGBA", image.size, (255, 255, 255, 255))
        composited = Image.alpha_composite(white_bg, image).convert("RGB")
        if resize_to is not None and composited.size != resize_to:
            composited = composited.resize(resize_to, Image.Resampling.BILINEAR)
        return composited

    def _make_error_canvas(self) -> Image.Image:
        image = Image.new("RGB", (self.canvas_size, self.canvas_size), self.fallback_bg)
        draw = ImageDraw.Draw(image)
        pad = max(4, self.canvas_size // 16)
        draw.line((pad, pad, self.canvas_size - pad, self.canvas_size - pad), fill=self.error_line_color, width=4)
        draw.line((self.canvas_size - pad, pad, pad, self.canvas_size - pad), fill=self.error_line_color, width=4)
        return image

    def _make_triptych(self, target: Image.Image, rendered: Image.Image, diff: Image.Image) -> Image.Image:
        w, h = target.size
        canvas = Image.new("RGB", (w * 3, h), (255, 255, 255))
        canvas.paste(target, (0, 0))
        canvas.paste(rendered, (w, 0))
        canvas.paste(diff, (w * 2, 0))
        return canvas

    @staticmethod
    def _to_ycbcr(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        y = 0.299 * r + 0.587 * g + 0.114 * b
        cb = (b - y) * 0.564
        cr = (r - y) * 0.713
        return y, cb, cr

    @staticmethod
    def _gradmag(x: np.ndarray) -> np.ndarray:
        gx = np.zeros_like(x, dtype=np.float32)
        gy = np.zeros_like(x, dtype=np.float32)
        gx[:, 1:-1] = (x[:, 2:] - x[:, :-2]) * 0.5
        gx[:, 0] = x[:, 1] - x[:, 0]
        gx[:, -1] = x[:, -1] - x[:, -2]
        gy[1:-1, :] = (x[2:, :] - x[:-2, :]) * 0.5
        gy[0, :] = x[1, :] - x[0, :]
        gy[-1, :] = x[-1, :] - x[-2, :]
        return np.sqrt(gx * gx + gy * gy)

    @staticmethod
    def _robust_norm(x: np.ndarray, percentile: float) -> np.ndarray:
        s = float(np.percentile(x, percentile))
        s = max(s, 1e-6)
        return np.clip(x / s, 0.0, 1.0)

    def _build_diff_map(self, target_rgb: np.ndarray, pred_rgb: np.ndarray, percentile: float) -> np.ndarray:
        yt, cbt, crt = self._to_ycbcr(target_rgb)
        yp, cbp, crp = self._to_ycbcr(pred_rgb)

        d_luma = np.abs(yt - yp)
        d_chroma = np.sqrt((cbt - cbp) ** 2 + (crt - crp) ** 2)
        d_edge = np.abs(self._gradmag(yt) - self._gradmag(yp))

        nl = self._robust_norm(d_luma, percentile)
        nc = self._robust_norm(d_chroma, percentile)
        ne = self._robust_norm(d_edge, percentile)

        score = 0.25 * nl + 0.35 * nc + 0.40 * ne
        score = np.clip(score, 0.0, 1.0) ** 0.75
        return score

    @staticmethod
    def _colorize_heatmap(x: np.ndarray) -> np.ndarray:
        x = np.clip(x, 0.0, 1.0)
        anchors = np.array(
            [
                [0, 0, 0],
                [0, 0, 180],
                [0, 210, 255],
                [255, 230, 0],
                [255, 80, 0],
                [255, 0, 0],
            ],
            dtype=np.float32,
        )
        pos = np.array([0.0, 0.2, 0.4, 0.65, 0.82, 1.0], dtype=np.float32)
        out = np.zeros((*x.shape, 3), dtype=np.float32)
        for c in range(3):
            out[..., c] = np.interp(x, pos, anchors[:, c])
        return np.clip(out, 0, 255).astype(np.uint8)
