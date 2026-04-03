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

"""
Ablation variant: 2-panel feedback tool (no diff heatmap).

Returns a diptych image:
  left  : target image
  right : rendered SVG

Identical to SVGReactFeedbackTool except all diff-related computation
(_build_diff_map, _colorize_heatmap, _to_ycbcr, _gradmag, _robust_norm)
is removed.
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

import cairosvg
from PIL import Image, ImageDraw

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionParametersSchema, OpenAIFunctionPropertySchema, OpenAIFunctionSchema
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class SVGReactFeedbackToolNoDiff(BaseTool):
    """
    Render SVG to image, compare against target image, and return a 2-panel diptych:

    left   : target image
    right  : current rendered SVG

    No diff heatmap is produced (ablation: remove visual diff feedback).
    """

    _executor: ThreadPoolExecutor | None = None

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self.canvas_size = int(config.get("canvas_size", 256))
        self.fallback_bg = tuple(config.get("fallback_bg", [255, 255, 255]))
        self.error_line_color = tuple(config.get("error_line_color", [220, 20, 60]))
        if SVGReactFeedbackToolNoDiff._executor is None:
            SVGReactFeedbackToolNoDiff._executor = ThreadPoolExecutor(
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
                    "and return a 2-panel feedback image: target | render."
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

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor, self._execute_sync, svg, target_image
        )

    def _execute_sync(self, svg: str, target_image: Any) -> tuple[ToolResponse, float, dict]:
        """Synchronous rendering, runs in thread pool."""
        target_rgb = self._to_white_rgb_pil(target_image, resize_to=(self.canvas_size, self.canvas_size))

        render_error = None
        try:
            rendered_rgb = self._render_svg_to_rgb(svg, size=self.canvas_size)
        except Exception as e:
            render_error = str(e)
            logger.warning("Failed to render SVG with CairoSVG: %s", e)
            rendered_rgb = self._make_error_canvas()

        diptych = self._make_diptych(target_rgb, rendered_rgb)

        if render_error is None:
            text = "SVG rendered successfully."
        else:
            text = f"SVG rendering failed and fallback image was used. error={render_error}"

        metrics = {
            "render_success": render_error is None,
        }
        return ToolResponse(text=text, image=[diptych]), 0.0, metrics

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

    def _make_diptych(self, target: Image.Image, rendered: Image.Image) -> Image.Image:
        w, h = target.size
        canvas = Image.new("RGB", (w * 2, h), (255, 255, 255))
        canvas.paste(target, (0, 0))
        canvas.paste(rendered, (w, 0))
        return canvas
