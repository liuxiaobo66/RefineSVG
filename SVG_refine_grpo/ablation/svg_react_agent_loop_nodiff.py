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
Ablation variant: agent loop with 2-panel feedback (no diff heatmap).

Subclasses SVGReactAgentLoop and only overrides DEFAULT_FEEDBACK_TEXT
to remove diff heatmap description.  All other logic is inherited.
"""

from verl.experimental.agent_loop.agent_loop import register
from SVG_refine_grpo.svg_react_agent_loop import SVGReactAgentLoop


@register("svg_react_agent")
class SVGReactAgentLoopNoDiff(SVGReactAgentLoop):
    """Two-stage ReAct loop with 2-panel visual feedback (no diff heatmap)."""

    DEFAULT_FEEDBACK_TEXT = (
        "This feedback image shows, from left to right: (1) the target image and "
        "(2) the rendered result of your previous SVG draft. "
        "Use this feedback to correct the draft. "
        "Preserve correct parts and fix incorrect geometry, colors, structure, and "
        "missing elements. If there is no meaningful difference, return the draft unchanged. "
        "Output only one complete corrected SVG wrapped in <SVG_FINAL>...</SVG_FINAL>."
    )

    # Reset class initialization flag so this subclass initializes independently
    _class_initialized = False
