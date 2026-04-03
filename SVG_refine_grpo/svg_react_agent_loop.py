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
import asyncio
import copy
import json
import logging
import os
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Optional
from uuid import uuid4

from omegaconf import OmegaConf
from SVG_refine_grpo.svg_reward import SVGRewardConfig, build_svg_reward_result
from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.tools.schemas import ToolResponse
from verl.tools.utils.tool_registry import initialize_tools_from_config
from verl.utils.profiler import simple_timer

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class SVGReactAgentData:
    def __init__(
        self,
        messages: list[dict[str, Any]],
        image_data: Optional[list[Any]],
        metrics: dict[str, Any],
        request_id: str,
        tools_kwargs: dict[str, Any],
    ):
        self.messages = messages
        self.metrics = metrics
        self.request_id = request_id
        self.tools_kwargs = tools_kwargs

        self.original_image_data = copy.deepcopy(image_data)
        self.image_data = copy.deepcopy(image_data)

        self.initial_prompt_ids: list[int] = []
        self.trajectory_ids: list[int] = []
        self.response_mask: list[int] = []
        self.response_logprobs: list[float] = []
        self.assistant_turns = 0
        self.user_turns = 0

        self.draft_svg: Optional[str] = None
        self.final_svg: Optional[str] = None
        self.tool_reward: Optional[float] = None
        self.tool_metrics: dict[str, Any] = {}
        self.tool_text: Optional[str] = None


@dataclass
class StageResult:
    """Diagnostic info returned from _generate_stage."""
    text: str
    raw_text: str  # full decoded text BEFORE stop-text truncation
    token_count: int
    raw_token_count: int  # token count before truncation
    was_truncated: bool  # True if stop_text was found and text was truncated
    has_open_tag: bool
    has_close_tag: bool
    elapsed_sec: float = 0.0


@register("svg_react_agent")
class SVGReactAgentLoop(AgentLoopBase):
    """
    A custom two-stage ReAct loop for image -> SVG tasks:

    1. First generation: output <SVG_DRAFT>...</SVG_DRAFT>
    2. Trigger tool with extracted SVG payload
    3. Feed tool result back as a new user turn (usually feedback image + text)
    4. Final generation: output <SVG_FINAL>...</SVG_FINAL>
    """

    DEFAULT_FEEDBACK_TEXT = (
        "This feedback image shows, from left to right: (1) the target image, "
        "(2) the rendered result of your previous SVG draft, and (3) a diff heatmap, "
        "where blue indicates small differences, yellow indicates medium differences, "
        "and red indicates large differences. Use this feedback to correct the draft. "
        "Preserve correct parts and fix incorrect geometry, colors, structure, and "
        "missing elements. If there is no meaningful difference, return the draft unchanged. "
        "Output only one complete corrected SVG wrapped in <SVG_FINAL>...</SVG_FINAL>."
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        rollout_cfg = self.config.actor_rollout_ref.rollout
        self.prompt_length = rollout_cfg.prompt_length
        self.response_length = rollout_cfg.response_length
        self.max_model_len = rollout_cfg.prompt_length + rollout_cfg.response_length
        self.draft_max_tokens = int(getattr(self, "draft_max_tokens", 0) or 0)
        self.final_max_tokens = int(getattr(self, "final_max_tokens", 0) or 0)

    @classmethod
    def init_class(cls, config, tokenizer, processor, **kwargs):
        if cls._class_initialized:
            return
        cls._class_initialized = True

        rollout_cfg = config.actor_rollout_ref.rollout
        cls.tokenizer = tokenizer
        cls.processor = processor
        cls.apply_chat_template_kwargs = config.data.get("apply_chat_template_kwargs", {})
        cls.system_prompt = tokenizer.apply_chat_template(
            [{}], add_generation_prompt=False, tokenize=True, **cls.apply_chat_template_kwargs
        )
        # Pre-compute the system-prompt text that apply_chat_template prepends
        # (e.g. "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n").
        # Used by _encode_delta_messages to strip the duplicate system prefix
        # when encoding only a delta (feedback turn) for VL models.
        if cls.processor is not None:
            cls._processor_system_text = cls.processor.apply_chat_template(
                [{}], add_generation_prompt=False, tokenize=False, **cls.apply_chat_template_kwargs
            )
        else:
            cls._processor_system_text = None

        cls.draft_tag = kwargs.get("draft_tag", "SVG_DRAFT")
        cls.final_tag = kwargs.get("final_tag", "SVG_FINAL")
        cls.tool_argument_name = kwargs.get("tool_argument_name", "svg")
        cls.tool_name = kwargs.get("tool_name", None)
        cls.feedback_text = kwargs.get("feedback_text", cls.DEFAULT_FEEDBACK_TEXT)
        cls.include_tool_text_in_feedback = kwargs.get("include_tool_text_in_feedback", False)
        cls.force_feedback_role = kwargs.get("feedback_role", "user")
        cls.include_stop_str_in_output = kwargs.get("include_stop_str_in_output", True)
        cls.require_feedback_image = kwargs.get("require_feedback_image", True)
        cls.terminate_on_draft_render_failure = bool(kwargs.get("terminate_on_draft_render_failure", True))
        cls.draft_max_tokens = int(kwargs.get("draft_max_tokens", 14000))
        cls.final_max_tokens = int(kwargs.get("final_max_tokens", 14000))
        cls.stage_timeout = float(kwargs.get("stage_timeout", 0))  # 0 = no timeout
        cls.max_tool_response_length = rollout_cfg.multi_turn.max_tool_response_length
        cls.tool_response_truncate_side = rollout_cfg.multi_turn.tool_response_truncate_side

        # Trajectory dump: write per-sample diagnostic JSONL during validation.
        # Set via agent config yaml or env var.  Empty string disables.
        cls.trajectory_dump_dir = kwargs.get(
            "trajectory_dump_dir",
            os.getenv("SVG_TRAJECTORY_DUMP_DIR", ""),
        )
        if cls.trajectory_dump_dir:
            os.makedirs(cls.trajectory_dump_dir, exist_ok=True)
            logger.warning("[SVGReact] trajectory_dump_dir=%s", cls.trajectory_dump_dir)

        tool_config_path = rollout_cfg.multi_turn.tool_config_path
        if not tool_config_path:
            raise ValueError("SVGReactAgentLoop requires actor_rollout_ref.rollout.multi_turn.tool_config_path")

        tool_list = initialize_tools_from_config(tool_config_path)
        cls.tools = {tool.name: tool for tool in tool_list}
        if not cls.tools:
            raise ValueError("No tools initialized for SVGReactAgentLoop")
        if cls.tool_name is None:
            if len(cls.tools) != 1:
                raise ValueError(
                    "SVGReactAgentLoop needs explicit tool_name when multiple tools are configured. "
                    f"Available tools: {list(cls.tools.keys())}"
                )
            cls.tool_name = next(iter(cls.tools.keys()))

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        t_start = time.monotonic()
        messages = list(kwargs["raw_prompt"])
        image_data = self._ensure_list(copy.deepcopy((kwargs.get("multi_modal_data") or {}).get("image", None)))
        request_id = uuid4().hex
        metrics: dict[str, Any] = {}
        tools_kwargs = kwargs.get("tools_kwargs", {})

        agent_data = SVGReactAgentData(
            messages=messages,
            image_data=image_data,
            metrics=metrics,
            request_id=request_id,
            tools_kwargs=tools_kwargs,
        )

        # Trajectory diagnostic record — populated incrementally, dumped at end.
        traj: dict[str, Any] = {
            "request_id": request_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "pid": os.getpid(),
        }

        agent_data.initial_prompt_ids = await self._build_prompt_ids(agent_data.messages, agent_data.image_data)
        agent_data.trajectory_ids = list(agent_data.initial_prompt_ids)
        traj["prompt_tokens"] = len(agent_data.initial_prompt_ids)
        traj["num_images"] = len(agent_data.image_data or [])
        traj["max_model_len"] = self.max_model_len
        logger.warning(
            "[SVGReact] request=%s init_prompt_tokens=%d images=%d max_model_len=%d response_length=%d",
            agent_data.request_id,
            len(agent_data.initial_prompt_ids),
            len(agent_data.image_data or []),
            self.max_model_len,
            self.response_length,
        )

        # Stage 1: draft generation
        t0 = time.monotonic()
        draft_result = await self._generate_stage(
            agent_data=agent_data,
            sampling_params=sampling_params,
            stop_text=f"</{self.draft_tag}>",
            stage_name="draft",
            stage_max_tokens=self.draft_max_tokens,
        )
        t_draft = time.monotonic() - t0
        draft_result.elapsed_sec = t_draft
        agent_data.draft_svg = self._extract_svg_payload(draft_result.text, self.draft_tag)

        # Record draft diagnostics
        traj["draft"] = {
            "raw_token_count": draft_result.raw_token_count,
            "token_count": draft_result.token_count,
            "was_truncated": draft_result.was_truncated,
            "has_open_tag": draft_result.has_open_tag,
            "has_close_tag": draft_result.has_close_tag,
            "svg_extracted": agent_data.draft_svg is not None,
            "svg_chars": len(agent_data.draft_svg or ""),
            "elapsed_sec": round(t_draft, 2),
            # First 2000 chars of raw model output for analysis
            "raw_text_preview": draft_result.raw_text[:2000] if draft_result.raw_text else "",
            # Last 500 chars to see how the output ends
            "raw_text_tail": draft_result.raw_text[-500:] if draft_result.raw_text and len(draft_result.raw_text) > 500 else "",
        }

        logger.warning(
            "[SVGReact] request=%s draft_tokens=%d trajectory_tokens=%d extracted_draft=%s draft_chars=%d",
            agent_data.request_id,
            len(agent_data.trajectory_ids) - len(agent_data.initial_prompt_ids),
            len(agent_data.trajectory_ids),
            agent_data.draft_svg is not None,
            len(agent_data.draft_svg or ""),
        )

        if not agent_data.draft_svg:
            t_total = time.monotonic() - t_start
            timed_out = draft_result.token_count == 0 and draft_result.elapsed_sec >= self.stage_timeout > 0
            traj["outcome"] = "draft_timeout" if timed_out else "draft_extract_failed"
            traj["total_sec"] = round(t_total, 2)
            self._dump_trajectory(traj)
            logger.warning(
                "[SVGReact] request=%s %s (tokens=%d elapsed=%.1fs)",
                agent_data.request_id,
                traj["outcome"],
                draft_result.token_count,
                draft_result.elapsed_sec,
            )
            return self._build_early_terminated_output(agent_data)

        # Tool call
        t0 = time.monotonic()
        tool_response, tool_reward, tool_metrics = await self._call_tool(agent_data, agent_data.draft_svg)
        t_tool = time.monotonic() - t0
        agent_data.tool_reward = tool_reward
        agent_data.tool_metrics = tool_metrics
        agent_data.tool_text = tool_response.text

        traj["tool"] = {
            "render_success": bool(tool_metrics.get("render_success", False)),
            "tool_text": tool_response.text[:500] if tool_response.text else "",
            "elapsed_sec": round(t_tool, 2),
        }

        if self.terminate_on_draft_render_failure and not bool(tool_metrics.get("render_success", True)):
            t_total = time.monotonic() - t_start
            logger.warning(
                "[SVGReact] request=%s TIMING draft=%.2fs tool=%.2fs total=%.2fs (early_terminated)",
                agent_data.request_id, t_draft, t_tool, t_total,
            )
            logger.warning(
                "[SVGReact] request=%s draft_render_failed terminate_early reward=%s error=%s",
                agent_data.request_id,
                self._get_format_failure_penalty(),
                tool_response.text,
            )
            traj["outcome"] = "draft_render_failed"
            traj["total_sec"] = round(t_total, 2)
            self._dump_trajectory(traj)
            return self._build_early_terminated_output(agent_data)

        feedback_images = self._ensure_list(tool_response.image)
        if self.require_feedback_image and not feedback_images:
            raise ValueError("SVGReactAgentLoop expects the feedback tool to return at least one image.")

        t0 = time.monotonic()
        await self._append_feedback_turn(agent_data, tool_response, feedback_images)
        t_feedback = time.monotonic() - t0
        logger.warning(
            "[SVGReact] request=%s feedback_added images=%d tool_text_chars=%d trajectory_tokens=%d remaining_budget=%d",
            agent_data.request_id,
            len(feedback_images),
            len(tool_response.text or ""),
            len(agent_data.trajectory_ids),
            self.max_model_len - len(agent_data.trajectory_ids),
        )

        # Stage 2: final generation
        t0 = time.monotonic()
        final_result = await self._generate_stage(
            agent_data=agent_data,
            sampling_params=sampling_params,
            stop_text=f"</{self.final_tag}>",
            stage_name="final",
            stage_max_tokens=self.final_max_tokens,
        )
        t_final = time.monotonic() - t0
        final_result.elapsed_sec = t_final
        agent_data.final_svg = self._extract_svg_payload(final_result.text, self.final_tag)

        # Record final diagnostics
        traj["final"] = {
            "raw_token_count": final_result.raw_token_count,
            "token_count": final_result.token_count,
            "was_truncated": final_result.was_truncated,
            "has_open_tag": final_result.has_open_tag,
            "has_close_tag": final_result.has_close_tag,
            "svg_extracted": agent_data.final_svg is not None,
            "svg_chars": len(agent_data.final_svg or ""),
            "elapsed_sec": round(t_final, 2),
            "raw_text_preview": final_result.raw_text[:2000] if final_result.raw_text else "",
            "raw_text_tail": final_result.raw_text[-500:] if final_result.raw_text and len(final_result.raw_text) > 500 else "",
        }

        t_total = time.monotonic() - t_start
        draft_tokens = len(agent_data.trajectory_ids) - len(agent_data.initial_prompt_ids)
        logger.warning(
            "[SVGReact] request=%s TIMING draft=%.2fs tool=%.2fs feedback_encode=%.2fs final=%.2fs total=%.2fs tokens=%d",
            agent_data.request_id, t_draft, t_tool, t_feedback, t_final, t_total, draft_tokens,
        )

        traj["outcome"] = "completed"
        traj["total_sec"] = round(t_total, 2)
        self._dump_trajectory(traj)

        response_ids = agent_data.trajectory_ids[len(agent_data.initial_prompt_ids) :]
        response_mask = agent_data.response_mask[: len(response_ids)]
        response_logprobs = agent_data.response_logprobs[: len(response_ids)] if agent_data.response_logprobs else None

        output = AgentLoopOutput(
            prompt_ids=agent_data.initial_prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            response_logprobs=response_logprobs[: self.response_length] if response_logprobs else None,
            multi_modal_data={"image": agent_data.image_data} if agent_data.image_data else {},
            num_turns=1 + agent_data.user_turns + agent_data.assistant_turns,
            metrics=agent_data.metrics,
            extra_fields={
                "draft_svg": agent_data.draft_svg,
                "final_svg": agent_data.final_svg,
                "tool_reward": agent_data.tool_reward,
                "tool_metrics": agent_data.tool_metrics,
                "tool_name": self.tool_name,
                "time_draft_sec": round(t_draft, 3),
                "time_tool_sec": round(t_tool, 3),
                "time_final_sec": round(t_final, 3),
                "time_total_sec": round(t_total, 3),
            },
        )
        return output

    def _dump_trajectory(self, traj: dict[str, Any]) -> None:
        """Append a trajectory diagnostic record as one JSONL line.

        Each Ray worker (process) writes to its own file to avoid contention.
        File: {trajectory_dump_dir}/traj_pid{pid}.jsonl
        """
        dump_dir = getattr(self, "trajectory_dump_dir", "")
        if not dump_dir:
            return
        try:
            path = os.path.join(dump_dir, f"traj_pid{os.getpid()}.jsonl")
            line = json.dumps(traj, ensure_ascii=False, default=str)
            with open(path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception as exc:
            logger.warning("[SVGReact] trajectory dump failed: %s", exc)

    def _get_reward_cfg(self) -> SVGRewardConfig:
        reward_kwargs = OmegaConf.to_container(
            self.config.get("custom_reward_function", {}).get("reward_kwargs", {}),
            resolve=True,
        )
        reward_kwargs = reward_kwargs or {}
        return SVGRewardConfig(
            alpha_t=float(reward_kwargs.get("alpha_t", 3.0)),
            clip_threshold=float(reward_kwargs.get("clip_threshold", 0.3)),
            clip_api_url=reward_kwargs.get("clip_api_url", os.getenv("SVG_REWARD_CLIP_API_URL", "http://127.0.0.1:18080")),
            clip_api_timeout=float(reward_kwargs.get("clip_api_timeout", 30.0)),
            canvas_size=int(reward_kwargs.get("canvas_size", 256)),
            tokenizer_json_path=reward_kwargs.get("tokenizer_json_path"),
            format_failure_penalty=float(reward_kwargs.get("format_failure_penalty", -0.5)),
            svg_missing_penalty=float(reward_kwargs.get("svg_missing_penalty", -0.5)),
            render_failure_penalty=float(reward_kwargs.get("render_failure_penalty", -0.5)),
            weight_l2=float(reward_kwargs.get("weight_l2", 0.0)),
            weight_dino=float(reward_kwargs.get("weight_dino", 1.0)),
            weight_efficiency=float(reward_kwargs.get("weight_efficiency", 0.0)),
        )

    def _get_format_failure_penalty(self) -> float:
        return self._get_reward_cfg().format_failure_penalty

    def _build_early_terminated_output(self, agent_data: SVGReactAgentData) -> AgentLoopOutput:
        response_ids = agent_data.trajectory_ids[len(agent_data.initial_prompt_ids) :]
        response_mask = agent_data.response_mask[: len(response_ids)]
        response_logprobs = agent_data.response_logprobs[: len(response_ids)] if agent_data.response_logprobs else None

        reward_cfg = self._get_reward_cfg()
        reward_extra_info = build_svg_reward_result(
            reward_cfg,
            score=reward_cfg.format_failure_penalty,
            weighted_sum=0.0,
            format_penalty=reward_cfg.format_failure_penalty,
            total_penalty=reward_cfg.format_failure_penalty,
            format_ok=False,
            has_svg_draft_tag=agent_data.draft_svg is not None,
            has_svg_final_tag=False,
            svg_extracted=False,
            render_success=False,
        )

        return AgentLoopOutput(
            prompt_ids=agent_data.initial_prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            response_logprobs=response_logprobs[: self.response_length] if response_logprobs else None,
            multi_modal_data={"image": agent_data.image_data} if agent_data.image_data else {},
            reward_score=reward_cfg.format_failure_penalty,
            num_turns=1 + agent_data.user_turns + agent_data.assistant_turns,
            metrics=agent_data.metrics,
            extra_fields={
                "draft_svg": agent_data.draft_svg,
                "final_svg": None,
                "tool_reward": agent_data.tool_reward,
                "tool_metrics": agent_data.tool_metrics,
                "tool_name": self.tool_name,
                "reward_extra_info": reward_extra_info,
            },
        )

    async def _generate_stage(
        self,
        agent_data: SVGReactAgentData,
        sampling_params: dict[str, Any],
        stop_text: str,
        stage_name: str,
        stage_max_tokens: int,
    ) -> StageResult:
        stage_sampling_params = dict(sampling_params)
        # NOTE: Do NOT set stage_sampling_params["stop"] here.
        # verl uses skip_tokenizer_init=True for SGLang async mode, so SGLang's
        # scheduler has no tokenizer and would crash at schedule_batch.py:742
        # trying to decode tokens for stop-string matching.  Instead we handle
        # stop-string truncation *after* generation using the agent loop's own
        # tokenizer (self.tokenizer).
        #
        # However, we CAN set stop_token_ids (integer IDs, no tokenizer needed)
        # as a safety net.  This tells SGLang to stop when the model emits EOS
        # (<|im_end|>), preventing wasted decode steps after the model has
        # finished its output but before reaching max_new_tokens.
        eos_id = self.tokenizer.eos_token_id
        if eos_id is not None:
            stage_sampling_params["stop_token_ids"] = [eos_id]
        remaining_budget = self.max_model_len - len(agent_data.trajectory_ids)
        requested_max_tokens = min(remaining_budget, stage_max_tokens) if stage_max_tokens > 0 else remaining_budget
        if requested_max_tokens <= 0:
            raise ValueError(
                f"SVGReactAgentLoop has no remaining generation budget before stage={stage_name} stop={stop_text}. "
                f"trajectory_tokens={len(agent_data.trajectory_ids)}, max_model_len={self.max_model_len}. "
                f"Please increase prompt+response budget or reduce intermediate context."
            )
        stage_sampling_params["max_new_tokens"] = requested_max_tokens
        logger.warning(
            "[SVGReact] request=%s stage=%s stage_stop=%s prompt_tokens=%d remaining_budget=%d requested_max_tokens=%d image_inputs=%d",
            agent_data.request_id,
            stage_name,
            stop_text,
            len(agent_data.trajectory_ids),
            remaining_budget,
            requested_max_tokens,
            len(agent_data.image_data or []),
        )

        with simple_timer("generate_sequences", agent_data.metrics):
            generate_coro = self.server_manager.generate(
                request_id=agent_data.request_id,
                prompt_ids=agent_data.trajectory_ids,
                sampling_params=stage_sampling_params,
                image_data=agent_data.image_data,
            )
            if self.stage_timeout > 0:
                try:
                    output = await asyncio.wait_for(generate_coro, timeout=self.stage_timeout)
                except asyncio.TimeoutError:
                    logger.warning(
                        "[SVGReact] request=%s stage=%s TIMEOUT after %.0fs — returning empty result",
                        agent_data.request_id,
                        stage_name,
                        self.stage_timeout,
                    )
                    return StageResult(
                        text="",
                        raw_text="",
                        token_count=0,
                        raw_token_count=0,
                        was_truncated=False,
                        has_open_tag=False,
                        has_close_tag=False,
                        elapsed_sec=self.stage_timeout,
                    )
            else:
                output = await generate_coro

        token_ids = list(output.token_ids)
        log_probs = list(output.log_probs) if output.log_probs else None

        # Decode and truncate at stop_text in a thread pool to avoid blocking
        # the asyncio event loop.  With 6k tokens the binary search does ~13
        # decode calls × ~5-10 ms each ≈ 65-130 ms of CPU time.
        original_token_ids = list(token_ids)
        original_len = len(token_ids)
        text, token_ids, log_probs = await self.loop.run_in_executor(
            None, self._truncate_at_stop, token_ids, log_probs, stop_text
        )
        truncated = len(token_ids) < original_len
        if truncated:
            logger.warning(
                "[SVGReact] request=%s stage=%s truncated at stop_text=%s: %d -> %d tokens",
                agent_data.request_id,
                stage_name,
                stop_text,
                original_len,
                len(token_ids),
            )

        # Decode the raw (untruncated) text for diagnostics.
        # We already have `text` which is the truncated version.
        if truncated:
            raw_text = await self.loop.run_in_executor(
                None, self.tokenizer.decode, original_token_ids, True  # skip_special_tokens
            )
        else:
            raw_text = text

        # Detect whether the model produced the open/close wrapper tags.
        open_tag = f"<{stage_name.upper().replace('DRAFT', self.draft_tag).replace('FINAL', self.final_tag) if stage_name in ('draft', 'final') else stage_name}>"
        # Simpler: just use the stop_text to derive the close tag and the open tag.
        close_tag = stop_text  # e.g. "</SVG_DRAFT>"
        wrapper_tag = self.draft_tag if stage_name == "draft" else self.final_tag
        open_tag = f"<{wrapper_tag}>"
        has_open_tag = open_tag.lower() in raw_text.lower()
        has_close_tag = close_tag.lower() in raw_text.lower()

        # After truncation at the stop_text (e.g. </SVG_DRAFT>), the EOS token
        # (<|im_end|>, id=eos_token_id) that the model naturally produced right
        # after the closing tag is lost because _truncate_at_stop uses
        # skip_special_tokens=True.  We must re-append it so that:
        #   1. The training trajectory includes the termination signal.
        #   2. The model continues to receive GRPO reinforcement for learning
        #      when to stop generating.
        eos_token_id = self.tokenizer.eos_token_id
        if truncated and eos_token_id is not None:
            # Check if the original output had EOS right after the truncation point.
            trunc_len = len(token_ids)
            if trunc_len < len(original_token_ids) and original_token_ids[trunc_len] == eos_token_id:
                token_ids.append(eos_token_id)
                if log_probs is not None and trunc_len < len(output.log_probs or []):
                    log_probs.append(list(output.log_probs)[trunc_len])
                elif log_probs is not None:
                    log_probs.append(0.0)

        agent_data.assistant_turns += 1
        agent_data.trajectory_ids.extend(token_ids)
        agent_data.response_mask.extend([1] * len(token_ids))
        if log_probs:
            agent_data.response_logprobs.extend(log_probs)

        logger.warning(
            "[SVGReact] request=%s stage=%s stage_stop=%s generated_tokens=%d text_chars=%d",
            agent_data.request_id,
            stage_name,
            stop_text,
            len(token_ids),
            len(text),
        )
        return StageResult(
            text=text,
            raw_text=raw_text,
            token_count=len(token_ids),
            raw_token_count=original_len,
            was_truncated=truncated,
            has_open_tag=has_open_tag,
            has_close_tag=has_close_tag,
        )

    async def _append_feedback_turn(
        self,
        agent_data: SVGReactAgentData,
        tool_response: ToolResponse,
        feedback_images: list[Any],
    ) -> None:
        feedback_text = self.feedback_text
        if self.include_tool_text_in_feedback and tool_response.text:
            feedback_text = f"{feedback_text}\n\nAdditional tool feedback:\n{tool_response.text}"

        if self.force_feedback_role != "user":
            raise ValueError(f"Unsupported feedback_role={self.force_feedback_role}. Only 'user' is supported.")

        message = self._build_feedback_user_message(feedback_images, feedback_text)
        add_messages = [message]
        agent_data.messages.extend(add_messages)
        agent_data.user_turns += 1

        if feedback_images:
            if agent_data.image_data is None:
                agent_data.image_data = []
            agent_data.image_data.extend(feedback_images)

        response_ids = await self._encode_delta_messages(add_messages=add_messages, images=feedback_images)
        agent_data.trajectory_ids.extend(response_ids)
        agent_data.response_mask.extend([0] * len(response_ids))
        if agent_data.response_logprobs:
            agent_data.response_logprobs.extend([0.0] * len(response_ids))

    def _build_feedback_user_message(self, feedback_images: list[Any], feedback_text: str) -> dict[str, Any]:
        if self.processor is not None:
            content = []
            for _ in feedback_images:
                content.append({"type": "image"})
            content.append({"type": "text", "text": feedback_text})
            return {"role": "user", "content": content}
        prefix = "<image>" * max(1, len(feedback_images))
        return {"role": "user", "content": f"{prefix}{feedback_text}"}

    async def _encode_delta_messages(self, add_messages: list[dict[str, Any]], images: Optional[list[Any]]) -> list[int]:
        if self.processor is not None:
            raw_full = await self.loop.run_in_executor(
                None,
                lambda: self.processor.apply_chat_template(
                    add_messages,
                    add_generation_prompt=True,
                    tokenize=False,
                    **self.apply_chat_template_kwargs,
                ),
            )
            # apply_chat_template always prepends the default system message
            # (e.g. "<|im_start|>system\nYou are ...<|im_end|>\n").
            # Strip it so we only encode the delta (user turn + gen prompt).
            raw_delta = raw_full
            if self._processor_system_text and raw_full.startswith(self._processor_system_text):
                raw_delta = raw_full[len(self._processor_system_text):]
            model_inputs = self.processor(text=[raw_delta], images=images or None, return_tensors="pt")
            return model_inputs.pop("input_ids").squeeze(0).tolist()

        response_ids = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.apply_chat_template(
                add_messages,
                add_generation_prompt=True,
                tokenize=True,
                **self.apply_chat_template_kwargs,
            ),
        )
        return response_ids[len(self.system_prompt) :]

    async def _build_prompt_ids(self, messages: list[dict[str, Any]], images: Optional[list[Any]]) -> list[int]:
        if self.processor is not None:
            raw_prompt = await self.loop.run_in_executor(
                None,
                lambda: self.processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False,
                    **self.apply_chat_template_kwargs,
                ),
            )
            model_inputs = self.processor(text=[raw_prompt], images=images, return_tensors="pt")
            return model_inputs.pop("input_ids").squeeze(0).tolist()

        return await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                **self.apply_chat_template_kwargs,
            ),
        )

    async def _call_tool(
        self, agent_data: SVGReactAgentData, svg_payload: str
    ) -> tuple[ToolResponse, Optional[float], dict[str, Any]]:
        tool = self.tools[self.tool_name]
        instance_id = None

        target_images = self._ensure_list(agent_data.original_image_data)
        target_image = target_images[0] if target_images else None
        tool_kwargs = agent_data.tools_kwargs.get(self.tool_name, {})

        try:
            instance_id, _ = await tool.create(create_kwargs=tool_kwargs.get("create_kwargs", {}))
            tool_response, tool_reward, tool_metrics = await tool.execute(
                instance_id,
                {self.tool_argument_name: svg_payload},
                target_image=target_image,
                target_images=target_images,
                current_images=self._ensure_list(agent_data.image_data),
                messages=copy.deepcopy(agent_data.messages),
                draft_svg=svg_payload,
                **tool_kwargs.get("execute_kwargs", {}),
            )
        finally:
            if instance_id is not None:
                await tool.release(instance_id, **tool_kwargs.get("release_kwargs", {}))

        if tool_response.text:
            tool_response.text = self._truncate_text(tool_response.text)
        return tool_response, tool_reward, tool_metrics

    def _truncate_text(self, text: str) -> str:
        if len(text) <= self.max_tool_response_length:
            return text
        if self.tool_response_truncate_side == "left":
            return text[: self.max_tool_response_length] + "...(truncated)"
        if self.tool_response_truncate_side == "right":
            return "(truncated)..." + text[-self.max_tool_response_length :]
        length = self.max_tool_response_length // 2
        return text[:length] + "...(truncated)..." + text[-length:]

    @staticmethod
    def _ensure_list(value: Optional[Any]) -> list[Any]:
        if value is None:
            return []
        if isinstance(value, list):
            return value
        return [value]

    def _truncate_at_stop(
        self, token_ids: list[int], log_probs: list[float] | None, stop_text: str
    ) -> tuple[str, list[int], list[float] | None]:
        """Decode token_ids and truncate at the first occurrence of stop_text.

        Uses binary search over token prefixes: O(log N) decode calls.
        Designed to run in a thread pool (CPU-bound, but tokenizer releases the GIL).
        """
        text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
        if stop_text not in text:
            return text, token_ids, log_probs
        lo, hi = 1, len(token_ids)
        while lo < hi:
            mid = (lo + hi) // 2
            partial = self.tokenizer.decode(token_ids[:mid], skip_special_tokens=True)
            if stop_text in partial:
                hi = mid
            else:
                lo = mid + 1
        token_ids = token_ids[:lo]
        if log_probs is not None:
            log_probs = log_probs[:lo]
        text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
        return text, token_ids, log_probs

    @staticmethod
    def _extract_svg_payload(text: str, wrapper_tag: str) -> Optional[str]:
        wrapper_pattern = re.compile(
            rf"<{wrapper_tag}>\s*(.*?)\s*</{wrapper_tag}>",
            flags=re.DOTALL | re.IGNORECASE,
        )
        match = wrapper_pattern.search(text)
        if match:
            return match.group(1).strip()

        svg_pattern = re.compile(r"(<svg[\s\S]*?</svg>)", flags=re.DOTALL | re.IGNORECASE)
        svg_match = svg_pattern.search(text)
        if svg_match:
            return svg_match.group(1).strip()

        open_tag = f"<{wrapper_tag}>"
        if open_tag.lower() in text.lower():
            start_idx = text.lower().find(open_tag.lower()) + len(open_tag)
            remainder = text[start_idx:].strip()
            if remainder:
                return remainder
        return None
