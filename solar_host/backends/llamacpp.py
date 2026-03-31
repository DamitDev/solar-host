"""LlamaCpp backend runner implementation."""

import re
from typing import List, Optional, Any, Dict
from datetime import datetime, timezone

from solar_host.backends.base import BackendRunner, RuntimeStateUpdate
from solar_host.models.base import InstancePhase, GenerationMetrics
from solar_host.config import settings


class LlamaCppRunner(BackendRunner):
    """Backend runner for llama.cpp server instances."""

    def __init__(self):
        # Compile regex patterns for parsing llama-server logs
        self._re_launch = re.compile(
            r"slot\s+launch_slot_:\s+id\s+(\d+)\s*\|\s*task\s+(-?\d+)\s*\|\s*processing task"
        )
        self._re_progress = re.compile(
            r"prompt processing progress.*progress\s*=\s*([0-9.]+)"
        )
        self._re_prompt_done = re.compile(r"\|\s*prompt done\b")
        self._re_release = re.compile(
            r"slot\s+release:\s+id\s+(\d+)\s*\|\s*task\s+(-?\d+)\s*\|\s*stop processing"
        )
        self._re_all_idle = re.compile(r"srv\s+update_slots:\s+all slots are idle")
        self._re_new_prompt = re.compile(
            r"slot\s+update_slots:\s+id\s+(\d+)\s*\|\s*task\s+(-?\d+)\s*\|\s*new prompt.*task\.n_tokens\s*=\s*(\d+)"
        )
        self._re_checkpoint = re.compile(
            r"created context checkpoint\s+(\d+)\s+of\s+(\d+)"
        )
        self._re_print_timing = re.compile(
            r"slot\s+print_timing:\s+id\s+(\d+)\s*\|\s*task\s+(-?\d+)\s*\|"
        )
        self._re_prompt_eval_line = re.compile(
            r"prompt eval time\s*=\s*([0-9.]+)\s*ms\s*/\s*(\d+)\s*tokens\s*\(\s*([0-9.]+)\s*ms per token,\s*([0-9.]+)\s*tokens per second\)"
        )
        self._re_decode_eval_line = re.compile(
            r"\s*eval time\s*=\s*([0-9.]+)\s*ms\s*/\s*(\d+)\s*tokens\s*\(\s*([0-9.]+)\s*ms per token,\s*([0-9.]+)\s*tokens per second\)"
        )

    def get_backend_type(self) -> str:
        return "llamacpp"

    def build_command(self, instance: Any) -> List[str]:
        """Build llama-server command from instance config."""
        config = instance.config
        cmd = [
            "llama-server",
            "--model",
            config.model,
            "--alias",
            config.alias,
            "--threads",
            str(config.threads),
            "--n_gpu_layers",
            str(config.n_gpu_layers),
            "--temp",
            str(config.temp),
            "--top_p",
            str(config.top_p),
            "--top_k",
            str(config.top_k),
            "--min_p",
            str(config.min_p),
            "--ctx-size",
            str(config.ctx_size),
            "--host",
            config.host,
            "--port",
            str(instance.port),
            "--api-key",
            settings.api_key,
            "--no-warmup",
        ]

        if config.chat_template_file:
            cmd.extend(["--jinja", "--chat-template-file", config.chat_template_file])
        else:
            cmd.extend(["--jinja"])

        chat_template_kwargs = getattr(config, "chat_template_kwargs", None)
        if chat_template_kwargs and chat_template_kwargs.strip():
            cmd.extend(["--chat-template-kwargs", chat_template_kwargs.strip()])

        reasoning_budget = getattr(config, "reasoning_budget", None)
        if reasoning_budget is not None:
            cmd.extend(["--reasoning-budget", str(int(reasoning_budget))])

        cache_type_k = getattr(config, "cache_type_k", None)
        if cache_type_k and cache_type_k.strip():
            cmd.extend(["-ctk", cache_type_k.strip()])

        cache_type_v = getattr(config, "cache_type_v", None)
        if cache_type_v and cache_type_v.strip():
            cmd.extend(["-ctv", cache_type_v.strip()])

        rope_scaling = getattr(config, "rope_scaling", None)
        if rope_scaling and rope_scaling.strip():
            cmd.extend(["--rope-scaling", rope_scaling.strip()])

        rope_scale = getattr(config, "rope_scale", None)
        if rope_scale is not None:
            cmd.extend(["--rope-scale", str(rope_scale)])

        yarn_orig_ctx = getattr(config, "yarn_orig_ctx", None)
        if yarn_orig_ctx is not None:
            cmd.extend(["--yarn-orig-ctx", str(int(yarn_orig_ctx))])

        if getattr(config, "special", False):
            cmd.append("--special")

        ot_value = getattr(config, "ot", None)
        if ot_value and ot_value.strip():
            cmd.extend(["-ot", ot_value])

        # Model type flags
        model_type = getattr(config, "model_type", "llm")
        if model_type == "embedding":
            cmd.append("--embedding")
        elif model_type == "reranker":
            cmd.append("--rerank")

        # Pooling flag (only valid for embedding models)
        if model_type == "embedding":
            pooling = getattr(config, "pooling", None)
            if pooling and pooling.strip():
                cmd.extend(["--pooling", pooling])

        return cmd

    def get_health_endpoint(self) -> str:
        return "/health"

    def get_supported_endpoints(self) -> List[str]:
        return [
            "/v1/chat/completions",
            "/v1/completions",
            "/v1/models",
            "/v1/embeddings",
            "/v1/rerank",
        ]

    def get_supported_endpoints_for_model_type(self, model_type: str) -> List[str]:
        """Return endpoints based on llama.cpp model_type (llm, embedding, reranker)."""
        if model_type == "embedding":
            return ["/v1/embeddings", "/v1/models"]
        elif model_type == "reranker":
            return ["/v1/rerank", "/v1/models"]
        return ["/v1/chat/completions", "/v1/completions", "/v1/models"]

    def initialize_context(self) -> Dict[str, Any]:
        """Initialize parsing context for llama.cpp log parsing."""
        return {
            "active_slots": set(),
            "pending_generations_by_slot": {},
            "recent_generations": [],
            "last_state": {
                "busy": False,
                "phase": InstancePhase.IDLE.value,
                "prefill_progress": None,
                "active_slots": 0,
                "slot_id": None,
                "task_id": None,
            },
        }

    def parse_log_line(
        self, instance_id: str, line: str, context: Dict[str, Any]
    ) -> Optional[RuntimeStateUpdate]:
        """Parse a llama-server log line and return state update if changed."""
        slots = context.get("active_slots", set())
        last_state = context.get("last_state", {})
        pending_by_slot = context.get("pending_generations_by_slot", {})

        # slot launch → add slot, busy true
        m = self._re_launch.search(line)
        if m:
            try:
                slot_id = int(m.group(1))
            except Exception:
                slot_id = -1
            slots.add(slot_id)
            context["active_slots"] = slots
            return self._create_update(
                busy=True,
                phase=InstancePhase.PREFILL,
                prefill_progress=last_state.get("prefill_progress"),
                active_slots=len(slots),
                slot_id=slot_id,
                task_id=last_state.get("task_id"),
                last_state=last_state,
                context=context,
            )

        # new prompt → phase becomes prefill; capture task_id and prompt tokens
        m = self._re_new_prompt.search(line)
        if m:
            try:
                slot_id = int(m.group(1))
                task_id = int(m.group(2))
                prompt_tokens = int(m.group(3))
            except Exception:
                slot_id, task_id, prompt_tokens = -1, -1, None
            slots.add(slot_id)
            context["active_slots"] = slots

            # Initialize pending generation metrics for this slot
            pending = pending_by_slot.get(slot_id) or {}
            pending.update(
                {
                    "slot_id": slot_id,
                    "task_id": task_id,
                    "prompt_tokens": prompt_tokens,
                    "started_at": datetime.now(timezone.utc).isoformat(),
                }
            )
            pending_by_slot[slot_id] = pending
            context["pending_generations_by_slot"] = pending_by_slot

            return self._create_update(
                busy=True,
                phase=InstancePhase.PREFILL,
                prefill_progress=0.0,
                active_slots=len(slots),
                slot_id=slot_id,
                task_id=task_id,
                prefill_prompt_tokens=prompt_tokens,
                last_state=last_state,
                context=context,
            )

        # prompt processing progress → update progress
        m = self._re_progress.search(line)
        if m:
            try:
                progress = float(m.group(1))
            except Exception:
                progress = None
            return self._create_update(
                busy=True if len(slots) > 0 else last_state.get("busy", False),
                phase=InstancePhase.PREFILL,
                prefill_progress=progress,
                active_slots=len(slots),
                slot_id=last_state.get("slot_id"),
                task_id=last_state.get("task_id"),
                last_state=last_state,
                context=context,
            )

        # prompt done → set progress to 1.0
        if self._re_prompt_done.search(line):
            return self._create_update(
                busy=True if len(slots) > 0 else last_state.get("busy", False),
                phase=(
                    InstancePhase.GENERATING if len(slots) > 0 else InstancePhase.IDLE
                ),
                prefill_progress=1.0,
                active_slots=len(slots),
                slot_id=last_state.get("slot_id"),
                task_id=last_state.get("task_id"),
                last_state=last_state,
                context=context,
            )

        # context checkpoint progress (still prefill phase)
        m = self._re_checkpoint.search(line)
        if m:
            try:
                idx = int(m.group(1))
                total = int(m.group(2))
            except Exception:
                idx, total = None, None
            return self._create_update(
                busy=True if len(slots) > 0 else last_state.get("busy", False),
                phase=InstancePhase.PREFILL,
                prefill_progress=last_state.get("prefill_progress"),
                active_slots=len(slots),
                slot_id=last_state.get("slot_id"),
                task_id=last_state.get("task_id"),
                checkpoint_index=idx,
                checkpoint_total=total,
                last_state=last_state,
                context=context,
            )

        # decode timing metrics after generation finishes
        if self._re_print_timing.search(line):
            # Subsequent lines include timing metrics; rely on later matches
            return None

        m = self._re_decode_eval_line.search(line)
        if m:
            try:
                gen_tokens = int(m.group(2))
                ms_per_tok = float(m.group(3))
                tps = float(m.group(4))
            except Exception:
                gen_tokens, ms_per_tok, tps = None, None, None

            # Update pending metrics for last active slot
            last_slot_id = last_state.get("slot_id")
            if isinstance(last_slot_id, int):
                pending: dict[str, Any] = pending_by_slot.get(last_slot_id) or {
                    "slot_id": last_slot_id
                }
                if gen_tokens is not None:
                    pending["generated_tokens"] = gen_tokens
                if tps is not None:
                    pending["decode_tps"] = tps
                if ms_per_tok is not None:
                    pending["decode_ms_per_token"] = ms_per_tok
                pending_by_slot[last_slot_id] = pending
                context["pending_generations_by_slot"] = pending_by_slot

            phase_str = last_state.get("phase", InstancePhase.IDLE.value)
            phase = (
                InstancePhase(phase_str)
                if phase_str in [p.value for p in InstancePhase]
                else InstancePhase.IDLE
            )
            if len(slots) > 0:
                phase = InstancePhase.GENERATING

            return self._create_update(
                busy=True if len(slots) > 0 else last_state.get("busy", False),
                phase=phase,
                prefill_progress=last_state.get("prefill_progress"),
                active_slots=len(slots),
                slot_id=last_state.get("slot_id"),
                task_id=last_state.get("task_id"),
                generated_tokens=gen_tokens,
                decode_tps=tps,
                decode_ms_per_token=ms_per_tok,
                last_state=last_state,
                context=context,
            )

        # slot release → remove slot; if none remain, clear busy and progress
        m = self._re_release.search(line)
        if m:
            try:
                slot_id = int(m.group(1))
            except Exception:
                slot_id = -1
            if slot_id in slots:
                slots.discard(slot_id)
            context["active_slots"] = slots

            # Finalize any pending generation for this slot
            pending = pending_by_slot.pop(slot_id, None)
            if pending is not None:
                metrics = GenerationMetrics(
                    instance_id=instance_id,
                    slot_id=pending.get("slot_id"),
                    task_id=pending.get("task_id"),
                    prompt_tokens=pending.get("prompt_tokens"),
                    generated_tokens=pending.get("generated_tokens"),
                    decode_tps=pending.get("decode_tps"),
                    decode_ms_per_token=pending.get("decode_ms_per_token"),
                    started_at=pending.get("started_at"),
                    finished_at=datetime.now(timezone.utc).isoformat(),
                )
                recent = context.get("recent_generations", [])
                recent.append(metrics)
                # Keep only last 100 generations
                if len(recent) > 100:
                    recent = recent[-100:]
                context["recent_generations"] = recent
            context["pending_generations_by_slot"] = pending_by_slot

            if len(slots) == 0:
                return self._create_update(
                    busy=False,
                    phase=InstancePhase.IDLE,
                    prefill_progress=None,
                    active_slots=0,
                    slot_id=None,
                    task_id=None,
                    checkpoint_index=None,
                    checkpoint_total=None,
                    last_state=last_state,
                    context=context,
                )
            else:
                phase_str = last_state.get("phase", InstancePhase.GENERATING.value)
                phase = (
                    InstancePhase(phase_str)
                    if phase_str in [p.value for p in InstancePhase]
                    else InstancePhase.GENERATING
                )
                return self._create_update(
                    busy=True,
                    phase=phase,
                    prefill_progress=last_state.get("prefill_progress"),
                    active_slots=len(slots),
                    slot_id=last_state.get("slot_id"),
                    task_id=last_state.get("task_id"),
                    last_state=last_state,
                    context=context,
                )

        # all slots idle → force clear
        if self._re_all_idle.search(line):
            slots.clear()
            context["active_slots"] = slots
            return self._create_update(
                busy=False,
                phase=InstancePhase.IDLE,
                prefill_progress=None,
                active_slots=0,
                slot_id=None,
                task_id=None,
                checkpoint_index=None,
                checkpoint_total=None,
                last_state=last_state,
                context=context,
            )

        return None

    def _create_update(
        self,
        busy: bool,
        phase: InstancePhase,
        prefill_progress: Optional[float],
        active_slots: int,
        last_state: Dict[str, Any],
        context: Dict[str, Any],
        slot_id: Optional[int] = None,
        task_id: Optional[int] = None,
        prefill_prompt_tokens: Optional[int] = None,
        generated_tokens: Optional[int] = None,
        decode_tps: Optional[float] = None,
        decode_ms_per_token: Optional[float] = None,
        checkpoint_index: Optional[int] = None,
        checkpoint_total: Optional[int] = None,
    ) -> Optional[RuntimeStateUpdate]:
        """Create a RuntimeStateUpdate if state has changed."""
        # Normalize prefill_progress
        pp: Optional[float] = None
        if prefill_progress is not None:
            try:
                pp = float(prefill_progress)
            except Exception:
                pp = None

        # Check if state changed
        changed = (
            last_state.get("busy") != busy
            or last_state.get("phase") != phase.value
            or last_state.get("prefill_progress") != pp
            or last_state.get("active_slots") != active_slots
            or last_state.get("slot_id") != slot_id
            or last_state.get("task_id") != task_id
            or last_state.get("prefill_prompt_tokens") != prefill_prompt_tokens
            or last_state.get("generated_tokens") != generated_tokens
            or last_state.get("decode_tps") != decode_tps
            or last_state.get("decode_ms_per_token") != decode_ms_per_token
            or last_state.get("checkpoint_index") != checkpoint_index
            or last_state.get("checkpoint_total") != checkpoint_total
        )

        if not changed:
            return None

        # Update last_state in context
        context["last_state"] = {
            "busy": busy,
            "phase": phase.value,
            "prefill_progress": pp,
            "active_slots": active_slots,
            "slot_id": slot_id,
            "task_id": task_id,
            "prefill_prompt_tokens": prefill_prompt_tokens,
            "generated_tokens": generated_tokens,
            "decode_tps": decode_tps,
            "decode_ms_per_token": decode_ms_per_token,
            "checkpoint_index": checkpoint_index,
            "checkpoint_total": checkpoint_total,
        }

        return RuntimeStateUpdate(
            busy=busy,
            phase=phase,
            prefill_progress=pp,
            active_slots=active_slots,
            slot_id=slot_id,
            task_id=task_id,
            prefill_prompt_tokens=prefill_prompt_tokens,
            generated_tokens=generated_tokens,
            decode_tps=decode_tps,
            decode_ms_per_token=decode_ms_per_token,
            checkpoint_index=checkpoint_index,
            checkpoint_total=checkpoint_total,
        )

    def get_last_generation(
        self, context: Dict[str, Any]
    ) -> Optional[GenerationMetrics]:
        """Get the last generation metrics from context."""
        recent = context.get("recent_generations", [])
        if not recent:
            return None
        return recent[-1]
