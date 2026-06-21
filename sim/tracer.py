"""
CaptureTracer — an AgentCallbackHandler that records a full per-turn execution
trace from the framework's own observability callbacks.

The framework fires, in order, within one `process_turn`:
    on_turn_start
    on_agent_skipped*          (Phase-1 eligibility rejections, with reason)
    on_phase_start(1, names)
      on_agent_start / on_agent_finish / on_agent_error   (parallel, Phase 1)
    on_phase_end(1, event_names)
    [ on_phase_start(2, names) ... on_phase_end(2, event_names) ]   (if events)
    on_turn_end(response, duration)

Because agents within a phase run in parallel, `on_agent_finish` can interleave;
we attribute each finish to the phase most recently announced by
`on_phase_start`, which is safe (the engine awaits all of a phase's agents before
starting the next phase).
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

# Imported lazily-safe: the framework is guaranteed importable before this runs.
from xubb_agents.core.callbacks import AgentCallbackHandler


def _serialize_insight(ins: Any) -> Dict[str, Any]:
    return {
        "agent_id": getattr(ins, "agent_id", None),
        "agent_name": getattr(ins, "agent_name", None),
        "type": getattr(getattr(ins, "type", None), "value", str(getattr(ins, "type", ""))),
        "content": getattr(ins, "content", ""),
        "confidence": getattr(ins, "confidence", None),
        "expiry": getattr(ins, "expiry", None),
        "action_label": getattr(ins, "action_label", None),
        "metadata": getattr(ins, "metadata", {}) or {},
    }


def _serialize_fact(f: Any) -> Dict[str, Any]:
    return {
        "type": getattr(f, "type", None),
        "key": getattr(f, "key", None),
        "value": getattr(f, "value", None),
        "confidence": getattr(f, "confidence", None),
        "priority": getattr(f, "priority", None),
        "source_agent": getattr(f, "source_agent", None),
    }


class CaptureTracer(AgentCallbackHandler):
    """Accumulates the current turn's trace; `snapshot()` then `reset()` per turn."""

    def __init__(self):
        # Binding persists across per-turn resets (set once per session by the driver).
        self._sim_ctx = None       # mock_llm.SimContext, frozen per phase for fidelity
        self._live_bb = None       # the live Blackboard the engine mutates
        self.reset()

    def bind(self, sim_ctx, live_bb) -> None:
        """Wire the mock's SimContext + the live Blackboard for snapshot fidelity.

        The real engine runs each phase against `blackboard.snapshot()` (a deep
        copy). To mirror that exactly, we re-point the mock's view at a fresh
        snapshot on every `on_phase_start`, so mock predicates (has_event /
        var_equals / has_fact / queue_not_empty / $inc) read the SAME immutable
        per-phase state a real DynamicAgent would see — not the live board.
        """
        self._sim_ctx = sim_ctx
        self._live_bb = live_bb

    def reset(self) -> None:
        self._skipped: List[Dict[str, str]] = []
        self._phases: Dict[int, Dict[str, Any]] = {}
        self._phase_order: List[int] = []
        self._current_phase: int = 1
        self._errors: Dict[str, str] = {}
        self._turn_real_latency_ms: Optional[float] = None
        self._final_insight_count: Optional[int] = None
        self._agent_starts: Dict[str, float] = {}   # perf_counter at on_agent_start

    # ----------------------------------------------------------- turn level
    async def on_turn_start(self, context) -> None:
        self.reset()

    async def on_turn_end(self, response, duration: float) -> None:
        self._turn_real_latency_ms = duration * 1000.0
        try:
            self._final_insight_count = len(response.insights)
        except Exception:
            self._final_insight_count = None

    async def on_chain_error(self, error: Exception) -> None:
        self._errors["__chain__"] = str(error)

    # ---------------------------------------------------------- phase level
    async def on_phase_start(self, phase: int, agent_names: List[str]) -> None:
        self._current_phase = phase
        # Fidelity: freeze the mock's blackboard view to a per-phase snapshot,
        # exactly as the engine does for the real agents (engine._run_phase).
        if self._sim_ctx is not None and self._live_bb is not None:
            try:
                self._sim_ctx.blackboard = self._live_bb.snapshot()
            except Exception:
                self._sim_ctx.blackboard = self._live_bb
        if phase not in self._phases:
            self._phases[phase] = {
                "phase": phase,
                "ran": list(agent_names),
                "agents": [],
                "events_emitted": [],
            }
            self._phase_order.append(phase)
        else:
            self._phases[phase]["ran"] = list(agent_names)

    async def on_phase_end(self, phase: int, event_names: List[str]) -> None:
        if phase in self._phases:
            self._phases[phase]["events_emitted"] = list(event_names)

    # ---------------------------------------------------------- agent level
    async def on_agent_skipped(self, agent_name: str, reason: str) -> None:
        self._skipped.append({"agent": agent_name, "reason": reason})

    async def on_agent_start(self, agent_name: str, context) -> None:
        # Measure true wall time ourselves: the framework's own duration is
        # computed with `time.time()`, which inside agent.py is the (frozen)
        # virtual clock — so its per-agent duration is always ~0. perf_counter
        # is never patched, so this is the real per-agent latency (meaningful in
        # real-LLM mode; ~0 in mock mode, where the UI uses modeled latency).
        self._agent_starts[agent_name] = time.perf_counter()

    async def on_agent_error(self, agent_name: str, error: Exception) -> None:
        self._errors[agent_name] = str(error)

    async def on_agent_finish(self, agent_name: str, response, duration: float) -> None:
        phase = self._current_phase
        bucket = self._phases.setdefault(
            phase,
            {"phase": phase, "ran": [], "agents": [], "events_emitted": []},
        )
        if phase not in self._phase_order:
            self._phase_order.append(phase)

        start = self._agent_starts.get(agent_name)
        wall_ms = (time.perf_counter() - start) * 1000.0 if start is not None else None
        rec: Dict[str, Any] = {
            "agent": agent_name,
            "real_latency_ms": round(wall_ms, 2) if wall_ms is not None else None,
        }
        if response is None:
            rec["status"] = "no_response"
            rec["spoke"] = False
        else:
            insights = [_serialize_insight(i) for i in getattr(response, "insights", [])]
            is_error = any(i["type"] == "error" for i in insights)
            rec["status"] = "error" if is_error else "success"
            rec["spoke"] = bool(insights)
            rec["insights"] = insights
            rec["events"] = [getattr(e, "name", "") for e in getattr(response, "events", [])]
            rec["facts"] = [_serialize_fact(f) for f in getattr(response, "facts", [])]
            rec["variable_updates"] = dict(getattr(response, "variable_updates", {}) or {})
            rec["queue_pushes"] = {
                k: len(v) for k, v in (getattr(response, "queue_pushes", {}) or {}).items()
            }
            rec["memory_updates_keys"] = list((getattr(response, "memory_updates", {}) or {}).keys())
            # FULL TRANSPARENCY: the exact messages the framework sent to the LLM
            # (system prompt = the agent's text + everything the host/framework wraps
            # around it; user = the transcript) plus the raw LLM output. So the user
            # sees the COMPLETE prompt, with nothing hidden behind the scenes.
            dbg = getattr(response, "debug_info", None) or {}
            if dbg.get("prompt_messages"):
                rec["prompt"] = dbg["prompt_messages"]
                rec["model"] = dbg.get("model")
                rec["llm_output"] = dbg.get("llm_output")
        bucket["agents"].append(rec)

    # -------------------------------------------------------------- export
    def snapshot(self) -> Dict[str, Any]:
        # Attach any captured exceptions onto the matching agent records.
        for phase in self._phases.values():
            for rec in phase["agents"]:
                if rec["agent"] in self._errors:
                    rec["status"] = "error"
                    rec["error"] = self._errors[rec["agent"]]
        ordered = [self._phases[p] for p in self._phase_order if p in self._phases]
        return {
            "skipped": list(self._skipped),
            "phases": ordered,
            "errors": dict(self._errors),
            "real_latency_ms": self._turn_real_latency_ms,
            "final_insight_count": self._final_insight_count,
        }
