"""
SimulationSession — replays a scripted conversation through a real AgentEngine.

This module IS a reference host integration. It does exactly what a production
host (xubb_server, a CLI, a desktop app) must do:

  * build an AgentEngine and register DynamicAgents from config,
  * maintain a Blackboard and a sliding transcript window for the session,
  * call `engine.process_turn(context, trigger_type=...)` once per conversational
    event, choosing the trigger type (turn / keyword / silence / interval),
  * read insights off the aggregated response and render them.

On top of that it captures a rich per-turn trace (via CaptureTracer + blackboard
deltas + modeled latency) so the web UI can show *why* each whisper did or didn't
fire.
"""

from __future__ import annotations

import asyncio
import copy
from typing import Any, Dict, List, Optional

from .clock import ActiveClock, VirtualClock, install_clock_shim, uninstall_clock_shim
from .mock_llm import MockLLMClient, SimContext
from .tracer import CaptureTracer

# Framework imports (resolved by sim.framework before this module is used).
from xubb_agents import (
    AgentEngine,
    AgentContext,
    Blackboard,
    DynamicAgent,
    TranscriptSegment,
    TriggerType,
)

_TRIGGER_MAP = {
    "turn_based": TriggerType.TURN_BASED,
    "keyword": TriggerType.KEYWORD,
    "silence": TriggerType.SILENCE,
    "interval": TriggerType.INTERVAL,
    "event": TriggerType.EVENT,
    "force": TriggerType.FORCE,
}


class SimulationSession:
    def __init__(
        self,
        suite: Dict[str, Any],
        scenario: Dict[str, Any],
        mode: str = "mock",
        api_key: Optional[str] = None,
    ):
        self.suite = suite
        self.scenario = scenario
        self.mode = mode
        self.api_key = api_key

        self.window: int = int(scenario.get("window", 12))
        self.user_context: Optional[str] = scenario.get("user_context")
        self.language_directive: Optional[str] = scenario.get("language_directive")
        self.session_id: str = scenario.get("session_id", "sim_session")

        self.steps: List[Dict[str, Any]] = scenario.get("steps", [])

        # Modeled latency per agent NAME (the tracer reports by name).
        self._latency_by_name: Dict[str, float] = {}

        self._lock = asyncio.Lock()
        self._clock = VirtualClock()
        self._closed = False
        self._sim_ctx = SimContext()
        self._tracer = CaptureTracer()

        self._build_engine()
        self.reset()

    # ------------------------------------------------------------ engine build
    def _build_engine(self) -> None:
        # Process-wide, reference-counted shim install (concurrency-safe across
        # sessions). The active clock is selected per turn via `ActiveClock`.
        install_clock_shim()
        key = self.api_key if self.mode == "real" else None
        self.engine = AgentEngine(api_key=key, callbacks=[self._tracer])

        self._agent_meta: List[Dict[str, Any]] = []
        for raw in self.suite.get("agents", []):
            config = dict(raw)
            mock_spec = config.pop("_mock", {}) or {}
            doc = config.pop("_doc", "") or ""
            agent = DynamicAgent(config)
            self.engine.register_agent(agent)

            name = agent.config.name
            latency = float(mock_spec.get("latency_ms", 0.0))
            self._latency_by_name[name] = latency
            if self.mode == "mock":
                agent.llm = MockLLMClient(
                    agent_id=agent.config.id,
                    rules=mock_spec.get("rules", []),
                    sim_ctx=self._sim_ctx,
                    latency_ms=latency,
                )
            self._agent_meta.append(
                {
                    "id": agent.config.id,
                    "name": name,
                    "priority": agent.config.priority,
                    "cooldown": agent.config.cooldown,
                    "triggers": [t.value for t in agent.config.trigger_types],
                    "keywords": list(agent.config.trigger_keywords),
                    "subscribed_events": list(getattr(agent.config, "subscribed_events", []) or []),
                    "output_format": agent.config.output_format,
                    "latency_ms": latency,
                    "doc": doc,
                    # The portable prompt the user ports to the real app + its wiring.
                    "text": config.get("text") or "",
                    "trigger_conditions": config.get("trigger_conditions"),
                }
            )

    # ------------------------------------------------------------------- reset
    def reset(self) -> None:
        self.blackboard = Blackboard()
        self.segments: List[TranscriptSegment] = []
        self.turn_count = 0
        self.cursor = 0  # index into self.steps
        self.history: List[Dict[str, Any]] = []
        # agent NAME -> {"turn": int, "spoke": bool}; powers the cooldown causal chain.
        self._last_fire: Dict[str, Dict[str, Any]] = {}

        self._sim_ctx.blackboard = self.blackboard
        self._sim_ctx.turn_count = 0
        self._sim_ctx.segments = self.segments

        # Reset per-agent runtime state so cooldowns/memory start fresh, and make
        # the FIRST turn always eligible despite the virtual clock starting near 0.
        for agent in self.engine.agents:
            agent.last_run_time = float("-inf")
            agent.private_state = {}

    # ------------------------------------------------------------------- meta
    def meta(self) -> Dict[str, Any]:
        return {
            "scenario": {
                "name": self.scenario.get("name"),
                "description": self.scenario.get("description"),
                "window": self.window,
                "user_context": self.user_context,
                "language_directive": self.language_directive,
                "user_speaker": self.scenario.get("user_speaker"),
                "total_steps": len(self.steps),
            },
            "suite": {
                "name": self.suite.get("name"),
                "description": self.suite.get("description"),
            },
            "agents": self._agent_meta,
            "mode": self.mode,
        }

    @property
    def finished(self) -> bool:
        return self.cursor >= len(self.steps)

    # ------------------------------------------------------------------- step
    async def step(self) -> Optional[Dict[str, Any]]:
        """Advance one scenario step; returns its turn trace (or None if done)."""
        async with self._lock:
            if self.finished:
                return None
            step = self.steps[self.cursor]
            self.cursor += 1
            return await self._run_step(step)

    async def run_all(self) -> List[Dict[str, Any]]:
        traces: List[Dict[str, Any]] = []
        while not self.finished:
            t = await self.step()
            if t is None:
                break
            traces.append(t)
        return traces

    async def _run_step(self, step: Dict[str, Any]) -> Dict[str, Any]:
        timestamp = float(step.get("timestamp", self.turn_count))
        self._clock.set(timestamp)

        has_text = bool(step.get("text"))
        if has_text:
            seg = TranscriptSegment(
                speaker=step.get("speaker", "SPEAKER"),
                text=step["text"],
                timestamp=timestamp,
                is_final=step.get("is_final", True),
            )
            self.segments.append(seg)

        self.turn_count += 1
        self._sim_ctx.turn_count = self.turn_count
        self._sim_ctx.segments = self.segments

        trigger_name = step.get("trigger", "turn_based")
        trigger_type = _TRIGGER_MAP.get(trigger_name, TriggerType.TURN_BASED)
        self._sim_ctx.trigger_type = trigger_name

        trigger_metadata = dict(step.get("trigger_metadata", {}) or {})
        allowed_agent_ids: Optional[List[str]] = None

        # Keyword trigger: emulate the host's keyword-detection responsibility.
        if trigger_type == TriggerType.KEYWORD and has_text:
            matches = self.engine.check_keyword_triggers(step["text"])
            allowed_agent_ids = [a.config.id for a, _ in matches]
            if matches and "keyword" not in trigger_metadata:
                trigger_metadata["keyword"] = matches[0][1]
        elif trigger_type == TriggerType.SILENCE and "silence_duration" not in trigger_metadata:
            trigger_metadata["silence_duration"] = float(step.get("silence_duration", 0.0))

        context = AgentContext(
            session_id=self.session_id,
            recent_segments=self.segments[-100:],
            blackboard=self.blackboard,
            turn_count=self.turn_count,
            user_context=self.user_context,
            language_directive=self.language_directive,
        )

        bb_before = self.blackboard.to_dict()
        self._tracer.reset()
        # Bind the tracer so it can freeze a per-phase snapshot for mock fidelity.
        self._tracer.bind(self._sim_ctx, self.blackboard)

        # Activate THIS session's virtual clock for the duration of the turn so
        # cooldown math (and the gather-spawned agent tasks) read scenario time.
        with ActiveClock(self._clock):
            response = await self.engine.process_turn(
                context,
                allowed_agent_ids=allowed_agent_ids,
                trigger_type=trigger_type,
                trigger_metadata=trigger_metadata,
            )

        # The mock's view may hold the last phase's snapshot; restore the live board.
        self._sim_ctx.blackboard = self.blackboard

        trace = self._tracer.snapshot()
        bb_after = self.blackboard.to_dict()

        turn_record = self._assemble(
            step=step,
            timestamp=timestamp,
            trigger_name=trigger_name,
            trigger_metadata=trigger_metadata,
            allowed_agent_ids=allowed_agent_ids,
            has_text=has_text,
            trace=trace,
            response=response,
            bb_before=bb_before,
            bb_after=bb_after,
        )
        # Update last-fire AFTER assembling (gated rows referenced the prior state).
        for phase in trace["phases"]:
            for rec in phase["agents"]:
                self._last_fire[rec["agent"]] = {
                    "turn": self.turn_count,
                    "spoke": bool(rec.get("spoke")),
                }
        self.history.append(turn_record)
        return turn_record

    # --------------------------------------------------------------- assembly
    def _assemble(self, **kw) -> Dict[str, Any]:
        step = kw["step"]
        trace = kw["trace"]
        response = kw["response"]
        bb_before = kw["bb_before"]
        bb_after = kw["bb_after"]

        # Modeled latency = sum over phases of the slowest agent that ran in it
        # (phases are sequential; agents within a phase are parallel). This makes
        # the latency-vs-coordination-depth tradeoff visible.
        modeled_total = 0.0
        phase_latencies = []
        gated: List[Dict[str, str]] = []
        for phase in trace["phases"]:
            finished_names = [a["agent"] for a in phase["agents"]]
            # Cooldown gating is invisible to callbacks: the framework's cooldown
            # check returns BEFORE on_agent_start fires. So an agent the engine
            # deemed eligible (announced in on_phase_start -> `ran`) that produced
            # no response was gated by its own cooldown. Recover and surface it.
            for name in phase.get("ran", []):
                if name not in finished_names:
                    g = {"agent": name, "reason": "cooldown", "phase": phase["phase"]}
                    # Causal chain: when did this agent last actually run, and did
                    # it speak? (A *silent* prior run still spent the cooldown.)
                    last = self._last_fire.get(name)
                    if last:
                        g["last_turn"] = last["turn"]
                        g["last_spoke"] = last["spoke"]
                    gated.append(g)
            phase["eligible"] = list(phase.get("ran", []))
            phase["gated"] = [g["agent"] for g in gated if g["phase"] == phase["phase"]]
            phase_max = max(
                (self._latency_by_name.get(n, 0.0) for n in finished_names), default=0.0
            )
            phase_latencies.append({"phase": phase["phase"], "modeled_latency_ms": phase_max})
            modeled_total += phase_max

        insights = [
            {
                "agent_id": getattr(i, "agent_id", None),
                "agent_name": getattr(i, "agent_name", None),
                "type": getattr(getattr(i, "type", None), "value", str(getattr(i, "type", ""))),
                "content": getattr(i, "content", ""),
                "confidence": getattr(i, "confidence", None),
                "expiry": getattr(i, "expiry", None),
                "action_label": getattr(i, "action_label", None),
                "metadata": getattr(i, "metadata", {}) or {},
            }
            for i in getattr(response, "insights", [])
        ]

        events = [
            {"name": getattr(e, "name", ""), "source_agent": getattr(e, "source_agent", ""),
             "payload": getattr(e, "payload", {})}
            for e in getattr(response, "events", [])
        ]

        return {
            "turn_index": self.turn_count,
            "cursor": self.cursor,
            "sim_time": kw["timestamp"],
            "trigger_type": kw["trigger_name"],
            "trigger_metadata": kw["trigger_metadata"],
            "allowed_agent_ids": kw["allowed_agent_ids"],
            "segment": (
                {
                    "speaker": step.get("speaker", "SPEAKER"),
                    "text": step.get("text", ""),
                    "timestamp": kw["timestamp"],
                }
                if kw["has_text"]
                else None
            ),
            "note": step.get("note"),
            "insights": insights,
            "events": events,
            "phases": trace["phases"],
            "skipped": trace["skipped"],
            "gated": gated,
            "errors": trace["errors"],
            "phase_latencies": phase_latencies,
            "modeled_latency_ms": round(modeled_total, 2),
            "real_latency_ms": trace["real_latency_ms"],
            "blackboard_before": bb_before,
            "blackboard_after": bb_after,
            "blackboard_delta": self._delta(bb_before, bb_after),
        }

    @staticmethod
    def _delta(before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, Any]:
        d: Dict[str, Any] = {"variables": {}, "facts_added": [], "queues": {}, "memory": []}

        bvars, avars = before.get("variables", {}), after.get("variables", {})
        for k, v in avars.items():
            if k.startswith("sys."):
                continue
            if k not in bvars or bvars[k] != v:
                d["variables"][k] = {"from": bvars.get(k, None), "to": v}

        def fact_key(f):
            return (f.get("type"), f.get("key"))

        before_keys = {fact_key(f) for f in before.get("facts", [])}
        for f in after.get("facts", []):
            if fact_key(f) not in before_keys:
                d["facts_added"].append(f)

        bq, aq = before.get("queues", {}), after.get("queues", {})
        for name, items in aq.items():
            before_len = len(bq.get(name, []))
            after_len = len(items)
            if after_len != before_len:
                d["queues"][name] = {"from": before_len, "to": after_len}

        bm, am = before.get("memory", {}), after.get("memory", {})
        for agent_id, mem in am.items():
            if bm.get(agent_id) != mem:
                d["memory"].append({"agent_id": agent_id, "memory": mem})

        return d

    def close(self) -> None:
        # Idempotent: each session holds exactly one shim install reference.
        if not self._closed:
            self._closed = True
            uninstall_clock_shim()
