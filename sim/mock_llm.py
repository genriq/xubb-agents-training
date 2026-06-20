"""
Deterministic, rule-based stand-in for the framework's OpenAI LLM client.

Why a mock:
- It's free and reproducible — perfect for studying the *framework mechanics*
  (phases, events, snapshot isolation, cooldowns, priority merges) without
  burning API budget or fighting non-determinism.
- It is a *pure function of the prompt*, exactly like a real LLM. The rules match
  against the rendered transcript the agent actually sent — so what you learn
  about "the agent only reacts to what's in its prompt" transfers directly.

Interface parity:
- Mirrors `xubb_agents.core.llm.LLMClient.generate_json(model, messages, ...)`.
- One `MockLLMClient` instance per agent, holding that agent's rules and id, so
  the mock knows who it is speaking for.

Output shape:
- The mock emits a *superset* dict so it works with any shipped schema
  (`default`, `default_v2`, `v2_raw`): it sets both `has_insight` (the gate),
  `content`/`message` (content aliases), a nested `insight` object (for the
  `v2_raw` root_key), and top-level `events`/`variable_updates`/`queue_pushes`/
  `facts`/`memory_updates` (which the DynamicAgent parser reads by default name
  regardless of schema). The parser simply ignores keys its schema doesn't map.

Rule format (see data/suites/*.json for worked examples):

    {
      "when": {                         # all predicates AND together; omit => always
        "any_keywords": ["price", "expensive"],   # case-insensitive substring
        "all_keywords": ["budget", "quarter"],
        "not_keywords": ["later"],
        "regex": "\\$\\s?\\d",                    # python regex, IGNORECASE
        "speaker": "CUSTOMER",                     # last segment's speaker
        "min_turn": 3, "max_turn": 20,             # sys turn-count gates
        "has_event": "question_detected",          # live blackboard (Phase 2)
        "var_equals": {"phase": "negotiation"},    # blackboard variable check
        "has_fact": {"type": "budget"},            # blackboard fact check
        "queue_not_empty": "pending_questions"
      },
      "scope": "last",                  # "last" (default) | "window" | "all"
      "speak": {                        # omit => stay silent, side effects still apply
        "content": "Price objection — anchor on value, not discount.",
        "type": "warning",             # suggestion|warning|opportunity|fact|praise
        "confidence": 0.9,
        "action_label": "Handle objection",   # optional
        "expiry": 20                           # optional, seconds
      },
      "events": [{"name": "objection_raised", "payload": {"kind": "price"}}],
      "facts":  [{"type": "objection", "key": "price", "value": "...", "confidence": 0.8}],
      "variable_updates": {"phase": "negotiation"},
      "queue_pushes": {"pending_questions": ["What's the budget range?"]},
      "memory_updates": {"objection_count": {"$inc": 1}}    # $inc reads live memory
    }

The first rule whose `when` matches wins; order rules most-specific-first.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple


class SimContext:
    """Per-session state the mock reads when evaluating rules.

    `blackboard` is re-pointed to a fresh `blackboard.snapshot()` (a deep copy) at
    every `on_phase_start` by the tracer — exactly mirroring the framework, which
    runs each phase against `context.blackboard.snapshot()`. So mock predicates
    (`has_event`, `var_equals`, `has_fact`, `queue_not_empty`, and `$inc`) read the
    SAME immutable per-phase snapshot a real DynamicAgent would see in its Jinja
    template:
      * Phase 1 sees pre-turn state (no Phase-1 sibling's uncommitted write — the
        engine merges only after all of a phase's agents finish).
      * Phase 2 sees Phase-1's committed writes and emitted events.
    This keeps the abstraction faithful: a rule cannot observe anything the real
    engine's snapshot isolation would hide.
    """

    def __init__(self):
        self.turn_count: int = 0
        self.segments: list = []            # list[TranscriptSegment]
        self.blackboard: Optional[Any] = None
        self.trigger_type: str = "turn_based"


_TRANSCRIPT_MARKER = "### TRANSCRIPT:"


class MockLLMClient:
    """Per-agent deterministic LLM stand-in."""

    def __init__(
        self,
        agent_id: str,
        rules: Optional[List[Dict[str, Any]]],
        sim_ctx: SimContext,
        latency_ms: float = 0.0,
    ):
        self.agent_id = agent_id
        self.rules = rules or []
        self.sim_ctx = sim_ctx
        self.latency_ms = float(latency_ms)
        # Parity with the real client's public surface.
        self.last_error_category: Optional[str] = None
        self.client = None

    # ------------------------------------------------------------------ API
    async def generate_json(
        self,
        model: str,
        messages: list,
        max_tokens: Optional[int] = None,
        timeout: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """Return a schema-superset dict, or a silent gate. Never raises."""
        self.last_error_category = None
        transcript = self._extract_transcript(messages)
        full_prompt = self._concat(messages)

        # First matching rule wins. (Rules are ordered most-specific-first.)
        for rule in self.rules:
            if self._matches(rule, transcript, full_prompt):
                return self._build_output(rule)

        return {"has_insight": False}

    # -------------------------------------------------------------- matching
    def _matches(
        self,
        rule: Dict[str, Any],
        transcript: List[Tuple[str, str]],
        full_prompt: str,
    ) -> bool:
        when = rule.get("when") or {}
        if not when:
            return True

        scope = rule.get("scope", "last")
        if scope == "last":
            text = transcript[-1][1] if transcript else ""
        elif scope == "all":
            text = full_prompt
        else:  # "window"
            text = "\n".join(t for _, t in transcript)
        text_l = text.lower()

        any_kw = when.get("any_keywords")
        if any_kw and not any(k.lower() in text_l for k in any_kw):
            return False

        all_kw = when.get("all_keywords")
        if all_kw and not all(k.lower() in text_l for k in all_kw):
            return False

        not_kw = when.get("not_keywords")
        if not_kw and any(k.lower() in text_l for k in not_kw):
            return False

        rx = when.get("regex")
        if rx:
            try:
                if not re.search(rx, text, re.IGNORECASE):
                    return False
            except re.error:
                # A malformed rule regex must not crash the agent/turn — treat as
                # no-match (the mock's never-raise contract mirrors the real client).
                return False

        speaker = when.get("speaker")
        if speaker is not None:
            last_speaker = transcript[-1][0] if transcript else None
            if last_speaker != speaker:
                return False

        min_turn = when.get("min_turn")
        if min_turn is not None and self.sim_ctx.turn_count < min_turn:
            return False

        max_turn = when.get("max_turn")
        if max_turn is not None and self.sim_ctx.turn_count > max_turn:
            return False

        bb = self.sim_ctx.blackboard

        ev = when.get("has_event")
        if ev is not None:
            if not (bb and bb.has_event(ev)):
                return False

        var_eq = when.get("var_equals")
        if var_eq:
            if not bb:
                return False
            for k, v in var_eq.items():
                if bb.get_var(k) != v:
                    return False

        has_fact = when.get("has_fact")
        if has_fact:
            if not bb:
                return False
            if not bb.has_fact(has_fact.get("type"), has_fact.get("key")):
                return False

        q_ne = when.get("queue_not_empty")
        if q_ne is not None:
            if not (bb and bb.queue_length(q_ne) > 0):
                return False

        return True

    # ---------------------------------------------------------------- output
    def _build_output(self, rule: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        speak = rule.get("speak")
        if speak:
            content = speak.get("content", "")
            itype = speak.get("type", "suggestion")
            conf = speak.get("confidence", 0.9)
            out["has_insight"] = True
            out["content"] = content
            out["message"] = content  # `default` schema content_field alias
            out["type"] = itype
            out["confidence"] = conf
            if "expiry" in speak:
                out["expiry"] = speak["expiry"]
            if "action_label" in speak:
                out["action_label"] = speak["action_label"]
            # `v2_raw` reads its insight from a nested root object.
            out["insight"] = {"content": content, "type": itype, "confidence": conf}
        else:
            out["has_insight"] = False

        if rule.get("events"):
            out["events"] = rule["events"]

        if rule.get("variable_updates"):
            out["variable_updates"] = rule["variable_updates"]
            out["state_snapshot"] = rule["variable_updates"]  # `v2_raw` var field

        if rule.get("queue_pushes"):
            out["queue_pushes"] = rule["queue_pushes"]

        if rule.get("facts"):
            out["facts"] = rule["facts"]

        if rule.get("memory_updates"):
            out["memory_updates"] = self._resolve_memory_ops(rule["memory_updates"])

        return out

    def _resolve_memory_ops(self, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve simple `$inc` ops against the agent's live committed memory."""
        bb = self.sim_ctx.blackboard
        current: Dict[str, Any] = {}
        if bb is not None:
            try:
                current = bb.get_memory(self.agent_id) or {}
            except Exception:
                current = {}
        resolved: Dict[str, Any] = {}
        for key, val in updates.items():
            if isinstance(val, dict) and "$inc" in val:
                base = current.get(key, 0)
                try:
                    base = float(base)
                except (TypeError, ValueError):
                    base = 0
                inc = val["$inc"]
                new_val = base + inc
                # Keep ints clean.
                if isinstance(inc, int) and float(new_val).is_integer():
                    new_val = int(new_val)
                resolved[key] = new_val
            else:
                resolved[key] = val
        return resolved

    # --------------------------------------------------------------- helpers
    @staticmethod
    def _concat(messages: list) -> str:
        parts = []
        for m in messages or []:
            try:
                parts.append(str(m.get("content", "")))
            except AttributeError:
                parts.append(str(m))
        return "\n".join(parts)

    def _extract_transcript(self, messages: list) -> List[Tuple[str, str]]:
        """Parse the `### TRANSCRIPT:` user message into (speaker, text) lines.

        The DynamicAgent renders the window as `SPEAKER: text` lines in the user
        message. We parse the *last* user message containing the marker.
        """
        transcript_block = ""
        for m in messages or []:
            content = ""
            try:
                content = str(m.get("content", ""))
            except AttributeError:
                content = str(m)
            if _TRANSCRIPT_MARKER in content:
                transcript_block = content.split(_TRANSCRIPT_MARKER, 1)[1]
        lines: List[Tuple[str, str]] = []
        for raw in transcript_block.splitlines():
            line = raw.strip()
            if not line:
                continue
            if ":" in line:
                speaker, text = line.split(":", 1)
                lines.append((speaker.strip(), text.strip()))
            else:
                lines.append(("", line))
        return lines
