"""
Agent generator — an LLM designs a starter copilot suite from a goal.

It bakes in the hard-won lessons (strict role, silence-by-default, pinned event
names for coordination, no overlap) so the generated agents are GOOD by
construction; the Self-Improve loop then refines them against real conversations.

`generator` is an injectable seam so this is testable without an API key.
"""

from __future__ import annotations

import copy
import json
from typing import Any, Dict, Optional

from .driver import SimulationSession
from . import learnings

try:
    from openai import AsyncOpenAI
except ImportError:  # pragma: no cover
    AsyncOpenAI = None


_GEN_SYSTEM = """You design agent suites for the xubb_agents real-time conversational-copilot \
framework. Given a GOAL, output a suite of specialized agents that whisper concise PRIVATE coaching \
to the user during a live conversation.

Framework facts you MUST respect:
- Each agent is a config: {id (kebab-case), name, output_format: "default_v2", trigger_config, text}.
- trigger_config: {mode: "turn_based" | "keyword" | "silence" | "interval" | ["turn_based","keyword"], \
cooldown: <seconds int>, keywords: [...], subscribed_events: [...], priority: <int>}.
- `text` is the agent's SYSTEM PROMPT — ALL behavior lives there.
- output_format default_v2 lets the model return: has_insight (bool gate), content, \
type (suggestion|warning|opportunity|fact|praise), confidence, events:[{name,payload}], \
variable_updates, queue_pushes, facts, memory_updates.
HOW AGENTS FIRE — READ CAREFULLY (this is where suites usually break):
- The conversation is replayed as a stream of TURN_BASED turns. An agent only runs if its trigger \
type matches the turn. So EVERY agent MUST be able to fire on a turn_based turn:
  * use mode "turn_based" (or multi-mode ["turn_based","keyword"]) for ALL agents that should whisper \
    on the conversation, AND
  * the ONLY exception is a pure event SUBSCRIBER (mode "event", subscribed_events:[...]), which fires \
    in Phase 2 of a turn where another agent EMITTED that exact event.
- Do NOT use "silence" or "interval" as an agent's mode, and do NOT make any agent keyword-ONLY — those \
  will NEVER fire here (there are no silence/interval/keyword-routed turns). Per-turn gating (objection? \
  question? filler?) belongs in the PROMPT (has_insight=false unless ...), not in the trigger mode.
- trigger_conditions gate on BLACKBOARD STATE that a DIFFERENT agent writes on an EARLIER turn \
  (variables/facts/queues), via {"mode":"all"|"any","rules":[{"var":"x","op":"eq","value":"y"},\
  {"fact":"t","op":"exists"},{"queue":"q","op":"not_empty"}]} (ops: eq/neq/gt/gte/lt/lte/in/not_in/\
  contains/exists/present/not_exists/not_empty/empty/mod). A condition that references state nothing sets \
  makes the agent NEVER fire. When unsure, OMIT trigger_conditions entirely and gate in the prompt.
- For an event pair: the EMITTER is a turn_based agent whose prompt tells it to emit an event with an \
  EXACT name; the SUBSCRIBER (mode "event") lists that exact name in subscribed_events. The name must \
  match literally in both.

Design rules — write GOOD agents from the start:
1. STRICT ROLE: each prompt defines ONE narrow job and explicitly forbids generic advice.
2. SILENCE BY DEFAULT: the prompt MUST instruct the model to return has_insight=false UNLESS this \
agent's specific trigger is present in the MOST RECENT turn. Pure detector/tracker/extractor agents \
that only emit events or write state must NEVER whisper (has_insight always false).
3. COORDINATION — USE SPARINGLY (this is the #1 source of broken suites): PREFER a single \
turn_based agent that detects a situation AND whispers about it directly. Only create an event pair \
when 2+ DISTINCT agents must react to the SAME signal — otherwise a needless emitter→subscriber pair \
is just a fragile point of failure (LLM event-emission is unreliable). If you DO create an emitter: \
(a) pin the EXACT event name in both prompts, and (b) state explicitly that it MUST return an \
`events` array when triggered — being silent (has_insight=false) means NO whisper, NOT no event. \
When in doubt, do NOT use coordination; use one turn_based agent.
4. NO OVERLAP: keep each agent's territory disjoint so two agents never give the same advice.
5. RELEVANCE: tell each agent to address only the latest turn; keep whispers short.
Keep it tight: 3-6 agents unless the goal clearly needs more.

Return ONLY JSON: {"name": "...", "description": "...", "agents": [ \
{"id": "...", "name": "...", "output_format": "default_v2", "trigger_config": {...}, "text": "..."}, ... ]}"""


def speaker_orientation(user_speaker: Optional[str]) -> str:
    """The fixed framing EVERY agent must encode: who is the user vs the audience,
    and that the whisper is private advice TO the user. Parameterized by the
    scenario's user_speaker (e.g. 'YOU' in production, 'REP'/'ME' in demos)."""
    us = user_speaker or "YOU"
    return (
        "\n\nSPEAKER ORIENTATION — bake this into EVERY agent's prompt:\n"
        f"- In the transcript, the speaker labeled \"{us}\" is the USER you privately coach — those "
        "are the user's OWN words.\n"
        "- EVERY other speaker label is the AUDIENCE / counterparty (could be one person or many).\n"
        f"- Each agent MUST: (a) attribute correctly — distinguish what the user (\"{us}\") said from "
        "what the audience said; (b) frame its whisper as PRIVATE advice spoken TO the user "
        "(\"you…\", \"ask them…\"), NEVER addressed to the audience; (c) watch the AUDIENCE for "
        f"questions / objections / buying-signals, and watch the USER (\"{us}\") for the user's own "
        "missteps / over-promises."
    )


async def generate_suite(
    goal: str,
    api_key: Optional[str],
    model: str = "gpt-4o",
    session_context: Optional[str] = None,
    baseline_sample: Optional[str] = None,
    learned_principles: Optional[list] = None,
    user_speaker: Optional[str] = None,
    generator=None,
) -> Dict[str, Any]:
    """Generate a suite for `goal`. Returns a normalized, build-validated suite dict.

    Learned principles (auto-distilled from past Self-Improve runs) are injected
    into the system prompt so each generation starts smarter than the last.
    """
    if generator is not None:
        suite = await generator(goal, session_context, baseline_sample)
    else:
        if AsyncOpenAI is None:
            raise RuntimeError("openai package not available")
        if not api_key:
            raise RuntimeError("Agent generation needs an OpenAI API key.")
        principles = (learned_principles if learned_principles is not None
                      else learnings.active_principles(scope="all"))
        system = _GEN_SYSTEM + speaker_orientation(user_speaker) + learnings.principles_block(principles)
        client = AsyncOpenAI(api_key=api_key)
        user = f"GOAL:\n{goal}\n"
        if session_context:
            user += ("\nThe copilot will run on real conversations like this excerpt — design for "
                     f"THIS kind of talk:\n{session_context[:1800]}\n")
        if baseline_sample:
            user += ("\nThe current production agents whisper things like the following. Do BETTER — "
                     f"sharper, more specialized, far less spammy:\n{baseline_sample[:1400]}\n")
        resp = await client.chat.completions.create(
            model=model, temperature=0.4,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
        )
        suite = json.loads(resp.choices[0].message.content)

    suite = _normalize(suite)
    if not suite.get("agents"):
        raise ValueError("generator returned no agents")
    # Auto-fix the firing logic the LLM commonly gets wrong, THEN build-validate.
    suite, fixes = _lint_and_fix(suite)
    suite["_lint"] = fixes
    SimulationSession(suite=suite, scenario={"steps": []}, mode="mock").close()
    return suite


# Modes that can actually fire on the turn_based replay (everything else is dead
# unless the scenario happens to contain that trigger, which imported real
# sessions never do).
_FIRING_MODES = {"turn_based", "event"}


def _as_mode_list(mode) -> list:
    if isinstance(mode, list):
        return [m for m in mode if isinstance(m, str)]
    if isinstance(mode, str):
        return [mode]
    return []


def _lint_and_fix(suite: Dict[str, Any]):
    """Repair the firing errors LLMs make, so generated agents actually trigger.

    Returns (suite, human-readable fix notes). Deterministic safety net that runs
    regardless of how well the model followed the prompt.
    """
    fixes = []
    agents = suite.get("agents", []) or []
    for a in agents:
        tc = a.setdefault("trigger_config", {})
        # 1. Strip trigger_conditions — the #1 fail-closed cause. A starter suite
        #    has no guaranteed prior-turn state to gate on; gate in the prompt.
        if a.pop("trigger_conditions", None):
            fixes.append(f"{a['id']}: removed trigger_conditions (would fail-closed on unset state — gate in the prompt instead)")
        if tc.pop("trigger_conditions", None):
            fixes.append(f"{a['id']}: removed trigger_config.trigger_conditions")
        # 2. Ensure the agent can fire on a turn_based turn. Pure
        #    keyword/silence/interval agents never fire on the replay; add
        #    'turn_based' so they run and gate via their prompt. Event-only
        #    subscribers are left alone (they fire in Phase 2).
        modes = _as_mode_list(tc.get("mode", "turn_based")) or ["turn_based"]
        if not (set(modes) & _FIRING_MODES):
            modes = modes + ["turn_based"]
            fixes.append(f"{a['id']}: mode {tc.get('mode')!r} can't fire on turn-based turns — added 'turn_based'")
        tc["mode"] = modes[0] if len(modes) == 1 else modes
    # 3. Warn about orphan event subscribers (no other agent mentions emitting it).
    for a in agents:
        subs = (a.get("trigger_config", {}).get("subscribed_events")) or []
        others = " ".join((b.get("text", "") or "") for b in agents if b is not a).lower()
        for ev in subs:
            if ev and ev.lower() not in others:
                fixes.append(f"WARNING {a['id']}: subscribes to '{ev}' but no other agent's prompt emits it — it may never fire")
    return suite, fixes


def _normalize(suite: Dict[str, Any]) -> Dict[str, Any]:
    suite = copy.deepcopy(suite if isinstance(suite, dict) else {})
    # Some models wrap the array — accept {"suite": {...}} or a bare list.
    if "agents" not in suite:
        if isinstance(suite.get("suite"), dict):
            suite = suite["suite"]
        elif isinstance(suite.get("agents_list"), list):
            suite["agents"] = suite.pop("agents_list")
    suite.setdefault("name", "Generated suite")
    suite.setdefault("description", "")
    seen = set()
    for i, a in enumerate(suite.get("agents", []) or []):
        aid = (a.get("id") or a.get("name", f"agent-{i}")).lower().replace(" ", "-")[:48]
        while aid in seen:
            aid = f"{aid}-{i}"
        seen.add(aid)
        a["id"] = aid
        a.setdefault("name", aid)
        a.setdefault("output_format", "default_v2")
        tc = a.setdefault("trigger_config", {})
        tc.setdefault("mode", "turn_based")
        tc.setdefault("cooldown", tc.get("cooldown", 8))
        a.setdefault("text", "")
    return suite
