"""
Self-improvement loop for agent prompts.

Press "Self-Improve" and this runs:

    run scenario (real LLM) -> judge results vs the objective -> rewrite the
    agent prompts -> re-run ... until the score hits the target or plateaus.

It keeps the best-scoring suite and documents every round. The intelligence that
makes a real agent good lives in its `text` (system prompt) and its trigger
wiring — NOT in the simulator's mock rules — so this optimizes exactly that.

Seams: `judge` and `optimizer` are injectable so the loop can be exercised
deterministically in tests without an API key. In production both are LLM-backed
(OpenAI, temperature 0).
"""

from __future__ import annotations

import copy
import json
import re
from typing import Any, Callable, Dict, List, Optional

from .driver import SimulationSession
from .llm_compat import create_json
from . import learnings
from .generator import speaker_orientation

try:
    from openai import AsyncOpenAI
except ImportError:  # pragma: no cover
    AsyncOpenAI = None


# =============================================================================
# Deterministic metrics — these catch structural failures regardless of the LLM
# =============================================================================
_WORD = re.compile(r"[a-z0-9]+")


def _tokens(s: str) -> set:
    return set(_WORD.findall((s or "").lower()))


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def compute_metrics(traces: List[Dict[str, Any]], suite: Dict[str, Any]) -> Dict[str, Any]:
    """Objective signals derived from a run's traces."""
    agents = suite.get("agents", [])
    subs = {
        a["id"]: (a.get("trigger_config", {}).get("subscribed_events") or [])
        for a in agents
    }

    emitted_names: set = set()
    whispers_per_turn: List[int] = []
    total_whispers = 0
    redundant_pairs = 0
    per_agent_whispers: Dict[str, int] = {}

    for t in traces:
        w = t.get("insights", []) or []
        whispers_per_turn.append(len(w))
        total_whispers += len(w)
        for ins in w:
            aid = ins.get("agent_id")
            per_agent_whispers[aid] = per_agent_whispers.get(aid, 0) + 1
        for p in t.get("phases", []) or []:
            for a in p.get("agents", []) or []:
                for ev in (a.get("events") or []):
                    emitted_names.add(ev)
        # near-duplicate whispers within one turn (HUD redundancy)
        toks = [_tokens(x.get("content", "")) for x in w]
        for i in range(len(toks)):
            for j in range(i + 1, len(toks)):
                if _jaccard(toks[i], toks[j]) > 0.5:
                    redundant_pairs += 1

    # Coordination integrity: every subscription must see a matching emitted name.
    broken = []
    for aid, evs in subs.items():
        if evs and not any(e in emitted_names for e in evs):
            broken.append(
                {"agent": aid, "subscribed": evs, "emitted_seen": sorted(emitted_names)}
            )

    n = max(1, len(traces))
    return {
        "turns": len(traces),
        "total_whispers": total_whispers,
        "avg_whispers_per_turn": round(total_whispers / n, 2),
        "max_whispers_per_turn": max(whispers_per_turn or [0]),
        "within_turn_redundant_pairs": redundant_pairs,
        "coordination_broken": broken,
        "emitted_event_names": sorted(emitted_names),
        "whispers_by_agent": per_agent_whispers,
    }


# =============================================================================
# Compact run rendering for the judge
# =============================================================================
def render_run(scenario: Dict[str, Any], traces: List[Dict[str, Any]]) -> str:
    lines = []
    for t in traces:
        seg = t.get("segment")
        who = f"{seg['speaker']}: {seg['text']}" if seg else f"({t['trigger_type']} tick)"
        lines.append(f"\n— Turn {t['turn_index']} [{t['trigger_type']}] {who}")
        if t.get("note"):
            lines.append(f"  EXPECTED: {t['note']}")
        evs = sorted({e for p in t.get('phases', []) for e in (p.get('events_emitted') or [])})
        if evs:
            lines.append(f"  events emitted: {evs}")
        for g in t.get("gated", []) or []:
            lines.append(f"  gated(cooldown): {g['agent']}")
        if t.get("insights"):
            for i in t["insights"]:
                lines.append(f"  WHISPER [{i['type']}] {i['agent_name']}: {i['content']}")
        else:
            lines.append("  (no whispers)")
    return "\n".join(lines)


# =============================================================================
# LLM helpers
# =============================================================================
async def _openai_json(client, model: str, system: str, user: str) -> Dict[str, Any]:
    return await create_json(
        client,
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )


_JUDGE_SYSTEM = """You are a STRICT evaluator of a real-time conversational-copilot agent suite \
running on the xubb_agents framework. You score how well the suite behaved on one scripted \
scenario, judged against the per-turn EXPECTED notes and these hard rules:

- COORDINATION: if `coordination_broken` is non-empty in the metrics, an event subscriber never \
  fired because the emitter used a different event name. This is a CRITICAL failure — the score \
  MUST be below 45.
- ANTI-SPAM: a good copilot whispers rarely and only with high-signal advice. Penalize heavily if \
  avg_whispers_per_turn > 2, or if multiple agents repeat the same advice in one turn \
  (within_turn_redundant_pairs > 0).
- ROLE FIDELITY: each agent must stay strictly in its lane. A detector/tracker/monitor that emits \
  generic coaching whispers is OFF-ROLE. Silent state agents that correctly stay silent are GOOD.
- RELEVANCE: whispers must address the LATEST turn, not a stale earlier topic.
- COVERAGE: the agent the EXPECTED note calls for should actually fire; reward matches.

Return ONLY JSON:
{
 "score": 0-100,
 "coordination_ok": true/false,
 "per_agent": [{"agent_id": "...", "verdict": "good|too_chatty|off_role|stale|silent_ok|missing|broken", "problems": ["..."], "fix_hint": "..."}],
 "global_problems": ["..."],
 "summary": "one paragraph"
}"""


async def llm_judge(client, model, scenario, suite, traces, metrics, objective, baseline=None) -> Dict[str, Any]:
    roles = "\n".join(
        f"- {a['id']} ({a.get('name')}): {a.get('_doc') or (a.get('text','')[:160])}"
        for a in suite.get("agents", [])
    )
    user = (
        f"OBJECTIVE:\n{objective}\n\n"
        f"AGENT ROLES:\n{roles}\n\n"
        f"DETERMINISTIC METRICS:\n{json.dumps(metrics, indent=2)}\n\n"
        f"RUN (scenario '{scenario.get('name')}'):\n{render_run(scenario, traces)}\n"
    )
    if baseline:
        blines = "\n".join(
            f"- {b.get('agent_id')} [{b.get('insight_type')}]: {(b.get('content') or '')[:160]}"
            for b in baseline[:40]
        )
        user += (
            "\nPRODUCTION BASELINE — what the CURRENT deployed agents actually whispered on this "
            "(often spammy/redundant). The suite under test must be MORE useful and LESS noisy than "
            f"this baseline; reward it for beating the baseline, penalize it for matching the noise:\n{blines}\n"
        )
    user += "\nScore the suite per the rules. Be harsh about spam, redundancy, off-role whispers, and broken coordination."
    return await _openai_json(client, model, _JUDGE_SYSTEM, user)


_OPT_SYSTEM = """You improve the agents of a xubb_agents conversational-copilot suite so it scores \
higher next round. You may rewrite, per agent:
- `text`  : the agent's SYSTEM PROMPT (this is what the real LLM sees and where ALL behavior lives),
- `trigger_conditions` : optional preconditions, in the EXACT schema below,
- `cooldown`, `subscribed_events`, `keywords`.

CRITICAL — trigger_conditions schema. They gate on BLACKBOARD STATE ONLY (variables/facts/queues that \
agents have written on prior turns), NEVER on transcript text, speaker, or "the latest turn". Format:
  {"mode": "all"|"any", "rules": [ {"var": "<name>", "op": "<op>", "value": <v>}, \
{"fact": "<type>", "op": "exists"}, {"queue": "<name>", "op": "not_empty"} ]}
Valid ops ONLY: eq, neq, gt, gte, lt, lte, in, not_in, contains, exists, present, not_exists, not_empty, empty, mod.
You may NOT invent keys like "speaker", "ends_with", "contains_keywords" — the engine ignores any rules-less \
object (it silently passes), so those do nothing. Per-turn gating on text/speaker/"is this a question" MUST \
live in the agent's PROMPT (has_insight=false unless ...), not in trigger_conditions. Only add \
trigger_conditions when a real blackboard variable/fact/queue exists to test; otherwise OMIT them.

Apply these principles, hard:
1. STRICT ROLE: each prompt must define one narrow job and forbid everything else ("Do NOT give \
   generic interview/sales advice. You ONLY do X.").
2. SILENCE BY DEFAULT: instruct the agent to return has_insight=false UNLESS its specific trigger \
   appears in the MOST RECENT turn. Detectors/trackers that exist to write state or emit events \
   must set has_insight=false ALWAYS (they never whisper).
3. FIX COORDINATION: an emitter MUST emit the EXACT event name its subscriber listens for. Pin the \
   literal name in the emitter's prompt, e.g. 'emit an event named exactly "interviewer_question"'. \
   Use the subscribers' `subscribed_events` as the source of truth for the name.
4. NO REDUNDANCY: keep each agent's territory disjoint so two agents never give the same advice.
5. RELEVANCE: tell the agent to address only the latest turn.
6. USE THE TRAJECTORY: you are given the history of prior rounds (the change made and the score it \
   produced). KEEP and build on changes that RAISED the score; do NOT re-apply a change already shown \
   not to help; if the most recent change LOWERED the score, reconsider or revert that specific change \
   rather than piling on. Make targeted edits, not a full rewrite each round.
7. UNBLOCK BROKEN COORDINATION — do NOT keep re-tuning a dead emitter. If `coordination_broken` is \
   non-empty and the emitter "did not emit" (especially if a prior round already tried to fix its \
   wording), STOP wording-tweaks: LLM event-emission is unreliable. COLLAPSE the pair into one \
   turn_based agent: patch the SUBSCRIBER to {"subscribed_events": [], "mode": "turn_based"} and \
   rewrite its prompt to detect the condition DIRECTLY from the latest transcript turn and whisper; \
   then neutralize the now-redundant emitter (rewrite it to a silent no-op, since you cannot delete \
   agents). One turn_based detect-and-advise agent is far more robust than a fragile emitter→subscriber.

Keep each agent's `id` unchanged. Only return agents you actually changed.
Return ONLY JSON:
{"patches": [{"id": "...", "text": "...", "trigger_conditions": {...}|null, "cooldown": int|null, "subscribed_events": [...]|null, "keywords": [...]|null}], "rationale": "what you changed and why"}"""


def build_history(rounds) -> List[Dict[str, Any]]:
    """Compact OPRO-style trajectory from the accumulated rounds: each entry pairs
    the change made after a round with the score delta it then produced."""
    scored = [r for r in rounds if "error" not in r and r.get("score") is not None]
    hist = []
    for idx, rec in enumerate(scored):
        entry = {"round": rec["round"], "score": rec.get("score")}
        rw = rec.get("rewrite")
        if rw:
            entry["change"] = {"agents": rw.get("patched_agents"), "rationale": rw.get("rationale")}
            if idx + 1 < len(scored):  # change produced the NEXT round's score
                entry["result_delta"] = round(scored[idx + 1]["score"] - rec["score"], 1)
        j = rec.get("judgement") or {}
        probs = [f"{pa.get('agent_id')}: {pa['problems'][0]}"
                 for pa in (j.get("per_agent") or []) if pa.get("problems")]
        probs += (j.get("global_problems") or [])
        if probs:
            entry["problems"] = probs[:4]
        hist.append(entry)
    return hist


def render_trajectory(history) -> str:
    """Readable (change → outcome) history so the optimizer learns what helped."""
    lines = []
    for h in history or []:
        line = f"Round {h['round']}: score {h.get('score')}."
        ch = h.get("change")
        if ch:
            rd = h.get("result_delta")
            eff = (f" → outcome {'+' if rd >= 0 else ''}{rd}" if rd is not None
                   else " (its effect is the current round)")
            line += f" Applied [{', '.join(ch.get('agents') or [])}]: {ch.get('rationale')}.{eff}"
        if h.get("problems"):
            line += f"   [open: {'; '.join(h['problems'])}]"
        lines.append(line)
    return "\n".join(lines)


async def llm_optimize(client, model, suite, judgement, objective, history=None, learned_principles=None, user_speaker=None) -> Dict[str, Any]:
    agents_view = [
        {
            "id": a["id"],
            "name": a.get("name"),
            "role_doc": a.get("_doc", ""),
            "current_text": a.get("text", ""),
            "trigger_config": a.get("trigger_config", {}),
            "trigger_conditions": a.get("trigger_conditions"),
        }
        for a in suite.get("agents", [])
    ]
    user = f"OBJECTIVE:\n{objective}\n\n"
    if history:
        user += (
            "TRAJECTORY SO FAR — keep the changes that raised the score, do NOT repeat ones that "
            "didn't help, and if the latest change lowered the score, reconsider/revert it:\n"
            f"{render_trajectory(history)}\n\n"
        )
    user += (
        f"CURRENT ROUND DIAGNOSIS:\n{json.dumps(judgement, indent=2)}\n\n"
        f"CURRENT AGENTS:\n{json.dumps(agents_view, indent=2)}\n\n"
        "Make targeted edits to fix the diagnosis, informed by the trajectory. "
        "Return patches only for agents you change."
    )
    system = (_OPT_SYSTEM + speaker_orientation(user_speaker)
              + learnings.principles_block(learned_principles or []))
    return await _openai_json(client, model, system, user)


# =============================================================================
# Apply patches to a suite (by id; preserves _mock/_doc/structure)
# =============================================================================
def apply_patches(suite: Dict[str, Any], patches: List[Dict[str, Any]]) -> Dict[str, Any]:
    new = copy.deepcopy(suite)
    by_id = {a["id"]: a for a in new.get("agents", [])}
    for p in patches or []:
        a = by_id.get(p.get("id"))
        if not a:
            continue
        if p.get("text"):
            a["text"] = p["text"]
        if p.get("trigger_conditions") is not None:
            a["trigger_conditions"] = p["trigger_conditions"]
        tc = a.setdefault("trigger_config", {})
        if p.get("cooldown") is not None:
            tc["cooldown"] = p["cooldown"]
        if p.get("subscribed_events") is not None:
            tc["subscribed_events"] = p["subscribed_events"]
        if p.get("keywords") is not None:
            tc["keywords"] = p["keywords"]
    return new


# =============================================================================
# The loop
# =============================================================================
async def run_self_improvement(
    suite: Dict[str, Any],
    scenario: Dict[str, Any],
    api_key: Optional[str],
    objective: str,
    target_score: int = 85,
    max_rounds: int = 5,
    optimizer_model: str = "gpt-4o",
    run_mode: str = "real",
    baseline: Optional[list] = None,
    learned_principles: Optional[list] = None,
    progress_cb: Optional[Callable[[Dict[str, Any]], None]] = None,
    judge: Optional[Callable] = None,
    optimizer: Optional[Callable] = None,
) -> Dict[str, Any]:
    """Run the optimize loop. Returns {best, rounds, report_md}.

    `judge`/`optimizer` may be injected (for tests). By default they are the
    OpenAI-backed `llm_judge`/`llm_optimize`, called with a client built from
    `api_key`.
    """
    client = None
    if judge is None or optimizer is None:
        if AsyncOpenAI is None:
            raise RuntimeError("openai package not available")
        if not api_key:
            raise RuntimeError("Self-Improvement needs an OpenAI API key (real mode).")
        client = AsyncOpenAI(api_key=api_key)

    async def _judge(s, su, tr, m):
        if judge is not None:
            return await judge(s, su, tr, m)
        return await llm_judge(client, optimizer_model, s, su, tr, m, objective, baseline)

    if learned_principles is None:
        try:
            learned_principles = learnings.active_principles(scope="all")
        except Exception:
            learned_principles = []

    _user_speaker = scenario.get("user_speaker")

    async def _optimize(su, jd, hist):
        if optimizer is not None:
            return await optimizer(su, jd, hist)
        return await llm_optimize(client, optimizer_model, su, jd, objective, hist,
                                  learned_principles, _user_speaker)

    def emit(record):
        if progress_cb:
            try:
                progress_cb(record)
            except Exception:
                pass

    current = copy.deepcopy(suite)
    best = {"score": -1, "suite": current, "round": 0}
    rounds: List[Dict[str, Any]] = []

    for r in range(1, max_rounds + 1):
        emit({"type": "round_start", "round": r})

        # 1. Run the scenario through the current suite.
        try:
            sess = SimulationSession(
                suite=current, scenario=scenario, mode=run_mode, api_key=api_key
            )
            traces = await sess.run_all()
            sess.close()
        except Exception as e:
            rec = {"round": r, "error": f"run failed: {e}"}
            rounds.append(rec)
            emit({"type": "round_error", **rec})
            break

        # 2. Objective metrics + LLM judgement.
        metrics = compute_metrics(traces, current)
        try:
            judgement = await _judge(scenario, current, traces, metrics)
        except Exception as e:
            rec = {"round": r, "error": f"judge failed: {e}", "metrics": metrics}
            rounds.append(rec)
            emit({"type": "round_error", **rec})
            break

        score = float(judgement.get("score", 0))
        rec = {
            "round": r,
            "score": score,
            "metrics": metrics,
            "judgement": judgement,
            "suite": copy.deepcopy(current),
        }
        rounds.append(rec)
        if score > best["score"]:
            best = {"score": score, "suite": copy.deepcopy(current), "round": r}
        emit({"type": "round_done", "round": r, "score": score,
              "metrics": metrics, "judgement": judgement})

        # 3. Stop conditions: target hit, last round, or plateau.
        if score >= target_score:
            emit({"type": "stop", "reason": "target_reached", "round": r})
            break
        if r == max_rounds:
            emit({"type": "stop", "reason": "max_rounds", "round": r})
            break
        if len(rounds) >= 3:
            recent = [x.get("score", 0) for x in rounds[-3:]]
            if max(recent) - min(recent) < 1.5:
                emit({"type": "stop", "reason": "plateau", "round": r})
                break

        # 4. Rewrite the prompts, informed by the trajectory so far (OPRO-style).
        try:
            opt = await _optimize(current, judgement, build_history(rounds))
            candidate = apply_patches(current, opt.get("patches", []))
            # Validate it still builds before adopting.
            SimulationSession(suite=candidate, scenario=scenario, mode="mock").close()
            rec["rewrite"] = {
                "rationale": opt.get("rationale", ""),
                "patched_agents": [p.get("id") for p in opt.get("patches", [])],
            }
            current = candidate
            emit({"type": "rewrite", "round": r,
                  "patched_agents": rec["rewrite"]["patched_agents"],
                  "rationale": rec["rewrite"]["rationale"]})
        except Exception as e:
            rec["rewrite_error"] = str(e)
            emit({"type": "rewrite_error", "round": r, "error": str(e)})
            break

    report = build_report(scenario, suite, best, rounds)
    return {"best": best, "rounds": rounds, "report_md": report}


def build_report(scenario, original_suite, best, rounds) -> str:
    lines = [
        f"# Self-Improvement Report — {original_suite.get('name')}",
        f"Scenario: **{scenario.get('name')}**",
        f"Best score: **{best['score']:.0f}/100** (round {best['round']}) over {len(rounds)} round(s).",
        "",
        "## Round-by-round",
    ]
    for rec in rounds:
        if "error" in rec:
            lines.append(f"\n### Round {rec['round']} — ERROR\n{rec['error']}")
            continue
        m = rec.get("metrics", {})
        j = rec.get("judgement", {})
        lines.append(f"\n### Round {rec['round']} — score {rec.get('score', 0):.0f}/100")
        lines.append(
            f"- whispers/turn avg **{m.get('avg_whispers_per_turn')}** "
            f"(max {m.get('max_whispers_per_turn')}), "
            f"redundant pairs **{m.get('within_turn_redundant_pairs')}**, "
            f"coordination_broken **{len(m.get('coordination_broken', []))}**"
        )
        if j.get("summary"):
            lines.append(f"- judge: {j['summary']}")
        for pa in j.get("per_agent", []) or []:
            if pa.get("verdict") not in ("good", "silent_ok") and pa.get("problems"):
                lines.append(f"  - `{pa.get('agent_id')}` [{pa.get('verdict')}]: "
                             f"{'; '.join(pa.get('problems', []))}")
        rw = rec.get("rewrite")
        if rw:
            # Outcome: did this change raise the next round's score? (the learning)
            later = [x for x in rounds if "error" not in x
                     and x.get("score") is not None and x["round"] > rec["round"]]
            outcome = ""
            if later:
                d = later[0]["score"] - rec.get("score", 0)
                outcome = f"  → next round **{'+' if d >= 0 else ''}{d:.0f}**"
            lines.append(f"- rewrote: {', '.join(rw['patched_agents'])} — {rw['rationale']}{outcome}")
    lines.append("\n## Final optimized prompts")
    for a in best["suite"].get("agents", []):
        lines.append(f"\n### {a.get('name')} (`{a['id']}`)")
        if a.get("trigger_conditions"):
            lines.append(f"trigger_conditions: `{json.dumps(a['trigger_conditions'])}`")
        lines.append(f"```\n{a.get('text', '')}\n```")
    return "\n".join(lines)
