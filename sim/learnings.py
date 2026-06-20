"""
Learning store — distill generalizable lessons from Self-Improve runs, consolidate
them (dedupe + support counts), and inject the proven ones back into future agent
generation and optimization.

Three kinds of lesson, routed differently:
  - structural : deterministic/universal wiring rules -> surfaced as SUGGESTED lint
                 (never auto-written to code; a human promotes them).
  - stylistic  : fuzzy quality/judgment principles -> AUTO-injected into the
                 generator + optimizer prompts once support >= SUPPORT_THRESHOLD.
  - domain     : specific to a conversation domain -> injected when generating for
                 that scope.

Source of truth is `sim/data/learnings.json`; `learnings.md` is a generated,
human-readable view. The `distiller` is an injectable seam so this is testable
without an API key.
"""

from __future__ import annotations

import copy
import json
import os
import re
from typing import Any, Dict, List, Optional

try:
    from openai import AsyncOpenAI
except ImportError:  # pragma: no cover
    AsyncOpenAI = None

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
STORE_PATH = os.path.join(_DATA_DIR, "learnings.json")
MD_PATH = os.path.join(_DATA_DIR, "learnings.md")

SUPPORT_THRESHOLD = 2  # stylistic lessons auto-activate (inject into prompts) at this support


# =========================================================================
# Store I/O
# =========================================================================
def load_store() -> Dict[str, Any]:
    if os.path.exists(STORE_PATH):
        try:
            with open(STORE_PATH, "r", encoding="utf-8") as f:
                store = json.load(f)
                store.setdefault("lessons", [])
                return store
        except Exception:
            pass
    return {"lessons": []}


def save_store(store: Dict[str, Any]) -> None:
    os.makedirs(_DATA_DIR, exist_ok=True)
    with open(STORE_PATH, "w", encoding="utf-8") as f:
        json.dump(store, f, indent=2)
    with open(MD_PATH, "w", encoding="utf-8") as f:
        f.write(render_md(store))


# =========================================================================
# Dedupe / consolidate
# =========================================================================
# Generic filler dropped before dedup matching. Negations / quantifiers
# (never, always, no, not, only) are KEPT — they flip a principle's meaning, so
# two opposite lessons must never merge.
_STOP = {"a", "an", "the", "to", "of", "and", "or", "so", "set", "must", "should",
         "will", "would", "that", "this", "it", "its", "in", "on", "for", "with",
         "you", "your", "be", "is", "are", "as", "at", "by", "an", "if", "when"}


def _tokens(s: str) -> set:
    toks = re.findall(r"[a-z0-9]+", (s or "").lower())
    out = set()
    for t in toks:
        if t in _STOP:
            continue
        if len(t) > 3 and t.endswith("s"):  # crude singularize: agents -> agent
            t = t[:-1]
        out.add(t)
    return out


def _similar(a: str, b: str, thresh: float = 0.55) -> bool:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return False
    return len(ta & tb) / len(ta | tb) >= thresh


def _slug(s: str, n: int = 40) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (s or "lesson").lower()).strip("-")[:n] or "lesson"


def consolidate(store: Dict[str, Any], proposed: List[Dict[str, Any]]) -> (Dict[str, Any], List[Dict[str, Any]]):
    """Merge proposed lessons into the store. Returns (store, change_summary)."""
    store = copy.deepcopy(store)
    lessons = store.setdefault("lessons", [])
    changes: List[Dict[str, Any]] = []
    seen_ids = {l.get("id") for l in lessons}

    for p in proposed or []:
        principle = (p.get("principle") or "").strip()
        if not principle:
            continue
        kind = p.get("kind", "stylistic")
        if kind not in ("structural", "stylistic", "domain"):
            kind = "stylistic"
        delta = 0.0
        try:
            delta = float(p.get("delta") or 0)
        except (TypeError, ValueError):
            delta = 0.0

        match = next(
            (l for l in lessons if l.get("kind") == kind and _similar(l.get("principle", ""), principle)),
            None,
        )
        if match:
            ev = match.setdefault("evidence", {"support": 1, "avg_delta": 0.0})
            n = ev.get("support", 1)
            ev["avg_delta"] = round((ev.get("avg_delta", 0.0) * n + delta) / (n + 1), 1)
            ev["support"] = n + 1
            if kind == "stylistic" and ev["support"] >= SUPPORT_THRESHOLD:
                match["status"] = "active"
            changes.append({"action": "reinforced", "kind": kind,
                            "principle": match["principle"], "support": ev["support"]})
        else:
            lid = _slug(principle)
            while lid in seen_ids:
                lid += "-x"
            seen_ids.add(lid)
            # stylistic starts as a candidate (needs corroboration); structural and
            # domain are surfaced immediately (structural as suggested-lint).
            status = "candidate" if kind == "stylistic" else "active"
            lessons.append({
                "id": lid,
                "principle": principle,
                "kind": kind,
                "before": (p.get("before") or "")[:240],
                "after": (p.get("after") or "")[:240],
                "scope": p.get("scope") or ["all"],
                "evidence": {"support": 1, "avg_delta": round(delta, 1)},
                "status": status,
            })
            changes.append({"action": "new", "kind": kind, "principle": principle})
    return store, changes


# =========================================================================
# Injection: the proven lessons that flow into prompts
# =========================================================================
def active_principles(store: Optional[Dict[str, Any]] = None, scope: str = "all") -> List[str]:
    """Stylistic lessons that have reached the support threshold, plus domain
    lessons in scope — these are auto-injected into generator/optimizer prompts."""
    store = store or load_store()
    out = []
    for l in store.get("lessons", []):
        if l.get("status") != "active":
            continue
        sc = l.get("scope") or ["all"]
        in_scope = ("all" in sc) or (scope in sc)
        if l.get("kind") == "stylistic" and in_scope:
            out.append(l["principle"])
        elif l.get("kind") == "domain" and (scope in sc):
            out.append(l["principle"])
    return out


def structural_suggestions(store: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    store = store or load_store()
    return [l for l in store.get("lessons", []) if l.get("kind") == "structural"]


def principles_block(principles: List[str]) -> str:
    if not principles:
        return ""
    return ("\n\n## Learned principles (proven across past optimization runs — apply them):\n"
            + "\n".join(f"- {p}" for p in principles))


# =========================================================================
# Distill (LLM reads a run's trajectory -> generalizable lessons)
# =========================================================================
_DISTILL_SYSTEM = """You extract GENERALIZABLE, reusable lessons from one agent-suite \
prompt-optimization run, so future suites are designed better. You are given the trajectory: each \
round's change and the score delta it produced.

Rules:
- Output 1-4 lessons, ONLY from changes that RAISED the score (positive delta). Skip changes that \
  didn't help.
- Each lesson must be a GENERAL operational rule that would help OTHER suites — NOT a fact about this \
  specific suite or conversation. Do not quote transcript content.
- Classify each: "structural" (a deterministic/universal wiring rule, e.g. event names must match), \
  "stylistic" (a fuzzy quality/judgment principle, e.g. trackers should never whisper), or "domain" \
  (specific to this conversation domain).
- Keep before/after to short abstract phrases.

Return ONLY JSON: {"lessons": [{"principle": "...", "kind": "structural|stylistic|domain", \
"before": "...", "after": "...", "delta": <number>, "scope": ["all"] or ["<domain>"]}]}"""


def _render_trajectory_for_distill(run_result: Dict[str, Any]) -> str:
    lines = []
    rounds = [r for r in run_result.get("rounds", []) if "error" not in r and r.get("score") is not None]
    for idx, rec in enumerate(rounds):
        rw = rec.get("rewrite")
        if not rw:
            continue
        delta = None
        if idx + 1 < len(rounds):
            delta = round(rounds[idx + 1]["score"] - rec["score"], 1)
        j = rec.get("judgement") or {}
        lines.append(
            f"Round {rec['round']} (score {rec['score']}): changed {rw.get('patched_agents')} — "
            f"{rw.get('rationale')}  => score change {('+' if (delta or 0) >= 0 else '')}{delta}. "
            f"Problems addressed: {j.get('summary', '')[:160]}"
        )
    return "\n".join(lines) or "(no scored changes)"


async def distill(run_result: Dict[str, Any], api_key: Optional[str],
                  model: str = "gpt-4o", distiller=None) -> List[Dict[str, Any]]:
    if distiller is not None:
        return await distiller(run_result)
    if AsyncOpenAI is None or not api_key:
        return []
    traj = _render_trajectory_for_distill(run_result)
    if traj.strip() == "(no scored changes)":
        return []
    client = AsyncOpenAI(api_key=api_key)
    resp = await client.chat.completions.create(
        model=model, temperature=0,
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": _DISTILL_SYSTEM},
                  {"role": "user", "content": f"TRAJECTORY:\n{traj}\n\nExtract the lessons."}],
    )
    try:
        return (json.loads(resp.choices[0].message.content) or {}).get("lessons", [])
    except Exception:
        return []


async def learn_from_run(run_result: Dict[str, Any], api_key: Optional[str],
                         model: str = "gpt-4o", distiller=None) -> Dict[str, Any]:
    """Distill -> consolidate -> persist. Returns a summary of what was learned.
    Never raises into the caller (the optimize run already succeeded)."""
    try:
        proposed = await distill(run_result, api_key, model=model, distiller=distiller)
    except Exception as e:
        return {"error": f"distill failed: {e}", "changes": []}
    store = load_store()
    store, changes = consolidate(store, proposed)
    try:
        save_store(store)
    except Exception as e:
        return {"error": f"save failed: {e}", "changes": changes}
    return {
        "changes": changes,
        "new": [c for c in changes if c["action"] == "new"],
        "reinforced": [c for c in changes if c["action"] == "reinforced"],
        "total_lessons": len(store["lessons"]),
        "active_count": sum(1 for l in store["lessons"] if l.get("status") == "active"),
    }


# =========================================================================
# Human-readable view
# =========================================================================
def render_md(store: Dict[str, Any]) -> str:
    lessons = store.get("lessons", [])
    by_kind = {"structural": [], "stylistic": [], "domain": []}
    for l in lessons:
        by_kind.get(l.get("kind", "stylistic"), by_kind["stylistic"]).append(l)

    def fmt(l):
        ev = l.get("evidence", {})
        sup = ev.get("support", 1)
        avg = ev.get("avg_delta", 0)
        tag = "ACTIVE" if l.get("status") == "active" else "candidate"
        return f"- **{l['principle']}** _(support {sup}, avg +{avg}, {tag})_"

    out = ["# Learned principles",
           "",
           "Auto-distilled from Self-Improve runs. Stylistic lessons auto-inject into the "
           f"generator + optimizer prompts once support ≥ {SUPPORT_THRESHOLD}.",
           ""]
    out.append("## Structural — suggested lint rules (promote to code by hand)")
    out += [fmt(l) for l in by_kind["structural"]] or ["_(none yet)_"]
    out.append("\n## Stylistic — injected into prompts when ACTIVE")
    out += [fmt(l) for l in by_kind["stylistic"]] or ["_(none yet)_"]
    out.append("\n## Domain-specific")
    out += [fmt(l) for l in by_kind["domain"]] or ["_(none yet)_"]
    return "\n".join(out) + "\n"
