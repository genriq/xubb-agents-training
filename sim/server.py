"""
FastAPI server for the Xubb Agents Simulator.

Serves the interactive web sandbox and exposes a small API that drives
SimulationSession instances. Mock mode needs no API key; real mode forwards your
OpenAI-compatible key to the framework's LLM client.
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from typing import Any, Dict, List, Optional

# Ensure the framework is importable BEFORE importing the driver (which imports it).
from .framework import ensure_framework_importable

ensure_framework_importable()

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import HTMLResponse, JSONResponse  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from .driver import SimulationSession  # noqa: E402
from .optimizer import run_self_improvement  # noqa: E402
from .generator import generate_suite  # noqa: E402
from . import db as xubb_db  # noqa: E402
from . import learnings  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(HERE, "static")
DATA_DIR = os.path.join(HERE, "data")
SCENARIO_DIR = os.path.join(DATA_DIR, "scenarios")
SUITE_DIR = os.path.join(DATA_DIR, "suites")
REPORT_DIR = os.path.join(DATA_DIR, "reports")
ENV_PATH = os.path.join(os.path.dirname(HERE), ".env")  # project root


def _load_dotenv() -> None:
    """Minimal, dependency-free .env loader (KEY=VALUE per line).

    The .env is THIS app's config store (written by the in-app Save button), so it
    is AUTHORITATIVE: a value in .env overrides a shell-exported one. With no .env
    (or no matching key), a shell-exported var is used as the fallback."""
    if not os.path.exists(ENV_PATH):
        return
    try:
        with open(ENV_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k, v = k.strip(), v.strip().strip('"').strip("'")
                if k:
                    os.environ[k] = v
    except Exception:
        pass


_load_dotenv()

app = FastAPI(title="Xubb Agents Simulator", version="0.1.0")

# In-memory session registry. Single-process; fine for a local dev tool.
# Bounded so a long-lived server doesn't accumulate sessions without limit
# (each holds an engine + a clock-shim reference).
_SESSIONS: Dict[str, SimulationSession] = {}
_MAX_SESSIONS = 50


def _evict_if_full() -> None:
    while len(_SESSIONS) >= _MAX_SESSIONS:
        oldest_id, oldest = next(iter(_SESSIONS.items()))
        _SESSIONS.pop(oldest_id, None)
        try:
            oldest.close()
        except Exception:
            pass


# ----------------------------------------------------------------- helpers
def _list_json(directory: str) -> List[Dict[str, str]]:
    out = []
    if not os.path.isdir(directory):
        return out
    for fn in sorted(os.listdir(directory)):
        if fn.endswith(".json"):
            name = fn[:-5]
            try:
                with open(os.path.join(directory, fn), "r", encoding="utf-8") as f:
                    doc = json.load(f)
                label = doc.get("name", name)
                desc = doc.get("description", "")
            except Exception as e:
                label, desc = name, f"(failed to parse: {e})"
            out.append({"id": name, "name": label, "description": desc})
    return out


def _load_json(directory: str, name: str) -> Dict[str, Any]:
    # Guard against path traversal.
    safe = os.path.basename(name)
    path = os.path.join(directory, safe + ".json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"Not found: {name}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ------------------------------------------------------------------- models
class CreateSessionBody(BaseModel):
    scenario_name: Optional[str] = None
    suite_name: Optional[str] = None
    scenario: Optional[Dict[str, Any]] = None
    suite: Optional[Dict[str, Any]] = None
    mode: str = "mock"
    api_key: Optional[str] = None


class ValidateBody(BaseModel):
    suite: Optional[Dict[str, Any]] = None
    scenario: Optional[Dict[str, Any]] = None


# ------------------------------------------------------------------- routes
@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    index_path = os.path.join(STATIC_DIR, "index.html")
    with open(index_path, "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.get("/api/scenarios")
async def list_scenarios():
    return _list_json(SCENARIO_DIR)


@app.get("/api/suites")
async def list_suites():
    return _list_json(SUITE_DIR)


@app.get("/api/scenario/{name}")
async def get_scenario(name: str):
    return _load_json(SCENARIO_DIR, name)


@app.get("/api/suite/{name}")
async def get_suite(name: str):
    return _load_json(SUITE_DIR, name)


@app.post("/api/validate")
async def validate(body: ValidateBody):
    """Try to build a session from the given suite/scenario; report errors."""
    errors: List[str] = []
    agents_summary: List[Dict[str, Any]] = []
    if body.suite is None or body.scenario is None:
        return {"ok": False, "errors": ["Both suite and scenario are required."]}
    try:
        sess = SimulationSession(suite=body.suite, scenario=body.scenario, mode="mock")
        agents_summary = sess.meta()["agents"]
        sess.close()
    except Exception as e:
        errors.append(f"{type(e).__name__}: {e}")
    return {"ok": not errors, "errors": errors, "agents": agents_summary}


@app.post("/api/session")
async def create_session(body: CreateSessionBody):
    scenario = body.scenario or (
        _load_json(SCENARIO_DIR, body.scenario_name) if body.scenario_name else None
    )
    suite = body.suite or (_load_json(SUITE_DIR, body.suite_name) if body.suite_name else None)
    if scenario is None or suite is None:
        raise HTTPException(status_code=400, detail="scenario and suite are required")

    if body.mode == "real" and not body.api_key and not os.environ.get("OPENAI_API_KEY"):
        raise HTTPException(status_code=400, detail="real mode requires an api_key")

    try:
        session = SimulationSession(
            suite=suite, scenario=scenario, mode=body.mode, api_key=body.api_key
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to build session: {e}")

    _evict_if_full()
    sid = uuid.uuid4().hex[:12]
    _SESSIONS[sid] = session
    return {"session_id": sid, "meta": session.meta(), "finished": session.finished}


def _get_session(sid: str) -> SimulationSession:
    sess = _SESSIONS.get(sid)
    if sess is None:
        raise HTTPException(status_code=404, detail="session not found")
    return sess


@app.post("/api/session/{sid}/step")
async def step(sid: str):
    sess = _get_session(sid)
    trace = await sess.step()
    if trace is None:
        return {"done": True}
    return {"done": False, "trace": trace, "finished": sess.finished}


@app.post("/api/session/{sid}/run")
async def run(sid: str):
    sess = _get_session(sid)
    traces = await sess.run_all()
    return {"traces": traces, "finished": sess.finished}


@app.post("/api/session/{sid}/reset")
async def reset(sid: str):
    sess = _get_session(sid)
    sess.reset()
    return {"ok": True, "finished": sess.finished}


@app.get("/api/session/{sid}")
async def get_session(sid: str):
    sess = _get_session(sid)
    return {
        "meta": sess.meta(),
        "history": sess.history,
        "cursor": sess.cursor,
        "finished": sess.finished,
    }


@app.delete("/api/session/{sid}")
async def delete_session(sid: str):
    sess = _SESSIONS.pop(sid, None)
    if sess is not None:
        sess.close()
    return {"ok": True}


# =========================================================================
# Self-Improvement (prompt optimization loop)
# =========================================================================
_OPT_JOBS: Dict[str, Dict[str, Any]] = {}

DEFAULT_OBJECTIVE = (
    "Optimize for a real-time copilot that whispers RARELY and with high signal. "
    "Judge against each turn's EXPECTED note: the agent the note calls for should fire, "
    "and agents meant to stay silent (detectors/trackers/monitors) must stay silent. "
    "Hard rules: (1) event coordination MUST work — every subscriber must actually fire; "
    "(2) no more than ~1-2 whispers per turn; (3) each agent stays strictly in its lane; "
    "(4) no two agents repeat the same advice in one turn; (5) whispers address the latest turn only."
)


class OptimizeBody(BaseModel):
    session_id: str
    objective: Optional[str] = None
    target_score: int = 85
    max_rounds: int = 5
    optimizer_model: str = "gpt-4o"
    api_key: Optional[str] = None


@app.get("/api/optimize/default-objective")
async def optimize_default_objective():
    return {"objective": DEFAULT_OBJECTIVE}


class KeyBody(BaseModel):
    api_key: Optional[str] = None


@app.get("/api/key-status")
async def key_status():
    """Whether a server-side key is configured. Never returns the key itself."""
    key = os.environ.get("OPENAI_API_KEY")
    return {"has_server_key": bool(key), "hint": (key[:3] + "…" + key[-4:]) if key else None}


@app.post("/api/key")
async def set_key(body: KeyBody):
    """Persist the key to .env (gitignored) and activate it for this process.
    Pass an empty key to clear it."""
    key = (body.api_key or "").strip()
    # Rewrite .env, replacing/removing the OPENAI_API_KEY line.
    lines: List[str] = []
    if os.path.exists(ENV_PATH):
        try:
            with open(ENV_PATH, "r", encoding="utf-8") as f:
                lines = [ln for ln in f.read().splitlines()
                         if not ln.strip().startswith("OPENAI_API_KEY=")]
        except Exception:
            lines = []
    if key:
        lines.append(f"OPENAI_API_KEY={key}")
        os.environ["OPENAI_API_KEY"] = key
    else:
        os.environ.pop("OPENAI_API_KEY", None)
    try:
        with open(ENV_PATH, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"could not write .env: {e}")
    return {"ok": True, "has_server_key": bool(os.environ.get("OPENAI_API_KEY"))}


@app.get("/api/learnings")
async def get_learnings():
    store = learnings.load_store()
    return {
        "lessons": store.get("lessons", []),
        "active": learnings.active_principles(store),
        "md": learnings.render_md(store),
    }


# =========================================================================
# Production DB (real conversations) + agent generation
# =========================================================================
@app.get("/api/db/available")
async def db_available():
    if not xubb_db.available():
        return {"available": False}
    try:
        return {"available": True, "stats": xubb_db.stats()}
    except Exception as e:
        return {"available": False, "error": str(e)}


@app.get("/api/db/sessions")
async def db_sessions(search: str = ""):
    if not xubb_db.available():
        raise HTTPException(status_code=404, detail="no production DB present (sim/db/xubb.db)")
    return xubb_db.list_sessions(search=search or None)


@app.get("/api/db/session/{sid}/scenario")
async def db_session_scenario(sid: str, max_turns: int = 40, offset: int = 0):
    if not xubb_db.available():
        raise HTTPException(status_code=404, detail="no production DB")
    try:
        return xubb_db.session_to_scenario(sid, max_turns=max_turns, offset=offset)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/db/session/{sid}/baseline")
async def db_session_baseline(sid: str):
    if not xubb_db.available():
        raise HTTPException(status_code=404, detail="no production DB")
    return xubb_db.session_baseline(sid)


class GenerateBody(BaseModel):
    goal: str
    api_key: Optional[str] = None
    model: str = "gpt-4o"
    db_session_id: Optional[str] = None
    use_baseline: bool = True


@app.post("/api/generate-suite")
async def generate(body: GenerateBody):
    key = body.api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise HTTPException(status_code=400, detail="agent generation needs an OpenAI API key")
    session_context = baseline_sample = None
    user_speaker = "YOU"  # production transcripts label the user's own words "YOU"
    if body.db_session_id and xubb_db.available():
        try:
            scen = xubb_db.session_to_scenario(body.db_session_id, max_turns=24)
            session_context = "\n".join(f"{s['speaker']}: {s['text']}" for s in scen["steps"])
            user_speaker = scen.get("user_speaker") or user_speaker
        except Exception:
            pass
        if body.use_baseline:
            try:
                base = xubb_db.session_baseline(body.db_session_id, limit=15)
                baseline_sample = "\n".join(
                    f"{b['agent_id']} [{b['insight_type']}]: {b['content']}" for b in base
                )
            except Exception:
                pass
    try:
        suite = await generate_suite(
            body.goal, api_key=key, model=body.model,
            session_context=session_context, baseline_sample=baseline_sample,
            user_speaker=user_speaker,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"generation failed: {e}")
    return {"suite": suite}


@app.post("/api/optimize")
async def optimize_start(body: OptimizeBody):
    sess = _get_session(body.session_id)
    key = body.api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise HTTPException(status_code=400, detail="Self-Improvement needs an OpenAI API key (it runs the agents for real).")

    jid = uuid.uuid4().hex[:12]
    objective = body.objective or DEFAULT_OBJECTIVE
    job: Dict[str, Any] = {
        "id": jid, "status": "running", "round": 0, "max_rounds": body.max_rounds,
        "target": body.target_score, "objective": objective,
        "suite_name": sess.suite.get("name"), "scenario_name": sess.scenario.get("name"),
        "rounds": [], "result": None, "error": None,
    }
    _OPT_JOBS[jid] = job

    def progress_cb(rec: Dict[str, Any]) -> None:
        t = rec.get("type")
        if t == "round_start":
            job["round"] = rec["round"]
        elif t == "round_done":
            job["rounds"].append({
                "round": rec["round"], "score": rec["score"],
                "metrics": rec["metrics"], "judgement": rec["judgement"],
            })
        elif t == "rewrite" and job["rounds"]:
            job["rounds"][-1]["rewrite"] = {
                "patched_agents": rec.get("patched_agents", []),
                "rationale": rec.get("rationale", ""),
            }
        elif t in ("round_error", "rewrite_error"):
            job["rounds"].append({"round": rec.get("round"), "error": rec.get("error")})

    # If the scenario was imported from a real DB session, optimize to BEAT the
    # production whispers (the baseline-to-beat objective).
    baseline = None
    src = (sess.scenario.get("source") or {}).get("db_session_id")
    if src and xubb_db.available():
        try:
            baseline = xubb_db.session_baseline(src)
        except Exception:
            baseline = None
    job["baseline_count"] = len(baseline) if baseline else 0

    async def runner():
        try:
            res = await run_self_improvement(
                suite=sess.suite, scenario=sess.scenario, api_key=key,
                objective=objective, target_score=body.target_score,
                max_rounds=body.max_rounds, optimizer_model=body.optimizer_model,
                run_mode="real", baseline=baseline, progress_cb=progress_cb,
            )
            job["result"] = {
                "best_score": res["best"]["score"], "best_round": res["best"]["round"],
                "total_rounds": len(res["rounds"]), "report_md": res["report_md"],
                "improved_suite": res["best"]["suite"],
            }
            # Step (b): auto-distill generalizable lessons and fold them into the
            # store so the next generation/optimization starts smarter.
            try:
                job["result"]["learnings"] = await learnings.learn_from_run(
                    {"rounds": res["rounds"]}, api_key=key, model=body.optimizer_model
                )
            except Exception as e:
                job["result"]["learnings"] = {"error": str(e), "changes": []}
            job["status"] = "done"
        except Exception as e:
            job["error"] = str(e)
            job["status"] = "error"

    asyncio.create_task(runner())
    return {"job_id": jid}


@app.get("/api/optimize/{jid}")
async def optimize_status(jid: str):
    job = _OPT_JOBS.get(jid)
    if job is None:
        raise HTTPException(status_code=404, detail="optimize job not found")
    return job


@app.post("/api/optimize/{jid}/save")
async def optimize_save(jid: str):
    job = _OPT_JOBS.get(jid)
    if job is None or not job.get("result"):
        raise HTTPException(status_code=400, detail="no completed result to save")
    suite = dict(job["result"]["improved_suite"])
    name = suite.get("name", "Suite")
    if not name.rstrip().endswith("(improved)"):
        name = name + " (improved)"
    suite["name"] = name
    base = (name.lower().replace(" (improved)", "").replace(" ", "_").replace("/", "_"))[:40]
    suite_id = f"{base}_improved"
    with open(os.path.join(SUITE_DIR, suite_id + ".json"), "w", encoding="utf-8") as f:
        json.dump(suite, f, indent=2)
    os.makedirs(REPORT_DIR, exist_ok=True)
    report_path = os.path.join(REPORT_DIR, suite_id + "_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(job["result"]["report_md"])
    return {"ok": True, "suite_id": suite_id, "report": os.path.basename(report_path)}
