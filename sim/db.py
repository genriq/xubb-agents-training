"""
Read-only access to the production Xubb database (`sim/db/xubb.db`).

This is the user's REAL conversation data — it is opened strictly read-only (we
never risk corrupting the live app DB), and it must never be committed to git
(see .gitignore). It provides:

- real sessions to use as optimization scenarios (transcript_segments → steps),
- the production whispers per session (ai_agent_logs) as the "baseline to beat",
- the real prompts (the production agent suite), for reference.
"""

from __future__ import annotations

import json
import os
import pathlib
import sqlite3
from typing import Any, Dict, List, Optional

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "db", "xubb.db")


def available() -> bool:
    return os.path.exists(DB_PATH)


def _connect() -> sqlite3.Connection:
    if not available():
        raise FileNotFoundError(
            f"Production DB not found at {DB_PATH}. Place xubb.db in sim/db/."
        )
    uri = pathlib.Path(DB_PATH).as_uri() + "?mode=ro"
    con = sqlite3.connect(uri, uri=True, timeout=5)
    con.row_factory = sqlite3.Row
    return con


def _loads(s: Any) -> Any:
    if not s:
        return None
    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError):
        return s


# =========================================================================
# Sessions
# =========================================================================
def list_sessions(search: Optional[str] = None, limit: int = 60) -> List[Dict[str, Any]]:
    """Browse sessions that actually have transcript segments."""
    con = _connect()
    try:
        cur = con.cursor()
        # segment counts per session
        counts = {
            r["session_id"]: r["c"]
            for r in cur.execute(
                "SELECT session_id, COUNT(*) c FROM transcript_segments GROUP BY session_id"
            )
        }
        params: List[Any] = []
        where = "WHERE (is_deleted IS NULL OR is_deleted = 0)"
        if search:
            where += " AND (session_name LIKE ? OR classification LIKE ? OR summary LIKE ?)"
            like = f"%{search}%"
            params += [like, like, like]
        rows = cur.execute(
            f"""SELECT session_id, session_name, classification, created_timestamp,
                       duration_seconds, word_count, active_agents, ai_language, summary
                FROM sessions {where}
                ORDER BY created_timestamp DESC LIMIT 400""",
            params,
        ).fetchall()
        out = []
        for r in rows:
            segs = counts.get(r["session_id"], 0)
            if segs <= 0:
                continue
            out.append({
                "session_id": r["session_id"],
                "name": r["session_name"] or "(untitled)",
                "classification": r["classification"] or "",
                "created": r["created_timestamp"],
                "segments": segs,
                "words": r["word_count"] or 0,
                "duration_s": r["duration_seconds"] or 0,
                "language": r["ai_language"] or "en",
                "active_agents": _loads(r["active_agents"]) or [],
                "summary": (r["summary"] or "")[:240],
            })
            if len(out) >= limit:
                break
        return out
    finally:
        con.close()


def session_to_scenario(
    session_id: str, max_turns: int = 40, offset: int = 0, final_only: bool = True
) -> Dict[str, Any]:
    """Convert a real session's transcript into a simulator scenario."""
    con = _connect()
    try:
        cur = con.cursor()
        s = cur.execute(
            """SELECT session_name, classification, context_information, ai_language, summary
               FROM sessions WHERE session_id = ?""",
            (session_id,),
        ).fetchone()
        if s is None:
            raise ValueError(f"session {session_id} not found")

        final_clause = "AND (is_final = 1 OR is_final IS NULL)" if final_only else ""
        seg_rows = cur.execute(
            f"""SELECT speaker, start_time, text_content, is_final
                FROM transcript_segments
                WHERE session_id = ? AND text_content IS NOT NULL AND length(trim(text_content)) > 0
                {final_clause}
                ORDER BY start_time
                LIMIT ? OFFSET ?""",
            (session_id, max_turns, offset),
        ).fetchall()

        steps = []
        for r in seg_rows:
            steps.append({
                "speaker": r["speaker"] or "SPEAKER",
                "text": r["text_content"].strip(),
                "timestamp": float(r["start_time"] or 0.0),
                "is_final": bool(r["is_final"]) if r["is_final"] is not None else True,
            })

        lang = (s["ai_language"] or "en")
        name = s["session_name"] or "Imported session"
        return {
            "name": f"{name} (real)",
            "description": f"Imported real session{' — ' + s['classification'] if s['classification'] else ''}. "
                           f"{(s['summary'] or '')[:200]}",
            "session_id": "db_" + session_id[:8],
            "window": 12,
            "user_speaker": "YOU",
            "user_context": (s["context_information"] or "")[:500] or None,
            "language_directive": f"Respond in {'English' if lang.startswith('en') else lang}.",
            "source": {"db_session_id": session_id},
            "steps": steps,
        }
    finally:
        con.close()


def session_baseline(session_id: str, limit: int = 300) -> List[Dict[str, Any]]:
    """The production whispers for a session (the 'baseline to beat')."""
    con = _connect()
    try:
        cur = con.cursor()
        rows = cur.execute(
            """SELECT agent_id, insight_type, trigger_type, content, created_at
               FROM ai_agent_logs
               WHERE session_id = ? AND content IS NOT NULL AND length(trim(content)) > 0
               ORDER BY created_at LIMIT ?""",
            (session_id, limit),
        ).fetchall()
        return [{
            "agent_id": r["agent_id"], "insight_type": r["insight_type"],
            "trigger_type": r["trigger_type"], "content": r["content"],
        } for r in rows]
    finally:
        con.close()


# =========================================================================
# Real prompts (production agent suite, for reference / import)
# =========================================================================
def list_prompts() -> List[Dict[str, Any]]:
    con = _connect()
    try:
        cur = con.cursor()
        rows = cur.execute(
            """SELECT id, name, description, text, trigger_config, model_config,
                      output_format, type
               FROM prompts WHERE text IS NOT NULL AND length(trim(text)) > 0
               ORDER BY name"""
        ).fetchall()
        return [{
            "id": str(r["id"]), "name": r["name"], "description": r["description"] or "",
            "text": r["text"], "trigger_config": _loads(r["trigger_config"]) or {},
            "model_config": _loads(r["model_config"]) or {},
            "output_format": r["output_format"] or "default", "type": r["type"] or "agent",
        } for r in rows]
    finally:
        con.close()


def stats() -> Dict[str, Any]:
    con = _connect()
    try:
        cur = con.cursor()
        def one(q):
            return cur.execute(q).fetchone()[0]
        return {
            "sessions": one("SELECT COUNT(*) FROM sessions WHERE is_deleted IS NULL OR is_deleted=0"),
            "segments": one("SELECT COUNT(*) FROM transcript_segments"),
            "whispers": one("SELECT COUNT(*) FROM ai_agent_logs"),
            "prompts": one("SELECT COUNT(*) FROM prompts WHERE text IS NOT NULL"),
        }
    finally:
        con.close()
