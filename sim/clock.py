"""
Virtual clock for deterministic replay.

`BaseAgent.process()` in the framework enforces cooldowns with `time.time()`
(wall-clock). If we replayed a scenario at full speed, every turn would land in
the same wall-clock instant and cooldowns (default 10-15s) would suppress every
agent after turn 1 — making cooldown behavior impossible to study.

The fix: drive cooldowns from *scenario time*. The scenario's segment timestamps
already are "seconds since session start", which is exactly the clock cooldowns
should compare against. We surgically replace the `time` reference in the
framework's `agent` module namespace with a shim whose `.time()` returns our
virtual clock. This affects ONLY `core/agent.py` cooldown math — `core/engine.py`
imports `time` independently, so its real-wall-clock latency measurement is
untouched.

Concurrency-safe design (important: the server holds many sessions at once):
- The shim is installed **once**, process-wide, reference-counted. We never let a
  per-session patch capture another session's shim as the "original".
- The active clock is resolved per async context from a `ContextVar`. Each
  session sets its own `VirtualClock` into the ContextVar for the duration of one
  turn (`with ActiveClock(clock): await engine.process_turn(...)`). Tasks spawned
  by `asyncio.gather` inside `process_turn` inherit the ContextVar, so every agent
  reads the clock of the session that is currently stepping — and two sessions
  stepping concurrently stay isolated.

This is the same well-understood technique `freezegun` uses, made multi-session
safe. Fully contained to the simulator and reversible.
"""

from __future__ import annotations

import contextvars
import time as _real_time
from typing import Optional

# The clock active for the current async context (None => use real wall-clock).
_active_clock: "contextvars.ContextVar[Optional[VirtualClock]]" = contextvars.ContextVar(
    "xubb_sim_active_clock", default=None
)


class VirtualClock:
    """A mutable virtual clock. `now` is "seconds since session start"."""

    def __init__(self, start: float = 0.0):
        self.now: float = float(start)

    def set(self, seconds: float) -> None:
        self.now = float(seconds)

    def time(self) -> float:  # matches time.time() signature used by the framework
        return self.now


class _DispatchShim:
    """Process-global stand-in for the `time` module inside `core/agent.py`.

    Resolves the active VirtualClock from the ContextVar on every call; falls
    back to real wall-clock when no session is stepping. Delegates everything
    else (sleep, monotonic, perf_counter, …) to the real `time` module.
    """

    def time(self) -> float:
        clk = _active_clock.get()
        return clk.time() if clk is not None else _real_time.time()

    def __getattr__(self, name):
        return getattr(_real_time, name)


# --- Reference-counted, process-wide install of the single shim ----------------
_install_count = 0
_original_time = None
_agent_module = None


def install_clock_shim() -> None:
    """Install the dispatch shim into `xubb_agents.core.agent` (idempotent)."""
    global _install_count, _original_time, _agent_module
    if _install_count == 0:
        from xubb_agents.core import agent as agent_module

        _agent_module = agent_module
        _original_time = agent_module.time  # the real `time` module, captured once
        agent_module.time = _DispatchShim()
    _install_count += 1


def uninstall_clock_shim() -> None:
    """Drop one install reference; restore the real `time` module at zero."""
    global _install_count, _original_time, _agent_module
    if _install_count <= 0:
        return
    _install_count -= 1
    if _install_count == 0 and _agent_module is not None:
        _agent_module.time = _original_time
        _original_time = None
        _agent_module = None


class ActiveClock:
    """Context manager: make `clock` the active clock for the current async context."""

    def __init__(self, clock: VirtualClock):
        self.clock = clock
        self._token = None

    def __enter__(self) -> "ActiveClock":
        self._token = _active_clock.set(self.clock)
        return self

    def __exit__(self, *exc) -> None:
        if self._token is not None:
            _active_clock.reset(self._token)
            self._token = None
