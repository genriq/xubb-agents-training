# Xubb Agents Simulator

An interactive sandbox for **learning and stress-testing the [`xubb_agents`](../xubb_agents) framework**.
It replays scripted conversations through a **real `AgentEngine`** and shows you the agent
"whispers" (insights) exactly as a live copilot would surface them — turn by turn — alongside
*why* each agent did or didn't fire.

> This is a **separate project** that *consumes* `xubb_agents` as a library. It writes nothing into
> the framework repo. The simulator's driver doubles as a reference example of how to host the
> framework correctly.

![concept](https://img.shields.io/badge/mode-mock%20%7C%20real-blue) ![status](https://img.shields.io/badge/v1-visualize%20%26%20debug-green)

---

## Why this exists

The whole value of `xubb_agents` is *timely, coordinated, real-time conversational intelligence*.
But the dynamics that make or break a real implementation are invisible in unit tests:

- **Phase-1 → Phase-2 event coordination** (and the extra LLM round-trip it costs)
- **Cooldown gating** — including the subtle trap that *a silent run still consumes the cooldown*
- **Trigger-type routing** — an `event`/`silence`/`interval` agent simply won't fire on a `turn_based` turn
- **Priority** deciding who wins on colliding writes / fact conflicts
- **The Blackboard** (variables, facts, queues, memory) accreting across turns
- **Events being transient** — gone at the end of every turn

This tool makes all of that *visible* so you can answer the real question:
**what is the best way to design and wire agents for this framework?**

---

## Quick start

Requires Python 3.8+. The framework is auto-detected at `../xubb_agents` (the conventional sibling
layout); no install needed.

```bash
cd xubb_agents_simulator
pip install -r requirements.txt        # fastapi + uvicorn (openai/pydantic/jinja2 come with the framework)
python run.py                          # -> http://127.0.0.1:8000
```

Then open the URL, pick a **scenario** + **suite**, press **Load**, and **Step** or **Play**.

### Bundled demos

Five domain pairs ship in `sim/data/`, each chosen to make a different framework behavior visible:

| Scenario | Suite | What it teaches |
|----------|-------|-----------------|
| **Sales discovery call** | Sales copilot | Phase-1→Phase-2 coordination, cooldown gating, facts/queues/memory, a silence turn with no subscriber |
| **Support chat: outage** | Support copilot | High-priority escalation, a **silence**-triggered SLA monitor, sentiment as a pure-function-of-the-prompt, cooldown recovery |
| **Procurement negotiation** | Negotiation copilot | **Priority collisions** on a shared variable, an **interval** BATNA reminder, keyword **allow-list routing** |
| **Mock job interview** | Interview copilot | Memory accumulation (`$inc`), event-driven follow-up prediction, filler-word detection |
| **Team standup** | Standup copilot | Action-item queues, per-speaker memory, an interval/silence time-box monitor |

If the framework isn't a sibling:

```bash
# either install it editable…
pip install -e /path/to/xubb_agents
# …or point at it explicitly
XUBB_AGENTS_PATH=/path/to/xubb_agents python run.py
```

**Mock mode** (default) needs no API key — agent behavior comes from deterministic rules.
**Real mode** uses an OpenAI-compatible key (paste it in the UI or set `OPENAI_API_KEY`) to drive
the agents with an actual LLM.

---

## What you see

| Pane | Shows |
|------|-------|
| **Conversation & Whispers** | The transcript streaming in, with each agent whisper attached to the turn that produced it, color-coded by type. Per-turn chips flag `P1→P2 coordination`, latency, emitted events, cooldown-gated agents, and variable changes. |
| **Turn detail** | Phase-by-phase breakdown: which agents ran, who spoke vs stayed silent, what each wrote (events/facts/vars/queues/memory), who was **cooldown-gated**, who was **skipped** and why, and the **modeled latency** (Phase 1 + Phase 2 = the real cost of coordination). |
| **Blackboard** | The live shared state — variables, facts (with confidence + priority + source), queues, agent-private memory — evolving as you step. |
| **Agents** | The suite roster: triggers, cooldown, priority, keywords, event subscriptions, and a one-line description of what to watch for each. |
| **Suite editor** | Edit the agents + mock rules as JSON and **Apply** to restart the session against the same scenario. This is the loop for "try a design, watch it run." |

---

## How it works

```
 scenario.json (scripted turns)          suite.json (DynamicAgent configs + mock rules)
            │                                          │
            ▼                                          ▼
 ┌───────────────────────────── SimulationSession (a reference host) ──────────────────────────┐
 │  • builds a real xubb_agents.AgentEngine and registers the agents                            │
 │  • per step: sets the virtual clock, builds an AgentContext, calls engine.process_turn(...)  │
 │  • MockLLMClient (per agent) stands in for the OpenAI call — deterministic, rule-based        │
 │  • CaptureTracer (an AgentCallbackHandler) records the full per-turn execution trace          │
 └──────────────────────────────────────────┬───────────────────────────────────────────────────┘
                                             ▼
                              FastAPI  ◄──►  single-page web UI
```

Key design choices (all in `sim/`):

- **`mock_llm.py`** — the mock is a *pure function of the rendered prompt*, exactly like a real LLM.
  Rules match the transcript text (keywords/regex) plus optional live-blackboard predicates. This
  keeps lessons honest: *an agent only reacts to what's in its prompt.*
- **`clock.py`** — cooldowns use `time.time()` in the framework. A fast replay would land every turn
  in the same instant and suppress everything. We point the framework's `agent` module at a
  **virtual clock** driven by scenario timestamps, so cooldowns gate in *scenario seconds*. Surgical
  and reversible; the engine's own latency measurement is untouched.
- **`tracer.py`** — reconstructs phases, per-agent results, emitted events, and skips from the
  framework's own callback stream. Cooldown gating is *inferred* (an agent the engine deemed
  eligible that produced no response was gated — the framework's cooldown check returns before any
  callback fires).
- **`driver.py`** — the host loop. Read it as the canonical "how to integrate `xubb_agents`."

---

## Authoring scenarios & suites

Drop JSON files into `sim/data/scenarios/` and `sim/data/suites/`; they appear in the UI dropdowns.

### Scenario

```jsonc
{
  "name": "Sales discovery call",
  "description": "...",
  "session_id": "sales_demo",
  "window": 12,                       // transcript turns each agent sees
  "user_context": "You are Alex, an AE on a live call…",
  "language_directive": "Respond in English.",
  "steps": [
    { "speaker": "CUSTOMER", "text": "How much does it cost?", "timestamp": 19,
      "note": "A question → Phase-2 Answer Suggester fires" },
    { "trigger": "silence", "silence_duration": 16, "timestamp": 100,
      "note": "dead air — only SILENCE agents would fire" }
  ]
}
```

A step's optional `"trigger"` is one of `turn_based` (default), `keyword`, `silence`, `interval`,
`force`. For `keyword`, the driver runs the framework's `check_keyword_triggers` and restricts the
turn to matched agents (the host's keyword-detection responsibility, demonstrated).

### Suite

Each agent is a normal `DynamicAgent` config plus two simulator-only keys:

- **`_doc`** — a description shown in the Agents tab.
- **`_mock`** — `{ "latency_ms": <int>, "rules": [ … ] }` driving the mock LLM.

```jsonc
{
  "id": "objection_handler",
  "name": "Objection Handler",
  "output_format": "default_v2",
  "trigger_config": { "mode": ["turn_based", "keyword"],
                      "keywords": ["expensive", "discount"], "cooldown": 10, "priority": 5 },
  "text": "Spot price objections and coach the rep.",
  "_doc": "Cooldown 10s on purpose: two objections <10s apart → the 2nd is gated.",
  "_mock": {
    "latency_ms": 320,
    "rules": [
      {
        "when": { "any_keywords": ["expensive", "too much", "discount"] },
        "speak": { "content": "Price objection. Anchor on value, not discount.",
                   "type": "warning", "confidence": 0.9, "action_label": "Reframe value" },
        "facts": [ { "type": "objection", "key": "price", "value": "…", "confidence": 0.8 } ],
        "memory_updates": { "objection_count": { "$inc": 1 } }
      }
    ]
  }
}
```

**Mock rule grammar** (full reference in `sim/mock_llm.py`):

| Section | Fields |
|---------|--------|
| `when` (all AND; omit = always) | `any_keywords`, `all_keywords`, `not_keywords`, `regex`, `speaker`, `min_turn`, `max_turn`, `has_event`, `var_equals`, `has_fact`, `queue_not_empty` |
| `scope` | `last` (default) · `window` · `all` — which text the keyword/regex matches |
| `speak` (omit = silent) | `content`, `type` (`suggestion`/`warning`/`opportunity`/`fact`/`praise`), `confidence`, `action_label`, `expiry` |
| side effects | `events`, `facts`, `variable_updates`, `queue_pushes`, `memory_updates` (`{"$inc": n}` supported) |
| control | `stop` (first matching rule wins; default true) |

`has_event` is how you wire **Phase-2 coordination**: a `mode:"event"` agent with
`subscribed_events:[…]` and a rule `"when": { "has_event": "question_detected" }` fires in Phase 2 of
any turn where a Phase-1 agent emitted that event.

---

## Modes

- **Mock (default):** deterministic, free, reproducible. Best for studying *mechanics*.
- **Real:** drives the actual `DynamicAgent` prompts through OpenAI. Best for validating *prompts*.
  Set the model per agent via `model_config.model`. Latency shown is the real measured latency.

## 🔁 Self-Improvement (automatic prompt optimization)

The **Self-Improve** tab runs a closed optimization loop on the loaded suite:

```
run scenario (real LLM) → AI judge scores it vs the per-turn expectations + anti-spam/
coordination rules → AI rewrites the agent prompts → re-run … keep the best, stop at the
target score or when it plateaus.
```

It exists because **real agent quality lives in the prompt, not the mock rules** — and the loop
optimizes exactly that (each agent's `text`, plus `trigger_conditions`, cooldowns, and event names).
It deterministically catches the classic failure where an emitter and a subscriber disagree on an
event name (so Phase-2 coordination silently dies), and penalizes HUD spam, redundancy, and
off-role whispers.

- **Objective:** judged against each scenario step's authored `note` (ground truth) plus baked-in
  rules (coordination must fire; ≤1–2 whispers/turn; agents stay in their lane; no redundancy).
  Editable in the tab.
- **Output:** a live round-by-round view (score bar, metrics, judge critique, what it rewrote), the
  best suite, and a saved **Markdown report** documenting the whole journey. **Save improved suite**
  writes `<suite>_improved.json` into the dropdown so you can A/B it against the original.
- **Cost:** runs in **real mode** — needs your OpenAI key. Each round ≈ one full scenario run
  (turns × agents LLM calls) + a judge call + a rewrite call. Defaults: target 85, up to 5 rounds
  (stops early), optimizer model `gpt-4o`.

Implementation: [`sim/optimizer.py`](sim/optimizer.py) (`compute_metrics` / `llm_judge` /
`llm_optimize` / `run_self_improvement`); the judge and optimizer are injectable seams so the loop
is testable without an API key.

### It learns across runs (a growing playbook)

Two mechanisms make each run smarter than the last:

- **Trajectory-aware optimizer (OPRO-style):** each round, the optimizer sees the *history* of
  changes and the score each produced (`pinned event name → +22`, `added silence gate → −2`), so it
  builds on what worked and reverts what didn't instead of re-deriving every round. The saved report
  shows these outcome deltas.
- **A distilled learning store** ([`sim/learnings.py`](sim/learnings.py)): after every run, an LLM
  distills *generalizable* lessons from the trajectory, tagged `structural` / `stylistic` / `domain`,
  and consolidates them into [`sim/data/learnings.json`](sim/data/learnings.json) (+ a readable
  `learnings.md`) — deduping near-duplicate phrasings and counting corroboration. Routing:
  - **stylistic** lessons **auto-inject** into the generator + optimizer prompts once **support ≥ 2**
    (corroborated by ≥2 runs), so generation starts from the accumulated playbook;
  - **structural** lessons are surfaced as *suggested lint rules* (you promote them to code by hand —
    deterministic guarantees shouldn't live in a prose file);
  - **domain** lessons are scoped and injected when generating for that domain.

  Opposite principles never merge (negations are preserved in matching), and the support threshold +
  the deterministic-rules-stay-in-code split guard against a self-reinforcing loop drifting.

The Self-Improve panel shows what each run **learned** (new vs reinforced); `GET /api/learnings`
returns the current store.

## 📚 Optimizing against your real data (the production DB)

Drop your production SQLite DB at **`sim/db/xubb.db`** and the **📚 Data** tab unlocks a full pipeline:

1. **Browse real conversations** — search your 350 sessions (`transcript_segments`), preview, and
   **Use as scenario** (a real session → simulator scenario; speaker `YOU` is the coached user).
2. **Generate an agent suite** ([`sim/generator.py`](sim/generator.py)) — describe the copilot you
   want and an LLM writes a starter suite of specialized, silence-disciplined agents with event
   coordination wired in. Optionally seed it with the picked session's transcript + the production
   whispers so it's designed for *your* kind of talk.
3. **Load the combo and Self-Improve** — the optimizer now judges against the **production baseline**:
   it's shown what your *real deployed agents* actually whispered on that conversation
   (`ai_agent_logs`) and is rewarded for being **more useful and less spammy than production**.

The DB is opened **strictly read-only** and is **git-ignored** (290MB of private conversations).
Only the transcript text sent to OpenAI during a real run leaves your machine — the same as your
production app. Real sessions are kept in memory (never written to `data/`).

> **A note on `trigger_conditions`:** they gate on **blackboard state only** (variables/facts/queues),
> never on transcript text or speaker. Per-turn "is this a question / did they use filler" gating
> belongs in the agent's *prompt* (`has_insight=false unless …`). The generator and optimizer are
> taught this so they don't emit conditions the engine would silently ignore.

---

## Project layout

```
xubb_agents_simulator/
├── run.py                       # launcher: python run.py [--port N]
├── smoke_test.py                # headless end-to-end replay (no server)
├── requirements.txt
└── sim/
    ├── framework.py             # locate + import xubb_agents
    ├── clock.py                 # virtual clock for scenario-time cooldowns
    ├── mock_llm.py              # deterministic rule-based LLM stand-in
    ├── tracer.py                # per-turn trace via the framework's callbacks
    ├── driver.py                # SimulationSession — the reference host loop
    ├── server.py                # FastAPI app + API
    ├── static/index.html        # the web sandbox (no build step)
    └── data/
        ├── scenarios/*.json
        └── suites/*.json
```

---

## Lessons it's built to teach

1. **Coordination is shallow and costs latency.** Watch the modeled latency jump on `P1→P2` turns —
   that's a second sequential LLM round-trip. Anything deeper than one hop must wait for the next turn.
2. **Silent runs spend cooldown.** An agent that runs but says nothing still resets its timer, so a
   chatty cooldown can make it miss a later *real* trigger. (See the Objection Handler at turn 6.)
3. **Trigger type is routing, not preference.** An `event` agent is invisible to a `turn_based` turn;
   a `silence` agent needs a silence to exist. Most "my agent never fires" bugs live here.
4. **State is the only thing that crosses turns** — variables/facts/queues/memory persist; events
   don't. Cross-turn coordination is polling durable state, clocked by the conversation.

These mirror the framework's real architectural limits — the simulator just lets you *feel* them.
