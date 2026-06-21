#!/usr/bin/env python
"""Headless end-to-end check: replay the summary-agent demo (mock mode), print the trace.

Mock mode is kept for this internal headless check only — the app UI always runs real."""

import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sim.framework import ensure_framework_importable

ensure_framework_importable()
from sim.driver import SimulationSession  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def load(rel):
    with open(os.path.join(HERE, "sim", "data", rel), "r", encoding="utf-8") as f:
        return json.load(f)


async def main():
    suite = load("suites/summary_demo_suite.json")
    scenario = load("scenarios/summary_demo.json")
    sess = SimulationSession(suite=suite, scenario=scenario, mode="mock")
    print(f"AGENTS: {[a['name'] for a in sess.meta()['agents']]}\n")

    traces = await sess.run_all()
    for t in traces:
        seg = t["segment"]
        head = f"[turn {t['turn_index']} | t={t['sim_time']}s | {t['trigger_type']}]"
        if seg:
            print(f"{head} {seg['speaker']}: {seg['text']}")
        else:
            print(f"{head} (no speech)")
        for ins in t["insights"]:
            print(f"    >> WHISPER [{ins['type']}] ({ins['agent_name']}): {ins['content']}")
        # phase / coordination view
        for ph in t["phases"]:
            ran = [a["agent"] + ("*" if a.get("spoke") else "") for a in ph["agents"]]
            ev = ph["events_emitted"]
            extra = f"  events={ev}" if ev else ""
            if ran:
                print(f"    phase {ph['phase']}: ran {ran}{extra}")
        if t["skipped"]:
            sk = [f"{s['agent']}({s['reason']})" for s in t["skipped"]]
            print(f"    skipped: {sk}")
        d = t["blackboard_delta"]
        if d["variables"] or d["facts_added"] or d["queues"] or d["memory"]:
            print(f"    delta: vars={d['variables']} facts+={[f['type'] for f in d['facts_added']]} "
                  f"queues={d['queues']} mem={[m['agent_id'] for m in d['memory']]}")
        print(f"    modeled_latency={t['modeled_latency_ms']}ms "
              f"(phases={[p['modeled_latency_ms'] for p in t['phase_latencies']]})")
        print()

    print("FINAL BLACKBOARD:")
    print(json.dumps(traces[-1]["blackboard_after"], indent=2, default=str))
    sess.close()


if __name__ == "__main__":
    asyncio.run(main())
