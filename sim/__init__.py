"""
Xubb Agents Simulator
=====================

A standalone host application that *consumes* the `xubb_agents` framework to
replay scripted conversations through a real `AgentEngine` and visualize the
agent "whispers" (insights) it produces, turn by turn.

This package intentionally lives OUTSIDE the framework repo: it is a reference
host, and the cleanest way to learn the framework is to drive it exactly the way
a production host (e.g. `xubb_server`) would.

Modules:
- framework : locates and imports the `xubb_agents` library
- clock     : virtual clock so cooldowns operate in scenario-time, not wall-clock
- mock_llm  : deterministic rule-based stand-in for the OpenAI LLM client
- tracer    : an AgentCallbackHandler that captures a full per-turn execution trace
- driver    : SimulationSession — builds the engine, replays a scenario, returns traces
- server    : FastAPI app + the interactive web sandbox
"""

__version__ = "0.1.0"
