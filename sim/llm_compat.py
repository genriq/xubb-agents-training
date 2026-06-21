"""Compatibility shim for OpenAI chat-completions across model families.

Newer models (gpt-5.x, the o-series reasoning models) reject any non-default
sampling param — e.g. `temperature` may only be the default (1). Older models
(gpt-4o, gpt-4.1) accept `temperature=0` for determinism. Rather than hardcode
which is which, we send the caller's preferred params and, on a 400 that names an
offending param, drop just that param and retry. So one generator/optimizer call
works whether the user picks gpt-4o or gpt-5.5 from the model dropdown.
"""
from __future__ import annotations

import json
from typing import Any, Dict

# Sampling params a model may reject with an "unsupported_value" 400 — safe to drop
# (the model falls back to its default), unlike structural params (messages, format).
_DROPPABLE = ("temperature", "top_p", "frequency_penalty", "presence_penalty")


async def create_json(client, **params) -> Dict[str, Any]:
    """`client.chat.completions.create` returning parsed JSON, tolerant of models
    that only allow default sampling params. On a 400 naming a droppable param we're
    sending, drop it and retry; terminates once no offending param remains."""
    params.setdefault("response_format", {"type": "json_object"})
    while True:
        try:
            resp = await client.chat.completions.create(**params)
            return json.loads(resp.choices[0].message.content)
        except Exception as e:                       # noqa: BLE001 — inspect + maybe retry
            if not _drop_unsupported_param(params, e):
                raise


def _drop_unsupported_param(params: dict, err: Exception) -> bool:
    """If `err` is an 'unsupported param' 400 for a droppable param we're sending,
    remove it in place and return True (caller retries); else False."""
    msg = str(err).lower()
    if "unsupported" not in msg and "does not support" not in msg:
        return False
    for p in _DROPPABLE:
        if p in params and p in msg:
            params.pop(p, None)
            return True
    return False
