"""
Locate and import the `xubb_agents` framework.

The simulator is a separate project, so `xubb_agents` is an external dependency.
We resolve it in this order:

1. Already importable (e.g. you ran `pip install -e ../xubb_agents`).
2. The `XUBB_AGENTS_PATH` environment variable, pointing at the framework folder
   (the directory that contains `__init__.py`, i.e. the package directory itself).
3. The conventional sibling layout: `.../Projects/xubb_agents` next to
   `.../Projects/xubb_agents_simulator`.

To import a package by folder, the *parent* of the package directory must be on
`sys.path` (so `import xubb_agents` finds `<parent>/xubb_agents/__init__.py`).
"""

from __future__ import annotations

import os
import sys
from typing import List, Optional


def _is_package_dir(path: str) -> bool:
    return os.path.isdir(path) and os.path.exists(os.path.join(path, "__init__.py"))


def _add_parent_to_path(package_dir: str) -> None:
    parent = os.path.dirname(os.path.abspath(package_dir))
    if parent not in sys.path:
        sys.path.insert(0, parent)


def ensure_framework_importable(explicit_path: Optional[str] = None) -> str:
    """Make `import xubb_agents` succeed and return the resolved package path.

    Raises ImportError with an actionable message if the framework cannot be
    found anywhere.
    """
    # 1. Fast path: already importable.
    try:
        import xubb_agents  # noqa: F401

        return os.path.dirname(os.path.abspath(xubb_agents.__file__))
    except ImportError:
        pass

    candidates: List[str] = []
    if explicit_path:
        candidates.append(explicit_path)
    env_path = os.environ.get("XUBB_AGENTS_PATH")
    if env_path:
        candidates.append(env_path)

    # Conventional sibling layout.
    simulator_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    projects_dir = os.path.dirname(simulator_root)
    candidates.append(os.path.join(projects_dir, "xubb_agents"))

    for package_dir in candidates:
        if _is_package_dir(package_dir):
            _add_parent_to_path(package_dir)
            try:
                import xubb_agents  # noqa: F401

                return os.path.dirname(os.path.abspath(xubb_agents.__file__))
            except ImportError:
                continue

    tried = "\n  - ".join(candidates) or "(none)"
    raise ImportError(
        "Could not locate the `xubb_agents` framework. Tried:\n  - "
        + tried
        + "\n\nFix one of:\n"
        "  * pip install -e <path-to-xubb_agents>\n"
        "  * set XUBB_AGENTS_PATH to the framework folder (the one with __init__.py)\n"
        "  * place the framework at ../xubb_agents next to this simulator."
    )
