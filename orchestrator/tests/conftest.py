"""
conftest.py — ensure the orchestrator/ directory is on sys.path so that
``upload``, ``schema_validate``, ``campaign``, ``plan``, etc. are importable
when pytest is invoked from orchestrator/ or from the project root.
"""

from __future__ import annotations

import sys
from pathlib import Path

# orchestrator/ is the parent of tests/
_ORCHESTRATOR_DIR = Path(__file__).resolve().parent.parent
if str(_ORCHESTRATOR_DIR) not in sys.path:
    sys.path.insert(0, str(_ORCHESTRATOR_DIR))
