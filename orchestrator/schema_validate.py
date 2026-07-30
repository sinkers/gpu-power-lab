"""
schema_validate.py — validate a rung summary JSON against the schema.

CLI usage::

    python schema_validate.py path/to/summary.json   # exits 0 if valid, 1 if not

Programmatic usage::

    from schema_validate import validate_summary
    validate_summary(Path("summary.json"))   # raises jsonschema.ValidationError on failure
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import jsonschema
import jsonschema.exceptions

# schema/ lives two levels above orchestrator/schema_validate.py
_SCHEMA_PATH: Path = (
    Path(__file__).resolve().parent.parent / "schema" / "rung-summary.schema.json"
)


def _load_schema() -> dict:
    """Load and return the rung-summary JSON Schema (cached on first call)."""
    return json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))


def validate_summary(path: Path) -> None:
    """
    Load *path* as JSON and validate it against ``schema/rung-summary.schema.json``.

    Raises
    ------
    jsonschema.ValidationError
        When the instance does not conform to the schema.  The message includes
        the failing JSON path and a human-readable description of the violation.
    jsonschema.SchemaError
        When the schema itself is malformed (programming error).
    json.JSONDecodeError
        When *path* cannot be parsed as JSON.
    FileNotFoundError
        When *path* or the schema file does not exist.
    """
    schema = _load_schema()
    instance = json.loads(path.read_text(encoding="utf-8"))

    validator = jsonschema.Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(instance), key=lambda e: list(e.absolute_path))
    if not errors:
        return

    first = errors[0]
    path_str = " → ".join(str(p) for p in first.absolute_path) or "<root>"
    extra = f" (and {len(errors) - 1} more error{'s' if len(errors) > 2 else ''})" if len(errors) > 1 else ""
    raise jsonschema.ValidationError(
        f"Summary {path} failed validation at [{path_str}]: {first.message}{extra}",
        context=first.context,
    )


def main() -> int:
    logging_fmt = "%(levelname)s: %(message)s"
    ap = argparse.ArgumentParser(
        description="Validate a rung summary JSON against rung-summary.schema.json."
    )
    ap.add_argument("path", help="Path to the summary.json file to validate.")
    ns = ap.parse_args()

    target = Path(ns.path)
    if not target.exists():
        print(f"ERROR: file not found: {target}", file=sys.stderr)
        return 1
    if not target.is_file():
        print(f"ERROR: not a file: {target}", file=sys.stderr)
        return 1

    try:
        validate_summary(target)
        print(f"OK: {target} is valid against rung-summary.schema.json")
        return 0
    except json.JSONDecodeError as exc:
        print(f"ERROR: JSON parse failure: {exc}", file=sys.stderr)
        return 1
    except (jsonschema.ValidationError, jsonschema.SchemaError) as exc:
        print(f"INVALID: {exc.message}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
