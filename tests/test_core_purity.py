"""`core/` is pure. This test is what makes that true rather than aspirational.

Purity buys three things:

* Part 3 idempotency is *provable*. Identical input produces byte-identical
  output because there is no clock inside the extractor to move underneath it.
* The four evaluation arms are swappable with no branching, because nothing in
  `core/` knows where its data came from.
* The whole suite runs with no fixtures, no network, no model download.

Enforced by a test rather than by discipline, because under time pressure in the
last hours discipline loses.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

CORE = Path(__file__).resolve().parents[1] / "src" / "saral" / "core"

FORBIDDEN_MODULES = {
    "sqlite3",
    "requests",
    "httpx",
    "urllib",
    "fastapi",
    "torch",
    "sentence_transformers",
    "transformers",
    "ollama",
    "boto3",
    "joblib",
    "yaml",
    "os",
    "pathlib",
}

# `datetime` itself is allowed (the `date` type is a value object); reading the
# clock is not.
FORBIDDEN_CALLS = {"now", "utcnow", "today", "time", "monotonic", "perf_counter"}
FORBIDDEN_CALL_ROOTS = {"datetime", "date", "time"}


def _core_files() -> list[Path]:
    return sorted(CORE.rglob("*.py"))


def test_core_has_files():
    assert _core_files(), "no files found under src/saral/core"


@pytest.mark.parametrize("path", _core_files(), ids=lambda p: p.name)
def test_core_imports_nothing_impure(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        modules: set[str] = set()
        if isinstance(node, ast.Import):
            modules = {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                modules = {node.module.split(".")[0]}
        bad = modules & FORBIDDEN_MODULES
        assert not bad, f"{path.relative_to(CORE)} imports {sorted(bad)}"


@pytest.mark.parametrize("path", _core_files(), ids=lambda p: p.name)
def test_core_reads_no_clock(path: Path):
    """`computed_at` and `as_of` are injected. Nothing in core reads the clock."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr in FORBIDDEN_CALLS:
            root = func.value
            root_name = getattr(root, "id", None) or getattr(
                getattr(root, "value", None), "id", None
            )
            if root_name in FORBIDDEN_CALL_ROOTS:
                raise AssertionError(
                    f"{path.relative_to(CORE)} reads the clock: {root_name}.{func.attr}()"
                )
