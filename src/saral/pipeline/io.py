"""Filesystem I/O for the pipeline. Deliberately dull."""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

from saral.contracts.models import JobSpec
from saral.core.dates import derive_snapshot_date

ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "out"
CONFIG_DIR = ROOT / "config"

#: Frozen so `make all` is reproducible: `computed_at` must not be the wall
#: clock, or two runs over identical input produce different bytes and the
#: idempotency test has to grow an exclusion list. Overridable via --computed-at.
DEFAULT_COMPUTED_AT = datetime(2026, 8, 17, 10, 3, 11, tzinfo=timezone.utc)


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            text = line.strip()
            if not text:
                continue
            try:
                yield json.loads(text)
            except json.JSONDecodeError as exc:
                # A malformed line on row 7 of a million must not kill the batch.
                print(f"[warn] {path.name}:{line_no} unparseable JSON, skipped: {exc}")


def write_jsonl(path: Path, records: Iterable[Any]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            payload = record if isinstance(record, dict) else record.model_dump(mode="json")
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            count += 1
    return count


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_candidates(path: Path | None = None) -> list[dict[str, Any]]:
    return list(read_jsonl(path or DATA_DIR / "candidates.jsonl"))


def load_delta(path: Path | None = None) -> list[dict[str, Any]]:
    return list(read_jsonl(path or DATA_DIR / "candidates_delta.jsonl"))


def load_jobs(path: Path | None = None) -> list[JobSpec]:
    raw = json.loads((path or DATA_DIR / "jobs.json").read_text(encoding="utf-8"))
    return [JobSpec.model_validate(job) for job in raw]


def load_labels(path: Path | None = None) -> list[dict[str, Any]]:
    import csv

    with (path or DATA_DIR / "labels.csv").open("r", encoding="utf-8", newline="") as handle:
        return [
            {
                "job_id": row["job_id"],
                "candidate_id": row["candidate_id"],
                "relevance": int(row["relevance"]),
            }
            for row in csv.DictReader(handle)
        ]


def corpus_as_of(profiles: list[dict[str, Any]], fallback: date | None = None) -> date:
    """The date the corpus was crawled, derived from the data. See dates.py."""
    return derive_snapshot_date(profiles, fallback or DEFAULT_COMPUTED_AT.date())
