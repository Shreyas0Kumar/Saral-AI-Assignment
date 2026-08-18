"""SQLite repository.

Chosen over flat JSONL for three concrete reasons: real transactions, so a delta
that fails halfway does not leave half-applied state; a `GET /health` that can
actually query something; and per-field state that is stored rather than
reconstructed on every run.

Not Postgres, despite INFRA.md proposing Aurora, because a compose file is one
more thing a reviewer has to run. The repository sits behind this class, so the
swap is one adapter -- which is an argument, not a proof, and INFRA.md states it
as one.

`input_hash` is stored on both `raw_profiles` and `signals` so the system can
distinguish "nothing changed" from "never computed", which Appendix A calls out
explicitly.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from saral.contracts.models import ChangeEvent, SignalRecord
from saral.contracts.versions import SIGNALS_VERSION
from saral.core.delta.apply import DeltaState, FieldState
from saral.core.hashing import input_hash

SCHEMA = """
CREATE TABLE IF NOT EXISTS raw_profiles (
    candidate_id TEXT PRIMARY KEY,
    payload      TEXT NOT NULL,
    input_hash   TEXT NOT NULL,
    observed_at  TEXT,
    source       TEXT
);
CREATE TABLE IF NOT EXISTS signals (
    candidate_id    TEXT PRIMARY KEY,
    payload         TEXT NOT NULL,
    input_hash      TEXT NOT NULL,
    signals_version TEXT NOT NULL,
    computed_at     TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS field_state (
    candidate_id    TEXT NOT NULL,
    field_group     TEXT NOT NULL,
    normalized_hash TEXT NOT NULL,
    observed_at     TEXT NOT NULL,
    PRIMARY KEY (candidate_id, field_group)
);
CREATE TABLE IF NOT EXISTS change_events (
    event_id          TEXT PRIMARY KEY,
    candidate_id      TEXT NOT NULL,
    observed_at       TEXT,
    field             TEXT,
    materiality       TEXT,
    signals_recomputed INTEGER,
    payload           TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS runs (
    run_id   TEXT PRIMARY KEY,
    manifest TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_candidate ON change_events(candidate_id);
CREATE INDEX IF NOT EXISTS idx_events_materiality ON change_events(materiality);
"""


class SqliteRepo:
    def __init__(self, path: Path | str) -> None:
        self.path = str(path)
        self._connection = sqlite3.connect(self.path)
        self._connection.row_factory = sqlite3.Row
        self._connection.executescript(SCHEMA)
        self._connection.commit()

    def close(self) -> None:
        self._connection.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """All-or-nothing. A delta that fails halfway rolls back completely."""
        try:
            yield self._connection
            self._connection.commit()
        except Exception:
            self._connection.rollback()
            raise

    # -- raw profiles ------------------------------------------------------
    def upsert_profile(
        self, profile: dict[str, Any], observed_at: str | None = None, source: str | None = None
    ) -> None:
        self._connection.execute(
            "INSERT INTO raw_profiles (candidate_id, payload, input_hash, observed_at, source) "
            "VALUES (?,?,?,?,?) ON CONFLICT(candidate_id) DO UPDATE SET "
            "payload=excluded.payload, input_hash=excluded.input_hash, "
            "observed_at=excluded.observed_at, source=excluded.source",
            (
                profile["id"],
                json.dumps(profile, sort_keys=True, ensure_ascii=False),
                input_hash(profile),
                observed_at,
                source,
            ),
        )

    def load_profiles(self) -> list[dict[str, Any]]:
        rows = self._connection.execute(
            "SELECT payload FROM raw_profiles ORDER BY candidate_id"
        ).fetchall()
        return [json.loads(row["payload"]) for row in rows]

    def profile_hashes(self) -> dict[str, str]:
        rows = self._connection.execute(
            "SELECT candidate_id, input_hash FROM raw_profiles"
        ).fetchall()
        return {row["candidate_id"]: row["input_hash"] for row in rows}

    # -- signals -----------------------------------------------------------
    def upsert_signal(self, record: SignalRecord) -> None:
        self._connection.execute(
            "INSERT INTO signals (candidate_id, payload, input_hash, signals_version, computed_at) "
            "VALUES (?,?,?,?,?) ON CONFLICT(candidate_id) DO UPDATE SET "
            "payload=excluded.payload, input_hash=excluded.input_hash, "
            "signals_version=excluded.signals_version, computed_at=excluded.computed_at",
            (
                record.candidate_id,
                record.model_dump_json(),
                record.input_hash,
                record.signals_version,
                record.computed_at,
            ),
        )

    def load_signals(self) -> list[SignalRecord]:
        rows = self._connection.execute(
            "SELECT payload FROM signals ORDER BY candidate_id"
        ).fetchall()
        return [SignalRecord.model_validate_json(row["payload"]) for row in rows]

    def signal_state(self) -> dict[str, tuple[str, str]]:
        """``{candidate_id: (input_hash, signals_version)}``.

        This is what tells "nothing changed" apart from "never computed": a
        missing key means never computed, a matching hash means nothing changed.
        """
        rows = self._connection.execute(
            "SELECT candidate_id, input_hash, signals_version FROM signals"
        ).fetchall()
        return {r["candidate_id"]: (r["input_hash"], r["signals_version"]) for r in rows}

    def stale_candidates(self, current_hashes: dict[str, str]) -> list[str]:
        stored = self.signal_state()
        out = []
        for candidate_id, hash_value in current_hashes.items():
            entry = stored.get(candidate_id)
            if entry is None or entry[0] != hash_value or entry[1] != SIGNALS_VERSION:
                out.append(candidate_id)
        return sorted(out)

    # -- field state -------------------------------------------------------
    def save_field_state(self, state: dict[tuple[str, str], FieldState]) -> None:
        self._connection.executemany(
            "INSERT INTO field_state (candidate_id, field_group, normalized_hash, observed_at) "
            "VALUES (?,?,?,?) ON CONFLICT(candidate_id, field_group) DO UPDATE SET "
            "normalized_hash=excluded.normalized_hash, observed_at=excluded.observed_at",
            [
                (candidate_id, group, value.normalized_hash, value.observed_at)
                for (candidate_id, group), value in state.items()
            ],
        )

    def load_field_state(self) -> dict[tuple[str, str], FieldState]:
        rows = self._connection.execute("SELECT * FROM field_state").fetchall()
        return {
            (row["candidate_id"], row["field_group"]): FieldState(
                normalized_hash=row["normalized_hash"], observed_at=row["observed_at"]
            )
            for row in rows
        }

    def load_state(self) -> DeltaState:
        return DeltaState(
            profiles={p["id"]: p for p in self.load_profiles()},
            field_state=self.load_field_state(),
        )

    # -- change events -----------------------------------------------------
    def record_events(self, events: list[ChangeEvent]) -> None:
        self._connection.executemany(
            "INSERT OR IGNORE INTO change_events "
            "(event_id, candidate_id, observed_at, field, materiality, signals_recomputed, payload) "
            "VALUES (?,?,?,?,?,?,?)",
            [
                (
                    e.event_id,
                    e.candidate_id,
                    e.observed_at,
                    e.field,
                    e.materiality,
                    int(e.signals_recomputed),
                    e.model_dump_json(),
                )
                for e in events
            ],
        )

    def event_count(self) -> int:
        return self._connection.execute("SELECT COUNT(*) AS n FROM change_events").fetchone()["n"]

    # -- runs / health -----------------------------------------------------
    def record_run(self, run_id: str, manifest: dict) -> None:
        self._connection.execute(
            "INSERT OR REPLACE INTO runs (run_id, manifest) VALUES (?,?)",
            (run_id, json.dumps(manifest)),
        )

    def health(self) -> dict[str, Any]:
        """A 200 that means something.

        Checks the signals table is readable, has rows, and that those rows were
        written by the version of the code currently running. A health check
        that returns 200 unconditionally is a lie with a status code.
        """
        try:
            row = self._connection.execute(
                "SELECT COUNT(*) AS n, "
                "SUM(CASE WHEN signals_version = ? THEN 1 ELSE 0 END) AS current "
                "FROM signals",
                (SIGNALS_VERSION,),
            ).fetchone()
        except sqlite3.Error as exc:
            return {"status": "unhealthy", "reason": f"signals table unreadable: {exc}"}

        total = row["n"] or 0
        current = row["current"] or 0
        if total == 0:
            return {"status": "unhealthy", "reason": "signals table is empty", "signals": 0}
        if current != total:
            return {
                "status": "degraded",
                "reason": (
                    f"{total - current} of {total} signal rows were written by a different "
                    f"signals_version than the running code ({SIGNALS_VERSION}); "
                    "re-run `saral extract`"
                ),
                "signals": total,
                "at_current_version": current,
            }
        return {
            "status": "healthy",
            "signals": total,
            "signals_version": SIGNALS_VERSION,
            "change_events": self.event_count(),
        }
