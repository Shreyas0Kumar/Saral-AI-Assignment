"""Stage timing, memory and counters.

Built in Phase 1 rather than bolted on at the end, because the dashboard, the
cost-per-million arithmetic in ``INFRA.md`` and the incremental-saving report in
Part 3 all read from here. A number that is calculated by hand in a document is
a number nobody can reproduce.
"""

from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass, field
from typing import Iterator

try:  # psutil is optional; RSS is reported as None without it.
    import psutil

    _PROCESS = psutil.Process()
except Exception:  # pragma: no cover
    _PROCESS = None


def _rss_mb() -> float | None:
    if _PROCESS is None:
        return None
    try:
        return _PROCESS.memory_info().rss / (1024 * 1024)
    except Exception:  # pragma: no cover
        return None


@dataclass
class StageReport:
    name: str
    records_in: int = 0
    records_out: int = 0
    wall_ms: float = 0.0
    rss_start_mb: float | None = None
    rss_end_mb: float | None = None
    peak_rss_mb: float | None = None
    counters: dict[str, float] = field(default_factory=dict)

    def count(self, key: str, value: float = 1) -> None:
        self.counters[key] = self.counters.get(key, 0) + value

    def to_dict(self) -> dict:
        return {
            "records_in": self.records_in,
            "records_out": self.records_out,
            "wall_ms": round(self.wall_ms, 3),
            "peak_rss_mb": round(self.peak_rss_mb, 1) if self.peak_rss_mb else None,
            "rss_delta_mb": (
                round(self.rss_end_mb - self.rss_start_mb, 1)
                if self.rss_end_mb is not None and self.rss_start_mb is not None
                else None
            ),
            "counters": {k: round(v, 4) for k, v in sorted(self.counters.items())},
        }


class Telemetry:
    """Collects stage reports for one run."""

    def __init__(self) -> None:
        self.stages: dict[str, StageReport] = {}

    @contextlib.contextmanager
    def stage(self, name: str, records_in: int = 0) -> Iterator[StageReport]:
        report = self.stages.get(name)
        if report is None:
            report = StageReport(name=name)
            self.stages[name] = report
        report.records_in += records_in
        report.rss_start_mb = report.rss_start_mb if report.rss_start_mb is not None else _rss_mb()
        start = time.perf_counter()
        try:
            yield report
        finally:
            report.wall_ms += (time.perf_counter() - start) * 1000
            report.rss_end_mb = _rss_mb()
            if report.rss_end_mb is not None:
                report.peak_rss_mb = max(report.peak_rss_mb or 0.0, report.rss_end_mb)

    def to_dict(self) -> dict:
        return {name: report.to_dict() for name, report in self.stages.items()}


#: Process-wide default. Passed explicitly where it matters; this exists so the
#: LLM adapter can increment token counters without threading a handle through
#: five layers of call stack.
TELEMETRY = Telemetry()
