"""The run manifest.

Everything needed to reproduce or dispute a number in ``out/metrics.json``:
the git sha, the exact machine, the config hashes, per-stage throughput, and the
derived cost per million profiles. The cost figure is computed here from
measured throughput -- never by hand in a markdown file.
"""

from __future__ import annotations

import json
import platform
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from saral.contracts.versions import LEXICON_VERSION, SCORING_VERSION, SIGNALS_VERSION
from saral.telemetry.stages import Telemetry

#: ECS Fargate on-demand, ap-south-1 (Mumbai), Linux/X86, as of 2026-08.
#: Used only to turn measured CPU-seconds into a currency figure; the
#: arithmetic is shown in INFRA.md so the rate can be substituted.
FARGATE_VCPU_HOUR_USD = 0.04656
FARGATE_GB_HOUR_USD = 0.00511


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def machine_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "processor": platform.processor() or platform.machine(),
    }
    try:
        import psutil

        info["cores_physical"] = psutil.cpu_count(logical=False)
        info["cores_logical"] = psutil.cpu_count(logical=True)
        info["ram_gb"] = round(psutil.virtual_memory().total / 1e9, 1)
    except Exception:  # pragma: no cover
        pass
    return info


@dataclass
class RunManifest:
    run_id: str
    arm: str
    telemetry: Telemetry
    config_hashes: dict[str, str] = field(default_factory=dict)
    models: dict[str, str] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def start(cls, arm: str, telemetry: Telemetry, **kwargs: Any) -> "RunManifest":
        return cls(
            run_id=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            arm=arm,
            telemetry=telemetry,
            **kwargs,
        )

    def cost_per_1m_profiles(self, stage_name: str = "extract") -> dict[str, Any] | None:
        """Derive Fargate cost for one full pass over 1M profiles.

        Deliberately single-threaded arithmetic: the measured stage is
        single-threaded, so scaling it by vCPU-hours is the honest conversion.
        """
        stage = self.telemetry.stages.get(stage_name)
        if not stage or not stage.records_out or stage.wall_ms <= 0:
            return None
        ms_per_record = stage.wall_ms / stage.records_out
        cpu_hours = (ms_per_record * 1_000_000) / 1000 / 3600
        gb = max(0.5, (stage.peak_rss_mb or 512) / 1024)
        usd = cpu_hours * FARGATE_VCPU_HOUR_USD + cpu_hours * gb * FARGATE_GB_HOUR_USD
        return {
            "stage": stage_name,
            "ms_per_record": round(ms_per_record, 4),
            "cpu_hours_per_1m": round(cpu_hours, 4),
            "assumed_gb": round(gb, 2),
            "usd_per_1m": round(usd, 4),
            "rate_note": (
                f"fargate ap-south-1 on-demand ${FARGATE_VCPU_HOUR_USD}/vCPU-hr "
                f"+ ${FARGATE_GB_HOUR_USD}/GB-hr"
            ),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "git_sha": _git_sha(),
            "arm": self.arm,
            "versions": {
                "signals": SIGNALS_VERSION,
                "scoring": SCORING_VERSION,
                "lexicon": LEXICON_VERSION,
            },
            "machine": machine_info(),
            "models": self.models,
            "config_hashes": self.config_hashes,
            "stages": self.telemetry.to_dict(),
            "derived": {"cost_per_1m_profiles": self.cost_per_1m_profiles()},
            **self.extra,
        }

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
