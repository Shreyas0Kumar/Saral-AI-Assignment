"""`saral` command line entry point.

    saral extract     raw profiles      -> out/candidate_signals.jsonl
    saral score       signals + jobs    -> out/rankings.jsonl
    saral evaluate    rankings + labels -> out/metrics.json
    saral delta       delta feed        -> out/change_events.jsonl
    saral all         everything, in order
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path

from saral.config_loader import load_all
from saral.pipeline import io
from saral.pipeline.arms import ARMS, SHIPPED_ARM
from saral.pipeline.extract_pass import run_extract, write_signals
from saral.pipeline.score_pass import run_scoring, write_rankings
from saral.telemetry.manifest import RunManifest
from saral.telemetry.stages import Telemetry


def _parse_date(value: str | None) -> date | None:
    return datetime.strptime(value, "%Y-%m-%d").date() if value else None


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def cmd_extract(args) -> int:
    profiles = io.load_candidates(Path(args.candidates) if args.candidates else None)
    records, _traces, telemetry = run_extract(
        profiles,
        arm=args.arm,
        computed_at=_parse_dt(args.computed_at),
        as_of=_parse_date(args.as_of),
    )
    count = write_signals(records, Path(args.out) if args.out else None)
    print(f"extract: {count} signal records -> {args.out or 'out/candidate_signals.jsonl'}")
    print(f"  {json.dumps(telemetry.to_dict()['extract'])}")
    return 0


def cmd_score(args) -> int:
    profiles = io.load_candidates()
    jobs = io.load_jobs()
    records, traces, telemetry = run_extract(profiles, arm=args.arm)
    rankings = run_scoring(records, traces, jobs, telemetry=telemetry)
    count = write_rankings(rankings)
    print(f"score: {count} ranking records across {len(jobs)} jobs -> out/rankings.jsonl")
    return 0


def cmd_evaluate(args) -> int:
    from saral.pipeline.run_eval import run_full_evaluation

    payload = run_full_evaluation(skip_baseline=args.skip_baseline)
    io.write_json(io.OUT_DIR / "metrics.json", payload)
    print("evaluate: -> out/metrics.json")
    print(f"  verdict: {payload['uncertainty']['verdict'][:160]}...")
    return 0


def cmd_delta(args) -> int:
    from saral.pipeline.delta_pass import run_delta

    result = run_delta(
        db_path=Path(args.db) if args.db else None,
        reapply=args.reapply,
    )
    print(
        f"delta: {result['events_written']} change events "
        f"({result['materiality_counts']}) -> out/change_events.jsonl"
    )
    print(f"  recomputed {result['recomputed']}/{result['candidates_total']} candidates")
    return 0


def cmd_all(args) -> int:
    telemetry = Telemetry()
    cfg, _weights = load_all()

    profiles = io.load_candidates()
    jobs = io.load_jobs()
    records, traces, telemetry = run_extract(profiles, arm=args.arm, telemetry=telemetry)
    write_signals(records)
    print(f"[1/4] extract: {len(records)} signals")

    rankings = run_scoring(records, traces, jobs, telemetry=telemetry)
    write_rankings(rankings)
    print(f"[2/4] score: {sum(len(v) for v in rankings.values())} rankings")

    from saral.pipeline.run_eval import run_full_evaluation

    payload = run_full_evaluation(skip_baseline=args.skip_baseline, telemetry=telemetry)
    io.write_json(io.OUT_DIR / "metrics.json", payload)
    print("[3/4] evaluate: out/metrics.json")

    from saral.pipeline.delta_pass import run_delta

    result = run_delta()
    print(f"[4/4] delta: {result['events_written']} change events")

    manifest = RunManifest.start(
        arm=args.arm,
        telemetry=telemetry,
        config_hashes=cfg.config_hashes,
        models={"baseline": "sentence-transformers/all-MiniLM-L6-v2"},
    )
    manifest.write(io.OUT_DIR / "run_manifest.json")
    print("wrote out/run_manifest.json")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="saral", description=__doc__)
    parser.add_argument("--arm", default=SHIPPED_ARM, choices=sorted(ARMS))
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("extract", help="raw profiles -> signal records")
    p.add_argument("--candidates")
    p.add_argument("--out")
    p.add_argument("--computed-at")
    p.add_argument("--as-of", help="YYYY-MM-DD; default is derived from the corpus")
    p.set_defaults(func=cmd_extract)

    p = sub.add_parser("score", help="signals + jobs -> rankings")
    p.set_defaults(func=cmd_score)

    p = sub.add_parser("evaluate", help="rankings + labels -> metrics")
    p.add_argument("--skip-baseline", action="store_true", help="skip the embedding baseline")
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("delta", help="apply the change feed")
    p.add_argument("--db")
    p.add_argument("--reapply", action="store_true", help="apply twice, to demonstrate idempotency")
    p.set_defaults(func=cmd_delta)

    p = sub.add_parser("all", help="run everything")
    p.add_argument("--skip-baseline", action="store_true")
    p.set_defaults(func=cmd_all)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
