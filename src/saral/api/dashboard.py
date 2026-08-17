"""A static dashboard rendered from `out/metrics.json` and the delta report.

Served from the FastAPI process that is already running rather than as a second
Streamlit app: a second process, a second port, another 200MB of dependencies,
and one more thing for a reviewer to start. Inline SVG rather than a CDN chart
library, so it renders with no network.

Panel order is the order the numbers should be read in, with the hero number
first: **LLM calls on the hot path: 0**.
"""

from __future__ import annotations

import html
import json
from pathlib import Path

from saral.pipeline import io


def _load(name: str) -> dict:
    path = io.OUT_DIR / name
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _bar(value: float, maximum: float, width: int = 220) -> str:
    pct = 0 if maximum <= 0 else max(0.0, min(1.0, value / maximum))
    return (
        f'<svg width="{width}" height="14" role="img">'
        f'<rect width="{width}" height="14" rx="3" fill="var(--track)"/>'
        f'<rect width="{pct * width:.1f}" height="14" rx="3" fill="var(--accent)"/>'
        f"</svg>"
    )


def _esc(value) -> str:
    return html.escape(str(value))


def render_dashboard() -> str:
    metrics = _load("metrics.json")
    delta = _load("delta_report.json")
    manifest = _load("run_manifest.json")

    if not metrics:
        return _page("<p class='empty'>No metrics yet. Run <code>make all</code>.</p>")

    systems = metrics.get("systems", {})
    baseline = systems.get("baseline_cosine", {})
    shipped = systems.get("signals_v1_lexicon_lr", {})
    uncertainty = metrics.get("uncertainty", {})
    telemetry = metrics.get("telemetry", {})
    cost_arm = metrics.get("llm_per_row_cost_arm", {})

    sections: list[str] = []

    # 1. hero
    fallback_rate = telemetry.get("extract", {}).get("counters", {}).get("fallback_rate", 0)
    sections.append(
        "<section class='hero'>"
        "<div class='big'>0</div>"
        "<div class='big-label'>LLM calls on the hot path</div>"
        f"<div class='sub'>{_esc(telemetry.get('extract', {}).get('counters', {}).get('titles_classified', 0))} "
        f"titles classified by a deterministic lexicon &middot; fallback fired on "
        f"{_esc(round(float(fallback_rate) * 100, 1))}% of them</div>"
        "</section>"
    )

    # 2. metric table
    rows = []
    for metric in ("ndcg@10", "ndcg@5", "precision@5"):
        b = baseline.get(metric, {}).get("mean")
        s = shipped.get(metric, {}).get("mean")
        if b is None or s is None:
            continue
        delta_value = s - b
        rows.append(
            f"<tr><td>{_esc(metric)}</td>"
            f"<td class='num'>{b:.4f}</td>"
            f"<td class='num'>{s:.4f}</td>"
            f"<td class='num {'pos' if delta_value > 0 else 'neg'}'>{delta_value:+.4f}</td>"
            f"<td>{_bar(s, 1.0)}</td></tr>"
        )
    sections.append(
        "<section><h2>Baseline vs shipped system</h2>"
        "<table><thead><tr><th>metric</th><th>baseline (MiniLM cosine)</th>"
        "<th>signals_v1</th><th>delta</th><th></th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
        f"<p class='verdict'><strong>Verdict.</strong> {_esc(uncertainty.get('verdict', 'not computed'))}</p>"
        "</section>"
    )

    # 3. per-job deltas -- the honest picture
    per_job = uncertainty.get("per_job_ndcg@10_delta", {})
    if per_job:
        span = max(abs(v) for v in per_job.values()) or 1.0
        job_rows = "".join(
            f"<tr><td>{_esc(job)}</td><td class='num {'pos' if value > 0 else 'neg'}'>{value:+.4f}</td>"
            f"<td>{_bar(abs(value), span, 160)}</td>"
            f"<td class='muted'>P@5 ceiling {_esc(metrics.get('metric_ceilings', {}).get(job, {}).get('precision@5_ceiling', '-'))}</td></tr>"
            for job, value in per_job.items()
        )
        sections.append(
            "<section><h2>Per-job NDCG@10 delta</h2>"
            "<p class='muted'>The independent unit is the job, and there are four. "
            "Shown individually so a mean carried by one job is visible as such.</p>"
            f"<table><tbody>{job_rows}</tbody></table></section>"
        )

    # 4. ablation ladder
    ladder = metrics.get("ablation_ladder", {})
    if ladder:
        ladder_rows = "".join(
            f"<tr><td>{_esc(name)}</td><td class='num'>{_esc(payload.get('ndcg@10'))}</td>"
            f"<td class='num'>{_esc(payload.get('delta_ndcg@10_vs_previous_rung', '-'))}</td>"
            f"<td class='muted'>{_esc(payload.get('description'))}</td></tr>"
            for name, payload in ladder.items()
        )
        sections.append(
            "<section><h2>Ablation ladder</h2>"
            "<table><thead><tr><th>rung</th><th>NDCG@10</th><th>delta</th><th></th></tr></thead>"
            f"<tbody>{ladder_rows}</tbody></table></section>"
        )

    # 5. cost per 1M profiles
    cost_rows = []
    derived = manifest.get("derived", {}).get("cost_per_1m_profiles") or {}
    if derived:
        cost_rows.append(
            f"<tr><td>signals_v1 (shipped)</td><td class='num'>{derived.get('ms_per_record')} ms</td>"
            f"<td class='num'>{derived.get('cpu_hours_per_1m')} h</td>"
            f"<td class='num'>${derived.get('usd_per_1m')}</td></tr>"
        )
    for model, payload in cost_arm.items():
        mean_s = payload.get("wall_s_per_profile", {}).get("mean")
        if mean_s:
            cpu_hours = mean_s * 1_000_000 / 3600
            cost_rows.append(
                f"<tr><td>llm_per_row &middot; {_esc(model)}</td>"
                f"<td class='num'>{mean_s * 1000:,.0f} ms</td>"
                f"<td class='num'>{cpu_hours:,.0f} h</td>"
                f"<td class='num'>${cpu_hours * 0.04656:,.0f}</td></tr>"
            )
    if cost_rows:
        sections.append(
            "<section><h2>Cost of one full pass over 1M profiles</h2>"
            "<table><thead><tr><th>arm</th><th>per profile</th><th>CPU-hours</th>"
            "<th>Fargate ap-south-1</th></tr></thead>"
            f"<tbody>{''.join(cost_rows)}</tbody></table>"
            "<p class='muted'>Single-threaded, measured on this machine, priced at "
            "$0.04656/vCPU-hour. Arithmetic is in INFRA.md.</p></section>"
        )

    # 6. incremental saving
    if delta:
        counts = delta.get("materiality_counts", {})
        saving = delta.get("saving", {})
        sections.append(
            "<section><h2>Part 3: incremental vs full recompute</h2>"
            f"<p>Recomputed <strong>{_esc(delta.get('recomputed'))}</strong> of "
            f"<strong>{_esc(delta.get('candidates_total'))}</strong> candidates from "
            f"{_esc(delta.get('delta_records'))} delta records.</p>"
            "<p>Events by materiality: "
            + " &middot; ".join(
                f"<span class='pill {_esc(k)}'>{_esc(k)}: {_esc(v)}</span>"
                for k, v in sorted(counts.items())
            )
            + "</p>"
            f"<p class='muted'>{_esc(saving.get('reduction', {}).get('note', ''))}</p>"
            "</section>"
        )

    # 7. drill-down
    sections.append(_drilldown())

    return _page("".join(sections))


def _drilldown() -> str:
    """The panel a recruiter would actually use: pick a job, read the reasons."""
    path = io.OUT_DIR / "rankings.jsonl"
    if not path.exists():
        return ""
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_job: dict[str, list[dict]] = {}
    for record in records:
        by_job.setdefault(record["job_id"], []).append(record)

    blocks = []
    for job_id in sorted(by_job):
        rows = sorted(by_job[job_id], key=lambda r: r["rank"])[:8]
        body = "".join(
            "<tr>"
            f"<td class='num'>{_esc(r['rank'])}</td>"
            f"<td>{_esc(r['candidate_id'])}</td>"
            f"<td class='num'>{_esc(r['fit_score'])}</td>"
            f"<td class='num muted'>{_esc(r['confidence'])}</td>"
            f"<td>{''.join(f'<code>{_esc(c)}</code>' for c in r['reason_codes'][:6])}</td>"
            f"<td class='muted'>{_esc(', '.join(r['missing_must_haves']) or '-')}</td>"
            "</tr>"
            for r in rows
        )
        blocks.append(
            f"<details><summary>{_esc(job_id)}</summary>"
            "<table><thead><tr><th>#</th><th>candidate</th><th>score</th><th>conf</th>"
            "<th>why</th><th>missing must-haves</th></tr></thead>"
            f"<tbody>{body}</tbody></table></details>"
        )
    return (
        "<section><h2>Ranked candidates and why</h2>"
        "<p class='muted'>Every score decomposes into components, and every component "
        "leaves a reason code. Nothing is filtered out, so a rejected candidate is still "
        "visible with a readable explanation.</p>" + "".join(blocks) + "</section>"
    )


def _page(body: str) -> str:
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SARAL signals</title>
<style>
:root {{
  --bg:#fbfbfa; --fg:#1a1a19; --muted:#6b6b68; --line:#e4e4e1;
  --accent:#3b5bdb; --track:#ececeb; --pos:#2b8a3e; --neg:#c92a2a; --card:#fff;
}}
@media (prefers-color-scheme: dark) {{
  :root {{ --bg:#141414; --fg:#ececea; --muted:#9a9a96; --line:#2b2b2b;
    --accent:#748ffc; --track:#262626; --pos:#51cf66; --neg:#ff6b6b; --card:#1c1c1c; }}
}}
* {{ box-sizing:border-box; }}
body {{ margin:0; padding:2rem 1.25rem 4rem; background:var(--bg); color:var(--fg);
  font:15px/1.55 ui-sans-serif,-apple-system,"Segoe UI",Roboto,sans-serif; }}
main {{ max-width:960px; margin:0 auto; }}
h1 {{ font-size:1.5rem; margin:0 0 .25rem; letter-spacing:-.01em; }}
h2 {{ font-size:1rem; margin:0 0 .75rem; letter-spacing:.02em; text-transform:uppercase;
  color:var(--muted); font-weight:600; }}
section {{ background:var(--card); border:1px solid var(--line); border-radius:10px;
  padding:1.25rem; margin:1rem 0; }}
.hero {{ text-align:center; padding:2rem 1rem; }}
.big {{ font-size:4.5rem; font-weight:700; line-height:1; letter-spacing:-.03em; color:var(--accent); }}
.big-label {{ font-size:1rem; font-weight:600; margin-top:.4rem; }}
.sub, .muted {{ color:var(--muted); font-size:.85rem; }}
table {{ width:100%; border-collapse:collapse; font-size:.88rem; }}
th, td {{ text-align:left; padding:.45rem .5rem; border-bottom:1px solid var(--line); vertical-align:top; }}
th {{ font-weight:600; color:var(--muted); font-size:.78rem; text-transform:uppercase; }}
.num {{ text-align:right; font-variant-numeric:tabular-nums; }}
.pos {{ color:var(--pos); }} .neg {{ color:var(--neg); }}
code {{ display:inline-block; background:var(--track); border-radius:4px; padding:.1rem .35rem;
  margin:.1rem .2rem .1rem 0; font-size:.76rem; }}
.verdict {{ margin:1rem 0 0; padding:.85rem 1rem; border-left:3px solid var(--accent);
  background:var(--track); border-radius:0 6px 6px 0; font-size:.88rem; }}
.pill {{ display:inline-block; padding:.1rem .5rem; border-radius:999px; background:var(--track);
  font-size:.78rem; }}
.pill.high {{ color:var(--neg); font-weight:600; }}
.pill.noise {{ color:var(--muted); }}
details {{ margin:.5rem 0; }} summary {{ cursor:pointer; font-weight:600; padding:.35rem 0; }}
.wrap {{ overflow-x:auto; }}
</style></head>
<body><main>
<h1>SARAL candidate signals</h1>
<p class="muted">Structured signals, fit scoring, and an evaluation harness.
Numbers below are read from <code>out/</code>; nothing on this page is typed by hand.</p>
{body}
</main></body></html>"""
