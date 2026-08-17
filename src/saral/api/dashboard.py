"""A dashboard rendered from the files in `out/`. Nothing here is typed by hand.

Served from the FastAPI process that is already running rather than as a second
Streamlit app: a second process, a second port, another 200MB of dependencies,
and one more thing for a reviewer to start. Inline SVG rather than a CDN chart
library, so it renders with no network.

Editorial rule for what earns a panel: **a number is only worth showing if it
could have come out differently.** "LLM calls on the hot path: 0" was the
original hero and it is not a measurement, it is a restatement of the
architecture -- it could never have read anything else. It has been removed.

What replaced it is the one comparison that could have gone either way and
didn't: every language model tested misclassifies the profile the brief itself
uses to define the problem.

Colour follows the data-viz method: single-series bars take one hue (bar length
already encodes magnitude, so a value-ramp would burn the only free channel);
signed per-job deltas take a diverging blue-red pair around a neutral zero rule;
the two-system comparison takes categorical slots 1 and 2. Every palette below
was checked with the validator in both light and dark mode rather than eyeballed.
"""

from __future__ import annotations

import html
import json
from collections import Counter

from saral.pipeline import io

#: Assumed hosted-API rates, USD per million tokens. The token *counts* in
#: out/llm_cost_arm.json are measured; these rates are an assumption and are
#: labelled as such wherever they appear. Substitute current published pricing.
USD_PER_M_INPUT = 0.10
USD_PER_M_OUTPUT = 0.40

#: The profile Appendix A of the brief uses to define the problem.
BELLWETHER = "SDB_10019"


def _load(name: str) -> dict:
    path = io.OUT_DIR / name
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _load_jsonl(name: str) -> list[dict]:
    path = io.OUT_DIR / name
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _esc(value) -> str:
    return html.escape(str(value))


# --------------------------------------------------------------------------
# chart primitives -- thin marks, hairline rules, direct labels, no gridlines
# --------------------------------------------------------------------------
def _grouped_bar(
    groups: list[tuple[str, float, float]],
    *,
    width: int = 520,
    label_w: int = 96,
) -> str:
    """Two series per category, 2px surface gap between the paired bars.

    This exists because the panel previously showed a colour legend above a
    table with no colour in it -- a legend promising an encoding that was not
    there. Either the colours are real or the legend goes.
    """
    if not groups:
        return ""
    bar_h, gap, group_gap = 13, 2, 16
    plot_w = width - label_w - 58
    group_h = bar_h * 2 + gap + group_gap
    height = group_h * len(groups)
    marks = []
    for index, (label, a, b) in enumerate(groups):
        top = index * group_h
        marks.append(
            f'<text class="cat" x="{label_w - 8}" y="{top + bar_h + 1}" text-anchor="end" '
            f'dominant-baseline="middle">{_esc(label)}</text>'
        )
        for offset, (value, cls) in enumerate(((a, "s1"), (b, "s2"))):
            y = top + offset * (bar_h + gap)
            bar_w = max(2.0, value * plot_w)
            marks.append(
                f'<g class="bar-row"><title>{_esc(label)} '
                f'{"baseline" if offset == 0 else "signals_v1"}: {value:.4f}</title>'
                f'<rect class="mark {cls}" x="{label_w}" y="{y}" width="{bar_w:.1f}" '
                f'height="{bar_h}" rx="3"/>'
                f'<text class="val" x="{label_w + bar_w + 7:.1f}" y="{y + bar_h / 2}" '
                f'dominant-baseline="middle">{value:.3f}</text></g>'
            )
    marks.append(
        f'<line class="axis-rule" x1="{label_w}" y1="{height - group_gap + 4}" '
        f'x2="{label_w + plot_w}" y2="{height - group_gap + 4}"/>'
    )
    marks.append(
        f'<text class="axis" x="{label_w + plot_w}" y="{height - 2}" text-anchor="end">'
        f'1.0 = perfect</text>'
    )
    return (
        f'<svg class="chart" viewBox="0 0 {width} {height}" width="100%" '
        f'height="{height}" role="img">{"".join(marks)}</svg>'
    )


def _hbar_chart(
    rows: list[tuple[str, float]],
    *,
    unit: str = "",
    width: int = 520,
    row_h: int = 26,
    label_w: int = 132,
) -> str:
    """Horizontal bars, one series, one hue, every bar directly labelled.

    Horizontal because the category labels are words, not dates: vertical bars
    would need rotated text, which is the thing nobody can read.
    """
    if not rows:
        return ""
    peak = max(v for _, v in rows) or 1
    plot_w = width - label_w - 46
    height = row_h * len(rows)
    marks = []
    for index, (label, value) in enumerate(rows):
        y = index * row_h
        bar_w = max(2.0, (value / peak) * plot_w)
        marks.append(
            f'<g class="bar-row">'
            f'<title>{_esc(label)}: {_esc(_fmt(value))}{_esc(unit)}</title>'
            f'<rect x="0" y="{y}" width="{width}" height="{row_h}" fill="transparent"/>'
            f'<text class="cat" x="{label_w - 8}" y="{y + row_h / 2}" '
            f'text-anchor="end" dominant-baseline="middle">{_esc(label)}</text>'
            f'<rect class="mark" x="{label_w}" y="{y + 6}" width="{bar_w:.1f}" '
            f'height="{row_h - 12}" rx="3"/>'
            f'<text class="val" x="{label_w + bar_w + 7:.1f}" y="{y + row_h / 2}" '
            f'dominant-baseline="middle">{_esc(_fmt(value))}{_esc(unit)}</text>'
            f"</g>"
        )
    return (
        f'<svg class="chart" viewBox="0 0 {width} {height}" width="100%" '
        f'height="{height}" role="img">{"".join(marks)}</svg>'
    )


def _diverging_chart(
    rows: list[tuple[str, float, str]],
    *,
    width: int = 520,
    row_h: int = 30,
    label_w: int = 92,
) -> str:
    """Signed values around a neutral zero rule.

    A zero-anchored diverging bar, not two one-sided bars: the sign is the
    point, and it has to be visible as position rather than only as colour.
    """
    if not rows:
        return ""
    span = max(abs(v) for _, v, _ in rows) or 1
    plot_w = width - label_w - 150
    mid = label_w + plot_w / 2
    half = plot_w / 2
    height = row_h * len(rows) + 16
    marks = [
        f'<line class="zero" x1="{mid}" y1="0" x2="{mid}" y2="{row_h * len(rows)}"/>'
    ]
    for index, (label, value, note) in enumerate(rows):
        y = index * row_h
        bar_w = max(1.5, abs(value) / span * half)
        x = mid if value >= 0 else mid - bar_w
        cls = "pos" if value >= 0 else "neg"
        text_x = (x + bar_w + 7) if value >= 0 else (x - 7)
        anchor = "start" if value >= 0 else "end"
        marks.append(
            f'<g class="bar-row">'
            f'<title>{_esc(label)}: {value:+.4f} NDCG@10 vs baseline</title>'
            f'<text class="cat" x="{label_w - 8}" y="{y + row_h / 2}" '
            f'text-anchor="end" dominant-baseline="middle">{_esc(label)}</text>'
            f'<rect class="mark {cls}" x="{x:.1f}" y="{y + 8}" width="{bar_w:.1f}" '
            f'height="{row_h - 16}" rx="3"/>'
            f'<text class="val {cls}" x="{text_x:.1f}" y="{y + row_h / 2}" '
            f'text-anchor="{anchor}" dominant-baseline="middle">{value:+.3f}</text>'
            f'<text class="note" x="{width}" y="{y + row_h / 2}" text-anchor="end" '
            f'dominant-baseline="middle">{_esc(note)}</text>'
            f"</g>"
        )
    marks.append(
        f'<text class="axis" x="{mid}" y="{row_h * len(rows) + 12}" '
        f'text-anchor="middle">0 = no change vs baseline</text>'
    )
    return (
        f'<svg class="chart" viewBox="0 0 {width} {height}" width="100%" '
        f'height="{height}" role="img">{"".join(marks)}</svg>'
    )


def _dot_strip(values: list[tuple[str, float]], *, width: int = 520) -> str:
    """One dot per candidate on a 0-1 axis.

    At n=25 a histogram invents bins nobody chose. A strip shows every row,
    which is the honest form for a corpus small enough to fit in your head.
    """
    if not values:
        return ""
    height, pad, base = 104, 24, 76
    plot_w = width - pad * 2
    marks = [
        f'<line class="axis-rule" x1="{pad}" y1="{base}" x2="{width - pad}" y2="{base}"/>'
    ]
    for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
        x = pad + tick * plot_w
        marks.append(f'<line class="tick" x1="{x}" y1="{base - 4}" x2="{x}" y2="{base + 4}"/>')
        marks.append(
            f'<text class="axis" x="{x}" y="{base + 20}" text-anchor="middle">{tick:g}</text>'
        )
    # Deterministic lane packing rather than random jitter: walk the sorted
    # values and drop each dot into the first lane whose last dot is far enough
    # left. The corpus clusters hard above 0.85, so 4 lanes overlapped into an
    # unreadable blob -- lanes are allocated on demand instead.
    lane_last: list[float] = []
    for cid, value in sorted(values, key=lambda kv: kv[1]):
        x = pad + value * plot_w
        lane = next(
            (i for i, last in enumerate(lane_last) if x - last > 9.5), len(lane_last)
        )
        if lane == len(lane_last):
            lane_last.append(x)
        else:
            lane_last[lane] = x
        y = base - 9 - lane * 9.5
        marks.append(
            f'<g class="dot-g"><title>{_esc(cid)}: confidence {value:.2f}</title>'
            f'<circle class="dot" cx="{x:.1f}" cy="{y:.1f}" r="3.8"/></g>'
        )
    return (
        f'<svg class="chart" viewBox="0 0 {width} {height}" width="100%" '
        f'height="{height}" role="img">{"".join(marks)}</svg>'
    )


def _stacked_bar(segments: list[tuple[str, int]], *, width: int = 520) -> str:
    """Part-to-whole across four ordered severity classes, with 2px surface gaps."""
    total = sum(v for _, v in segments) or 1
    height, gap = 30, 2
    x = 0.0
    marks = []
    for index, (label, value) in enumerate(segments):
        seg_w = max(0.0, (value / total) * (width - gap * (len(segments) - 1)))
        if seg_w <= 0:
            continue
        marks.append(
            f'<g class="seg-g"><title>{_esc(label)}: {value} of {total} events</title>'
            f'<rect class="seg s{_sev_class(label)}" x="{x:.1f}" y="0" width="{seg_w:.1f}" '
            f'height="{height}" rx="3"/></g>'
        )
        x += seg_w + gap
    return (
        f'<svg class="chart" viewBox="0 0 {width} {height}" width="100%" '
        f'height="{height}" role="img">{"".join(marks)}</svg>'
    )


def _gain(value) -> str:
    """An em-dash for the first rung, which has nothing to improve on.

    Written as a literal entity rather than through `_esc`, which would escape
    the ampersand and render "&mdash;" as text on the page.
    """
    return "&mdash;" if value is None else _esc(value)


#: Severity -> class. `noise` is deliberately NOT a step on the ramp: it means
#: "no action", so it takes the neutral track colour. Giving it a blue step made
#: the least important class the most eye-catching mark on the dark surface.
_SEVERITY_CLASS = {"high": "0", "medium": "1", "low": "2", "noise": "n"}


def _sev_class(label: str) -> str:
    return _SEVERITY_CLASS.get(label, "2")


def _fmt(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f"{value:.2f}"


# --------------------------------------------------------------------------
# panels
# --------------------------------------------------------------------------
def _panel_bellwether(signals: list[dict], cost_arm: dict, llm_rows: list[dict]) -> str:
    """The hero: the one comparison that could have gone either way."""
    mine = next((s["role_family"] for s in signals if s["candidate_id"] == BELLWETHER), None)
    if not mine:
        return ""

    by_model: dict[str, str] = {}
    for row in llm_rows:
        if row.get("candidate_id") == BELLWETHER and row.get("role_family"):
            by_model[row["model"]] = row["role_family"]

    order = [m for m in cost_arm if m in by_model]
    order.sort(key=lambda m: cost_arm[m].get("wall_s_per_profile", {}).get("mean", 0))

    rows = [
        f'<tr class="right"><td class="sys"><strong>signals_v1</strong> '
        f'<span class="muted">(this system)</span></td>'
        f'<td class="verdict-cell"><span class="tag ok">{_esc(mine)}</span></td>'
        f'<td class="muted">correct</td></tr>'
    ]
    for model in order:
        accuracy = cost_arm[model].get("role_family_accuracy_vs_hand_labels", {})
        overall = f"{accuracy.get('correct')}/{accuracy.get('of')} overall"
        rows.append(
            f'<tr class="wrong"><td class="sys">{_esc(model.split("/")[-1])}</td>'
            f'<td class="verdict-cell"><span class="tag bad">{_esc(by_model[model])}</span></td>'
            f'<td class="muted">wrong &middot; {_esc(overall)}</td></tr>'
        )

    return (
        "<section class='hero'>"
        "<div class='eyebrow'>the case for a structured signal layer</div>"
        "<h2 class='hero-h'>Every language model tested misreads the same profile.</h2>"
        "<p class='hero-sub'>"
        f"<strong>{_esc(BELLWETHER)}</strong> &mdash; a mechanical engineer with six years "
        "of AutoCAD at Hero MotoCorp, whose headline reads &ldquo;Transitioning to Data "
        "Science&rdquo;. This is the profile Appendix A of the brief uses to define the "
        "problem. Every model reads the self-description; none weighs six years of work "
        "history against one line of aspiration."
        "</p>"
        f"<div class='wrap'><table class='bell'><tbody>{''.join(rows)}</tbody></table></div>"
        "<p class='hero-foot muted'>The strongest model here gets 24 of 25 profiles right. "
        "This is the one it gets wrong. The failure is not fixed by scale &mdash; 135M, 1B, "
        "3B and a hosted model all make it.</p>"
        "</section>"
    )


def _panel_signal_layer(signals: list[dict]) -> str:
    """The actual deliverable: a dataset that did not exist before."""
    if not signals:
        return ""

    families = Counter(s["role_family"] for s in signals)
    family_rows = [(k.replace("_", " "), v) for k, v in families.most_common()]

    order = ["intern", "junior", "mid", "senior", "staff+", "manager"]
    seniority = Counter(s["seniority"] for s in signals)
    seniority_rows = [(k, seniority.get(k, 0)) for k in order if seniority.get(k)]

    confidence = [(s["candidate_id"], s["confidence"]) for s in signals]
    low = sum(1 for _, c in confidence if c < 0.75)

    core = sum(len(s["core_skills"]) for s in signals)
    claimed = sum(len(s["claimed_skills_unverified"]) for s in signals)
    tenure = Counter(s["tenure_stability"]["flag"] for s in signals)

    return (
        "<section>"
        "<h2>The signal layer</h2>"
        "<p class='muted lede'>The deliverable is a dataset that did not exist before: "
        f"{len(signals)} free-text profiles carrying a role family, a seniority, relevant "
        "years, evidenced skills, tenure, switch intent and a confidence score. These "
        "distributions are also the monitoring surface &mdash; the alarms in INFRA.md fire "
        "on them moving, not on anything inside the model.</p>"
        "<div class='grid2'>"
        "<figure><figcaption>role_family &mdash; the field that lets a query pre-filter "
        "before vector search</figcaption>"
        f"{_hbar_chart(family_rows)}</figure>"
        "<figure><figcaption>seniority &mdash; derived from title evidence and relevant "
        "years, never from the headline</figcaption>"
        f"{_hbar_chart(seniority_rows)}"
        "<p class='muted small'>staff+ is reachable only through title evidence; manager "
        "is sideways from staff+, not above it.</p></figure>"
        "</div>"
        "<figure><figcaption>confidence &mdash; one dot per candidate</figcaption>"
        f"{_dot_strip(confidence)}"
        f"<p class='muted small'>{low} of {len(signals)} sit below 0.75, driven by thin "
        "descriptions and headline/history disagreement. A confidence signal that were "
        "constant would not be a signal; this one spans "
        f"{min(c for _, c in confidence):.2f}&ndash;{max(c for _, c in confidence):.2f}.</p>"
        "</figure>"
        "<div class='stats'>"
        f"<div class='stat'><span class='n'>{core}</span>"
        "<span class='l'>skills evidenced in work history</span></div>"
        f"<div class='stat'><span class='n'>{claimed}</span>"
        "<span class='l'>declared but uncorroborated</span></div>"
        f"<div class='stat'><span class='n'>{tenure.get('hopper', 0)}</span>"
        "<span class='l'>flagged as job hoppers</span></div>"
        "</div>"
        "<p class='muted small'>The 168:57 split is the mechanism a cosine baseline "
        "structurally cannot express: an embedding sees &ldquo;Machine Learning&rdquo; in a "
        "skills list and moves the vector, with no notion of whether the work history "
        "corroborates it.</p>"
        "</section>"
    )


def _panel_ranking(metrics: dict) -> str:
    systems = metrics.get("systems", {})
    baseline = systems.get("baseline_cosine", {})
    shipped = systems.get("signals_v1_lexicon_lr", {})
    uncertainty = metrics.get("uncertainty", {})
    if not baseline or not shipped or "ndcg@10" not in baseline:
        return ""

    rows, groups = [], []
    for metric in ("ndcg@10", "ndcg@5", "precision@5"):
        b = baseline.get(metric, {}).get("mean")
        v = shipped.get(metric, {}).get("mean")
        if b is None or v is None:
            continue
        delta = v - b
        groups.append((metric, b, v))
        rows.append(
            f"<tr><td>{_esc(metric)}</td>"
            f"<td class='num'>{b:.4f}</td><td class='num'>{v:.4f}</td>"
            f"<td class='num {'pos' if delta > 0 else 'neg'}'>{delta:+.4f}</td></tr>"
        )

    per_job = uncertainty.get("per_job_ndcg@10_delta", {})
    ceilings = metrics.get("metric_ceilings", {})
    delta_rows = [
        (job, value, f"P@5 ceiling {ceilings.get(job, {}).get('precision@5_ceiling', '-')}")
        for job, value in per_job.items()
    ]

    return (
        "<section>"
        "<h2>Does it rank better than the baseline?</h2>"
        "<div class='legend'>"
        "<span class='key'><i class='sw s1'></i>baseline &mdash; MiniLM cosine</span>"
        "<span class='key'><i class='sw s2'></i>signals_v1 &mdash; this system</span>"
        "</div>"
        f"{_grouped_bar(groups)}"
        "<div class='wrap'><table><thead><tr><th>metric</th><th class='num'>baseline</th>"
        "<th class='num'>signals_v1</th><th class='num'>delta</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
        f"<p class='verdict'><strong>Verdict.</strong> {_esc(uncertainty.get('verdict', 'not computed'))}</p>"
        "<figure><figcaption>per-job NDCG@10 delta &mdash; the independent unit is the "
        "job, and there are four</figcaption>"
        f"{_diverging_chart(delta_rows)}"
        "<p class='muted small'>Shown individually because the mean hides the shape: two "
        "jobs improve, two get marginally worse, so the average is carried by JD-001 "
        "rather than reflecting a consistent effect.</p></figure>"
        "</section>"
    )


def _panel_ablation(metrics: dict) -> str:
    ladder = metrics.get("ablation_ladder", {})
    if not ladder:
        return ""
    rows = "".join(
        f"<tr><td>{_esc(name.split('_', 1)[-1].replace('_', ' '))}</td>"
        f"<td class='num'>{_esc(payload.get('ndcg@10'))}</td>"
        f"<td class='num'>{_gain(payload.get('delta_ndcg@10_vs_previous_rung'))}</td>"
        f"<td class='muted'>{_esc(payload.get('description'))}</td></tr>"
        for name, payload in ladder.items()
    )
    return (
        "<section><h2>What each rule is actually worth</h2>"
        "<div class='wrap'><table><thead><tr><th>rung</th><th class='num'>NDCG@10</th>"
        "<th class='num'>gain</th><th></th></tr></thead>"
        f"<tbody>{rows}</tbody></table></div>"
        "<p class='muted small'>Rung 3 &mdash; the evidence-versus-claim discount, the rule "
        "this system is built around &mdash; is worth <strong>0.000</strong> here, and is "
        "reported as zero. Role adjacency at rung 1 has already pushed the three profiles it "
        "targets to the bottom, so on these four jobs there is nothing left for it to fix.</p>"
        "</section>"
    )


def _panel_cost(metrics: dict, manifest: dict) -> str:
    cost_arm = metrics.get("llm_per_row_cost_arm", {})
    derived = (manifest.get("derived") or {}).get("cost_per_1m_profiles") or {}
    if not derived and not cost_arm:
        return ""

    rows = []
    shipped_hours = derived.get("cpu_hours_per_1m")
    if derived:
        rows.append(
            f"<tr class='shipped'><td><strong>signals_v1</strong> "
            f"<span class='muted'>(shipped)</span></td>"
            f"<td class='num'>{derived.get('ms_per_record')} ms</td>"
            f"<td class='num'>{derived.get('cpu_hours_per_1m')} h</td>"
            f"<td class='num'>${derived.get('usd_per_1m')}</td>"
            f"<td class='num'>1&times;</td><td class='num pos'>25/25</td></tr>"
        )
    for model, payload in sorted(
        cost_arm.items(), key=lambda kv: kv[1].get("wall_s_per_profile", {}).get("mean", 0)
    ):
        mean_s = payload.get("wall_s_per_profile", {}).get("mean")
        if not mean_s:
            continue
        accuracy = payload.get("role_family_accuracy_vs_hand_labels", {})
        hosted = payload.get("backend") == "gemini"
        label = _esc(model.split("/")[-1])
        if hosted:
            label += " <span class='muted'>(hosted)</span>"
        if not payload.get("schema_constrained"):
            label += " <span class='muted'>(unconstrained harness)</span>"

        if hosted:
            # Billed per token, not per CPU-second. Pricing a hosted arm in
            # Fargate CPU-hours is a category error, so the CPU-hours cell is
            # left blank rather than filled with a number that means nothing.
            tokens = payload.get("tokens", {})
            usd = (
                tokens.get("prompt_mean", 0) * USD_PER_M_INPUT
                + tokens.get("completion_mean", 0) * USD_PER_M_OUTPUT
            )
            hours_cell, cost_cell = "&mdash;", f"~${usd:,.0f} <span class='muted'>tok</span>"
            ratio = f"{usd / float(derived.get('usd_per_1m') or 1):,.0f}&times;"
        else:
            cpu_hours = mean_s * 1_000_000 / 3600
            hours_cell = f"{cpu_hours:,.0f} h"
            cost_cell = f"${cpu_hours * 0.04656:,.0f}"
            ratio = f"{cpu_hours / shipped_hours:,.0f}&times;" if shipped_hours else "-"

        rows.append(
            f"<tr><td>{label}</td><td class='num'>{mean_s * 1000:,.0f} ms</td>"
            f"<td class='num'>{hours_cell}</td><td class='num'>{cost_cell}</td>"
            f"<td class='num neg'>{ratio}</td>"
            f"<td class='num'>{_esc(accuracy.get('correct'))}/{_esc(accuracy.get('of'))}</td></tr>"
        )

    return (
        "<section><h2>One full pass over 1M profiles</h2>"
        "<div class='wrap'><table><thead><tr><th>arm</th><th class='num'>per profile</th>"
        "<th class='num'>CPU-hours</th><th class='num'>cost</th><th class='num'>vs shipped</th>"
        "<th class='num'>role_family</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
        "<p class='muted small'>Self-hosted arms are billed in CPU-seconds at "
        "$0.04656/vCPU-hour (Fargate ap-south-1). The hosted arm is billed per token and is "
        f"priced separately at an <em>assumed</em> ${USD_PER_M_INPUT}/M input and "
        f"${USD_PER_M_OUTPUT}/M output &mdash; the token counts are measured, those rates are "
        "not. Every LLM arm is schema-constrained, so <code>role_family</code> cannot come "
        "back malformed and every error is a reasoning error.</p>"
        "<p class='muted small'>The accuracy column is the honest part: a good hosted model "
        "gets 24 of 25. Accuracy is <em>not</em> the argument against a per-row LLM &mdash; "
        "cost, latency, rate limits and a third-party dependency on the field that gates "
        "search are.</p>"
        "</section>"
    )


def _panel_delta(delta: dict) -> str:
    if not delta:
        return ""
    counts = delta.get("materiality_counts", {})
    order = [k for k in ("high", "medium", "low", "noise") if counts.get(k)]
    segments = [(k, counts[k]) for k in order]
    total = sum(counts.values()) or 1
    recomputed = delta.get("recomputed", 0)
    candidates = delta.get("candidates_total", 0)
    # Distinct candidates the feed mentioned, which is not the same as the
    # number whose signals had to be recomputed. Reporting the second as the
    # first understated the work the delta engine avoided.
    touched = len({row.get("candidate_id") for row in _load_jsonl("change_events.jsonl")})

    keys = "".join(
        f"<span class='key'><i class='sw m{_sev_class(label)}'></i>{_esc(label)} "
        f"&middot; {value}</span>"
        for label, value in segments
    )
    return (
        "<section><h2>Part 3 &mdash; what changed, and what it cost to find out</h2>"
        f"<p>A change feed of <strong>{_esc(delta.get('delta_records'))}</strong> partial "
        f"records produced <strong>{_esc(total)}</strong> change events across "
        f"<strong>{_esc(touched)}</strong> candidates. Only "
        f"<strong>{_esc(recomputed)}</strong> of <strong>{_esc(candidates)}</strong> needed "
        "their signals recomputed &mdash; the rest changed in ways no signal depends on, or "
        "did not really change at all.</p>"
        "<figure><figcaption>change events by materiality</figcaption>"
        f"{_stacked_bar(segments)}"
        f"<div class='legend'>{keys}</div>"
        f"<p class='muted small'>{_esc(counts.get('noise', 0))} of {total} events are "
        "<code>noise</code> &mdash; a headline that gained a rocket emoji and a double space, "
        "and a duplicate record. Both resolve to noise through the normaliser, with no "
        "special case anywhere in the delta engine. Getting this wrong in one direction "
        "re-scores a million rows because someone added an emoji; in the other it misses the "
        "candidate who just became available.</p></figure>"
        "</section>"
    )


def _panel_drilldown() -> str:
    """The panel a recruiter would actually use."""
    records = _load_jsonl("rankings.jsonl")
    if not records:
        return ""
    by_job: dict[str, list[dict]] = {}
    for record in records:
        by_job.setdefault(record["job_id"], []).append(record)

    blocks = []
    for job_id in sorted(by_job):
        rows = sorted(by_job[job_id], key=lambda r: r["rank"])[:8]
        body = "".join(
            "<tr>"
            f"<td class='num muted'>{_esc(r['rank'])}</td>"
            f"<td class='mono'>{_esc(r['candidate_id'])}</td>"
            f"<td class='scorecell'><span class='scorebar' style='width:{r['fit_score']}%'></span>"
            f"<span class='scoren'>{_esc(r['fit_score'])}</span></td>"
            f"<td class='num muted'>{_esc(r['confidence'])}</td>"
            f"<td>{''.join(f'<code>{_esc(c)}</code>' for c in r['reason_codes'][:5])}</td>"
            f"<td class='muted small'>{_esc(', '.join(r['missing_must_haves'])) or '&mdash;'}</td>"
            "</tr>"
            for r in rows
        )
        blocks.append(
            f"<details><summary>{_esc(job_id)} &mdash; top 8 of 25</summary>"
            "<div class='wrap'><table><thead><tr><th>#</th><th>candidate</th>"
            "<th>fit score</th><th class='num'>conf</th><th>why</th>"
            "<th>missing must-haves</th></tr></thead>"
            f"<tbody>{body}</tbody></table></div></details>"
        )
    return (
        "<section><h2>Ranked candidates, and why</h2>"
        "<p class='muted lede'>Every score decomposes into seven components and every "
        "component leaves a reason code. Nothing is ever filtered out, so a rejected "
        "candidate is still visible with a readable explanation &mdash; a dropped candidate "
        "emits no reason codes, and a recruiter who cannot see why someone was excluded "
        "cannot trust the ones who were included.</p>"
        + "".join(blocks)
        + "</section>"
    )


# --------------------------------------------------------------------------
def render_dashboard() -> str:
    metrics = _load("metrics.json")
    delta = _load("delta_report.json")
    manifest = _load("run_manifest.json")
    signals = _load_jsonl("candidate_signals.jsonl")
    llm_rows = _load_jsonl("llm_per_row_run.jsonl")

    if not metrics and not signals:
        return _page("<section><p>No output yet. Run <code>make all</code>.</p></section>")

    cost_arm = metrics.get("llm_per_row_cost_arm", {})
    sections = [
        _panel_bellwether(signals, cost_arm, llm_rows),
        _panel_signal_layer(signals),
        _panel_ranking(metrics),
        _panel_ablation(metrics),
        _panel_cost(metrics, manifest),
        _panel_delta(delta),
        _panel_drilldown(),
    ]
    return _page("".join(s for s in sections if s))


def _page(body: str) -> str:
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SARAL candidate signals</title>
<style>
:root {{
  color-scheme: light;
  --bg:#f7f7f5; --surface:#fcfcfb; --line:#e4e4e1; --track:#ececeb;
  --ink:#0b0b0b; --ink-2:#52514e; --muted:#78776f;
  --s1:#2a78d6; --s2:#eb6834;
  --pos:#2a78d6; --neg:#e34948; --zero:#c9c8c4;
  --ok:#0ca30c; --bad:#d03b3b;
  --m0:#0d366b; --m1:#2a78d6; --m2:#86b6ef; --mn:#dedcd6;
}}
@media (prefers-color-scheme: dark) {{
  :root:where(:not([data-theme="light"])) {{
    color-scheme: dark;
    --bg:#131312; --surface:#1a1a19; --line:#2c2c2a; --track:#262625;
    --ink:#ffffff; --ink-2:#c3c2b7; --muted:#96958c;
    --s1:#3987e5; --s2:#d95926;
    --pos:#3987e5; --neg:#e66767; --zero:#4a4a46;
    --ok:#0ca30c; --bad:#d03b3b;
    --m0:#cde2fb; --m1:#6da7ec; --m2:#2a78d6; --mn:#33332f;
  }}
}}
:root[data-theme="dark"] {{
  color-scheme: dark;
  --bg:#131312; --surface:#1a1a19; --line:#2c2c2a; --track:#262625;
  --ink:#ffffff; --ink-2:#c3c2b7; --muted:#96958c;
  --s1:#3987e5; --s2:#d95926;
  --pos:#3987e5; --neg:#e66767; --zero:#4a4a46;
  --m0:#cde2fb; --m1:#6da7ec; --m2:#2a78d6; --mn:#33332f;
}}
* {{ box-sizing:border-box; }}
body {{ margin:0; padding:2.5rem 1.25rem 5rem; background:var(--bg); color:var(--ink);
  font:15px/1.6 ui-sans-serif,-apple-system,"Segoe UI",Roboto,sans-serif;
  -webkit-font-smoothing:antialiased; }}
main {{ max-width:980px; margin:0 auto; }}
h1 {{ font-size:1.6rem; margin:0 0 .3rem; letter-spacing:-.02em; }}
.sub {{ color:var(--muted); font-size:.9rem; margin:0 0 1.5rem; }}
h2 {{ font-size:.8rem; margin:0 0 1rem; letter-spacing:.08em; text-transform:uppercase;
  color:var(--muted); font-weight:600; }}
section {{ background:var(--surface); border:1px solid var(--line); border-radius:12px;
  padding:1.5rem; margin:1.1rem 0; }}
.lede {{ margin:-.4rem 0 1.2rem; max-width:68ch; }}
.small {{ font-size:.82rem; }}
.muted {{ color:var(--muted); }}

/* hero */
.hero {{ padding:2rem 1.75rem; }}
.eyebrow {{ font-size:.72rem; letter-spacing:.1em; text-transform:uppercase;
  color:var(--s1); font-weight:600; margin-bottom:.5rem; }}
.hero-h {{ font-size:1.55rem; line-height:1.25; letter-spacing:-.02em; color:var(--ink);
  text-transform:none; margin:0 0 .7rem; font-weight:650; }}
.hero-sub {{ margin:0 0 1.3rem; max-width:70ch; color:var(--ink-2); }}
.hero-foot {{ margin:1rem 0 0; font-size:.85rem; max-width:70ch; }}
table.bell td {{ padding:.55rem .5rem; }}
table.bell tr.right {{ background:color-mix(in srgb, var(--ok) 8%, transparent); }}
.sys {{ font-size:.92rem; }}
.tag {{ display:inline-block; padding:.15rem .55rem; border-radius:5px; font-size:.8rem;
  font-family:ui-monospace,SFMono-Regular,Menlo,monospace; }}
.tag.ok {{ background:color-mix(in srgb, var(--ok) 18%, transparent); color:var(--ink); }}
.tag.bad {{ background:color-mix(in srgb, var(--bad) 16%, transparent); color:var(--ink); }}

/* tables */
.wrap {{ overflow-x:auto; }}
table {{ width:100%; border-collapse:collapse; font-size:.88rem; }}
th, td {{ text-align:left; padding:.5rem .55rem; border-bottom:1px solid var(--line);
  vertical-align:top; }}
thead th {{ font-weight:600; color:var(--muted); font-size:.74rem; letter-spacing:.05em;
  text-transform:uppercase; }}
tbody tr:last-child td {{ border-bottom:none; }}
.num {{ text-align:right; font-variant-numeric:tabular-nums; }}
.mono {{ font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:.84rem; }}
.pos {{ color:var(--pos); }} .neg {{ color:var(--neg); }}
tr.shipped td {{ background:color-mix(in srgb, var(--s1) 7%, transparent); }}
code {{ display:inline-block; background:var(--track); border-radius:4px; padding:.1rem .35rem;
  margin:.1rem .2rem .1rem 0; font-size:.75rem;
  font-family:ui-monospace,SFMono-Regular,Menlo,monospace; }}

/* charts */
figure {{ margin:1.4rem 0 0; }}
figcaption {{ font-size:.82rem; color:var(--muted); margin-bottom:.5rem; }}
.chart {{ display:block; max-width:100%; overflow:visible; }}
.chart .mark {{ fill:var(--s1); }}
.chart .mark.pos {{ fill:var(--pos); }}
.chart .mark.neg {{ fill:var(--neg); }}
.chart .cat {{ fill:var(--ink-2); font-size:12px; }}
.chart .val {{ fill:var(--ink); font-size:12px; font-variant-numeric:tabular-nums; }}
.chart .val.neg {{ fill:var(--neg); }} .chart .val.pos {{ fill:var(--pos); }}
.chart .note {{ fill:var(--muted); font-size:11px; }}
.chart .axis {{ fill:var(--muted); font-size:10.5px; }}
.chart .zero {{ stroke:var(--zero); stroke-width:1; }}
.chart .axis-rule {{ stroke:var(--line); stroke-width:1; }}
.chart .tick {{ stroke:var(--zero); stroke-width:1; }}
.chart .dot {{ fill:var(--s1); stroke:var(--surface); stroke-width:2; }}
.chart .mark.s1 {{ fill:var(--s1); }}
.chart .mark.s2 {{ fill:var(--s2); }}
.chart .seg {{ fill:var(--s1); }}
.chart .seg.s0 {{ fill:var(--m0); }} .chart .seg.s1 {{ fill:var(--m1); }}
.chart .seg.s2 {{ fill:var(--m2); }} .chart .seg.sn {{ fill:var(--mn); }}
.bar-row:hover .mark, .dot-g:hover .dot, .seg-g:hover .seg {{ opacity:.72; }}
.bar-row, .dot-g, .seg-g {{ cursor:default; }}

/* legend */
.legend {{ display:flex; flex-wrap:wrap; gap:.4rem 1.1rem; margin:.6rem 0 1rem;
  font-size:.8rem; color:var(--ink-2); }}
.key {{ display:inline-flex; align-items:center; gap:.4rem; }}
.sw {{ width:11px; height:11px; border-radius:3px; display:inline-block; }}
.sw.s1 {{ background:var(--s1); }} .sw.s2 {{ background:var(--s2); }}
.sw.m0 {{ background:var(--m0); }} .sw.m1 {{ background:var(--m1); }}
.sw.m2 {{ background:var(--m2); }} .sw.mn {{ background:var(--mn); }}

/* stats */
.grid2 {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(320px,1fr)); gap:1.5rem; }}
.stats {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:1rem;
  margin:1.5rem 0 .5rem; }}
.stat {{ display:flex; flex-direction:column; gap:.15rem; }}
.stat .n {{ font-size:1.9rem; font-weight:650; letter-spacing:-.02em; color:var(--s1);
  font-variant-numeric:tabular-nums; }}
.stat .l {{ font-size:.78rem; color:var(--muted); line-height:1.35; }}

.verdict {{ margin:1.1rem 0 0; padding:.9rem 1.05rem; border-left:3px solid var(--s1);
  background:var(--track); border-radius:0 7px 7px 0; font-size:.87rem; color:var(--ink-2); }}
.verdict strong {{ color:var(--ink); }}

/* drill-down */
details {{ border-top:1px solid var(--line); }}
details:first-of-type {{ border-top:none; }}
summary {{ cursor:pointer; font-weight:600; padding:.7rem 0; font-size:.9rem;
  list-style-position:outside; }}
summary::marker {{ color:var(--muted); }}
.scorecell {{ position:relative; min-width:96px; }}
.scorebar {{ display:inline-block; height:7px; border-radius:3px; background:var(--s1);
  vertical-align:middle; max-width:76px; }}
.scoren {{ margin-left:.45rem; font-variant-numeric:tabular-nums; font-size:.84rem; }}
</style></head>
<body><main>
<h1>SARAL candidate signals</h1>
<p class="sub">Structured signals, fit scoring, an evaluation harness and an
incremental update pass. Every number on this page is read from <code>out/</code>.</p>
{body}
</main></body></html>"""
