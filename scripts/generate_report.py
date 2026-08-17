#!/usr/bin/env python3
"""
Render a campaign of rung summaries into a single self-contained HTML report.

    python3 scripts/generate_report.py results/t2-b300 -o report.html
    python3 scripts/generate_report.py results/t2-b300 --story results/t2-b300/story.json

Reads every rung-*.json in the directory (and duty.json, if present) and emits
one page: verdict tiles, the O1 ladder against the enforced limit, the O2 swing
comparison, and the full table. No external assets — the CSP on published
artifacts blocks CDN fonts and scripts, so everything is inline and the charts
are plain HTML/CSS rather than a charting library.

Narrative is optional. Without a story file you get the data; with one you get
the data plus the prose. Keeping the two separate means a re-run of a campaign
regenerates the numbers without disturbing the writing.
"""

import argparse
import glob
import html
import json
import os
import sys

# Scale headroom above the enforced limit, so the cap renders as a rule the
# bars approach rather than as the right edge of the track.
SCALE_HEADROOM = 1.09

# Rungs whose tensor work cannot reach a 5th-gen tensor core are drawn in the
# second hue. This is the distinction the whole B300 result turned on, so it is
# encoded in the chart rather than left to the caption.
def reaches_tensor_core(rung: dict) -> bool:
    """True when this rung's tensor work goes through cuBLAS, and therefore
    through tcgen05 on Blackwell. Older summaries do not echo the backend, so
    fall back to an explicit map in story.json, then to the rung name."""
    inv = rung.get("invocation", {})
    if "tensor_backend" in inv:
        return inv["tensor_backend"] == "cublas"
    if rung.get("_backend") is not None:
        return rung["_backend"] == "cublas"
    return "cublas" in rung["_name"]


def backend_label(rung: dict) -> str:
    if not has_tensor_work(rung):
        return "—"
    return "cuBLAS" if reaches_tensor_core(rung) else "wmma"


def has_tensor_work(rung: dict) -> bool:
    inv = rung.get("invocation", {})
    if "mix_tensor" in inv:
        return inv["mix_tensor"] > 0
    return rung.get("_tensor", "tensor" in rung["_name"])


def load_campaign(d: str):
    rungs = []
    for path in sorted(glob.glob(os.path.join(d, "rung-*.json"))):
        with open(path) as f:
            r = json.load(f)
        r["_name"] = os.path.basename(path)[len("rung-"):-len(".json")]
        # The runner does not yet echo tensor_backend into invocation; infer it
        # from the rung name the campaign used. Harmless when it does.
        r["_backend"] = None
        rungs.append(r)
    duty = None
    dp = os.path.join(d, "duty.json")
    if os.path.exists(dp):
        with open(dp) as f:
            duty = json.load(f)
    return rungs, duty


def fmt(x, n=1):
    return f"{x:,.{n}f}"


def bar_rows(rungs, limit_w, scale_max, cap_pct):
    out = []
    for r in sorted(rungs, key=lambda r: -r["power"]["avg_w"]):
        w = r["power"]["avg_w"]
        pct_track = 100.0 * w / scale_max
        inside = pct_track > 32
        cls = "bar" + ("" if reaches_tensor_core(r) else " alt") + (" wide" if inside else "")
        out.append(
            f'        <span class="blabel">{html.escape(r["_name"])}</span>\n'
            f'        <div class="track"><div class="{cls}" style="width:{pct_track:.2f}%">'
            f'<span class="bval">{fmt(w)} W</span></div>'
            f'<div class="ceiling" style="left:{cap_pct:.2f}%"></div></div>'
        )
    return "\n".join(out)


def table_rows(rungs):
    out = []
    ranked = sorted(rungs, key=lambda r: -r["power"]["avg_w"])
    for i, r in enumerate(ranked):
        p = r["power"]
        reasons = ", ".join(r["throttle"]["reasons"]) or "none"
        eff = r.get("efficiency", {})
        out.append(
            f'          <tr{" class=\"best\"" if i == 0 else ""}>'
            f'<td>{html.escape(r["_name"])}</td>'
            f'<td>{backend_label(r)}</td>'
            f'<td class="num">{fmt(p["avg_w"])}</td>'
            f'<td class="num">{fmt(p["pct_of_enforced_limit"])}</td>'
            f'<td class="num">{fmt(r["clocks"]["sm_avg_mhz"], 0)}</td>'
            f'<td>{html.escape(reasons)}</td></tr>'
        )
    return "\n".join(out)



# ---------------------------------------------------------------------------
# Sample trace: every sample, not a summary of them.
#
# A mean hides everything this project cares about. The trace renders the full
# NDJSON as a min/max envelope per pixel column with the mean drawn through it,
# so a spike narrower than one column still shows as envelope height rather
# than being averaged away. Naive downsampling would erase exactly the
# behaviour we are trying to measure.
# ---------------------------------------------------------------------------

def load_trace(path):
    """Read an NDJSON telemetry file into parallel lists."""
    t, pw, pi, temp, phase = [], [], [], [], []
    with open(path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            t.append(i)
            pw.append(r.get("power_w", 0.0))
            v = r.get("power_instant_w", -1.0)
            pi.append(v if v is not None and v >= 0 else None)
            temp.append(r.get("temp_c", 0.0))
            phase.append(r.get("phase", ""))
    return {"t": t, "pw": pw, "pi": pi, "temp": temp, "phase": phase}


def envelope(vals, cols):
    """Bucket into `cols` columns, returning (min, max, mean) per column.
    None entries are skipped; empty buckets are dropped."""
    n = len(vals)
    if n == 0:
        return []
    per = max(1, n / float(cols))
    out = []
    for c in range(cols):
        lo_i, hi_i = int(c * per), int((c + 1) * per)
        chunk = [v for v in vals[lo_i:max(hi_i, lo_i + 1)] if v is not None]
        if not chunk:
            continue
        out.append((c, min(chunk), max(chunk), sum(chunk) / len(chunk)))
    return out


def svg_trace(vals, cols, width, height, y_min, y_max, color, soft,
              rules=(), y_label="", x_label=""):
    """Envelope + mean line as inline SVG. `rules` is a list of
    (value, label, dashed) drawn as horizontal reference lines."""
    env = envelope(vals, cols)
    if not env:
        return '<p class="nodata">no samples</p>'

    pad_l, pad_r, pad_t, pad_b = 58, 92, 12, 26
    pw_ = width - pad_l - pad_r
    ph_ = height - pad_t - pad_b
    span = (y_max - y_min) or 1.0

    def x(c):  return pad_l + (c / float(max(cols - 1, 1))) * pw_
    def y(v):  return pad_t + ph_ - ((v - y_min) / span) * ph_

    top = " ".join(f"{x(c):.1f},{y(hi):.1f}" for c, lo, hi, m in env)
    bot = " ".join(f"{x(c):.1f},{y(lo):.1f}" for c, lo, hi, m in reversed(env))
    mean_pts = " ".join(f"{x(c):.1f},{y(m):.1f}" for c, lo, hi, m in env)

    # Reference lines sit at their true value, but their labels are nudged
    # apart when two rules land within a line-height of each other - the cap
    # and the peak often nearly coincide, which is exactly the case where
    # both labels need to stay readable.
    placed = []
    rule_svg = []
    for val, label, dashed in sorted([r for r in rules if y_min <= r[0] <= y_max],
                                     key=lambda r: -r[0]):
        yy = y(val)
        ly = yy + 4
        for prev in placed:
            if abs(ly - prev) < 13:
                ly = prev + 13
        placed.append(ly)
        rule_svg.append(
            f'<line x1="{pad_l}" y1="{yy:.1f}" x2="{pad_l + pw_}" y2="{yy:.1f}" '
            f'class="rule{" dash" if dashed else ""}"/>'
            f'<text x="{pad_l + pw_ + 8}" y="{ly:.1f}" class="rlab">{label}</text>')

    ticks = []
    for frac in (0.0, 0.5, 1.0):
        v = y_min + frac * span
        yy = y(v)
        ticks.append(f'<text x="{pad_l - 10}" y="{yy + 4:.1f}" class="ytick">{v:,.0f}</text>')

    return (
        f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" '
        f'role="img" aria-label="{html.escape(y_label)} over time">'
        f'<polygon class="env" points="{top} {bot}" fill="{soft}"/>'
        f'<polyline class="meanline" points="{mean_pts}" stroke="{color}"/>'
        + "".join(rule_svg) + "".join(ticks) +
        f'<text x="{pad_l}" y="{height - 6}" class="xlab">{html.escape(x_label)}</text>'
        f'<text x="6" y="{pad_t + 10}" class="ylab">{html.escape(y_label)}</text>'
        f'</svg>')


def trace_figure(dirpath, name, limit, title, caption):
    """Two stacked panels sharing an x axis: power, then temperature.
    Deliberately not a dual-axis chart - two measures of different scale get
    two plots, never two y-scales on one."""
    # Campaigns name traces either <rung>.ndjson or rung-<rung>.ndjson,
    # depending on how the sweep script invoked --out-metrics.
    path = None
    for cand in (name + ".ndjson", "rung-" + name + ".ndjson"):
        c = os.path.join(dirpath, cand)
        if os.path.exists(c):
            path = c
            break
    if path is None:
        return ""
    tr = load_trace(path)
    if not tr["t"]:
        return ""

    n = len(tr["t"])
    pw = [v for v in tr["pw"] if v is not None]
    peak, mean = max(pw), sum(pw) / len(pw)
    srt = sorted(pw)
    p99 = srt[min(len(srt) - 1, int(0.99 * len(srt)))]
    temps = [v for v in tr["temp"] if v]
    cols = 600

    power_svg = svg_trace(
        tr["pw"], cols, 1000, 250, 0, limit * 1.09,
        "var(--signal)", "var(--signal-soft)",
        rules=[(limit, f"{fmt(limit,0)} W cap", True),
               (peak, f"peak {fmt(peak)} W", False),
               (mean, f"mean {fmt(mean)} W", False)],
        y_label="watts", x_label=f"{n:,} samples")

    temp_svg = ""
    if temps:
        tmax, tmean = max(temps), sum(temps) / len(temps)
        temp_svg = (
            '<div class="panelgap"></div>' +
            svg_trace(tr["temp"], cols, 1000, 170,
                      max(0, min(temps) - 5), tmax + 5,
                      "var(--oxide)", "var(--oxide-soft)",
                      rules=[(tmax, f"peak {fmt(tmax,0)} °C", False),
                             (tmean, f"mean {fmt(tmean,0)} °C", False)],
                      y_label="°C", x_label="same window"))

    stats = (f'<div class="tracestats">'
             f'<span><b>{fmt(peak)} W</b> peak</span>'
             f'<span><b>{fmt(p99)} W</b> p99</span>'
             f'<span><b>{fmt(mean)} W</b> mean</span>'
             f'<span><b>{fmt(peak - min(pw))} W</b> observed range</span>'
             f'<span><b>{n:,}</b> samples</span>'
             + (f'<span><b>{fmt(max(temps),0)} °C</b> peak temp</span>' if temps else '')
             + '</div>')

    return (f'    <figure>\n      <div class="figtitle">{html.escape(title)}</div>\n'
            f'      <div class="chart trace">{stats}{power_svg}{temp_svg}</div>\n'
            f'      <figcaption>{html.escape(caption)}</figcaption>\n    </figure>')


CSS = """
  :root {
    --ground:#F4F6F5; --surface:#FFFFFF; --surface-sunk:#ECEFEE;
    --ink:#131A20; --ink-2:#3C4854; --muted:#61707C;
    --rule:#D5DBDA; --signal:#0A6E9B; --signal-soft:#0A6E9B1A;
    --oxide:#B4552A; --oxide-soft:#B4552A1A; --ceiling:#131A2073;
    --serif: Charter,"Bitstream Charter","Iowan Old Style",Cambria,Georgia,serif;
    --mono: ui-monospace,"SF Mono",SFMono-Regular,"JetBrains Mono",Menlo,Consolas,monospace;
    --measure:66ch;
    --step--1:0.815rem; --step-0:1rem; --step-1:1.27rem;
    --step-2:1.6rem; --step-3:2.1rem; --step-4:2.9rem;
  }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
      --ground:#0F141A; --surface:#161D25; --surface-sunk:#1D2630;
      --ink:#E7EDF2; --ink-2:#BDC8D2; --muted:#8B9AA7;
      --rule:#2A353F; --signal:#2E97BA; --signal-soft:#2E97BA26;
      --oxide:#C46F3F; --oxide-soft:#C46F3F26; --ceiling:#E7EDF280;
    }
  }
  :root[data-theme="dark"] {
    --ground:#0F141A; --surface:#161D25; --surface-sunk:#1D2630;
    --ink:#E7EDF2; --ink-2:#BDC8D2; --muted:#8B9AA7;
    --rule:#2A353F; --signal:#2E97BA; --signal-soft:#2E97BA26;
    --oxide:#C46F3F; --oxide-soft:#C46F3F26; --ceiling:#E7EDF280;
  }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--ground); color:var(--ink);
         font-family:var(--serif); font-size:var(--step-0); line-height:1.62;
         -webkit-font-smoothing:antialiased; }
  .wrap { max-width:60rem; margin:0 auto;
          padding:clamp(1.5rem,4vw,4rem) clamp(1.1rem,4vw,2.5rem) 6rem;
          display:flex; flex-direction:column; gap:3.25rem; }
  .prose { max-width:var(--measure); display:flex; flex-direction:column; gap:1rem; }
  p { margin:0; }
  .eyebrow { font-family:var(--mono); font-size:var(--step--1); letter-spacing:.13em;
             text-transform:uppercase; color:var(--muted); margin:0; }
  h1,h2,h3 { margin:0; text-wrap:balance; font-weight:600; line-height:1.14; }
  h1 { font-size:var(--step-4); letter-spacing:-.018em; }
  h2 { font-size:var(--step-2); letter-spacing:-.01em; }
  h3 { font-size:var(--step-1); }
  header { display:flex; flex-direction:column; gap:1.1rem; }
  .lede { font-size:var(--step-1); color:var(--ink-2); max-width:54ch; }
  .specbar { display:flex; flex-wrap:wrap; gap:0 2rem; font-family:var(--mono);
             font-size:var(--step--1); color:var(--muted);
             border-top:1px solid var(--rule); border-bottom:1px solid var(--rule);
             padding:.7rem 0; }
  .specbar span { white-space:nowrap; }
  .specbar b { font-weight:500; color:var(--ink-2); }
  .tiles { display:grid; grid-template-columns:repeat(auto-fit,minmax(13.5rem,1fr));
           gap:1px; background:var(--rule); border:1px solid var(--rule); }
  .tile { background:var(--surface); padding:1.35rem 1.4rem;
          display:flex; flex-direction:column; gap:.4rem; }
  .tile .k { font-family:var(--mono); font-size:var(--step--1); letter-spacing:.1em;
             text-transform:uppercase; color:var(--muted); }
  .tile .v { font-family:var(--mono); font-size:var(--step-3); font-weight:600;
             letter-spacing:-.02em; font-variant-numeric:tabular-nums; line-height:1; }
  .tile .v.sig { color:var(--signal); } .tile .v.ox { color:var(--oxide); }
  .tile .n { font-size:var(--step--1); color:var(--muted); line-height:1.45; }
  figure { margin:0; display:flex; flex-direction:column; gap:1rem; }
  figcaption { font-size:var(--step--1); color:var(--muted); max-width:var(--measure); }
  .chart { background:var(--surface); border:1px solid var(--rule);
           padding:1.5rem 1.4rem 1.1rem; overflow-x:auto; }
  .bars { display:grid; grid-template-columns:minmax(11rem,auto) 1fr;
          gap:.42rem 1rem; align-items:center; min-width:32rem; }
  .blabel { font-family:var(--mono); font-size:var(--step--1); color:var(--ink-2);
            text-align:right; white-space:nowrap; }
  .blabel .sub { color:var(--muted); }
  .ceiling { position:absolute; top:0; bottom:0; width:0; pointer-events:none;
             border-left:1.5px dashed var(--ceiling); }
  .track { position:relative; height:1.55rem; background:var(--surface-sunk); }
  .bar { position:absolute; inset:0 auto 0 0; border-radius:0 4px 4px 0;
         background:var(--signal); transition:filter 120ms ease; }
  .bar.alt { background:var(--oxide); }
  .track:hover .bar { filter:brightness(1.12); }
  .bval { font-family:var(--mono); font-size:var(--step--1);
          font-variant-numeric:tabular-nums; color:var(--ink-2);
          position:absolute; left:calc(100% + .5rem); top:50%;
          transform:translateY(-50%); white-space:nowrap; }
  .bar.wide .bval { left:auto; right:.5rem; color:#fff; }
  .axis { grid-column:2; position:relative; height:1.3rem;
          border-top:1px solid var(--rule); margin-top:.35rem; }
  .tick { position:absolute; top:.2rem; transform:translateX(-50%);
          font-family:var(--mono); font-size:var(--step--1); color:var(--muted);
          white-space:nowrap; }
  .tick.cap { color:var(--ink-2); }
  .legend { display:flex; flex-wrap:wrap; gap:1.1rem; font-family:var(--mono);
            font-size:var(--step--1); color:var(--ink-2); padding-top:.9rem;
            margin-top:.6rem; border-top:1px solid var(--rule); }
  .legend i { display:inline-block; width:.85rem; height:.5rem; border-radius:2px;
              background:var(--signal); margin-right:.45rem; }
  .legend i.alt { background:var(--oxide); }
  .legend i.dash { width:1rem; height:0; border-top:1.5px dashed var(--ceiling);
                   border-radius:0; }
  .trace { padding:1.3rem 1.2rem 1rem; }
  .trace svg { display:block; }
  .env { stroke:none; }
  .meanline { fill:none; stroke-width:1.5; }
  .rule { stroke:var(--ink-2); stroke-width:1; opacity:.65; }
  .rule.dash { stroke-dasharray:5 4; opacity:.9; }
  .rlab, .ytick, .xlab, .ylab { font-family:var(--mono); font-size:11px;
                                fill:var(--muted); }
  .rlab { fill:var(--ink-2); }
  .ytick { text-anchor:end; }
  .tracestats { display:flex; flex-wrap:wrap; gap:1.4rem; font-family:var(--mono);
                font-size:var(--step--1); color:var(--muted);
                padding-bottom:1rem; margin-bottom:.4rem;
                border-bottom:1px solid var(--rule); }
  .tracestats b { color:var(--ink); font-weight:600;
                  font-variant-numeric:tabular-nums; }
  .figtitle { font-family:var(--mono); font-size:var(--step--1);
              letter-spacing:.1em; text-transform:uppercase; color:var(--muted); }
  .panelgap { height:.7rem; }
  .nodata { font-family:var(--mono); font-size:var(--step--1); color:var(--muted); }
  .rangerow { position:relative; height:2.1rem; background:var(--surface-sunk); }
  .band { position:absolute; top:0; bottom:0; background:var(--signal-soft);
          border-left:3px solid var(--signal); border-right:3px solid var(--signal); }
  .band.alt { background:var(--oxide-soft); border-color:var(--oxide); }
  .bandnum { position:absolute; top:50%; transform:translateY(-50%);
             font-family:var(--mono); font-size:var(--step--1);
             font-variant-numeric:tabular-nums; color:var(--ink-2);
             background:var(--surface); padding:0 .3rem; }
  .tablewrap { overflow-x:auto; border:1px solid var(--rule); background:var(--surface); }
  table { border-collapse:collapse; width:100%; font-family:var(--mono);
          font-size:var(--step--1); }
  th,td { text-align:right; padding:.5rem .85rem; border-bottom:1px solid var(--rule);
          white-space:nowrap; }
  th:first-child, td:first-child { text-align:left; }
  thead th { color:var(--muted); font-weight:500; letter-spacing:.06em;
             text-transform:uppercase; position:sticky; top:0; background:var(--surface); }
  tbody tr:last-child td { border-bottom:0; }
  tbody tr:hover td { background:var(--surface-sunk); }
  td.num { font-variant-numeric:tabular-nums; }
  tr.best td { color:var(--signal); font-weight:600; }
  .callout { border-left:3px solid var(--oxide); background:var(--surface);
             padding:1.1rem 1.3rem; display:flex; flex-direction:column; gap:.55rem;
             max-width:var(--measure); }
  .callout .eyebrow { color:var(--oxide); }
  section { display:flex; flex-direction:column; gap:1.3rem; }
  .divider { border:0; border-top:1px solid var(--rule); margin:0; }
  ul { margin:0; padding-left:1.15rem; display:flex; flex-direction:column;
       gap:.5rem; max-width:var(--measure); }
  li::marker { color:var(--muted); }
  code { font-family:var(--mono); font-size:.88em; background:var(--surface-sunk);
         padding:.08em .34em; border-radius:3px; }
  pre { margin:0; background:var(--surface); border:1px solid var(--rule);
        padding:1.1rem 1.2rem; overflow-x:auto; font-family:var(--mono);
        font-size:var(--step--1); line-height:1.55; color:var(--ink-2); }
  pre code { background:none; padding:0; font-size:1em; }
  pre .c { color:var(--muted); font-style:italic; }
  pre .k { color:var(--signal); }
  .codecap { font-family:var(--mono); font-size:var(--step--1); color:var(--muted);
             display:flex; gap:.6rem; align-items:baseline; }
  .codecap b { font-weight:500; color:var(--ink-2); }
  .wideblock { display:flex; flex-direction:column; gap:.7rem; }
  .cmp th:first-child, .cmp td:first-child { color:var(--muted); font-weight:500; }
  .cmp td { vertical-align:top; white-space:normal; min-width:11rem; }
  .cmp thead th { text-align:left; }
  .cmp td, .cmp th { text-align:left; }
  footer { font-family:var(--mono); font-size:var(--step--1); color:var(--muted);
           border-top:1px solid var(--rule); padding-top:1rem;
           display:flex; flex-direction:column; gap:.3rem; }
  a { color:var(--signal); }
  a:focus-visible { outline:2px solid var(--signal); outline-offset:2px; }
  @media (prefers-reduced-motion: reduce) { * { transition:none !important; } }
"""


def render(rungs, duty, story, out_path, campaign_dir=None):
    # story.json may carry {"backends": {"<rung>": "cublas"|"wmma"|"none"}} for
    # campaigns whose summaries predate the runner echoing the backend.
    bmap = (story or {}).get("backends", {})
    for r in rungs:
        if r["_name"] in bmap:
            r["_backend"] = bmap[r["_name"]]
            r["_tensor"] = bmap[r["_name"]] != "none"
    if not rungs:
        sys.exit("no rung-*.json found")

    best = max(rungs, key=lambda r: r["power"]["avg_w"])
    dev = best["device"]
    limit = best["power"]["enforced_limit_w"]
    scale_max = limit * SCALE_HEADROOM
    cap_pct = 100.0 / SCALE_HEADROOM

    s = story or {}
    title = s.get("title", f'{dev["name"]} power ceiling')
    lede = s.get("lede", "")

    # --- tiles -----------------------------------------------------------
    tiles = [
        ("Best sustained", f'{fmt(best["power"]["pct_of_enforced_limit"])}%', "sig",
         f'{fmt(best["power"]["avg_w"])} W of a {fmt(limit,0)} W cap · {html.escape(best["_name"])}'
         + (". No throttle bit set, so the shortfall is the workload, not the controller."
            if not best["throttle"]["any_throttled"] else ". Throttled — see the table.")),
    ]
    if duty:
        dp = duty["power"]
        peak = max(dp["peak_w"], dp.get("peak_instant_w", 0))
        tiles.append(("Peak transient", f'{fmt(100.0*peak/limit)}%', "sig",
                      f'{fmt(peak)} W instantaneous during a '
                      f'{fmt(duty["transient"]["duty_on_ms"],0)} ms burst.'))
    for label in s.get("extra_tiles", []):
        tiles.append((label["k"], label["v"], label.get("tone", "ox"), label["n"]))

    tile_html = "\n".join(
        f'    <div class="tile"><span class="k">{html.escape(k)}</span>'
        f'<span class="v {tone}">{html.escape(v)}</span>'
        f'<span class="n">{n}</span></div>'
        for k, v, tone, n in tiles)

    # --- swing figure ----------------------------------------------------
    swing_html = ""
    if duty and s.get("swing"):
        sw = s["swing"]  # {"instant":[min,max], "usage":[min,max]}
        def band(kind, lo, hi, alt, cap_pct=cap_pct):
            l = 100.0 * lo / scale_max
            r = 100.0 - 100.0 * hi / scale_max
            return (f'          <span class="blabel">{html.escape(kind)}</span>\n'
                    f'          <div class="rangerow">'
                    f'<div class="ceiling" style="left:{cap_pct:.2f}%"></div>'
                    f'<div class="band{" alt" if alt else ""}" '
                    f'style="left:{l:.2f}%; right:{r:.2f}%"></div>'
                    f'<span class="bandnum" style="left:{l+0.5:.2f}%">{fmt(lo)} W</span>'
                    f'<span class="bandnum" style="right:{r+0.5:.2f}%">{fmt(hi)} W</span>'
                    f'</div>')
        swing_html = f"""
  <section>
    <p class="eyebrow">O2 · transients</p>
    <h2>{html.escape(s.get("swing_title", "How much can one GPU swing?"))}</h2>
    <div class="prose">{s.get("swing_prose", "")}</div>
    <figure>
      <div class="chart">
        <div class="bars">
{band("POWER_INSTANT", *sw["instant"], False)}
{band("GetPowerUsage averaged", *sw["usage"], True)}
          <div class="axis">
            <span class="tick" style="left:0">0</span>
            <span class="tick cap" style="left:{cap_pct:.2f}%">{fmt(limit,0)} W cap</span>
          </div>
        </div>
        <div class="legend">
          <span><i></i>instantaneous — swing {fmt(sw["instant"][1]-sw["instant"][0])} W</span>
          <span><i class="alt"></i>driver-averaged — swing {fmt(sw["usage"][1]-sw["usage"][0])} W</span>
          <span><i class="dash"></i>enforced limit</span>
        </div>
      </div>
      <figcaption>{html.escape(s.get("swing_caption", ""))}</figcaption>
    </figure>
  </section>"""

    def render_sections(key):
        """Each section is: eyebrow, heading, prose, an optional full-width
        block (code listings, wide comparison tables), then optional closing
        prose. The wide block sits outside .prose so it is not squeezed to the
        66ch measure."""
        out = []
        for sec in s.get(key, []):
            parts = [
                f'  <section>',
                f'    <p class="eyebrow">{html.escape(sec.get("eyebrow", ""))}</p>',
                f'    <h2>{html.escape(sec["h2"])}</h2>',
            ]
            if sec.get("body"):
                parts.append(f'    <div class="prose">{sec["body"]}</div>')
            if sec.get("wide"):
                parts.append(f'    <div class="wideblock">{sec["wide"]}</div>')
            if sec.get("body_after"):
                parts.append(f'    <div class="prose">{sec["body_after"]}</div>')
            parts.append('  </section>')
            out.append("\n".join(parts))
        return "\n".join(out)

    sections = render_sections("sections")
    closing = render_sections("closing")

    # Sample traces, for every rung that kept its NDJSON. Without --out-metrics
    # there is nothing to draw; the summary alone cannot show spikiness.
    tdir = campaign_dir or os.path.dirname(out_path) or "."
    trace_figs = []
    for r in sorted(rungs, key=lambda r: -r["power"]["avg_w"]):
        fig = trace_figure(tdir, r["_name"], limit,
                           f'{r["_name"]} — every sample',
                           s.get("trace_caption", ""))
        if fig:
            trace_figs.append(fig)
    duty_fig = trace_figure(tdir, "duty", limit,
                            "duty cycle — every sample",
                            s.get("trace_caption", ""))
    if duty_fig:
        trace_figs.insert(0, duty_fig)

    if trace_figs:
        traces = ('  <section>\n    <p class="eyebrow">Raw telemetry</p>\n'
                  '    <h2>' + html.escape(s.get("trace_title", "Every sample")) + '</h2>\n'
                  '    <div class="prose">' + s.get("trace_prose", "") + '</div>\n'
                  + "\n".join(trace_figs) + '\n  </section>')
    else:
        traces = ('  <section>\n    <p class="eyebrow">Raw telemetry</p>\n'
                  '    <h2>Every sample</h2>\n    <div class="prose">'
                  '<p>No per-sample traces in this campaign — the rungs were run '
                  'without <code>--out-metrics</code>, so only the aggregates '
                  'survive. Summary statistics cannot show spikiness; keep the '
                  'NDJSON.</p></div>\n  </section>')

    page = f"""<title>{html.escape(title)}</title>
<style>{CSS}</style>
<div class="wrap">
  <header>
    <p class="eyebrow">{html.escape(s.get("eyebrow", "gpu-power-lab · field report"))}</p>
    <h1>{html.escape(s.get("headline", title))}</h1>
    <p class="lede">{lede}</p>
    <div class="specbar">
      <span><b>Device</b> {html.escape(dev["name"])}</span>
      <span><b>Limit</b> {fmt(limit,0)} W</span>
      <span><b>Driver</b> {html.escape(dev["driver"])}</span>
      <span><b>CUDA</b> {html.escape(dev["cuda_runtime"])}</span>
      <span><b>Rungs</b> {len(rungs)}</span>
      <span><b>Date</b> {html.escape(s.get("date", ""))}</span>
    </div>
  </header>

  <div class="tiles">
{tile_html}
  </div>

  <hr class="divider">

  <section>
    <p class="eyebrow">O1 · the ceiling</p>
    <h2>{html.escape(s.get("ladder_title", "Mean power by workload"))}</h2>
    <div class="prose">{s.get("ladder_prose", "")}</div>
    <figure>
      <div class="chart">
        <div class="bars">
{bar_rows(rungs, limit, scale_max, cap_pct)}
          <div class="axis">
            <span class="tick" style="left:0">0</span>
            <span class="tick" style="left:{100*550/scale_max:.2f}%">550 W</span>
            <span class="tick cap" style="left:{cap_pct:.2f}%">{fmt(limit,0)} W cap</span>
          </div>
        </div>
        <div class="legend">
          <span><i></i>reaches the 5th-gen tensor core (cuBLAS → tcgen05)</span>
          <span><i class="alt"></i>cannot reach it (warp-synchronous wmma, or no tensor work)</span>
          <span><i class="dash"></i>enforced limit</span>
        </div>
      </div>
      <figcaption>{html.escape(s.get("ladder_caption", ""))}</figcaption>
    </figure>
  </section>

{sections}
{swing_html}

{traces}

  <section>
    <p class="eyebrow">Full data</p>
    <h2>Every rung</h2>
    <div class="tablewrap">
      <table>
        <thead><tr><th>Rung</th><th>Tensor backend</th><th>Mean W</th>
          <th>% of limit</th><th>SM clock</th><th>Peak °C</th>
          <th>EDP</th><th>EDPp</th><th>Throttle</th></tr></thead>
        <tbody>
{table_rows(rungs)}
        </tbody>
      </table>
    </div>
  </section>

{closing}

  <footer>
    <span>{html.escape(s.get("footer", "gpu-power-lab"))}</span>
  </footer>
</div>
"""
    with open(out_path, "w") as f:
        f.write(page)
    print(f"wrote {out_path}  ({len(rungs)} rungs, limit {fmt(limit,0)} W)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("campaign_dir")
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("--story", default=None,
                    help="optional JSON with headline/prose; defaults to "
                         "<campaign_dir>/story.json when present")
    a = ap.parse_args()

    rungs, duty = load_campaign(a.campaign_dir)
    story_path = a.story or os.path.join(a.campaign_dir, "story.json")
    story = None
    if os.path.exists(story_path):
        with open(story_path) as f:
            story = json.load(f)
    out = a.out or os.path.join(a.campaign_dir, "report.html")
    render(rungs, duty, story, out, campaign_dir=a.campaign_dir)


if __name__ == "__main__":
    main()
