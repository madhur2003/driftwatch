"""A self-contained monitoring dashboard served by the API.

A single HTML page (inline CSS + vanilla JS, no external assets) polls one JSON
endpoint and renders the flag, model quality, the live forecast, and the drift
history. It is kept as an embedded string so it ships in the wheel and the
container with no package-data or static-file plumbing.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd

from . import drift as D
from . import features as F
from . import model as M


def _report_summary(row) -> dict:
    return {
        "id": row["id"],
        "generated_at": row["generated_at_utc"],
        "status": row["status"],
        "flagged": bool(row["flagged"]),
        "value_psi": row["value_psi"],
        "ks_statistic": row["ks_statistic"],
        "error_mae": row["error_mae"],
        "baseline_mae": row["baseline_mae"],
        "mae_ratio": row["mae_ratio"],
    }


def dashboard_data(
    settings,
    artifact: M.Artifact | None,
    *,
    respondent: str = "PJM",
    data_type: str = "D",
    history_hours: int = 72,
    horizon: int = 24,
    window_hours: int = 168,
    history_limit: int = 40,
) -> dict:
    """Assemble everything the dashboard renders in a single payload."""
    from . import db

    model_block: dict = {"loaded": artifact is not None}
    if artifact is not None:
        model_block.update(
            {
                "trained_at": artifact.trained_at_utc,
                "metrics": artifact.metadata.get("metrics"),
                "n_train": artifact.metadata.get("n_train"),
                "n_val": artifact.metadata.get("n_val"),
                "train_start": artifact.metadata.get("train_start_utc"),
                "train_end": artifact.metadata.get("train_end_utc"),
                "has_reference": bool(artifact.reference),
            }
        )

    with db.get_connection(settings.db_path) as conn:
        observations = db.observation_count(conn)
        latest = db.latest_period(conn, respondent, data_type)
        rows = db.select_observations(conn, respondent, data_type)
        history = [
            _report_summary(r)
            for r in db.recent_drift_reports(conn, respondent=respondent, limit=history_limit)
        ]

    frame = F.frame_from_observations(rows)

    actuals: list[dict] = []
    forecast: list[dict] = []
    drift_block: dict | None = None

    if not frame.empty:
        last = frame.index.max()
        recent = frame[frame.index >= last - pd.Timedelta(hours=history_hours - 1)]
        actuals = [
            {"period": idx.isoformat(), "value": (None if pd.isna(v) else round(float(v), 1))}
            for idx, v in recent["value"].items()
        ]

        if artifact is not None:
            periods = [
                pd.Timestamp(last).to_pydatetime() + timedelta(hours=k)
                for k in range(1, horizon + 1)
            ]
            preds = M.predict(artifact, frame, periods)
            forecast = [
                {
                    "period": pd.Timestamp(idx).isoformat(),
                    "value": (None if pd.isna(v) else round(float(v), 1)),
                }
                for idx, v in preds.items()
            ]

            if artifact.reference:
                current = frame[frame.index >= last - pd.Timedelta(hours=window_hours - 1)]
                error_periods = frame.index[
                    (frame.index >= last - pd.Timedelta(hours=window_hours - 1))
                    & frame["value"].notna()
                ]
                report = D.analyze(
                    artifact,
                    respondent=respondent,
                    data_type=data_type,
                    current_frame=current,
                    history=frame,
                    error_periods=list(error_periods),
                )
                drift_block = report.to_dict()
            else:
                drift_block = {"status": None, "detail": "no drift reference; retrain"}

    return {
        "respondent": respondent,
        "data_type": data_type,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "observations": observations,
        "latest_period": latest,
        "model": model_block,
        "drift": drift_block,
        "series": {"history": actuals, "forecast": forecast},
        "history": history,
    }


DASHBOARD_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Driftwatch — monitoring</title>
<style>
  :root {
    --bg: #f6f7f9; --panel: #ffffff; --ink: #1a1d21; --muted: #6b7280;
    --line: #e5e7eb; --accent: #2563eb; --accent2: #9333ea;
    --ok: #16a34a; --warn: #d97706; --alert: #dc2626;
    --ok-bg: #dcfce7; --warn-bg: #fef3c7; --alert-bg: #fee2e2; --none-bg: #e5e7eb;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #0e1116; --panel: #171b22; --ink: #e6e9ee; --muted: #9aa4b2;
      --line: #262c36; --accent: #60a5fa; --accent2: #c084fc;
      --ok: #4ade80; --warn: #fbbf24; --alert: #f87171;
      --ok-bg: #052e16; --warn-bg: #3a2a06; --alert-bg: #3b0d0d; --none-bg: #232833;
    }
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--ink);
         font: 14px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
  .wrap { max-width: 1040px; margin: 0 auto; padding: 24px 20px 48px; }
  header { display: flex; align-items: baseline; justify-content: space-between; flex-wrap: wrap; gap: 8px; }
  h1 { font-size: 20px; margin: 0; letter-spacing: .2px; }
  h1 span { color: var(--muted); font-weight: 500; }
  .sub { color: var(--muted); font-size: 12px; }
  .btn { border: 1px solid var(--line); background: var(--panel); color: var(--ink);
         border-radius: 8px; padding: 6px 12px; font-size: 13px; cursor: pointer; }
  .btn:hover { border-color: var(--accent); }
  .banner { margin: 18px 0; border-radius: 14px; padding: 18px 20px; display: flex;
            align-items: center; gap: 16px; border: 1px solid var(--line); }
  .banner .dot { width: 14px; height: 14px; border-radius: 50%; flex: none; }
  .banner .label { font-size: 22px; font-weight: 700; letter-spacing: .3px; }
  .banner .why { color: var(--muted); font-size: 13px; }
  .s-ok    { background: var(--ok-bg); }    .s-ok .dot { background: var(--ok); }       .s-ok .label { color: var(--ok); }
  .s-warn  { background: var(--warn-bg); }  .s-warn .dot { background: var(--warn); }    .s-warn .label { color: var(--warn); }
  .s-alert { background: var(--alert-bg); } .s-alert .dot { background: var(--alert); }  .s-alert .label { color: var(--alert); }
  .s-none  { background: var(--none-bg); }  .s-none .dot { background: var(--muted); }   .s-none .label { color: var(--muted); }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; }
  .tile { background: var(--panel); border: 1px solid var(--line); border-radius: 12px; padding: 14px 16px; }
  .tile .k { color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: .6px; }
  .tile .v { font-size: 22px; font-weight: 650; margin-top: 4px; }
  .tile .v small { font-size: 13px; font-weight: 500; color: var(--muted); }
  .card { background: var(--panel); border: 1px solid var(--line); border-radius: 14px;
          padding: 16px 18px; margin-top: 18px; }
  .card h2 { font-size: 14px; margin: 0 0 4px; }
  .card .hint { color: var(--muted); font-size: 12px; margin: 0 0 12px; }
  .legend { display: flex; gap: 16px; font-size: 12px; color: var(--muted); margin-bottom: 6px; }
  .legend i { display: inline-block; width: 18px; height: 3px; border-radius: 2px; vertical-align: middle; margin-right: 6px; }
  svg { width: 100%; height: auto; display: block; }
  .chart-empty { color: var(--muted); font-size: 13px; padding: 24px 0; text-align: center; }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th, td { text-align: right; padding: 7px 10px; border-bottom: 1px solid var(--line); white-space: nowrap; }
  th:first-child, td:first-child { text-align: left; }
  th { color: var(--muted); font-weight: 600; font-size: 11px; text-transform: uppercase; letter-spacing: .5px; }
  .pill { display: inline-block; padding: 2px 9px; border-radius: 999px; font-size: 11px; font-weight: 700; text-transform: uppercase; }
  .pill.ok { background: var(--ok-bg); color: var(--ok); }
  .pill.warn { background: var(--warn-bg); color: var(--warn); }
  .pill.alert { background: var(--alert-bg); color: var(--alert); }
  .scroll { overflow-x: auto; }
  footer { color: var(--muted); font-size: 12px; margin-top: 22px; }
  a { color: var(--accent); }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>Driftwatch <span id="resp">·</span></h1>
    <div>
      <span class="sub" id="updated">loading…</span>
      <button class="btn" id="refresh">Refresh</button>
    </div>
  </header>

  <div class="banner s-none" id="banner">
    <span class="dot"></span>
    <div>
      <div class="label" id="status-label">—</div>
      <div class="why" id="status-why">Contacting service…</div>
    </div>
  </div>

  <div class="grid" id="tiles"></div>

  <div class="card">
    <h2>Demand — recent actuals &amp; forecast</h2>
    <p class="hint">Last 72h observed, next 24h predicted. The dashed line begins at "now".</p>
    <div class="legend">
      <span><i style="background:var(--accent)"></i>actual</span>
      <span><i style="background:var(--accent2)"></i>forecast</span>
    </div>
    <div id="forecast-chart"><div class="chart-empty">no data</div></div>
  </div>

  <div class="card">
    <h2>Drift history</h2>
    <p class="hint">Recorded drift checks (newest first) — the operational trail written by <code>driftwatch drift</code> / <code>POST /drift</code>.</p>
    <div class="scroll"><table id="history">
      <thead><tr><th>Time (UTC)</th><th>Status</th><th>PSI</th><th>KS</th><th>Recent MAE</th><th>vs baseline</th></tr></thead>
      <tbody><tr><td colspan="6" class="chart-empty">no drift checks recorded yet — run <code>driftwatch drift</code></td></tr></tbody>
    </table></div>
  </div>

  <footer>
    Driftwatch monitoring · <a href="/docs">API docs</a> · auto-refreshes every 30s
  </footer>
</div>

<script>
const params = new URLSearchParams(location.search);
const RESPONDENT = params.get("respondent") || "PJM";
const fmtNum = (x, d=0) => (x==null ? "—" : Number(x).toLocaleString(undefined, {maximumFractionDigits:d}));
const fmtPct = (x, d=1) => (x==null ? "—" : (100*x).toFixed(d) + "%");
const shortTime = s => s ? s.replace("T"," ").replace(/:\d\d(\+00:00|Z)$/,"") : "—";

function setBanner(status, why) {
  const b = document.getElementById("banner");
  const cls = status ? status : "none";
  b.className = "banner s-" + cls;
  document.getElementById("status-label").textContent = status ? status.toUpperCase() : "NO MODEL";
  document.getElementById("status-why").textContent = why;
}

function tile(k, v, sub) {
  return `<div class="tile"><div class="k">${k}</div><div class="v">${v}${sub?` <small>${sub}</small>`:""}</div></div>`;
}

function renderTiles(d) {
  const m = d.model || {};
  const metrics = m.metrics || {};
  const drift = d.drift || {};
  const err = drift.error || {};
  const psi = (drift.features && drift.features[0]) ? drift.features[0].psi : null;
  const tiles = [
    tile("Model MAPE", metrics.mape!=null ? metrics.mape.toFixed(2)+"%" : "—"),
    tile("Skill vs naive", metrics.skill_vs_baseline!=null ? fmtPct(metrics.skill_vs_baseline) : "—"),
    tile("Current PSI", psi!=null ? Number(psi).toFixed(3) : "—"),
    tile("Recent error", err.mae_ratio!=null ? err.mae_ratio.toFixed(2)+"×" : "—", "vs baseline"),
    tile("Observations", fmtNum(d.observations)),
    tile("Model trained", m.trained_at ? shortTime(m.trained_at) : "—"),
  ];
  document.getElementById("tiles").innerHTML = tiles.join("");
}

function lineChart(el, series, opts={}) {
  const all = series.flatMap(s => s.points);
  if (!all.length) { el.innerHTML = '<div class="chart-empty">no data</div>'; return; }
  const W = 960, H = 300, pad = {l:64, r:16, t:12, b:28};
  const xs = all.map(p=>p.x), ys = all.map(p=>p.y);
  let xmin=Math.min(...xs), xmax=Math.max(...xs), ymin=Math.min(...ys), ymax=Math.max(...ys);
  const yp = (ymax-ymin)*0.12 || 1; ymin-=yp; ymax+=yp;
  const sx = x => pad.l + (xmax===xmin?0:(x-xmin)/(xmax-xmin))*(W-pad.l-pad.r);
  const sy = y => H-pad.b - (ymax===ymin?0:(y-ymin)/(ymax-ymin))*(H-pad.t-pad.b);
  let svg = `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="xMidYMid meet" role="img">`;
  for (let i=0;i<=4;i++){
    const y = ymin + (ymax-ymin)*i/4, py = sy(y);
    svg += `<line x1="${pad.l}" y1="${py}" x2="${W-pad.r}" y2="${py}" stroke="var(--line)" stroke-width="1"/>`;
    svg += `<text x="${pad.l-8}" y="${py+4}" text-anchor="end" font-size="11" fill="var(--muted)">${fmtNum(y)}</text>`;
  }
  if (opts.nowX!=null) {
    const nx = sx(opts.nowX);
    svg += `<line x1="${nx}" y1="${pad.t}" x2="${nx}" y2="${H-pad.b}" stroke="var(--muted)" stroke-width="1" stroke-dasharray="3 3"/>`;
    svg += `<text x="${nx+4}" y="${pad.t+12}" font-size="10" fill="var(--muted)">now</text>`;
  }
  for (const s of series) {
    const pts = s.points.filter(p=>p.y!=null).map(p=>`${sx(p.x).toFixed(1)},${sy(p.y).toFixed(1)}`).join(" ");
    if (pts) svg += `<polyline points="${pts}" fill="none" stroke="${s.color}" stroke-width="2" ${s.dashed?'stroke-dasharray="5 4"':''} stroke-linejoin="round" stroke-linecap="round"/>`;
  }
  const t0 = new Date(xmin), t1 = new Date(xmax);
  svg += `<text x="${pad.l}" y="${H-8}" font-size="11" fill="var(--muted)">${shortTime(t0.toISOString())}</text>`;
  svg += `<text x="${W-pad.r}" y="${H-8}" text-anchor="end" font-size="11" fill="var(--muted)">${shortTime(t1.toISOString())}</text>`;
  svg += `</svg>`;
  el.innerHTML = svg;
}

function renderChart(d) {
  const hist = (d.series.history||[]).map(p=>({x:Date.parse(p.period), y:p.value}));
  const fc = (d.series.forecast||[]).map(p=>({x:Date.parse(p.period), y:p.value}));
  const nowX = hist.length ? hist[hist.length-1].x : (fc.length?fc[0].x:null);
  const series = [
    {name:"actual", color:"var(--accent)", points: hist},
    {name:"forecast", color:"var(--accent2)", dashed:true,
     points: (hist.length?[hist[hist.length-1]]:[]).concat(fc)},
  ];
  lineChart(document.getElementById("forecast-chart"), series, {nowX});
}

function renderHistory(d) {
  const tb = document.querySelector("#history tbody");
  const rows = d.history || [];
  if (!rows.length) return;
  tb.innerHTML = rows.map(r => `<tr>
    <td>${shortTime(r.generated_at)}</td>
    <td><span class="pill ${r.status}">${r.status}</span></td>
    <td>${r.value_psi!=null ? Number(r.value_psi).toFixed(3) : "—"}</td>
    <td>${r.ks_statistic!=null ? Number(r.ks_statistic).toFixed(3) : "—"}</td>
    <td>${fmtNum(r.error_mae)}</td>
    <td>${r.mae_ratio!=null ? r.mae_ratio.toFixed(2)+"×" : "—"}</td>
  </tr>`).join("");
}

function statusWhy(d) {
  const m = d.model || {};
  if (!m.loaded) return "No trained model is loaded — POST to /train, then retrain the reference.";
  const drift = d.drift || {};
  if (!drift || drift.status == null) return drift && drift.detail ? drift.detail : "Model loaded; awaiting data.";
  const psi = (drift.features && drift.features[0]) ? drift.features[0].psi : null;
  const ratio = drift.error ? drift.error.mae_ratio : null;
  return `Input PSI ${psi!=null?Number(psi).toFixed(3):"—"} · recent error ${ratio!=null?ratio.toFixed(2)+"× baseline":"—"} `
       + `· window ${shortTime(drift.current_start_utc)} → ${shortTime(drift.current_end_utc)}`;
}

async function load() {
  try {
    const res = await fetch(`/dashboard/data?respondent=${encodeURIComponent(RESPONDENT)}`);
    const d = await res.json();
    document.getElementById("resp").textContent = "· " + d.respondent;
    const status = d.drift && d.drift.status ? d.drift.status : null;
    setBanner(status, statusWhy(d));
    renderTiles(d);
    renderChart(d);
    renderHistory(d);
    document.getElementById("updated").textContent = "updated " + new Date().toLocaleTimeString();
  } catch (e) {
    setBanner(null, "Failed to reach the service: " + e);
  }
}

document.getElementById("refresh").addEventListener("click", load);
load();
setInterval(load, 30000);
</script>
</body>
</html>
"""
