from __future__ import annotations

from pathlib import Path

import duckdb
from jinja2 import Environment, select_autoescape

from legal_eval.models import RunSummary, project_root

TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Legal Agent Eval Dashboard</title>
  <style>
    :root { color-scheme: light dark; --bg:#f8fafc; --panel:#ffffff; --ink:#172033; --muted:#64748b; --line:#dbe3ef; --accent:#2f6fed; --ok:#0e9f6e; --bad:#c2410c; }
    @media (prefers-color-scheme: dark) { :root { --bg:#10141c; --panel:#171d29; --ink:#eef4ff; --muted:#9aa8bc; --line:#2a3446; --accent:#7aa2ff; --ok:#38c989; --bad:#ff955c; } }
    * { box-sizing: border-box; }
    body { margin:0; background:var(--bg); color:var(--ink); font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    main { max-width:1180px; margin:0 auto; padding:32px 20px 48px; }
    header { display:flex; justify-content:space-between; gap:20px; align-items:flex-end; margin-bottom:24px; }
    h1 { font-size:28px; margin:0 0 8px; letter-spacing:0; }
    p { color:var(--muted); margin:0; }
    .grid { display:grid; gap:16px; }
    .metrics { grid-template-columns: repeat(4, minmax(0, 1fr)); }
    .panel { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:18px; box-shadow:0 14px 30px rgba(15,23,42,.06); }
    .metric strong { display:block; font-size:26px; line-height:1.1; }
    .metric span { color:var(--muted); font-size:13px; }
    .charts { grid-template-columns: 1.2fr .8fr; margin-top:16px; }
    .bar-row { display:grid; grid-template-columns: 160px 1fr 54px; gap:12px; align-items:center; margin:14px 0; }
    .track { height:20px; border-radius:999px; background:color-mix(in srgb, var(--line) 70%, transparent); overflow:hidden; border:1px solid var(--line); }
    .fill { height:100%; border-radius:999px; background:linear-gradient(90deg, var(--accent), var(--ok)); min-width:2px; }
    .failure-tile { display:grid; grid-template-columns:1fr auto; gap:10px; align-items:center; padding:12px 0; border-bottom:1px solid var(--line); }
    .failure-tile:last-child { border-bottom:0; }
    .heat { width:120px; height:18px; border-radius:999px; background:linear-gradient(90deg, var(--ok), #facc15, var(--bad)); position:relative; overflow:hidden; border:1px solid var(--line); }
    .heat::after { content:""; position:absolute; inset:0; background:rgba(255,255,255,.7); transform:translateX(var(--mask)); }
    table { border-collapse: collapse; width:100%; margin-top:16px; font-size:14px; }
    th, td { border-bottom:1px solid var(--line); padding:10px 8px; text-align:left; }
    th { color:var(--muted); font-weight:600; }
    .pass { color:var(--ok); font-weight:700; }
    .fail { color:var(--bad); font-weight:700; }
    @media (max-width: 760px) { header { display:block; } .metrics, .charts { grid-template-columns:1fr; } }
  </style>
</head>
<body>
<main>
  <header>
    <div>
      <h1>Legal Agent Eval Dashboard</h1>
      <p>Run {{ summary.run_id }} · deterministic citation, redline, jurisdiction, and plan-routing evaluation.</p>
    </div>
    <p class="{{ 'pass' if verification.passed else 'fail' }}">{{ 'Verification passed' if verification.passed else 'Verification failed' }}</p>
  </header>
  <section class="grid metrics">
    <div class="panel metric"><strong>{{ summary.result_count }}</strong><span>case-model results</span></div>
    <div class="panel metric"><strong>{{ summary.unique_cases }}</strong><span>unique legal tasks</span></div>
    <div class="panel metric"><strong>{{ '%.0f'|format(summary.regression_precision * 100) }}%</strong><span>regression precision</span></div>
    <div class="panel metric"><strong>{{ '%.2f'|format(summary.runtime_seconds) }}s</strong><span>local runtime</span></div>
  </section>
  <section class="grid charts">
    <div class="panel">
      <h2 style="margin:0 0 8px; font-size:18px">Quality Index</h2>
      {% for row in summary.leaderboard %}
        <div class="bar-row">
          <strong>{{ row.model }}</strong>
          <div class="track" aria-label="{{ row.model }} quality {{ row.quality_index }}"><div class="fill" style="width: {{ row.quality_index * 100 }}%"></div></div>
          <span>{{ '%.2f'|format(row.quality_index) }}</span>
        </div>
      {% endfor %}
    </div>
    <div class="panel">
      <h2 style="margin:0 0 8px; font-size:18px">Regression Failures</h2>
      {% for row in summary.leaderboard %}
        <div class="failure-tile">
          <span>{{ row.model }}</span>
          <span><strong>{{ row.failures }}</strong> / {{ row.cases }}</span>
          <div class="heat" style="--mask: {{ (row.failures / row.cases) * 100 }}%"></div>
        </div>
      {% endfor %}
    </div>
  </section>
  <section class="panel" style="margin-top:16px">
    <h2 style="margin:0 0 8px; font-size:18px">Leaderboard</h2>
    <table>
      <thead><tr><th>Model</th><th>Quality</th><th>Citation F1</th><th>Redline</th><th>Jurisdiction</th><th>Plan Recall</th><th>Failures</th></tr></thead>
      <tbody>
      {% for row in summary.leaderboard %}
        <tr>
          <td>{{ row.model }}</td><td>{{ row.quality_index }}</td><td>{{ row.citation_f1 }}</td><td>{{ row.redline_delta }}</td><td>{{ row.jurisdiction_match }}</td><td>{{ row.plan_recall_at_k }}</td><td>{{ row.failures }}</td>
        </tr>
      {% endfor %}
      </tbody>
    </table>
  </section>
  <section class="panel" style="margin-top:16px">
    <h2 style="margin:0 0 8px; font-size:18px">Verification Gates</h2>
    <table>
      <tbody>
      {% for key, value in verification.checks.items() %}
        <tr><td>{{ key }}</td><td class="{{ 'pass' if value else 'fail' }}">{{ value }}</td></tr>
      {% endfor %}
      </tbody>
    </table>
  </section>
</main>
</body>
</html>
"""


def build_dashboard() -> Path:
    root = project_root()
    summary_path = root / "outputs" / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError("Run `uv run legal-eval run` before generating dashboard.")
    summary = RunSummary.model_validate_json(summary_path.read_text(encoding="utf-8"))
    from legal_eval.runner import verify_outputs

    verification = verify_outputs()
    env = Environment(autoescape=select_autoescape(["html", "xml"]))
    template = env.from_string(TEMPLATE)
    html = template.render(
        summary=summary,
        verification=verification,
    )
    target = root / "outputs" / "dashboard.html"
    target.write_text(html, encoding="utf-8")
    return target


def benchmark_summary() -> dict[str, float]:
    root = project_root()
    db_path = root / "runs" / "latest" / "results.duckdb"
    if not db_path.exists():
        raise FileNotFoundError("Run `uv run legal-eval run` first.")
    conn = duckdb.connect(str(db_path), read_only=True)
    row = conn.execute(
        """
        select
          avg(latency_ms) as avg_latency_ms,
          avg(token_estimate) as avg_token_estimate,
          count(*) as rows
        from case_results
        """
    ).fetchone()
    conn.close()
    return {"avg_latency_ms": float(row[0]), "avg_token_estimate": float(row[1]), "rows": float(row[2])}
