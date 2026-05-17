# Legal Agent Evaluation Harness

A fully local evaluation harness for legal AI agents that need defensible regression testing across model upgrades. The project grades four failure modes that matter in professional legal work:

- citation faithfulness against exact source spans
- redline correctness with section-aware costs
- jurisdiction grounding
- plan and tool-routing recall

The load-bearing citation metric is deterministic and does not use an LLM judge. The demo target is synthetic, local, and designed to make regressions obvious without sending data to external services.

## Quick Start

```bash
uv sync
uv run legora-eval init-demo
uv run legora-eval run --iterations 4
uv run legora-eval verify
uv run legora-eval dashboard
```

Open `outputs/dashboard.html` after running the dashboard command.

## What It Produces

- `runs/latest/results.duckdb` with run, case, model, and metric rows
- `outputs/summary.json` with aggregate leaderboard metrics
- `outputs/regression_report.json` with precision and detected model-swap failures
- `outputs/dashboard.html` with self-contained visual drilldowns
- `outputs/demo_pack/` with a portable evidence bundle

## Design Notes

The harness intentionally separates plan quality from answer quality. A legal agent can sometimes produce the right answer through the wrong workflow, which hides future reliability and cost regressions. This project grades the trace separately so routing regressions surface immediately.

All fixtures are public-domain-shaped synthetic documents. No client data, private legal data, credentials, or external API calls are required.
