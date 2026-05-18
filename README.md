# Legal Agent Evaluation Harness

A fully local evaluation harness for legal AI agents that need defensible regression testing across model upgrades. The project grades four failure modes that matter in professional legal work:

- citation faithfulness against exact source spans - redline correctness with section-aware costs - jurisdiction grounding - plan and tool-routing recall

## Problem shape

Local legal-agent evaluation harness for citation grounding, redlines, jurisdiction, and tool routing.

## What the harness exercises

- Replays the main `legora-eval` scenario from source-controlled fixtures.
- Pushes degraded `Legal Agent Evaluation Harness` cases through the same path as clean cases, then compares the evidence.
- Frames `Legal Agent Evaluation Harness` as a working evaluator rather than a static concept mock.
- Leaves `legora-eval` generated state outside git while keeping the rebuild path short.

## Local workflow

```bash
uv sync
uv run legora-eval init-demo
uv run legora-eval run --iterations 4
uv run legora-eval verify
uv run legora-eval dashboard
```

## Review surfaces

- `outputs/summary.json` for headline metrics and gate status
- `outputs/reports.json` for per-case results
- `outputs/dashboard.html` for visual inspection
- `outputs/demo-pack.zip` or `outputs/demo_pack/` for portable review

## Quality checks

```bash
uv run ruff check .
uv run pytest -q
uv run legora-eval verify
```

## Repository hygiene

Every example in `legora-eval` is fabricated for repeatability. Generated outputs are rebuildable artifacts, not source material.
