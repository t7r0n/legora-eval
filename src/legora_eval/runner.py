from __future__ import annotations

import json
import shutil
import time
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any

import duckdb

from legora_eval.corpus import SuiteFile, load_suite
from legora_eval.models import AgentTrace, CaseResult, Citation, MetricBundle, RedlineOp, RunSummary, ToolCall, project_root
from legora_eval.scorers import harmonic_quality, regression_precision, score_case

MODELS = ("frontier-legal-a", "frontier-legal-b", "model-swap-regression")


def init_demo(force: bool = False) -> dict[str, str]:
    root = project_root()
    for directory in ("data", "runs", "outputs"):
        path = root / directory
        if force and path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)
    return {
        "suite": str(root / "suites" / "nightly.json"),
        "runs": str(root / "runs"),
        "outputs": str(root / "outputs"),
    }


def connect_store(path: Path) -> duckdb.DuckDBPyConnection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(path))
    conn.execute(
        """
        create table if not exists case_results (
          run_id varchar,
          case_id varchar,
          suite varchar,
          model varchar,
          citation_f1 double,
          citation_precision double,
          citation_recall double,
          hallucinated_citations integer,
          redline_delta double,
          jurisdiction_match double,
          plan_recall_at_k double,
          tool_sequence_exact double,
          failed boolean,
          expected_regression boolean,
          latency_ms integer,
          token_estimate integer
        )
        """
    )
    return conn


def clone_citations(citations: list[Citation]) -> list[Citation]:
    return [Citation.model_validate(citation.model_dump()) for citation in citations]


def clone_redlines(ops: list[RedlineOp]) -> list[RedlineOp]:
    return [RedlineOp.model_validate(op.model_dump()) for op in ops]


def clone_tools(calls: list[ToolCall]) -> list[ToolCall]:
    return [ToolCall.model_validate(call.model_dump()) for call in calls]


def synthesize_trace(case_id: str, suite: SuiteFile, model: str, iteration: int) -> AgentTrace:
    case = next(item for item in suite.cases if item.id == case_id)
    plan = [call.name for call in case.gold_tool_calls]
    tool_calls = clone_tools(case.gold_tool_calls)
    citations = clone_citations(case.gold_citations)
    redlines = clone_redlines(case.gold_redline_ops)
    jurisdiction = case.jurisdiction_tag
    answer = case.expected_answer
    latency_ms = 145 + iteration * 9 + len(case.prompt)
    token_estimate = 900 + len(case.prompt.split()) * 18 + len(plan) * 40

    if model == "frontier-legal-b":
        latency_ms += 35
        token_estimate += 110
        if redlines:
            redlines[0] = redlines[0].model_copy(update={"text": redlines[0].text.replace("material", "Material")})

    if model == "model-swap-regression":
        latency_ms -= 25
        token_estimate -= 160
        if citations:
            bad = citations[0]
            citations[0] = bad.model_copy(update={"section_id": "7.3", "quote": "commercially reasonable efforts"})
        if redlines:
            redlines[0] = redlines[0].model_copy(update={"text": "replace with commercially reasonable language"})
        if plan:
            plan = [name for name in plan if name not in {"tabular_review", "legal_research"}]
            plan.append("direct_answer")
        jurisdiction = "US-DE" if case.jurisdiction_tag != "US-DE" else "EU"
        answer = f"{answer} The analysis also cites an unsupported adjacent clause."

    return AgentTrace(
        case_id=case.id,
        model=model,
        plan=plan,
        tool_calls=tool_calls,
        answer=answer,
        citations=citations,
        redline_ops=redlines,
        jurisdiction=jurisdiction,
        latency_ms=max(latency_ms, 1),
        token_estimate=max(token_estimate, 1),
    )


def persist_result(conn: duckdb.DuckDBPyConnection, result: CaseResult) -> None:
    metrics = result.metrics
    conn.execute(
        """
        insert into case_results values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            result.run_id,
            result.case.id,
            result.case.suite.value,
            result.trace.model,
            metrics.citation_f1,
            metrics.citation_precision,
            metrics.citation_recall,
            metrics.hallucinated_citations,
            metrics.redline_delta,
            metrics.jurisdiction_match,
            metrics.plan_recall_at_k,
            metrics.tool_sequence_exact,
            metrics.failed,
            result.trace.model in result.case.regression_expected_for,
            result.trace.latency_ms,
            result.trace.token_estimate,
        ],
    )


def summarize(run_id: str, results: list[CaseResult], runtime_seconds: float) -> RunSummary:
    by_model: dict[str, list[CaseResult]] = defaultdict(list)
    expected: list[bool] = []
    detected: list[bool] = []
    for result in results:
        by_model[result.trace.model].append(result)
        expected.append(result.trace.model in result.case.regression_expected_for)
        detected.append(result.metrics.failed)

    leaderboard: list[dict[str, Any]] = []
    for model, model_results in by_model.items():
        citation = sum(item.metrics.citation_f1 for item in model_results) / len(model_results)
        redline = sum(item.metrics.redline_delta for item in model_results) / len(model_results)
        jurisdiction = sum(item.metrics.jurisdiction_match for item in model_results) / len(model_results)
        plan = sum(item.metrics.plan_recall_at_k for item in model_results) / len(model_results)
        leaderboard.append(
            {
                "model": model,
                "citation_f1": round(citation, 4),
                "redline_delta": round(redline, 4),
                "jurisdiction_match": round(jurisdiction, 4),
                "plan_recall_at_k": round(plan, 4),
                "quality_index": round(harmonic_quality([citation, redline, jurisdiction, plan]), 4),
                "failures": sum(1 for item in model_results if item.metrics.failed),
                "cases": len(model_results),
            }
        )
    leaderboard.sort(key=lambda item: item["quality_index"], reverse=True)
    return RunSummary(
        run_id=run_id,
        result_count=len(results),
        unique_cases=len({result.case.id for result in results}),
        models=sorted(by_model),
        leaderboard=leaderboard,
        regression_precision=round(regression_precision(expected, detected), 4),
        detected_regressions=sum(1 for item in detected if item),
        expected_regressions=sum(1 for item in expected if item),
        runtime_seconds=round(runtime_seconds, 4),
    )


def run_suite(iterations: int = 4, models: tuple[str, ...] = MODELS, run_dir: Path | None = None) -> RunSummary:
    init_demo()
    suite = load_suite()
    run_id = f"run-{uuid.uuid4().hex[:12]}"
    root = project_root()
    target_dir = run_dir or root / "runs" / "latest"
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    conn = connect_store(target_dir / "results.duckdb")
    started = time.perf_counter()
    results: list[CaseResult] = []
    selected_cases = suite.cases * iterations
    for iteration, case in enumerate(selected_cases):
        for model in models:
            trace = synthesize_trace(case.id, suite, model, iteration)
            metrics = MetricBundle.model_validate(score_case(case, trace, suite))
            result = CaseResult(run_id=run_id, case=case, trace=trace, metrics=metrics)
            persist_result(conn, result)
            results.append(result)
    conn.close()
    summary = summarize(run_id, results, time.perf_counter() - started)
    outputs = root / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    (outputs / "summary.json").write_text(summary.model_dump_json(indent=2), encoding="utf-8")
    regression_report = {
        "precision": summary.regression_precision,
        "detected_regressions": summary.detected_regressions,
        "expected_regressions": summary.expected_regressions,
        "threshold": 0.95,
        "pass": summary.regression_precision >= 0.95,
    }
    (outputs / "regression_report.json").write_text(json.dumps(regression_report, indent=2), encoding="utf-8")
    return summary


def verify_outputs() -> dict[str, Any]:
    root = project_root()
    summary_path = root / "outputs" / "summary.json"
    report_path = root / "outputs" / "regression_report.json"
    db_path = root / "runs" / "latest" / "results.duckdb"
    if not summary_path.exists() or not report_path.exists() or not db_path.exists():
        raise FileNotFoundError("Run `uv run legora-eval run` before verification.")
    summary = RunSummary.model_validate_json(summary_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    conn = duckdb.connect(str(db_path), read_only=True)
    result_count = conn.execute("select count(*) from case_results").fetchone()[0]
    failing_non_regressed = conn.execute(
        "select count(*) from case_results where failed = true and expected_regression = false"
    ).fetchone()[0]
    conn.close()
    checks = {
        "required_outputs_present": summary_path.exists() and report_path.exists() and db_path.exists(),
        "at_least_sixty_results": result_count >= 60,
        "four_eval_axes_present": all(
            key in summary.leaderboard[0]
            for key in ("citation_f1", "redline_delta", "jurisdiction_match", "plan_recall_at_k")
        ),
        "regression_precision_at_least_0_95": report["precision"] >= 0.95,
        "full_suite_under_12_minutes": summary.runtime_seconds < 720,
        "no_false_positive_regressions": failing_non_regressed == 0,
    }
    return {
        "run_id": summary.run_id,
        "result_count": result_count,
        "models": summary.models,
        "leaderboard": summary.leaderboard,
        "checks": checks,
        "passed": all(checks.values()),
    }


def export_demo_pack() -> Path:
    root = project_root()
    pack = root / "outputs" / "demo_pack"
    if pack.exists():
        shutil.rmtree(pack)
    pack.mkdir(parents=True, exist_ok=True)
    for source in ("summary.json", "regression_report.json", "dashboard.html"):
        path = root / "outputs" / source
        if path.exists():
            shutil.copy2(path, pack / source)
    shutil.copy2(root / "suites" / "nightly.json", pack / "nightly.json")
    return pack
