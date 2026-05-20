from __future__ import annotations

from pathlib import Path

from legal_eval.corpus import load_suite
from legal_eval.dashboard import build_dashboard
from legal_eval.models import Citation, RedlineOp
from legal_eval.runner import init_demo, run_suite, synthesize_trace, verify_outputs
from legal_eval.scorers import score_citations, score_redlines


def test_end_to_end_run_and_verify() -> None:
    init_demo(force=True)
    summary = run_suite(iterations=4)
    report = verify_outputs()
    assert summary.result_count >= 60
    assert summary.regression_precision >= 0.95
    assert report["passed"] is True


def test_citation_scorer_rejects_adjacent_wrong_clause() -> None:
    suite = load_suite()
    case = next(item for item in suite.cases if item.id == "citation-uk-warranty")
    bad = [
        Citation(
            document_id="spa-uk-001",
            section_id="7.3",
            quote="no material customer contract has been terminated",
            jurisdiction="UK",
        )
    ]
    metrics = score_citations(case.gold_citations, bad, suite)
    assert metrics["citation_f1"] == 0.0
    assert metrics["hallucinated_citations"] == 1.0


def test_redline_defined_term_penalty_is_strict() -> None:
    suite = load_suite()
    case = next(item for item in suite.cases if item.id == "redline-defined-term")
    weak = [RedlineOp(op="replace", section_id="12.2", text="Material Contract means a commercially reasonable contract.")]
    strong = case.gold_redline_ops
    assert score_redlines(strong, strong, case, suite) == 1.0
    assert score_redlines(strong, weak, case, suite) < 0.6


def test_model_swap_regression_changes_plan_and_jurisdiction() -> None:
    suite = load_suite()
    case = next(item for item in suite.cases if item.id == "tool-routing-diligence")
    trace = synthesize_trace(case.id, suite, "model-swap-regression", 0)
    assert "tabular_review" not in trace.plan
    assert trace.jurisdiction != case.jurisdiction_tag


def test_dashboard_is_generated_with_visual_containers() -> None:
    init_demo(force=True)
    run_suite(iterations=4)
    path = build_dashboard()
    html = Path(path).read_text(encoding="utf-8")
    assert "Quality Index" in html
    assert "Regression Failures" in html
    assert "Legal Agent Eval Dashboard" in html
    assert "Verification passed" in html
