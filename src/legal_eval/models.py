from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class SuiteName(StrEnum):
    CITATION = "citation"
    REDLINE = "redline"
    JURISDICTION = "jurisdiction"
    TOOL_ROUTING = "tool-routing"


class Citation(BaseModel):
    document_id: str
    section_id: str
    quote: str
    jurisdiction: str


class RedlineOp(BaseModel):
    op: str
    section_id: str
    text: str
    weight: float = 1.0


class ToolCall(BaseModel):
    name: str
    args: dict[str, Any] = Field(default_factory=dict)


class LegalSection(BaseModel):
    id: str
    title: str
    text: str
    jurisdiction: str
    defined_terms: bool = False


class LegalDocument(BaseModel):
    id: str
    title: str
    jurisdiction: str
    sections: list[LegalSection]


class EvalCase(BaseModel):
    id: str
    suite: SuiteName
    title: str
    prompt: str
    document_ids: list[str]
    expected_answer: str
    gold_citations: list[Citation]
    gold_redline_ops: list[RedlineOp]
    gold_tool_calls: list[ToolCall]
    jurisdiction_tag: str
    regression_expected_for: list[str] = Field(default_factory=list)


class AgentTrace(BaseModel):
    case_id: str
    model: str
    plan: list[str]
    tool_calls: list[ToolCall]
    answer: str
    citations: list[Citation]
    redline_ops: list[RedlineOp]
    jurisdiction: str
    latency_ms: int
    token_estimate: int


class MetricBundle(BaseModel):
    citation_precision: float
    citation_recall: float
    citation_f1: float
    hallucinated_citations: int
    redline_delta: float
    jurisdiction_match: float
    plan_recall_at_k: float
    tool_sequence_exact: float
    failed: bool


class CaseResult(BaseModel):
    run_id: str
    case: EvalCase
    trace: AgentTrace
    metrics: MetricBundle


class RunSummary(BaseModel):
    run_id: str
    result_count: int
    unique_cases: int
    models: list[str]
    leaderboard: list[dict[str, Any]]
    regression_precision: float
    detected_regressions: int
    expected_regressions: int
    runtime_seconds: float


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]
