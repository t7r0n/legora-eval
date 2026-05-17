from __future__ import annotations

import math
import re
from collections.abc import Sequence

from legora_eval.corpus import SuiteFile, section_index, section_meta
from legora_eval.models import AgentTrace, Citation, EvalCase, RedlineOp, ToolCall


TOKEN_RE = re.compile(r"[a-z0-9]+")


def normalize(text: str) -> str:
    return " ".join(TOKEN_RE.findall(text.lower()))


def token_set(text: str) -> set[str]:
    return set(TOKEN_RE.findall(text.lower()))


def f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def citation_key(citation: Citation) -> tuple[str, str, str]:
    return (citation.document_id, citation.section_id, normalize(citation.quote))


def quote_grounded(citation: Citation, suite: SuiteFile) -> bool:
    sections = section_index(suite)
    source_text = sections.get((citation.document_id, citation.section_id), "")
    exact_match = normalize(citation.quote) in normalize(source_text)
    if exact_match:
        return True
    source_tokens = token_set(source_text)
    quote_tokens = token_set(citation.quote)
    if not quote_tokens:
        return False
    return len(source_tokens & quote_tokens) / len(quote_tokens) >= 0.92


def score_citations(gold: Sequence[Citation], claimed: Sequence[Citation], suite: SuiteFile) -> dict[str, float]:
    gold_keys = {citation_key(citation) for citation in gold if quote_grounded(citation, suite)}
    claimed_grounded = {
        citation_key(citation)
        for citation in claimed
        if quote_grounded(citation, suite) and citation.jurisdiction
    }
    correct = len(gold_keys & claimed_grounded)
    precision = correct / len(claimed) if claimed else 0.0
    recall = correct / len(gold_keys) if gold_keys else 1.0
    hallucinated = len(claimed) - correct
    return {
        "citation_precision": precision,
        "citation_recall": recall,
        "citation_f1": f1(precision, recall),
        "hallucinated_citations": float(max(hallucinated, 0)),
    }


def lcs_ratio(a: list[str], b: list[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    previous = [0] * (len(b) + 1)
    for left in a:
        current = [0]
        for j, right in enumerate(b, start=1):
            current.append(previous[j - 1] + 1 if left == right else max(previous[j], current[-1]))
        previous = current
    return previous[-1] / max(len(a), len(b))


def op_similarity(gold: RedlineOp, claimed: RedlineOp, defined_terms: bool) -> float:
    if gold.op != claimed.op or gold.section_id != claimed.section_id:
        return 0.0
    base = lcs_ratio(TOKEN_RE.findall(gold.text.lower()), TOKEN_RE.findall(claimed.text.lower()))
    if defined_terms and base < 0.85:
        return base * 0.55
    return base


def score_redlines(gold: Sequence[RedlineOp], claimed: Sequence[RedlineOp], case: EvalCase, suite: SuiteFile) -> float:
    if not gold and not claimed:
        return 1.0
    if not gold or not claimed:
        return 0.0
    meta = section_meta(suite)
    claimed_remaining = list(claimed)
    scores: list[float] = []
    for gold_op in gold:
        defined = any(meta.get((doc_id, gold_op.section_id), ("", False))[1] for doc_id in case.document_ids)
        best_index = -1
        best_score = 0.0
        for index, candidate in enumerate(claimed_remaining):
            score = op_similarity(gold_op, candidate, defined) * gold_op.weight
            if score > best_score:
                best_score = score
                best_index = index
        scores.append(min(best_score / max(gold_op.weight, 0.01), 1.0))
        if best_index >= 0:
            claimed_remaining.pop(best_index)
    extra_penalty = 0.04 * len(claimed_remaining)
    return max(sum(scores) / len(gold) - extra_penalty, 0.0)


def score_jurisdiction(gold: str, claimed: str) -> float:
    return 1.0 if gold.lower() == claimed.lower() else 0.0


def score_plan(gold_calls: Sequence[ToolCall], actual_plan: Sequence[str], k: int = 5) -> tuple[float, float]:
    gold_names = [call.name for call in gold_calls]
    actual_top_k = list(actual_plan[:k])
    if not gold_names:
        return 1.0, 1.0
    recalled = sum(1 for name in gold_names if name in actual_top_k)
    exact = 1.0 if actual_top_k[: len(gold_names)] == gold_names else 0.0
    return recalled / len(gold_names), exact


def score_case(case: EvalCase, trace: AgentTrace, suite: SuiteFile) -> dict[str, float | bool]:
    citation_metrics = score_citations(case.gold_citations, trace.citations, suite)
    redline_delta = score_redlines(case.gold_redline_ops, trace.redline_ops, case, suite)
    jurisdiction_match = score_jurisdiction(case.jurisdiction_tag, trace.jurisdiction)
    plan_recall, exact = score_plan(case.gold_tool_calls, trace.plan)
    failed = (
        citation_metrics["citation_f1"] < 0.92
        or redline_delta < 0.84
        or jurisdiction_match < 1.0
        or plan_recall < 0.8
    )
    return {
        **citation_metrics,
        "redline_delta": redline_delta,
        "jurisdiction_match": jurisdiction_match,
        "plan_recall_at_k": plan_recall,
        "tool_sequence_exact": exact,
        "failed": failed,
    }


def regression_precision(expected: Sequence[bool], detected: Sequence[bool]) -> float:
    true_positive = sum(1 for e, d in zip(expected, detected, strict=True) if e and d)
    false_positive = sum(1 for e, d in zip(expected, detected, strict=True) if not e and d)
    if true_positive + false_positive == 0:
        return 1.0
    return true_positive / (true_positive + false_positive)


def harmonic_quality(values: Sequence[float]) -> float:
    clean = [max(v, 0.001) for v in values]
    return len(clean) / sum(1 / value for value in clean) if clean else math.nan
