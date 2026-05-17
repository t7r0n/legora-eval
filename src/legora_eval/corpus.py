from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel

from legora_eval.models import EvalCase, LegalDocument, project_root


class SuiteFile(BaseModel):
    documents: list[LegalDocument]
    cases: list[EvalCase]


def default_suite_path() -> Path:
    return project_root() / "suites" / "nightly.json"


def load_suite(path: Path | None = None) -> SuiteFile:
    suite_path = path or default_suite_path()
    with suite_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return SuiteFile.model_validate(payload)


def document_index(suite: SuiteFile) -> dict[str, LegalDocument]:
    return {document.id: document for document in suite.documents}


def section_index(suite: SuiteFile) -> dict[tuple[str, str], str]:
    index: dict[tuple[str, str], str] = {}
    for document in suite.documents:
        for section in document.sections:
            index[(document.id, section.id)] = section.text
    return index


def section_meta(suite: SuiteFile) -> dict[tuple[str, str], tuple[str, bool]]:
    meta: dict[tuple[str, str], tuple[str, bool]] = {}
    for document in suite.documents:
        for section in document.sections:
            meta[(document.id, section.id)] = (section.jurisdiction, section.defined_terms)
    return meta
