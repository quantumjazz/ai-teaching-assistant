import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass(frozen=True)
class RagEvalCase:
    case_id: str
    question: str
    expected_sources: List[str]
    expected_terms: List[str]
    expected_no_answer: bool = False
    max_duplicate_sources: Optional[int] = None


@dataclass(frozen=True)
class RagEvalResult:
    case_id: str
    passed: bool
    matched_source: bool
    matched_terms: bool
    matched_no_answer: bool
    duplicate_source_count: int
    duplicate_source_passed: bool
    returned_sources: List[str]
    missing_terms: List[str]

    def to_dict(self):
        return {
            "case_id": self.case_id,
            "passed": self.passed,
            "matched_source": self.matched_source,
            "matched_terms": self.matched_terms,
            "matched_no_answer": self.matched_no_answer,
            "duplicate_source_count": self.duplicate_source_count,
            "duplicate_source_passed": self.duplicate_source_passed,
            "returned_sources": self.returned_sources,
            "missing_terms": self.missing_terms,
        }


def load_eval_cases(path):
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        raw_cases = json.load(f)

    if not isinstance(raw_cases, list):
        raise ValueError("RAG eval cases file must contain a JSON list")

    cases = []
    for index, raw_case in enumerate(raw_cases):
        if not isinstance(raw_case, dict):
            raise ValueError(f"case {index} must be an object")
        case_id = str(raw_case.get("id") or f"case_{index + 1}")
        question = str(raw_case.get("question") or "").strip()
        if not question:
            raise ValueError(f"{case_id} is missing question")
        cases.append(
            RagEvalCase(
                case_id=case_id,
                question=question,
                expected_sources=[
                    str(source).lower() for source in raw_case.get("expected_sources", [])
                ],
                expected_terms=[
                    str(term).lower() for term in raw_case.get("expected_terms", [])
                ],
                expected_no_answer=bool(raw_case.get("expected_no_answer", False)),
                max_duplicate_sources=read_optional_nonnegative_int(
                    raw_case.get("max_duplicate_sources")
                ),
            )
        )
    return cases


def score_retrieval(case, retrieved_context):
    returned_sources = [source.filename for source in retrieved_context.sources]
    returned_sources_lower = [source.lower() for source in returned_sources]
    context_text = retrieved_context.text.lower()

    matched_source = True
    if case.expected_sources:
        matched_source = any(
            expected in returned_source
            for expected in case.expected_sources
            for returned_source in returned_sources_lower
        )

    missing_terms = [
        term for term in case.expected_terms if term not in context_text
    ]
    matched_terms = len(missing_terms) == 0
    matched_no_answer = True
    if case.expected_no_answer:
        matched_no_answer = not retrieved_context.answerable

    duplicate_source_count = count_duplicate_sources(retrieved_context.sources)
    duplicate_source_passed = True
    if case.max_duplicate_sources is not None:
        duplicate_source_passed = duplicate_source_count <= case.max_duplicate_sources

    return RagEvalResult(
        case_id=case.case_id,
        passed=(
            matched_source
            and matched_terms
            and matched_no_answer
            and duplicate_source_passed
        ),
        matched_source=matched_source,
        matched_terms=matched_terms,
        matched_no_answer=matched_no_answer,
        duplicate_source_count=duplicate_source_count,
        duplicate_source_passed=duplicate_source_passed,
        returned_sources=returned_sources,
        missing_terms=missing_terms,
    )


def read_optional_nonnegative_int(value):
    if value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("max_duplicate_sources must be a nonnegative integer") from exc
    if parsed < 0:
        raise ValueError("max_duplicate_sources must be a nonnegative integer")
    return parsed


def count_duplicate_sources(sources):
    counts = {}
    for source in sources:
        key = source.source_path or source.filename
        counts[key] = counts.get(key, 0) + 1
    return sum(max(0, count - 1) for count in counts.values())


def summarize_results(results):
    total = len(results)
    passed = sum(1 for result in results if result.passed)
    return {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": passed / total if total else 0.0,
        "results": [result.to_dict() for result in results],
    }
