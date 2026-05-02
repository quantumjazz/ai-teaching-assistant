import json
import tempfile
import unittest
from pathlib import Path

from src.evaluation import load_eval_cases, score_retrieval, summarize_results
from src.retrieval import RetrievedContext, Source


class EvaluationTests(unittest.TestCase):
    def test_load_eval_cases_validates_and_normalizes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cases.json"
            path.write_text(
                json.dumps(
                    [
                        {
                            "id": "CaseA",
                            "question": "Where is grading explained?",
                            "expected_sources": ["Syllabus"],
                            "expected_terms": ["Grade"],
                            "expected_no_answer": False,
                            "max_duplicate_sources": 0,
                        }
                    ]
                ),
                encoding="utf-8",
            )

            cases = load_eval_cases(path)

        self.assertEqual(cases[0].case_id, "CaseA")
        self.assertEqual(cases[0].expected_sources, ["syllabus"])
        self.assertEqual(cases[0].expected_terms, ["grade"])
        self.assertFalse(cases[0].expected_no_answer)
        self.assertEqual(cases[0].max_duplicate_sources, 0)

    def test_score_retrieval_passes_expected_source_and_terms(self):
        case = load_eval_case_for_test(
            expected_sources=["syllabus"],
            expected_terms=["grade", "policy"],
        )
        retrieved = RetrievedContext(
            text="Source: syllabus.pdf\nThe grade policy is explained here.",
            sources=[Source(filename="syllabus.pdf", chunk_index=0)],
        )

        result = score_retrieval(case, retrieved)

        self.assertTrue(result.passed)

    def test_score_retrieval_reports_missing_terms(self):
        case = load_eval_case_for_test(
            expected_sources=["syllabus"],
            expected_terms=["grade", "office"],
        )
        retrieved = RetrievedContext(
            text="Source: syllabus.pdf\nThe grade policy is explained here.",
            sources=[Source(filename="syllabus.pdf", chunk_index=0)],
        )

        result = score_retrieval(case, retrieved)

        self.assertFalse(result.passed)
        self.assertEqual(result.missing_terms, ["office"])

    def test_score_retrieval_passes_expected_no_answer(self):
        case = load_eval_case_for_test(expected_no_answer=True)
        retrieved = RetrievedContext(
            text="",
            sources=[],
            confidence=0.0,
            answerable=False,
        )

        result = score_retrieval(case, retrieved)

        self.assertTrue(result.passed)
        self.assertTrue(result.matched_no_answer)

    def test_score_retrieval_reports_duplicate_sources(self):
        case = load_eval_case_for_test(max_duplicate_sources=0)
        retrieved = RetrievedContext(
            text="Context",
            sources=[
                Source(filename="course.txt", chunk_index=0, source_path="documents/course.txt"),
                Source(filename="course.txt", chunk_index=1, source_path="documents/course.txt"),
            ],
        )

        result = score_retrieval(case, retrieved)

        self.assertFalse(result.passed)
        self.assertEqual(result.duplicate_source_count, 1)
        self.assertFalse(result.duplicate_source_passed)

    def test_summarize_results_counts_passes(self):
        passing = score_retrieval(
            load_eval_case_for_test(expected_terms=["grade"]),
            RetrievedContext("grade", [Source("source.txt", 0)]),
        )
        failing = score_retrieval(
            load_eval_case_for_test(expected_terms=["missing"]),
            RetrievedContext("grade", [Source("source.txt", 0)]),
        )

        summary = summarize_results([passing, failing])

        self.assertEqual(summary["total"], 2)
        self.assertEqual(summary["passed"], 1)
        self.assertEqual(summary["failed"], 1)


def load_eval_case_for_test(
    expected_sources=None,
    expected_terms=None,
    expected_no_answer=False,
    max_duplicate_sources=None,
):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "cases.json"
        path.write_text(
            json.dumps(
                [
                    {
                        "id": "case",
                        "question": "Question?",
                        "expected_sources": expected_sources or [],
                        "expected_terms": expected_terms or [],
                        "expected_no_answer": expected_no_answer,
                        "max_duplicate_sources": max_duplicate_sources,
                    }
                ]
            ),
            encoding="utf-8",
        )
        return load_eval_cases(path)[0]


if __name__ == "__main__":
    unittest.main()
