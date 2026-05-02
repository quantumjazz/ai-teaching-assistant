import json
import pickle
import tempfile
import unittest
from pathlib import Path

from src.health import get_health_report
from src.index_status import document_records_for_report


class HealthTests(unittest.TestCase):
    def test_health_ready_with_report_present(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_ready_project(root, include_report=True)

            report = get_health_report(root, env=ready_env())

        self.assertEqual(report.status, "ready")
        checks = {check.name: check for check in report.checks}
        self.assertEqual(checks["index_report"].status, "ready")
        self.assertIn("valid", checks["index_report"].message)

    def test_health_missing_api_key_is_incomplete(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_ready_project(root)

            report = get_health_report(root, env={})

        checks = {check.name: check for check in report.checks}
        self.assertEqual(report.status, "incomplete")
        self.assertEqual(checks["openai_key"].status, "incomplete")

    def test_health_missing_flask_secret_is_incomplete(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_ready_project(root)

            report = get_health_report(root, env={"OPENAI_API_KEY": "test-key"})

        checks = {check.name: check for check in report.checks}
        self.assertEqual(report.status, "incomplete")
        self.assertEqual(checks["flask_secret"].status, "incomplete")

    def test_health_placeholder_env_values_are_incomplete(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_ready_project(root)
            (root / ".env").write_text(
                "OPENAI_API_KEY=your_api_key_here\n"
                "FLASK_SECRET_KEY=replace_with_a_long_random_secret\n",
                encoding="utf-8",
            )

            report = get_health_report(root, env={})

        checks = {check.name: check for check in report.checks}
        self.assertEqual(report.status, "incomplete")
        self.assertEqual(checks["openai_key"].status, "incomplete")
        self.assertEqual(checks["flask_secret"].status, "incomplete")

    def test_health_whitespace_env_values_are_incomplete(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_ready_project(root)

            report = get_health_report(
                root,
                env={"OPENAI_API_KEY": "   ", "FLASK_SECRET_KEY": "\t"},
            )

        checks = {check.name: check for check in report.checks}
        self.assertEqual(report.status, "incomplete")
        self.assertEqual(checks["openai_key"].status, "incomplete")
        self.assertEqual(checks["flask_secret"].status, "incomplete")

    def test_health_missing_documents_is_incomplete(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_ready_project(root)
            for path in (root / "documents").iterdir():
                path.unlink()

            report = get_health_report(root, env=ready_env())

        checks = {check.name: check for check in report.checks}
        self.assertEqual(report.status, "incomplete")
        self.assertEqual(checks["documents"].status, "incomplete")

    def test_health_missing_faiss_artifact_is_incomplete(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_ready_project(root)
            (root / "data" / "faiss_index.bin").unlink()

            report = get_health_report(root, env=ready_env())

        checks = {check.name: check for check in report.checks}
        self.assertEqual(report.status, "incomplete")
        self.assertEqual(checks["faiss_index"].status, "incomplete")

    def test_health_malformed_metadata_is_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_ready_project(root)
            (root / "data" / "faiss_metadata.json").write_text("{}", encoding="utf-8")

            report = get_health_report(root, env=ready_env())

        checks = {check.name: check for check in report.checks}
        self.assertEqual(report.status, "error")
        self.assertEqual(checks["faiss_metadata"].status, "error")

    def test_health_rejects_embedded_data_without_source_fingerprints(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_ready_project(root)
            with open(root / "data" / "embedded_data.pkl", "wb") as f:
                pickle.dump(
                    [
                        {
                            "filename": "course.txt",
                            "chunk_index": 0,
                            "chunk_text": "Course content",
                            "embedding": [0.1, 0.2],
                        }
                    ],
                    f,
                )

            report = get_health_report(root, env=ready_env())

        checks = {check.name: check for check in report.checks}
        self.assertEqual(report.status, "error")
        self.assertEqual(checks["embedded_data"].status, "error")
        self.assertIn("source_path", checks["embedded_data"].message)

    def test_health_rejects_embedded_data_with_empty_source_fingerprint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_ready_project(root)
            with open(root / "data" / "embedded_data.pkl", "rb") as f:
                embedded = pickle.load(f)
            embedded[0]["source_path"] = " "
            with open(root / "data" / "embedded_data.pkl", "wb") as f:
                pickle.dump(embedded, f)

            report = get_health_report(root, env=ready_env())

        checks = {check.name: check for check in report.checks}
        self.assertEqual(report.status, "error")
        self.assertEqual(checks["embedded_data"].status, "error")
        self.assertIn("empty source_path", checks["embedded_data"].message)

    def test_health_missing_index_report_is_incomplete(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_ready_project(root, include_report=False)

            report = get_health_report(root, env=ready_env())

        checks = {check.name: check for check in report.checks}
        self.assertEqual(report.status, "incomplete")
        self.assertEqual(checks["index_report"].status, "incomplete")
        self.assertEqual(checks["index_freshness"].status, "incomplete")

    def test_health_stale_index_is_incomplete(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_ready_project(root)
            (root / "documents" / "course.txt").write_text("Changed content", encoding="utf-8")

            report = get_health_report(root, env=ready_env())

        checks = {check.name: check for check in report.checks}
        self.assertEqual(report.status, "incomplete")
        self.assertEqual(checks["index_freshness"].status, "incomplete")
        self.assertIn("Changed supported document", checks["index_freshness"].message)

    def _write_ready_project(self, root, include_report=True):
        data_dir = root / "data"
        documents_dir = root / "documents"
        data_dir.mkdir()
        documents_dir.mkdir()
        (root / "settings.txt").write_text(
            "classname=Test Course\n"
            "embedding_method=openai\n"
            "openai_embedding_model=test-embedding\n",
            encoding="utf-8",
        )
        (documents_dir / "course.txt").write_text("Course content", encoding="utf-8")
        source_document = document_records_for_report(root)[0]
        (data_dir / "chopped_text.csv").write_text(
            "filename,chunk_index,chunk_text\ncourse.txt,0,Course content\n",
            encoding="utf-8",
        )
        embedded = [
            {
                "filename": "course.txt",
                "chunk_index": 0,
                "chunk_text": "Course content",
                "embedding": [0.1, 0.2],
                "source_path": source_document["path"],
                "source_name": source_document["name"],
                "source_extension": source_document["extension"],
                "source_size_bytes": source_document["size_bytes"],
                "source_modified_at": source_document["modified_at"],
                "source_modified_at_ns": source_document["modified_at_ns"],
                "source_sha256": source_document["sha256"],
            }
        ]
        with open(data_dir / "embedded_data.pkl", "wb") as f:
            pickle.dump(embedded, f)
        (data_dir / "faiss_index.bin").write_bytes(b"index")
        (data_dir / "faiss_metadata.json").write_text(
            json.dumps(
                [
                    {
                        "filename": "course.txt",
                        "chunk_index": 0,
                        "chunk_text": "Course content",
                    }
                ]
            ),
            encoding="utf-8",
        )
        if include_report:
            (data_dir / "index_report.json").write_text(
                json.dumps(
                    {
                        "created_at": "2026-04-29T00:00:00+00:00",
                        "source_document_count": 1,
                        "chunk_count": 1,
                        "embedding_count": 1,
                        "faiss_vector_count": 1,
                        "embedding_model": "test-embedding",
                        "input_documents": document_records_for_report(root),
                        "artifacts": {},
                    }
                ),
                encoding="utf-8",
            )


def ready_env():
    return {
        "OPENAI_API_KEY": "test-key",
        "FLASK_SECRET_KEY": "test-secret",
    }


if __name__ == "__main__":
    unittest.main()
