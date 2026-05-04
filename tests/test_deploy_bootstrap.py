import sys
import tempfile
import unittest
from pathlib import Path

from scripts.bootstrap_index import bootstrap_index
from src.index_status import document_records_for_report


class DeployBootstrapTests(unittest.TestCase):
    def test_ready_index_skips_bootstrap_commands(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_ready_project(root)
            calls = []

            result = bootstrap_index(root, env=ready_env(), runner=calls.append)

        self.assertEqual(result, "ready")
        self.assertEqual(calls, [])

    def test_missing_index_with_key_runs_pipeline(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_project_with_document(root)
            calls = []

            def runner(args, cwd, check):
                calls.append((args, cwd, check))

            result = bootstrap_index(root, env=ready_env(), runner=runner)

        self.assertEqual(result, "built")
        self.assertEqual(len(calls), 3)
        self.assertEqual([call[0][0] for call in calls], [sys.executable] * 3)
        self.assertTrue(calls[0][0][1].endswith("prepare_documents.py"))
        self.assertTrue(calls[1][0][1].endswith("embed_documents.py"))
        self.assertTrue(calls[2][0][1].endswith("create_final_data.py"))
        self.assertTrue(all(call[2] for call in calls))

    def test_course_root_env_selects_external_course_data(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_project_with_document(root)
            calls = []

            def runner(args, cwd, check):
                calls.append((args, cwd, check))

            result = bootstrap_index(
                env={**ready_env(), "COURSE_ROOT": str(root)},
                runner=runner,
            )

        self.assertEqual(result, "built")
        self.assertEqual(len(calls), 3)

    def test_missing_key_skips_bootstrap_commands(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_project_with_document(root)
            calls = []

            result = bootstrap_index(root, env={}, runner=calls.append)

        self.assertEqual(result, "skipped")
        self.assertEqual(calls, [])

    def test_auto_index_flag_can_disable_bootstrap(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_project_with_document(root)
            calls = []

            result = bootstrap_index(
                root,
                env={"OPENAI_API_KEY": "test-key", "AUTO_INDEX_ON_STARTUP": "false"},
                runner=calls.append,
            )

        self.assertEqual(result, "disabled")
        self.assertEqual(calls, [])

    def _write_project_with_document(self, root):
        (root / "documents").mkdir()
        (root / "data").mkdir()
        (root / "documents" / "course.txt").write_text(
            "Course content for deployment bootstrap.",
            encoding="utf-8",
        )

    def _write_ready_project(self, root):
        self._write_project_with_document(root)
        input_documents = document_records_for_report(root)
        data_dir = root / "data"
        (data_dir / "chopped_text.csv").write_text("chunks", encoding="utf-8")
        (data_dir / "embedded_data.pkl").write_bytes(b"embedded")
        (data_dir / "faiss_index.bin").write_bytes(b"index")
        (data_dir / "faiss_metadata.json").write_text("[]", encoding="utf-8")
        (data_dir / "index_report.json").write_text(
            "{"
            '"created_at":"2026-04-29T00:00:00+00:00",'
            f'"source_document_count":{len(input_documents)},'
            '"chunk_count":1,'
            '"embedding_count":1,'
            '"faiss_vector_count":1,'
            '"embedding_model":"test-embedding",'
            f'"input_documents":{_json_documents(input_documents)},'
            '"artifacts":{}'
            "}",
            encoding="utf-8",
        )


def ready_env():
    return {
        "OPENAI_API_KEY": "test-key",
        "FLASK_SECRET_KEY": "test-secret",
    }


def _json_documents(documents):
    import json

    return json.dumps(documents)


if __name__ == "__main__":
    unittest.main()
