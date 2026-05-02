import json
import tempfile
import unittest
from pathlib import Path

from src.index_status import (
    document_records_for_report,
    get_index_status,
    list_document_inventory,
)


class IndexStatusTests(unittest.TestCase):
    def test_index_status_ready_when_report_matches_documents(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_indexed_project(root)

            status = get_index_status(root)

        self.assertEqual(status.status, "ready")
        self.assertEqual(status.message, "The local index matches the current supported documents.")

    def test_index_status_detects_changed_document(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_indexed_project(root)
            (root / "documents" / "course.txt").write_text("Updated content", encoding="utf-8")

            status = get_index_status(root)

        self.assertEqual(status.status, "stale")
        self.assertTrue(any("Changed supported document" in reason for reason in status.stale_reasons))

    def test_index_status_detects_added_supported_document(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_indexed_project(root)
            (root / "documents" / "extra.txt").write_text("Extra content", encoding="utf-8")

            status = get_index_status(root)

        self.assertEqual(status.status, "stale")
        self.assertTrue(any("New supported document" in reason for reason in status.stale_reasons))

    def test_index_status_lists_unsupported_documents_without_marking_stale(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_indexed_project(root)
            (root / "documents" / "notes.md").write_text("Unsupported", encoding="utf-8")

            status = get_index_status(root)

        self.assertEqual(status.status, "ready")
        payload = status.to_dict()
        self.assertEqual(payload["supported_document_count"], 1)
        self.assertEqual(payload["unsupported_document_count"], 1)

    def test_document_inventory_records_fingerprint_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            documents_dir = root / "documents"
            documents_dir.mkdir()
            (documents_dir / "course.txt").write_text("Course content", encoding="utf-8")

            inventory = list_document_inventory(root)

        self.assertEqual(len(inventory), 1)
        self.assertEqual(inventory[0].path, "documents/course.txt")
        self.assertEqual(inventory[0].extension, ".txt")
        self.assertTrue(inventory[0].supported)
        self.assertEqual(len(inventory[0].sha256), 64)

    def _write_indexed_project(self, root):
        data_dir = root / "data"
        documents_dir = root / "documents"
        data_dir.mkdir()
        documents_dir.mkdir()
        (documents_dir / "course.txt").write_text("Course content", encoding="utf-8")
        (data_dir / "chopped_text.csv").write_text("chunks", encoding="utf-8")
        (data_dir / "embedded_data.pkl").write_bytes(b"embedded")
        (data_dir / "faiss_index.bin").write_bytes(b"index")
        (data_dir / "faiss_metadata.json").write_text("[]", encoding="utf-8")
        input_documents = document_records_for_report(root)
        (data_dir / "index_report.json").write_text(
            json.dumps(
                {
                    "created_at": "2026-04-29T00:00:00+00:00",
                    "source_document_count": len(input_documents),
                    "chunk_count": 1,
                    "embedding_count": 1,
                    "faiss_vector_count": 1,
                    "embedding_model": "test-embedding",
                    "input_documents": input_documents,
                    "artifacts": {},
                }
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    unittest.main()
