import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src import clients, retrieval, settings


class RuntimeCacheTests(unittest.TestCase):
    def setUp(self):
        clients.clear_client_cache()
        retrieval.clear_retrieval_cache()
        settings.clear_settings_cache()

    def tearDown(self):
        clients.clear_client_cache()
        retrieval.clear_retrieval_cache()
        settings.clear_settings_cache()

    def test_settings_cache_reuses_and_invalidates_by_mtime(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            settings_path = Path(tmpdir) / "settings.txt"
            settings_path.write_text(
                "classname=First\nembedding_method=openai\n",
                encoding="utf-8",
            )
            settings.clear_settings_cache()

            first = settings.load_course_settings(settings_path)
            second = settings.load_course_settings(settings_path)
            self.assertIs(first, second)

            settings_path.write_text(
                "classname=Second\nembedding_method=openai\n",
                encoding="utf-8",
            )
            current_mtime = settings_path.stat().st_mtime_ns
            os.utime(settings_path, ns=(current_mtime + 1_000_000, current_mtime + 1_000_000))

            third = settings.load_course_settings(settings_path)

        self.assertEqual(third.classname, "Second")
        self.assertIsNot(first, third)

    def test_faiss_resources_cache_reuses_index_until_artifacts_change(self):
        fake_faiss = SimpleNamespace(read_count=0)

        def read_index(path):
            fake_faiss.read_count += 1
            return {"path": path, "read_count": fake_faiss.read_count}

        fake_faiss.read_index = read_index

        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            (data_dir / "faiss_index.bin").write_bytes(b"index")
            (data_dir / "faiss_metadata.json").write_text(
                json.dumps([{"chunk_text": "Context"}]),
                encoding="utf-8",
            )
            retrieval.clear_retrieval_cache()

            with patch("src.retrieval.load_faiss_module", return_value=fake_faiss):
                first_index, _first_metadata = retrieval.load_faiss_resources(data_dir)
                second_index, _second_metadata = retrieval.load_faiss_resources(data_dir)

                metadata_path = data_dir / "faiss_metadata.json"
                metadata_path.write_text(
                    json.dumps([{"chunk_text": "Updated context"}]),
                    encoding="utf-8",
                )
                current_mtime = metadata_path.stat().st_mtime_ns
                os.utime(metadata_path, ns=(current_mtime + 1_000_000, current_mtime + 1_000_000))
                third_index, third_metadata = retrieval.load_faiss_resources(data_dir)

        self.assertIs(first_index, second_index)
        self.assertEqual(fake_faiss.read_count, 2)
        self.assertEqual(third_index["read_count"], 2)
        self.assertEqual(third_metadata[0]["chunk_text"], "Updated context")

    def test_configure_openai_caches_default_module(self):
        fake_openai = SimpleNamespace(api_key=None)
        clients.clear_client_cache()

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}), patch(
            "src.clients.load_dotenv_if_available", return_value=True
        ), patch("src.clients.load_openai_module", return_value=fake_openai) as load_module:
            first = clients.configure_openai()
            second = clients.configure_openai()

        self.assertIs(first, second)
        self.assertEqual(fake_openai.api_key, "test-key")
        self.assertEqual(load_module.call_count, 1)


if __name__ == "__main__":
    unittest.main()
