import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts import create_final_data, embed_documents, prepare_documents
from src.retrieval import RetrievalCandidate, Source, _format_context


class IngestionTests(unittest.TestCase):
    def test_prepare_rejects_empty_extracted_text(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "empty.txt"
            path.write_text("   \n", encoding="utf-8")

            with redirect_stdout(io.StringIO()):
                with self.assertRaises(prepare_documents.IngestionError) as caught:
                    prepare_documents.prepare_chunks([str(path)])

        self.assertIn("run OCR first", str(caught.exception))

    def test_prepare_chunks_includes_source_fingerprints(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            documents_dir = root / "documents"
            documents_dir.mkdir()
            path = documents_dir / "course.txt"
            path.write_text("Course content for indexing", encoding="utf-8")

            with redirect_stdout(io.StringIO()):
                chunks = prepare_documents.prepare_chunks(
                    [str(path)],
                    chunk_size=20,
                    overlap=0,
                    project_root=str(root),
                )

        self.assertEqual(chunks[0].filename, "course.txt")
        self.assertEqual(chunks[0].source_path, "documents/course.txt")
        self.assertEqual(chunks[0].source_name, "course.txt")
        self.assertEqual(chunks[0].source_extension, ".txt")
        self.assertEqual(len(chunks[0].source_sha256), 64)
        self.assertEqual(chunks[0].document_title, "course")
        self.assertEqual(chunks[0].chunk_start_char, 0)
        self.assertGreater(chunks[0].chunk_word_count, 0)

    def test_prepare_chunks_keeps_section_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            documents_dir = root / "documents"
            documents_dir.mkdir()
            path = documents_dir / "course.txt"
            path.write_text(
                "Assessment\n\nThe grade policy uses assignments and an exam.",
                encoding="utf-8",
            )

            with redirect_stdout(io.StringIO()):
                chunks = prepare_documents.prepare_chunks(
                    [str(path)],
                    chunk_size=20,
                    overlap=0,
                    project_root=str(root),
        )

        self.assertEqual(chunks[0].section_title, "Assessment")
        self.assertNotIn("Section: Assessment.", chunks[0].chunk_text)
        self.assertIn("The grade policy uses assignments and an exam.", chunks[0].chunk_text)

    def test_txt_heading_heuristic_keeps_short_title_cased_body_text(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            documents_dir = root / "documents"
            documents_dir.mkdir()
            path = documents_dir / "course.txt"
            path.write_text(
                "Important Context Follows\n\nShort note continues here.",
                encoding="utf-8",
            )

            with redirect_stdout(io.StringIO()):
                chunks = prepare_documents.prepare_chunks(
                    [str(path)],
                    chunk_size=20,
                    overlap=0,
                    project_root=str(root),
                )

        self.assertEqual(chunks[0].section_title, "")
        self.assertIn("Important Context Follows", chunks[0].chunk_text)

    def test_legacy_chunk_format_helpers_are_removed(self):
        self.assertFalse(hasattr(prepare_documents, "chunk_text"))
        self.assertFalse(hasattr(prepare_documents, "format_chunk_text"))

    def test_format_context_adds_source_metadata_to_pristine_chunk_body(self):
        context = _format_context(
            [
                RetrievalCandidate(
                    "The grade policy uses assignments and an exam.",
                    Source(
                        filename="course.txt",
                        chunk_index=2,
                        page_number=3,
                        section_title="Assessment",
                        chunk_start_char=10,
                        chunk_end_char=56,
                    ),
                    vector_rank=0,
                )
            ]
        )

        self.assertIn("Source: course.txt, chunk 2, page 3, section Assessment", context)
        self.assertIn("chars 10-56", context)
        self.assertIn("The grade policy uses assignments and an exam.", context)

    def test_read_chopped_csv_rejects_malformed_rows(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "chopped_text.csv"
            path.write_text("filename,chunk_index\ncourse.txt,0\n", encoding="utf-8")

            with self.assertRaises(embed_documents.IngestionError):
                embed_documents.read_chopped_csv(str(path))

    def test_read_chopped_csv_rejects_malformed_source_fingerprint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "chopped_text.csv"
            path.write_text(
                "filename,chunk_index,chunk_text,source_path,source_name,source_extension,"
                "source_size_bytes,source_modified_at,source_modified_at_ns,source_sha256\n"
                "course.txt,0,Text,documents/course.txt,course.txt,.txt,not-a-size,"
                "2026-04-29T00:00:00+00:00,1,abc\n",
                encoding="utf-8",
            )

            with self.assertRaises(embed_documents.IngestionError):
                embed_documents.read_chopped_csv(str(path))

    def test_read_chopped_csv_rejects_missing_source_fingerprints(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "chopped_text.csv"
            path.write_text(
                "filename,chunk_index,chunk_text\n"
                "course.txt,0,Text\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                embed_documents.IngestionError,
                "source fingerprint",
            ):
                embed_documents.read_chopped_csv(str(path))

    def test_read_chopped_csv_preserves_chunk_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "chopped_text.csv"
            path.write_text(
                "filename,chunk_index,chunk_text,source_path,source_name,source_extension,"
                "source_size_bytes,source_modified_at,source_modified_at_ns,source_sha256,"
                "document_title,page_number,section_title,"
                "chunk_start_char,chunk_end_char,chunk_word_count\n"
                "course.txt,0,Text,documents/course.txt,course.txt,.txt,4,"
                "2026-04-29T00:00:00+00:00,1,abc,course,2,Assessment,10,25,3\n",
                encoding="utf-8",
            )

            records = embed_documents.read_chopped_csv(str(path))

        self.assertEqual(records[0]["document_title"], "course")
        self.assertEqual(records[0]["page_number"], 2)
        self.assertEqual(records[0]["section_title"], "Assessment")
        self.assertEqual(records[0]["chunk_start_char"], 10)

    def test_attach_embeddings_rejects_count_mismatch(self):
        data = [{"filename": "course.txt", "chunk_index": 0, "chunk_text": "Text"}]

        with self.assertRaises(embed_documents.IngestionError):
            embed_documents.attach_embeddings(data, [])

    def test_validate_embedded_data_rejects_invalid_dimensions(self):
        data = [
            {
                "filename": "a.txt",
                "chunk_index": 0,
                "chunk_text": "A",
                "embedding": [0.1, 0.2],
            },
            {
                "filename": "b.txt",
                "chunk_index": 0,
                "chunk_text": "B",
                "embedding": [0.1],
            },
        ]

        with self.assertRaises(create_final_data.IngestionError):
            create_final_data.validate_embedded_data(data)

    def test_create_index_report_success(self):
        embedded = [
            {
                "filename": "a.txt",
                "chunk_index": 0,
                "chunk_text": "A",
                "embedding": [0.1, 0.2],
            }
        ]
        metadata = [{"filename": "a.txt", "chunk_index": 0, "chunk_text": "A"}]
        faiss_index = SimpleNamespace(ntotal=1)

        report = create_final_data.create_index_report(
            embedded_data=embedded,
            metadata_list=metadata,
            faiss_index=faiss_index,
            embedding_model="test-embedding",
            artifact_paths={"faiss_index": "data/faiss_index.bin"},
            input_documents=[
                {
                    "path": "documents/a.txt",
                    "name": "a.txt",
                    "extension": ".txt",
                    "size_bytes": 1,
                    "modified_at": "2026-04-29T00:00:00+00:00",
                    "modified_at_ns": 1,
                    "sha256": "abc",
                }
            ],
        )

        self.assertEqual(report["source_document_count"], 1)
        self.assertEqual(report["chunk_count"], 1)
        self.assertEqual(report["embedding_count"], 1)
        self.assertEqual(report["faiss_vector_count"], 1)
        self.assertEqual(report["embedding_model"], "test-embedding")
        self.assertEqual(report["input_documents"][0]["path"], "documents/a.txt")

    def test_input_documents_rejects_missing_source_fingerprints(self):
        embedded = [
            {
                "filename": "a.txt",
                "chunk_index": 0,
                "chunk_text": "A",
                "embedding": [0.1, 0.2],
            }
        ]

        with self.assertRaisesRegex(
            create_final_data.IngestionError,
            "source fingerprint",
        ):
            create_final_data.input_documents_from_embedded_data(embedded)

    def test_embed_with_openai_default_does_not_sleep_between_batches(self):
        fake_openai = SimpleNamespace()
        fake_openai.embeddings = SimpleNamespace()

        def create(model, input):
            return SimpleNamespace(
                data=[SimpleNamespace(embedding=[float(index)]) for index, _ in enumerate(input)]
            )

        fake_openai.embeddings.create = create

        with patch("scripts.embed_documents.time.sleep") as sleep:
            embeddings = embed_documents.embed_with_openai(
                ["one", "two"],
                model="test-embedding",
                max_tokens_per_batch=1,
                openai_module=fake_openai,
            )

        self.assertEqual(len(embeddings), 2)
        sleep.assert_not_called()

    def test_embed_with_openai_sleeps_only_on_rate_limit(self):
        class RateLimitError(Exception):
            pass

        class FakeEmbeddings:
            def __init__(self):
                self.calls = 0

            def create(self, model, input):
                self.calls += 1
                if self.calls == 1:
                    raise RateLimitError("slow down")
                return SimpleNamespace(data=[SimpleNamespace(embedding=[1.0])])

        fake_openai = SimpleNamespace(
            embeddings=FakeEmbeddings(),
            RateLimitError=RateLimitError,
        )

        with patch("scripts.embed_documents.time.sleep") as sleep:
            embeddings = embed_documents.embed_with_openai(
                ["one"],
                model="test-embedding",
                max_tokens_per_batch=10,
                openai_module=fake_openai,
                sleep_seconds=1,
            )

        self.assertEqual(embeddings, [[1.0]])
        sleep.assert_called_once_with(1)

    def test_build_faiss_index_copies_chunk_metadata(self):
        embedded = [
            {
                "filename": "a.txt",
                "chunk_index": 0,
                "chunk_text": "A",
                "embedding": [0.1, 0.2],
                "document_title": "a",
                "page_number": None,
                "section_title": "Intro",
                "chunk_start_char": 0,
                "chunk_end_char": 1,
                "chunk_word_count": 1,
            }
        ]

        class FakeNumpy:
            float32 = "float32"

            def array(self, value, dtype=None):
                return value

            def vstack(self, values):
                return values

        class FakeIndex:
            ntotal = 0

            def __init__(self, dim):
                self.dim = dim

            def add(self, vectors):
                self.ntotal = len(vectors)

        class FakeFaiss:
            def IndexFlatL2(self, dim):
                return FakeIndex(dim)

        _index, metadata = create_final_data.build_faiss_index(
            embedded,
            embedding_dim=2,
            faiss_module=FakeFaiss(),
            np_module=FakeNumpy(),
        )

        self.assertEqual(metadata[0]["document_title"], "a")
        self.assertEqual(metadata[0]["section_title"], "Intro")
        self.assertEqual(metadata[0]["chunk_word_count"], 1)


if __name__ == "__main__":
    unittest.main()
