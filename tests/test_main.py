import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src import main
from src.prompts import build_messages
from src.retrieval import RetrievedContext, Source
from src.settings import CourseSettings
from src.sessions import ConversationState
from src.service import answer_query_with_sources


class FakeNumpy:
    float32 = "float32"

    def array(self, value, dtype=None):
        return value

    def expand_dims(self, value, axis=0):
        return [value]


class FakeEmbeddings:
    def __init__(self):
        self.calls = []

    def create(self, model, input):
        self.calls.append({"model": model, "input": input})
        return SimpleNamespace(data=[SimpleNamespace(embedding=[0.1, 0.2, 0.3])])


class FakeCompletions:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        reply = self.replies.pop(0)
        message = SimpleNamespace(content=reply)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class FakeOpenAI:
    def __init__(self, replies):
        self.api_key = None
        self.embeddings = FakeEmbeddings()
        self.chat = SimpleNamespace(completions=FakeCompletions(replies))


class FakeIndex:
    def __init__(self):
        self.last_k = None

    def search(self, query_embedding, k):
        self.last_k = k
        return [[0.0, 1.0]], [[0, 1]]


class FakeFaiss:
    def __init__(self, index):
        self.index = index

    def read_index(self, path):
        return self.index


class MainServiceTests(unittest.TestCase):
    def test_read_settings_skips_comments_and_blank_lines(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            settings_path = Path(tmpdir) / "settings.txt"
            settings_path.write_text(
                "\n# embedding_method=sentence-transformers\n"
                "embedding_method=openai\n"
                "num_chunks=4\n"
                "hybrid_retrieval=false\n"
                "rrf_k=42\n"
                "max_chunks_per_source=1\n"
                "minimum_retrieval_confidence=0.25\n",
                encoding="utf-8",
            )

            settings = main.read_settings(settings_path)
            loaded = main.load_course_settings(settings_path)

        self.assertEqual(settings["embedding_method"], "openai")
        self.assertEqual(settings["num_chunks"], "4")
        self.assertNotIn("# embedding_method", settings)
        self.assertFalse(loaded.hybrid_retrieval)
        self.assertEqual(loaded.rrf_k, 42)
        self.assertEqual(loaded.max_chunks_per_source, 1)
        self.assertEqual(loaded.minimum_retrieval_confidence, 0.25)
        self.assertEqual(loaded.minimum_hybrid_retrieval_confidence, 0.25)
        self.assertEqual(loaded.minimum_lexical_retrieval_confidence, 0.25)
        self.assertEqual(loaded.minimum_vector_retrieval_confidence, 0.25)

    def test_load_faiss_resources_raises_clear_setup_error_when_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(main.KnowledgeBaseNotReady) as caught:
                main.load_faiss_resources(
                    data_dir=tmpdir, faiss_module=FakeFaiss(FakeIndex())
                )

        self.assertIn("Course knowledge base is not ready", str(caught.exception))
        self.assertEqual(caught.exception.code, "knowledge_base_not_ready")

    def test_load_course_settings_reads_mode_specific_confidence_thresholds(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            settings_path = Path(tmpdir) / "settings.txt"
            settings_path.write_text(
                "embedding_method=openai\n"
                "minimum_hybrid_retrieval_confidence=0.03\n"
                "minimum_lexical_retrieval_confidence=0.4\n"
                "minimum_vector_retrieval_confidence=0.5\n",
                encoding="utf-8",
            )

            loaded = main.load_course_settings(settings_path)

        self.assertEqual(loaded.minimum_hybrid_retrieval_confidence, 0.03)
        self.assertEqual(loaded.minimum_lexical_retrieval_confidence, 0.4)
        self.assertEqual(loaded.minimum_vector_retrieval_confidence, 0.5)

    def test_answer_check_mode_is_disabled_without_requiring_data(self):
        reply = main.answer_query("a: is this answer correct?")

        self.assertIn("needs a previous course question", reply)

    def test_answer_check_disabled_message_matches_bulgarian_input(self):
        result = answer_query_with_sources("a: това е отговор")

        self.assertIn("Режимът за проверка", result.response)

    def test_multiple_choice_requires_prompt_after_prefix(self):
        with self.assertRaises(main.UserInputError) as caught:
            answer_query_with_sources("m:")

        self.assertEqual(str(caught.exception), "No query provided")
        self.assertEqual(caught.exception.status_code, 400)

    def test_answer_query_uses_mocked_retrieval_and_openai(self):
        fake_index = FakeIndex()
        fake_openai = FakeOpenAI(["No", "Course answer."])

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            settings_path = root / "settings.txt"
            data_dir = root / "data"
            data_dir.mkdir()
            settings_path.write_text(
                "classname=Test Course\n"
                "assistantname=Test Assistant\n"
                "classdescription=testing\n"
                "embedding_method=openai\n"
                "openai_embedding_model=test-embedding\n"
                "chat_model=test-chat\n"
                "num_chunks=2\n"
                "minimum_hybrid_retrieval_confidence=0.01\n",
                encoding="utf-8",
            )
            (data_dir / "faiss_index.bin").write_text("fake", encoding="utf-8")
            (data_dir / "faiss_metadata.json").write_text(
                json.dumps(
                    [
                        {
                            "filename": "lecture-1.pdf",
                            "chunk_index": 0,
                            "chunk_text": "First context",
                        },
                        {
                            "filename": "lecture-2.pdf",
                            "chunk_index": 3,
                            "chunk_text": "Second context",
                        },
                    ]
                ),
                encoding="utf-8",
            )

            with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}), patch(
                "src.retrieval.load_numpy_module", return_value=FakeNumpy()
            ):
                result = answer_query_with_sources(
                    "What is covered?",
                    settings_path=settings_path,
                    data_dir=data_dir,
                    openai_module=fake_openai,
                    faiss_module=FakeFaiss(fake_index),
                    verify=False,
                )

        self.assertEqual(result.response, "Course answer.")
        self.assertEqual([source.filename for source in result.sources], ["lecture-1.pdf", "lecture-2.pdf"])
        self.assertEqual([source.chunk_index for source in result.sources], [0, 3])
        self.assertEqual([source.snippet for source in result.sources], ["First context", "Second context"])
        self.assertGreater(result.sources[0].hybrid_score, 0)
        self.assertEqual(result.sources[0].vector_rank, 0)
        self.assertIn("Source: lecture-1.pdf, chunk 0", fake_openai.chat.completions.calls[1]["messages"][0]["content"])
        self.assertEqual(fake_openai.api_key, "test-key")
        self.assertEqual(fake_openai.embeddings.calls[0]["model"], "test-embedding")
        self.assertEqual(fake_index.last_k, 2)

    def test_answer_query_keeps_string_compatibility(self):
        fake_index = FakeIndex()
        fake_openai = FakeOpenAI(["No", "Course answer."])

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            settings_path = root / "settings.txt"
            data_dir = root / "data"
            data_dir.mkdir()
            settings_path.write_text(
                "embedding_method=openai\n"
                "openai_embedding_model=test-embedding\n"
                "chat_model=test-chat\n"
                "minimum_hybrid_retrieval_confidence=0.01\n",
                encoding="utf-8",
            )
            (data_dir / "faiss_index.bin").write_text("fake", encoding="utf-8")
            (data_dir / "faiss_metadata.json").write_text(
                json.dumps([{"chunk_text": "Context"}]),
                encoding="utf-8",
            )

            with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}), patch(
                "src.retrieval.load_numpy_module", return_value=FakeNumpy()
            ):
                reply = main.answer_query(
                    "What is covered?",
                    settings_path=settings_path,
                    data_dir=data_dir,
                    openai_module=fake_openai,
                    faiss_module=FakeFaiss(fake_index),
                    verify=False,
                )

        self.assertEqual(reply, "Course answer.")

    def test_answer_query_returns_no_answer_when_retrieval_confidence_is_low(self):
        fake_index = FakeIndex()
        fake_openai = FakeOpenAI(["No"])

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            settings_path = root / "settings.txt"
            data_dir = root / "data"
            data_dir.mkdir()
            settings_path.write_text(
                "embedding_method=openai\n"
                "openai_embedding_model=test-embedding\n"
                "chat_model=test-chat\n"
                "minimum_retrieval_confidence=1.0\n",
                encoding="utf-8",
            )
            (data_dir / "faiss_index.bin").write_text("fake", encoding="utf-8")
            (data_dir / "faiss_metadata.json").write_text(
                json.dumps([{"filename": "course.txt", "chunk_index": 0, "chunk_text": "Context"}]),
                encoding="utf-8",
            )

            with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}), patch(
                "src.retrieval.load_numpy_module", return_value=FakeNumpy()
            ):
                result = answer_query_with_sources(
                    "Unrelated question?",
                    settings_path=settings_path,
                    data_dir=data_dir,
                    openai_module=fake_openai,
                    faiss_module=FakeFaiss(fake_index),
                    verify=False,
                )

        self.assertIn("I don't know", result.response)
        self.assertEqual(len(fake_openai.chat.completions.calls), 1)

    def test_answer_query_returns_bulgarian_no_answer_for_bulgarian_input(self):
        fake_index = FakeIndex()
        fake_openai = FakeOpenAI(["No"])

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            settings_path = root / "settings.txt"
            data_dir = root / "data"
            data_dir.mkdir()
            settings_path.write_text(
                "embedding_method=openai\n"
                "openai_embedding_model=test-embedding\n"
                "chat_model=test-chat\n"
                "minimum_retrieval_confidence=1.0\n",
                encoding="utf-8",
            )
            (data_dir / "faiss_index.bin").write_text("fake", encoding="utf-8")
            (data_dir / "faiss_metadata.json").write_text(
                json.dumps([{"filename": "course.txt", "chunk_index": 0, "chunk_text": "Context"}]),
                encoding="utf-8",
            )

            with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}), patch(
                "src.retrieval.load_numpy_module", return_value=FakeNumpy()
            ):
                result = answer_query_with_sources(
                    "Какво се оценява?",
                    settings_path=settings_path,
                    data_dir=data_dir,
                    openai_module=fake_openai,
                    faiss_module=FakeFaiss(fake_index),
                    verify=False,
                )

        self.assertIn("Не знам", result.response)
        self.assertEqual(len(fake_openai.chat.completions.calls), 1)

    def test_build_messages_omits_empty_context_block(self):
        messages = build_messages(
            "normal",
            original_question="What is covered?",
            final_query="What is covered?",
            context="",
            settings=CourseSettings(classname="Test Course"),
        )

        self.assertNotIn("Context:", messages[0]["content"])
        self.assertIn("Match the language", messages[0]["content"])

    def test_verification_fallback_preserves_original_when_alt_retrieval_is_not_answerable(self):
        fake_openai = FakeOpenAI(["No", "Draft answer.", "No"])
        first_retrieval = RetrievedContext(
            "Good context",
            [Source(filename="course.txt", chunk_index=0)],
            confidence=0.5,
            answerable=True,
        )
        weak_retrieval = RetrievedContext(
            "",
            [],
            confidence=0.0,
            answerable=False,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            settings_path = Path(tmpdir) / "settings.txt"
            settings_path.write_text(
                "embedding_method=openai\n"
                "openai_embedding_model=test-embedding\n"
                "chat_model=test-chat\n"
                "minimum_hybrid_retrieval_confidence=0.01\n",
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}), patch(
                "src.service.load_faiss_resources", return_value=(object(), [])
            ), patch(
                "src.service.get_context_from_query",
                side_effect=[first_retrieval, weak_retrieval],
            ) as get_context:
                result = answer_query_with_sources(
                    "What is covered?",
                    settings_path=settings_path,
                    openai_module=fake_openai,
                )

        self.assertEqual(result.response, "Draft answer.")
        self.assertEqual(result.sources, first_retrieval.sources)
        self.assertEqual(len(fake_openai.chat.completions.calls), 3)
        self.assertEqual(get_context.call_args_list[1].args[0], "What is covered?")
        self.assertNotIn("Source:", get_context.call_args_list[1].args[0])

    def test_verification_fallback_preserves_original_when_followup_fails_verification(self):
        fake_openai = FakeOpenAI(
            ["No", "Draft answer.", "No", "Wider answer.", "No"]
        )
        first_retrieval = RetrievedContext(
            "Good context",
            [Source(filename="course.txt", chunk_index=0)],
            confidence=0.5,
            answerable=True,
        )
        alt_retrieval = RetrievedContext(
            "Wider context",
            [Source(filename="course.txt", chunk_index=1)],
            confidence=0.5,
            answerable=True,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            settings_path = Path(tmpdir) / "settings.txt"
            settings_path.write_text(
                "embedding_method=openai\n"
                "openai_embedding_model=test-embedding\n"
                "chat_model=test-chat\n"
                "minimum_hybrid_retrieval_confidence=0.01\n",
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}), patch(
                "src.service.load_faiss_resources", return_value=(object(), [])
            ), patch(
                "src.service.get_context_from_query",
                side_effect=[first_retrieval, alt_retrieval],
            ):
                result = answer_query_with_sources(
                    "What is covered?",
                    settings_path=settings_path,
                    openai_module=fake_openai,
                )

        self.assertEqual(result.response, "Draft answer.")
        self.assertEqual(result.sources, first_retrieval.sources)
        self.assertEqual(len(fake_openai.chat.completions.calls), 5)

    def test_followup_includes_previous_conversation_summary(self):
        fake_index = FakeIndex()
        fake_openai = FakeOpenAI(["No", "Follow-up answer."])
        state = ConversationState()
        state.add_turn(
            "What is the grading policy?",
            "The final grade uses assignments.",
            "Context about grades.",
            [],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            settings_path = root / "settings.txt"
            data_dir = root / "data"
            data_dir.mkdir()
            settings_path.write_text(
                "embedding_method=openai\n"
                "openai_embedding_model=test-embedding\n"
                "chat_model=test-chat\n"
                "minimum_hybrid_retrieval_confidence=0.01\n",
                encoding="utf-8",
            )
            (data_dir / "faiss_index.bin").write_text("fake", encoding="utf-8")
            (data_dir / "faiss_metadata.json").write_text(
                json.dumps([{"filename": "course.txt", "chunk_index": 0, "chunk_text": "Context"}]),
                encoding="utf-8",
            )

            with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}), patch(
                "src.retrieval.load_numpy_module", return_value=FakeNumpy()
            ):
                result = answer_query_with_sources(
                    "What about the exam?",
                    settings_path=settings_path,
                    data_dir=data_dir,
                    openai_module=fake_openai,
                    faiss_module=FakeFaiss(fake_index),
                    verify=False,
                    conversation_state=state,
                )

        embedded_query = fake_openai.embeddings.calls[0]["input"][0]
        self.assertIn("previous chat", embedded_query)
        self.assertIn("What is the grading policy?", embedded_query)
        self.assertEqual(result.response, "Follow-up answer.")
        self.assertEqual(len(state.turns), 2)

    def test_answer_check_uses_previous_context(self):
        fake_openai = FakeOpenAI(["The answer is mostly correct."])
        state = ConversationState()
        state.add_turn(
            "What is the grading policy?",
            "Assignments determine the grade.",
            "Grading context.",
            [],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            settings_path = Path(tmpdir) / "settings.txt"
            settings_path.write_text(
                "classname=Test Course\n"
                "assistantname=Test Assistant\n"
                "embedding_method=openai\n"
                "chat_model=test-chat\n",
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
                result = answer_query_with_sources(
                    "a: assignments determine the grade",
                    settings_path=settings_path,
                    openai_module=fake_openai,
                    conversation_state=state,
                )

        self.assertEqual(result.response, "The answer is mostly correct.")
        messages = fake_openai.chat.completions.calls[0]["messages"]
        self.assertIn("Grading context.", messages[0]["content"])
        self.assertIn("assignments determine the grade", messages[1]["content"])
        self.assertEqual(state.turn_count(), 1)


if __name__ == "__main__":
    unittest.main()
