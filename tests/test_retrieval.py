import unittest
from types import SimpleNamespace
from unittest.mock import patch

from src import retrieval as retrieval_module
from src.retrieval import (
    BM25_B,
    BM25_K1,
    RetrievalCandidate,
    Source,
    _search_k,
    _tokenize,
    bm25_candidates,
    get_context_from_query,
    merge_ranked_candidates,
    rank_hybrid_candidates,
    retrieval_confidence,
    retrieval_confidence_threshold,
    select_diverse_candidates,
    rank_candidates,
)
from src.settings import CourseSettings


class FakeNumpy:
    float32 = "float32"

    def array(self, value, dtype=None):
        return value

    def expand_dims(self, value, axis=0):
        return [value]


class FakeEmbeddings:
    def create(self, model, input):
        return SimpleNamespace(data=[SimpleNamespace(embedding=[0.1, 0.2])])


class FakeOpenAI:
    embeddings = FakeEmbeddings()


class FakeIndex:
    def search(self, query_embedding, k):
        return [[0.1, 0.2, 0.3]], [[0, 1, 2]]


class RetrievalTests(unittest.TestCase):
    def test_lexical_rerank_promotes_matching_candidate(self):
        settings = CourseSettings(num_chunks=2, lexical_rerank=True)
        candidates = [
            RetrievalCandidate(
                chunk_text="General class overview.",
                source=Source(filename="overview.txt", chunk_index=0),
                vector_rank=0,
            ),
            RetrievalCandidate(
                chunk_text="The final grade policy uses assignments and an exam.",
                source=Source(filename="syllabus.txt", chunk_index=1),
                vector_rank=1,
            ),
        ]

        ranked = rank_candidates("final grade policy", candidates, settings)

        self.assertEqual(ranked[0].source.filename, "syllabus.txt")
        self.assertGreater(ranked[0].source.lexical_score, ranked[1].source.lexical_score)

    def test_lexical_rerank_can_be_disabled(self):
        settings = CourseSettings(num_chunks=2, lexical_rerank=False)
        candidates = [
            RetrievalCandidate("General class overview.", Source("overview.txt", 0), 0),
            RetrievalCandidate("The final grade policy.", Source("syllabus.txt", 1), 1),
        ]

        ranked = rank_candidates("final grade policy", candidates, settings)

        self.assertEqual([item.source.filename for item in ranked], ["overview.txt", "syllabus.txt"])

    def test_bm25_candidates_rank_matching_chunks(self):
        retrieval_module.clear_retrieval_cache()
        metadata = [
            {"filename": "overview.txt", "chunk_index": 0, "chunk_text": "General overview."},
            {
                "filename": "syllabus.txt",
                "chunk_index": 1,
                "chunk_text": "The final grade policy uses assignments and an exam.",
            },
        ]

        candidates = bm25_candidates("final grade policy", metadata, limit=2)

        self.assertEqual(candidates[0].source.filename, "syllabus.txt")
        self.assertGreater(candidates[0].source.lexical_score, 0)
        self.assertEqual(candidates[0].source.lexical_rank, 0)
        self.assertIsNone(candidates[0].source.vector_rank)

    def test_bm25_candidates_reuses_cached_corpus_until_cache_clear(self):
        retrieval_module.clear_retrieval_cache()
        metadata = [
            {"filename": "overview.txt", "chunk_index": 0, "chunk_text": "General overview."},
            {
                "filename": "syllabus.txt",
                "chunk_index": 1,
                "chunk_text": "The final grade policy uses assignments and an exam.",
            },
        ]

        bm25_candidates("final grade policy", metadata, limit=2)
        first_info = retrieval_module._bm25_corpus_cached.cache_info()
        bm25_candidates("exam policy", metadata, limit=2)
        second_info = retrieval_module._bm25_corpus_cached.cache_info()
        retrieval_module.clear_retrieval_cache()
        cleared_info = retrieval_module._bm25_corpus_cached.cache_info()

        self.assertEqual(first_info.misses, 1)
        self.assertEqual(second_info.hits, 1)
        self.assertEqual(cleared_info.hits, 0)
        self.assertEqual(cleared_info.misses, 0)

    def test_rrf_merge_promotes_candidate_found_by_both_methods(self):
        vector_candidates = [
            RetrievalCandidate("A", Source("a.txt", 0), vector_rank=0, metadata_index=0),
            RetrievalCandidate("B", Source("b.txt", 0), vector_rank=1, metadata_index=1),
        ]
        lexical_candidates = [
            RetrievalCandidate(
                "B",
                Source("b.txt", 0, lexical_score=2.0),
                vector_rank=None,
                metadata_index=1,
                lexical_rank=0,
            )
        ]

        merged = merge_ranked_candidates(vector_candidates, lexical_candidates, rrf_k=60)

        self.assertEqual(merged[0].source.filename, "b.txt")
        self.assertEqual(merged[0].source.vector_rank, 1)
        self.assertEqual(merged[0].source.lexical_rank, 0)
        self.assertGreater(merged[0].source.hybrid_score, merged[1].source.hybrid_score)

    def test_hybrid_lexical_rerank_keeps_hybrid_score_primary(self):
        settings = CourseSettings(hybrid_retrieval=True, lexical_rerank=True)
        candidates = [
            RetrievalCandidate(
                "The grading policy is listed in the syllabus.",
                Source("syllabus.txt", 0, hybrid_score=0.05),
                vector_rank=1,
            ),
            RetrievalCandidate(
                "A general course overview.",
                Source("overview.txt", 0, hybrid_score=0.06),
                vector_rank=0,
            ),
        ]

        ranked = rank_hybrid_candidates("grading policy", candidates, settings)

        self.assertEqual([item.source.filename for item in ranked], ["overview.txt", "syllabus.txt"])

    def test_hybrid_lexical_rerank_breaks_hybrid_ties(self):
        settings = CourseSettings(hybrid_retrieval=True, lexical_rerank=True)
        candidates = [
            RetrievalCandidate(
                "A general course overview.",
                Source("overview.txt", 0, hybrid_score=0.05),
                vector_rank=0,
            ),
            RetrievalCandidate(
                "The grading policy is listed in the syllabus.",
                Source("syllabus.txt", 0, hybrid_score=0.05),
                vector_rank=1,
            ),
        ]

        ranked = rank_hybrid_candidates("grading policy", candidates, settings)

        self.assertEqual([item.source.filename for item in ranked], ["syllabus.txt", "overview.txt"])

    def test_get_context_hybrid_end_to_end_uses_vector_and_bm25(self):
        retrieval_module.clear_retrieval_cache()
        settings = CourseSettings(
            num_chunks=2,
            hybrid_retrieval=True,
            lexical_rerank=True,
            minimum_hybrid_retrieval_confidence=0.0,
        )
        metadata = [
            {"filename": "overview.txt", "chunk_index": 0, "chunk_text": "General course overview."},
            {
                "filename": "syllabus.txt",
                "chunk_index": 1,
                "chunk_text": "The final grade policy uses assignments and an exam.",
            },
            {"filename": "calendar.txt", "chunk_index": 2, "chunk_text": "Week one introduces methods."},
        ]

        with patch("src.retrieval.load_numpy_module", return_value=FakeNumpy()):
            retrieved = get_context_from_query(
                "final grade policy",
                index=FakeIndex(),
                metadata=metadata,
                settings=settings,
                openai_module=FakeOpenAI(),
            )

        self.assertTrue(retrieved.answerable)
        self.assertEqual(retrieved.sources[0].filename, "syllabus.txt")
        self.assertGreater(retrieved.sources[0].hybrid_score, 0)
        self.assertEqual(retrieved.sources[0].vector_rank, 1)
        self.assertEqual(retrieved.sources[0].lexical_rank, 0)

    def test_cyrillic_hybrid_query_can_use_strong_vector_hit_without_lexical_match(self):
        retrieval_module.clear_retrieval_cache()
        settings = CourseSettings(
            num_chunks=2,
            hybrid_retrieval=True,
            lexical_rerank=True,
            minimum_hybrid_retrieval_confidence=0.025,
            minimum_vector_retrieval_confidence=0.20,
        )
        metadata = [
            {
                "filename": "property.txt",
                "chunk_index": 0,
                "chunk_text": "Residual decision authority under incomplete contracts.",
            },
            {"filename": "overview.txt", "chunk_index": 1, "chunk_text": "General overview."},
        ]

        with patch("src.retrieval.load_numpy_module", return_value=FakeNumpy()):
            retrieved = get_context_from_query(
                "Какво е това?\n\nEnglish retrieval query: agency theory",
                index=FakeIndex(),
                metadata=metadata,
                settings=settings,
                openai_module=FakeOpenAI(),
            )

        self.assertTrue(retrieved.answerable)
        self.assertEqual(retrieved.sources[0].filename, "property.txt")
        self.assertIsNone(retrieved.sources[0].lexical_rank)
        self.assertLess(retrieved.confidence, settings.minimum_hybrid_retrieval_confidence)

    def test_cyrillic_hybrid_query_checks_any_selected_vector_hit(self):
        retrieval_module.clear_retrieval_cache()
        settings = CourseSettings(
            num_chunks=2,
            hybrid_retrieval=True,
            lexical_rerank=True,
            minimum_hybrid_retrieval_confidence=0.025,
            minimum_vector_retrieval_confidence=0.20,
        )
        metadata = [
            {
                "filename": "property.txt",
                "chunk_index": 0,
                "chunk_text": "Residual decision authority under incomplete contracts.",
            },
            {"filename": "overview.txt", "chunk_index": 1, "chunk_text": "General overview."},
            {"filename": "schedule.txt", "chunk_index": 2, "chunk_text": "Class schedule."},
            {"filename": "agency.txt", "chunk_index": 3, "chunk_text": "Agency theory."},
        ]

        with patch("src.retrieval.load_numpy_module", return_value=FakeNumpy()):
            retrieved = get_context_from_query(
                "Какво е това?\n\nEnglish retrieval query: agency theory",
                index=FakeIndex(),
                metadata=metadata,
                settings=settings,
                openai_module=FakeOpenAI(),
            )

        self.assertTrue(retrieved.answerable)
        self.assertEqual(retrieved.sources[0].filename, "agency.txt")
        self.assertIsNone(retrieved.sources[0].vector_rank)
        self.assertEqual(retrieved.sources[1].filename, "property.txt")
        self.assertEqual(retrieved.sources[1].vector_rank, 0)
        self.assertLess(retrieved.confidence, settings.minimum_hybrid_retrieval_confidence)

    def test_english_hybrid_query_still_requires_hybrid_confidence(self):
        retrieval_module.clear_retrieval_cache()
        settings = CourseSettings(
            num_chunks=2,
            hybrid_retrieval=True,
            lexical_rerank=True,
            minimum_hybrid_retrieval_confidence=0.025,
            minimum_vector_retrieval_confidence=0.20,
        )
        metadata = [
            {
                "filename": "property.txt",
                "chunk_index": 0,
                "chunk_text": "Residual decision authority under incomplete contracts.",
            },
            {"filename": "overview.txt", "chunk_index": 1, "chunk_text": "General overview."},
        ]

        with patch("src.retrieval.load_numpy_module", return_value=FakeNumpy()):
            retrieved = get_context_from_query(
                "agency theory",
                index=FakeIndex(),
                metadata=metadata,
                settings=settings,
                openai_module=FakeOpenAI(),
            )

        self.assertFalse(retrieved.answerable)
        self.assertEqual(retrieved.sources[0].filename, "property.txt")

    def test_diversity_limits_repeated_sources_before_filling(self):
        candidates = [
            RetrievalCandidate("A1", Source("a.txt", 0, source_path="documents/a.txt"), 0),
            RetrievalCandidate("A2", Source("a.txt", 1, source_path="documents/a.txt"), 1),
            RetrievalCandidate("B1", Source("b.txt", 0, source_path="documents/b.txt"), 2),
        ]

        selected = select_diverse_candidates(
            candidates,
            limit=2,
            max_chunks_per_source=1,
        )

        self.assertEqual([candidate.source.filename for candidate in selected], ["a.txt", "b.txt"])

    def test_retrieval_confidence_uses_hybrid_score(self):
        candidates = [
            RetrievalCandidate("A", Source("a.txt", 0, hybrid_score=0.25), 0),
        ]

        self.assertEqual(retrieval_confidence(candidates), 0.25)

    def test_retrieval_confidence_threshold_uses_retrieval_mode(self):
        self.assertEqual(
            retrieval_confidence_threshold(CourseSettings(hybrid_retrieval=True)),
            0.025,
        )
        self.assertEqual(
            retrieval_confidence_threshold(
                CourseSettings(hybrid_retrieval=False, lexical_rerank=True)
            ),
            0.20,
        )
        self.assertEqual(
            retrieval_confidence_threshold(
                CourseSettings(hybrid_retrieval=False, lexical_rerank=False)
            ),
            0.20,
        )

    def test_search_k_uses_fetch_multiplier_with_metadata_cap(self):
        settings = CourseSettings(num_chunks=3, retrieval_fetch_multiplier=4)

        self.assertEqual(_search_k(settings, [{"chunk_text": "x"}] * 20), 12)
        self.assertEqual(_search_k(settings, [{"chunk_text": "x"}] * 5), 5)

    def test_tokenize_ignores_tokens_shorter_than_three_characters(self):
        self.assertEqual(_tokenize("of by is a cat или"), ["cat", "или"])

    def test_bm25_constants_are_named_defaults(self):
        self.assertEqual(BM25_K1, 1.5)
        self.assertEqual(BM25_B, 0.75)


if __name__ == "__main__":
    unittest.main()
