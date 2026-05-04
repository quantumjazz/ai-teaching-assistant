import json
import math
import re
from dataclasses import dataclass
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

from .clients import load_faiss_module, load_numpy_module
from .errors import KnowledgeBaseNotReady
from .settings import DATA_DIR


BM25_K1 = 1.5
BM25_B = 0.75
MIN_TOKEN_LENGTH = 3


@dataclass(frozen=True)
class Source:
    filename: str
    chunk_index: int
    distance: Optional[float] = None
    snippet: str = ""
    lexical_score: float = 0.0
    hybrid_score: float = 0.0
    vector_rank: Optional[int] = None
    lexical_rank: Optional[int] = None
    source_path: str = ""
    document_title: str = ""
    page_number: Optional[int] = None
    section_title: str = ""
    chunk_start_char: Optional[int] = None
    chunk_end_char: Optional[int] = None

    def to_dict(self):
        return {
            "filename": self.filename,
            "chunk_index": self.chunk_index,
            "distance": self.distance,
            "snippet": self.snippet,
            "lexical_score": self.lexical_score,
            "hybrid_score": self.hybrid_score,
            "vector_rank": self.vector_rank,
            "lexical_rank": self.lexical_rank,
            "source_path": self.source_path,
            "document_title": self.document_title,
            "page_number": self.page_number,
            "section_title": self.section_title,
            "chunk_start_char": self.chunk_start_char,
            "chunk_end_char": self.chunk_end_char,
        }


@dataclass(frozen=True)
class RetrievedContext:
    text: str
    sources: List[Source]
    confidence: float = 0.0
    answerable: bool = True


@dataclass(frozen=True)
class RetrievalCandidate:
    chunk_text: str
    source: Source
    vector_rank: Optional[int]
    metadata_index: int = -1
    lexical_rank: Optional[int] = None


def load_faiss_resources(data_dir=DATA_DIR, faiss_module=None):
    data_dir = Path(data_dir)
    faiss_index_path = data_dir / "faiss_index.bin"
    metadata_path = data_dir / "faiss_metadata.json"

    if not faiss_index_path.exists() or not metadata_path.exists():
        raise KnowledgeBaseNotReady(
            "Course knowledge base is not ready. Add course files to documents/ "
            "and run the indexing scripts from run_instructions.txt."
        )

    if faiss_module is None:
        return _load_faiss_resources_cached(
            str(data_dir.resolve()),
            faiss_index_path.stat().st_mtime_ns,
            metadata_path.stat().st_mtime_ns,
        )

    faiss_module = load_faiss_module(faiss_module)
    index = faiss_module.read_index(str(faiss_index_path))
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return index, metadata


@lru_cache(maxsize=4)
def _load_faiss_resources_cached(data_dir, _index_mtime_ns, _metadata_mtime_ns):
    data_dir = Path(data_dir)
    faiss_module = load_faiss_module()
    index = faiss_module.read_index(str(data_dir / "faiss_index.bin"))
    with open(data_dir / "faiss_metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return index, metadata


def clear_retrieval_cache():
    _load_faiss_resources_cached.cache_clear()
    _bm25_corpus_cached.cache_clear()


def embed_query(query, embedding_model, openai_module=None):
    np = load_numpy_module()
    response = openai_module.embeddings.create(model=embedding_model, input=[query])
    embedding = response.data[0].embedding
    return np.array(embedding, dtype=np.float32)


def get_context_from_query(query, index, metadata, settings, openai_module=None):
    np = load_numpy_module()
    search_k = _search_k(settings, metadata)
    query_embedding = embed_query(
        query,
        embedding_model=settings.openai_embedding_model,
        openai_module=openai_module,
    )
    query_embedding = np.expand_dims(query_embedding, axis=0)
    distances, indices = index.search(query_embedding, search_k)

    vector_candidates = []
    for position, idx in enumerate(indices[0]):
        idx = int(idx)
        if 0 <= idx < len(metadata):
            record = metadata[idx]
            vector_candidates.append(
                RetrievalCandidate(
                    chunk_text=record["chunk_text"],
                    source=_source_from_record(record, idx, distances[0], position),
                    vector_rank=position,
                    metadata_index=idx,
                )
            )

    if settings.hybrid_retrieval:
        lexical_candidates = bm25_candidates(query, metadata, search_k)
        candidates = merge_ranked_candidates(
            vector_candidates,
            lexical_candidates,
            rrf_k=settings.rrf_k,
        )
        candidates = rank_hybrid_candidates(query, candidates, settings)
    else:
        candidates = rank_candidates(query, vector_candidates, settings)

    candidates = select_diverse_candidates(
        candidates,
        limit=settings.num_chunks,
        max_chunks_per_source=settings.max_chunks_per_source,
    )
    confidence = retrieval_confidence(candidates)
    return RetrievedContext(
        text=_format_context(candidates),
        sources=[candidate.source for candidate in candidates],
        confidence=confidence,
        answerable=retrieval_answerable(query, candidates, settings),
    )


def rank_candidates(query, candidates, settings):
    if not settings.lexical_rerank:
        return list(candidates)

    query_terms = _tokenize(query)
    if not query_terms:
        return list(candidates)

    ranked = []
    for candidate in candidates:
        lexical_score = lexical_overlap_score(query_terms, candidate.chunk_text)
        ranked.append(
            RetrievalCandidate(
                chunk_text=candidate.chunk_text,
                source=replace(candidate.source, lexical_score=lexical_score),
                vector_rank=candidate.vector_rank,
                metadata_index=candidate.metadata_index,
                lexical_rank=candidate.lexical_rank,
            )
        )
    return sorted(
        ranked,
        key=lambda item: (-item.source.lexical_score, _rank_sort_value(item.vector_rank)),
    )


def rank_hybrid_candidates(query, candidates, settings):
    if not settings.lexical_rerank:
        return list(candidates)

    query_terms = _tokenize(query)
    if not query_terms:
        return list(candidates)

    return sorted(
        candidates,
        key=lambda candidate: _hybrid_sort_key(
            candidate,
            lexical_overlap_score(query_terms, candidate.chunk_text),
        ),
    )


def bm25_candidates(query, metadata, limit):
    query_terms = _tokenize(query)
    if not query_terms or not metadata:
        return []

    chunk_texts = tuple(str(record.get("chunk_text", "")) for record in metadata)
    documents, document_frequencies, document_count, average_length = _bm25_corpus_cached(
        chunk_texts
    )
    if average_length == 0:
        return []

    scored = []
    for index, document in enumerate(documents):
        score = bm25_score(
            query_terms=query_terms,
            document_terms=document,
            document_frequencies=document_frequencies,
            document_count=document_count,
            average_length=average_length,
        )
        if score <= 0:
            continue
        record = metadata[index]
        scored.append(
            RetrievalCandidate(
                chunk_text=record["chunk_text"],
                source=replace(
                    _source_from_record(record, index, [], None),
                    lexical_score=score,
                    lexical_rank=0,
                ),
                vector_rank=None,
                metadata_index=index,
                lexical_rank=0,
            )
        )

    scored.sort(key=lambda candidate: (-candidate.source.lexical_score, candidate.metadata_index))
    ranked = []
    for rank, candidate in enumerate(scored[:limit]):
        ranked.append(
            RetrievalCandidate(
                chunk_text=candidate.chunk_text,
                source=replace(candidate.source, lexical_rank=rank),
                vector_rank=candidate.vector_rank,
                metadata_index=candidate.metadata_index,
                lexical_rank=rank,
            )
        )
    return ranked


@lru_cache(maxsize=8)
def _bm25_corpus_cached(chunk_texts):
    documents = tuple(_tokenize(text) for text in chunk_texts)
    document_count = len(documents)
    if document_count == 0:
        return documents, {}, 0, 0.0

    average_length = sum(len(document) for document in documents) / document_count
    document_frequencies: Dict[str, int] = {}
    for document in documents:
        for term in set(document):
            document_frequencies[term] = document_frequencies.get(term, 0) + 1
    return documents, document_frequencies, document_count, average_length


def bm25_score(query_terms, document_terms, document_frequencies, document_count, average_length):
    if not document_terms:
        return 0.0
    term_counts = {}
    for term in document_terms:
        term_counts[term] = term_counts.get(term, 0) + 1

    document_length = len(document_terms)
    score = 0.0
    for term in query_terms:
        frequency = term_counts.get(term, 0)
        if frequency == 0:
            continue
        df = document_frequencies.get(term, 0)
        idf = math.log(1 + (document_count - df + 0.5) / (df + 0.5))
        denominator = frequency + BM25_K1 * (
            1 - BM25_B + BM25_B * document_length / average_length
        )
        score += idf * (frequency * (BM25_K1 + 1)) / denominator
    return score


def merge_ranked_candidates(vector_candidates, lexical_candidates, rrf_k=60):
    merged = {}

    for rank, candidate in enumerate(vector_candidates):
        key = candidate.metadata_index
        existing = merged.get(key)
        source = replace(
            candidate.source,
            vector_rank=rank,
            hybrid_score=(existing.source.hybrid_score if existing else 0.0)
            + reciprocal_rank_score(rank, rrf_k),
        )
        merged[key] = RetrievalCandidate(
            chunk_text=candidate.chunk_text,
            source=source,
            vector_rank=rank,
            metadata_index=key,
            lexical_rank=existing.lexical_rank if existing else None,
        )

    for rank, candidate in enumerate(lexical_candidates):
        key = candidate.metadata_index
        existing = merged.get(key)
        if existing:
            source = replace(
                existing.source,
                lexical_score=candidate.source.lexical_score,
                lexical_rank=rank,
                hybrid_score=existing.source.hybrid_score + reciprocal_rank_score(rank, rrf_k),
            )
            merged[key] = RetrievalCandidate(
                chunk_text=existing.chunk_text,
                source=source,
                vector_rank=existing.vector_rank,
                metadata_index=key,
                lexical_rank=rank,
            )
            continue
        source = replace(
            candidate.source,
            lexical_rank=rank,
            hybrid_score=reciprocal_rank_score(rank, rrf_k),
        )
        merged[key] = RetrievalCandidate(
            chunk_text=candidate.chunk_text,
            source=source,
            vector_rank=candidate.vector_rank,
            metadata_index=key,
            lexical_rank=rank,
        )

    return sorted(
        merged.values(),
        key=lambda candidate: _hybrid_sort_key(candidate),
    )


def reciprocal_rank_score(rank, rrf_k):
    return 1.0 / (rrf_k + rank + 1)


def select_diverse_candidates(candidates, limit, max_chunks_per_source):
    selected = []
    source_counts = {}
    deferred = []

    for candidate in candidates:
        key = _diversity_key(candidate.source)
        count = source_counts.get(key, 0)
        if count < max_chunks_per_source:
            selected.append(candidate)
            source_counts[key] = count + 1
        else:
            deferred.append(candidate)
        if len(selected) == limit:
            return selected

    for candidate in deferred:
        selected.append(candidate)
        if len(selected) == limit:
            break
    return selected


def retrieval_confidence(candidates):
    if not candidates:
        return 0.0
    top = candidates[0].source
    if top.hybrid_score:
        return top.hybrid_score
    if top.lexical_score:
        return top.lexical_score
    if top.distance is not None:
        return 1.0 / (1.0 + max(top.distance, 0.0))
    return 0.0


def retrieval_answerable(query, candidates, settings):
    if not candidates:
        return False

    if retrieval_confidence(candidates) >= retrieval_confidence_threshold(settings):
        return True

    top = candidates[0].source
    if (
        settings.hybrid_retrieval
        and _uses_cyrillic(query)
        and top.vector_rank is not None
        and vector_confidence(top) > settings.minimum_vector_retrieval_confidence
    ):
        return True

    return False


def vector_confidence(source):
    if source.distance is None:
        return 0.0
    return 1.0 / (1.0 + max(source.distance, 0.0))


def retrieval_confidence_threshold(settings):
    if settings.hybrid_retrieval:
        return settings.minimum_hybrid_retrieval_confidence
    if settings.lexical_rerank:
        return settings.minimum_lexical_retrieval_confidence
    return settings.minimum_vector_retrieval_confidence


def lexical_overlap_score(query_terms, text):
    text_terms = set(_tokenize(text))
    if not text_terms:
        return 0.0
    matches = sum(1 for term in query_terms if term in text_terms)
    return matches / len(query_terms)


def _tokenize(text):
    """Tokenize Unicode words and ignore tokens shorter than MIN_TOKEN_LENGTH."""
    terms = re.findall(r"\w+", text.lower(), flags=re.UNICODE)
    return [term for term in terms if len(term) >= MIN_TOKEN_LENGTH]


def _uses_cyrillic(text):
    return any("\u0400" <= char <= "\u04ff" for char in text)


def _search_k(settings, metadata):
    metadata_count = len(metadata)
    if metadata_count == 0:
        return settings.num_chunks
    return min(
        metadata_count,
        settings.num_chunks * settings.retrieval_fetch_multiplier,
    )


def _rank_sort_value(rank):
    return rank is None, rank if rank is not None else 0


def _hybrid_sort_key(candidate, lexical_overlap=0.0):
    return (
        -candidate.source.hybrid_score,
        -lexical_overlap,
        _rank_sort_value(candidate.source.vector_rank),
        _rank_sort_value(candidate.source.lexical_rank),
    )


def _format_context(candidates):
    parts = []
    for candidate in candidates:
        source = candidate.source
        details = [f"Source: {source.filename}", f"chunk {source.chunk_index}"]
        if source.page_number:
            details.append(f"page {source.page_number}")
        if source.section_title:
            details.append(f"section {source.section_title}")
        if source.chunk_start_char is not None and source.chunk_end_char is not None:
            details.append(f"chars {source.chunk_start_char}-{source.chunk_end_char}")
        parts.append(
            ", ".join(details)
            + "\n"
            f"{candidate.chunk_text}"
        )
    return "\n\n".join(parts)


def _source_from_record(record, fallback_chunk_index, distances, position):
    try:
        chunk_index = int(record.get("chunk_index", fallback_chunk_index))
    except (TypeError, ValueError):
        chunk_index = fallback_chunk_index

    distance = None
    if position is not None and position < len(distances):
        try:
            distance = float(distances[position])
        except (TypeError, ValueError):
            distance = None

    return Source(
        filename=str(record.get("filename", "Unknown source")),
        chunk_index=chunk_index,
        distance=distance,
        snippet=_snippet(record.get("chunk_text", "")),
        vector_rank=position,
        source_path=str(record.get("source_path", "")),
        document_title=str(record.get("document_title", "")),
        page_number=_optional_int(record.get("page_number")),
        section_title=str(record.get("section_title", "")),
        chunk_start_char=_optional_int(record.get("chunk_start_char")),
        chunk_end_char=_optional_int(record.get("chunk_end_char")),
    )


def _diversity_key(source):
    if source.source_path:
        return source.source_path
    return source.filename


def _optional_int(value):
    if value in ("", None):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _snippet(text, max_length=240):
    text = " ".join(str(text).split())
    if len(text) <= max_length:
        return text
    return text[: max_length - 3].rstrip() + "..."
