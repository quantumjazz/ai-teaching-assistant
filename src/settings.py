from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from .errors import ConfigurationError


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SETTINGS_PATH = PROJECT_ROOT / "settings.txt"
DATA_DIR = PROJECT_ROOT / "data"

DEFAULT_CHAT_MODEL = "gpt-4o-mini"
DEFAULT_EMBEDDING_MODEL = "text-embedding-ada-002"
DEFAULT_NUM_CHUNKS = 3
DEFAULT_RETRIEVAL_FETCH_MULTIPLIER = 3
DEFAULT_RRF_K = 60
DEFAULT_MAX_CHUNKS_PER_SOURCE = 2
DEFAULT_MINIMUM_RETRIEVAL_CONFIDENCE = 0.01
DEFAULT_MINIMUM_HYBRID_RETRIEVAL_CONFIDENCE = 0.025
DEFAULT_MINIMUM_LEXICAL_RETRIEVAL_CONFIDENCE = 0.20
DEFAULT_MINIMUM_VECTOR_RETRIEVAL_CONFIDENCE = 0.20


@dataclass(frozen=True)
class CourseSettings:
    classname: str = ""
    professor: str = ""
    assistants: str = ""
    classdescription: str = ""
    instructions: str = ""
    assistant_name: str = "AI Assistant"
    num_chunks: int = DEFAULT_NUM_CHUNKS
    embedding_method: str = "openai"
    openai_embedding_model: str = DEFAULT_EMBEDDING_MODEL
    chat_model: str = DEFAULT_CHAT_MODEL
    retrieval_fetch_multiplier: int = DEFAULT_RETRIEVAL_FETCH_MULTIPLIER
    lexical_rerank: bool = True
    hybrid_retrieval: bool = True
    rrf_k: int = DEFAULT_RRF_K
    max_chunks_per_source: int = DEFAULT_MAX_CHUNKS_PER_SOURCE
    minimum_retrieval_confidence: float = DEFAULT_MINIMUM_RETRIEVAL_CONFIDENCE
    minimum_hybrid_retrieval_confidence: float = DEFAULT_MINIMUM_HYBRID_RETRIEVAL_CONFIDENCE
    minimum_lexical_retrieval_confidence: float = DEFAULT_MINIMUM_LEXICAL_RETRIEVAL_CONFIDENCE
    minimum_vector_retrieval_confidence: float = DEFAULT_MINIMUM_VECTOR_RETRIEVAL_CONFIDENCE


def read_settings(file_path):
    settings = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            settings[key.strip()] = value.strip()
    return settings


def _read_positive_int(raw_value, default):
    try:
        value = int(raw_value)
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


def _read_bool(raw_value, default):
    if raw_value is None:
        return default
    value = str(raw_value).strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return default


def _read_nonnegative_float(raw_value, default):
    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        return default
    return value if value >= 0 else default


def load_course_settings(settings_path=SETTINGS_PATH):
    settings_path = Path(settings_path)
    if not settings_path.exists():
        raise ConfigurationError(f"Settings file not found: {settings_path}")

    return _load_course_settings_cached(
        str(settings_path.resolve()),
        settings_path.stat().st_mtime_ns,
    )


@lru_cache(maxsize=8)
def _load_course_settings_cached(settings_path, _mtime_ns):
    settings_path = Path(settings_path)
    raw = read_settings(settings_path)
    embedding_method = raw.get("embedding_method", "openai").lower()
    if embedding_method != "openai":
        raise ConfigurationError(
            "Only embedding_method=openai is supported in this reliability pass."
        )

    legacy_minimum_confidence = _read_nonnegative_float(
        raw.get("minimum_retrieval_confidence"),
        DEFAULT_MINIMUM_RETRIEVAL_CONFIDENCE,
    )

    return CourseSettings(
        classname=raw.get("classname", ""),
        professor=raw.get("professor", ""),
        assistants=raw.get("assistants", ""),
        classdescription=raw.get("classdescription", ""),
        instructions=raw.get("instructions", ""),
        assistant_name=raw.get("assistantname", "AI Assistant"),
        num_chunks=_read_positive_int(raw.get("num_chunks"), DEFAULT_NUM_CHUNKS),
        embedding_method=embedding_method,
        openai_embedding_model=raw.get(
            "openai_embedding_model", DEFAULT_EMBEDDING_MODEL
        ),
        chat_model=raw.get("chat_model", DEFAULT_CHAT_MODEL),
        retrieval_fetch_multiplier=_read_positive_int(
            raw.get("retrieval_fetch_multiplier"),
            DEFAULT_RETRIEVAL_FETCH_MULTIPLIER,
        ),
        lexical_rerank=_read_bool(raw.get("lexical_rerank"), True),
        hybrid_retrieval=_read_bool(raw.get("hybrid_retrieval"), True),
        rrf_k=_read_positive_int(raw.get("rrf_k"), DEFAULT_RRF_K),
        max_chunks_per_source=_read_positive_int(
            raw.get("max_chunks_per_source"),
            DEFAULT_MAX_CHUNKS_PER_SOURCE,
        ),
        minimum_retrieval_confidence=legacy_minimum_confidence,
        minimum_hybrid_retrieval_confidence=_read_nonnegative_float(
            raw.get("minimum_hybrid_retrieval_confidence"),
            legacy_minimum_confidence
            if "minimum_retrieval_confidence" in raw
            else DEFAULT_MINIMUM_HYBRID_RETRIEVAL_CONFIDENCE,
        ),
        minimum_lexical_retrieval_confidence=_read_nonnegative_float(
            raw.get("minimum_lexical_retrieval_confidence"),
            legacy_minimum_confidence
            if "minimum_retrieval_confidence" in raw
            else DEFAULT_MINIMUM_LEXICAL_RETRIEVAL_CONFIDENCE,
        ),
        minimum_vector_retrieval_confidence=_read_nonnegative_float(
            raw.get("minimum_vector_retrieval_confidence"),
            legacy_minimum_confidence
            if "minimum_retrieval_confidence" in raw
            else DEFAULT_MINIMUM_VECTOR_RETRIEVAL_CONFIDENCE,
        ),
    )


def clear_settings_cache():
    _load_course_settings_cached.cache_clear()
