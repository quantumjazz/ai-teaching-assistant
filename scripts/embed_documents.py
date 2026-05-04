import csv
import os
import pickle
import sys
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.clients import load_openai_module
from src.env_utils import load_dotenv_if_available
from src.settings import read_settings, runtime_paths


class IngestionError(Exception):
    pass


def read_chopped_csv(csv_path: str):
    """
    Reads chunked data from a CSV file.
    Returns a list of dicts, each with keys:
      'filename', 'chunk_index', 'chunk_text'
    """
    required = {"filename", "chunk_index", "chunk_text"}
    data = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
            raise IngestionError(
                "chopped_text.csv must contain filename, chunk_index, and chunk_text columns"
            )
        source_fingerprint_fields = {
            "source_path",
            "source_name",
            "source_extension",
            "source_size_bytes",
            "source_modified_at",
            "source_modified_at_ns",
            "source_sha256",
        }
        fieldnames = set(reader.fieldnames)
        if not source_fingerprint_fields.issubset(fieldnames):
            missing = source_fingerprint_fields.difference(fieldnames)
            raise IngestionError(
                "chopped_text.csv is missing source fingerprint columns: "
                + ", ".join(sorted(missing))
                + ". Rerun scripts/prepare_documents.py."
            )
        chunk_metadata_fields = {
            "document_title",
            "page_number",
            "section_title",
            "chunk_start_char",
            "chunk_end_char",
            "chunk_word_count",
        }
        present_chunk_metadata = chunk_metadata_fields.intersection(fieldnames)
        if present_chunk_metadata and not chunk_metadata_fields.issubset(fieldnames):
            missing = chunk_metadata_fields.difference(fieldnames)
            raise IngestionError(
                "chopped_text.csv has incomplete chunk metadata columns: "
                + ", ".join(sorted(missing))
            )
        has_chunk_metadata = chunk_metadata_fields.issubset(fieldnames)
        for row_number, row in enumerate(reader, start=2):
            filename = (row.get("filename") or "").strip()
            chunk_text = (row.get("chunk_text") or "").strip()
            if not filename:
                raise IngestionError(f"Row {row_number} has an empty filename")
            if not chunk_text:
                raise IngestionError(f"Row {row_number} has empty chunk_text")
            try:
                chunk_index = int(row.get("chunk_index", ""))
            except ValueError as exc:
                raise IngestionError(f"Row {row_number} has invalid chunk_index") from exc
            record = {
                "filename": filename,
                "chunk_index": chunk_index,
                "chunk_text": chunk_text,
            }
            record.update(read_source_fingerprint(row, row_number))
            if has_chunk_metadata:
                record.update(read_chunk_metadata(row, row_number))
            data.append(record)
    return data


def read_source_fingerprint(row, row_number):
    source_path = (row.get("source_path") or "").strip()
    source_name = (row.get("source_name") or "").strip()
    source_extension = (row.get("source_extension") or "").strip()
    source_modified_at = (row.get("source_modified_at") or "").strip()
    source_sha256 = (row.get("source_sha256") or "").strip()
    if not source_path:
        raise IngestionError(f"Row {row_number} has empty source_path")
    if not source_name:
        raise IngestionError(f"Row {row_number} has empty source_name")
    if not source_extension:
        raise IngestionError(f"Row {row_number} has empty source_extension")
    if not source_modified_at:
        raise IngestionError(f"Row {row_number} has empty source_modified_at")
    if not source_sha256:
        raise IngestionError(f"Row {row_number} has empty source_sha256")
    try:
        source_size_bytes = int(row.get("source_size_bytes", ""))
    except ValueError as exc:
        raise IngestionError(f"Row {row_number} has invalid source_size_bytes") from exc
    try:
        source_modified_at_ns = int(row.get("source_modified_at_ns", ""))
    except ValueError as exc:
        raise IngestionError(f"Row {row_number} has invalid source_modified_at_ns") from exc
    if source_size_bytes < 0:
        raise IngestionError(f"Row {row_number} has invalid source_size_bytes")
    if source_modified_at_ns <= 0:
        raise IngestionError(f"Row {row_number} has invalid source_modified_at_ns")
    return {
        "source_path": source_path,
        "source_name": source_name,
        "source_extension": source_extension,
        "source_size_bytes": source_size_bytes,
        "source_modified_at": source_modified_at,
        "source_modified_at_ns": source_modified_at_ns,
        "source_sha256": source_sha256,
    }


def read_chunk_metadata(row, row_number):
    document_title = (row.get("document_title") or "").strip()
    page_number = (row.get("page_number") or "").strip()
    section_title = (row.get("section_title") or "").strip()
    try:
        chunk_start_char = int(row.get("chunk_start_char", ""))
    except ValueError as exc:
        raise IngestionError(f"Row {row_number} has invalid chunk_start_char") from exc
    try:
        chunk_end_char = int(row.get("chunk_end_char", ""))
    except ValueError as exc:
        raise IngestionError(f"Row {row_number} has invalid chunk_end_char") from exc
    try:
        chunk_word_count = int(row.get("chunk_word_count", ""))
    except ValueError as exc:
        raise IngestionError(f"Row {row_number} has invalid chunk_word_count") from exc
    if chunk_start_char < 0:
        raise IngestionError(f"Row {row_number} has invalid chunk_start_char")
    if chunk_end_char < chunk_start_char:
        raise IngestionError(f"Row {row_number} has invalid chunk_end_char")
    if chunk_word_count <= 0:
        raise IngestionError(f"Row {row_number} has invalid chunk_word_count")

    metadata = {
        "document_title": document_title,
        "page_number": None,
        "section_title": section_title,
        "chunk_start_char": chunk_start_char,
        "chunk_end_char": chunk_end_char,
        "chunk_word_count": chunk_word_count,
    }
    if page_number:
        try:
            metadata["page_number"] = int(page_number)
        except ValueError as exc:
            raise IngestionError(f"Row {row_number} has invalid page_number") from exc
        if metadata["page_number"] <= 0:
            raise IngestionError(f"Row {row_number} has invalid page_number")
    return metadata


def embed_with_openai(
    texts,
    model: str,
    max_tokens_per_batch: int,
    openai_module=None,
    sleep_seconds: int = 0,
):
    """
    Batches texts based on a simple token count (words) and sends them to OpenAI's API.
    Uses openai.embeddings.create with the new API.
    """
    if not texts:
        raise IngestionError("No chunk text provided for embedding")
    if max_tokens_per_batch <= 0:
        raise IngestionError("max_tokens_per_batch must be greater than zero")

    openai_module = load_openai_module(openai_module)

    def count_tokens(text: str) -> int:
        return len(text.split())

    embeddings = []
    batch = []
    current_tokens = 0

    def embed_current_batch(has_more_batches):
        batch_embeddings = _embed_batch_with_rate_limit_retry(
            openai_module,
            model,
            batch,
            sleep_seconds,
        )
        if has_more_batches and sleep_seconds > 0:
            print(f"Waiting {sleep_seconds} seconds before the next embedding batch...")
            time.sleep(sleep_seconds)
        return batch_embeddings

    for text in texts:
        tokens = count_tokens(text)
        if batch and current_tokens + tokens > max_tokens_per_batch:
            embeddings.extend(embed_current_batch(has_more_batches=True))
            batch = []
            current_tokens = 0
        batch.append(text)
        current_tokens += tokens
    if batch:
        embeddings.extend(embed_current_batch(has_more_batches=False))
    return embeddings


def _embed_batch_with_rate_limit_retry(openai_module, model, batch, sleep_seconds):
    try:
        return _embed_batch(openai_module, model, batch)
    except Exception as exc:
        if not _is_rate_limit_error(openai_module, exc) or sleep_seconds <= 0:
            raise
        print(f"Rate limit from OpenAI. Waiting {sleep_seconds} seconds before retrying...")
        time.sleep(sleep_seconds)
        return _embed_batch(openai_module, model, batch)


def _is_rate_limit_error(openai_module, exc):
    rate_limit_error = getattr(openai_module, "RateLimitError", None)
    if isinstance(rate_limit_error, type) and isinstance(exc, rate_limit_error):
        return True
    return exc.__class__.__name__ == "RateLimitError"


def _embed_batch(openai_module, model, batch):
    response = openai_module.embeddings.create(model=model, input=batch)
    embeddings = [item.embedding for item in response.data]
    if len(embeddings) != len(batch):
        raise IngestionError(
            f"OpenAI returned {len(embeddings)} embedding(s) for {len(batch)} input chunk(s)"
        )
    return embeddings


def attach_embeddings(data, embeddings):
    if len(data) != len(embeddings):
        raise IngestionError(
            f"Embedding count mismatch: {len(embeddings)} embedding(s) for {len(data)} chunk(s)"
        )
    for record, embedding in zip(data, embeddings):
        if not isinstance(embedding, list) or not embedding:
            raise IngestionError("Received an invalid empty embedding")
        record["embedding"] = embedding
    return data


def read_positive_int(raw_value: str, default: int) -> int:
    try:
        value = int(raw_value)
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


def read_nonnegative_int(raw_value: str, default: int) -> int:
    try:
        value = int(raw_value)
    except (TypeError, ValueError):
        return default
    return value if value >= 0 else default


def main():
    load_dotenv_if_available()
    paths = runtime_paths()
    load_dotenv_if_available(paths.env_path, override=True)
    data_dir = str(paths.data_dir)
    settings = read_settings(paths.settings_path)
    chopped_csv_path = str(paths.data_dir / "chopped_text.csv")
    output_pickle_path = str(paths.data_dir / "embedded_data.pkl")

    try:
        openai_module = load_openai_module()
        openai_module.api_key = os.getenv("OPENAI_API_KEY")
        if not openai_module.api_key:
            raise IngestionError("OPENAI_API_KEY not found in environment or .env file")

        if not os.path.exists(chopped_csv_path):
            raise IngestionError(
                f"Chopped CSV file not found: {chopped_csv_path}. Run prepare_documents.py first."
            )

        data = read_chopped_csv(chopped_csv_path)
        if not data:
            raise IngestionError("No chunk rows found in chopped_text.csv")

        texts = [d["chunk_text"] for d in data]
        model_name = settings.get("openai_embedding_model", "text-embedding-ada-002")
        max_tokens_per_batch = read_positive_int(
            settings.get("max_tokens_per_batch"), 100000
        )
        rate_limit_sleep_seconds = read_nonnegative_int(
            settings.get("embedding_rate_limit_sleep_seconds"), 0
        )

        print("Generating embeddings using OpenAI model:", model_name)
        embeddings = embed_with_openai(
            texts,
            model=model_name,
            max_tokens_per_batch=max_tokens_per_batch,
            openai_module=openai_module,
            sleep_seconds=rate_limit_sleep_seconds,
        )
        data = attach_embeddings(data, embeddings)

        os.makedirs(data_dir, exist_ok=True)
        with open(output_pickle_path, "wb") as f:
            pickle.dump(data, f)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Successfully wrote embeddings to {output_pickle_path}")
    print("Sample record:", data[0])
    print("Done!")


if __name__ == "__main__":
    main()
