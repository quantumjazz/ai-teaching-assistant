import json
import os
import pickle
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.clients import load_faiss_module, load_numpy_module
from src.index_status import validate_index_report
from src.settings import read_settings


class IngestionError(Exception):
    pass


SOURCE_FINGERPRINT_FIELDS = (
    "source_path",
    "source_name",
    "source_extension",
    "source_size_bytes",
    "source_modified_at",
    "source_modified_at_ns",
    "source_sha256",
)


def validate_embedded_data(embedded_data: List[Dict[str, Any]]) -> int:
    if not isinstance(embedded_data, list):
        raise IngestionError("Embedded data must be a list")
    if not embedded_data:
        raise IngestionError("No embedded data found")

    embedding_dim = None
    for index, record in enumerate(embedded_data):
        if not isinstance(record, dict):
            raise IngestionError(f"Record {index} is not an object")
        for key in ("filename", "chunk_index", "chunk_text", "embedding"):
            if key not in record:
                raise IngestionError(f"Record {index} is missing {key}")
        if not str(record["filename"]).strip():
            raise IngestionError(f"Record {index} has empty filename")
        if not str(record["chunk_text"]).strip():
            raise IngestionError(f"Record {index} has empty chunk_text")
        try:
            int(record["chunk_index"])
        except (TypeError, ValueError) as exc:
            raise IngestionError(f"Record {index} has invalid chunk_index") from exc

        embedding = record["embedding"]
        if not isinstance(embedding, list) or not embedding:
            raise IngestionError(f"Record {index} has an invalid embedding")
        if not all(isinstance(value, (int, float)) for value in embedding):
            raise IngestionError(f"Record {index} embedding contains non-numeric values")
        if embedding_dim is None:
            embedding_dim = len(embedding)
        elif len(embedding) != embedding_dim:
            raise IngestionError(
                f"Record {index} embedding dimension {len(embedding)} does not match {embedding_dim}"
            )
        validate_optional_source_fingerprint(record, index)
        validate_optional_chunk_metadata(record, index)

    return embedding_dim


def validate_optional_source_fingerprint(record, index):
    source_fields = set(SOURCE_FINGERPRINT_FIELDS)
    present = source_fields.intersection(record)
    if not present:
        return
    missing = source_fields.difference(record)
    if missing:
        raise IngestionError(
            f"Record {index} has incomplete source fingerprint fields: {', '.join(sorted(missing))}"
        )
    if not str(record["source_path"]).strip():
        raise IngestionError(f"Record {index} has empty source_path")
    if not str(record["source_sha256"]).strip():
        raise IngestionError(f"Record {index} has empty source_sha256")
    if not isinstance(record["source_size_bytes"], int) or record["source_size_bytes"] < 0:
        raise IngestionError(f"Record {index} has invalid source_size_bytes")
    if not isinstance(record["source_modified_at_ns"], int) or record["source_modified_at_ns"] <= 0:
        raise IngestionError(f"Record {index} has invalid source_modified_at_ns")


def validate_optional_chunk_metadata(record, index):
    metadata_fields = {
        "document_title",
        "page_number",
        "section_title",
        "chunk_start_char",
        "chunk_end_char",
        "chunk_word_count",
    }
    present = metadata_fields.intersection(record)
    if not present:
        return
    missing = metadata_fields.difference(record)
    if missing:
        raise IngestionError(
            f"Record {index} has incomplete chunk metadata fields: {', '.join(sorted(missing))}"
        )
    page_number = record["page_number"]
    if page_number is not None and (
        not isinstance(page_number, int) or page_number <= 0
    ):
        raise IngestionError(f"Record {index} has invalid page_number")
    if not isinstance(record["chunk_start_char"], int) or record["chunk_start_char"] < 0:
        raise IngestionError(f"Record {index} has invalid chunk_start_char")
    if (
        not isinstance(record["chunk_end_char"], int)
        or record["chunk_end_char"] < record["chunk_start_char"]
    ):
        raise IngestionError(f"Record {index} has invalid chunk_end_char")
    if not isinstance(record["chunk_word_count"], int) or record["chunk_word_count"] <= 0:
        raise IngestionError(f"Record {index} has invalid chunk_word_count")


def build_faiss_index(
    embedded_data: List[Dict[str, Any]],
    embedding_dim: int,
    faiss_module=None,
    np_module=None,
) -> Tuple[Any, List[Dict[str, Any]]]:
    """
    Builds a FAISS index (using L2 distance) from the embedded data.
    Returns:
      - A FAISS index containing all embeddings.
      - A metadata list with each record's 'filename', 'chunk_index', and 'chunk_text'.
    """
    faiss_module = load_faiss_module(faiss_module)
    np = load_numpy_module(np_module)

    vectors = []
    metadata = []
    for record in embedded_data:
        embedding = np.array(record["embedding"], dtype=np.float32)
        vectors.append(embedding)
        metadata_record = {
            "filename": record["filename"],
            "chunk_index": int(record["chunk_index"]),
            "chunk_text": record["chunk_text"],
        }
        for key in (
            "source_path",
            "source_name",
            "source_extension",
            "source_size_bytes",
            "source_modified_at",
            "source_modified_at_ns",
            "source_sha256",
        ):
            if key in record:
                metadata_record[key] = record[key]
        for key in (
            "document_title",
            "page_number",
            "section_title",
            "chunk_start_char",
            "chunk_end_char",
            "chunk_word_count",
        ):
            if key in record:
                metadata_record[key] = record[key]
        metadata.append(metadata_record)
    if not vectors:
        raise IngestionError("No vectors available for FAISS index")
    vectors_np = np.vstack(vectors)
    index = faiss_module.IndexFlatL2(embedding_dim)
    index.add(vectors_np)
    return index, metadata


def create_index_report(
    embedded_data,
    metadata_list,
    faiss_index,
    embedding_model,
    artifact_paths,
    input_documents,
):
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_document_count": len(input_documents),
        "chunk_count": len(metadata_list),
        "embedding_count": len(embedded_data),
        "faiss_vector_count": int(getattr(faiss_index, "ntotal", len(metadata_list))),
        "embedding_model": embedding_model,
        "input_documents": input_documents,
        "artifacts": artifact_paths,
    }
    validate_index_report(report)
    return report


def input_documents_from_embedded_data(embedded_data):
    documents = {}
    for index, record in enumerate(embedded_data):
        missing = [
            field for field in SOURCE_FINGERPRINT_FIELDS if field not in record
        ]
        if missing:
            raise IngestionError(
                f"Embedded record {index} is missing source fingerprint fields: "
                + ", ".join(missing)
                + ". Rerun scripts/prepare_documents.py and scripts/embed_documents.py."
            )
        path = record["source_path"]
        documents[path] = {
            "path": path,
            "name": record["source_name"],
            "extension": record["source_extension"],
            "size_bytes": record["source_size_bytes"],
            "modified_at": record["source_modified_at"],
            "modified_at_ns": record["source_modified_at_ns"],
            "sha256": record["source_sha256"],
        }
    return [documents[path] for path in sorted(documents)]


def write_index_report(report_path, report):
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


def main():
    base_dir = BASE_DIR
    data_dir = os.path.join(base_dir, "data")
    settings = read_settings(os.path.join(base_dir, "settings.txt"))
    embedded_data_path = os.path.join(data_dir, "embedded_data.pkl")
    faiss_index_path = os.path.join(data_dir, "faiss_index.bin")
    metadata_path = os.path.join(data_dir, "faiss_metadata.json")
    report_path = os.path.join(data_dir, "index_report.json")

    try:
        if not os.path.exists(embedded_data_path):
            raise IngestionError(
                f"Embedded data not found at {embedded_data_path}. Run embed_documents.py first."
            )

        with open(embedded_data_path, "rb") as f:
            embedded_data = pickle.load(f)

        embedding_dim = validate_embedded_data(embedded_data)
        print("Detected embedding dimension:", embedding_dim)

        input_documents = input_documents_from_embedded_data(embedded_data)
        if not input_documents:
            raise IngestionError("No supported input document fingerprints available")

        faiss_index, metadata_list = build_faiss_index(embedded_data, embedding_dim)
        print(f"FAISS index built with {len(metadata_list)} vectors.")

        os.makedirs(data_dir, exist_ok=True)
        faiss_module = load_faiss_module()
        faiss_module.write_index(faiss_index, faiss_index_path)
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata_list, f, ensure_ascii=False, indent=2)

        report = create_index_report(
            embedded_data=embedded_data,
            metadata_list=metadata_list,
            faiss_index=faiss_index,
            embedding_model=settings.get("openai_embedding_model", "text-embedding-ada-002"),
            artifact_paths={
                "embedded_data": "data/embedded_data.pkl",
                "faiss_index": "data/faiss_index.bin",
                "faiss_metadata": "data/faiss_metadata.json",
                "index_report": "data/index_report.json",
            },
            input_documents=input_documents,
        )
        write_index_report(report_path, report)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    print("FAISS index, metadata, and index report saved successfully!")


if __name__ == "__main__":
    main()
