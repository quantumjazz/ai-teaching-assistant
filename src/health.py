import csv
import json
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

from .env_utils import (
    is_configured_env_value,
    is_placeholder_env_value,
    read_env_file,
)
from .index_status import get_index_status, validate_index_report
from .settings import (
    DATA_DIR,
    DOCUMENTS_DIR,
    ENV_PATH,
    PROJECT_ROOT,
    SETTINGS_PATH,
    load_course_settings,
    runtime_paths,
)
SUPPORTED_DOCUMENT_EXTENSIONS = {".pdf", ".docx", ".txt"}
PLACEHOLDER_DOCUMENT_NAMES = {"files.txt"}


@dataclass(frozen=True)
class HealthCheck:
    name: str
    status: str
    message: str

    def to_dict(self):
        return {
            "name": self.name,
            "status": self.status,
            "message": self.message,
        }


@dataclass(frozen=True)
class HealthReport:
    status: str
    checks: List[HealthCheck]

    def to_dict(self):
        return {
            "status": self.status,
            "checks": [check.to_dict() for check in self.checks],
        }


def get_health_report(project_root=None, env=None):
    env = os.environ if env is None else env
    paths = runtime_paths(project_root, env=env)

    metadata_records = None
    checks = [
        check_settings(paths.settings_path),
        check_openai_key(paths.env_path, env),
        check_flask_secret(paths.env_path, env),
        check_documents(paths.documents_dir),
        check_chopped_csv(paths.data_dir / "chopped_text.csv"),
        check_embedded_data(paths.data_dir / "embedded_data.pkl"),
        check_file_present("faiss_index", paths.data_dir / "faiss_index.bin"),
    ]

    metadata_check, metadata_records = check_metadata(paths.data_dir / "faiss_metadata.json")
    checks.append(metadata_check)
    checks.append(check_metadata_count(metadata_records))
    checks.append(check_index_report(paths.data_dir / "index_report.json"))
    checks.append(check_index_freshness(paths.course_root, env=env))

    return HealthReport(status=_overall_status(checks), checks=checks)


def check_settings(settings_path=SETTINGS_PATH):
    settings_path = Path(settings_path)
    if not settings_path.exists():
        return HealthCheck("settings", "incomplete", "settings.txt is missing.")
    try:
        settings = load_course_settings(settings_path)
    except Exception as exc:
        return HealthCheck("settings", "error", f"settings.txt is invalid: {exc}")
    return HealthCheck(
        "settings",
        "ready",
        f"Loaded settings for {settings.classname or 'unnamed course'}.",
    )


def check_openai_key(env_path=ENV_PATH, env=None):
    env = os.environ if env is None else env
    env_path = Path(env_path)
    env_value = env.get("OPENAI_API_KEY")
    if is_configured_env_value(env_value):
        return HealthCheck("openai_key", "ready", "OPENAI_API_KEY is set.")
    if is_placeholder_env_value(env_value):
        return HealthCheck(
            "openai_key",
            "incomplete",
            "OPENAI_API_KEY is still set to a placeholder value.",
        )

    env_values = read_env_file(env_path)
    env_file_value = env_values.get("OPENAI_API_KEY")
    if is_configured_env_value(env_file_value):
        return HealthCheck("openai_key", "ready", ".env contains OPENAI_API_KEY.")
    if is_placeholder_env_value(env_file_value):
        return HealthCheck(
            "openai_key",
            "incomplete",
            ".env contains a placeholder OPENAI_API_KEY.",
        )

    if env_path.exists():
        return HealthCheck("openai_key", "incomplete", ".env does not contain OPENAI_API_KEY.")
    return HealthCheck(
        "openai_key",
        "incomplete",
        "OPENAI_API_KEY is not set and .env is missing.",
    )


def check_flask_secret(env_path=ENV_PATH, env=None):
    env = os.environ if env is None else env
    env_path = Path(env_path)
    env_value = env.get("FLASK_SECRET_KEY")
    if is_configured_env_value(env_value):
        return HealthCheck("flask_secret", "ready", "FLASK_SECRET_KEY is set.")
    if is_placeholder_env_value(env_value):
        return HealthCheck(
            "flask_secret",
            "incomplete",
            "FLASK_SECRET_KEY is still set to a placeholder value.",
        )

    env_values = read_env_file(env_path)
    env_file_value = env_values.get("FLASK_SECRET_KEY")
    if is_configured_env_value(env_file_value):
        return HealthCheck("flask_secret", "ready", ".env contains FLASK_SECRET_KEY.")
    if is_placeholder_env_value(env_file_value):
        return HealthCheck(
            "flask_secret",
            "incomplete",
            ".env contains a placeholder FLASK_SECRET_KEY.",
        )

    if env_path.exists():
        return HealthCheck(
            "flask_secret",
            "incomplete",
            ".env does not contain FLASK_SECRET_KEY; sessions will use an ephemeral key.",
        )
    return HealthCheck(
        "flask_secret",
        "incomplete",
        "FLASK_SECRET_KEY is not set and .env is missing; sessions will use an ephemeral key.",
    )


def check_documents(documents_dir=DOCUMENTS_DIR):
    files = list_document_files(documents_dir)
    if not files:
        return HealthCheck(
            "documents",
            "incomplete",
            "No course PDF, DOCX, or TXT files found in documents/.",
        )
    return HealthCheck("documents", "ready", f"Found {len(files)} course document(s).")


def list_document_files(documents_dir=DOCUMENTS_DIR):
    documents_dir = Path(documents_dir)
    if not documents_dir.exists():
        return []

    files = []
    for path in documents_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.name in PLACEHOLDER_DOCUMENT_NAMES:
            continue
        if path.suffix.lower() in SUPPORTED_DOCUMENT_EXTENSIONS:
            files.append(path)
    return sorted(files)


def check_chopped_csv(csv_path):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return HealthCheck("chopped_text", "incomplete", "data/chopped_text.csv is missing.")
    try:
        row_count = count_valid_csv_rows(csv_path)
    except Exception as exc:
        return HealthCheck("chopped_text", "error", f"data/chopped_text.csv is invalid: {exc}")
    if row_count == 0:
        return HealthCheck("chopped_text", "incomplete", "data/chopped_text.csv has no chunks.")
    return HealthCheck("chopped_text", "ready", f"Found {row_count} chunk row(s).")


def count_valid_csv_rows(csv_path):
    required = {"filename", "chunk_index", "chunk_text"}
    count = 0
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
            raise ValueError("missing required columns filename, chunk_index, chunk_text")
        for row_number, row in enumerate(reader, start=2):
            if not (row.get("filename") or "").strip():
                raise ValueError(f"row {row_number} has an empty filename")
            if not (row.get("chunk_text") or "").strip():
                raise ValueError(f"row {row_number} has empty chunk text")
            try:
                int(row.get("chunk_index", ""))
            except ValueError as exc:
                raise ValueError(f"row {row_number} has an invalid chunk_index") from exc
            count += 1
    return count


def check_embedded_data(pickle_path):
    pickle_path = Path(pickle_path)
    if not pickle_path.exists():
        return HealthCheck("embedded_data", "incomplete", "data/embedded_data.pkl is missing.")
    try:
        with open(pickle_path, "rb") as f:
            data = pickle.load(f)
        count = validate_embedded_records(data)
    except Exception as exc:
        return HealthCheck("embedded_data", "error", f"data/embedded_data.pkl is invalid: {exc}")
    return HealthCheck("embedded_data", "ready", f"Found {count} embedded chunk(s).")


def validate_embedded_records(data):
    if not isinstance(data, list):
        raise ValueError("expected a list of embedded records")
    if not data:
        raise ValueError("embedded record list is empty")
    for index, record in enumerate(data):
        if not isinstance(record, dict):
            raise ValueError(f"record {index} is not an object")
        required_keys = (
            "filename",
            "chunk_index",
            "chunk_text",
            "embedding",
            "source_path",
            "source_name",
            "source_extension",
            "source_size_bytes",
            "source_modified_at",
            "source_modified_at_ns",
            "source_sha256",
        )
        for key in required_keys:
            if key not in record:
                raise ValueError(f"record {index} is missing {key}")
        if not record["chunk_text"]:
            raise ValueError(f"record {index} has empty chunk_text")
        if not isinstance(record["embedding"], list) or not record["embedding"]:
            raise ValueError(f"record {index} has an invalid embedding")
        validate_source_fingerprint_fields(record, index)
        validate_optional_metadata_fields(record, index)
    return len(data)


def validate_source_fingerprint_fields(record, index):
    for key in (
        "source_path",
        "source_name",
        "source_extension",
        "source_modified_at",
        "source_sha256",
    ):
        if not str(record[key]).strip():
            raise ValueError(f"record {index} has empty {key}")
    if not isinstance(record["source_size_bytes"], int) or record["source_size_bytes"] < 0:
        raise ValueError(f"record {index} has invalid source_size_bytes")
    if (
        not isinstance(record["source_modified_at_ns"], int)
        or record["source_modified_at_ns"] <= 0
    ):
        raise ValueError(f"record {index} has invalid source_modified_at_ns")


def check_file_present(name, file_path):
    file_path = Path(file_path)
    if not file_path.exists():
        return HealthCheck(name, "incomplete", f"{_relative_path(file_path)} is missing.")
    if file_path.stat().st_size == 0:
        return HealthCheck(name, "error", f"{_relative_path(file_path)} is empty.")
    return HealthCheck(name, "ready", f"{_relative_path(file_path)} exists.")


def check_metadata(metadata_path):
    metadata_path = Path(metadata_path)
    if not metadata_path.exists():
        return (
            HealthCheck("faiss_metadata", "incomplete", "data/faiss_metadata.json is missing."),
            None,
        )
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            records = json.load(f)
        validate_metadata_records(records)
    except Exception as exc:
        return (
            HealthCheck("faiss_metadata", "error", f"data/faiss_metadata.json is invalid: {exc}"),
            None,
        )
    return (
        HealthCheck("faiss_metadata", "ready", "data/faiss_metadata.json is valid."),
        records,
    )


def validate_metadata_records(records):
    if not isinstance(records, list):
        raise ValueError("expected a list of metadata records")
    if not records:
        raise ValueError("metadata record list is empty")
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"record {index} is not an object")
        for key in ("filename", "chunk_index", "chunk_text"):
            if key not in record:
                raise ValueError(f"record {index} is missing {key}")
        if not str(record["filename"]).strip():
            raise ValueError(f"record {index} has empty filename")
        if not str(record["chunk_text"]).strip():
            raise ValueError(f"record {index} has empty chunk_text")
        try:
            int(record["chunk_index"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"record {index} has invalid chunk_index") from exc
        validate_optional_metadata_fields(record, index)


def validate_optional_metadata_fields(record, index):
    integer_fields = {
        "page_number": 1,
        "source_size_bytes": 0,
        "source_modified_at_ns": 1,
        "chunk_start_char": 0,
        "chunk_end_char": 0,
        "chunk_word_count": 1,
    }
    for field, minimum in integer_fields.items():
        if field not in record or record[field] in ("", None):
            continue
        try:
            value = int(record[field])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"record {index} has invalid {field}") from exc
        if value < minimum:
            raise ValueError(f"record {index} has invalid {field}")
    if (
        "chunk_start_char" in record
        and "chunk_end_char" in record
        and record.get("chunk_start_char") not in ("", None)
        and record.get("chunk_end_char") not in ("", None)
        and int(record["chunk_end_char"]) < int(record["chunk_start_char"])
    ):
        raise ValueError(f"record {index} has invalid chunk character range")


def check_metadata_count(records):
    if records is None:
        return HealthCheck("metadata_count", "incomplete", "No valid metadata records available.")
    return HealthCheck("metadata_count", "ready", f"Found {len(records)} metadata record(s).")


def check_index_report(report_path):
    report_path = Path(report_path)
    if not report_path.exists():
        return HealthCheck(
            "index_report",
            "incomplete",
            "data/index_report.json is missing; rerun the indexing pipeline.",
        )
    try:
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)
        validate_index_report(report)
    except Exception as exc:
        return HealthCheck("index_report", "error", f"data/index_report.json is invalid: {exc}")
    return HealthCheck("index_report", "ready", "data/index_report.json is valid.")


def check_index_freshness(project_root=None, env=None):
    index_status = get_index_status(project_root, env=env)
    if index_status.status == "ready":
        return HealthCheck("index_freshness", "ready", index_status.message)
    if index_status.status == "error":
        return HealthCheck("index_freshness", "error", index_status.message)
    if index_status.status == "stale":
        return HealthCheck(
            "index_freshness",
            "incomplete",
            index_status.message + " " + " ".join(index_status.stale_reasons),
        )
    return HealthCheck("index_freshness", "incomplete", index_status.message)


def _overall_status(checks: Iterable[HealthCheck]):
    statuses = [check.status for check in checks]
    if "error" in statuses:
        return "error"
    if "incomplete" in statuses:
        return "incomplete"
    return "ready"


def _relative_path(path):
    try:
        return str(Path(path).resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path)
