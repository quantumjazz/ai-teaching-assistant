import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .settings import PROJECT_ROOT


SUPPORTED_DOCUMENT_EXTENSIONS = {".pdf", ".docx", ".txt"}
PLACEHOLDER_DOCUMENT_NAMES = {"files.txt"}
ARTIFACTS = (
    ("chopped_text", "data/chopped_text.csv"),
    ("embedded_data", "data/embedded_data.pkl"),
    ("faiss_index", "data/faiss_index.bin"),
    ("faiss_metadata", "data/faiss_metadata.json"),
    ("index_report", "data/index_report.json"),
)


@dataclass(frozen=True)
class DocumentRecord:
    path: str
    name: str
    extension: str
    supported: bool
    size_bytes: int
    modified_at: str
    modified_at_ns: int
    sha256: str

    def to_dict(self):
        return {
            "path": self.path,
            "name": self.name,
            "extension": self.extension,
            "supported": self.supported,
            "size_bytes": self.size_bytes,
            "modified_at": self.modified_at,
            "modified_at_ns": self.modified_at_ns,
            "sha256": self.sha256,
            "sha256_short": self.sha256[:12],
        }

    def to_report_dict(self):
        return {
            "path": self.path,
            "name": self.name,
            "extension": self.extension,
            "size_bytes": self.size_bytes,
            "modified_at": self.modified_at,
            "modified_at_ns": self.modified_at_ns,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class ArtifactRecord:
    name: str
    path: str
    exists: bool
    size_bytes: int = 0
    modified_at: str = ""
    modified_at_ns: int = 0

    def to_dict(self):
        return {
            "name": self.name,
            "path": self.path,
            "exists": self.exists,
            "size_bytes": self.size_bytes,
            "modified_at": self.modified_at,
            "modified_at_ns": self.modified_at_ns,
        }


@dataclass(frozen=True)
class IndexStatus:
    status: str
    message: str
    documents: List[DocumentRecord]
    artifacts: List[ArtifactRecord]
    report: Optional[Dict]
    stale_reasons: List[str]

    def to_dict(self):
        return {
            "status": self.status,
            "message": self.message,
            "documents": [document.to_dict() for document in self.documents],
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            "report": self.report,
            "stale_reasons": list(self.stale_reasons),
            "supported_document_count": len(supported_documents(self.documents)),
            "unsupported_document_count": len(unsupported_documents(self.documents)),
        }


def get_index_status(project_root=PROJECT_ROOT):
    project_root = Path(project_root)
    documents = list_document_inventory(project_root)
    artifacts = list_artifact_status(project_root)
    supported = supported_documents(documents)
    missing_artifacts = [
        artifact.path for artifact in artifacts if artifact.name != "index_report" and not artifact.exists
    ]

    report = None
    stale_reasons = []
    try:
        report = load_index_report(project_root)
        if report is not None:
            validate_index_report(report)
    except Exception as exc:
        return IndexStatus(
            "error",
            f"data/index_report.json is invalid: {exc}",
            documents,
            artifacts,
            None,
            [],
        )

    if not supported:
        return IndexStatus(
            "missing",
            "No supported course documents found in documents/.",
            documents,
            artifacts,
            report,
            [],
        )

    if missing_artifacts:
        return IndexStatus(
            "missing",
            "Missing generated artifact(s): " + ", ".join(missing_artifacts),
            documents,
            artifacts,
            report,
            [],
        )

    if report is None:
        return IndexStatus(
            "missing",
            "data/index_report.json is missing; rerun the indexing pipeline.",
            documents,
            artifacts,
            None,
            [],
        )

    stale_reasons = compare_reported_documents(supported, report["input_documents"])
    if stale_reasons:
        return IndexStatus(
            "stale",
            "The current documents differ from the indexed document fingerprints.",
            documents,
            artifacts,
            report,
            stale_reasons,
        )

    return IndexStatus(
        "ready",
        "The local index matches the current supported documents.",
        documents,
        artifacts,
        report,
        [],
    )


def list_document_inventory(project_root=PROJECT_ROOT):
    project_root = Path(project_root)
    documents_dir = project_root / "documents"
    if not documents_dir.exists():
        return []

    records = []
    for path in sorted(documents_dir.rglob("*")):
        if not path.is_file() or path.name in PLACEHOLDER_DOCUMENT_NAMES:
            continue
        records.append(fingerprint_document(path, project_root))
    return records


def list_supported_document_records(project_root=PROJECT_ROOT):
    return supported_documents(list_document_inventory(project_root))


def supported_documents(documents: Iterable[DocumentRecord]):
    return [document for document in documents if document.supported]


def unsupported_documents(documents: Iterable[DocumentRecord]):
    return [document for document in documents if not document.supported]


def fingerprint_document(file_path, project_root=PROJECT_ROOT):
    file_path = Path(file_path)
    project_root = Path(project_root)
    stat = file_path.stat()
    extension = file_path.suffix.lower()
    return DocumentRecord(
        path=_relative_posix(file_path, project_root),
        name=file_path.name,
        extension=extension,
        supported=extension in SUPPORTED_DOCUMENT_EXTENSIONS,
        size_bytes=stat.st_size,
        modified_at=_iso_from_ns(stat.st_mtime_ns),
        modified_at_ns=stat.st_mtime_ns,
        sha256=sha256_file(file_path),
    )


def document_records_for_report(project_root=PROJECT_ROOT):
    return [record.to_report_dict() for record in list_supported_document_records(project_root)]


def list_artifact_status(project_root=PROJECT_ROOT):
    project_root = Path(project_root)
    records = []
    for name, relative_path in ARTIFACTS:
        path = project_root / relative_path
        if not path.exists():
            records.append(ArtifactRecord(name=name, path=relative_path, exists=False))
            continue
        stat = path.stat()
        records.append(
            ArtifactRecord(
                name=name,
                path=relative_path,
                exists=True,
                size_bytes=stat.st_size,
                modified_at=_iso_from_ns(stat.st_mtime_ns),
                modified_at_ns=stat.st_mtime_ns,
            )
        )
    return records


def load_index_report(project_root=PROJECT_ROOT):
    report_path = Path(project_root) / "data" / "index_report.json"
    if not report_path.exists():
        return None
    with open(report_path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_index_report(report):
    if not isinstance(report, dict):
        raise ValueError("expected an object")
    required_keys = {
        "created_at",
        "source_document_count",
        "chunk_count",
        "embedding_count",
        "faiss_vector_count",
        "embedding_model",
        "artifacts",
        "input_documents",
    }
    missing = required_keys.difference(report)
    if missing:
        raise ValueError(f"missing required keys: {', '.join(sorted(missing))}")
    if not isinstance(report["input_documents"], list):
        raise ValueError("input_documents must be a list")
    if report["source_document_count"] != len(report["input_documents"]):
        raise ValueError("source_document_count does not match input_documents")
    for index, document in enumerate(report["input_documents"]):
        validate_report_document(document, index)


def validate_report_document(document, index):
    if not isinstance(document, dict):
        raise ValueError(f"input_documents[{index}] is not an object")
    required_keys = {
        "path",
        "name",
        "extension",
        "size_bytes",
        "modified_at",
        "modified_at_ns",
        "sha256",
    }
    missing = required_keys.difference(document)
    if missing:
        raise ValueError(
            f"input_documents[{index}] missing required keys: {', '.join(sorted(missing))}"
        )
    if not str(document["path"]).strip():
        raise ValueError(f"input_documents[{index}] has empty path")
    if not str(document["sha256"]).strip():
        raise ValueError(f"input_documents[{index}] has empty sha256")
    if not isinstance(document["size_bytes"], int) or document["size_bytes"] < 0:
        raise ValueError(f"input_documents[{index}] has invalid size_bytes")
    if not isinstance(document["modified_at_ns"], int) or document["modified_at_ns"] <= 0:
        raise ValueError(f"input_documents[{index}] has invalid modified_at_ns")


def compare_reported_documents(current_documents, reported_documents):
    current_by_path = {document.path: document for document in current_documents}
    reported_by_path = {document["path"]: document for document in reported_documents}

    reasons = []
    added = sorted(set(current_by_path) - set(reported_by_path))
    removed = sorted(set(reported_by_path) - set(current_by_path))
    changed = []

    for path in sorted(set(current_by_path) & set(reported_by_path)):
        current = current_by_path[path]
        reported = reported_by_path[path]
        if (
            current.sha256 != reported.get("sha256")
            or current.size_bytes != reported.get("size_bytes")
        ):
            changed.append(path)

    if added:
        reasons.append("New supported document(s): " + ", ".join(added))
    if removed:
        reasons.append("Indexed document(s) no longer present: " + ", ".join(removed))
    if changed:
        reasons.append("Changed supported document(s): " + ", ".join(changed))
    return reasons


def sha256_file(file_path):
    digest = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_posix(path, project_root):
    path = Path(path)
    project_root = Path(project_root)
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return path.name


def _iso_from_ns(timestamp_ns):
    return datetime.fromtimestamp(timestamp_ns / 1_000_000_000, timezone.utc).isoformat()
