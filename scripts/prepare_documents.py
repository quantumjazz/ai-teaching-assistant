import csv
import glob
import os
import re
import sys
from dataclasses import dataclass
from typing import List, Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.index_status import fingerprint_document
from src.env_utils import load_dotenv_if_available
from src.settings import runtime_paths


SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}
PLACEHOLDER_DOCUMENT_NAMES = {"files.txt"}


class IngestionError(Exception):
    pass


@dataclass(frozen=True)
class TextBlock:
    text: str
    page_number: Optional[int]
    section_title: str
    start_char: int
    end_char: int


@dataclass(frozen=True)
class ChunkRecord:
    filename: str
    chunk_index: int
    chunk_text: str
    source_path: str
    source_name: str
    source_extension: str
    source_size_bytes: int
    source_modified_at: str
    source_modified_at_ns: int
    source_sha256: str
    document_title: str
    page_number: Optional[int]
    section_title: str
    chunk_start_char: int
    chunk_end_char: int
    chunk_word_count: int

    def to_csv_row(self):
        return (
            self.filename,
            self.chunk_index,
            self.chunk_text,
            self.source_path,
            self.source_name,
            self.source_extension,
            self.source_size_bytes,
            self.source_modified_at,
            self.source_modified_at_ns,
            self.source_sha256,
            self.document_title,
            "" if self.page_number is None else self.page_number,
            self.section_title,
            self.chunk_start_char,
            self.chunk_end_char,
            self.chunk_word_count,
        )


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a PDF file using PyPDF2."""
    return "\n\n".join(block.text for block in extract_blocks_from_pdf(pdf_path))


def extract_blocks_from_pdf(pdf_path: str) -> List[TextBlock]:
    try:
        import PyPDF2
    except ImportError as exc:
        raise IngestionError(
            "PyPDF2 is not installed. Run `pip install -r requirements.txt`."
        ) from exc

    blocks = []
    cursor = 0
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page_number, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text()
            page_text = normalize_text(page_text or "")
            if not page_text:
                continue
            start = cursor
            end = start + len(page_text)
            blocks.append(TextBlock(page_text, page_number, "", start, end))
            cursor = end + 2
    return blocks


def extract_text_from_docx(docx_path: str) -> str:
    """Extract text from a DOCX file using python-docx."""
    return "\n\n".join(block.text for block in extract_blocks_from_docx(docx_path))


def extract_blocks_from_docx(docx_path: str) -> List[TextBlock]:
    try:
        import docx
    except ImportError as exc:
        raise IngestionError(
            "python-docx is not installed. Run `pip install -r requirements.txt`."
        ) from exc

    doc = docx.Document(docx_path)
    blocks = []
    cursor = 0
    section_title = ""
    for para in doc.paragraphs:
        text = normalize_text(para.text)
        if not text:
            continue
        if _is_docx_heading(para):
            section_title = text
            continue
        start = cursor
        end = start + len(text)
        blocks.append(TextBlock(text, None, section_title, start, end))
        cursor = end + 2
    return blocks


def extract_text_from_txt(txt_path: str) -> str:
    """Extract text from a TXT file."""
    return "\n\n".join(block.text for block in extract_blocks_from_txt(txt_path))


def extract_blocks_from_txt(txt_path: str) -> List[TextBlock]:
    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        raw_text = f.read()
    paragraphs = split_paragraphs(raw_text)
    blocks = []
    cursor = 0
    section_title = ""
    detect_headings = len(paragraphs) > 1
    for index, paragraph in enumerate(paragraphs):
        text = normalize_text(paragraph)
        if not text:
            continue
        next_text = ""
        if index + 1 < len(paragraphs):
            next_text = normalize_text(paragraphs[index + 1])
        if detect_headings and is_probable_section_heading(text, next_text):
            section_title = text
            continue
        start = cursor
        end = start + len(text)
        blocks.append(TextBlock(text, None, section_title, start, end))
        cursor = end + 2
    return blocks


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def split_paragraphs(text: str):
    return [part.strip() for part in re.split(r"\n\s*\n+", text) if part.strip()]


def _is_docx_heading(paragraph):
    style_name = getattr(getattr(paragraph, "style", None), "name", "") or ""
    return style_name.lower().startswith("heading")


def is_probable_section_heading(text: str, next_text: str = "") -> bool:
    words = text.split()
    if len(words) > 12:
        return False
    if text.endswith(":"):
        return True
    if re.match(r"^(\d+(\.\d+)*|[IVXLCM]+)[.)]\s+\S+", text, re.IGNORECASE):
        return True
    if re.search(r"[.!?]$", text):
        return False
    if len(words) <= 8 and len(text) <= 90 and _looks_mostly_uppercase(text):
        return True
    if (
        len(words) <= 4
        and len(text) <= 60
        and _looks_like_title(text)
        and count_words(next_text) >= 8
    ):
        return True
    return False


def _looks_mostly_uppercase(text):
    letters = [char for char in text if char.isalpha()]
    if not letters:
        return False
    uppercase = sum(1 for char in letters if char.isupper())
    return uppercase / len(letters) >= 0.75


def _looks_like_title(text):
    letters = [char for char in text if char.isalpha()]
    if not letters:
        return False
    if text.upper() == text:
        return True
    words = [word.strip("0123456789.,;:()[]{}") for word in text.split()]
    words = [word for word in words if word]
    return bool(words) and all(word[:1].isupper() for word in words)


def chunk_blocks(
    blocks: List[TextBlock],
    chunk_size: int = 200,
    overlap: int = 100,
):
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if overlap < 0:
        raise ValueError("overlap cannot be negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []
    current_blocks = []
    current_words = 0
    current_page = None
    current_section = ""

    def flush():
        nonlocal current_blocks, current_words, current_page, current_section
        if not current_blocks:
            return
        chunk_body = "\n\n".join(block.text for block in current_blocks)
        chunks.append(
            {
                "text": chunk_body,
                "page_number": current_page,
                "section_title": current_section,
                "start_char": current_blocks[0].start_char,
                "end_char": current_blocks[-1].end_char,
                "word_count": count_words(chunk_body),
            }
        )
        current_blocks = []
        current_words = 0
        current_page = None
        current_section = ""

    for block in blocks:
        if not block.text:
            continue
        if count_words(block.text) > chunk_size:
            flush()
            chunks.extend(split_long_block(block, chunk_size, overlap))
            continue

        block_words = count_words(block.text)
        same_scope = (
            not current_blocks
            or (current_page == block.page_number and current_section == block.section_title)
        )
        if current_blocks and (not same_scope or current_words + block_words > chunk_size):
            flush()

        current_blocks.append(block)
        current_words += block_words
        current_page = block.page_number
        current_section = block.section_title

    flush()
    return [
        SimpleChunk(
            chunk["text"],
            chunk["page_number"],
            chunk["section_title"],
            chunk["start_char"],
            chunk["end_char"],
            chunk["word_count"],
        )
        for chunk in chunks
    ]


@dataclass(frozen=True)
class SimpleChunk:
    chunk_text: str
    page_number: Optional[int]
    section_title: str
    start_char: int
    end_char: int
    word_count: int


def split_long_block(block: TextBlock, chunk_size: int, overlap: int):
    words = block.text.split()
    step = chunk_size - overlap
    chunks = []
    start_word = 0
    while start_word < len(words):
        end_word = min(start_word + chunk_size, len(words))
        body = " ".join(words[start_word:end_word])
        start_char, end_char = word_window_offsets(block.text, start_word, end_word)
        chunks.append(
            {
                "text": body,
                "page_number": block.page_number,
                "section_title": block.section_title,
                "start_char": block.start_char + start_char,
                "end_char": block.start_char + end_char,
                "word_count": count_words(body),
            }
        )
        if end_word == len(words):
            break
        start_word += step
    return chunks


def word_window_offsets(text, start_word, end_word):
    matches = list(re.finditer(r"\S+", text))
    if not matches:
        return 0, 0
    start_index = min(start_word, len(matches) - 1)
    end_index = min(max(end_word - 1, start_index), len(matches) - 1)
    return matches[start_index].start(), matches[end_index].end()


def count_words(text):
    return len(text.split())


def gather_document_files(documents_dir: str):
    patterns = [
        os.path.join(documents_dir, "**", "*.pdf"),
        os.path.join(documents_dir, "**", "*.docx"),
        os.path.join(documents_dir, "**", "*.txt"),
    ]
    files = []
    for pattern in patterns:
        files.extend(glob.glob(pattern, recursive=True))
    return sorted(
        fpath
        for fpath in files
        if os.path.basename(fpath) not in PLACEHOLDER_DOCUMENT_NAMES
    )


def extract_text(file_path: str) -> str:
    return "\n\n".join(block.text for block in extract_blocks(file_path))


def extract_blocks(file_path: str) -> List[TextBlock]:
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return extract_blocks_from_pdf(file_path)
    if ext == ".docx":
        return extract_blocks_from_docx(file_path)
    if ext == ".txt":
        return extract_blocks_from_txt(file_path)
    raise IngestionError(f"Unsupported document type: {file_path}")


def prepare_chunks(files, chunk_size=200, overlap=100, project_root=None):
    if not files:
        raise IngestionError("No course documents found in documents/.")

    project_root = project_root or BASE_DIR
    all_chunks = []
    for file_path in files:
        print(f"Processing: {file_path}")
        blocks = extract_blocks(file_path)
        if not blocks:
            raise IngestionError(
                f"No extractable text found in {file_path}. If this is a scanned PDF, "
                "run OCR first and retry."
            )

        fingerprint = fingerprint_document(file_path, project_root)
        filename_only = os.path.basename(file_path)
        document_title = os.path.splitext(filename_only)[0]
        chunks = chunk_blocks(
            blocks,
            chunk_size=chunk_size,
            overlap=overlap,
        )
        if not chunks:
            raise IngestionError(f"No chunks produced for {file_path}")

        for i, chunk in enumerate(chunks):
            all_chunks.append(
                ChunkRecord(
                    filename_only,
                    i,
                    chunk.chunk_text,
                    fingerprint.path,
                    fingerprint.name,
                    fingerprint.extension,
                    fingerprint.size_bytes,
                    fingerprint.modified_at,
                    fingerprint.modified_at_ns,
                    fingerprint.sha256,
                    document_title,
                    chunk.page_number,
                    chunk.section_title,
                    chunk.start_char,
                    chunk.end_char,
                    chunk.word_count,
                )
            )

    if not all_chunks:
        raise IngestionError("No chunks were produced from the provided documents.")
    return all_chunks


def write_chunks_csv(chunks, output_csv_path):
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    with open(output_csv_path, "w", encoding="utf-8", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "filename",
                "chunk_index",
                "chunk_text",
                "source_path",
                "source_name",
                "source_extension",
                "source_size_bytes",
                "source_modified_at",
                "source_modified_at_ns",
                "source_sha256",
                "document_title",
                "page_number",
                "section_title",
                "chunk_start_char",
                "chunk_end_char",
                "chunk_word_count",
            ]
        )
        writer.writerows(chunk.to_csv_row() for chunk in chunks)


def main():
    load_dotenv_if_available()
    paths = runtime_paths()
    load_dotenv_if_available(paths.env_path)
    documents_dir = str(paths.documents_dir)
    output_csv_path = str(paths.data_dir / "chopped_text.csv")

    try:
        files = gather_document_files(documents_dir)
        chunks = prepare_chunks(
            files,
            chunk_size=200,
            overlap=100,
            project_root=str(paths.course_root),
        )
        write_chunks_csv(chunks, output_csv_path)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Done! Wrote {len(chunks)} chunks to {output_csv_path}")


if __name__ == "__main__":
    main()
