# AI Teaching Assistant

This is a Flask-based retrieval-augmented teaching assistant for course materials
from the Quantitative Methods research lab at New Bulgarian University.

The reliable local path is:

1. Add course PDF, DOCX, or TXT files under `documents/`.
2. Create a local `.env` file with `OPENAI_API_KEY=...` and `FLASK_SECRET_KEY=...`.
3. Run the indexing pipeline.
4. Start the Flask app from `src.app`.
5. Open the local browser URL and ask a question.

Generated course data and source documents are intentionally ignored by Git.

## Local Setup On Ubuntu LTS

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Create `.env`:

```bash
OPENAI_API_KEY=your_api_key_here
FLASK_SECRET_KEY=replace_with_a_long_random_secret
# Set this only when serving over HTTPS:
# SESSION_COOKIE_SECURE=true
```

You can also start from the tracked template:

```bash
cp .env.example .env
```

## Build The Knowledge Base

```bash
python scripts/prepare_documents.py
python scripts/embed_documents.py
python scripts/create_final_data.py
```

Or use:

```bash
make index
```

These commands create generated files under `data/`. If those files are missing,
the web app still starts and returns a clear setup-needed message.
After pulling changes that affect chunking or retrieval settings, rerun the full
indexing pipeline so generated embeddings and FAISS metadata match the code.

Answers include source metadata from the retrieved chunks, shown as filenames
and chunk indexes in the chat UI. When available, sources also include page,
section, and chunk offset metadata.

The chat keeps short browser-session context for follow-up questions. Use the
`New chat` button to clear that context. To check a student answer against the
most recent course question, start the message with `a:`. Answer-check messages
do not record a new chat turn, so repeated checks still refer to the most recent
course question.

The assistant is instructed to answer in the student's latest question language.
The built-in no-answer and answer-check fallback messages match Bulgarian or
English input without adding a separate language setting.

## Run Locally

```bash
python -m src.app
```

Or use:

```bash
make run
```

Then open `http://127.0.0.1:5000`.

Open `http://127.0.0.1:5000/setup` to see a local setup checklist. The same
diagnostics are available as JSON at `http://127.0.0.1:5000/api/health`.
Open `http://127.0.0.1:5000/documents` to see local document inventory,
generated artifact status, and whether the current FAISS index is stale.

## Run Tests

```bash
python -m unittest discover -s tests -v
```

Or use:

```bash
make test
```

To remove generated local index artifacts while keeping the tracked data
directory placeholder:

```bash
make clean-data
```

## Evaluate Retrieval Quality

Copy the example file and edit it with course-specific questions:

```bash
cp eval/rag_cases.example.json eval/rag_cases.json
```

Each case checks whether retrieval returns an expected source filename and
expected terms in the retrieved context. Run:

```bash
make evaluate
```

This uses the local FAISS index and OpenAI embeddings, so it requires `.env` and
the indexing pipeline to be complete.

The current reliability pass supports OpenAI embeddings only.

For real deployments, set a stable `FLASK_SECRET_KEY`. Without one, every app
restart uses a new ephemeral signing key and existing browser chat sessions
become unreachable. Flask session cookies are `HttpOnly` and `SameSite=Strict`
by default; set `SESSION_COOKIE_SECURE=true` only when serving the app over
HTTPS.

The included Docker image runs gunicorn with one worker because conversation
state is intentionally in-process. If you deploy multiple workers or replicas,
move conversation state to shared storage first.

The indexing pipeline is intentionally strict: missing documents, empty
extractions, malformed chunk data, invalid embeddings, or inconsistent embedding
dimensions stop the command with a nonzero exit code. Successful indexing writes
`data/index_report.json` alongside the FAISS index and metadata. That report
includes input document fingerprints so the app can detect when local course
documents have changed and the index should be rebuilt.

Chunking is paragraph-aware instead of a fixed sliding word window. Chunk bodies
are embedded without display metadata; PDF page numbers, section titles, and
character offsets are stored as structured metadata and added to prompts only at
retrieval time. Very long blocks are split with overlap.

Retrieval is hybrid by default:

- FAISS vector search retrieves semantic candidates.
- Local BM25 lexical search retrieves exact-term candidates.
- Reciprocal rank fusion merges both lists.
- When `hybrid_retrieval=true` and `lexical_rerank=true`, lexical overlap is
  applied as a secondary rerank after fusion; the hybrid RRF score remains the
  primary rank.
- Diversity filtering avoids overloading the context with repeated chunks from
  the same source document.
- A low-confidence retrieval gate returns a clear "I don't know" response rather
  than sending weak context to the chat model.

Relevant settings:

```text
hybrid_retrieval=true
lexical_rerank=true
rrf_k=60
max_chunks_per_source=2
embedding_rate_limit_sleep_seconds=0
minimum_hybrid_retrieval_confidence=0.025
minimum_lexical_retrieval_confidence=0.20
minimum_vector_retrieval_confidence=0.20
```

Retrieval evaluation cases may also include:

```json
{
  "expected_no_answer": true,
  "max_duplicate_sources": 1
}
```
