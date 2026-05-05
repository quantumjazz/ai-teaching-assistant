import unittest
from unittest.mock import Mock, patch

from src.main import KnowledgeBaseNotReady
from src.health import HealthCheck, HealthReport
from src.retrieval import Source
from src.sessions import ConversationState, conversation_store
from src.service import AssistantResponse

try:
    from src.app import app
except ModuleNotFoundError as exc:
    if exc.name != "flask":
        raise
    app = None


@unittest.skipIf(app is None, "Flask is not installed")
class AppTests(unittest.TestCase):
    def setUp(self):
        app.config.update(TESTING=True)
        self.client = app.test_client()
        conversation_store._states.clear()

    def test_chat_requires_query(self):
        response = self.client.post("/api/chat", json={})

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json()["error"], "No query provided")

    def test_session_cookie_hardening_defaults(self):
        self.assertTrue(app.config["SESSION_COOKIE_HTTPONLY"])
        self.assertEqual(app.config["SESSION_COOKIE_SAMESITE"], "Strict")
        self.assertFalse(app.config["SESSION_COOKIE_SECURE"])

    def test_index_links_to_setup(self):
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Setup", response.data)
        self.assertIn(b"/setup", response.data)
        self.assertIn(b"Documents", response.data)
        self.assertIn(b"/documents", response.data)
        self.assertIn(b"styles.css?v=", response.data)
        self.assertIn(b"queryProcess.js?v=", response.data)

    def test_chat_returns_answer_from_service(self):
        result = AssistantResponse(
            "Hello from service.",
            [Source(filename="lecture.pdf", chunk_index=2, distance=0.25)],
        )
        with patch("src.app.answer_query_with_sources", return_value=result) as answer:
            response = self.client.post("/api/chat", json={"query": "Hello"})

        payload = response.get_json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["response"], "Hello from service.")
        self.assertEqual(payload["turn_count"], 0)
        self.assertIsInstance(answer.call_args.kwargs["conversation_state"], ConversationState)
        self.assertEqual(payload["sources"][0]["filename"], "lecture.pdf")
        self.assertEqual(payload["sources"][0]["chunk_index"], 2)
        self.assertEqual(payload["sources"][0]["distance"], 0.25)
        self.assertIn("hybrid_score", payload["sources"][0])

    def test_chat_maps_missing_knowledge_base_to_503(self):
        error = KnowledgeBaseNotReady("Course knowledge base is not ready.")
        with patch("src.app.answer_query_with_sources", side_effect=error):
            response = self.client.post("/api/chat", json={"query": "Hello"})

        payload = response.get_json()
        self.assertEqual(response.status_code, 503)
        self.assertEqual(payload["code"], "knowledge_base_not_ready")
        self.assertIn("Course knowledge base is not ready", payload["error"])

    def test_chat_hides_unexpected_error_details(self):
        with patch("src.app.answer_query_with_sources", side_effect=RuntimeError("secret path")):
            response = self.client.post("/api/chat", json={"query": "Hello"})

        payload = response.get_json()
        self.assertEqual(response.status_code, 500)
        self.assertEqual(payload["error"], "Unexpected error processing request.")
        self.assertNotIn("secret path", payload["error"])

    def test_clear_chat_session_removes_state(self):
        with self.client.session_transaction() as cookie_session:
            cookie_session["chat_session_id"] = "session-a"
        conversation_store.get("session-a").add_turn("q", "a", "ctx", [])

        response = self.client.delete("/api/chat/session")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["turn_count"], 0)
        self.assertEqual(conversation_store.get("session-a").turns, [])

    def test_clear_chat_session_after_real_turn_behaves_like_fresh_session(self):
        def answer(query, conversation_state):
            conversation_state.add_turn(query, "answer", "context", [])
            return AssistantResponse("answer", [])

        with patch("src.app.answer_query_with_sources", side_effect=answer):
            first = self.client.post("/api/chat", json={"query": "Hello"})

        cleared = self.client.delete("/api/chat/session")

        with patch("src.app.answer_query_with_sources", side_effect=answer):
            second = self.client.post("/api/chat", json={"query": "Hello again"})

        self.assertEqual(first.get_json()["turn_count"], 1)
        self.assertEqual(cleared.get_json()["turn_count"], 0)
        self.assertEqual(second.get_json()["turn_count"], 1)

    def test_api_health_returns_report(self):
        report = HealthReport(
            "ready",
            [HealthCheck("settings", "ready", "Loaded settings.")],
        )
        with patch("src.app.get_health_report", return_value=report):
            response = self.client.get("/api/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["status"], "ready")
        self.assertEqual(response.get_json()["checks"][0]["name"], "settings")

    def test_api_documents_returns_index_status(self):
        index_status = Mock(
            to_dict=Mock(
                return_value={
                    "status": "stale",
                    "message": "Documents changed.",
                    "documents": [],
                    "artifacts": [],
                    "report": None,
                    "stale_reasons": ["Changed supported document(s): documents/course.txt"],
                }
            )
        )
        with patch("src.app.get_index_status", return_value=index_status):
            response = self.client.get("/api/documents")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["status"], "stale")

    def test_documents_page_renders_index_status(self):
        index_status = Mock(
            to_dict=Mock(
                return_value={
                    "status": "ready",
                    "message": "The local index matches the current supported documents.",
                    "documents": [
                        {
                            "path": "documents/course.txt",
                            "supported": True,
                            "size_bytes": 12,
                            "modified_at": "2026-04-29T00:00:00+00:00",
                            "sha256_short": "abc123",
                        }
                    ],
                    "artifacts": [],
                    "report": None,
                    "stale_reasons": [],
                }
            )
        )
        with patch("src.app.get_index_status", return_value=index_status):
            response = self.client.get("/documents")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Documents", response.data)
        self.assertIn(b"documents/course.txt", response.data)

    def test_setup_page_renders_report(self):
        report = HealthReport(
            "incomplete",
            [HealthCheck("documents", "incomplete", "No documents.")],
        )
        with patch("src.app.get_health_report", return_value=report):
            response = self.client.get("/setup")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Setup Status", response.data)
        self.assertIn(b"No documents.", response.data)


if __name__ == "__main__":
    unittest.main()
