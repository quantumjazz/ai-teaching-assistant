import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class StaticAndOpsTests(unittest.TestCase):
    def test_chat_rendering_uses_text_content_not_inner_html(self):
        script = (PROJECT_ROOT / "static" / "js" / "queryProcess.js").read_text(
            encoding="utf-8"
        )

        self.assertIn("textContent", script)
        self.assertIn("renderMarkdownMessage", script)
        self.assertIn("appendInlineMarkdown", script)
        self.assertIn("document.createElement('strong')", script)
        self.assertIn("document.createElement('ul')", script)
        self.assertNotIn("innerHTML", script)

    def test_chat_form_documents_enter_shortcut_and_scopes_submit_width(self):
        html = (PROJECT_ROOT / "templates" / "index.html").read_text(encoding="utf-8")
        css = (PROJECT_ROOT / "static" / "css" / "styles.css").read_text(
            encoding="utf-8"
        )

        self.assertIn("Enter to send", html)
        self.assertIn("Shift+Enter", html)
        self.assertIn("min-height: 100dvh", css)
        self.assertIn("#submit-button", css)
        self.assertNotIn("height: 100vh", css)

    def test_dockerfile_is_hardened_for_local_production_path(self):
        dockerfile = (PROJECT_ROOT / "Dockerfile").read_text(encoding="utf-8")
        start_script = (PROJECT_ROOT / "scripts" / "start_web.sh").read_text(
            encoding="utf-8"
        )

        self.assertIn("--no-install-recommends", dockerfile)
        self.assertIn("rm -rf /var/lib/apt/lists/*", dockerfile)
        self.assertIn("scripts/start_web.sh", dockerfile)
        self.assertIn("--workers 1", start_script)
        self.assertIn("${PORT:-8080}", start_script)
        self.assertIn("USER appuser", dockerfile)

    def test_dockerignore_and_make_clean_data_exist(self):
        dockerignore = (PROJECT_ROOT / ".dockerignore").read_text(encoding="utf-8")
        makefile = (PROJECT_ROOT / "Makefile").read_text(encoding="utf-8")

        self.assertIn(".env", dockerignore)
        self.assertIn("data/*", dockerignore)
        self.assertIn("documents/*", dockerignore)
        self.assertIn("!documents/demo_*.txt", dockerignore)
        self.assertIn("clean-data:", makefile)

    def test_render_blueprint_declares_demo_web_service(self):
        render_yaml = (PROJECT_ROOT / "render.yaml").read_text(encoding="utf-8")

        self.assertIn("runtime: docker", render_yaml)
        self.assertIn("healthCheckPath: /api/health", render_yaml)
        self.assertIn("OPENAI_API_KEY", render_yaml)
        self.assertIn("AUTO_INDEX_ON_STARTUP", render_yaml)

    def test_dependencies_are_major_pinned(self):
        requirements = (PROJECT_ROOT / "requirements.txt").read_text(encoding="utf-8")

        for package in (
            "PyPDF2",
            "python-docx",
            "openai",
            "python-dotenv",
            "faiss-cpu",
            "numpy",
            "Flask",
            "gunicorn",
        ):
            self.assertRegex(requirements, rf"(?m)^{package}>=.+,<.+$", package)

    def test_dead_root_config_removed(self):
        self.assertFalse((PROJECT_ROOT / "config.py").exists())


if __name__ == "__main__":
    unittest.main()
