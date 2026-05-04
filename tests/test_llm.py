import unittest
from types import SimpleNamespace

from src.llm import chat_completion
from src.settings import CourseSettings


class FakeCompletions:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        message = SimpleNamespace(content="ok")
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class LlmTests(unittest.TestCase):
    def test_chat_completion_translates_token_limit_for_gpt5_models(self):
        completions = FakeCompletions()
        openai_module = SimpleNamespace(chat=SimpleNamespace(completions=completions))

        reply = chat_completion(
            [{"role": "user", "content": "Say ok."}],
            settings=CourseSettings(chat_model="gpt-5.4"),
            openai_module=openai_module,
            max_tokens=5,
            temperature=0.0,
        )

        self.assertEqual(reply, "ok")
        call = completions.calls[0]
        self.assertEqual(call["model"], "gpt-5.4")
        self.assertEqual(call["max_completion_tokens"], 5)
        self.assertNotIn("max_tokens", call)
        self.assertNotIn("temperature", call)

    def test_chat_completion_preserves_legacy_parameters_for_gpt4_models(self):
        completions = FakeCompletions()
        openai_module = SimpleNamespace(chat=SimpleNamespace(completions=completions))

        chat_completion(
            [{"role": "user", "content": "Say ok."}],
            settings=CourseSettings(chat_model="gpt-4o-mini"),
            openai_module=openai_module,
            max_tokens=5,
            temperature=0.0,
        )

        call = completions.calls[0]
        self.assertEqual(call["model"], "gpt-4o-mini")
        self.assertEqual(call["max_tokens"], 5)
        self.assertEqual(call["temperature"], 0.0)


if __name__ == "__main__":
    unittest.main()
