import unittest

from src.sessions import ConversationState


class ConversationStateTests(unittest.TestCase):
    def test_add_turn_trims_to_max_turns_and_tracks_count(self):
        state = ConversationState()

        for index in range(6):
            state.add_turn(
                f"question {index}",
                f"answer {index}",
                f"context {index}",
                [],
                max_turns=5,
            )

        self.assertEqual(state.turn_count(), 5)
        self.assertEqual(state.turns[0].user, "question 1")
        self.assertEqual(state.latest().user, "question 5")

    def test_latest_and_summary_are_safe_after_multiple_turns(self):
        state = ConversationState()
        state.add_turn("first question", "first answer", "first context", [])
        state.add_turn("second question", "second answer", "second context", [])

        latest = state.latest()
        summary = state.summary(max_turns=1)

        self.assertEqual(latest.user, "second question")
        self.assertIn("second question", summary)
        self.assertNotIn("first question", summary)

    def test_latest_returns_none_for_empty_state(self):
        state = ConversationState()

        self.assertIsNone(state.latest())
        self.assertEqual(state.summary(), "")
        self.assertEqual(state.turn_count(), 0)


if __name__ == "__main__":
    unittest.main()
