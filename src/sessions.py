from dataclasses import dataclass, field
from threading import Lock
from typing import Dict, List, Optional


@dataclass
class ConversationTurn:
    user: str
    assistant: str
    context: str
    sources: List[dict]


@dataclass
class ConversationState:
    turns: List[ConversationTurn] = field(default_factory=list)
    _lock: Lock = field(default_factory=Lock, init=False, repr=False, compare=False)

    def add_turn(self, user, assistant, context, sources, max_turns=5):
        with self._lock:
            self.turns.append(
                ConversationTurn(
                    user=user,
                    assistant=assistant,
                    context=context,
                    sources=sources,
                )
            )
            if len(self.turns) > max_turns:
                del self.turns[:-max_turns]

    def latest(self) -> Optional[ConversationTurn]:
        with self._lock:
            if not self.turns:
                return None
            return self.turns[-1]

    def summary(self, max_turns=3):
        with self._lock:
            recent = self.turns[-max_turns:]
            parts = []
            for turn in recent:
                parts.append(f"Student: {turn.user}\nAssistant: {turn.assistant}")
            return "\n\n".join(parts)

    def turn_count(self):
        with self._lock:
            return len(self.turns)


class ConversationStore:
    def __init__(self):
        self._states: Dict[str, ConversationState] = {}
        self._lock = Lock()

    def get(self, session_id):
        with self._lock:
            return self._states.setdefault(session_id, ConversationState())

    def clear(self, session_id):
        with self._lock:
            self._states.pop(session_id, None)


conversation_store = ConversationStore()
