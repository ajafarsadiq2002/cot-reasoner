"""Conversation memory for maintaining context across reasoning calls."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class ConversationTurn:
    """A single turn in the conversation."""

    query: str
    answer: str
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "answer": self.answer,
            "timestamp": self.timestamp.isoformat(),
        }


class ConversationMemory:
    """Manages conversation history for contextual reasoning.

    Example:
        ```python
        memory = ConversationMemory()
        memory.add_turn("What is 15% of 240?", "36")
        memory.add_turn("Double that", "72")

        # Get context for next query
        context = memory.get_context()
        ```
    """

    def __init__(self, max_turns: int = 10):
        """Initialize conversation memory.

        Args:
            max_turns: Maximum number of turns to keep in memory (default: 10)
        """
        self.max_turns = max_turns
        self._history: list[ConversationTurn] = []

    def add_turn(self, query: str, answer: str) -> None:
        """Add a conversation turn to memory.

        Args:
            query: The user's question
            answer: The reasoner's answer
        """
        turn = ConversationTurn(query=query, answer=answer)
        self._history.append(turn)

        # Trim to max_turns
        if len(self._history) > self.max_turns:
            self._history = self._history[-self.max_turns:]

    def get_context(self, last_n: Optional[int] = None) -> str:
        """Get formatted conversation context for prompts.

        Args:
            last_n: Only include last N turns (default: all)

        Returns:
            Formatted string of conversation history
        """
        if not self._history:
            return ""

        turns = self._history
        if last_n:
            turns = self._history[-last_n:]

        lines = ["=== CONVERSATION HISTORY (use this for context) ==="]
        for i, turn in enumerate(turns, 1):
            lines.append(f"User Question {i}: {turn.query}")
            lines.append(f"Your Answer {i}: {turn.answer}")
            lines.append("")
        lines.append("=== END OF HISTORY ===")
        lines.append("")
        lines.append("Use the above history to understand references like 'that', 'it', 'the result', etc.")
        lines.append("")

        return "\n".join(lines)

    def get_history(self) -> list[ConversationTurn]:
        """Get the full conversation history."""
        return self._history.copy()

    def clear(self) -> None:
        """Clear all conversation history."""
        self._history = []

    @property
    def turn_count(self) -> int:
        """Get number of turns in memory."""
        return len(self._history)

    @property
    def is_empty(self) -> bool:
        """Check if memory is empty."""
        return len(self._history) == 0

    def __len__(self) -> int:
        return len(self._history)

    def __bool__(self) -> bool:
        """Memory object is always truthy when it exists (regardless of turn count)."""
        return True

    def __repr__(self) -> str:
        return f"ConversationMemory(turns={self.turn_count}, max={self.max_turns})"
