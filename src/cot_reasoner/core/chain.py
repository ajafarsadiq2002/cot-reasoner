"""Data models for Chain of Thought reasoning."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import json


@dataclass
class ReasoningStep:
    """Represents a single step in the reasoning chain."""

    number: int
    content: str
    confidence: float = 1.0
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        """Convert step to dictionary."""
        return {
            "number": self.number,
            "content": self.content,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ReasoningStep":
        """Create step from dictionary."""
        return cls(
            number=data["number"],
            content=data["content"],
            confidence=data.get("confidence", 1.0),
            timestamp=datetime.fromisoformat(data["timestamp"]) if "timestamp" in data else datetime.now(),
        )

    def __str__(self) -> str:
        return f"Step {self.number}: {self.content}"


@dataclass
class ReasoningChain:
    """Represents a complete chain of reasoning with final answer."""

    query: str
    steps: list[ReasoningStep] = field(default_factory=list)
    answer: Optional[str] = None
    confidence: float = 0.0
    provider: str = ""
    model: str = ""
    strategy: str = ""
    total_tokens: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    metadata: dict = field(default_factory=dict)

    def add_step(self, content: str, confidence: float = 1.0) -> ReasoningStep:
        """Add a new reasoning step to the chain."""
        step = ReasoningStep(
            number=len(self.steps) + 1,
            content=content,
            confidence=confidence,
        )
        self.steps.append(step)
        return step

    def set_answer(self, answer: str, confidence: float = 1.0) -> None:
        """Set the final answer with confidence score."""
        self.answer = answer
        self.confidence = confidence

    @property
    def is_complete(self) -> bool:
        """Check if reasoning chain has a final answer."""
        return self.answer is not None

    @property
    def step_count(self) -> int:
        """Get number of reasoning steps."""
        return len(self.steps)

    def to_dict(self) -> dict:
        """Convert chain to dictionary."""
        return {
            "query": self.query,
            "steps": [step.to_dict() for step in self.steps],
            "answer": self.answer,
            "confidence": self.confidence,
            "provider": self.provider,
            "model": self.model,
            "strategy": self.strategy,
            "total_tokens": self.total_tokens,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ReasoningChain":
        """Create chain from dictionary."""
        chain = cls(
            query=data["query"],
            answer=data.get("answer"),
            confidence=data.get("confidence", 0.0),
            provider=data.get("provider", ""),
            model=data.get("model", ""),
            strategy=data.get("strategy", ""),
            total_tokens=data.get("total_tokens", 0),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.now(),
            metadata=data.get("metadata", {}),
        )
        chain.steps = [ReasoningStep.from_dict(s) for s in data.get("steps", [])]
        return chain

    def to_json(self, indent: int = 2) -> str:
        """Convert chain to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> "ReasoningChain":
        """Create chain from JSON string."""
        return cls.from_dict(json.loads(json_str))

    def format_steps(self, include_confidence: bool = False) -> str:
        """Format reasoning steps as readable string."""
        lines = []
        for step in self.steps:
            if include_confidence:
                lines.append(f"Step {step.number} (confidence: {step.confidence:.2f}): {step.content}")
            else:
                lines.append(f"Step {step.number}: {step.content}")
        return "\n".join(lines)

    def __str__(self) -> str:
        output = f"Query: {self.query}\n\n"
        output += "Reasoning:\n"
        output += self.format_steps()
        if self.answer:
            output += f"\n\nAnswer: {self.answer}"
            output += f" (confidence: {self.confidence:.2f})"
        return output
