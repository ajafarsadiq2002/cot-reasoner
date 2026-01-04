"""Tests for ReasoningStep and ReasoningChain."""

import pytest
from datetime import datetime

from cot_reasoner.core.chain import ReasoningStep, ReasoningChain


class TestReasoningStep:
    """Tests for ReasoningStep dataclass."""

    def test_create_step(self):
        """Test creating a reasoning step."""
        step = ReasoningStep(number=1, content="First step")
        assert step.number == 1
        assert step.content == "First step"
        assert step.confidence == 1.0

    def test_step_with_confidence(self):
        """Test step with custom confidence."""
        step = ReasoningStep(number=2, content="Second step", confidence=0.8)
        assert step.confidence == 0.8

    def test_step_to_dict(self):
        """Test serializing step to dict."""
        step = ReasoningStep(number=1, content="Test step")
        data = step.to_dict()
        assert data["number"] == 1
        assert data["content"] == "Test step"
        assert "timestamp" in data

    def test_step_from_dict(self):
        """Test deserializing step from dict."""
        data = {
            "number": 3,
            "content": "From dict",
            "confidence": 0.9,
            "timestamp": datetime.now().isoformat(),
        }
        step = ReasoningStep.from_dict(data)
        assert step.number == 3
        assert step.content == "From dict"
        assert step.confidence == 0.9

    def test_step_str(self):
        """Test string representation."""
        step = ReasoningStep(number=1, content="Hello")
        assert str(step) == "Step 1: Hello"


class TestReasoningChain:
    """Tests for ReasoningChain dataclass."""

    def test_create_chain(self):
        """Test creating a reasoning chain."""
        chain = ReasoningChain(query="What is 2+2?")
        assert chain.query == "What is 2+2?"
        assert chain.steps == []
        assert chain.answer is None
        assert not chain.is_complete

    def test_add_step(self):
        """Test adding steps to chain."""
        chain = ReasoningChain(query="Test")
        step1 = chain.add_step("First reasoning")
        step2 = chain.add_step("Second reasoning")

        assert len(chain.steps) == 2
        assert step1.number == 1
        assert step2.number == 2
        assert chain.step_count == 2

    def test_set_answer(self):
        """Test setting final answer."""
        chain = ReasoningChain(query="Test")
        chain.add_step("Reasoning step")
        chain.set_answer("42", confidence=0.95)

        assert chain.answer == "42"
        assert chain.confidence == 0.95
        assert chain.is_complete

    def test_chain_to_dict(self):
        """Test serializing chain to dict."""
        chain = ReasoningChain(
            query="Test query",
            provider="openai",
            model="gpt-4",
            strategy="standard",
        )
        chain.add_step("Step 1")
        chain.set_answer("Answer")

        data = chain.to_dict()
        assert data["query"] == "Test query"
        assert data["provider"] == "openai"
        assert len(data["steps"]) == 1
        assert data["answer"] == "Answer"

    def test_chain_to_json(self):
        """Test JSON serialization."""
        chain = ReasoningChain(query="JSON test")
        chain.add_step("A step")
        chain.set_answer("Result")

        json_str = chain.to_json()
        assert "JSON test" in json_str
        assert "Result" in json_str

    def test_chain_from_json(self):
        """Test JSON deserialization."""
        chain = ReasoningChain(query="Original")
        chain.add_step("Step")
        chain.set_answer("Answer")

        json_str = chain.to_json()
        restored = ReasoningChain.from_json(json_str)

        assert restored.query == "Original"
        assert restored.answer == "Answer"
        assert len(restored.steps) == 1

    def test_format_steps(self):
        """Test formatting steps as string."""
        chain = ReasoningChain(query="Test")
        chain.add_step("First step")
        chain.add_step("Second step")

        formatted = chain.format_steps()
        assert "Step 1: First step" in formatted
        assert "Step 2: Second step" in formatted

    def test_chain_str(self):
        """Test string representation of chain."""
        chain = ReasoningChain(query="What is 1+1?")
        chain.add_step("Adding numbers")
        chain.set_answer("2")

        output = str(chain)
        assert "Query: What is 1+1?" in output
        assert "Adding numbers" in output
        assert "Answer: 2" in output
