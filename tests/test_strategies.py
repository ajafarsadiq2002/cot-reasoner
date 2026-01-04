"""Tests for CoT strategies."""

import pytest
from unittest.mock import Mock, AsyncMock

from cot_reasoner.core.chain import ReasoningChain
from cot_reasoner.providers.base import BaseLLMProvider, LLMResponse
from cot_reasoner.strategies.standard import StandardCoTStrategy
from cot_reasoner.strategies.zero_shot import ZeroShotCoTStrategy
from cot_reasoner.strategies.base import BaseStrategy


@pytest.fixture
def mock_provider():
    """Create a mock LLM provider."""
    provider = Mock(spec=BaseLLMProvider)
    provider.provider_name = "mock"
    provider.model = "mock-model"
    return provider


class TestBaseStrategy:
    """Tests for base strategy parsing."""

    def test_parse_numbered_steps(self, mock_provider):
        """Test parsing numbered step format."""

        class TestStrategy(BaseStrategy):
            @property
            def strategy_name(self):
                return "test"

            def reason(self, query, **kwargs):
                pass

            async def reason_async(self, query, **kwargs):
                pass

        strategy = TestStrategy(mock_provider)
        chain = ReasoningChain(query="Test")

        response = """Step 1: First I analyze the problem
Step 2: Then I calculate the result
Step 3: Finally I verify
Answer: 42"""

        result = strategy.parse_response(response, chain)

        assert len(result.steps) == 3
        assert result.steps[0].content == "First I analyze the problem"
        assert result.steps[1].content == "Then I calculate the result"
        assert result.answer == "42"

    def test_parse_alternative_format(self, mock_provider):
        """Test parsing alternative numbering format."""

        class TestStrategy(BaseStrategy):
            @property
            def strategy_name(self):
                return "test"

            def reason(self, query, **kwargs):
                pass

            async def reason_async(self, query, **kwargs):
                pass

        strategy = TestStrategy(mock_provider)
        chain = ReasoningChain(query="Test")

        response = """1. First step here
2. Second step here
3. Third step here
Final Answer: The result is 100"""

        result = strategy.parse_response(response, chain)

        assert len(result.steps) >= 3
        assert result.answer is not None


class TestStandardCoTStrategy:
    """Tests for Standard CoT strategy."""

    def test_strategy_name(self, mock_provider):
        """Test strategy name property."""
        strategy = StandardCoTStrategy(mock_provider)
        assert strategy.strategy_name == "standard"

    def test_reason_sync(self, mock_provider):
        """Test synchronous reasoning."""
        mock_provider.generate.return_value = LLMResponse(
            content="""Step 1: Analyze the problem
Step 2: Calculate 15% of 240 = 0.15 * 240 = 36
Answer: 36""",
            model="mock-model",
            total_tokens=100,
        )

        strategy = StandardCoTStrategy(mock_provider)
        result = strategy.reason("What is 15% of 240?")

        assert isinstance(result, ReasoningChain)
        assert result.query == "What is 15% of 240?"
        assert len(result.steps) >= 1
        assert result.total_tokens == 100
        mock_provider.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_reason_async(self, mock_provider):
        """Test asynchronous reasoning."""
        mock_provider.generate_async = AsyncMock(
            return_value=LLMResponse(
                content="""Step 1: Think about it
Answer: 42""",
                model="mock-model",
                total_tokens=50,
            )
        )

        strategy = StandardCoTStrategy(mock_provider)
        result = await strategy.reason_async("What is the meaning?")

        assert isinstance(result, ReasoningChain)
        assert result.total_tokens == 50
        mock_provider.generate_async.assert_called_once()


class TestZeroShotCoTStrategy:
    """Tests for Zero-shot CoT strategy."""

    def test_strategy_name(self, mock_provider):
        """Test strategy name property."""
        strategy = ZeroShotCoTStrategy(mock_provider)
        assert strategy.strategy_name == "zero_shot"

    def test_reason_sync(self, mock_provider):
        """Test synchronous reasoning."""
        mock_provider.generate.return_value = LLMResponse(
            content="""Let me think step by step.
Step 1: The question asks about percentages
Step 2: 15% means 15/100 = 0.15
Step 3: 0.15 * 240 = 36
Answer: 36""",
            model="mock-model",
            total_tokens=80,
        )

        strategy = ZeroShotCoTStrategy(mock_provider)
        result = strategy.reason("Calculate 15% of 240")

        assert isinstance(result, ReasoningChain)
        assert result.strategy == "zero_shot"
        mock_provider.generate.assert_called_once()


class TestPromptFormatting:
    """Tests for prompt formatting in strategies."""

    def test_standard_prompt_contains_query(self, mock_provider):
        """Test that the query is included in the prompt."""
        mock_provider.generate.return_value = LLMResponse(
            content="Step 1: Test\nAnswer: Test",
            model="mock-model",
        )

        strategy = StandardCoTStrategy(mock_provider)
        strategy.reason("My specific question here")

        # Check the prompt passed to generate
        call_args = mock_provider.generate.call_args
        prompt = call_args.kwargs.get("prompt") or call_args.args[0]
        assert "My specific question here" in prompt
