"""Tests for the main Reasoner class."""

import pytest
from unittest.mock import Mock, patch

from cot_reasoner.core.reasoner import Reasoner, PROVIDERS, STRATEGIES
from cot_reasoner.providers.base import BaseLLMProvider, LLMResponse
from cot_reasoner.strategies.base import BaseStrategy


class TestReasonerInit:
    """Tests for Reasoner initialization."""

    def test_list_providers(self):
        """Test listing available providers."""
        providers = Reasoner.list_providers()
        assert "openai" in providers
        assert "anthropic" in providers

    def test_list_strategies(self):
        """Test listing available strategies."""
        strategies = Reasoner.list_strategies()
        assert "standard" in strategies
        assert "zero_shot" in strategies
        assert "self_consistency" in strategies

    def test_invalid_provider(self):
        """Test error on invalid provider."""
        with pytest.raises(ValueError, match="Unknown provider"):
            Reasoner(provider="invalid_provider", api_key="test")

    def test_invalid_strategy(self):
        """Test error on invalid strategy."""
        with pytest.raises(ValueError, match="Unknown strategy"):
            Reasoner(provider="openai", strategy="invalid_strategy", api_key="test")


class TestReasonerRegistration:
    """Tests for custom provider/strategy registration."""

    def test_register_custom_provider(self):
        """Test registering a custom provider."""

        class MockProvider(BaseLLMProvider):
            DEFAULT_MODEL = "mock-model"

            @property
            def provider_name(self):
                return "mock"

            def generate(self, prompt, system_prompt=None, **kwargs):
                return LLMResponse(content="Mock response", model=self.model)

            async def generate_async(self, prompt, system_prompt=None, **kwargs):
                return LLMResponse(content="Mock response", model=self.model)

            def stream(self, prompt, system_prompt=None, **kwargs):
                yield "Mock"

            async def stream_async(self, prompt, system_prompt=None, **kwargs):
                yield "Mock"

        Reasoner.register_provider("mock", MockProvider)
        assert "mock" in Reasoner.list_providers()

        # Cleanup
        del PROVIDERS["mock"]

    def test_register_invalid_provider(self):
        """Test error when registering non-provider class."""

        class NotAProvider:
            pass

        with pytest.raises(TypeError, match="must extend BaseLLMProvider"):
            Reasoner.register_provider("invalid", NotAProvider)

    def test_register_custom_strategy(self):
        """Test registering a custom strategy."""

        class MockStrategy(BaseStrategy):
            @property
            def strategy_name(self):
                return "mock"

            def reason(self, query, **kwargs):
                chain = self._create_chain(query)
                chain.add_step("Mock reasoning")
                chain.set_answer("Mock answer")
                return chain

            async def reason_async(self, query, **kwargs):
                return self.reason(query, **kwargs)

        # Need a mock provider first
        mock_provider = Mock(spec=BaseLLMProvider)
        mock_provider.provider_name = "mock"
        mock_provider.model = "mock-model"

        Reasoner.register_strategy("mock_strategy", MockStrategy)
        assert "mock_strategy" in Reasoner.list_strategies()

        # Cleanup
        del STRATEGIES["mock_strategy"]


class TestReasonerRepr:
    """Tests for Reasoner string representation."""

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    def test_repr(self):
        """Test repr output."""
        reasoner = Reasoner(provider="openai", strategy="standard")
        repr_str = repr(reasoner)
        assert "openai" in repr_str
        assert "standard" in repr_str
