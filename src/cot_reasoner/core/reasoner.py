"""Main Reasoner class - the primary interface for Chain of Thought reasoning."""

from typing import Iterator, Optional, Type

from cot_reasoner.core.chain import ReasoningChain
from cot_reasoner.core.memory import ConversationMemory
from cot_reasoner.providers.base import BaseLLMProvider
from cot_reasoner.providers.openai import OpenAIProvider
from cot_reasoner.providers.anthropic import AnthropicProvider
from cot_reasoner.strategies.base import BaseStrategy
from cot_reasoner.strategies.standard import StandardCoTStrategy
from cot_reasoner.strategies.zero_shot import ZeroShotCoTStrategy
from cot_reasoner.strategies.self_consistency import SelfConsistencyStrategy


# Provider registry
PROVIDERS: dict[str, Type[BaseLLMProvider]] = {
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
}

# Strategy registry
STRATEGIES: dict[str, Type[BaseStrategy]] = {
    "standard": StandardCoTStrategy,
    "zero_shot": ZeroShotCoTStrategy,
    "self_consistency": SelfConsistencyStrategy,
}


class Reasoner:
    """Main Chain of Thought Reasoner class.

    This is the primary interface for performing Chain of Thought reasoning.
    It orchestrates LLM providers and reasoning strategies.

    Example usage:
        ```python
        from cot_reasoner import Reasoner

        # Create reasoner with defaults (OpenAI, standard CoT)
        reasoner = Reasoner()

        # Or customize provider and strategy
        reasoner = Reasoner(
            provider="anthropic",
            model="claude-sonnet-4-20250514",
            strategy="self_consistency"
        )

        # Perform reasoning
        result = reasoner.reason("What is 15% of 240?")

        # Access results
        for step in result.steps:
            print(f"Step {step.number}: {step.content}")
        print(f"Answer: {result.answer}")

        # With conversation memory
        reasoner = Reasoner(memory=True)
        result1 = reasoner.reason("What is 15% of 240?")  # Answer: 36
        result2 = reasoner.reason("Double that")  # Answer: 72 (remembers context)
        reasoner.clear_memory()  # Reset conversation
        ```
    """

    def __init__(
        self,
        provider: str = "openai",
        model: Optional[str] = None,
        strategy: str = "standard",
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        memory: bool = False,
        max_memory_turns: int = 10,
        **kwargs,
    ):
        """Initialize the Reasoner.

        Args:
            provider: LLM provider name ("openai" or "anthropic")
            model: Model identifier (uses provider default if not specified)
            strategy: Reasoning strategy ("standard", "zero_shot", "self_consistency")
            api_key: API key for the provider (or read from environment)
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            memory: Enable conversation memory for contextual reasoning
            max_memory_turns: Maximum conversation turns to remember (default: 10)
            **kwargs: Additional provider or strategy options
        """
        self._provider_name = provider
        self._strategy_name = strategy
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._memory_enabled = memory

        # Initialize conversation memory
        self._memory = ConversationMemory(max_turns=max_memory_turns) if memory else None

        # Initialize provider
        self._provider = self._create_provider(
            provider=provider,
            model=model,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

        # Initialize strategy
        self._strategy = self._create_strategy(
            strategy=strategy,
            provider=self._provider,
            **kwargs,
        )

    @property
    def provider(self) -> BaseLLMProvider:
        """Get the current LLM provider."""
        return self._provider

    @property
    def strategy(self) -> BaseStrategy:
        """Get the current reasoning strategy."""
        return self._strategy

    @property
    def model(self) -> str:
        """Get the current model name."""
        return self._provider.model

    @property
    def memory(self) -> Optional[ConversationMemory]:
        """Get the conversation memory (None if disabled)."""
        return self._memory

    @property
    def has_memory(self) -> bool:
        """Check if memory is enabled."""
        return self._memory_enabled

    def clear_memory(self) -> None:
        """Clear conversation memory."""
        if self._memory:
            self._memory.clear()

    def _create_provider(
        self,
        provider: str,
        model: Optional[str] = None,
        **kwargs,
    ) -> BaseLLMProvider:
        """Create an LLM provider instance."""
        if provider not in PROVIDERS:
            available = ", ".join(PROVIDERS.keys())
            raise ValueError(f"Unknown provider '{provider}'. Available: {available}")

        provider_class = PROVIDERS[provider]

        # Use provider's default model if not specified
        if model is None:
            model = provider_class.DEFAULT_MODEL

        return provider_class(model=model, **kwargs)

    def _create_strategy(
        self,
        strategy: str,
        provider: BaseLLMProvider,
        **kwargs,
    ) -> BaseStrategy:
        """Create a reasoning strategy instance."""
        if strategy not in STRATEGIES:
            available = ", ".join(STRATEGIES.keys())
            raise ValueError(f"Unknown strategy '{strategy}'. Available: {available}")

        strategy_class = STRATEGIES[strategy]

        # Handle strategy-specific kwargs
        if strategy == "self_consistency":
            num_samples = kwargs.pop("num_samples", 3)
            return strategy_class(provider, num_samples=num_samples)

        return strategy_class(provider)

    def reason(self, query: str, **kwargs) -> ReasoningChain:
        """Perform Chain of Thought reasoning on a query.

        Args:
            query: The problem or question to reason about
            **kwargs: Additional options passed to the strategy

        Returns:
            ReasoningChain containing steps and final answer

        Example:
            ```python
            result = reasoner.reason("If a train travels at 60 mph for 2.5 hours, how far does it go?")
            print(result.answer)  # "150 miles"
            ```
        """
        # Get conversation context if memory is enabled
        context = None
        if self._memory and not self._memory.is_empty:
            context = self._memory.get_context()

        # Perform reasoning with context
        result = self._strategy.reason(query, context=context, **kwargs)

        # Save to memory if enabled
        if self._memory and result.answer:
            self._memory.add_turn(query, result.answer)

        return result

    async def reason_async(self, query: str, **kwargs) -> ReasoningChain:
        """Perform Chain of Thought reasoning asynchronously.

        Args:
            query: The problem or question to reason about
            **kwargs: Additional options passed to the strategy

        Returns:
            ReasoningChain containing steps and final answer
        """
        # Get conversation context if memory is enabled
        context = None
        if self._memory and not self._memory.is_empty:
            context = self._memory.get_context()

        # Perform reasoning with context
        result = await self._strategy.reason_async(query, context=context, **kwargs)

        # Save to memory if enabled
        if self._memory and result.answer:
            self._memory.add_turn(query, result.answer)

        return result

    def reason_stream(self, query: str, **kwargs) -> Iterator[str]:
        """Stream the reasoning process (shows raw LLM output).

        Note: Streaming returns raw text, not parsed ReasoningChain.
        Use `reason()` for structured output.

        Args:
            query: The problem or question to reason about
            **kwargs: Additional options

        Yields:
            Text chunks as they are generated
        """
        from cot_reasoner.core.prompts import get_prompt_template

        prompt = get_prompt_template(self._strategy_name, "user").format(query=query)
        system_prompt = get_prompt_template(self._strategy_name, "system")

        yield from self._provider.stream(
            prompt=prompt,
            system_prompt=system_prompt,
            **kwargs,
        )

    @classmethod
    def list_providers(cls) -> list[str]:
        """List available LLM providers."""
        return list(PROVIDERS.keys())

    @classmethod
    def list_strategies(cls) -> list[str]:
        """List available reasoning strategies."""
        return list(STRATEGIES.keys())

    @classmethod
    def register_provider(cls, name: str, provider_class: Type[BaseLLMProvider]) -> None:
        """Register a custom LLM provider.

        Args:
            name: Name to register the provider under
            provider_class: Provider class (must extend BaseLLMProvider)

        Example:
            ```python
            from cot_reasoner import Reasoner
            from my_providers import OllamaProvider

            Reasoner.register_provider("ollama", OllamaProvider)
            reasoner = Reasoner(provider="ollama", model="llama2")
            ```
        """
        if not issubclass(provider_class, BaseLLMProvider):
            raise TypeError("Provider must extend BaseLLMProvider")
        PROVIDERS[name] = provider_class

    @classmethod
    def register_strategy(cls, name: str, strategy_class: Type[BaseStrategy]) -> None:
        """Register a custom reasoning strategy.

        Args:
            name: Name to register the strategy under
            strategy_class: Strategy class (must extend BaseStrategy)

        Example:
            ```python
            from cot_reasoner import Reasoner
            from my_strategies import TreeOfThoughtStrategy

            Reasoner.register_strategy("tree_of_thought", TreeOfThoughtStrategy)
            reasoner = Reasoner(strategy="tree_of_thought")
            ```
        """
        if not issubclass(strategy_class, BaseStrategy):
            raise TypeError("Strategy must extend BaseStrategy")
        STRATEGIES[name] = strategy_class

    def __repr__(self) -> str:
        return (
            f"Reasoner(provider={self._provider_name!r}, "
            f"model={self.model!r}, "
            f"strategy={self._strategy_name!r})"
        )
