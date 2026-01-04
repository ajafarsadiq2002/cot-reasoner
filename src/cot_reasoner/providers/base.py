"""Base class for LLM providers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncIterator, Iterator, Optional


@dataclass
class LLMResponse:
    """Response from an LLM provider."""

    content: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    finish_reason: str = ""

    @property
    def tokens(self) -> int:
        """Total tokens used."""
        return self.total_tokens or (self.prompt_tokens + self.completion_tokens)


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers.

    To add a new provider, extend this class and implement:
    - generate(): Synchronous text generation
    - generate_async(): Asynchronous text generation
    - stream(): Synchronous streaming generation
    - stream_async(): Asynchronous streaming generation
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs,
    ):
        """Initialize the provider.

        Args:
            model: Model identifier
            api_key: API key (or read from environment)
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional provider-specific options
        """
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.kwargs = kwargs

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider name."""
        pass

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> LLMResponse:
        """Generate a response synchronously.

        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            **kwargs: Additional generation options

        Returns:
            LLMResponse with the generated content
        """
        pass

    @abstractmethod
    async def generate_async(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> LLMResponse:
        """Generate a response asynchronously.

        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            **kwargs: Additional generation options

        Returns:
            LLMResponse with the generated content
        """
        pass

    @abstractmethod
    def stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> Iterator[str]:
        """Stream a response synchronously.

        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            **kwargs: Additional generation options

        Yields:
            Text chunks as they are generated
        """
        pass

    @abstractmethod
    async def stream_async(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        """Stream a response asynchronously.

        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            **kwargs: Additional generation options

        Yields:
            Text chunks as they are generated
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model!r})"
