"""Anthropic LLM provider implementation."""

import os
from typing import AsyncIterator, Iterator, Optional

import anthropic

from cot_reasoner.providers.base import BaseLLMProvider, LLMResponse


class AnthropicProvider(BaseLLMProvider):
    """Anthropic API provider for Claude models."""

    DEFAULT_MODEL = "claude-sonnet-4-20250514"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs,
    ):
        """Initialize Anthropic provider.

        Args:
            model: Anthropic model to use (default: claude-sonnet-4-20250514)
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional options
        """
        super().__init__(model, api_key, temperature, max_tokens, **kwargs)
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("Anthropic API key required. Set ANTHROPIC_API_KEY environment variable.")

        self._client = anthropic.Anthropic(api_key=self.api_key)
        self._async_client = anthropic.AsyncAnthropic(api_key=self.api_key)

    @property
    def provider_name(self) -> str:
        return "anthropic"

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> LLMResponse:
        """Generate a response synchronously."""
        message_kwargs = {
            "model": self.model,
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "messages": [{"role": "user", "content": prompt}],
        }

        # Add temperature if not 1.0 (Anthropic default)
        temp = kwargs.get("temperature", self.temperature)
        if temp != 1.0:
            message_kwargs["temperature"] = temp

        if system_prompt:
            message_kwargs["system"] = system_prompt

        response = self._client.messages.create(**message_kwargs)

        content = ""
        if response.content:
            content = response.content[0].text

        return LLMResponse(
            content=content,
            model=response.model,
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens,
            total_tokens=response.usage.input_tokens + response.usage.output_tokens,
            finish_reason=response.stop_reason or "",
        )

    async def generate_async(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> LLMResponse:
        """Generate a response asynchronously."""
        message_kwargs = {
            "model": self.model,
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "messages": [{"role": "user", "content": prompt}],
        }

        temp = kwargs.get("temperature", self.temperature)
        if temp != 1.0:
            message_kwargs["temperature"] = temp

        if system_prompt:
            message_kwargs["system"] = system_prompt

        response = await self._async_client.messages.create(**message_kwargs)

        content = ""
        if response.content:
            content = response.content[0].text

        return LLMResponse(
            content=content,
            model=response.model,
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens,
            total_tokens=response.usage.input_tokens + response.usage.output_tokens,
            finish_reason=response.stop_reason or "",
        )

    def stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> Iterator[str]:
        """Stream a response synchronously."""
        message_kwargs = {
            "model": self.model,
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "messages": [{"role": "user", "content": prompt}],
        }

        temp = kwargs.get("temperature", self.temperature)
        if temp != 1.0:
            message_kwargs["temperature"] = temp

        if system_prompt:
            message_kwargs["system"] = system_prompt

        with self._client.messages.stream(**message_kwargs) as stream:
            for text in stream.text_stream:
                yield text

    async def stream_async(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        """Stream a response asynchronously."""
        message_kwargs = {
            "model": self.model,
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "messages": [{"role": "user", "content": prompt}],
        }

        temp = kwargs.get("temperature", self.temperature)
        if temp != 1.0:
            message_kwargs["temperature"] = temp

        if system_prompt:
            message_kwargs["system"] = system_prompt

        async with self._async_client.messages.stream(**message_kwargs) as stream:
            async for text in stream.text_stream:
                yield text
