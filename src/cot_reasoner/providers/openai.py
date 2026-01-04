"""OpenAI LLM provider implementation."""

import os
from typing import AsyncIterator, Iterator, Optional

from openai import AsyncOpenAI, OpenAI

from cot_reasoner.providers.base import BaseLLMProvider, LLMResponse


class OpenAIProvider(BaseLLMProvider):
    """OpenAI API provider for GPT models."""

    DEFAULT_MODEL = "gpt-4o"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs,
    ):
        """Initialize OpenAI provider.

        Args:
            model: OpenAI model to use (default: gpt-4o)
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional options
        """
        super().__init__(model, api_key, temperature, max_tokens, **kwargs)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY environment variable.")

        self._client = OpenAI(api_key=self.api_key)
        self._async_client = AsyncOpenAI(api_key=self.api_key)

    @property
    def provider_name(self) -> str:
        return "openai"

    def _build_messages(
        self, prompt: str, system_prompt: Optional[str] = None
    ) -> list[dict]:
        """Build message list for API call."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> LLMResponse:
        """Generate a response synchronously."""
        messages = self._build_messages(prompt, system_prompt)

        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
        )

        choice = response.choices[0]
        usage = response.usage

        return LLMResponse(
            content=choice.message.content or "",
            model=response.model,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            total_tokens=usage.total_tokens if usage else 0,
            finish_reason=choice.finish_reason or "",
        )

    async def generate_async(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> LLMResponse:
        """Generate a response asynchronously."""
        messages = self._build_messages(prompt, system_prompt)

        response = await self._async_client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
        )

        choice = response.choices[0]
        usage = response.usage

        return LLMResponse(
            content=choice.message.content or "",
            model=response.model,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            total_tokens=usage.total_tokens if usage else 0,
            finish_reason=choice.finish_reason or "",
        )

    def stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> Iterator[str]:
        """Stream a response synchronously."""
        messages = self._build_messages(prompt, system_prompt)

        stream = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            stream=True,
        )

        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    async def stream_async(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        """Stream a response asynchronously."""
        messages = self._build_messages(prompt, system_prompt)

        stream = await self._async_client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            stream=True,
        )

        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
