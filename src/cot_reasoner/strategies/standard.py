"""Standard Chain of Thought strategy implementation."""

from typing import Optional

from cot_reasoner.core.chain import ReasoningChain
from cot_reasoner.core.prompts import STANDARD_COT_PROMPT, STANDARD_COT_SYSTEM
from cot_reasoner.strategies.base import BaseStrategy


class StandardCoTStrategy(BaseStrategy):
    """Standard Chain of Thought reasoning strategy.

    This strategy uses explicit prompting to encourage step-by-step reasoning:
    "Let's think through this step by step..."

    The model is guided to:
    1. Break down the problem into numbered steps
    2. Show reasoning for each step
    3. Conclude with a clear final answer
    """

    @property
    def strategy_name(self) -> str:
        return "standard"

    def _build_prompt(self, query: str, context: Optional[str] = None) -> str:
        """Build prompt with optional conversation context."""
        if context:
            return f"{context}\nCurrent question: {STANDARD_COT_PROMPT.format(query=query)}"
        return STANDARD_COT_PROMPT.format(query=query)

    def reason(self, query: str, context: Optional[str] = None, **kwargs) -> ReasoningChain:
        """Execute standard CoT reasoning synchronously."""
        chain = self._create_chain(query)

        # Format the prompt with context
        prompt = self._build_prompt(query, context)

        # Generate response
        response = self.provider.generate(
            prompt=prompt,
            system_prompt=STANDARD_COT_SYSTEM,
            **kwargs,
        )

        # Update token count
        chain.total_tokens = response.total_tokens

        # Parse response into steps
        chain = self.parse_response(response.content, chain)

        return chain

    async def reason_async(self, query: str, context: Optional[str] = None, **kwargs) -> ReasoningChain:
        """Execute standard CoT reasoning asynchronously."""
        chain = self._create_chain(query)

        # Format the prompt with context
        prompt = self._build_prompt(query, context)

        # Generate response
        response = await self.provider.generate_async(
            prompt=prompt,
            system_prompt=STANDARD_COT_SYSTEM,
            **kwargs,
        )

        # Update token count
        chain.total_tokens = response.total_tokens

        # Parse response into steps
        chain = self.parse_response(response.content, chain)

        return chain
