"""Zero-shot Chain of Thought strategy implementation."""

from typing import Optional

from cot_reasoner.core.chain import ReasoningChain
from cot_reasoner.core.prompts import ZERO_SHOT_COT_PROMPT, ZERO_SHOT_COT_SYSTEM
from cot_reasoner.strategies.base import BaseStrategy


class ZeroShotCoTStrategy(BaseStrategy):
    """Zero-shot Chain of Thought reasoning strategy.

    This strategy uses minimal prompting with just "Let's think step by step"
    appended to the query. No examples or detailed instructions are provided.

    Key characteristics:
    - Simple prompt augmentation
    - Relies on model's inherent reasoning capabilities
    - Works well for straightforward problems
    - Lower token usage compared to few-shot approaches
    """

    @property
    def strategy_name(self) -> str:
        return "zero_shot"

    def _build_prompt(self, query: str, context: Optional[str] = None) -> str:
        """Build prompt with optional conversation context."""
        if context:
            return f"{context}\nCurrent question: {ZERO_SHOT_COT_PROMPT.format(query=query)}"
        return ZERO_SHOT_COT_PROMPT.format(query=query)

    def reason(self, query: str, context: Optional[str] = None, **kwargs) -> ReasoningChain:
        """Execute zero-shot CoT reasoning synchronously."""
        chain = self._create_chain(query)

        # Format the prompt with context
        prompt = self._build_prompt(query, context)

        # Generate response
        response = self.provider.generate(
            prompt=prompt,
            system_prompt=ZERO_SHOT_COT_SYSTEM,
            **kwargs,
        )

        # Update token count
        chain.total_tokens = response.total_tokens

        # Parse response into steps
        chain = self.parse_response(response.content, chain)

        return chain

    async def reason_async(self, query: str, context: Optional[str] = None, **kwargs) -> ReasoningChain:
        """Execute zero-shot CoT reasoning asynchronously."""
        chain = self._create_chain(query)

        # Format the prompt with context
        prompt = self._build_prompt(query, context)

        # Generate response
        response = await self.provider.generate_async(
            prompt=prompt,
            system_prompt=ZERO_SHOT_COT_SYSTEM,
            **kwargs,
        )

        # Update token count
        chain.total_tokens = response.total_tokens

        # Parse response into steps
        chain = self.parse_response(response.content, chain)

        return chain
