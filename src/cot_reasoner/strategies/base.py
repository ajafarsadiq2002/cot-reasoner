"""Base class for Chain of Thought strategies."""

import re
from abc import ABC, abstractmethod
from typing import Optional

from cot_reasoner.core.chain import ReasoningChain
from cot_reasoner.providers.base import BaseLLMProvider


class BaseStrategy(ABC):
    """Abstract base class for Chain of Thought reasoning strategies.

    Strategies define how to:
    1. Format the prompt for the LLM
    2. Execute the reasoning process
    3. Parse the response into structured steps
    """

    def __init__(self, provider: BaseLLMProvider):
        """Initialize strategy with an LLM provider.

        Args:
            provider: The LLM provider to use for generation
        """
        self.provider = provider

    @property
    @abstractmethod
    def strategy_name(self) -> str:
        """Return the strategy name."""
        pass

    @abstractmethod
    def reason(self, query: str, context: Optional[str] = None, **kwargs) -> ReasoningChain:
        """Execute the reasoning process synchronously.

        Args:
            query: The problem or question to reason about
            context: Optional conversation context from memory
            **kwargs: Additional strategy-specific options

        Returns:
            ReasoningChain with steps and final answer
        """
        pass

    @abstractmethod
    async def reason_async(self, query: str, context: Optional[str] = None, **kwargs) -> ReasoningChain:
        """Execute the reasoning process asynchronously.

        Args:
            query: The problem or question to reason about
            context: Optional conversation context from memory
            **kwargs: Additional strategy-specific options

        Returns:
            ReasoningChain with steps and final answer
        """
        pass

    def parse_response(self, response: str, chain: ReasoningChain) -> ReasoningChain:
        """Parse LLM response into reasoning steps and answer.

        This default implementation looks for:
        - Lines starting with "Step N:" as reasoning steps
        - Lines starting with "Answer:" as the final answer

        Override this method for custom parsing logic.

        Args:
            response: The raw LLM response
            chain: The ReasoningChain to populate

        Returns:
            Updated ReasoningChain with parsed steps and answer
        """
        lines = response.strip().split("\n")
        current_step_content = []
        current_step_num = 0

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check for step pattern: "Step N:" or "N."
            step_match = re.match(r"^(?:Step\s*)?(\d+)[.:\)]\s*(.*)$", line, re.IGNORECASE)
            answer_match = re.match(r"^(?:Final\s+)?Answer[:\s]+(.*)$", line, re.IGNORECASE)

            if answer_match:
                # Save any pending step
                if current_step_content:
                    chain.add_step(" ".join(current_step_content))
                    current_step_content = []

                # Extract answer
                answer_text = answer_match.group(1).strip()
                # Continue capturing answer if it spans multiple lines
                chain.set_answer(answer_text, confidence=0.9)

            elif step_match:
                # Save previous step if exists
                if current_step_content:
                    chain.add_step(" ".join(current_step_content))

                # Start new step
                current_step_num = int(step_match.group(1))
                step_text = step_match.group(2).strip()
                current_step_content = [step_text] if step_text else []

            elif current_step_num > 0:
                # Continue current step
                current_step_content.append(line)

            elif not chain.steps and not chain.answer:
                # No steps found yet, treat as reasoning content
                current_step_content.append(line)

        # Save any remaining step content
        if current_step_content:
            chain.add_step(" ".join(current_step_content))

        # If no explicit answer found, use last meaningful content
        if not chain.answer and chain.steps:
            last_step = chain.steps[-1].content
            # Check if last step contains an answer
            if "=" in last_step or "is" in last_step.lower():
                chain.set_answer(last_step, confidence=0.7)

        return chain

    def _create_chain(self, query: str) -> ReasoningChain:
        """Create a new ReasoningChain for the query."""
        return ReasoningChain(
            query=query,
            provider=self.provider.provider_name,
            model=self.provider.model,
            strategy=self.strategy_name,
        )
