"""Self-consistency Chain of Thought strategy implementation."""

from collections import Counter
from typing import Optional

from cot_reasoner.core.chain import ReasoningChain, ReasoningStep
from cot_reasoner.core.prompts import SELF_CONSISTENCY_PROMPT, SELF_CONSISTENCY_SYSTEM
from cot_reasoner.strategies.base import BaseStrategy


class SelfConsistencyStrategy(BaseStrategy):
    """Self-consistency Chain of Thought reasoning strategy.

    This strategy generates multiple reasoning paths and selects the most
    consistent answer through majority voting.

    Key characteristics:
    - Generates N independent reasoning chains
    - Extracts answers from each chain
    - Uses majority voting to select final answer
    - Higher confidence when multiple paths agree
    - More robust for complex problems

    Reference: "Self-Consistency Improves Chain of Thought Reasoning in Language Models"
    (Wang et al., 2022)
    """

    def __init__(self, provider, num_samples: int = 3, temperature: float = 0.7):
        """Initialize self-consistency strategy.

        Args:
            provider: The LLM provider to use
            num_samples: Number of reasoning paths to generate (default: 3)
            temperature: Temperature for diverse sampling (default: 0.7)
        """
        super().__init__(provider)
        self.num_samples = num_samples
        self.sample_temperature = temperature

    @property
    def strategy_name(self) -> str:
        return "self_consistency"

    def _build_prompt(self, query: str, context: Optional[str] = None) -> str:
        """Build prompt with optional conversation context."""
        if context:
            return f"{context}\nCurrent question: {SELF_CONSISTENCY_PROMPT.format(query=query)}"
        return SELF_CONSISTENCY_PROMPT.format(query=query)

    def reason(self, query: str, context: Optional[str] = None, **kwargs) -> ReasoningChain:
        """Execute self-consistency reasoning synchronously."""
        chain = self._create_chain(query)
        prompt = self._build_prompt(query, context)

        # Generate multiple reasoning paths
        answers = []
        all_chains = []
        total_tokens = 0

        for i in range(self.num_samples):
            response = self.provider.generate(
                prompt=prompt,
                system_prompt=SELF_CONSISTENCY_SYSTEM,
                temperature=self.sample_temperature,
                **kwargs,
            )

            total_tokens += response.total_tokens

            # Parse this response into a temporary chain
            temp_chain = ReasoningChain(query=query)
            temp_chain = self.parse_response(response.content, temp_chain)
            all_chains.append(temp_chain)

            if temp_chain.answer:
                # Normalize answer for comparison
                normalized_answer = self._normalize_answer(temp_chain.answer)
                answers.append((normalized_answer, temp_chain.answer, temp_chain))

        chain.total_tokens = total_tokens

        # Perform majority voting
        if answers:
            final_answer, confidence = self._majority_vote(answers)
            chain.set_answer(final_answer, confidence)

            # Add summary steps showing the voting process
            chain.add_step(f"Generated {self.num_samples} independent reasoning paths")

            # Count votes
            vote_counts = Counter(a[0] for a in answers)
            votes_summary = ", ".join(f"'{ans}': {count}" for ans, count in vote_counts.most_common())
            chain.add_step(f"Answer distribution: {votes_summary}")

            chain.add_step(f"Selected answer '{final_answer}' with {confidence:.0%} confidence")

            # Store detailed reasoning paths in metadata
            chain.metadata["reasoning_paths"] = [
                {"steps": [s.to_dict() for s in c.steps], "answer": c.answer}
                for c in all_chains
            ]

        return chain

    async def reason_async(self, query: str, context: Optional[str] = None, **kwargs) -> ReasoningChain:
        """Execute self-consistency reasoning asynchronously."""
        import asyncio

        chain = self._create_chain(query)
        prompt = self._build_prompt(query, context)

        # Generate multiple reasoning paths concurrently
        async def generate_path():
            response = await self.provider.generate_async(
                prompt=prompt,
                system_prompt=SELF_CONSISTENCY_SYSTEM,
                temperature=self.sample_temperature,
                **kwargs,
            )
            temp_chain = ReasoningChain(query=query)
            temp_chain = self.parse_response(response.content, temp_chain)
            return response, temp_chain

        # Run all generations concurrently
        results = await asyncio.gather(*[generate_path() for _ in range(self.num_samples)])

        answers = []
        all_chains = []
        total_tokens = 0

        for response, temp_chain in results:
            total_tokens += response.total_tokens
            all_chains.append(temp_chain)

            if temp_chain.answer:
                normalized_answer = self._normalize_answer(temp_chain.answer)
                answers.append((normalized_answer, temp_chain.answer, temp_chain))

        chain.total_tokens = total_tokens

        # Perform majority voting
        if answers:
            final_answer, confidence = self._majority_vote(answers)
            chain.set_answer(final_answer, confidence)

            chain.add_step(f"Generated {self.num_samples} independent reasoning paths")

            vote_counts = Counter(a[0] for a in answers)
            votes_summary = ", ".join(f"'{ans}': {count}" for ans, count in vote_counts.most_common())
            chain.add_step(f"Answer distribution: {votes_summary}")

            chain.add_step(f"Selected answer '{final_answer}' with {confidence:.0%} confidence")

            chain.metadata["reasoning_paths"] = [
                {"steps": [s.to_dict() for s in c.steps], "answer": c.answer}
                for c in all_chains
            ]

        return chain

    def _normalize_answer(self, answer: str) -> str:
        """Normalize answer for comparison.

        Handles variations like:
        - Different whitespace
        - Case differences
        - Common formatting differences
        """
        # Basic normalization
        normalized = answer.lower().strip()

        # Remove common prefixes
        prefixes = ["the answer is", "therefore", "so", "thus", "hence"]
        for prefix in prefixes:
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix):].strip()

        # Remove trailing punctuation
        normalized = normalized.rstrip(".,!?")

        return normalized

    def _majority_vote(
        self, answers: list[tuple[str, str, ReasoningChain]]
    ) -> tuple[str, float]:
        """Perform majority voting on answers.

        Args:
            answers: List of (normalized_answer, original_answer, chain) tuples

        Returns:
            Tuple of (selected_answer, confidence_score)
        """
        if not answers:
            return "", 0.0

        # Count normalized answers
        vote_counts = Counter(a[0] for a in answers)
        most_common = vote_counts.most_common(1)[0]
        winning_normalized = most_common[0]
        winning_count = most_common[1]

        # Get original answer for the winning normalized version
        winning_answer = next(a[1] for a in answers if a[0] == winning_normalized)

        # Calculate confidence as proportion of votes
        confidence = winning_count / len(answers)

        return winning_answer, confidence
