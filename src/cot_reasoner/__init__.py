"""
CoT Reasoner - A Chain of Thought Reasoning Library

A Python library for implementing Chain of Thought reasoning with multiple LLM providers.
"""

from cot_reasoner.core.reasoner import Reasoner
from cot_reasoner.core.chain import ReasoningStep, ReasoningChain
from cot_reasoner.core.memory import ConversationMemory
from cot_reasoner.providers.base import BaseLLMProvider
from cot_reasoner.strategies.base import BaseStrategy

__version__ = "0.1.0"
__all__ = [
    "Reasoner",
    "ReasoningStep",
    "ReasoningChain",
    "ConversationMemory",
    "BaseLLMProvider",
    "BaseStrategy",
]
