"""Chain of Thought reasoning strategies."""

from cot_reasoner.strategies.base import BaseStrategy
from cot_reasoner.strategies.standard import StandardCoTStrategy
from cot_reasoner.strategies.zero_shot import ZeroShotCoTStrategy
from cot_reasoner.strategies.self_consistency import SelfConsistencyStrategy

__all__ = [
    "BaseStrategy",
    "StandardCoTStrategy",
    "ZeroShotCoTStrategy",
    "SelfConsistencyStrategy",
]
