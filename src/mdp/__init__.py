"""
MDP Module for SIGR Framework

Provides explicit MDP formalization including:
- State representation (MDPState)
- Reward computation (RewardComputer)
- Exploration scheduling (ExplorationScheduler)
- History/trend analysis (TrendAnalyzer)

Theoretical foundations:
- Relative rewards based on RLHF reward shaping
- Exploration via UCB and ε-greedy strategies
- Trend analysis for contextual bandit-style decisions
"""

from .state import MDPState
from .reward import RewardComputer, RewardResult
from .exploration import ExplorationScheduler
from .history_analyzer import TrendAnalyzer, TrendAnalysis

__all__ = [
    'MDPState',
    'RewardComputer',
    'RewardResult',
    'ExplorationScheduler',
    'TrendAnalyzer',
    'TrendAnalysis',
]
