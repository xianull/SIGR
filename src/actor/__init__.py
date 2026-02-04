"""Actor module for SIGR framework.

Implements Actor-Critic based policy optimization using LLM reflection.
- Actor: LLM-based strategy proposer
- Critic: History-based value estimator for advantage computation
"""

from .strategy import Strategy, StrategyConfig, get_default_strategy
from .prompts import TASK_INITIAL_PROMPTS, TASK_EDGE_PRIORITIES
from .agent import ActorAgent
from .critic import SimpleCritic, GAECritic, StateValue, AdvantageEstimate

__all__ = [
    "Strategy",
    "StrategyConfig",
    "get_default_strategy",
    "TASK_INITIAL_PROMPTS",
    "TASK_EDGE_PRIORITIES",
    "ActorAgent",
    "SimpleCritic",
    "GAECritic",
    "StateValue",
    "AdvantageEstimate",
]
