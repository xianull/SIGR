"""Actor module for SIGR framework.

Implements Actor-Critic based policy optimization using LLM reflection.
- Actor: LLM-based strategy proposer (Computational Biologist paradigm)
- Critic: History-based value estimator for advantage computation
- HypothesisLedger: Scientific hypothesis tracking for Bio-CoT
"""

from .strategy import Strategy, StrategyConfig, get_default_strategy
from .prompts import TASK_INITIAL_PROMPTS, TASK_EDGE_PRIORITIES, MODE_CONSTRAINTS
from .agent import ActorAgent, ExperimentEntry, ExperimentLog, HistoryEntry
from .critic import SimpleCritic, GAECritic, StateValue, AdvantageEstimate
from .hypothesis import HypothesisLedger, Hypothesis, HypothesisStatus
from .knowledge import get_task_biological_context, get_domain_knowledge

__all__ = [
    # Strategy
    "Strategy",
    "StrategyConfig",
    "get_default_strategy",
    # Prompts
    "TASK_INITIAL_PROMPTS",
    "TASK_EDGE_PRIORITIES",
    "MODE_CONSTRAINTS",
    # Agent
    "ActorAgent",
    "ExperimentEntry",
    "ExperimentLog",
    "HistoryEntry",
    # Critic
    "SimpleCritic",
    "GAECritic",
    "StateValue",
    "AdvantageEstimate",
    # Hypothesis (Bio-CoT)
    "HypothesisLedger",
    "Hypothesis",
    "HypothesisStatus",
    # Knowledge
    "get_task_biological_context",
    "get_domain_knowledge",
]
