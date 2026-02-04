"""SIGR Configuration Module."""

from .client import LLMClient, DualModelClient, get_llm_client, get_dual_model_client
from .reward_config import (
    RewardWeights,
    get_reward_weights,
    TASK_REWARD_WEIGHTS,
    AdaptiveRewardConfig,
    AdaptiveRewardManager,
)

__all__ = [
    # LLM clients
    "LLMClient",
    "DualModelClient",
    "get_llm_client",
    "get_dual_model_client",
    # Reward configuration
    "RewardWeights",
    "get_reward_weights",
    "TASK_REWARD_WEIGHTS",
    "AdaptiveRewardConfig",
    "AdaptiveRewardManager",
]
