"""
UCB (Upper Confidence Bound) Selector for SIGR Framework

Implements UCB1 algorithm for intelligent exploration-exploitation trade-off
when selecting discrete parameter values.
"""

import logging
import math
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)


class UCBSelector:
    """
    UCB1 algorithm for selecting discrete parameter values.

    Balances exploration (trying less-tested values) with exploitation
    (using values that performed well historically).
    """

    def __init__(self, c: float = 1.414):
        """
        Initialize the UCB selector.

        Args:
            c: Exploration constant. Higher values encourage more exploration.
               Default is sqrt(2) which is optimal for many scenarios.
        """
        self.c = c
        # arms[param][value] = (total_reward, count)
        self.arms: Dict[str, Dict[str, Tuple[float, int]]] = defaultdict(dict)
        self.total_pulls = 0

        logger.info(f"UCBSelector initialized with c={c}")

    def _normalize_value(self, value: Any) -> str:
        """Convert parameter value to string key."""
        if isinstance(value, list):
            import json
            return json.dumps(sorted(value))
        elif isinstance(value, bool):
            return str(value).lower()
        else:
            return str(value)

    def select(
        self,
        param: str,
        possible_values: List[Any],
        exclude: Optional[List[Any]] = None
    ) -> Any:
        """
        Select the best value for a parameter using UCB1.

        Args:
            param: Parameter name
            possible_values: List of possible values
            exclude: Values to exclude from selection

        Returns:
            Selected value (original type, not normalized)
        """
        if not possible_values:
            raise ValueError(f"No possible values for parameter {param}")

        exclude_keys = set()
        if exclude:
            exclude_keys = {self._normalize_value(v) for v in exclude}

        # Filter available values
        available = [(v, self._normalize_value(v)) for v in possible_values
                     if self._normalize_value(v) not in exclude_keys]

        if not available:
            logger.warning(f"All values excluded for {param}, returning first")
            return possible_values[0]

        # Initialize arms for new values
        for _, value_key in available:
            if value_key not in self.arms[param]:
                self.arms[param][value_key] = (0.0, 0)

        # Select using UCB1
        best_value = None
        best_original = None
        best_ucb = float('-inf')

        for original_value, value_key in available:
            total_reward, count = self.arms[param][value_key]

            if count == 0:
                # Prioritize unexplored values
                logger.debug(f"UCB: {param}={value_key} unexplored, selecting")
                return original_value

            # UCB1 formula: mean + c * sqrt(ln(total_pulls) / count)
            mean_reward = total_reward / count
            exploration_bonus = self.c * math.sqrt(
                math.log(self.total_pulls + 1) / count
            )
            ucb = mean_reward + exploration_bonus

            if ucb > best_ucb:
                best_ucb = ucb
                best_value = value_key
                best_original = original_value

        logger.debug(
            f"UCB selected {param}={best_value} with UCB={best_ucb:.4f}"
        )
        return best_original

    def update(self, param: str, value: Any, reward: float):
        """
        Update the arm statistics after receiving reward.

        Args:
            param: Parameter name
            value: Value that was used
            reward: Reward received (normalized to [0, 1] for best results)
        """
        value_key = self._normalize_value(value)

        # Initialize if needed
        if value_key not in self.arms[param]:
            self.arms[param][value_key] = (0.0, 0)

        total_reward, count = self.arms[param][value_key]
        self.arms[param][value_key] = (total_reward + reward, count + 1)
        self.total_pulls += 1

        logger.debug(
            f"UCB updated {param}={value_key}: "
            f"reward={reward:.4f}, count={count+1}, total_pulls={self.total_pulls}"
        )

    def get_statistics(self, param: str) -> Dict[str, Dict[str, float]]:
        """
        Get statistics for all values of a parameter.

        Args:
            param: Parameter name

        Returns:
            Dictionary mapping value to {mean, count, ucb}
        """
        stats = {}
        for value_key, (total_reward, count) in self.arms[param].items():
            if count > 0:
                mean = total_reward / count
                ucb = mean + self.c * math.sqrt(
                    math.log(self.total_pulls + 1) / count
                )
            else:
                mean = 0.0
                ucb = float('inf')

            stats[value_key] = {
                'mean': mean,
                'count': count,
                'ucb': ucb,
            }
        return stats

    def get_best_arm(self, param: str) -> Optional[Tuple[str, float]]:
        """
        Get the best arm (highest mean reward) for a parameter.

        Args:
            param: Parameter name

        Returns:
            (value_key, mean_reward) or None if no data
        """
        if param not in self.arms:
            return None

        best_value = None
        best_mean = float('-inf')

        for value_key, (total_reward, count) in self.arms[param].items():
            if count > 0:
                mean = total_reward / count
                if mean > best_mean:
                    best_mean = mean
                    best_value = value_key

        if best_value is None:
            return None
        return (best_value, best_mean)

    def get_exploration_priority(self, param: str) -> List[Tuple[str, int]]:
        """
        Get values sorted by exploration priority (least explored first).

        Args:
            param: Parameter name

        Returns:
            List of (value_key, count) sorted by count ascending
        """
        if param not in self.arms:
            return []

        values_with_counts = [
            (value_key, count)
            for value_key, (_, count) in self.arms[param].items()
        ]
        return sorted(values_with_counts, key=lambda x: x[1])

    def reset(self):
        """Reset all arm statistics."""
        self.arms.clear()
        self.total_pulls = 0
        logger.info("UCBSelector reset")


class MultiArmedBandit:
    """
    Multi-armed bandit that manages UCB selectors for multiple parameters.

    Provides a unified interface for selecting and updating multiple
    discrete parameters simultaneously.
    """

    # Parameters suitable for UCB selection
    DISCRETE_PARAMS = {
        'sampling': ['top_k', 'random', 'weighted'],
        'description_length': ['short', 'medium', 'long'],
        'description_focus': ['balanced', 'network', 'function', 'phenotype'],
        'context_window': ['minimal', 'local', 'extended', 'full'],
        'prompt_style': ['analytical', 'narrative', 'structured', 'comparative'],
        'feature_selection': ['all', 'essential', 'diverse', 'task_specific'],
        'max_hops': [1, 2, 3],
        'generation_passes': [1, 2, 3],
        'include_statistics': [True, False],
    }

    def __init__(self, c: float = 1.414):
        """
        Initialize the multi-armed bandit.

        Args:
            c: Exploration constant for UCB
        """
        self.selector = UCBSelector(c=c)

    def select_all(
        self,
        current_strategy: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Select values for all discrete parameters.

        Args:
            current_strategy: Current strategy (for context)

        Returns:
            Dictionary with selected values for each parameter
        """
        selected = {}
        for param, possible_values in self.DISCRETE_PARAMS.items():
            selected[param] = self.selector.select(param, possible_values)
        return selected

    def update_all(self, strategy: Dict[str, Any], reward: float):
        """
        Update arm statistics for all parameters in the strategy.

        Args:
            strategy: Strategy dictionary
            reward: Reward received (should be normalized to [0, 1])
        """
        for param in self.DISCRETE_PARAMS:
            if param in strategy:
                self.selector.update(param, strategy[param], reward)

    def get_guidance(self) -> str:
        """
        Generate guidance text for strategy generation.

        Returns:
            Formatted string with UCB recommendations
        """
        lines = ["UCB-based parameter recommendations:"]

        for param in self.DISCRETE_PARAMS:
            best = self.selector.get_best_arm(param)
            if best:
                value_key, mean_reward = best
                lines.append(f"- {param}: best={value_key} (mean={mean_reward:.3f})")

            # Show least explored
            priority = self.selector.get_exploration_priority(param)
            if priority and priority[0][1] < 3:
                least_explored = priority[0][0]
                lines.append(f"  (consider exploring: {least_explored})")

        return "\n".join(lines)

    def reset(self):
        """Reset all bandit statistics."""
        self.selector.reset()
