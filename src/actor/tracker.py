"""
Parameter Effect Tracker for SIGR Framework

Tracks the effect of each parameter value on reward to guide strategy optimization.
"""

import logging
from typing import Dict, Any, List, Tuple, Optional
from collections import defaultdict
import json

logger = logging.getLogger(__name__)


class ParameterEffectTracker:
    """
    Tracks the effect of each parameter value on reward.

    Uses exponential moving average to smooth effects over time,
    giving more weight to recent observations.
    """

    # Parameters to track (discrete and continuous)
    TRACKED_PARAMS = [
        'edge_types',        # List of edge types
        'max_hops',          # 1-3
        'sampling',          # top_k, random, weighted
        'max_neighbors',     # 10-200
        'description_length', # short, medium, long
        'description_focus',  # balanced, network, function, phenotype
        'context_window',     # minimal, local, extended, full
        'prompt_style',       # analytical, narrative, structured, comparative
        'feature_selection',  # all, essential, diverse, task_specific
        'generation_passes',  # 1-3
        'include_statistics', # True/False
    ]

    def __init__(self, decay: float = 0.8):
        """
        Initialize the tracker.

        Args:
            decay: Decay factor for exponential moving average (0-1).
                   Higher values give more weight to historical data.
        """
        self.decay = decay
        self.effects: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.counts: Dict[str, Dict[str, int]] = defaultdict(dict)
        self.last_strategy: Optional[Dict[str, Any]] = None
        self.last_reward: Optional[float] = None

        logger.info(f"ParameterEffectTracker initialized with decay={decay}")

    def _normalize_value(self, value: Any) -> str:
        """Convert parameter value to string key."""
        if isinstance(value, list):
            return json.dumps(sorted(value))
        elif isinstance(value, dict):
            return json.dumps(value, sort_keys=True)
        elif isinstance(value, bool):
            return str(value).lower()
        else:
            return str(value)

    def record(self, strategy: Dict[str, Any], reward: float):
        """
        Record the effect of a strategy on reward.

        Args:
            strategy: Current strategy dictionary
            reward: Reward received for this strategy
        """
        # Calculate delta reward (improvement over last iteration)
        if self.last_reward is not None:
            delta = reward - self.last_reward
        else:
            delta = reward  # First iteration, use absolute reward

        # Update effects for each tracked parameter
        for param in self.TRACKED_PARAMS:
            if param not in strategy:
                continue

            value = strategy[param]
            value_key = self._normalize_value(value)

            # Get current effect estimate
            old_effect = self.effects[param].get(value_key, 0.0)
            old_count = self.counts[param].get(value_key, 0)

            # Update with exponential moving average
            if old_count == 0:
                new_effect = delta
            else:
                new_effect = self.decay * old_effect + (1 - self.decay) * delta

            self.effects[param][value_key] = new_effect
            self.counts[param][value_key] = old_count + 1

            logger.debug(
                f"Updated effect for {param}={value_key}: "
                f"{old_effect:.4f} -> {new_effect:.4f} (delta={delta:.4f})"
            )

        # Store for next iteration
        self.last_strategy = strategy.copy()
        self.last_reward = reward

    def get_best_values(self, param: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Get the top-k best values for a parameter based on historical effect.

        Args:
            param: Parameter name
            top_k: Number of values to return

        Returns:
            List of (value, effect) tuples, sorted by effect descending
        """
        if param not in self.effects:
            return []

        sorted_effects = sorted(
            self.effects[param].items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_effects[:top_k]

    def get_worst_values(self, param: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Get the top-k worst values for a parameter.

        Args:
            param: Parameter name
            top_k: Number of values to return

        Returns:
            List of (value, effect) tuples, sorted by effect ascending
        """
        if param not in self.effects:
            return []

        sorted_effects = sorted(
            self.effects[param].items(),
            key=lambda x: x[1]
        )
        return sorted_effects[:top_k]

    def get_recommendation(self, param: str) -> Optional[str]:
        """
        Get the recommended value for a parameter.

        Args:
            param: Parameter name

        Returns:
            Best value key or None if no data
        """
        best = self.get_best_values(param, top_k=1)
        if best:
            return best[0][0]
        return None

    def get_unexplored_values(
        self,
        param: str,
        all_values: List[Any],
        min_count: int = 2
    ) -> List[str]:
        """
        Get values that haven't been explored enough.

        Args:
            param: Parameter name
            all_values: All possible values for this parameter
            min_count: Minimum count to be considered explored

        Returns:
            List of value keys that need more exploration
        """
        unexplored = []
        for value in all_values:
            value_key = self._normalize_value(value)
            count = self.counts[param].get(value_key, 0)
            if count < min_count:
                unexplored.append(value_key)
        return unexplored

    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all tracked effects.

        Returns:
            Dictionary with effect summaries for each parameter
        """
        summary = {}
        for param in self.TRACKED_PARAMS:
            if param in self.effects and self.effects[param]:
                best = self.get_best_values(param, top_k=3)
                worst = self.get_worst_values(param, top_k=3)
                summary[param] = {
                    'best': best,
                    'worst': worst,
                    'total_observations': sum(self.counts[param].values()),
                }
        return summary

    def get_guidance_for_llm(self) -> str:
        """
        Generate guidance text for the LLM based on tracked effects.

        Returns:
            Formatted string with parameter recommendations
        """
        lines = ["Based on historical performance:"]

        for param in self.TRACKED_PARAMS:
            if param not in self.effects or not self.effects[param]:
                continue

            best = self.get_best_values(param, top_k=2)
            worst = self.get_worst_values(param, top_k=2)

            if best and best[0][1] > 0:
                lines.append(
                    f"- {param}: {best[0][0]} worked well (effect: +{best[0][1]:.3f})"
                )
            if worst and worst[0][1] < 0:
                lines.append(
                    f"- {param}: avoid {worst[0][0]} (effect: {worst[0][1]:.3f})"
                )

        if len(lines) == 1:
            return "No historical guidance available yet."

        return "\n".join(lines)

    def reset(self):
        """Reset all tracked effects."""
        self.effects.clear()
        self.counts.clear()
        self.last_strategy = None
        self.last_reward = None
        logger.info("ParameterEffectTracker reset")
