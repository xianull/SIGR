"""
Reward Computation for SIGR MDP Framework

Implements relative reward calculation based on RLHF reward shaping theory.
Key principle: Δ = metric_t - metric_{t-1} provides clearer learning signal
than absolute metric values.

Components:
- Relative reward: improvement over previous iteration (normalized to [-1, 1])
- Baseline reward: comparison to moving average (normalized to [-1, 1])
- Absolute performance: scaled to [-1, 1]
- Dynamic plateau penalty: increases with plateau duration

Theoretical basis:
- Reward shaping (Ng et al., 1999)
- RLHF relative comparisons (Christiano et al., 2017)
- All components normalized to [-1, 1] for consistent gradient signals

Now supports:
- Configurable weights via RewardWeights
- Task-specific default weights
- Adaptive weight adjustment
- Dynamic plateau penalty based on duration
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np

# Import configurable weights
try:
    from configs.reward_config import (
        RewardWeights,
        get_reward_weights,
        AdaptiveRewardManager,
        AdaptiveRewardConfig
    )
    HAS_REWARD_CONFIG = True
except ImportError:
    HAS_REWARD_CONFIG = False


logger = logging.getLogger(__name__)


@dataclass
class RewardResult:
    """Result of reward computation with breakdown."""
    total_reward: float  # Normalized to [-1, 1]
    relative_reward: float  # metric_t - metric_{t-1}, normalized
    baseline_reward: float  # metric_t - baseline, normalized
    absolute_reward: float  # Absolute performance scaled to [-1, 1]
    plateau_penalty: float  # Dynamic penalty for stagnation
    raw_metric: float  # Original absolute metric (for best_strategy selection)
    weights_used: Optional[dict] = None  # weights used for this computation
    plateau_duration: int = 0  # Number of consecutive plateau iterations

    def to_dict(self):
        """Convert to dictionary for logging."""
        result = {
            'total_reward': self.total_reward,
            'relative_reward': self.relative_reward,
            'baseline_reward': self.baseline_reward,
            'absolute_reward': self.absolute_reward,
            'plateau_penalty': self.plateau_penalty,
            'raw_metric': self.raw_metric,
            'plateau_duration': self.plateau_duration,
        }
        if self.weights_used:
            result['weights_used'] = self.weights_used
        return result


class RewardComputer:
    """
    Computes normalized rewards for MDP-based strategy optimization.

    All reward components are normalized to [-1, 1] for consistent gradient signals.

    Components:
    - Relative reward: (metric_t - metric_{t-1}) / metric_range, clipped to [-1, 1]
    - Baseline reward: (metric_t - baseline) / metric_range, clipped to [-1, 1]
    - Absolute reward: 2 * metric - 1 (converts [0,1] to [-1,1]) with non-linear bonus for high scores
    - Dynamic plateau penalty: base_penalty * (1 + duration * escalation_rate)
    - SOTA protection: guarantees positive reward when maintaining best performance

    Theoretical basis:
    - Reward shaping (Ng et al., 1999)
    - RLHF relative comparisons (Christiano et al., 2017)
    - Normalized rewards for stable learning

    Default weights (emphasizing absolute performance):
    - baseline_weight: 0.10
    - relative_weight: 0.10
    - raw_weight: 0.80
    """

    # Metric range for normalization (assuming metrics in [0, 1])
    METRIC_MIN = 0.0
    METRIC_MAX = 1.0

    # Non-linear scaling parameters for high scores
    HIGH_SCORE_THRESHOLD = 0.8  # Threshold for non-linear bonus
    HIGH_SCORE_SCALING = 5.0    # Scaling factor for bonus

    # SOTA protection parameters
    SOTA_TOLERANCE = 0.001      # Tolerance for SOTA comparison
    SOTA_MIN_REWARD = 0.1       # Minimum reward when at SOTA

    def __init__(
        self,
        # Legacy individual weight parameters (for backwards compatibility)
        baseline_weight: Optional[float] = None,
        relative_weight: Optional[float] = None,
        raw_weight: Optional[float] = None,
        improvement_bonus: Optional[float] = None,
        plateau_penalty: Optional[float] = None,
        # New configurable weights
        weights: Optional['RewardWeights'] = None,
        task_name: Optional[str] = None,
        # Adaptive configuration
        enable_adaptive: bool = False,
        adaptive_config: Optional['AdaptiveRewardConfig'] = None,
        # Other parameters
        baseline_window: int = 5,
        plateau_threshold: float = 0.01,  # 1% relative threshold
        use_relative_plateau_threshold: bool = True,  # Use relative instead of absolute
        # Dynamic plateau penalty parameters
        base_plateau_penalty: float = -0.05,
        plateau_escalation_rate: float = 0.3,
        max_plateau_penalty: float = -0.3,
        # Non-linear scaling
        enable_nonlinear_scaling: bool = True,
        # SOTA protection
        enable_sota_protection: bool = True,
    ):
        """
        Initialize RewardComputer with normalized reward calculation.

        Args:
            baseline_weight: Weight for baseline comparison (default 0.10)
            relative_weight: Weight for relative improvement (default 0.10)
            raw_weight: Weight for absolute performance (default 0.80)
            improvement_bonus: Deprecated, kept for compatibility
            plateau_penalty: Deprecated, use base_plateau_penalty instead
            weights: RewardWeights configuration object
            task_name: Task name for automatic weight selection
            enable_adaptive: Enable adaptive weight adjustment
            adaptive_config: Configuration for adaptive adjustment
            baseline_window: Number of iterations for moving average baseline
            plateau_threshold: Threshold for plateau detection (relative if use_relative_plateau_threshold=True)
            use_relative_plateau_threshold: If True, threshold is relative (0.01 = 1%), else absolute
            base_plateau_penalty: Starting penalty for plateau (-0.05)
            plateau_escalation_rate: Rate of penalty increase per plateau iteration
            max_plateau_penalty: Maximum penalty cap (-0.3)
            enable_nonlinear_scaling: Enable non-linear bonus for high scores (>0.8)
            enable_sota_protection: Enable SOTA protection (positive reward when at best)
        """
        # Non-linear scaling and SOTA protection flags
        self.enable_nonlinear_scaling = enable_nonlinear_scaling
        self.enable_sota_protection = enable_sota_protection
        self._best_metric: Optional[float] = None  # Track best metric for SOTA protection

        # Determine weights (priority: explicit weights > task_name > legacy > defaults)
        # New default weights: baseline=0.10, relative=0.10, raw=0.80 (emphasize absolute performance)
        if weights is not None and HAS_REWARD_CONFIG:
            self._weights = weights
        elif task_name is not None and HAS_REWARD_CONFIG:
            self._weights = get_reward_weights(task_name)
        elif any(w is not None for w in [baseline_weight, relative_weight, raw_weight]):
            if HAS_REWARD_CONFIG:
                self._weights = RewardWeights(
                    baseline_weight=baseline_weight if baseline_weight is not None else 0.10,
                    relative_weight=relative_weight if relative_weight is not None else 0.10,
                    raw_weight=raw_weight if raw_weight is not None else 0.80,
                    improvement_bonus=0.0,  # Deprecated
                    plateau_penalty=base_plateau_penalty,
                )
            else:
                self._weights = None
                self.baseline_weight = baseline_weight if baseline_weight is not None else 0.10
                self.relative_weight = relative_weight if relative_weight is not None else 0.10
                self.raw_weight = raw_weight if raw_weight is not None else 0.80
        else:
            if HAS_REWARD_CONFIG:
                self._weights = RewardWeights(
                    baseline_weight=0.10,  # Low - baseline comparison less important
                    relative_weight=0.10,  # Low - avoid over-penalizing small declines
                    raw_weight=0.80,       # High - emphasize absolute performance
                    improvement_bonus=0.0,
                    plateau_penalty=base_plateau_penalty,
                )
            else:
                self._weights = None
                self.baseline_weight = 0.10
                self.relative_weight = 0.10
                self.raw_weight = 0.80

        if self._weights is not None:
            self.baseline_weight = self._weights.baseline_weight
            self.relative_weight = self._weights.relative_weight
            self.raw_weight = self._weights.raw_weight

        self.baseline_window = baseline_window
        self.plateau_threshold = plateau_threshold
        self.use_relative_plateau_threshold = use_relative_plateau_threshold

        # Dynamic plateau penalty configuration
        self.base_plateau_penalty = base_plateau_penalty
        self.plateau_escalation_rate = plateau_escalation_rate
        self.max_plateau_penalty = max_plateau_penalty

        # Track plateau duration across calls
        self._plateau_duration = 0

        # Adaptive reward manager
        self._adaptive_manager = None
        if enable_adaptive and HAS_REWARD_CONFIG and self._weights is not None:
            self._adaptive_manager = AdaptiveRewardManager(
                initial_weights=self._weights,
                config=adaptive_config
            )
            logger.info("Adaptive reward adjustment enabled")

        logger.info(
            f"RewardComputer initialized (normalized): "
            f"weights=(relative={self.relative_weight}, absolute={self.raw_weight}, baseline={self.baseline_weight}), "
            f"plateau_penalty_base={self.base_plateau_penalty}, escalation={self.plateau_escalation_rate}, "
            f"nonlinear_scaling={self.enable_nonlinear_scaling}, sota_protection={self.enable_sota_protection}"
        )

    def _compute_absolute_reward(self, metric: float) -> float:
        """
        Compute absolute reward with optional non-linear scaling for high scores.

        For metrics > 0.8, applies a quadratic bonus to incentivize pushing higher.
        Formula: linear + ((metric - 0.8)^2 * scaling_factor)

        Args:
            metric: Current metric value (in [0, 1])

        Returns:
            Absolute reward in [-1, 1]
        """
        metric_range = self.METRIC_MAX - self.METRIC_MIN
        linear = 2 * (metric - self.METRIC_MIN) / metric_range - 1

        if self.enable_nonlinear_scaling and metric > self.HIGH_SCORE_THRESHOLD:
            # Quadratic bonus for high scores
            # e.g., metric=0.85 → bonus ≈ 0.0125
            # e.g., metric=0.90 → bonus ≈ 0.05
            # e.g., metric=0.95 → bonus ≈ 0.1125
            bonus = ((metric - self.HIGH_SCORE_THRESHOLD) ** 2) * self.HIGH_SCORE_SCALING
            result = linear + bonus
            logger.debug(
                f"Non-linear scaling applied: metric={metric:.4f}, "
                f"linear={linear:.4f}, bonus={bonus:.4f}, result={result:.4f}"
            )
            return min(result, 1.0)

        return np.clip(linear, -1.0, 1.0)

    def update_best_metric(self, metric: float):
        """
        Update the best metric seen (for SOTA protection).

        Args:
            metric: Current metric value
        """
        if self._best_metric is None or metric > self._best_metric:
            self._best_metric = metric
            logger.debug(f"Best metric updated: {metric:.4f}")

    def update_weights_for_trend(self, metric: float, trend: str = 'unknown'):
        """
        Update weights based on current trend (if adaptive is enabled).

        Args:
            metric: Latest evaluation metric
            trend: Current performance trend

        Returns:
            True if weights were updated
        """
        if self._adaptive_manager is None:
            return False

        updated_weights = self._adaptive_manager.update(metric, trend)
        self._weights = updated_weights
        self.baseline_weight = updated_weights.baseline_weight
        self.relative_weight = updated_weights.relative_weight
        self.raw_weight = updated_weights.raw_weight
        self.improvement_bonus = updated_weights.improvement_bonus
        self.plateau_penalty = updated_weights.plateau_penalty
        return True

    def get_weights(self) -> dict:
        """Get current reward weights as dictionary."""
        return {
            'baseline_weight': self.baseline_weight,
            'relative_weight': self.relative_weight,
            'raw_weight': self.raw_weight,
            'base_plateau_penalty': self.base_plateau_penalty,
            'plateau_escalation_rate': self.plateau_escalation_rate,
        }

    def set_weights(
        self,
        baseline_weight: Optional[float] = None,
        relative_weight: Optional[float] = None,
        raw_weight: Optional[float] = None,
        improvement_bonus: Optional[float] = None,  # Deprecated, ignored
        plateau_penalty: Optional[float] = None,  # Deprecated, use base_plateau_penalty
        base_plateau_penalty: Optional[float] = None,
        plateau_escalation_rate: Optional[float] = None,
    ):
        """
        Manually set reward weights.

        Args:
            baseline_weight: Weight for baseline comparison
            relative_weight: Weight for relative improvement
            raw_weight: Weight for absolute performance
            improvement_bonus: Deprecated, ignored
            plateau_penalty: Deprecated, maps to base_plateau_penalty
            base_plateau_penalty: Base penalty for plateau
            plateau_escalation_rate: Rate of penalty increase
        """
        if baseline_weight is not None:
            self.baseline_weight = baseline_weight
        if relative_weight is not None:
            self.relative_weight = relative_weight
        if raw_weight is not None:
            self.raw_weight = raw_weight
        if base_plateau_penalty is not None:
            self.base_plateau_penalty = base_plateau_penalty
        elif plateau_penalty is not None:
            # Legacy compatibility
            self.base_plateau_penalty = plateau_penalty
        if plateau_escalation_rate is not None:
            self.plateau_escalation_rate = plateau_escalation_rate

        # Update weights object if available
        if HAS_REWARD_CONFIG:
            self._weights = RewardWeights(
                baseline_weight=self.baseline_weight,
                relative_weight=self.relative_weight,
                raw_weight=self.raw_weight,
                improvement_bonus=0.0,
                plateau_penalty=self.base_plateau_penalty,
            )

        logger.info(f"Reward weights updated: {self.get_weights()}")

    def compute(
        self,
        current_metric: float,
        history: Optional[List[float]] = None
    ) -> RewardResult:
        """
        Compute normalized reward based on current metric and history.

        All components are normalized to [-1, 1] for consistent learning signals.
        Includes non-linear scaling for high scores and SOTA protection.

        Args:
            current_metric: Current iteration's primary metric value (in [0, 1])
            history: List of previous metric values (oldest to newest)

        Returns:
            RewardResult with total reward and breakdown (all in [-1, 1])
        """
        metric_range = self.METRIC_MAX - self.METRIC_MIN

        # Update best metric tracking for SOTA protection
        self.update_best_metric(current_metric)

        if history is None or len(history) == 0:
            # First iteration: use weighted formula with relative=0, baseline=0
            absolute_reward = self._compute_absolute_reward(current_metric)

            # Apply same weighted formula as subsequent iterations
            # This ensures consistent reward scale across all iterations
            total_reward = (
                self.relative_weight * 0.0 +        # No history, relative improvement = 0
                self.raw_weight * absolute_reward + # Weighted absolute performance
                self.baseline_weight * 0.0          # No history, baseline comparison = 0
            )
            total_reward = np.clip(total_reward, -1.0, 1.0)

            self._plateau_duration = 0
            return RewardResult(
                total_reward=total_reward,
                relative_reward=0.0,
                baseline_reward=0.0,
                absolute_reward=absolute_reward,
                plateau_penalty=0.0,
                raw_metric=current_metric,
                weights_used=self.get_weights(),
                plateau_duration=0,
            )

        # 1. Compute relative reward (normalized improvement over previous)
        previous_metric = history[-1]
        relative_delta = current_metric - previous_metric
        relative_reward = relative_delta / metric_range
        relative_reward = np.clip(relative_reward, -1.0, 1.0)

        # 2. Compute baseline reward (normalized comparison to moving average)
        window = min(len(history), self.baseline_window)
        baseline = np.mean(history[-window:])
        baseline_delta = current_metric - baseline
        baseline_reward = baseline_delta / metric_range
        baseline_reward = np.clip(baseline_reward, -1.0, 1.0)

        # 3. Compute absolute reward with non-linear scaling
        absolute_reward = self._compute_absolute_reward(current_metric)

        # 4. Dynamic plateau penalty
        plateau_penalty = 0.0

        # Compute effective threshold (relative or absolute)
        if self.use_relative_plateau_threshold:
            # Relative threshold: e.g., 0.01 = 1% of previous metric
            effective_threshold = self.plateau_threshold * max(previous_metric, 0.01)
        else:
            effective_threshold = self.plateau_threshold

        # Plateau detection requires at least 2 history points
        if len(history) >= 2:
            is_plateau = abs(relative_delta) < effective_threshold

            if is_plateau:
                # Check if truly in plateau (low variance in recent history)
                recent_std = np.std(history[-min(3, len(history)):])
                # Use a slightly larger threshold for variance check
                variance_threshold = effective_threshold * 1.5
                if recent_std < variance_threshold:
                    self._plateau_duration += 1
                    # Dynamic penalty: increases with duration
                    plateau_penalty = self.base_plateau_penalty * (
                        1 + self._plateau_duration * self.plateau_escalation_rate
                    )
                    # Cap the penalty
                    plateau_penalty = max(plateau_penalty, self.max_plateau_penalty)
                    logger.debug(
                        f"Plateau detected (duration={self._plateau_duration}), "
                        f"penalty={plateau_penalty:.4f}, threshold={effective_threshold:.6f}"
                    )
                else:
                    # High variance despite small delta - not a true plateau
                    self._plateau_duration = 0
            else:
                # Clear improvement or decline - reset plateau counter
                self._plateau_duration = 0
        # else: history < 2, don't modify plateau_duration

        # 5. Compute total reward (weighted sum, all components in [-1, 1])
        total_reward = (
            self.relative_weight * relative_reward +
            self.raw_weight * absolute_reward +
            self.baseline_weight * baseline_reward +
            plateau_penalty  # Already in [-0.3, 0]
        )

        # 6. SOTA Protection: if current metric >= best metric, ensure positive reward
        # This rewards maintaining the best performance, preventing unfair negative rewards
        if self.enable_sota_protection and self._best_metric is not None:
            if current_metric >= self._best_metric - self.SOTA_TOLERANCE:
                if total_reward < self.SOTA_MIN_REWARD:
                    logger.debug(
                        f"SOTA protection applied: total_reward {total_reward:.4f} -> {self.SOTA_MIN_REWARD:.4f} "
                        f"(metric={current_metric:.4f}, best={self._best_metric:.4f})"
                    )
                    total_reward = self.SOTA_MIN_REWARD

        # Clip final reward to [-1, 1] for safety
        total_reward = np.clip(total_reward, -1.0, 1.0)

        result = RewardResult(
            total_reward=total_reward,
            relative_reward=relative_reward,
            baseline_reward=baseline_reward,
            absolute_reward=absolute_reward,
            plateau_penalty=plateau_penalty,
            raw_metric=current_metric,
            weights_used=self.get_weights(),
            plateau_duration=self._plateau_duration,
        )

        logger.debug(
            f"Reward computed (normalized): total={total_reward:.4f}, "
            f"relative={relative_reward:.4f}, absolute={absolute_reward:.4f}, "
            f"baseline={baseline_reward:.4f}, plateau_penalty={plateau_penalty:.4f}"
        )

        return result

    def reset_plateau_tracking(self):
        """Reset plateau duration counter and best metric (call at start of new training run)."""
        self._plateau_duration = 0
        self._best_metric = None
        logger.debug("Plateau tracking and best metric reset")

    def compute_batch(
        self,
        metrics: List[float]
    ) -> List[RewardResult]:
        """
        Compute rewards for a sequence of metrics.

        Useful for retrospective analysis.

        Args:
            metrics: List of metrics in chronological order

        Returns:
            List of RewardResults
        """
        results = []
        for i, metric in enumerate(metrics):
            history = metrics[:i] if i > 0 else None
            result = self.compute(metric, history)
            results.append(result)
        return results

    def get_cumulative_reward(
        self,
        history: List[float],
        gamma: float = 0.99
    ) -> float:
        """
        Compute discounted cumulative reward from history.

        Args:
            history: List of metric values
            gamma: Discount factor

        Returns:
            Discounted cumulative reward
        """
        if not history:
            return 0.0

        results = self.compute_batch(history)
        cumulative = 0.0
        for i, result in enumerate(results):
            cumulative += (gamma ** i) * result.total_reward

        return cumulative
