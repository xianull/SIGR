"""
Critic Component for SIGR Actor System

Implements value estimation for Actor-Critic architecture.
The Critic estimates the value V(s) of the current state,
enabling advantage computation A(a,s) = Q(a,s) - V(s).

This is a simple, history-based Critic suitable for the
LLM-guided optimization setting where we don't have
differentiable value networks.

Theoretical basis:
- Actor-Critic methods (Konda & Tsitsiklis, 2000)
- Value function approximation
- Advantage estimation for variance reduction
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple

import numpy as np


logger = logging.getLogger(__name__)


@dataclass
class StateValue:
    """Value estimation for a state."""
    value: float  # V(s) estimate in [-1, 1]
    confidence: float  # Confidence in estimate [0, 1]
    components: Dict[str, float]  # Breakdown of value components


@dataclass
class AdvantageEstimate:
    """Advantage estimate for an action."""
    advantage: float  # A(a,s) = Q(a,s) - V(s)
    state_value: float  # V(s)
    action_value: float  # Q(a,s) approximation


class SimpleCritic:
    """
    History-based Critic for state value estimation.

    Estimates V(s) using:
    1. Performance trend (improving/plateau/declining)
    2. Best achieved metric
    3. Consistency of recent performance
    4. Progress relative to theoretical optimum

    This Critic doesn't require gradient-based training;
    it uses heuristic value estimation based on observable
    state features.
    """

    # Value factors for different trends
    # Adjusted to penalize plateau and decline more heavily
    TREND_VALUES = {
        'improving': 0.80,
        'plateau': 0.30,      # Lowered: plateau is not progress, shouldn't be rewarded
        'declining': 0.10,    # Lowered: clearly penalize declining performance
        'oscillating': 0.25,  # Lowered: oscillation indicates instability
        'unknown': 0.50,
    }

    def __init__(
        self,
        trend_weight: float = 0.35,
        best_metric_weight: float = 0.30,
        consistency_weight: float = 0.20,
        progress_weight: float = 0.15,
        decay_factor: float = 0.95,
        metric_min: float = 0.0,
        metric_max: float = 1.0,
    ):
        """
        Initialize SimpleCritic.

        Args:
            trend_weight: Weight for trend-based value component
            best_metric_weight: Weight for best achieved metric
            consistency_weight: Weight for performance consistency
            progress_weight: Weight for progress to optimum
            decay_factor: Temporal discount for historical values
            metric_min: Minimum possible metric value
            metric_max: Maximum possible metric value
        """
        self.trend_weight = trend_weight
        self.best_metric_weight = best_metric_weight
        self.consistency_weight = consistency_weight
        self.progress_weight = progress_weight
        self.decay_factor = decay_factor
        self.metric_min = metric_min
        self.metric_max = metric_max

        # History of value estimates for temporal consistency
        self._value_history: List[float] = []

        logger.info(
            f"SimpleCritic initialized: weights=(trend={trend_weight}, "
            f"best={best_metric_weight}, consistency={consistency_weight}, "
            f"progress={progress_weight})"
        )

    def estimate_value(
        self,
        metric_history: List[float],
        trend: str = 'unknown',
        best_metric: Optional[float] = None,
        current_metric: Optional[float] = None,
    ) -> StateValue:
        """
        Estimate the value V(s) of the current state.

        Args:
            metric_history: List of historical metric values
            trend: Current performance trend
            best_metric: Best metric achieved so far
            current_metric: Current iteration metric

        Returns:
            StateValue with estimate and breakdown
        """
        if not metric_history:
            # No history: return neutral value
            return StateValue(
                value=0.0,
                confidence=0.1,
                components={'default': 0.0}
            )

        components = {}

        # 1. Trend-based value (with actual decline detection override)
        # Override trend to 'declining' if actual decline is detected
        effective_trend = trend
        if self._detect_actual_decline(metric_history) and trend != 'declining':
            logger.debug(f"Overriding trend '{trend}' to 'declining' based on metric history")
            effective_trend = 'declining'

        trend_value = self.TREND_VALUES.get(effective_trend, 0.5)
        components['trend'] = trend_value

        # 2. Best metric value (normalized to [-1, 1])
        if best_metric is None:
            best_metric = max(metric_history)
        best_normalized = self._normalize_metric(best_metric)
        components['best_metric'] = best_normalized

        # 3. Consistency value (inverse of coefficient of variation)
        if len(metric_history) >= 3:
            recent = metric_history[-5:]
            mean_val = np.mean(recent)
            std_val = np.std(recent)
            if mean_val > 0:
                cv = std_val / mean_val
                # Low CV = high consistency = high value
                consistency = 1.0 - min(cv * 2, 1.0)  # Scale CV
            else:
                consistency = 0.5
        else:
            consistency = 0.5
        components['consistency'] = consistency

        # 4. Progress value (how close to theoretical maximum)
        if current_metric is None:
            current_metric = metric_history[-1]
        progress = (current_metric - self.metric_min) / (self.metric_max - self.metric_min)
        progress = np.clip(progress, 0.0, 1.0)
        components['progress'] = progress

        # Compute weighted value (in [0, 1] initially)
        raw_value = (
            self.trend_weight * trend_value +
            self.best_metric_weight * (best_normalized + 1) / 2 +  # Convert [-1,1] to [0,1]
            self.consistency_weight * consistency +
            self.progress_weight * progress
        )

        # Convert to [-1, 1] range
        value = 2 * raw_value - 1
        value = np.clip(value, -1.0, 1.0)

        # Compute confidence based on history length and consistency
        confidence = min(1.0, len(metric_history) / 10) * (0.5 + 0.5 * consistency)

        # Apply temporal smoothing
        if self._value_history:
            smoothed_value = 0.7 * value + 0.3 * self._value_history[-1]
        else:
            smoothed_value = value

        self._value_history.append(smoothed_value)
        # Keep history bounded
        if len(self._value_history) > 20:
            self._value_history = self._value_history[-20:]

        logger.debug(
            f"Value estimated: V(s)={smoothed_value:.4f}, confidence={confidence:.4f}, "
            f"components={components}"
        )

        return StateValue(
            value=smoothed_value,
            confidence=confidence,
            components=components
        )

    def compute_advantage(
        self,
        reward: float,
        metric_history: List[float],
        trend: str = 'unknown',
        best_metric: Optional[float] = None,
        current_metric: Optional[float] = None,
        gamma: float = 0.99,
    ) -> AdvantageEstimate:
        """
        Compute advantage A(a,s) = Q(a,s) - V(s).

        For our setting without explicit Q-values, we approximate:
        Q(a,s) ≈ r + γ * V(s')

        Since we don't have V(s') yet, we use:
        A(a,s) ≈ r - V(s) (immediate advantage)

        Args:
            reward: Immediate reward received
            metric_history: Historical metrics
            trend: Current trend
            best_metric: Best metric so far
            current_metric: Current metric
            gamma: Discount factor

        Returns:
            AdvantageEstimate with advantage and value components
        """
        state_value = self.estimate_value(
            metric_history, trend, best_metric, current_metric
        )

        # Approximate Q-value as immediate reward
        # This is a simplification; ideally we'd bootstrap with V(s')
        action_value = reward

        # Advantage = Q - V
        advantage = action_value - state_value.value

        # Clip advantage for stability
        advantage = np.clip(advantage, -2.0, 2.0)

        logger.debug(
            f"Advantage computed: A={advantage:.4f}, Q≈{action_value:.4f}, "
            f"V={state_value.value:.4f}"
        )

        return AdvantageEstimate(
            advantage=advantage,
            state_value=state_value.value,
            action_value=action_value
        )

    def compute_td_error(
        self,
        reward: float,
        current_value: float,
        next_value: float,
        gamma: float = 0.99,
    ) -> float:
        """
        Compute temporal difference error.

        TD error: δ = r + γ * V(s') - V(s)

        Args:
            reward: Immediate reward
            current_value: V(s)
            next_value: V(s')
            gamma: Discount factor

        Returns:
            TD error
        """
        td_error = reward + gamma * next_value - current_value
        return td_error

    def _normalize_metric(self, metric: float) -> float:
        """Normalize metric from [min, max] to [-1, 1]."""
        normalized = (metric - self.metric_min) / (self.metric_max - self.metric_min)
        normalized = 2 * normalized - 1
        return np.clip(normalized, -1.0, 1.0)

    def _detect_actual_decline(self, metric_history: List[float]) -> bool:
        """
        Detect if there's an actual declining trend in recent metrics.

        Returns True if the last 3+ metrics show consistent decline.

        Args:
            metric_history: List of historical metric values

        Returns:
            True if declining trend detected
        """
        if len(metric_history) < 3:
            return False
        recent = metric_history[-3:]
        return all(recent[i] <= recent[i-1] for i in range(1, len(recent)))

    def get_baseline_value(self) -> float:
        """Get baseline value for variance reduction."""
        if not self._value_history:
            return 0.0
        return np.mean(self._value_history[-5:])

    def reset(self):
        """Reset Critic state for new training run."""
        self._value_history = []
        logger.debug("Critic state reset")

    def get_state(self) -> Dict[str, Any]:
        """Get Critic state for serialization."""
        return {
            'value_history': list(self._value_history),
            'weights': {
                'trend': self.trend_weight,
                'best_metric': self.best_metric_weight,
                'consistency': self.consistency_weight,
                'progress': self.progress_weight,
            }
        }

    def load_state(self, state: Dict[str, Any]):
        """Load Critic state from serialization."""
        self._value_history = state.get('value_history', [])
        if 'weights' in state:
            self.trend_weight = state['weights'].get('trend', self.trend_weight)
            self.best_metric_weight = state['weights'].get('best_metric', self.best_metric_weight)
            self.consistency_weight = state['weights'].get('consistency', self.consistency_weight)
            self.progress_weight = state['weights'].get('progress', self.progress_weight)


class GAECritic(SimpleCritic):
    """
    Critic with Generalized Advantage Estimation (GAE).

    Extends SimpleCritic with GAE-λ for better bias-variance tradeoff
    in advantage estimation.

    GAE: A^GAE_t = Σ_{l=0}^{∞} (γλ)^l * δ_{t+l}
    where δ_t = r_t + γV(s_{t+1}) - V(s_t)
    """

    def __init__(
        self,
        gae_lambda: float = 0.95,
        **kwargs
    ):
        """
        Initialize GAE Critic.

        Args:
            gae_lambda: GAE lambda parameter (0 = TD(0), 1 = Monte Carlo)
            **kwargs: Arguments for SimpleCritic
        """
        super().__init__(**kwargs)
        self.gae_lambda = gae_lambda
        self._trajectory_rewards: List[float] = []
        self._trajectory_values: List[float] = []

        logger.info(f"GAECritic initialized with λ={gae_lambda}")

    def store_transition(self, reward: float, value: float):
        """
        Store a transition for GAE computation.

        Args:
            reward: Reward received
            value: Value estimate at that state
        """
        self._trajectory_rewards.append(reward)
        self._trajectory_values.append(value)

    def compute_gae(
        self,
        rewards: Optional[List[float]] = None,
        values: Optional[List[float]] = None,
        gamma: float = 0.99,
    ) -> List[float]:
        """
        Compute GAE advantages for a trajectory.

        Args:
            rewards: List of rewards (uses stored if None)
            values: List of value estimates (uses stored if None)
            gamma: Discount factor

        Returns:
            List of GAE advantages
        """
        if rewards is None:
            rewards = self._trajectory_rewards
        if values is None:
            values = self._trajectory_values

        if not rewards or len(rewards) != len(values):
            return []

        advantages = []
        gae = 0.0

        # Compute GAE backwards
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0.0  # Terminal state
            else:
                next_value = values[t + 1]

            delta = rewards[t] + gamma * next_value - values[t]
            gae = delta + gamma * self.gae_lambda * gae
            advantages.insert(0, gae)

        return advantages

    def reset_trajectory(self):
        """Reset stored trajectory."""
        self._trajectory_rewards = []
        self._trajectory_values = []

    def reset(self):
        """Reset all Critic state."""
        super().reset()
        self.reset_trajectory()
