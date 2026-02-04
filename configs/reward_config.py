"""
Reward Configuration for SIGR Framework

Provides configurable reward weights with task-specific defaults
and adaptive weight adjustment based on training dynamics.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, List
import logging

logger = logging.getLogger(__name__)


@dataclass
class RewardWeights:
    """Configurable reward computation weights.

    Default weights emphasize absolute performance:
    - baseline_weight: 0.10 (comparison to moving average)
    - relative_weight: 0.10 (improvement over previous iteration)
    - raw_weight: 0.80 (absolute performance - primary signal)
    Sum = 1.0

    This design rewards high absolute performance strongly,
    while keeping relative/baseline comparisons as minor adjustments.
    """
    baseline_weight: float = 0.10
    relative_weight: float = 0.10
    raw_weight: float = 0.80
    improvement_bonus: float = 0.0  # Deprecated, kept for compatibility
    plateau_penalty: float = -0.05

    def to_dict(self) -> Dict[str, float]:
        return {
            'baseline_weight': self.baseline_weight,
            'relative_weight': self.relative_weight,
            'raw_weight': self.raw_weight,
            'improvement_bonus': self.improvement_bonus,
            'plateau_penalty': self.plateau_penalty,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> 'RewardWeights':
        return cls(
            baseline_weight=data.get('baseline_weight', 0.10),
            relative_weight=data.get('relative_weight', 0.10),
            raw_weight=data.get('raw_weight', 0.80),
            improvement_bonus=data.get('improvement_bonus', 0.0),
            plateau_penalty=data.get('plateau_penalty', -0.05),
        )

    def validate(self) -> bool:
        """Validate that weights sum to approximately 1.0."""
        total = self.baseline_weight + self.relative_weight + self.raw_weight
        if abs(total - 1.0) > 0.01:
            logger.warning(f"Reward weights sum to {total:.3f}, expected ~1.0")
            return False
        return True

    def normalize(self) -> 'RewardWeights':
        """Normalize weights to sum to 1.0."""
        total = self.baseline_weight + self.relative_weight + self.raw_weight
        if total > 0:
            return RewardWeights(
                baseline_weight=self.baseline_weight / total,
                relative_weight=self.relative_weight / total,
                raw_weight=self.raw_weight / total,
                improvement_bonus=self.improvement_bonus,
                plateau_penalty=self.plateau_penalty,
            )
        return self


# Task-specific default weights
# All tasks now emphasize absolute performance (raw_weight=0.80)
# with minor adjustments for relative improvement and baseline comparison
TASK_REWARD_WEIGHTS: Dict[str, RewardWeights] = {
    # PPI: High absolute weight, some relative for interaction patterns
    'ppi': RewardWeights(
        baseline_weight=0.10,
        relative_weight=0.10,
        raw_weight=0.80,
        improvement_bonus=0.0,
        plateau_penalty=-0.05,
    ),
    # Gene type: Same emphasis on absolute performance
    'genetype': RewardWeights(
        baseline_weight=0.10,
        relative_weight=0.10,
        raw_weight=0.80,
        improvement_bonus=0.0,
        plateau_penalty=-0.05,
    ),
    # GGI: Similar to PPI
    'ggi': RewardWeights(
        baseline_weight=0.10,
        relative_weight=0.10,
        raw_weight=0.80,
        improvement_bonus=0.0,
        plateau_penalty=-0.05,
    ),
    # Cell: High absolute weight (clustering quality matters)
    'cell': RewardWeights(
        baseline_weight=0.10,
        relative_weight=0.10,
        raw_weight=0.80,
        improvement_bonus=0.0,
        plateau_penalty=-0.05,
    ),
    # Gene attributes: High absolute weight
    'geneattribute_dosage_sensitivity': RewardWeights(
        baseline_weight=0.10,
        relative_weight=0.10,
        raw_weight=0.80,
        improvement_bonus=0.0,
        plateau_penalty=-0.05,
    ),
    'geneattribute_lys4_only': RewardWeights(
        baseline_weight=0.10,
        relative_weight=0.10,
        raw_weight=0.80,
        improvement_bonus=0.0,
        plateau_penalty=-0.05,
    ),
    'geneattribute_no_methylation': RewardWeights(
        baseline_weight=0.10,
        relative_weight=0.10,
        raw_weight=0.80,
        improvement_bonus=0.0,
        plateau_penalty=-0.05,
    ),
    'geneattribute_bivalent': RewardWeights(
        baseline_weight=0.10,
        relative_weight=0.10,
        raw_weight=0.80,
        improvement_bonus=0.0,
        plateau_penalty=-0.05,
    ),
    # Perturbation: Regression task, very high absolute weight
    # since correlation is a meaningful absolute metric
    'perturbation': RewardWeights(
        baseline_weight=0.05,
        relative_weight=0.10,
        raw_weight=0.85,  # Even higher for regression
        improvement_bonus=0.0,
        plateau_penalty=-0.05,
    ),
}


def get_reward_weights(task_name: str) -> RewardWeights:
    """
    Get reward weights for a specific task.

    Args:
        task_name: Name of the downstream task

    Returns:
        RewardWeights configured for the task
    """
    if task_name in TASK_REWARD_WEIGHTS:
        return TASK_REWARD_WEIGHTS[task_name]

    # Check for partial match (e.g., geneattribute_*)
    for key, weights in TASK_REWARD_WEIGHTS.items():
        if task_name.startswith(key.split('_')[0]):
            return weights

    # Default weights
    return RewardWeights()


@dataclass
class AdaptiveRewardConfig:
    """
    Configuration for adaptive reward weight adjustment.

    Automatically adjusts reward weights based on training dynamics:
    - Increase relative_weight when stuck in plateau
    - Increase improvement_bonus when breaking out of plateau
    - Adjust baseline_window based on convergence rate
    """
    enabled: bool = True
    # Minimum iterations before adaptation
    warmup_iterations: int = 3
    # How often to adapt (every N iterations)
    adapt_interval: int = 2
    # Maximum adjustment per adaptation
    max_adjustment: float = 0.1
    # Plateau detection threshold
    plateau_threshold: float = 0.005
    # Oscillation detection window
    oscillation_window: int = 4


class AdaptiveRewardManager:
    """
    Manages adaptive adjustment of reward weights during training.

    Monitors training dynamics and adjusts reward computation
    to provide clearer learning signals in different phases.
    """

    def __init__(
        self,
        initial_weights: RewardWeights,
        config: Optional[AdaptiveRewardConfig] = None
    ):
        """
        Initialize the adaptive reward manager.

        Args:
            initial_weights: Starting reward weights
            config: Adaptive configuration (uses defaults if None)
        """
        self.weights = initial_weights
        self.config = config or AdaptiveRewardConfig()
        self.metric_history: List[float] = []
        self.iteration_count = 0
        self.last_adaptation = 0
        self._original_weights = RewardWeights(**initial_weights.to_dict())

        logger.info(f"AdaptiveRewardManager initialized: enabled={self.config.enabled}")

    def update(self, metric: float, trend: str = 'unknown') -> RewardWeights:
        """
        Update reward weights based on latest metric and trend.

        Args:
            metric: Latest evaluation metric value
            trend: Current trend ('improving', 'plateau', 'declining', 'oscillating')

        Returns:
            Potentially adjusted RewardWeights
        """
        self.metric_history.append(metric)
        self.iteration_count += 1

        if not self.config.enabled:
            return self.weights

        # Skip warmup phase
        if self.iteration_count < self.config.warmup_iterations:
            return self.weights

        # Check if it's time to adapt
        if (self.iteration_count - self.last_adaptation) < self.config.adapt_interval:
            return self.weights

        # Perform adaptation based on trend
        adjusted = self._adapt_weights(trend)

        if adjusted:
            self.last_adaptation = self.iteration_count
            logger.info(f"Reward weights adapted at iteration {self.iteration_count}: "
                       f"relative={self.weights.relative_weight:.3f}, "
                       f"baseline={self.weights.baseline_weight:.3f}")

        return self.weights

    def _adapt_weights(self, trend: str) -> bool:
        """
        Adapt weights based on training trend.

        Returns True if weights were modified.
        """
        adjustment = self.config.max_adjustment
        modified = False

        if trend == 'plateau':
            # In plateau: increase relative weight to emphasize changes
            # Also increase improvement bonus to reward breaking out
            self.weights.relative_weight = min(
                0.7,
                self.weights.relative_weight + adjustment * 0.5
            )
            self.weights.improvement_bonus = min(
                0.2,
                self.weights.improvement_bonus + adjustment * 0.3
            )
            # Increase plateau penalty to discourage stagnation
            self.weights.plateau_penalty = max(
                -0.05,
                self.weights.plateau_penalty - 0.01
            )
            modified = True
            logger.debug(f"Adapted for plateau: relative={self.weights.relative_weight:.3f}")

        elif trend == 'declining':
            # Declining: increase raw weight to stabilize
            # Reduce relative weight (negative changes are punishing)
            self.weights.raw_weight = min(
                0.4,
                self.weights.raw_weight + adjustment * 0.3
            )
            self.weights.relative_weight = max(
                0.3,
                self.weights.relative_weight - adjustment * 0.2
            )
            modified = True
            logger.debug(f"Adapted for decline: raw={self.weights.raw_weight:.3f}")

        elif trend == 'oscillating':
            # Oscillating: increase baseline weight for smoothing
            self.weights.baseline_weight = min(
                0.5,
                self.weights.baseline_weight + adjustment * 0.4
            )
            # Reduce improvement bonus (may cause oscillation)
            self.weights.improvement_bonus = max(
                0.05,
                self.weights.improvement_bonus - adjustment * 0.2
            )
            modified = True
            logger.debug(f"Adapted for oscillation: baseline={self.weights.baseline_weight:.3f}")

        elif trend == 'improving':
            # Improving: gradually return to original weights
            self._move_towards_original(0.3)
            modified = True

        # Normalize weights to maintain sum = 1.0
        if modified:
            self.weights = self.weights.normalize()

        return modified

    def _move_towards_original(self, rate: float = 0.2):
        """Gradually move weights back towards original values."""
        self.weights.baseline_weight += rate * (
            self._original_weights.baseline_weight - self.weights.baseline_weight
        )
        self.weights.relative_weight += rate * (
            self._original_weights.relative_weight - self.weights.relative_weight
        )
        self.weights.raw_weight += rate * (
            self._original_weights.raw_weight - self.weights.raw_weight
        )

    def reset(self):
        """Reset to original weights."""
        self.weights = RewardWeights(**self._original_weights.to_dict())
        self.metric_history = []
        self.iteration_count = 0
        self.last_adaptation = 0
        logger.info("AdaptiveRewardManager reset to original weights")

    def get_state(self) -> Dict:
        """Get current state for serialization."""
        return {
            'weights': self.weights.to_dict(),
            'original_weights': self._original_weights.to_dict(),
            'metric_history': self.metric_history,
            'iteration_count': self.iteration_count,
            'last_adaptation': self.last_adaptation,
        }

    def load_state(self, state: Dict):
        """Load state from serialized data."""
        self.weights = RewardWeights.from_dict(state.get('weights', {}))
        self._original_weights = RewardWeights.from_dict(state.get('original_weights', {}))
        self.metric_history = state.get('metric_history', [])
        self.iteration_count = state.get('iteration_count', 0)
        self.last_adaptation = state.get('last_adaptation', 0)
