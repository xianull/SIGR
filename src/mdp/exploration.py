"""
Exploration Scheduling for SIGR MDP Framework

Implements exploration-exploitation balance using:
- ε-greedy with polynomial decay (theoretically correct)
- UCB1 (Upper Confidence Bound) with proper reward tracking

Theoretical foundations:
- Multi-armed bandits (Auer et al., 2002)
- UCB1 algorithm (Auer, Cesa-Bianchi, Fischer, 2002)
- ε-greedy with polynomial decay for sublinear regret
"""

import logging
import random
import math
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple

import numpy as np


logger = logging.getLogger(__name__)


@dataclass
class ExplorationDecision:
    """Result of exploration decision."""
    should_explore: bool
    exploration_type: str  # 'none', 'epsilon', 'ucb', 'forced'
    reason: str
    exploration_rate: float  # current epsilon
    ucb_bonus: float  # UCB exploration bonus if applicable


class ExplorationScheduler:
    """
    Manages exploration-exploitation trade-off in strategy optimization.

    Implements theoretically-grounded algorithms:
    - UCB1: UCB_i = X̄_i + c * sqrt(ln(t) / n_i)
    - ε-greedy with polynomial decay: ε_t = ε_0 / (1 + α*t)

    The scheduler tracks rewards for each strategy arm to compute
    proper UCB values, enabling principled exploration.
    """

    def __init__(
        self,
        initial_epsilon: float = 0.2,
        min_epsilon: float = 0.05,
        decay_alpha: float = 0.3,  # Polynomial decay coefficient (3x faster for quicker exploitation)
        ucb_coefficient: float = 1.414,  # sqrt(2) is standard
        plateau_boost: float = 0.05,  # Reduced to avoid excessive exploration during plateau
        decline_boost: float = 0.1,
    ):
        """
        Initialize ExplorationScheduler.

        Args:
            initial_epsilon: Starting exploration probability (ε_0)
            min_epsilon: Minimum exploration probability
            decay_alpha: Polynomial decay rate (ε_t = ε_0 / (1 + α*t))
            ucb_coefficient: Exploration coefficient c in UCB1 (default sqrt(2))
            plateau_boost: Extra exploration when in plateau
            decline_boost: Extra exploration when declining
        """
        self.initial_epsilon = initial_epsilon
        self.current_epsilon = initial_epsilon
        self.min_epsilon = min_epsilon
        self.decay_alpha = decay_alpha
        self.ucb_coefficient = ucb_coefficient
        self.plateau_boost = plateau_boost
        self.decline_boost = decline_boost

        # UCB1 requires tracking rewards and counts per arm
        self.strategy_rewards: Dict[str, float] = {}  # Cumulative rewards
        self.strategy_counts: Dict[str, int] = {}     # Visit counts
        self.total_visits: int = 0

        # For compatibility with existing code
        self.strategy_visits = self.strategy_counts

        # Track exploration statistics
        self.explore_count: int = 0
        self.exploit_count: int = 0

        logger.info(
            f"ExplorationScheduler initialized: "
            f"ε_0={initial_epsilon}, min_ε={min_epsilon}, "
            f"decay_α={decay_alpha}, UCB_c={ucb_coefficient}"
        )

    def update_reward(self, strategy: Dict[str, Any], reward: float):
        """
        Update reward statistics for a strategy (required for UCB1).

        This method MUST be called after each iteration with the
        strategy used and reward received.

        Args:
            strategy: The strategy configuration used
            reward: The reward received (should be in [0, 1] for UCB1)
        """
        key = self._strategy_to_key(strategy)

        if key not in self.strategy_rewards:
            self.strategy_rewards[key] = 0.0
            self.strategy_counts[key] = 0

        self.strategy_rewards[key] += reward
        self.strategy_counts[key] += 1
        self.total_visits += 1

        logger.debug(
            f"Updated reward for {key}: "
            f"total={self.strategy_rewards[key]:.4f}, "
            f"count={self.strategy_counts[key]}, "
            f"mean={self.strategy_rewards[key]/self.strategy_counts[key]:.4f}"
        )

    def decide(
        self,
        trend: str = 'unknown',
        convergence_score: float = 0.0,
        iteration: int = 0,
        force_explore: bool = False,
        current_strategy: Optional[Dict[str, Any]] = None,
    ) -> ExplorationDecision:
        """
        Decide whether to explore or exploit.

        Decision flow:
        1. Check forced exploration
        2. Check UCB1 (if strategy provided and has enough data)
        3. Fall back to ε-greedy

        Args:
            trend: Current performance trend
            convergence_score: How close to convergence (0-1)
            iteration: Current iteration number
            force_explore: Force exploration regardless of epsilon
            current_strategy: Current strategy for UCB calculation

        Returns:
            ExplorationDecision with reasoning
        """
        # Calculate effective epsilon with context adjustments
        effective_epsilon = self._calculate_effective_epsilon(
            trend, convergence_score, iteration
        )

        # Force exploration in certain conditions
        if force_explore:
            self.explore_count += 1
            return ExplorationDecision(
                should_explore=True,
                exploration_type='forced',
                reason='Forced exploration requested',
                exploration_rate=effective_epsilon,
                ucb_bonus=0.0,
            )

        # UCB1-based decision (if we have strategy and enough data)
        if current_strategy and self.total_visits > 0:
            ucb_decision = self._ucb_decision(current_strategy, effective_epsilon)
            if ucb_decision is not None:
                if ucb_decision.should_explore:
                    self.explore_count += 1
                else:
                    self.exploit_count += 1
                return ucb_decision

        # Fall back to ε-greedy
        random_value = random.random()
        if random_value < effective_epsilon:
            self.explore_count += 1
            return ExplorationDecision(
                should_explore=True,
                exploration_type='epsilon',
                reason=f'ε-greedy exploration (ε={effective_epsilon:.3f}, roll={random_value:.3f})',
                exploration_rate=effective_epsilon,
                ucb_bonus=0.0,
            )

        # Default: exploit
        self.exploit_count += 1
        return ExplorationDecision(
            should_explore=False,
            exploration_type='none',
            reason='Exploiting current best strategy',
            exploration_rate=effective_epsilon,
            ucb_bonus=0.0,
        )

    def _ucb_decision(
        self,
        current_strategy: Dict[str, Any],
        effective_epsilon: float
    ) -> Optional[ExplorationDecision]:
        """
        Make exploration decision using UCB1.

        UCB1 formula: UCB_i = X̄_i + c * sqrt(ln(t) / n_i)

        Returns None if not enough data for UCB decision.
        """
        key = self._strategy_to_key(current_strategy)

        # If this strategy hasn't been tried, explore it
        if key not in self.strategy_counts or self.strategy_counts[key] == 0:
            return ExplorationDecision(
                should_explore=True,
                exploration_type='ucb',
                reason=f'UCB: unexplored strategy {key}',
                exploration_rate=effective_epsilon,
                ucb_bonus=float('inf'),
            )

        # Calculate UCB value for current strategy
        n = self.strategy_counts[key]
        mean_reward = self.strategy_rewards[key] / n

        # UCB1 exploration bonus: c * sqrt(ln(t) / n)
        # Note: log(1) = 0, which is mathematically correct but gives zero bonus
        log_visits = math.log(max(self.total_visits, 2))  # 确保至少有正的探索奖励
        exploration_bonus = self.ucb_coefficient * math.sqrt(log_visits / n)

        ucb_value = mean_reward + exploration_bonus

        # Find if there's an unexplored or higher UCB arm
        # This implements the "optimism in face of uncertainty" principle
        best_ucb = ucb_value
        best_key = key

        for other_key in self.strategy_counts:
            other_n = self.strategy_counts[other_key]
            if other_n == 0:
                # Unexplored arm has infinite UCB
                best_ucb = float('inf')
                best_key = other_key
                break

            other_mean = self.strategy_rewards[other_key] / other_n
            other_bonus = self.ucb_coefficient * math.sqrt(log_visits / other_n)
            other_ucb = other_mean + other_bonus

            if other_ucb > best_ucb:
                best_ucb = other_ucb
                best_key = other_key

        # If current strategy is not the best UCB, suggest exploration
        if best_key != key:
            return ExplorationDecision(
                should_explore=True,
                exploration_type='ucb',
                reason=f'UCB: {best_key} has higher UCB ({best_ucb:.4f}) than current ({ucb_value:.4f})',
                exploration_rate=effective_epsilon,
                ucb_bonus=exploration_bonus,
            )

        # Current strategy has best UCB, exploit it
        return ExplorationDecision(
            should_explore=False,
            exploration_type='none',
            reason=f'UCB: current strategy has best UCB ({ucb_value:.4f})',
            exploration_rate=effective_epsilon,
            ucb_bonus=exploration_bonus,
        )

    def _calculate_effective_epsilon(
        self,
        trend: str,
        convergence_score: float,
        iteration: int
    ) -> float:
        """
        Calculate effective epsilon with polynomial decay and context adjustments.

        Uses polynomial decay: ε_t = ε_0 / (1 + α*t)
        This provides sublinear regret bounds (Auer et al., 2002).
        """
        # Polynomial decay (theoretically correct)
        base_epsilon = self.initial_epsilon / (1 + self.decay_alpha * iteration)
        base_epsilon = max(self.min_epsilon, base_epsilon)

        # Update current_epsilon for tracking
        self.current_epsilon = base_epsilon

        epsilon = base_epsilon

        # Context-aware adjustments (additive, bounded)
        if trend == 'plateau':
            epsilon = min(1.0, epsilon + self.plateau_boost)
        elif trend == 'declining':
            epsilon = min(1.0, epsilon + self.decline_boost)
        elif trend == 'oscillating':
            epsilon = min(1.0, epsilon + self.plateau_boost * 0.5)

        # Reduce exploration when improving and converging
        if trend == 'improving' and convergence_score > 0.7:
            reduction = convergence_score * 0.2
            epsilon = max(self.min_epsilon, epsilon - reduction)

        return epsilon

    def decay_epsilon(self, iteration: int = None):
        """
        Apply polynomial epsilon decay.

        Note: This is now handled in _calculate_effective_epsilon.
        Kept for API compatibility.
        """
        if iteration is not None:
            old_epsilon = self.current_epsilon
            self.current_epsilon = self.initial_epsilon / (1 + self.decay_alpha * iteration)
            self.current_epsilon = max(self.min_epsilon, self.current_epsilon)
            logger.debug(
                f"Epsilon decayed: {old_epsilon:.4f} -> {self.current_epsilon:.4f}"
            )

    def reset_epsilon(self, new_epsilon: Optional[float] = None):
        """Reset epsilon, optionally to a new value."""
        self.current_epsilon = new_epsilon or self.initial_epsilon
        logger.info(f"Epsilon reset to {self.current_epsilon:.4f}")

    def record_strategy_visit(self, strategy: Dict[str, Any]):
        """
        Record a strategy visit (for backwards compatibility).

        Note: Use update_reward() instead for proper UCB1 operation.
        """
        key = self._strategy_to_key(strategy)
        self.strategy_counts[key] = self.strategy_counts.get(key, 0) + 1
        self.total_visits += 1

    def _strategy_to_key(self, strategy: Dict[str, Any]) -> str:
        """
        Convert strategy to hashable key for UCB tracking.

        Uses main strategy dimensions: edge_types, max_hops, max_neighbors, sampling.
        Edge types are sorted to ensure same strategies produce same keys regardless of order.
        Numerical parameters are bucketed to reduce key explosion from minor variations.
        """
        key_parts = []

        # Edge types (sorted for consistency - order shouldn't matter)
        if 'edge_types' in strategy:
            edge_types = strategy['edge_types']
            if isinstance(edge_types, list):
                sorted_edges = sorted(edge_types)
                key_parts.append('e:' + '-'.join(sorted_edges))

        # Max hops
        if 'max_hops' in strategy:
            key_parts.append(f'h:{strategy["max_hops"]}')

        # Max neighbors (bucketed with 25 granularity to reduce key fragmentation)
        if 'max_neighbors' in strategy:
            max_n = strategy['max_neighbors']
            if isinstance(max_n, (int, float)):
                bucket = (int(max_n) // 25) * 25
                key_parts.append(f'n:{bucket}')

        # Sampling method
        if 'sampling' in strategy:
            key_parts.append(f's:{strategy["sampling"]}')

        return ':'.join(key_parts) if key_parts else 'default'

    def generate_exploration_perturbation(
        self,
        current_strategy: Dict[str, Any],
        exploration_strength: float = 0.3,
    ) -> Dict[str, Any]:
        """
        Generate perturbations for exploration.

        Creates variations of the current strategy for exploration.
        """
        perturbations = {}

        if not isinstance(current_strategy, dict):
            logger.warning(f"Invalid strategy type: {type(current_strategy)}")
            return perturbations

        all_edge_types = ['PPI', 'GO', 'HPO', 'TRRUST', 'CellMarker', 'Reactome']

        # Perturb edge_types
        if 'edge_types' in current_strategy:
            edge_types_raw = current_strategy['edge_types']
            if isinstance(edge_types_raw, list):
                edge_types = list(edge_types_raw)
            elif isinstance(edge_types_raw, str):
                edge_types = [edge_types_raw]
            else:
                edge_types = ['PPI', 'GO', 'HPO']

            if random.random() < exploration_strength:
                if len(edge_types) > 1:
                    idx = random.randint(0, len(edge_types) - 2)
                    edge_types[idx], edge_types[idx + 1] = edge_types[idx + 1], edge_types[idx]
                    perturbations['edge_types'] = edge_types
            elif random.random() < exploration_strength * 0.5:
                unused = [et for et in all_edge_types if et not in edge_types]
                if unused:
                    new_type = random.choice(unused)
                    edge_types.append(new_type)
                    perturbations['edge_types'] = edge_types
            elif random.random() < exploration_strength * 0.3 and len(edge_types) > 2:
                remove_idx = random.randint(0, len(edge_types) - 1)
                edge_types.pop(remove_idx)
                perturbations['edge_types'] = edge_types

        # Perturb max_neighbors
        if 'max_neighbors' in current_strategy:
            max_n = current_strategy['max_neighbors']
            if isinstance(max_n, (int, float)) and random.random() < exploration_strength:
                change = random.uniform(-0.3, 0.3)
                new_val = int(max_n * (1 + change))
                new_val = max(10, min(200, new_val))
                if new_val != max_n:
                    perturbations['max_neighbors'] = new_val

        # Perturb max_hops
        if 'max_hops' in current_strategy:
            if random.random() < exploration_strength:
                depth = current_strategy['max_hops']
                new_depth = depth + random.choice([-1, 1])
                new_depth = max(1, min(3, new_depth))
                if new_depth != depth:
                    perturbations['max_hops'] = new_depth

        # Perturb sampling method
        if random.random() < exploration_strength * 0.5:
            methods = ['top_k', 'random', 'weighted']
            current = current_strategy.get('sampling', 'top_k')
            new_method = random.choice([m for m in methods if m != current])
            perturbations['sampling'] = new_method

        logger.debug(f"Generated exploration perturbations: {list(perturbations.keys())}")

        return perturbations

    def get_ucb_values(self) -> Dict[str, float]:
        """Get UCB values for all tracked strategies."""
        if self.total_visits == 0:
            return {}

        ucb_values = {}
        log_visits = math.log(max(self.total_visits, 2))  # 确保至少有正的探索奖励
        for key in self.strategy_counts:
            n = self.strategy_counts[key]
            if n == 0:
                ucb_values[key] = float('inf')
            else:
                mean = self.strategy_rewards.get(key, 0) / n
                bonus = self.ucb_coefficient * math.sqrt(log_visits / n)
                ucb_values[key] = mean + bonus

        return ucb_values

    def get_state(self) -> Dict[str, Any]:
        """Get scheduler state for logging/serialization."""
        return {
            'current_epsilon': self.current_epsilon,
            'initial_epsilon': self.initial_epsilon,
            'min_epsilon': self.min_epsilon,
            'decay_alpha': self.decay_alpha,
            'ucb_coefficient': self.ucb_coefficient,
            'total_visits': self.total_visits,
            'strategy_counts': dict(self.strategy_counts),
            'strategy_rewards': dict(self.strategy_rewards),
            'explore_count': self.explore_count,
            'exploit_count': self.exploit_count,
        }

    def load_state(self, state: Dict[str, Any]):
        """Load scheduler state."""
        self.current_epsilon = state.get('current_epsilon', self.initial_epsilon)
        self.total_visits = state.get('total_visits', 0)
        self.strategy_counts = state.get('strategy_counts', {})
        self.strategy_rewards = state.get('strategy_rewards', {})
        self.explore_count = state.get('explore_count', 0)
        self.exploit_count = state.get('exploit_count', 0)
        # Compatibility
        self.strategy_visits = self.strategy_counts
