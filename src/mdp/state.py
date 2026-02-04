"""
MDP State Representation for SIGR Framework

Provides explicit state representation for the MDP formulation.
State captures all information needed for policy decisions.

State components:
- Current iteration and strategy
- Metric and strategy history
- Trend analysis results
- Exploration parameters
- Best found solution
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from copy import deepcopy


logger = logging.getLogger(__name__)


@dataclass
class MDPState:
    """
    Explicit MDP state representation.

    Captures the complete state needed for:
    - Policy decision making (exploit vs explore)
    - Trend-aware strategy adjustment
    - Best solution tracking

    This formalization ensures the agent has clear
    context for each decision.
    """

    # Maximum history length to prevent memory issues
    MAX_HISTORY_SIZE: int = 100

    # Current iteration info
    iteration: int = 0
    current_strategy: Dict[str, Any] = field(default_factory=dict)

    # History tracking
    metric_history: List[float] = field(default_factory=list)
    strategy_history: List[Dict[str, Any]] = field(default_factory=list)
    reward_history: List[float] = field(default_factory=list)

    # Trend analysis
    trend: str = 'unknown'  # 'improving', 'declining', 'plateau', 'oscillating'
    trend_strength: float = 0.0  # 0-1 confidence in trend
    convergence_score: float = 0.0  # 0-1 how close to convergence

    # Exploration state
    exploration_rate: float = 0.3  # current epsilon
    exploration_count: int = 0  # number of exploration steps taken
    exploitation_count: int = 0  # number of exploitation steps taken

    # Best solution tracking
    best_metric: float = 0.0
    best_strategy: Dict[str, Any] = field(default_factory=dict)
    best_iteration: int = 0

    # Task context
    task_name: str = ''
    primary_metric: str = 'auroc'

    def update_after_iteration(
        self,
        metric: float,
        strategy: Dict[str, Any],
        reward: float,
        trend: str = None,
        trend_strength: float = None,
        convergence_score: float = None,
    ):
        """
        Update state after completing an iteration.

        Args:
            metric: Achieved metric value
            strategy: Strategy used
            reward: Computed reward
            trend: Optional trend analysis result
            trend_strength: Optional trend confidence
            convergence_score: Optional convergence measure
        """
        # Update history with size limit
        self.metric_history.append(metric)
        self.strategy_history.append(deepcopy(strategy))
        self.reward_history.append(reward)

        # Trim history if exceeds max size
        if len(self.metric_history) > self.MAX_HISTORY_SIZE:
            self.metric_history = self.metric_history[-self.MAX_HISTORY_SIZE:]
            self.strategy_history = self.strategy_history[-self.MAX_HISTORY_SIZE:]
            self.reward_history = self.reward_history[-self.MAX_HISTORY_SIZE:]

        # Update current
        self.current_strategy = deepcopy(strategy)
        self.iteration += 1

        # Update best if improved
        if metric > self.best_metric:
            self.best_metric = metric
            self.best_strategy = deepcopy(strategy)
            self.best_iteration = self.iteration - 1  # 0-indexed
            logger.info(
                f"New best metric: {metric:.4f} at iteration {self.best_iteration}"
            )

        # Update trend if provided
        if trend is not None:
            self.trend = trend
        if trend_strength is not None:
            self.trend_strength = trend_strength
        if convergence_score is not None:
            self.convergence_score = convergence_score

    def record_exploration(self, explored: bool):
        """Record whether this step was exploration or exploitation."""
        if explored:
            self.exploration_count += 1
        else:
            self.exploitation_count += 1

    def get_recent_history(self, n: int = 5) -> Dict[str, Any]:
        """
        Get recent history for context.

        Args:
            n: Number of recent iterations to include

        Returns:
            Dictionary with recent metrics, strategies, rewards
        """
        return {
            'metrics': self.metric_history[-n:] if self.metric_history else [],
            'strategies': self.strategy_history[-n:] if self.strategy_history else [],
            'rewards': self.reward_history[-n:] if self.reward_history else [],
        }

    def get_improvement_rate(self, window: int = 5) -> float:
        """
        Calculate recent improvement rate.

        Args:
            window: Number of iterations to consider

        Returns:
            Average improvement per iteration
        """
        if len(self.metric_history) < 2:
            return 0.0

        recent = self.metric_history[-window:]
        if len(recent) < 2:
            return 0.0

        # Linear regression slope
        import numpy as np
        x = np.arange(len(recent))
        slope, _ = np.polyfit(x, recent, 1)
        return float(slope)

    def is_converged(self, threshold: float = 0.9) -> bool:
        """
        Check if training has converged.

        Args:
            threshold: Convergence score threshold

        Returns:
            True if converged
        """
        return self.convergence_score >= threshold

    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for serialization."""
        return {
            'iteration': self.iteration,
            'current_strategy': self.current_strategy,
            'metric_history': self.metric_history,
            'strategy_history': self.strategy_history,  # Added for checkpoint recovery
            'reward_history': self.reward_history,
            'trend': self.trend,
            'trend_strength': self.trend_strength,
            'convergence_score': self.convergence_score,
            'exploration_rate': self.exploration_rate,
            'exploration_count': self.exploration_count,
            'exploitation_count': self.exploitation_count,
            'best_metric': self.best_metric,
            'best_strategy': self.best_strategy,
            'best_iteration': self.best_iteration,
            'task_name': self.task_name,
            'primary_metric': self.primary_metric,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MDPState':
        """Create state from dictionary."""
        state = cls()
        for key, value in data.items():
            if hasattr(state, key):
                setattr(state, key, value)
        return state

    def get_summary(self) -> str:
        """Get human-readable state summary."""
        lines = [
            f"=== MDP State (Iteration {self.iteration}) ===",
            f"Current metric: {self.metric_history[-1]:.4f}" if self.metric_history else "No metrics yet",
            f"Best metric: {self.best_metric:.4f} (iter {self.best_iteration})",
            f"Trend: {self.trend} (strength: {self.trend_strength:.2f})",
            f"Convergence: {self.convergence_score:.2f}",
            f"Exploration rate: {self.exploration_rate:.2f}",
            f"Explore/Exploit: {self.exploration_count}/{self.exploitation_count}",
        ]
        return '\n'.join(lines)
