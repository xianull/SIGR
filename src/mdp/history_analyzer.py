"""
Trend Analysis for SIGR MDP Framework

Analyzes metric and strategy history to:
- Detect performance trends
- Identify effective strategy patterns
- Suggest exploration vs exploitation actions

Based on contextual bandit principles for context-aware decisions.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter

import numpy as np


logger = logging.getLogger(__name__)


@dataclass
class TrendAnalysis:
    """Result of trend analysis."""
    # Trend detection
    trend_direction: str  # 'improving', 'declining', 'plateau', 'oscillating'
    trend_strength: float  # 0-1 confidence in trend
    trend_duration: int  # iterations this trend has persisted

    # Convergence analysis
    convergence_score: float  # 0-1 how close to convergence
    variance_recent: float  # recent metric variance

    # Strategy insights
    effective_strategies: List[Dict[str, Any]]  # historically effective patterns
    ineffective_strategies: List[Dict[str, Any]]  # historically poor patterns

    # Suggested action
    suggested_action: str  # 'exploit', 'explore', 'reset'
    action_reason: str  # explanation for suggestion

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'trend_direction': self.trend_direction,
            'trend_strength': self.trend_strength,
            'trend_duration': self.trend_duration,
            'convergence_score': self.convergence_score,
            'variance_recent': self.variance_recent,
            'effective_strategies': self.effective_strategies,
            'ineffective_strategies': self.ineffective_strategies,
            'suggested_action': self.suggested_action,
            'action_reason': self.action_reason,
        }


class TrendAnalyzer:
    """
    Analyzes training history to detect trends and suggest actions.

    Provides context for policy decisions based on:
    - Performance trajectory
    - Historical strategy effectiveness
    - Convergence indicators
    """

    def __init__(
        self,
        min_history: int = 3,
        plateau_threshold: float = 0.005,
        improvement_threshold: float = 0.01,
        convergence_window: int = 5,
        oscillation_threshold: float = 0.02,
    ):
        """
        Initialize TrendAnalyzer.

        Args:
            min_history: Minimum iterations needed for analysis
            plateau_threshold: Variance below which is considered plateau
            improvement_threshold: Minimum slope to be considered improving
            convergence_window: Window for convergence calculation
            oscillation_threshold: Threshold for oscillation detection
        """
        self.min_history = min_history
        self.plateau_threshold = plateau_threshold
        self.improvement_threshold = improvement_threshold
        self.convergence_window = convergence_window
        self.oscillation_threshold = oscillation_threshold

    def analyze(
        self,
        metric_history: List[float],
        strategy_history: List[Dict[str, Any]],
        reward_history: Optional[List[float]] = None,
    ) -> TrendAnalysis:
        """
        Analyze history and produce trend analysis.

        Args:
            metric_history: List of metric values (oldest to newest)
            strategy_history: List of strategy configurations
            reward_history: Optional list of rewards

        Returns:
            TrendAnalysis with insights and suggestions
        """
        if len(metric_history) < self.min_history:
            return self._insufficient_history_result()

        # Detect trend
        trend_direction, trend_strength, trend_duration = self._detect_trend(
            metric_history
        )

        # Calculate convergence
        convergence_score = self._calculate_convergence(metric_history)
        variance_recent = float(np.var(metric_history[-self.convergence_window:]))

        # Analyze strategy effectiveness
        effective, ineffective = self._analyze_strategy_effectiveness(
            metric_history, strategy_history
        )

        # Determine suggested action
        action, reason = self._suggest_action(
            trend_direction, trend_strength, convergence_score, variance_recent
        )

        return TrendAnalysis(
            trend_direction=trend_direction,
            trend_strength=trend_strength,
            trend_duration=trend_duration,
            convergence_score=convergence_score,
            variance_recent=variance_recent,
            effective_strategies=effective,
            ineffective_strategies=ineffective,
            suggested_action=action,
            action_reason=reason,
        )

    def _insufficient_history_result(self) -> TrendAnalysis:
        """Return default result when history is insufficient."""
        return TrendAnalysis(
            trend_direction='unknown',
            trend_strength=0.0,
            trend_duration=0,
            convergence_score=0.0,
            variance_recent=0.0,
            effective_strategies=[],
            ineffective_strategies=[],
            suggested_action='explore',
            action_reason='Insufficient history for analysis, exploring to gather data',
        )

    def _detect_trend(
        self,
        metric_history: List[float]
    ) -> Tuple[str, float, int]:
        """
        Detect trend in metric history.

        Returns:
            (direction, strength, duration)
        """
        recent = metric_history[-self.convergence_window:]
        x = np.arange(len(recent))

        # Linear regression for trend
        slope, intercept = np.polyfit(x, recent, 1)
        residuals = recent - (slope * x + intercept)
        r_squared = 1 - (np.var(residuals) / (np.var(recent) + 1e-10))
        # Clamp R² to valid range [0, 1]
        r_squared = max(0.0, min(1.0, r_squared))

        variance = np.var(recent)

        # Detect oscillation
        if len(recent) >= 4:
            differences = np.diff(recent)
            sign_changes = np.sum(np.diff(np.sign(differences)) != 0)
            # Fix: use max(1, ...) to avoid division by zero
            oscillation_ratio = sign_changes / max(1, len(differences) - 1)
        else:
            oscillation_ratio = 0

        # Classify trend
        if variance < self.plateau_threshold:
            direction = 'plateau'
            strength = 1.0 - (variance / self.plateau_threshold)
        elif oscillation_ratio > 0.6 and variance > self.oscillation_threshold:
            direction = 'oscillating'
            strength = oscillation_ratio
        elif slope > self.improvement_threshold:
            direction = 'improving'
            strength = min(1.0, slope / (self.improvement_threshold * 5))
        elif slope < -self.improvement_threshold:
            direction = 'declining'
            strength = min(1.0, abs(slope) / (self.improvement_threshold * 5))
        else:
            direction = 'plateau'
            strength = 0.5

        # Calculate duration
        duration = self._calculate_trend_duration(metric_history, direction)

        logger.debug(
            f"Trend detected: {direction} (strength={strength:.2f}, duration={duration})"
        )

        return direction, float(strength), duration

    def _calculate_trend_duration(
        self,
        metric_history: List[float],
        current_trend: str
    ) -> int:
        """Calculate how long the current trend has persisted."""
        if len(metric_history) < self.min_history:
            return 0

        duration = 0
        for i in range(len(metric_history) - self.min_history, -1, -1):
            window = metric_history[i:i + self.min_history]
            trend, _, _ = self._detect_trend_for_window(window)
            if trend == current_trend:
                duration += 1
            else:
                break

        return duration

    def _detect_trend_for_window(
        self,
        window: List[float]
    ) -> Tuple[str, float, int]:
        """Simplified trend detection for a single window."""
        if len(window) < 2:
            return 'unknown', 0.0, 0

        x = np.arange(len(window))
        slope, _ = np.polyfit(x, window, 1)
        variance = np.var(window)

        if variance < self.plateau_threshold:
            return 'plateau', 1.0, 0
        elif slope > self.improvement_threshold:
            return 'improving', 0.7, 0
        elif slope < -self.improvement_threshold:
            return 'declining', 0.7, 0
        else:
            return 'plateau', 0.5, 0

    def _calculate_convergence(self, metric_history: List[float]) -> float:
        """
        Calculate convergence score.

        High convergence means:
        - Low recent variance
        - Slow rate of change
        - Near-maximum values
        """
        if len(metric_history) < self.convergence_window:
            return 0.0

        recent = metric_history[-self.convergence_window:]

        # Variance component (lower variance = more converged)
        variance = np.var(recent)
        variance_score = max(0, 1 - variance / (self.plateau_threshold * 10))

        # Rate of change component
        slope = abs(np.polyfit(np.arange(len(recent)), recent, 1)[0])
        slope_score = max(0, 1 - slope / self.improvement_threshold)

        # Value component (higher is better for most metrics)
        max_possible = 1.0  # Assume metrics are in [0, 1]
        best_recent = max(recent)
        value_score = best_recent / max_possible

        # Combined score
        convergence = 0.4 * variance_score + 0.3 * slope_score + 0.3 * value_score

        return float(convergence)

    def _analyze_strategy_effectiveness(
        self,
        metric_history: List[float],
        strategy_history: List[Dict[str, Any]],
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Identify effective and ineffective strategy patterns.

        Analyzes which strategy configurations led to improvements.
        """
        if len(metric_history) < 2 or len(strategy_history) < 2:
            return [], []

        # Calculate improvement for each strategy
        improvements = []
        for i in range(1, len(metric_history)):
            improvement = metric_history[i] - metric_history[i - 1]
            improvements.append((improvement, strategy_history[i - 1]))

        # Sort by improvement
        improvements.sort(key=lambda x: x[0], reverse=True)

        # Top strategies (led to improvement)
        effective = []
        for imp, strategy in improvements[:3]:
            if imp > 0:
                effective.append({
                    'strategy': self._extract_key_features(strategy),
                    'improvement': imp,
                })

        # Bottom strategies (led to decline)
        ineffective = []
        for imp, strategy in improvements[-3:]:
            if imp < 0:
                ineffective.append({
                    'strategy': self._extract_key_features(strategy),
                    'decline': abs(imp),
                })

        return effective, ineffective

    def _extract_key_features(self, strategy: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key features from strategy for comparison."""
        key_fields = [
            'edge_priorities',
            'max_neighbors',
            'subgraph_depth',
            'prompt_template',
            'sampling_method',
        ]
        return {k: v for k, v in strategy.items() if k in key_fields}

    def _suggest_action(
        self,
        trend_direction: str,
        trend_strength: float,
        convergence_score: float,
        variance: float,
    ) -> Tuple[str, str]:
        """
        Suggest action based on analysis.

        Returns:
            (action, reason)
        """
        # High convergence - exploit
        if convergence_score > 0.8:
            return 'exploit', f'High convergence ({convergence_score:.2f}), fine-tuning current approach'

        # Strong improving trend - exploit
        if trend_direction == 'improving' and trend_strength > 0.7:
            return 'exploit', f'Strong improvement trend, continuing current direction'

        # Declining trend - explore
        if trend_direction == 'declining':
            return 'explore', f'Performance declining, exploring new strategies'

        # Plateau - explore to break out
        if trend_direction == 'plateau' and trend_strength > 0.6:
            return 'explore', f'Stuck in plateau (duration detected), exploring to break out'

        # Oscillating - partial reset
        if trend_direction == 'oscillating':
            return 'explore', f'Performance oscillating, exploring more stable strategies'

        # Default - balanced
        if convergence_score > 0.5:
            return 'exploit', f'Moderate convergence, primarily exploiting with minor exploration'
        else:
            return 'explore', f'Early stage or unclear trend, exploring to find better strategies'

    def get_trend_summary(self, analysis: TrendAnalysis) -> str:
        """Get human-readable trend summary."""
        return (
            f"Trend: {analysis.trend_direction} (strength: {analysis.trend_strength:.2f}, "
            f"duration: {analysis.trend_duration} iters)\n"
            f"Convergence: {analysis.convergence_score:.2f}\n"
            f"Suggested: {analysis.suggested_action} - {analysis.action_reason}"
        )
