"""
Reward Computation for SIGR MDP Framework

科学发现范式 (Scientific Discovery Paradigm)
==========================================

核心理念：Agent 是科学家，每次迭代是一次实验 (Experiment)

设计原则：
1. 效用势能 (Utility Potential): R_utility = tanh((metric_current - metric_best) / σ)
   - 光滑连续函数，自动包含"没改进=负值"逻辑
   - σ 控制敏感度（默认 0.05）

2. 探索代价 (Exploration Cost): R_cost = λ * (1 - strategy_distance)
   - 惩罚"无脑试错"（策略相似但没改进）
   - 鼓励"勇敢探索"（策略变化大）

3. 实验状态 (Experiment State):
   - BREAKTHROUGH: 突破历史最佳
   - EXPLORATION_FAILURE: 勇敢探索但失败
   - STAGNATION: 懒惰重复（策略相似+没改进）

最终公式：
- Total = R_utility (突破时)
- Total = R_utility - R_cost (失败时)

Theoretical basis:
- Information gain / surprise in scientific discovery
- Reward shaping (Ng et al., 1999)
- Exploration bonus via strategy novelty
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Union, Dict, Any

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


# =============================================================================
# Experiment State Enum (科学发现范式)
# =============================================================================

class ExperimentState(str, Enum):
    """
    实验结果状态 (Experiment Outcome State)

    用于指导 Actor 的反思模式 (Thinking Mode)
    """
    BREAKTHROUGH = "BREAKTHROUGH"              # 突破：metric > best_metric
    EXPLORATION_FAILURE = "EXPLORATION_FAILURE"  # 探索失败：策略变化大但没改进
    STAGNATION = "STAGNATION"                  # 停滞：策略相似且没改进


# =============================================================================
# Reward Signal (结构化奖励信号)
# =============================================================================

@dataclass
class RewardSignal:
    """
    结构化奖励信号，用于科学家风格的反馈 (Scientist-style feedback)

    核心组件：
    - utility_reward: 效用势能 R_utility = tanh((current - best) / σ)
    - exploration_cost: 探索代价 R_cost = λ * (1 - distance)
    - state: 实验状态 (BREAKTHROUGH / EXPLORATION_FAILURE / STAGNATION)
    - feedback_message: 人类可读的反馈消息，直接用于 LLM prompt
    """
    total_reward: float              # 最终奖励 [-1, 1]
    utility_reward: float            # R_utility = tanh((current - best) / σ)
    exploration_cost: float          # R_cost = λ * (1 - strategy_distance)
    state: ExperimentState           # 实验结果状态
    strategy_distance: float         # 策略距离 [0, 1]
    raw_metric: float               # 当前指标值
    best_metric: float              # 历史最佳指标
    metric_delta: float             # current - best
    feedback_message: str           # 人类可读反馈（用于 LLM prompt）

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            'total_reward': self.total_reward,
            'utility_reward': self.utility_reward,
            'exploration_cost': self.exploration_cost,
            'state': self.state.value,
            'strategy_distance': self.strategy_distance,
            'raw_metric': self.raw_metric,
            'best_metric': self.best_metric,
            'metric_delta': self.metric_delta,
            'feedback_message': self.feedback_message,
        }


# =============================================================================
# Legacy RewardResult (保持向后兼容)
# =============================================================================


@dataclass
class RewardResult:
    """Result of reward computation with breakdown (Legacy format for compatibility)."""
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

    @classmethod
    def from_reward_signal(cls, signal: RewardSignal) -> 'RewardResult':
        """
        Convert new RewardSignal to legacy RewardResult for backward compatibility.

        Args:
            signal: New-style RewardSignal

        Returns:
            Legacy RewardResult for existing logging/integration
        """
        return cls(
            total_reward=signal.total_reward,
            relative_reward=signal.metric_delta,  # Map metric_delta to relative
            baseline_reward=signal.utility_reward,  # Map utility to baseline
            absolute_reward=signal.raw_metric,
            plateau_penalty=-signal.exploration_cost,  # Map exploration cost as penalty
            raw_metric=signal.raw_metric,
            weights_used={
                'utility_scale': 0.05,
                'exploration_lambda': 0.15,
                'state': signal.state.value,
            },
            plateau_duration=1 if signal.state == ExperimentState.STAGNATION else 0,
        )


class RewardComputer:
    """
    Computes gated rewards for MDP-based strategy optimization.

    核心设计：没有改进就是惩罚 (No improvement = Negative reward)

    条件分支逻辑（代替线性加权）：
    - IF current_metric > best_metric: 正向奖励
    - ELSE: 负向惩罚（强制为负）

    Components:
    - Gated absolute reward: 只有改进时全额发放，否则衰减 90%
    - Dynamic baseline: 跟 best_metric 比，而不是 moving average
    - No-improvement penalty: 基础惩罚 -0.3，退步时更重
    - Plateau penalty: 持续停滞的额外惩罚
    - Forced negative ceiling: 没有改进时 reward 上限为 -0.2
    """

    # Metric range for normalization (assuming metrics in [0, 1])
    METRIC_MIN = 0.0
    METRIC_MAX = 1.0

    # Non-linear scaling parameters for high scores
    HIGH_SCORE_THRESHOLD = 0.8  # Threshold for non-linear bonus
    HIGH_SCORE_SCALING = 5.0    # Scaling factor for bonus

    # 门控参数 (Gated Reward Parameters)
    GATED_DECAY_FACTOR = 0.1           # 没有改进时 absolute_reward 衰减系数
    NO_IMPROVEMENT_BASE = -0.3         # 没有改进的基础惩罚
    NO_IMPROVEMENT_GAP_SCALE = 2.0     # 退步惩罚的放大系数
    NO_IMPROVEMENT_MAX_PENALTY = -0.6  # 最大惩罚上限
    FORCED_NEGATIVE_CEILING = -0.2     # 没有改进时的 reward 上限（强制负向）

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
        # No-improvement penalty (核心: 没有改进就是惩罚)
        enable_no_improvement_penalty: bool = True,
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
            enable_no_improvement_penalty: Enable "no improvement = punishment" strategy
        """
        # Non-linear scaling and gated reward flags
        self.enable_nonlinear_scaling = enable_nonlinear_scaling
        self.enable_no_improvement_penalty = enable_no_improvement_penalty
        self._best_metric: Optional[float] = None  # Track best metric for gating

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
            f"nonlinear_scaling={self.enable_nonlinear_scaling}, no_improvement_penalty={self.enable_no_improvement_penalty}"
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

    def set_baseline_metric(self, metric: float):
        """
        Legacy method kept for compatibility.

        新的门控逻辑不再需要单独设置 baseline，
        因为 best_metric 在第一次迭代时自动设置。
        """
        pass  # No-op, best_metric is set automatically in compute()

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
        Compute gated reward based on improvement over best metric.

        核心设计原则：没有改进就是惩罚 (No improvement = Negative reward)

        条件分支逻辑：
        - IF current_metric > best_metric: 正向奖励（鼓励改进）
        - ELSE: 负向惩罚（强制探索）

        关键机制：
        1. 门控绝对奖励：只有改进时才发放 absolute_reward
        2. 强制负向主导：没有改进时 reward 必须为负
        3. 动态基准线：跟 best_metric 比，而不是 moving average

        Args:
            current_metric: Current iteration's primary metric value (in [0, 1])
            history: List of previous metric values (oldest to newest)

        Returns:
            RewardResult with total reward and breakdown
        """
        metric_range = self.METRIC_MAX - self.METRIC_MIN

        # 判断是否有改进
        is_improvement = (self._best_metric is None) or (current_metric > self._best_metric)

        # 计算原始 absolute_reward
        raw_absolute_reward = self._compute_absolute_reward(current_metric)

        if history is None or len(history) == 0:
            # 第一次迭代：建立 baseline，给予适度正向奖励
            self._best_metric = current_metric
            self._plateau_duration = 0

            # 第一次迭代：正常发放 absolute_reward
            total_reward = self.raw_weight * raw_absolute_reward
            total_reward = np.clip(total_reward, -1.0, 1.0)

            logger.info(
                f"First iteration: metric={current_metric:.4f}, "
                f"absolute={raw_absolute_reward:.4f}, total={total_reward:.4f}"
            )

            return RewardResult(
                total_reward=total_reward,
                relative_reward=0.0,
                baseline_reward=0.0,
                absolute_reward=raw_absolute_reward,
                plateau_penalty=0.0,
                raw_metric=current_metric,
                weights_used=self.get_weights(),
                plateau_duration=0,
            )

        # ========== 有历史记录的情况 ==========

        # 1. 计算 relative_reward（相对于上一次的变化）
        previous_metric = history[-1]
        relative_delta = current_metric - previous_metric
        relative_reward = relative_delta / metric_range
        relative_reward = np.clip(relative_reward, -1.0, 1.0)

        # 2. 动态基准线：跟 best_metric 比，而不是 moving average
        if self._best_metric is not None:
            baseline_delta = current_metric - self._best_metric
            baseline_reward = baseline_delta / metric_range
        else:
            baseline_reward = 0.0
        baseline_reward = np.clip(baseline_reward, -1.0, 1.0)

        # 3. 门控绝对奖励 (Gated Absolute Reward)
        if is_improvement:
            # 改进：全额发放 absolute_reward
            gated_absolute = raw_absolute_reward
            no_improvement_penalty = 0.0
            logger.debug(f"Improvement detected: {self._best_metric:.4f} -> {current_metric:.4f}")
        else:
            # 没有改进：absolute_reward 衰减 90%
            gated_absolute = raw_absolute_reward * self.GATED_DECAY_FACTOR

            # 强制负向主导：计算 no_improvement_penalty
            improvement_gap = self._best_metric - current_metric
            no_improvement_penalty = self.NO_IMPROVEMENT_BASE - improvement_gap * self.NO_IMPROVEMENT_GAP_SCALE
            # 限制惩罚在 [MAX_PENALTY, 0] 范围内（MAX_PENALTY 是负数）
            no_improvement_penalty = float(np.clip(no_improvement_penalty, self.NO_IMPROVEMENT_MAX_PENALTY, 0.0))

            logger.debug(
                f"No improvement: best={self._best_metric:.4f}, current={current_metric:.4f}, "
                f"gap={improvement_gap:.4f}, penalty={no_improvement_penalty:.4f}"
            )

        # 4. Plateau penalty（持续停滞的额外惩罚）
        plateau_penalty = 0.0
        if self.use_relative_plateau_threshold:
            effective_threshold = self.plateau_threshold * max(previous_metric, 0.01)
        else:
            effective_threshold = self.plateau_threshold

        if len(history) >= 2 and abs(relative_delta) < effective_threshold:
            recent_std = np.std(history[-min(3, len(history)):])
            if recent_std < effective_threshold * 1.5:
                self._plateau_duration += 1
                plateau_penalty = self.base_plateau_penalty * (
                    1 + self._plateau_duration * self.plateau_escalation_rate
                )
                plateau_penalty = max(plateau_penalty, self.max_plateau_penalty)
            else:
                self._plateau_duration = 0
        elif is_improvement:
            self._plateau_duration = 0

        # 5. 计算总 reward
        total_reward = (
            self.relative_weight * relative_reward +
            self.raw_weight * gated_absolute +
            self.baseline_weight * baseline_reward +
            plateau_penalty +
            no_improvement_penalty
        )

        # 6. 强制负向：没有改进时确保 reward 为负
        if not is_improvement and self.enable_no_improvement_penalty:
            total_reward = min(total_reward, self.FORCED_NEGATIVE_CEILING)

        # Clip final reward
        total_reward = np.clip(total_reward, -1.0, 1.0)

        # 7. 更新 best_metric（只在改进时更新）
        if is_improvement:
            self._best_metric = current_metric

        # 合并所有 penalty 用于日志显示
        combined_penalty = plateau_penalty + no_improvement_penalty

        result = RewardResult(
            total_reward=total_reward,
            relative_reward=relative_reward,
            baseline_reward=baseline_reward,
            absolute_reward=gated_absolute,  # 返回门控后的值
            plateau_penalty=combined_penalty,
            raw_metric=current_metric,
            weights_used=self.get_weights(),
            plateau_duration=self._plateau_duration,
        )

        logger.debug(
            f"Reward: total={total_reward:.4f}, is_improvement={is_improvement}, "
            f"gated_abs={gated_absolute:.4f}, relative={relative_reward:.4f}, "
            f"baseline={baseline_reward:.4f}, no_improve={no_improvement_penalty:.4f}, "
            f"plateau={plateau_penalty:.4f}"
        )

        return result

    def reset_plateau_tracking(self):
        """Reset all tracking state (call at start of new training run)."""
        self._plateau_duration = 0
        self._best_metric = None
        logger.debug("All tracking state reset")

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

    # =========================================================================
    # 科学发现范式 (Scientific Discovery Paradigm) - New Methods
    # =========================================================================

    # 新范式参数 (New Paradigm Parameters)
    UTILITY_SCALE = 0.05          # σ for tanh sensitivity (控制敏感度)
    EXPLORATION_LAMBDA = 0.15     # λ for exploration cost weight (探索代价权重)
    STAGNATION_THRESHOLD = 0.2    # Strategy distance below this = "lazy" (停滞阈值)

    def compute_signal(
        self,
        current_metric: float,
        history: Optional[List[float]] = None,
        strategy_distance: float = 0.0
    ) -> RewardSignal:
        """
        科学发现范式的奖励计算 (Scientific Discovery Reward Computation)

        公式:
        - R_utility = tanh((current - best) / σ)
        - R_cost = λ * (1 - strategy_distance) 仅在失败时应用
        - Total = R_utility (突破) 或 R_utility - R_cost (失败)

        状态分类:
        - BREAKTHROUGH: current_metric > best_metric
        - STAGNATION: strategy_distance < threshold 且没有改进
        - EXPLORATION_FAILURE: 其他情况（勇敢探索但失败）

        Args:
            current_metric: 当前迭代的指标值 [0, 1]
            history: 历史指标列表（用于确定是否为第一次迭代）
            strategy_distance: 策略距离 [0, 1]，从 compute_strategy_distance() 获取

        Returns:
            RewardSignal: 结构化奖励信号，包含 feedback_message 用于 LLM prompt
        """
        # 确定是否为第一次迭代
        is_first_iteration = (history is None or len(history) == 0)

        # 获取或初始化 best_metric
        if self._best_metric is None:
            if is_first_iteration:
                self._best_metric = current_metric
            elif history:
                self._best_metric = max(history)
            else:
                # 逻辑矛盾：不是第一次迭代但没有历史
                logger.warning(
                    "Inconsistent state: not first iteration but no history. "
                    "Using current_metric as baseline."
                )
                self._best_metric = current_metric

        # 计算指标差异
        metric_delta = current_metric - self._best_metric

        # 1. 效用势能 (Utility Potential)
        # R_utility = tanh((current - best) / σ)
        utility_reward = float(np.tanh(metric_delta / self.UTILITY_SCALE))

        # 2. 判断是否为突破 (Breakthrough)
        is_breakthrough = current_metric > self._best_metric

        # 3. 确定实验状态 (Experiment State)
        if is_breakthrough:
            state = ExperimentState.BREAKTHROUGH
            exploration_cost = 0.0  # 突破时不惩罚
        elif strategy_distance < self.STAGNATION_THRESHOLD:
            state = ExperimentState.STAGNATION
            # 停滞时探索代价最高
            exploration_cost = self.EXPLORATION_LAMBDA * (1 - strategy_distance)
        else:
            state = ExperimentState.EXPLORATION_FAILURE
            # 探索失败但勇敢尝试，代价较低
            exploration_cost = self.EXPLORATION_LAMBDA * (1 - strategy_distance) * 0.5

        # 4. 计算总奖励 (Total Reward)
        if is_breakthrough:
            total_reward = utility_reward
        else:
            total_reward = utility_reward - exploration_cost

        # Clip to [-1, 1]
        total_reward = float(np.clip(total_reward, -1.0, 1.0))

        # 5. 生成反馈消息 (Feedback Message for LLM)
        feedback_message = self._generate_feedback_message(
            state=state,
            metric_delta=metric_delta,
            strategy_distance=strategy_distance,
            current_metric=current_metric,
            best_metric=self._best_metric
        )

        # 6. 更新 best_metric（仅在突破时）
        if is_breakthrough:
            self._best_metric = current_metric

        # 7. 构建 RewardSignal
        signal = RewardSignal(
            total_reward=total_reward,
            utility_reward=utility_reward,
            exploration_cost=exploration_cost,
            state=state,
            strategy_distance=strategy_distance,
            raw_metric=current_metric,
            best_metric=self._best_metric,
            metric_delta=metric_delta,
            feedback_message=feedback_message,
        )

        logger.info(
            f"RewardSignal: state={state.value}, total={total_reward:.4f}, "
            f"utility={utility_reward:.4f}, cost={exploration_cost:.4f}, "
            f"distance={strategy_distance:.2f}, delta={metric_delta:+.4f}"
        )

        return signal

    def _generate_feedback_message(
        self,
        state: ExperimentState,
        metric_delta: float,
        strategy_distance: float,
        current_metric: float,
        best_metric: float
    ) -> str:
        """
        生成人类可读的反馈消息，直接用于 LLM prompt

        Args:
            state: 实验状态
            metric_delta: 指标差异 (current - best)
            strategy_distance: 策略距离
            current_metric: 当前指标
            best_metric: 历史最佳指标

        Returns:
            str: 反馈消息
        """
        if state == ExperimentState.BREAKTHROUGH:
            return (
                f"BREAKTHROUGH! Your hypothesis was validated.\n"
                f"- Metric improved: {best_metric:.4f} → {current_metric:.4f} ({metric_delta:+.4f})\n"
                f"- This is a new best! Analyze what made this strategy successful."
            )
        elif state == ExperimentState.STAGNATION:
            similarity = (1 - strategy_distance) * 100
            return (
                f"STAGNATION WARNING: You are repeating similar experiments without progress.\n"
                f"- Strategy similarity: {similarity:.0f}% (distance={strategy_distance:.2f})\n"
                f"- Current metric: {current_metric:.4f}, Best: {best_metric:.4f} (gap: {metric_delta:+.4f})\n"
                f"- You MUST try significantly different parameters. Small tweaks are being penalized."
            )
        else:  # EXPLORATION_FAILURE
            return (
                f"Hypothesis invalidated. Your exploration did not yield improvement.\n"
                f"- Strategy distance: {strategy_distance:.2f} (brave exploration acknowledged)\n"
                f"- Current metric: {current_metric:.4f}, Best: {best_metric:.4f} (gap: {metric_delta:+.4f})\n"
                f"- Analyze why this hypothesis failed and propose a different approach."
            )
