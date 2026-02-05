"""
Actor Agent for SIGR Framework

计算生物学家范式 (Computational Biologist Paradigm)
================================================

核心理念：Agent 是计算生物学家，通过生物学推理进行策略优化

关键改进：
1. ExperimentEntry: 记录生物学诊断、假设、预期机制
2. 动态思维模式 (Dynamic Thinking Modes):
   - FINE_TUNE: 突破后微调巩固优势
   - HIGH_ENTROPY: 停滞时强制大幅探索
   - ANALYZE_AND_PIVOT: 探索失败时分析转向
3. 生物学背景知识注入

Architecture:
- Actor: LLM-based Computational Biologist (π(a|s))
- Critic: History-based value estimator (V(s))
- ExperimentLog: Simple historical facts (替代 UCB)
"""

import logging
from typing import Dict, List, Optional, Any, Protocol, Union
from dataclasses import dataclass, field

from .strategy import Strategy, StrategyConfig, get_default_strategy, compute_strategy_distance
from .critic import SimpleCritic, AdvantageEstimate
from .prompts import (
    TASK_INITIAL_PROMPTS,
    TASK_EDGE_PRIORITIES,
    MODE_CONSTRAINTS,
    get_reflection_prompt,
    get_prompt_optimization_prompt,
    get_self_critique_prompt,
    get_consistency_check_prompt,
    get_reflection_cot_prompt,
    get_scientist_reflection_prompt,
    get_biologist_reflection_prompt,
    get_bio_cot_prompt,
    format_strategy_summary,
)

# Import biological knowledge
from .knowledge import (
    get_task_biological_context,
    get_domain_knowledge,
)

# Import hypothesis tracking
from .hypothesis import HypothesisLedger, Hypothesis, HypothesisStatus

# Import new reward types
from ..mdp.reward import RewardSignal, ExperimentState


logger = logging.getLogger(__name__)


class LLMClient(Protocol):
    """Protocol for LLM client interface."""
    def generate(self, prompt: str) -> str:
        """Generate text from prompt."""
        ...


@dataclass
class HistoryEntry:
    """Entry in the Actor's history (Legacy format)."""
    strategy: Dict[str, Any]
    reward: float
    feedback: str


@dataclass
class ExperimentEntry:
    """
    实验记录 (Scientific Experiment Entry)

    科学家做实验不仅记录"怎么做的"和"结果如何"，
    最重要的是记录"当时为什么这么做" (Rationale/Hypothesis)

    Attributes:
        iteration: 迭代编号
        strategy: 策略配置
        hypothesis: 为什么选择这个策略的假设
        expected_outcome: 预期结果
        reward_signal: 结构化奖励信号
        actual_outcome: 实际发生的结果
        root_cause_analysis: 事后反思分析
    """
    iteration: int
    strategy: Dict[str, Any]
    hypothesis: str = ""
    expected_outcome: str = ""
    reward_signal: Optional[RewardSignal] = None
    actual_outcome: str = ""
    root_cause_analysis: str = ""

    def to_experiment_report(self) -> str:
        """Format as experiment report for LLM context."""
        state_str = self.reward_signal.state.value if self.reward_signal else "UNKNOWN"
        metric_str = f"{self.reward_signal.raw_metric:.4f}" if self.reward_signal else "N/A"
        reward_str = f"{self.reward_signal.total_reward:.4f}" if self.reward_signal else "N/A"

        return (
            f"## Experiment {self.iteration}\n"
            f"**Hypothesis**: {self.hypothesis}\n"
            f"**Expected**: {self.expected_outcome}\n"
            f"**Result**: {state_str} (metric={metric_str}, reward={reward_str})\n"
            f"**Analysis**: {self.root_cause_analysis}\n"
            f"**Strategy**: edge_types={self.strategy.get('edge_types', [])}, "
            f"max_neighbors={self.strategy.get('max_neighbors', 50)}, "
            f"sampling={self.strategy.get('sampling', 'top_k')}"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'iteration': self.iteration,
            'strategy': self.strategy,
            'hypothesis': self.hypothesis,
            'expected_outcome': self.expected_outcome,
            'reward_signal': self.reward_signal.to_dict() if self.reward_signal else None,
            'actual_outcome': self.actual_outcome,
            'root_cause_analysis': self.root_cause_analysis,
        }


@dataclass
class ParameterStat:
    """Simple statistics for a parameter value."""
    count: int = 0
    total_metric: float = 0.0
    metrics: List[float] = field(default_factory=list)

    @property
    def avg_metric(self) -> float:
        return self.total_metric / self.count if self.count > 0 else 0.0

    @property
    def best_metric(self) -> float:
        return max(self.metrics) if self.metrics else 0.0

    @property
    def worst_metric(self) -> float:
        return min(self.metrics) if self.metrics else 0.0


class ExperimentLog:
    """
    简单的实验日志 (Simple Experiment Log)

    替代 UCB 的设计：提供历史事实而非行动建议。
    让 LLM 自己从事实中推断，而非被算法"指导"。

    Design Philosophy:
    - Provide FACTS, not SUGGESTIONS
    - Let LLM reason from historical data
    - No exploration/exploitation trade-off calculation
    - No "recommended" parameters
    """

    def __init__(self):
        self.entries: List[ExperimentEntry] = []
        self.parameter_stats: Dict[str, Dict[str, ParameterStat]] = {}
        self._best_metric: float = 0.0
        self._best_strategy: Optional[Dict[str, Any]] = None

    def record(self, entry: ExperimentEntry):
        """
        Record an experiment entry.

        Args:
            entry: ExperimentEntry to record
        """
        self.entries.append(entry)
        self._update_parameter_stats(entry)

        # Track best
        if entry.reward_signal and entry.reward_signal.raw_metric > self._best_metric:
            self._best_metric = entry.reward_signal.raw_metric
            self._best_strategy = entry.strategy.copy()

    def _update_parameter_stats(self, entry: ExperimentEntry):
        """Update parameter statistics from experiment entry."""
        strategy = entry.strategy
        metric = entry.reward_signal.raw_metric if entry.reward_signal else 0.0

        # Track key parameters
        tracked_params = ['edge_types', 'max_hops', 'sampling', 'max_neighbors']

        for param in tracked_params:
            if param not in strategy:
                continue

            value = strategy[param]
            # Convert to string key
            if isinstance(value, list):
                value_key = ','.join(sorted(value))
            else:
                value_key = str(value)

            if param not in self.parameter_stats:
                self.parameter_stats[param] = {}
            if value_key not in self.parameter_stats[param]:
                self.parameter_stats[param][value_key] = ParameterStat()

            stat = self.parameter_stats[param][value_key]
            stat.count += 1
            stat.total_metric += metric
            stat.metrics.append(metric)

    def get_historical_facts(self) -> str:
        """
        Generate objective historical facts (NOT suggestions).

        Returns:
            String with historical facts for LLM context
        """
        if not self.entries:
            return "## Historical Facts\nNo experiments recorded yet."

        facts = ["## Historical Facts (for reference only)"]
        facts.append(f"Total experiments: {len(self.entries)}")
        facts.append(f"Best metric achieved: {self._best_metric:.4f}")

        # Parameter-level facts
        for param, values in self.parameter_stats.items():
            if not values:
                continue

            # Sort by average metric
            sorted_values = sorted(
                values.items(),
                key=lambda x: x[1].avg_metric,
                reverse=True
            )

            if sorted_values:
                best = sorted_values[0]
                facts.append(
                    f"\n### {param}")
                facts.append(
                    f"- Best performing: {best[0]} "
                    f"(avg={best[1].avg_metric:.4f}, n={best[1].count})"
                )

                # Report consistently poor performers (tried 2+ times, below average)
                overall_avg = sum(s.avg_metric for _, s in sorted_values) / len(sorted_values)
                poor_performers = [
                    (k, s) for k, s in sorted_values
                    if s.count >= 2 and s.avg_metric < overall_avg - 0.05
                ]
                for poor_key, poor_stat in poor_performers[:2]:  # Limit to top 2
                    facts.append(
                        f"- Consistently below average: {poor_key} "
                        f"(avg={poor_stat.avg_metric:.4f}, n={poor_stat.count})"
                    )

        # Recent experiment summaries
        facts.append("\n### Recent Experiments")
        for entry in self.entries[-3:]:  # Last 3
            state = entry.reward_signal.state.value if entry.reward_signal else "?"
            metric = entry.reward_signal.raw_metric if entry.reward_signal else 0
            facts.append(
                f"- Iter {entry.iteration}: {state} (metric={metric:.4f}) "
                f"| edges={entry.strategy.get('edge_types', [])} "
                f"| sampling={entry.strategy.get('sampling')}"
            )

        return "\n".join(facts)

    def get_failure_patterns(self) -> str:
        """
        Identify patterns in failed experiments.

        Returns:
            String describing failure patterns
        """
        failures = [
            e for e in self.entries
            if e.reward_signal and e.reward_signal.state == ExperimentState.EXPLORATION_FAILURE
        ]

        if not failures:
            return ""

        patterns = ["## Failure Patterns"]
        # Analyze common parameters in failures
        failure_params: Dict[str, Dict[str, int]] = {}
        for entry in failures:
            for param in ['edge_types', 'max_hops', 'sampling']:
                if param not in entry.strategy:
                    continue
                value = entry.strategy[param]
                if isinstance(value, list):
                    value = ','.join(sorted(value))
                else:
                    value = str(value)

                if param not in failure_params:
                    failure_params[param] = {}
                failure_params[param][value] = failure_params[param].get(value, 0) + 1

        # Report high-frequency failure parameters
        for param, values in failure_params.items():
            sorted_values = sorted(values.items(), key=lambda x: -x[1])
            for val, count in sorted_values[:2]:
                if count >= 2:
                    patterns.append(f"- {param}={val} appeared in {count} failures")

        return "\n".join(patterns) if len(patterns) > 1 else ""

    def get_best_strategy(self) -> Optional[Dict[str, Any]]:
        """Get the best performing strategy."""
        return self._best_strategy

    def get_best_metric(self) -> float:
        """Get the best metric achieved."""
        return self._best_metric


class ActorAgent:
    """
    科学家风格的 Actor-Critic Agent (Scientist-style Actor-Critic Agent)

    核心改进：
    - 维护 ExperimentEntry 列表（包含假设、预期、实际结果）
    - 根据 RewardSignal.state 动态选择思维模式
    - 生成科学家风格的反思 prompt

    The Actor maintains:
    - Current strategy (Action)
    - Experiment history with hypothesis/rationale
    - LLM client for reflection
    - Critic for value estimation

    Dynamic Thinking Modes:
    - FINE_TUNE: After breakthrough, make small adjustments
    - HIGH_ENTROPY: During stagnation, force significant changes
    - ANALYZE_AND_PIVOT: After exploration failure, analyze and pivot
    """

    # Thinking mode constants
    THINKING_MODE_FINE_TUNE = "FINE_TUNE"
    THINKING_MODE_HIGH_ENTROPY = "HIGH_ENTROPY"
    THINKING_MODE_ANALYZE_AND_PIVOT = "ANALYZE_AND_PIVOT"

    def __init__(
        self,
        llm_client: LLMClient,
        task_name: str,
        initial_strategy: Optional[Strategy] = None,
        enable_self_critique: bool = True,
        enable_consistency_check: bool = True,
        enable_cot_reasoning: bool = True,
        enable_critic: bool = True,
        enable_scientist_mode: bool = True,
    ):
        """
        Initialize the Computational Biologist Actor.

        Args:
            llm_client: LLM client for generating reflections
            task_name: Name of the downstream task
            initial_strategy: Optional initial strategy (uses default if None)
            enable_self_critique: Enable two-stage self-critique (default: True)
            enable_consistency_check: Enable soft consistency guidance (default: True)
            enable_cot_reasoning: Use Chain-of-Thought prompts (default: True)
            enable_critic: Enable Critic for value estimation (default: True)
            enable_scientist_mode: Enable biologist discovery paradigm (default: True)
        """
        self.llm = llm_client
        self.task_name = task_name

        # Self-verification configuration
        self.enable_self_critique = enable_self_critique
        self.enable_consistency_check = enable_consistency_check
        self.enable_cot_reasoning = enable_cot_reasoning
        self.enable_critic = enable_critic
        self.enable_scientist_mode = enable_scientist_mode

        # Initialize strategy
        if initial_strategy is not None:
            self.current_strategy = initial_strategy
        else:
            self.current_strategy = get_default_strategy(task_name)

        # Initialize Critic
        self.critic = SimpleCritic() if enable_critic else None

        # Legacy history for backward compatibility
        self.history: List[HistoryEntry] = []
        self.metric_history: List[float] = []  # For Critic

        # 实验历史（包含生物学诊断和假设）
        self.experiment_history: List[ExperimentEntry] = []
        self.experiment_log = ExperimentLog()  # 简单的实验日志（替代 UCB）
        self.hypothesis_ledger = HypothesisLedger()  # 假设账本（Bio-CoT 核心）
        self._last_strategy: Optional[Dict[str, Any]] = None  # 用于计算策略距离
        self._current_hypothesis: str = ""  # 当前假设
        self._current_expected_outcome: str = ""  # 当前预期
        self._current_biological_diagnosis: str = ""  # 当前生物学诊断
        self._current_hypothesis_id: Optional[str] = None  # 当前假设 ID
        self._previous_metric: Optional[float] = None  # 上一次实验指标

        # Track best strategy seen
        self.best_strategy: Optional[Strategy] = None
        self.best_reward: Optional[float] = None  # None allows tracking negative rewards

        # Last reflection and advantage for logging
        self._last_reflection: str = ""
        self._last_advantage: Optional[AdvantageEstimate] = None
        self._last_thinking_mode: str = ""

        # Exploration mode state
        self._exploration_mode: bool = False
        self._exploration_perturbations: Optional[Dict[str, Any]] = None

        # Track consecutive declines for rollback mechanism
        self._consecutive_decline_count: int = 0
        self._last_raw_metric: Optional[float] = None

        logger.info(f"ActorAgent (Biologist Mode) initialized for task: {task_name}")
        logger.info(f"  Self-critique: {enable_self_critique}")
        logger.info(f"  Consistency check (soft): {enable_consistency_check}")
        logger.info(f"  CoT reasoning: {enable_cot_reasoning}")
        logger.info(f"  Critic enabled: {enable_critic}")
        logger.info(f"  Scientist mode: {enable_scientist_mode}")

    def get_strategy(self) -> Dict[str, Any]:
        """
        Get current strategy as dictionary.

        If in exploration mode with perturbations, applies them.

        Returns:
            Strategy configuration dictionary
        """
        base_strategy = self.current_strategy.to_dict()

        # Apply exploration perturbations if in explore mode
        if self._exploration_mode and self._exploration_perturbations:
            logger.debug(f"Applying exploration perturbations: {list(self._exploration_perturbations.keys())}")
            for key, value in self._exploration_perturbations.items():
                if key in base_strategy:
                    base_strategy[key] = value

        return base_strategy

    def set_exploration_mode(
        self,
        explore: bool,
        perturbations: Optional[Dict[str, Any]] = None
    ):
        """
        Set exploration mode and optional perturbations.

        Args:
            explore: Whether to enable exploration mode
            perturbations: Optional perturbations to apply to strategy
        """
        self._exploration_mode = explore
        self._exploration_perturbations = perturbations
        if explore:
            logger.debug(f"Exploration mode enabled, perturbations: {list(perturbations.keys()) if perturbations else 'None'}")
        else:
            logger.debug("Exploitation mode enabled")

    def get_strategy_config(self) -> StrategyConfig:
        """
        Get current strategy configuration.

        Returns:
            StrategyConfig object
        """
        return self.current_strategy.get_config()

    def update_policy(
        self,
        reward: float,
        feedback: str,
        raw_metric: Optional[float] = None,
        strategy_dict: Optional[Dict[str, Any]] = None,
        trend_analysis: Optional[Dict[str, Any]] = None,
        kgbook_suggestions: Optional[str] = None,
        edge_effects: Optional[Dict[str, Dict]] = None
    ):
        """
        Update policy based on reward and feedback using LLM reflection.

        This is the core of the Actor-Critic loop:
        1. Critic estimates state value V(s)
        2. Compute advantage A(a,s) = r - V(s)
        3. LLM analyzes feedback with advantage context
        4. Self-critique: LLM validates its own reasoning
        5. Consistency check: Soft guidance from best strategy (not hard override)
        6. Update current strategy

        Args:
            reward: Reward value from evaluator (normalized to [-1, 1])
            feedback: Natural language feedback from evaluator
            raw_metric: Original metric value (for best strategy tracking)
            strategy_dict: Full strategy dict including is_baseline flag
            trend_analysis: Optional trend analysis dict for trend-aware reflection
            kgbook_suggestions: Optional KGBOOK suggestions when plateaued
        """
        # Check for baseline iteration - skip policy update
        is_baseline = strategy_dict.get('is_baseline', False) if strategy_dict else False
        if is_baseline:
            logger.info("Baseline iteration: skipping policy update, metric recorded as reference")
            return

        logger.info(f"Updating policy with reward: {reward:.4f}")

        # Track consecutive declines for rollback mechanism
        if raw_metric is not None and self._last_raw_metric is not None:
            # Use 0.01 threshold (~1% relative change) to avoid false triggers from noise
            if raw_metric < self._last_raw_metric - 0.01:
                self._consecutive_decline_count += 1
                logger.debug(f"Consecutive decline count: {self._consecutive_decline_count}")
            else:
                self._consecutive_decline_count = 0

        # Rollback to best strategy after 3 consecutive declines
        if self._consecutive_decline_count >= 3 and self.best_strategy is not None:
            logger.warning(
                f"3 consecutive declines detected (current: {raw_metric:.4f}). "
                f"Rolling back to best strategy (metric: {self.best_reward:.4f})"
            )
            self.current_strategy = Strategy(StrategyConfig.from_dict(
                self.best_strategy.to_dict()
            ))
            self._consecutive_decline_count = 0
            self._last_raw_metric = raw_metric
            # Skip reflection this round - just use the known-good strategy
            self.history.append(HistoryEntry(
                strategy=self.current_strategy.to_dict(),
                reward=reward,
                feedback=feedback + "\n[ROLLBACK: Reverted to best strategy after 3 consecutive declines]"
            ))
            return

        self._last_raw_metric = raw_metric

        # Compute advantage using Critic
        advantage = None
        if self.critic and raw_metric is not None:
            self.metric_history.append(raw_metric)
            trend = trend_analysis.get('trend', 'unknown') if trend_analysis else 'unknown'
            advantage = self.critic.compute_advantage(
                reward=reward,
                metric_history=self.metric_history,
                trend=trend,
                best_metric=self.best_reward,
                current_metric=raw_metric
            )
            self._last_advantage = advantage
            logger.info(
                f"Critic advantage: A={advantage.advantage:.4f}, "
                f"V(s)={advantage.state_value:.4f}"
            )

        # Add current state to history
        self.history.append(HistoryEntry(
            strategy=self.current_strategy.to_dict(),
            reward=reward,
            feedback=feedback
        ))

        # Track best strategy using raw_metric (not relative reward)
        # This ensures we keep the strategy with best absolute performance
        if raw_metric is None:
            logger.warning(
                "raw_metric not provided to update_policy, using composite reward for best tracking. "
                "This may result in suboptimal strategy selection."
            )
        metric_for_comparison = raw_metric if raw_metric is not None else reward
        # Use None check to allow tracking negative rewards
        if self.best_reward is None or metric_for_comparison > self.best_reward:
            self.best_reward = metric_for_comparison
            self.best_strategy = Strategy(StrategyConfig.from_dict(
                self.current_strategy.to_dict()
            ))
            logger.info(f"New best strategy found with metric: {metric_for_comparison:.4f}")

        # Generate reflection prompt (use CoT if enabled)
        # Include advantage information if available
        advantage_context = ""
        if advantage:
            advantage_context = (
                f"\n\nCritic Assessment:\n"
                f"- State Value V(s): {advantage.state_value:.4f}\n"
                f"- Advantage A(a,s): {advantage.advantage:.4f}\n"
                f"- Interpretation: {'Strategy outperformed baseline' if advantage.advantage > 0 else 'Strategy underperformed baseline'}"
            )

        # Combine all context for LLM
        enhanced_feedback = feedback + advantage_context

        # Add KGBOOK suggestions if available (when plateaued)
        if kgbook_suggestions:
            enhanced_feedback += f"\n\n{kgbook_suggestions}"
            logger.info("KGBOOK suggestions added to reflection context")

        if self.enable_cot_reasoning:
            reflection_prompt = get_reflection_cot_prompt(
                task_name=self.task_name,
                strategy_dict=self.current_strategy.to_dict(),
                reward=reward,
                feedback=enhanced_feedback,
                history=[
                    {'strategy': h.strategy, 'reward': h.reward, 'feedback': h.feedback}
                    for h in self.history
                ],
                trend_analysis=trend_analysis,
                best_reward=self.best_reward,
                edge_effects=edge_effects
            )
        else:
            reflection_prompt = get_reflection_prompt(
                task_name=self.task_name,
                strategy_dict=self.current_strategy.to_dict(),
                reward=reward,
                feedback=enhanced_feedback,
                history=[
                    {'strategy': h.strategy, 'reward': h.reward, 'feedback': h.feedback}
                    for h in self.history
                ],
                trend_analysis=trend_analysis
            )

        # Get LLM reflection
        # Always use strong model for Actor reasoning tasks
        try:
            if hasattr(self.llm, 'generate_strong'):
                # Use strong model for reflection (reasoning task)
                logger.info("Using strong model for reflection...")
                reflection_response = self.llm.generate_strong(reflection_prompt)
            else:
                # Fallback to regular generate (single model mode)
                logger.info("Using default model for reflection (no strong model available)")
                reflection_response = self.llm.generate(reflection_prompt)

            # Stage 2: Self-critique (if enabled)
            if self.enable_self_critique:
                logger.info("Running self-critique...")
                reflection_response = self._apply_self_critique(reflection_response)

            self._last_reflection = reflection_response

            # Parse new strategy from response (with retry if enabled)
            new_strategy = Strategy.parse_with_retry(
                reflection_response,
                llm_client=self.llm,
                fallback_strategy=self.current_strategy,  # Use current as fallback
                max_retries=2
            )

            # Stage 3: Soft consistency guidance (if enabled)
            # Changed from hard override to soft guidance
            if self.enable_consistency_check and self.best_strategy is not None:
                logger.info("Running soft consistency guidance...")
                new_strategy = self._apply_soft_consistency_guidance(new_strategy, advantage)

            # Preserve prompt template unless we need to optimize it
            # Only optimize prompt when performance is consistently poor
            if reward < 0.5 and len(self.history) >= 3:
                # Low performance for multiple iterations, optimize prompt
                new_prompt = self._optimize_prompt(reward, feedback)
                new_strategy.update(prompt_template=new_prompt)
            else:
                # Keep current prompt
                new_strategy.update(
                    prompt_template=self.current_strategy.get_config().prompt_template
                )

            self.current_strategy = new_strategy
            logger.info(f"Strategy updated: {new_strategy.get_config().edge_types}")

        except Exception as e:
            logger.error(f"Error during policy update: {e}")
            self._last_reflection = f"Error: {e}"
            # Keep current strategy on error

    def update_policy_scientist(
        self,
        reward_signal: RewardSignal,
        strategy_dict: Optional[Dict[str, Any]] = None,
        trend_analysis: Optional[Dict[str, Any]] = None,
        kgbook_suggestions: Optional[str] = None,
        edge_effects: Optional[Dict[str, Dict]] = None
    ):
        """
        科学家模式的策略更新 (Scientist-style Policy Update)

        根据 RewardSignal 的状态动态选择思维模式:
        - BREAKTHROUGH: FINE_TUNE 模式，微调巩固优势
        - STAGNATION: HIGH_ENTROPY 模式，强制大幅探索
        - EXPLORATION_FAILURE: ANALYZE_AND_PIVOT 模式，分析转向

        Args:
            reward_signal: 结构化奖励信号
            strategy_dict: 完整策略字典
            trend_analysis: 趋势分析
            kgbook_suggestions: KGBOOK 建议
            edge_effects: 边类型效果
        """
        # Check for baseline iteration
        is_baseline = strategy_dict.get('is_baseline', False) if strategy_dict else False
        if is_baseline:
            logger.info("Baseline iteration: skipping policy update")
            return

        logger.info(
            f"Scientist mode policy update: state={reward_signal.state.value}, "
            f"reward={reward_signal.total_reward:.4f}"
        )

        # 1. 确定思维模式 (Determine Thinking Mode)
        thinking_mode = self._determine_thinking_mode(reward_signal)
        self._last_thinking_mode = thinking_mode
        logger.info(f"Thinking mode: {thinking_mode}")

        # 2. 更新实验历史 (Update Experiment History)
        experiment_entry = ExperimentEntry(
            iteration=len(self.experiment_history) + 1,
            strategy=self.current_strategy.to_dict(),
            hypothesis=self._current_hypothesis,
            expected_outcome=self._current_expected_outcome,
            reward_signal=reward_signal,
            actual_outcome=reward_signal.state.value,
            root_cause_analysis=""  # Will be filled by reflection
        )
        self.experiment_history.append(experiment_entry)

        # 3. 更新 legacy history (for backward compatibility)
        self.history.append(HistoryEntry(
            strategy=self.current_strategy.to_dict(),
            reward=reward_signal.total_reward,
            feedback=reward_signal.feedback_message
        ))

        # 4. 更新 metric_history
        if reward_signal.raw_metric is not None:
            self.metric_history.append(reward_signal.raw_metric)

        # 5. 更新 best_strategy
        if self.best_reward is None or reward_signal.raw_metric > self.best_reward:
            self.best_reward = reward_signal.raw_metric
            self.best_strategy = Strategy(StrategyConfig.from_dict(
                self.current_strategy.to_dict()
            ))
            logger.info(f"New best strategy found with metric: {reward_signal.raw_metric:.4f}")

        # 6. 生成科学家风格的反思 prompt
        reflection_prompt = get_scientist_reflection_prompt(
            thinking_mode=thinking_mode,
            reward_signal=reward_signal,
            experiment_history=self.experiment_history,
            task_name=self.task_name,
            current_strategy=self.current_strategy.to_dict(),
            best_metric=self.best_reward,
            trend_analysis=trend_analysis,
            edge_effects=edge_effects,
            kgbook_suggestions=kgbook_suggestions,
        )

        # 7. LLM 反思
        try:
            if hasattr(self.llm, 'generate_strong'):
                logger.info("Using strong model for scientist reflection...")
                reflection_response = self.llm.generate_strong(reflection_prompt)
            else:
                reflection_response = self.llm.generate(reflection_prompt)

            # Self-critique if enabled
            if self.enable_self_critique:
                logger.info("Running self-critique...")
                reflection_response = self._apply_self_critique(reflection_response)

            self._last_reflection = reflection_response

            # 8. 解析新策略和假设
            new_strategy, hypothesis, expected_outcome = self._parse_scientist_response(
                reflection_response
            )

            # 存储下一次迭代的假设
            self._current_hypothesis = hypothesis
            self._current_expected_outcome = expected_outcome

            # 9. 验证策略距离满足思维模式要求
            new_strategy = self._validate_strategy_for_mode(
                new_strategy, thinking_mode, reward_signal.strategy_distance
            )

            # 10. 保持 prompt template
            new_strategy.update(
                prompt_template=self.current_strategy.get_config().prompt_template
            )

            # 更新策略
            self._last_strategy = self.current_strategy.to_dict()
            self.current_strategy = new_strategy
            logger.info(f"Strategy updated: {new_strategy.get_config().edge_types}")

        except Exception as e:
            logger.error(f"Error during scientist policy update: {e}")
            self._last_reflection = f"Error: {e}"

    def update_policy_biologist(
        self,
        reward_signal: RewardSignal,
        strategy_dict: Optional[Dict[str, Any]] = None,
        trend_analysis: Optional[Dict[str, Any]] = None,
        edge_effects: Optional[Dict[str, Dict]] = None
    ):
        """
        Bio-CoT 模式的策略更新 (Bio-CoT Policy Update)

        核心改变：
        1. 使用 HypothesisLedger 追踪假设的验证/证伪
        2. 强制 Bio-CoT 推理链：Observation → Diagnosis → Hypothesis → Design → Falsification
        3. 参数选择必须从生物学假设推导

        Args:
            reward_signal: 结构化奖励信号
            strategy_dict: 完整策略字典
            trend_analysis: 趋势分析
            edge_effects: 边类型效果
        """
        # Check for baseline iteration
        is_baseline = strategy_dict.get('is_baseline', False) if strategy_dict else False
        if is_baseline:
            logger.info("Baseline iteration: skipping policy update")
            return

        current_iteration = len(self.experiment_history) + 1
        logger.info(
            f"Bio-CoT policy update [Iteration {current_iteration}]: "
            f"state={reward_signal.state.value}, metric={reward_signal.raw_metric:.4f}"
        )

        # 1. 验证/证伪上一个假设 (Validate/Invalidate previous hypothesis)
        if self._current_hypothesis_id and self._previous_metric is not None:
            is_validated, evidence = self.hypothesis_ledger.evaluate_hypothesis(
                self._current_hypothesis_id,
                reward_signal.raw_metric,
                self._previous_metric
            )

            if is_validated:
                self.hypothesis_ledger.validate(
                    self._current_hypothesis_id,
                    current_iteration,
                    evidence
                )
                logger.info(f"Hypothesis {self._current_hypothesis_id} VALIDATED: {evidence}")
            else:
                self.hypothesis_ledger.invalidate(
                    self._current_hypothesis_id,
                    current_iteration,
                    evidence
                )
                logger.info(f"Hypothesis {self._current_hypothesis_id} INVALIDATED: {evidence}")

        # 2. 确定思维模式 (Determine Thinking Mode)
        thinking_mode = self._determine_thinking_mode(reward_signal)
        self._last_thinking_mode = thinking_mode
        logger.info(f"Thinking mode: {thinking_mode}")

        # 3. 更新实验历史 (Update Experiment History)
        experiment_entry = ExperimentEntry(
            iteration=current_iteration,
            strategy=self.current_strategy.to_dict(),
            hypothesis=self._current_hypothesis,
            expected_outcome=self._current_expected_outcome,
            reward_signal=reward_signal,
            actual_outcome=reward_signal.state.value,
            root_cause_analysis=self._current_biological_diagnosis
        )
        self.experiment_history.append(experiment_entry)
        self.experiment_log.record(experiment_entry)

        # 4. 更新 legacy history (for backward compatibility)
        self.history.append(HistoryEntry(
            strategy=self.current_strategy.to_dict(),
            reward=reward_signal.total_reward,
            feedback=reward_signal.feedback_message
        ))

        # 5. 更新 metric_history
        if reward_signal.raw_metric is not None:
            self.metric_history.append(reward_signal.raw_metric)

        # 6. 更新 best_strategy
        if self.best_reward is None or reward_signal.raw_metric > self.best_reward:
            self.best_reward = reward_signal.raw_metric
            self.best_strategy = Strategy(StrategyConfig.from_dict(
                self.current_strategy.to_dict()
            ))
            logger.info(f"New best strategy found with metric: {reward_signal.raw_metric:.4f}")

        # 7. 获取生物学背景知识
        task_biological_context = get_task_biological_context(self.task_name)

        # 8. 获取假设账本摘要
        hypothesis_ledger_summary = self.hypothesis_ledger.get_knowledge_summary()
        failure_patterns = self.hypothesis_ledger.get_failure_patterns()

        # 9. 获取思维模式约束 (已在顶部导入)
        mode_constraints = MODE_CONSTRAINTS.get(thinking_mode, "")

        # 10. 生成 Bio-CoT Prompt
        strategy_summary = format_strategy_summary(self.current_strategy.to_dict())
        # 安全获取指标值
        prev_metric_val = self._previous_metric if self._previous_metric is not None else 0.0
        curr_metric_val = reward_signal.raw_metric if reward_signal.raw_metric is not None else 0.0
        reflection_prompt = get_bio_cot_prompt(
            task_name=self.task_name,
            task_biological_context=task_biological_context,
            prev_metric=prev_metric_val,
            curr_metric=curr_metric_val,
            experiment_state=reward_signal.state.value,
            strategy_summary=strategy_summary,
            hypothesis_ledger_summary=hypothesis_ledger_summary,
            failure_patterns=failure_patterns,
            mode_constraints=mode_constraints,
        )

        # 11. LLM 反思 (Bio-CoT reasoning)
        try:
            if hasattr(self.llm, 'generate_strong'):
                logger.info("Using strong model for Bio-CoT reasoning...")
                reflection_response = self.llm.generate_strong(reflection_prompt)
            else:
                reflection_response = self.llm.generate(reflection_prompt)

            # Self-critique if enabled
            if self.enable_self_critique:
                logger.info("Running self-critique...")
                reflection_response = self._apply_self_critique(reflection_response)

            self._last_reflection = reflection_response

            # 12. 解析 Bio-CoT 响应
            parsed = self._parse_bio_cot_response(reflection_response)
            new_strategy = parsed['strategy']
            hypothesis_data = parsed['hypothesis']
            bio_diagnosis = parsed['biological_diagnosis']
            falsification_criteria = parsed['falsification_criteria']

            # 13. 注册新假设到账本
            hypothesis_statement = hypothesis_data.get('statement', '')
            if hypothesis_statement and hypothesis_statement.strip():
                h_id = self.hypothesis_ledger.propose(
                    statement=hypothesis_statement,
                    biological_basis=hypothesis_data.get('biological_basis', ''),
                    expected_outcome=hypothesis_data.get('expected_outcome', ''),
                    falsification_criteria=falsification_criteria,
                    iteration=current_iteration + 1,  # 下一次迭代验证
                    strategy=new_strategy.to_dict()
                )
                if h_id:  # 验证 h_id 有效
                    self._current_hypothesis_id = h_id
                    self._current_hypothesis = hypothesis_statement
                    self._current_expected_outcome = hypothesis_data.get('expected_outcome', '')
                    self._current_biological_diagnosis = bio_diagnosis
                    logger.info(f"New hypothesis registered: {h_id}")
                else:
                    logger.warning("Failed to register hypothesis, h_id is None")
            else:
                logger.warning("No valid hypothesis statement found, skipping registration")

            # 14. 验证策略距离满足思维模式要求
            new_strategy = self._validate_strategy_for_mode(
                new_strategy, thinking_mode, reward_signal.strategy_distance
            )

            # 15. 保持 prompt template
            new_strategy.update(
                prompt_template=self.current_strategy.get_config().prompt_template
            )

            # 更新策略和指标
            self._last_strategy = self.current_strategy.to_dict()
            self._previous_metric = reward_signal.raw_metric
            self.current_strategy = new_strategy
            logger.info(f"Strategy updated: {new_strategy.get_config().edge_types}")

        except Exception as e:
            logger.error(f"Error during Bio-CoT policy update: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self._last_reflection = f"Error: {e}"

    def _parse_bio_cot_response(self, response: str) -> Dict[str, Any]:
        """
        解析 Bio-CoT 响应

        期望的 JSON 格式：
        {
            "observation": {...},
            "biological_diagnosis": "...",
            "hypothesis": {
                "statement": "...",
                "biological_basis": "...",
                "expected_outcome": "..."
            },
            "experiment_design": {
                "rationale": "...",
                "edge_types": [...],
                ...
            },
            "falsification_criteria": "..."
        }

        Args:
            response: LLM 响应文本

        Returns:
            Dict: 解析后的响应数据
        """
        import json
        import re

        result = {
            'strategy': self.current_strategy,
            'hypothesis': {},
            'biological_diagnosis': '',
            'falsification_criteria': '',
        }

        # 尝试从 JSON 块中提取
        json_match = re.search(r'```json\s*([\s\S]*?)\s*```', response)
        if json_match:
            try:
                json_str = json_match.group(1)
                data = json.loads(json_str)

                # 提取生物学诊断
                result['biological_diagnosis'] = data.get('biological_diagnosis', '')

                # 提取假设
                hypothesis = data.get('hypothesis', {})
                if isinstance(hypothesis, dict):
                    result['hypothesis'] = hypothesis
                elif isinstance(hypothesis, str):
                    result['hypothesis'] = {'statement': hypothesis}

                # 提取证伪条件
                result['falsification_criteria'] = data.get('falsification_criteria', '')

                # 提取实验设计并转换为策略
                experiment_design = data.get('experiment_design', {})
                if experiment_design:
                    strategy_dict = self.current_strategy.to_dict()

                    # 从 experiment_design 更新策略参数
                    param_mapping = [
                        'edge_types', 'max_hops', 'sampling', 'max_neighbors',
                        'description_length', 'description_focus', 'context_window',
                        'prompt_style', 'feature_selection'
                    ]
                    for param in param_mapping:
                        if param in experiment_design:
                            strategy_dict[param] = experiment_design[param]

                    result['strategy'] = Strategy(StrategyConfig.from_dict(strategy_dict))

                return result

            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Failed to parse JSON from Bio-CoT response: {e}")

        # 回退到正则表达式提取
        logger.info("Falling back to regex extraction for Bio-CoT response")

        # 提取生物学诊断 (支持转义引号)
        bio_match = re.search(r'"biological_diagnosis":\s*"((?:[^"\\]|\\.)*)"', response)
        if bio_match:
            result['biological_diagnosis'] = bio_match.group(1).replace('\\"', '"')

        # 提取假设声明 (支持转义引号)
        stmt_match = re.search(r'"statement":\s*"((?:[^"\\]|\\.)*)"', response)
        if stmt_match:
            result['hypothesis']['statement'] = stmt_match.group(1).replace('\\"', '"')

        basis_match = re.search(r'"biological_basis":\s*"((?:[^"\\]|\\.)*)"', response)
        if basis_match:
            result['hypothesis']['biological_basis'] = basis_match.group(1).replace('\\"', '"')

        outcome_match = re.search(r'"expected_outcome":\s*"((?:[^"\\]|\\.)*)"', response)
        if outcome_match:
            result['hypothesis']['expected_outcome'] = outcome_match.group(1).replace('\\"', '"')

        # 提取证伪条件 (支持转义引号)
        falsif_match = re.search(r'"falsification_criteria":\s*"((?:[^"\\]|\\.)*)"', response)
        if falsif_match:
            result['falsification_criteria'] = falsif_match.group(1).replace('\\"', '"')

        # 使用现有的策略解析方法作为后备
        result['strategy'] = Strategy.parse_with_retry(
            response,
            llm_client=self.llm,
            fallback_strategy=self.current_strategy,
            max_retries=2
        )

        return result

    def _determine_thinking_mode(self, reward_signal: RewardSignal) -> str:
        """
        根据 RewardSignal 状态确定思维模式

        Args:
            reward_signal: 结构化奖励信号

        Returns:
            str: 思维模式 (FINE_TUNE / HIGH_ENTROPY / ANALYZE_AND_PIVOT)
        """
        if reward_signal.state == ExperimentState.BREAKTHROUGH:
            return self.THINKING_MODE_FINE_TUNE
        elif reward_signal.state == ExperimentState.STAGNATION:
            return self.THINKING_MODE_HIGH_ENTROPY
        elif reward_signal.state == ExperimentState.EXPLORATION_FAILURE:
            return self.THINKING_MODE_ANALYZE_AND_PIVOT
        else:
            # 未知状态，默认使用分析转向模式
            logger.warning(f"Unknown experiment state: {reward_signal.state}, defaulting to ANALYZE_AND_PIVOT")
            return self.THINKING_MODE_ANALYZE_AND_PIVOT

    def _parse_scientist_response(
        self,
        response: str
    ) -> tuple[Strategy, str, str]:
        """
        解析科学家风格的 LLM 响应

        Args:
            response: LLM 响应文本

        Returns:
            tuple: (new_strategy, hypothesis, expected_outcome)
        """
        import json
        import re

        hypothesis = ""
        expected_outcome = ""

        # Try to extract hypothesis and expected_outcome from response
        hypothesis_match = re.search(r'"hypothesis":\s*"([^"]*)"', response)
        if hypothesis_match:
            hypothesis = hypothesis_match.group(1)

        expected_match = re.search(r'"expected_outcome":\s*"([^"]*)"', response)
        if expected_match:
            expected_outcome = expected_match.group(1)

        # Parse strategy using existing method
        new_strategy = Strategy.parse_with_retry(
            response,
            llm_client=self.llm,
            fallback_strategy=self.current_strategy,
            max_retries=2
        )

        return new_strategy, hypothesis, expected_outcome

    def _validate_strategy_for_mode(
        self,
        proposed_strategy: Strategy,
        thinking_mode: str,
        current_distance: float
    ) -> Strategy:
        """
        验证策略是否满足思维模式的要求

        HIGH_ENTROPY 模式要求策略变化大 (distance > 0.3)
        FINE_TUNE 模式要求策略变化小 (distance < 0.2)

        Args:
            proposed_strategy: 提议的新策略
            thinking_mode: 当前思维模式
            current_distance: 当前策略距离

        Returns:
            Strategy: 验证后的策略（可能被调整）
        """
        proposed_dict = proposed_strategy.to_dict()
        current_dict = self.current_strategy.to_dict()

        # Compute distance to proposed strategy
        proposed_distance = compute_strategy_distance(proposed_dict, current_dict)

        if thinking_mode == self.THINKING_MODE_HIGH_ENTROPY:
            # HIGH_ENTROPY 模式需要大变化
            if proposed_distance < 0.3:
                logger.warning(
                    f"HIGH_ENTROPY mode but proposed distance={proposed_distance:.2f} < 0.3. "
                    f"Forcing larger changes."
                )
                # 强制增加变化：切换一些参数
                if proposed_dict.get('sampling') == current_dict.get('sampling'):
                    # 切换 sampling 方法
                    sampling_options = ['top_k', 'random', 'weighted']
                    current_sampling = current_dict.get('sampling', 'top_k')
                    other_samplings = [s for s in sampling_options if s != current_sampling]
                    if other_samplings:
                        new_sampling = other_samplings[0]
                        proposed_strategy.update(sampling=new_sampling)

                # 增大 max_neighbors 变化
                current_neighbors = current_dict.get('max_neighbors', 50)
                proposed_neighbors = proposed_dict.get('max_neighbors', 50)
                if abs(proposed_neighbors - current_neighbors) < 30:
                    new_neighbors = min(200, max(10, current_neighbors + 40))
                    proposed_strategy.update(max_neighbors=new_neighbors)

                logger.info("Forced larger changes for HIGH_ENTROPY mode")

        elif thinking_mode == self.THINKING_MODE_FINE_TUNE:
            # FINE_TUNE 模式需要小变化
            if proposed_distance > 0.4:
                logger.warning(
                    f"FINE_TUNE mode but proposed distance={proposed_distance:.2f} > 0.4. "
                    f"Suggesting smaller changes."
                )
                # 不强制修改，只是警告（允许 LLM 决定）

        return proposed_strategy

    def get_last_strategy(self) -> Optional[Dict[str, Any]]:
        """获取上一个策略（用于计算策略距离）"""
        return self._last_strategy

    def get_last_thinking_mode(self) -> str:
        """获取上次使用的思维模式"""
        return self._last_thinking_mode

    def get_experiment_history(self) -> List[Dict[str, Any]]:
        """获取实验历史"""
        return [entry.to_dict() for entry in self.experiment_history]

    def get_hypothesis_ledger(self) -> HypothesisLedger:
        """获取假设账本"""
        return self.hypothesis_ledger

    def get_hypothesis_summary(self) -> str:
        """获取假设账本的知识摘要"""
        return self.hypothesis_ledger.get_knowledge_summary()

    def get_current_hypothesis(self) -> Optional[Dict[str, Any]]:
        """获取当前活跃的假设"""
        h = self.hypothesis_ledger.get_current_hypothesis()
        return h.to_dict() if h else None

    def _apply_self_critique(self, initial_response: str) -> str:
        """
        Apply self-critique to validate the initial analysis.

        Args:
            initial_response: The initial reflection response from LLM

        Returns:
            Critiqued and potentially revised response
        """
        critique_prompt = get_self_critique_prompt(initial_response)

        try:
            if hasattr(self.llm, 'generate_strong'):
                critiqued_response = self.llm.generate_strong(critique_prompt)
            else:
                critiqued_response = self.llm.generate(critique_prompt)

            # Store critique for logging
            self._last_critique = critiqued_response

            # Log critique result
            if 'confidence' in critiqued_response.lower():
                logger.debug("Self-critique completed with confidence assessment")

            return critiqued_response

        except Exception as e:
            logger.warning(f"Self-critique failed: {e}, using original response")
            self._last_critique = f"Error: {e}"
            return initial_response

    def get_last_critique(self) -> str:
        """Get the last self-critique response."""
        return getattr(self, '_last_critique', "")

    def _apply_consistency_check(self, proposed_strategy: Strategy) -> Strategy:
        """
        DEPRECATED: Use _apply_soft_consistency_guidance instead.

        This hard consistency check has been replaced with soft guidance
        to avoid suppressing exploration.
        """
        logger.warning("_apply_consistency_check is deprecated, using soft guidance")
        return self._apply_soft_consistency_guidance(proposed_strategy, None)

    def _apply_soft_consistency_guidance(
        self,
        proposed_strategy: Strategy,
        advantage: Optional[AdvantageEstimate] = None
    ) -> Strategy:
        """
        Apply soft consistency guidance from historical best strategy.

        Unlike the original hard consistency check that would override changes,
        this version provides guidance but allows exploration:
        - If advantage > 0 (outperforming baseline): Allow more deviation
        - If advantage < 0 (underperforming): Suggest staying closer to best
        - Never hard-override the proposed strategy

        Args:
            proposed_strategy: The proposed new strategy
            advantage: Advantage estimate from Critic (if available)

        Returns:
            Possibly adjusted strategy (never hard-overridden)
        """
        if self.best_strategy is None:
            return proposed_strategy

        # Calculate how different the proposed strategy is from best
        proposed_dict = proposed_strategy.to_dict()
        best_dict = self.best_strategy.to_dict()

        # Check key dimensions
        changes = []
        if proposed_dict.get('edge_types') != best_dict.get('edge_types'):
            changes.append('edge_types')
        if proposed_dict.get('max_hops') != best_dict.get('max_hops'):
            changes.append('max_hops')
        if abs(proposed_dict.get('max_neighbors', 0) - best_dict.get('max_neighbors', 0)) > 20:
            changes.append('max_neighbors')
        if proposed_dict.get('sampling') != best_dict.get('sampling'):
            changes.append('sampling')

        if not changes:
            logger.debug("Proposed strategy similar to best, no guidance needed")
            return proposed_strategy

        # Determine guidance level based on advantage
        if advantage is not None:
            if advantage.advantage > 0.1:
                # Outperforming baseline significantly - allow exploration
                logger.info(
                    f"Positive advantage ({advantage.advantage:.4f}): "
                    f"allowing exploration with changes: {changes}"
                )
                return proposed_strategy
            elif advantage.advantage < -0.2:
                # Significantly underperforming - suggest closer to best
                # But still allow the change (soft guidance)
                logger.info(
                    f"Negative advantage ({advantage.advantage:.4f}): "
                    f"suggesting caution with changes: {changes}"
                )
                # Could add a prompt here to reconsider, but we trust the LLM
                return proposed_strategy

        # Neutral advantage or no Critic - log but allow
        logger.debug(f"Soft guidance: allowing changes {changes} from best strategy")
        return proposed_strategy

    def _optimize_prompt(self, reward: float, feedback: str) -> str:
        """
        Use LLM to optimize the generation prompt.

        Args:
            reward: Current reward value
            feedback: Feedback from evaluator

        Returns:
            Optimized prompt template
        """
        logger.info("Optimizing prompt template...")

        current_prompt = self.current_strategy.get_config().prompt_template

        optimization_prompt = get_prompt_optimization_prompt(
            task_name=self.task_name,
            current_prompt=current_prompt,
            feedback=feedback,
            reward=reward
        )

        try:
            # Use strong model for prompt optimization (reasoning task)
            if hasattr(self.llm, 'generate_strong'):
                logger.info("Using strong model for prompt optimization...")
                new_prompt = self.llm.generate_strong(optimization_prompt)
            else:
                new_prompt = self.llm.generate(optimization_prompt)

            # Basic validation - ensure it has required placeholders
            required_placeholders = ['{gene_id}', '{gene_name}']
            if all(p in new_prompt for p in required_placeholders):
                logger.info("Prompt template optimized successfully")
                return new_prompt
            else:
                logger.warning("Optimized prompt missing required placeholders, keeping original")
                return current_prompt

        except Exception as e:
            logger.error(f"Error optimizing prompt: {e}")
            return current_prompt

    def get_last_reflection(self) -> str:
        """
        Get the last reflection from the LLM.

        Returns:
            Last reflection text
        """
        return self._last_reflection

    def get_last_advantage(self) -> Optional[AdvantageEstimate]:
        """
        Get the last advantage estimate from Critic.

        Returns:
            Last AdvantageEstimate or None if Critic disabled
        """
        return self._last_advantage

    def get_critic_state_value(self) -> Optional[float]:
        """
        Get the current state value estimate from Critic.

        Returns:
            V(s) estimate or None if Critic disabled
        """
        if self.critic and self.metric_history:
            state_value = self.critic.estimate_value(
                self.metric_history,
                trend='unknown',
                best_metric=self.best_reward
            )
            return state_value.value
        return None

    def get_best_strategy(self) -> Optional[Dict[str, Any]]:
        """
        Get the best strategy seen so far.

        Returns:
            Best strategy dictionary or None if no iterations yet
        """
        if self.best_strategy is not None:
            return self.best_strategy.to_dict()
        return None

    def get_history(self) -> List[Dict[str, Any]]:
        """
        Get the full history of strategies and rewards.

        Returns:
            List of history entries as dictionaries
        """
        return [
            {
                'strategy': h.strategy,
                'reward': h.reward,
                'feedback': h.feedback[:500]  # Truncate for readability
            }
            for h in self.history
        ]

    def reset(self):
        """Reset the agent to initial state."""
        self.current_strategy = get_default_strategy(self.task_name)
        self.history = []
        self.metric_history = []
        self.experiment_history = []
        self.experiment_log = ExperimentLog()
        self.hypothesis_ledger.reset()
        self.best_strategy = None
        self.best_reward = None  # None to allow tracking negative rewards
        self._last_reflection = ""
        self._last_advantage = None
        self._last_thinking_mode = ""
        self._consecutive_decline_count = 0
        self._last_raw_metric = None
        self._current_hypothesis = ""
        self._current_expected_outcome = ""
        self._current_biological_diagnosis = ""
        self._current_hypothesis_id = None
        self._previous_metric = None
        if self.critic:
            self.critic.reset()
        logger.info("ActorAgent reset")

    def save_state(self) -> Dict[str, Any]:
        """
        Save agent state for persistence.

        Returns:
            Dictionary containing agent state
        """
        state = {
            'task_name': self.task_name,
            'current_strategy': self.current_strategy.to_dict(),
            'best_strategy': self.best_strategy.to_dict() if self.best_strategy else None,
            'best_reward': self.best_reward,
            'metric_history': list(self.metric_history),
            'history': [
                {'strategy': h.strategy, 'reward': h.reward, 'feedback': h.feedback}
                for h in self.history
            ],
            # Bio-CoT 相关状态
            'hypothesis_ledger': self.hypothesis_ledger.save_state(),
            'experiment_history': [e.to_dict() for e in self.experiment_history],
            'current_hypothesis': self._current_hypothesis,
            'current_expected_outcome': self._current_expected_outcome,
            'current_biological_diagnosis': self._current_biological_diagnosis,
            'current_hypothesis_id': self._current_hypothesis_id,
            'previous_metric': self._previous_metric,
            'last_thinking_mode': self._last_thinking_mode,
        }
        if self.critic:
            state['critic_state'] = self.critic.get_state()
        return state

    def load_state(self, state: Dict[str, Any]):
        """
        Load agent state from saved data.

        Args:
            state: Dictionary containing agent state
        """
        self.task_name = state['task_name']
        self.current_strategy = Strategy(StrategyConfig.from_dict(state['current_strategy']))
        self.best_reward = state['best_reward']
        self.metric_history = state.get('metric_history', [])

        if state['best_strategy']:
            self.best_strategy = Strategy(StrategyConfig.from_dict(state['best_strategy']))

        self.history = [
            HistoryEntry(
                strategy=h['strategy'],
                reward=h['reward'],
                feedback=h['feedback']
            )
            for h in state['history']
        ]

        # 加载 Bio-CoT 相关状态
        if 'hypothesis_ledger' in state:
            self.hypothesis_ledger.load_state(state['hypothesis_ledger'])

        if 'experiment_history' in state:
            from ..mdp.reward import RewardSignal
            self.experiment_history = []
            for e_dict in state['experiment_history']:
                entry = ExperimentEntry(
                    iteration=e_dict['iteration'],
                    strategy=e_dict['strategy'],
                    hypothesis=e_dict.get('hypothesis', ''),
                    expected_outcome=e_dict.get('expected_outcome', ''),
                    reward_signal=RewardSignal.from_dict(e_dict['reward_signal']) if e_dict.get('reward_signal') else None,
                    actual_outcome=e_dict.get('actual_outcome', ''),
                    root_cause_analysis=e_dict.get('root_cause_analysis', ''),
                )
                self.experiment_history.append(entry)

        self._current_hypothesis = state.get('current_hypothesis', '')
        self._current_expected_outcome = state.get('current_expected_outcome', '')
        self._current_biological_diagnosis = state.get('current_biological_diagnosis', '')
        self._current_hypothesis_id = state.get('current_hypothesis_id')
        self._previous_metric = state.get('previous_metric')
        self._last_thinking_mode = state.get('last_thinking_mode', '')

        if self.critic and 'critic_state' in state:
            self.critic.load_state(state['critic_state'])

        logger.info(f"ActorAgent state loaded for task: {self.task_name}")


class MockLLMClient:
    """
    Mock LLM client for testing without actual LLM calls.

    Returns predefined responses for strategy updates.
    """

    def __init__(self, task_name: str = 'ppi'):
        self.task_name = task_name
        self.call_count = 0

    def generate(self, prompt: str) -> str:
        """Generate mock response."""
        self.call_count += 1

        # Return a valid strategy JSON
        if 'improved prompt template' in prompt.lower() or 'new prompt template' in prompt.lower():
            # Prompt optimization request
            return TASK_INITIAL_PROMPTS.get(self.task_name, "")

        # Strategy update request
        import json
        response = {
            "edge_types": TASK_EDGE_PRIORITIES.get(self.task_name, ['PPI', 'GO', 'HPO']),
            "max_hops": 2,
            "sampling": "top_k",
            "max_neighbors": 60 + self.call_count * 10,  # Gradually increase
            "reasoning": f"Iteration {self.call_count}: Adjusting neighbors for better coverage"
        }
        return f"```json\n{json.dumps(response, indent=2)}\n```"
