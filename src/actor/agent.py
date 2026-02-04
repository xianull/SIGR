"""
Actor Agent for SIGR Framework

Implements Actor-Critic based policy optimization using LLM reflection.
The Actor decides the KG extraction strategy and prompt construction.
The Critic estimates state values for advantage computation.

Architecture:
- Actor: LLM-based strategy proposer (π(a|s))
- Critic: History-based value estimator (V(s))
- Advantage: A(a,s) = r - V(s) for variance reduction
- ParameterTracker: Tracks effect of each parameter on reward
- UCBSelector: Intelligent exploration-exploitation for discrete params
"""

import logging
from typing import Dict, List, Optional, Any, Protocol, Union
from dataclasses import dataclass

from .strategy import Strategy, StrategyConfig, get_default_strategy
from .critic import SimpleCritic, AdvantageEstimate
from .tracker import ParameterEffectTracker
from .ucb import UCBSelector, MultiArmedBandit
from .prompts import (
    TASK_INITIAL_PROMPTS,
    TASK_EDGE_PRIORITIES,
    get_reflection_prompt,
    get_prompt_optimization_prompt,
    get_self_critique_prompt,
    get_consistency_check_prompt,
    get_reflection_cot_prompt
)


logger = logging.getLogger(__name__)


class LLMClient(Protocol):
    """Protocol for LLM client interface."""
    def generate(self, prompt: str) -> str:
        """Generate text from prompt."""
        ...


@dataclass
class HistoryEntry:
    """Entry in the Actor's history."""
    strategy: Dict[str, Any]
    reward: float
    feedback: str


class ActorAgent:
    """
    Actor-Critic Agent using LLM reflection for policy updates.

    The Actor maintains:
    - Current strategy (Action)
    - History of (strategy, reward, feedback) tuples
    - LLM client for reflection
    - Critic for value estimation

    Policy Update:
    - Uses LLM reflection to analyze feedback and improve strategy
    - Critic provides value baseline for advantage estimation
    - Advantage-weighted updates for more stable learning

    Features:
    - Self-critique: Two-stage reasoning with self-verification
    - Consistency check: Soft guidance (not hard override) from best strategy
    - Chain-of-Thought: Structured step-by-step reasoning
    - Critic: Value estimation for advantage computation
    """

    def __init__(
        self,
        llm_client: LLMClient,
        task_name: str,
        initial_strategy: Optional[Strategy] = None,
        enable_self_critique: bool = True,
        enable_consistency_check: bool = True,
        enable_cot_reasoning: bool = True,
        enable_critic: bool = True,
        enable_parameter_tracking: bool = True,
        enable_ucb_selection: bool = True,
    ):
        """
        Initialize the Actor-Critic Agent.

        Args:
            llm_client: LLM client for generating reflections
            task_name: Name of the downstream task
            initial_strategy: Optional initial strategy (uses default if None)
            enable_self_critique: Enable two-stage self-critique (default: True)
            enable_consistency_check: Enable soft consistency guidance (default: True)
            enable_cot_reasoning: Use Chain-of-Thought prompts (default: True)
            enable_critic: Enable Critic for value estimation (default: True)
            enable_parameter_tracking: Enable parameter effect tracking (default: True)
            enable_ucb_selection: Enable UCB for discrete params (default: True)
        """
        self.llm = llm_client
        self.task_name = task_name

        # Self-verification configuration
        self.enable_self_critique = enable_self_critique
        self.enable_consistency_check = enable_consistency_check
        self.enable_cot_reasoning = enable_cot_reasoning
        self.enable_critic = enable_critic
        self.enable_parameter_tracking = enable_parameter_tracking
        self.enable_ucb_selection = enable_ucb_selection

        # Initialize strategy
        if initial_strategy is not None:
            self.current_strategy = initial_strategy
        else:
            self.current_strategy = get_default_strategy(task_name)

        # Initialize Critic
        self.critic = SimpleCritic() if enable_critic else None

        # Initialize Parameter Effect Tracker
        self.param_tracker = ParameterEffectTracker() if enable_parameter_tracking else None

        # Initialize UCB Selector for discrete parameters
        self.ucb_bandit = MultiArmedBandit() if enable_ucb_selection else None

        # History for policy learning
        self.history: List[HistoryEntry] = []
        self.metric_history: List[float] = []  # For Critic

        # Track best strategy seen
        self.best_strategy: Optional[Strategy] = None
        self.best_reward: Optional[float] = None  # None allows tracking negative rewards

        # Last reflection and advantage for logging
        self._last_reflection: str = ""
        self._last_advantage: Optional[AdvantageEstimate] = None

        # Exploration mode state
        self._exploration_mode: bool = False
        self._exploration_perturbations: Optional[Dict[str, Any]] = None

        # Track consecutive declines for rollback mechanism
        self._consecutive_decline_count: int = 0
        self._last_raw_metric: Optional[float] = None

        logger.info(f"ActorAgent initialized for task: {task_name}")
        logger.info(f"  Self-critique: {enable_self_critique}")
        logger.info(f"  Consistency check (soft): {enable_consistency_check}")
        logger.info(f"  CoT reasoning: {enable_cot_reasoning}")
        logger.info(f"  Critic enabled: {enable_critic}")
        logger.info(f"  Parameter tracking: {enable_parameter_tracking}")
        logger.info(f"  UCB selection: {enable_ucb_selection}")

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
        # Check for baseline iteration - only record metric, skip policy update
        is_baseline = strategy_dict.get('is_baseline', False) if strategy_dict else False
        if is_baseline:
            if self.param_tracker and raw_metric is not None:
                self.param_tracker.record(strategy_dict, raw_metric)
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

        # Update Parameter Effect Tracker with raw_metric for consistent tracking
        # Use strategy_dict if provided (contains is_baseline), otherwise use current_strategy
        tracker_strategy = strategy_dict if strategy_dict else self.current_strategy.to_dict()
        if self.param_tracker and raw_metric is not None:
            # Use raw_metric (AUC) for effect tracking - more consistent than composite reward
            self.param_tracker.record(tracker_strategy, raw_metric)
            logger.debug("Parameter effects updated with raw_metric")

        # Update UCB Bandit with normalized reward
        if self.ucb_bandit and raw_metric is not None:
            # Normalize reward to [0, 1] for UCB
            normalized_reward = max(0.0, min(1.0, raw_metric))
            self.ucb_bandit.update_all(tracker_strategy, normalized_reward)
            logger.debug(f"UCB updated with normalized reward: {normalized_reward:.4f}")

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

        # Generate guidance from trackers for LLM
        tracker_guidance = ""
        if self.param_tracker:
            tracker_guidance += "\n\n" + self.param_tracker.get_guidance_for_llm()
        if self.ucb_bandit:
            tracker_guidance += "\n\n" + self.ucb_bandit.get_guidance()

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
        enhanced_feedback = feedback + advantage_context + tracker_guidance

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
        self.best_strategy = None
        self.best_reward = None  # None to allow tracking negative rewards
        self._last_reflection = ""
        self._last_advantage = None
        self._consecutive_decline_count = 0
        self._last_raw_metric = None
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
            ]
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
