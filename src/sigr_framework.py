"""
SIGR Framework - Self-Iterative Gene Representation

Main framework integrating Actor, Generator, Encoder, and Evaluator
in an MDP-based training loop.
"""

import logging
import random
from pathlib import Path
from typing import Dict, Any, Optional, List, Protocol
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

import numpy as np
import networkx as nx

from .actor import ActorAgent
from .generator import extract_subgraph, TextGenerator, MockTextGenerator
from .encoder import GeneEncoder
from .evaluator import TaskEvaluator
from .utils import load_kg, get_all_genes
from .utils.logger import SIGRLogger, setup_logging
from .mdp import MDPState, TrendAnalyzer, ExplorationScheduler


logger = logging.getLogger(__name__)


# Available downstream tasks
AVAILABLE_TASKS = [
    'ppi',                              # PPI Prediction
    'genetype',                         # Gene Type Classification
    'ggi',                              # Gene-Gene Interaction
    'cell',                             # Cell Type Classification
    'geneattribute_dosage_sensitivity', # Dosage Sensitivity
    'geneattribute_lys4_only',          # H3K4me1 Methylation
    'geneattribute_no_methylation',     # No Methylation
    'geneattribute_bivalent',           # Bivalent Chromatin
]


class LLMClient(Protocol):
    """Protocol for LLM client interface."""
    def generate(self, prompt: str) -> str:
        """Generate text from prompt."""
        ...

    def generate_strong(self, prompt: str) -> str:
        """Generate text using strong model (optional, for dual-model clients)."""
        ...


class SIGRFramework:
    """
    Self-Iterative Gene Representation Framework.

    Implements MDP-based optimization:
    - State: Gene info + KG neighborhood + history
    - Action: Strategy (edge_types, hops, sampling, prompt)
    - Reward: Downstream task performance
    - Policy: LLM reflection for strategy updates
    """

    def __init__(
        self,
        kg_path: str,
        llm_client: LLMClient,
        task_name: str,
        log_dir: str = "logs",
        results_dir: str = "results",
        use_cross_validation: bool = False,
        n_folds: int = 5,
        max_workers: int = 8,
        enable_adaptive_reward: bool = False,
        enable_self_critique: bool = True,
        enable_consistency_check: bool = True,
        enable_cot_reasoning: bool = True
    ):
        """
        Initialize the SIGR framework.

        Args:
            kg_path: Path to the knowledge graph pickle file
            llm_client: LLM client for text generation and reflection
            task_name: Name of the downstream task
            log_dir: Directory for training logs
            results_dir: Directory for results and strategies
            use_cross_validation: Whether to use cross-validation
            n_folds: Number of folds for cross-validation
            max_workers: Maximum number of concurrent LLM requests
            enable_adaptive_reward: Enable adaptive reward weight adjustment
            enable_self_critique: Enable Actor self-critique (default: True)
            enable_consistency_check: Enable strategy consistency check (default: True)
            enable_cot_reasoning: Enable Chain-of-Thought reasoning (default: True)
        """
        # Validate task name
        if task_name not in AVAILABLE_TASKS:
            raise ValueError(
                f"Unknown task: {task_name}. "
                f"Available tasks: {AVAILABLE_TASKS}"
            )

        self.task_name = task_name
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.max_workers = max_workers
        self.enable_adaptive_reward = enable_adaptive_reward

        # Load knowledge graph
        logger.info(f"Loading knowledge graph from {kg_path}")
        self.kg = load_kg(kg_path)
        logger.info(f"KG loaded: {self.kg.number_of_nodes()} nodes, {self.kg.number_of_edges()} edges")

        # Get all genes
        self.all_genes = list(get_all_genes(self.kg))
        logger.info(f"Found {len(self.all_genes)} genes")

        # Initialize components
        self.actor = ActorAgent(
            llm_client,
            task_name,
            enable_self_critique=enable_self_critique,
            enable_consistency_check=enable_consistency_check,
            enable_cot_reasoning=enable_cot_reasoning,
            enable_critic=True  # Enable Critic for Actor-Critic architecture
        )
        self.generator = TextGenerator(llm_client)
        self.encoder = GeneEncoder()
        self.evaluator = TaskEvaluator(
            task_name,
            self.kg,
            use_cross_validation=use_cross_validation,
            n_folds=n_folds,
            enable_adaptive_reward=enable_adaptive_reward
        )
        self.logger = SIGRLogger(task_name, log_dir)

        # Get task-specific genes (for evaluation)
        self.task_genes = self.evaluator.get_task_genes()
        if self.task_genes:
            # Filter to genes that exist in KG
            kg_genes_set = set(self.all_genes)
            self.task_genes = [g for g in self.task_genes if g in kg_genes_set]
            logger.info(f"Task-specific genes: {len(self.task_genes)}")
        else:
            # Fallback to all genes
            self.task_genes = self.all_genes
            logger.info(f"Using all genes (no task-specific genes)")

        logger.info(f"SIGR Framework initialized for task: {task_name}")
        logger.info(f"Max concurrent LLM workers: {max_workers}")
        logger.info(f"Adaptive reward: {'enabled' if enable_adaptive_reward else 'disabled'}")

        # Initialize MDP components
        self.mdp_state = MDPState(task_name=task_name)
        self.trend_analyzer = TrendAnalyzer()
        self.exploration_scheduler = ExplorationScheduler()

    def train(
        self,
        n_iterations: int = 10,
        genes_per_iter: int = 100,
        random_seed: int = 42,
        run_baseline: bool = True
    ) -> Dict[str, Any]:
        """
        Run the MDP-based training loop.

        Args:
            n_iterations: Number of optimization iterations
            genes_per_iter: Number of genes to evaluate per iteration (0 = all)
            random_seed: Random seed for reproducibility
            run_baseline: Whether to run baseline (no-KG) evaluation first (default: True)

        Returns:
            Best strategy found
        """
        random.seed(random_seed)
        np.random.seed(random_seed)

        logger.info(f"Starting training: {n_iterations} iterations, {genes_per_iter} genes/iter")

        # Fix evaluation genes for fair comparison across iterations
        if genes_per_iter <= 0 or genes_per_iter >= len(self.task_genes):
            self.eval_genes = self.task_genes
        else:
            self.eval_genes = random.sample(self.task_genes, genes_per_iter)
        logger.info(f"Fixed evaluation set: {len(self.eval_genes)} genes")

        # Optional: Run baseline (Phase 0, not counted in iterations)
        if run_baseline:
            baseline_metrics = self._run_baseline(self.eval_genes)
            self.logger.log_baseline(baseline_metrics)

        for iteration in range(1, n_iterations + 1):
            print(f"\n{'='*60}")
            print(f"Iteration {iteration}/{n_iterations}")
            print(f"{'='*60}")

            # Analyze trend before iteration
            current_trend_analysis = None  # Will be set if enough history
            if len(self.mdp_state.metric_history) >= 3:
                trend_analysis = self.trend_analyzer.analyze(
                    self.mdp_state.metric_history,
                    self.mdp_state.strategy_history,
                    self.mdp_state.reward_history
                )
                current_trend_analysis = trend_analysis.to_dict()  # Convert to dict for Actor
                print(f"Trend: {trend_analysis.trend_direction} "
                      f"(strength: {trend_analysis.trend_strength:.2f})")
                print(f"Convergence: {trend_analysis.convergence_score:.2f}")
                print(f"Suggested: {trend_analysis.suggested_action}")

                # Update MDP state with trend info
                self.mdp_state.trend = trend_analysis.trend_direction
                self.mdp_state.trend_strength = trend_analysis.trend_strength
                self.mdp_state.convergence_score = trend_analysis.convergence_score

                # Decide exploration vs exploitation
                exploration_decision = self.exploration_scheduler.decide(
                    trend=trend_analysis.trend_direction,
                    convergence_score=trend_analysis.convergence_score,
                    iteration=iteration
                )
                print(f"Exploration: {exploration_decision.exploration_type} "
                      f"(ε={exploration_decision.exploration_rate:.3f})")

                # Pass exploration info to actor
                self.actor.set_exploration_mode(
                    explore=exploration_decision.should_explore,
                    perturbations=self.exploration_scheduler.generate_exploration_perturbation(
                        self.actor.get_strategy()
                    ) if exploration_decision.should_explore else None
                )

                # Record exploration decision
                self.mdp_state.record_exploration(exploration_decision.should_explore)
                self.mdp_state.exploration_rate = exploration_decision.exploration_rate
            else:
                # Early iterations: explore more
                self.actor.set_exploration_mode(explore=True, perturbations=None)
                # Also record exploration for early iterations
                self.mdp_state.record_exploration(explored=True)

            # Run one iteration
            reward, metrics = self._train_one_iteration(
                iteration=iteration,
                trend_analysis=current_trend_analysis
            )

            # Decay exploration rate after each iteration (polynomial decay)
            self.exploration_scheduler.decay_epsilon(iteration=iteration)

            print(f"Reward: {reward:.4f}")
            print(f"Metrics: {metrics}")
            print(f"\n--- MDP State Summary ---")
            print(self.mdp_state.get_summary())

        # Save final results
        best_strategy = self.actor.get_best_strategy()
        best_reward = self.actor.best_reward

        if best_strategy:
            self.logger.save_best_strategy(best_strategy, best_reward)
            self.logger.save_final_report(best_strategy, best_reward, n_iterations)

            # Save to results directory
            self._save_best_strategy(best_strategy, best_reward)

        return best_strategy or self.actor.get_strategy()

    def _train_one_iteration(
        self,
        iteration: int,
        trend_analysis: dict = None
    ) -> tuple:
        """
        Run one training iteration.

        Args:
            iteration: Current iteration number
            trend_analysis: Optional trend analysis dict for reflection

        Returns:
            Tuple of (reward, metrics)
        """
        # Start logging for this iteration
        self.logger.start_iteration(iteration)

        # Get current strategy from Actor (always use Actor strategy)
        strategy = self.actor.get_strategy()

        self.logger.log_strategy(strategy)

        edge_types_str = strategy.get('edge_types', [])
        if not edge_types_str:
            edge_types_str = "NONE (baseline)"
        logger.info(f"Strategy: edge_types={edge_types_str}, "
                   f"max_neighbors={strategy.get('max_neighbors', 0)}")

        # Use fixed evaluation genes (set in train() for fair comparison)
        eval_genes = self.eval_genes
        logger.info(f"Using fixed evaluation set: {len(eval_genes)} genes")

        # Generate embeddings for sampled genes
        embeddings, descriptions = self._generate_embeddings(
            eval_genes, strategy
        )

        # Handle empty embeddings (all generation failed)
        if not embeddings:
            logger.error("No embeddings generated - all gene descriptions failed")
            # Return a penalty reward and empty metrics
            penalty_reward = -0.5
            penalty_metrics = {self.evaluator.primary_metric: 0.0}

            # Update MDP state with failure
            self.mdp_state.update_after_iteration(
                metric=0.0,
                strategy=strategy,
                reward=penalty_reward,
            )

            # Generate failure feedback for Actor
            failure_feedback = (
                f"Generation failure: No embeddings were generated for {len(eval_genes)} genes. "
                f"This indicates severe issues with the current strategy. "
                f"Consider: 1) Reducing complexity of edge_types, 2) Checking if genes exist in KG, "
                f"3) Reviewing max_neighbors settings."
            )

            # Actor should still reflect on this failure
            self.actor.update_policy(
                penalty_reward,
                failure_feedback,
                raw_metric=0.0,
                trend_analysis=trend_analysis
            )

            self.logger.log_evaluation(penalty_metrics, penalty_reward)
            self.logger.log_feedback(failure_feedback)

            return penalty_reward, penalty_metrics

        # Log ALL descriptions (not just samples)
        for gene_id, desc in descriptions.items():
            self.logger.log_gene_description(gene_id, desc)

        # Warn if partial generation failure
        success_rate = len(embeddings) / len(eval_genes)
        if success_rate < 0.9:
            logger.warning(
                f"Partial generation failure: {len(embeddings)}/{len(eval_genes)} genes "
                f"({success_rate*100:.1f}%) - some genes may be missing from KG"
            )

        # Evaluate embeddings
        metrics = self.evaluator.evaluate(embeddings)

        # Compute reward using relative reward (improvement over history)
        history = self.mdp_state.metric_history if self.mdp_state.metric_history else None
        reward = self.evaluator.compute_reward(metrics, history=history)

        # Get detailed reward breakdown for logging
        reward_result = self.evaluator.compute_reward_detailed(metrics, history=history)
        logger.info(
            f"Reward breakdown: total={reward_result.total_reward:.4f}, "
            f"relative={reward_result.relative_reward:.4f}, "
            f"absolute={reward_result.absolute_reward:.4f}, "
            f"plateau_penalty={reward_result.plateau_penalty:.4f} (duration={reward_result.plateau_duration}), "
            f"raw={reward_result.raw_metric:.4f}"
        )

        # Update adaptive reward weights if enabled
        if self.enable_adaptive_reward and trend_analysis:
            trend_direction = trend_analysis.get('trend_direction', 'unknown')
            self.evaluator.update_reward_weights_for_trend(
                reward_result.raw_metric,
                trend_direction
            )

        # Update MDP state
        self.mdp_state.update_after_iteration(
            metric=reward_result.raw_metric,
            strategy=strategy,
            reward=reward,
        )

        # Record strategy visit for UCB
        self.exploration_scheduler.record_strategy_visit(strategy)

        # Update exploration scheduler with reward (required for UCB1)
        # Normalize reward to [0, 1] for UCB
        normalized_reward = min(1.0, max(0.0, reward_result.raw_metric))
        self.exploration_scheduler.update_reward(strategy, normalized_reward)

        # Log evaluation results
        self.logger.log_evaluation(metrics, reward)

        # Generate feedback for Actor
        feedback = self.evaluator.generate_feedback(
            metrics, embeddings, descriptions,
            best_metric=self.actor.best_reward
        )
        self.logger.log_feedback(feedback)

        # Actor reflects and updates policy (pass raw_metric for accurate best strategy tracking)
        self.actor.update_policy(
            reward,
            feedback,
            raw_metric=reward_result.raw_metric,
            trend_analysis=trend_analysis
        )
        self.logger.log_reflection(self.actor.get_last_reflection())

        # Save iteration summary
        self.logger.save_iteration_summary({
            'reward': reward,
            'metrics': metrics,
            'num_genes': len(eval_genes),
            'num_embeddings': len(embeddings)
        })

        return reward, metrics

    def _get_baseline_prompt_template(self) -> str:
        """
        Get prompt template for baseline iteration (no KG information).

        This baseline uses only gene name without any graph context,
        allowing us to measure the value added by KG information.

        Returns:
            Baseline prompt template string
        """
        return """Generate a CONCISE biological description (100-150 words) for gene {gene_id} ({gene_name}).

Based only on the gene name and your general biological knowledge, describe what is typically known about this gene.

CONSTRAINTS:
- Target length: 100-150 words (STRICT)
- Do NOT make predictions or judgments
- Do NOT output classification labels
- Factual description only
- This is a baseline description without knowledge graph context
"""

    def _run_baseline(self, eval_genes: List[str]) -> Dict[str, float]:
        """
        Run baseline evaluation without KG (optional Phase 0).

        This baseline uses only LLM prior knowledge without any KG context.
        Results are NOT added to MDP history - they serve only as a reference
        to measure the value added by KG information.

        Args:
            eval_genes: List of gene IDs to evaluate

        Returns:
            Baseline metrics dict
        """
        print(f"\n{'='*60}")
        print("BASELINE PHASE (no KG)")
        print("Measuring LLM prior knowledge only")
        print(f"{'='*60}")

        strategy = {
            'edge_types': [],  # Empty = no graph information
            'max_neighbors': 0,
            'max_hops': 0,
            'sampling': 'top_k',
            'prompt_template': self._get_baseline_prompt_template(),
            'is_baseline': True
        }

        logger.info("Running baseline evaluation without KG context...")

        embeddings, descriptions = self._generate_embeddings(eval_genes, strategy)

        if not embeddings:
            logger.error("Baseline: No embeddings generated")
            return {self.evaluator.primary_metric: 0.0}

        metrics = self.evaluator.evaluate(embeddings)

        logger.info("=" * 60)
        logger.info(f"BASELINE RESULTS: {metrics}")
        logger.info("This baseline measures performance without KG context")
        logger.info("=" * 60)

        print(f"Baseline {self.evaluator.primary_metric}: {metrics.get(self.evaluator.primary_metric, 0):.4f}")
        print(f"Full metrics: {metrics}")

        return metrics

    def _validate_and_build_strategy(self, strategy: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate strategy parameters and build extraction strategy.

        Args:
            strategy: Raw strategy from Actor or baseline

        Returns:
            Validated extraction strategy dict
        """
        # Extract parameters with defaults
        edge_types = strategy.get('edge_types', ['PPI', 'GO', 'HPO'])
        max_neighbors = strategy.get('max_neighbors', 50)
        sampling = strategy.get('sampling', 'top_k')
        max_hops = strategy.get('max_hops', 2)

        # Validate edge_types
        if not isinstance(edge_types, list):
            logger.warning(f"Invalid edge_types type: {type(edge_types)}, using default")
            edge_types = ['PPI', 'GO', 'HPO']

        valid_edge_types = {'PPI', 'GO', 'HPO', 'TRRUST', 'CellMarker', 'Reactome'}
        invalid_types = [et for et in edge_types if et not in valid_edge_types]
        if invalid_types:
            logger.warning(f"Invalid edge types removed: {invalid_types}")
            edge_types = [et for et in edge_types if et in valid_edge_types]

        # Validate max_neighbors
        if not isinstance(max_neighbors, (int, float)) or max_neighbors < 0:
            logger.warning(f"Invalid max_neighbors: {max_neighbors}, using default 50")
            max_neighbors = 50
        max_neighbors = int(max_neighbors)

        # Validate sampling method
        valid_sampling = {'top_k', 'random', 'weighted'}
        if sampling not in valid_sampling:
            logger.warning(f"Invalid sampling method: {sampling}, using default 'top_k'")
            sampling = 'top_k'

        # Validate max_hops
        if not isinstance(max_hops, int) or max_hops < 0 or max_hops > 3:
            logger.warning(f"Invalid max_hops: {max_hops}, using default 2")
            max_hops = 2

        return {
            'edge_types': edge_types,
            'max_neighbors': max_neighbors,
            'sampling': sampling,
            'max_hops': max_hops
        }

    def _generate_embeddings(
        self,
        gene_ids: List[str],
        strategy: Dict[str, Any]
    ) -> tuple:
        """
        Generate embeddings for a list of genes.

        Two-phase process:
        1. Generate all descriptions using LLM (concurrent, multi-threaded)
        2. Encode all descriptions using Sentence Transformer (batch, fast)

        Args:
            gene_ids: List of gene identifiers
            strategy: Current strategy configuration

        Returns:
            Tuple of (embeddings dict, descriptions dict)
        """
        descriptions = {}

        # Validate and build extraction strategy
        extraction_strategy = self._validate_and_build_strategy(strategy)

        # Log strategy mode
        if not extraction_strategy['edge_types']:
            logger.info("Extraction mode: BASELINE (no graph information)")
        else:
            logger.info(f"Extraction mode: NORMAL with edge_types={extraction_strategy['edge_types']}")

        # Phase 1: Generate all descriptions using LLM (multi-threaded)
        logger.info(f"Phase 1: Generating descriptions for {len(gene_ids)} genes (workers={self.max_workers})...")

        def generate_single_description(gene_id: str) -> tuple:
            """Generate description for a single gene."""
            try:
                # Extract subgraph using strategy
                subgraph = extract_subgraph(
                    gene_id=gene_id,
                    strategy=extraction_strategy,
                    kg=self.kg
                )

                # Generate description with strategy parameters
                description = self.generator.generate(
                    gene_id=gene_id,
                    subgraph=subgraph,
                    prompt_template=strategy.get('prompt_template', ''),
                    kg=self.kg,
                    strategy=strategy  # Pass full strategy for description_length, focus_keywords, etc.
                )
                return (gene_id, description, None)
            except Exception as e:
                return (gene_id, None, str(e))

        # Use ThreadPoolExecutor for concurrent LLM requests
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(generate_single_description, gene_id): gene_id
                for gene_id in gene_ids
            }

            # Collect results with progress bar
            for future in tqdm(as_completed(futures), total=len(gene_ids),
                              desc="Generating descriptions (LLM)"):
                try:
                    gene_id, description, error = future.result()
                    if description is not None:
                        descriptions[gene_id] = description
                    else:
                        logger.warning(f"Error generating description for {gene_id}: {error}")
                except Exception as e:
                    # Catch any unexpected exceptions from future.result()
                    gene_id = futures.get(future, "unknown")
                    logger.error(f"Unexpected error for gene {gene_id}: {e}")

        if not descriptions:
            logger.warning("No descriptions generated")
            return {}, {}

        # Phase 2: Batch encode all descriptions
        logger.info(f"Phase 2: Encoding {len(descriptions)} descriptions...")
        embeddings = self.encoder.encode_genes(
            descriptions,
            batch_size=64,
            show_progress=True
        )

        logger.info(f"Generated {len(embeddings)} embeddings")
        return embeddings, descriptions

    def _save_best_strategy(self, strategy: Dict[str, Any], reward: float):
        """Save best strategy to results directory."""
        import json

        strategies_dir = self.results_dir / "strategies"
        strategies_dir.mkdir(exist_ok=True)

        strategy_file = strategies_dir / f"{self.task_name}_best.json"
        with open(strategy_file, 'w') as f:
            json.dump({
                'task': self.task_name,
                'strategy': strategy,
                'reward': reward
            }, f, indent=2)

        logger.info(f"Best strategy saved to {strategy_file}")

    def evaluate_strategy(
        self,
        strategy: Dict[str, Any],
        gene_ids: Optional[List[str]] = None,
        sample_size: int = 100
    ) -> Dict[str, float]:
        """
        Evaluate a specific strategy.

        Args:
            strategy: Strategy configuration
            gene_ids: Optional specific genes to evaluate
            sample_size: Number of genes if gene_ids not provided

        Returns:
            Evaluation metrics
        """
        if gene_ids is None:
            gene_ids = random.sample(self.task_genes, min(sample_size, len(self.task_genes)))

        embeddings, _ = self._generate_embeddings(gene_ids, strategy)
        return self.evaluator.evaluate(embeddings)

    def save_checkpoint(self, iteration: int, checkpoint_dir: str = None) -> str:
        """
        Save training checkpoint for resume capability.

        Saves all state needed to resume training:
        - MDP state (history, metrics, best strategy)
        - Actor state (strategy, history)
        - Exploration scheduler state
        - Evaluation gene set

        Args:
            iteration: Current iteration number
            checkpoint_dir: Directory for checkpoints (default: results_dir/checkpoints)

        Returns:
            Path to saved checkpoint file
        """
        import json

        if checkpoint_dir is None:
            checkpoint_dir = self.results_dir / "checkpoints"
        else:
            checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'iteration': iteration,
            'task_name': self.task_name,
            'mdp_state': self.mdp_state.to_dict(),
            'actor_state': self.actor.save_state(),
            'exploration_scheduler': self.exploration_scheduler.get_state(),
            'eval_genes': self.eval_genes if hasattr(self, 'eval_genes') else None,
        }

        checkpoint_file = checkpoint_dir / f"checkpoint_iter{iteration}.json"
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint, f, indent=2)

        logger.info(f"Checkpoint saved to {checkpoint_file}")
        return str(checkpoint_file)

    def load_checkpoint(self, checkpoint_path: str) -> int:
        """
        Load training checkpoint and resume state.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            Iteration number to resume from (next iteration)
        """
        import json
        from .mdp import MDPState

        with open(checkpoint_path, 'r') as f:
            checkpoint = json.load(f)

        # Validate task name matches
        if checkpoint['task_name'] != self.task_name:
            raise ValueError(
                f"Checkpoint task '{checkpoint['task_name']}' does not match "
                f"current task '{self.task_name}'"
            )

        # Restore MDP state
        self.mdp_state = MDPState.from_dict(checkpoint['mdp_state'])

        # Restore Actor state
        self.actor.load_state(checkpoint['actor_state'])

        # Restore exploration scheduler
        scheduler_state = checkpoint['exploration_scheduler']
        self.exploration_scheduler.load_state(scheduler_state)

        # Restore evaluation genes
        if checkpoint.get('eval_genes'):
            self.eval_genes = checkpoint['eval_genes']

        resumed_iteration = checkpoint['iteration']
        logger.info(
            f"Checkpoint loaded from {checkpoint_path}. "
            f"Resuming from iteration {resumed_iteration + 1}"
        )
        return resumed_iteration + 1

    def train_with_checkpoint(
        self,
        n_iterations: int = 10,
        genes_per_iter: int = 100,
        random_seed: int = 42,
        checkpoint_interval: int = 5,
        resume_from: str = None,
        run_baseline: bool = True
    ) -> Dict[str, Any]:
        """
        Run training with checkpoint support.

        Args:
            n_iterations: Total number of optimization iterations
            genes_per_iter: Number of genes to evaluate per iteration (0 = all)
            random_seed: Random seed for reproducibility
            checkpoint_interval: Save checkpoint every N iterations
            resume_from: Path to checkpoint file to resume from
            run_baseline: Whether to run baseline (no-KG) evaluation first (default: True)

        Returns:
            Best strategy found
        """
        start_iteration = 1

        if resume_from:
            start_iteration = self.load_checkpoint(resume_from)
            start_iteration = max(1, start_iteration)  # Ensure at least 1
        else:
            random.seed(random_seed)
            np.random.seed(random_seed)

            # Fix evaluation genes for fair comparison across iterations
            if genes_per_iter <= 0 or genes_per_iter >= len(self.task_genes):
                self.eval_genes = self.task_genes
            else:
                self.eval_genes = random.sample(self.task_genes, genes_per_iter)

        logger.info(
            f"Starting training: iterations {start_iteration}-{n_iterations}, "
            f"{len(self.eval_genes)} genes/iter"
        )

        # Optional: Run baseline only on fresh start (not resume)
        if run_baseline and not resume_from:
            baseline_metrics = self._run_baseline(self.eval_genes)
            self.logger.log_baseline(baseline_metrics)

        for iteration in range(start_iteration, n_iterations + 1):
            print(f"\n{'='*60}")
            print(f"Iteration {iteration}/{n_iterations}")
            print(f"{'='*60}")

            # Analyze trend before iteration
            current_trend_analysis = None
            if len(self.mdp_state.metric_history) >= 3:
                trend_analysis = self.trend_analyzer.analyze(
                    self.mdp_state.metric_history,
                    self.mdp_state.strategy_history,
                    self.mdp_state.reward_history
                )
                current_trend_analysis = trend_analysis.to_dict()
                print(f"Trend: {trend_analysis.trend_direction} "
                      f"(strength: {trend_analysis.trend_strength:.2f})")
                print(f"Convergence: {trend_analysis.convergence_score:.2f}")
                print(f"Suggested: {trend_analysis.suggested_action}")

                # Update MDP state with trend info
                self.mdp_state.trend = trend_analysis.trend_direction
                self.mdp_state.trend_strength = trend_analysis.trend_strength
                self.mdp_state.convergence_score = trend_analysis.convergence_score

                # Decide exploration vs exploitation
                exploration_decision = self.exploration_scheduler.decide(
                    trend=trend_analysis.trend_direction,
                    convergence_score=trend_analysis.convergence_score,
                    iteration=iteration
                )
                print(f"Exploration: {exploration_decision.exploration_type} "
                      f"(ε={exploration_decision.exploration_rate:.3f})")

                self.actor.set_exploration_mode(
                    explore=exploration_decision.should_explore,
                    perturbations=self.exploration_scheduler.generate_exploration_perturbation(
                        self.actor.get_strategy()
                    ) if exploration_decision.should_explore else None
                )

                self.mdp_state.record_exploration(exploration_decision.should_explore)
                self.mdp_state.exploration_rate = exploration_decision.exploration_rate
            else:
                self.actor.set_exploration_mode(explore=True, perturbations=None)
                # Also record exploration for early iterations
                self.mdp_state.record_exploration(explored=True)

            # Run one iteration
            reward, metrics = self._train_one_iteration(
                iteration=iteration,
                trend_analysis=current_trend_analysis
            )

            # Decay exploration rate
            self.exploration_scheduler.decay_epsilon()

            print(f"Reward: {reward:.4f}")
            print(f"Metrics: {metrics}")
            print(f"\n--- MDP State Summary ---")
            print(self.mdp_state.get_summary())

            # Save checkpoint at intervals
            if checkpoint_interval > 0 and iteration % checkpoint_interval == 0:
                self.save_checkpoint(iteration)

        # Save final results
        best_strategy = self.actor.get_best_strategy()
        best_reward = self.actor.best_reward

        if best_strategy:
            self.logger.save_best_strategy(best_strategy, best_reward)
            self.logger.save_final_report(best_strategy, best_reward, n_iterations)
            self._save_best_strategy(best_strategy, best_reward)

        return best_strategy or self.actor.get_strategy()


class MockLLMClient:
    """Mock LLM client for testing without actual API calls."""

    def __init__(self):
        self.call_count = 0

    def generate(self, prompt: str) -> str:
        """Generate mock response."""
        self.call_count += 1

        if 'updated strategy' in prompt.lower() or 'json' in prompt.lower():
            import json
            return json.dumps({
                "edge_types": ["PPI", "GO", "HPO"],
                "max_hops": 2,
                "sampling": "top_k",
                "max_neighbors": 50,
                "reasoning": "Mock strategy update"
            })
        else:
            return f"Mock description for gene. Call #{self.call_count}"


def list_available_tasks() -> List[str]:
    """Return list of available downstream tasks."""
    return AVAILABLE_TASKS.copy()


def get_task_info(task_name: str) -> Dict[str, str]:
    """
    Get information about a task.

    Args:
        task_name: Name of the task

    Returns:
        Dictionary with task information
    """
    task_info = {
        'ppi': {
            'name': 'PPI Prediction',
            'description': 'Protein-Protein Interaction prediction (binary classification)',
            'metric': 'auc',
            'data': 'PPI graph edges'
        },
        'genetype': {
            'name': 'Gene Type Classification',
            'description': 'Multi-class classification of gene biotypes',
            'metric': 'f1',
            'data': 'data/downstreams/genetype/'
        },
        'ggi': {
            'name': 'Gene-Gene Interaction',
            'description': 'Gene-Gene Interaction prediction (binary classification)',
            'metric': 'auc',
            'data': 'data/downstreams/ggi/'
        },
        'cell': {
            'name': 'Cell Type Classification',
            'description': 'Cell type classification and clustering',
            'metric': 'f1',
            'data': 'data/downstreams/cell/'
        },
        'geneattribute_dosage_sensitivity': {
            'name': 'Dosage Sensitivity',
            'description': 'Dosage sensitivity prediction',
            'metric': 'auc',
            'data': 'data/downstreams/geneattribute/'
        },
        'geneattribute_lys4_only': {
            'name': 'H3K4me1 Methylation',
            'description': 'H3K4me1 only methylation prediction',
            'metric': 'auc',
            'data': 'data/downstreams/geneattribute/'
        },
        'geneattribute_no_methylation': {
            'name': 'No Methylation',
            'description': 'No methylation prediction',
            'metric': 'auc',
            'data': 'data/downstreams/geneattribute/'
        },
        'geneattribute_bivalent': {
            'name': 'Bivalent Chromatin',
            'description': 'Bivalent chromatin state prediction',
            'metric': 'auc',
            'data': 'data/downstreams/geneattribute/'
        },
    }
    return task_info.get(task_name, {'name': task_name, 'description': 'Unknown task'})


def run_training(
    task_name: str,
    kg_path: str = "data/kg/sigr_kg_v2.pkl",
    n_iterations: int = 10,
    genes_per_iter: int = 100,
    log_dir: str = "logs",
    results_dir: str = "results",
    llm_client: Optional[LLMClient] = None,
    use_mock: bool = False,
    # API configuration
    api_base: str = "https://yunwu.ai/v1",
    api_key: Optional[str] = None,
    model: str = "gpt-4o-mini",
    # Dual model configuration
    use_dual_model: bool = False,
    fast_model: str = "gpt-4o-mini",
    strong_model: str = "gemini-3-pro-preview",
    # Evaluation options
    use_cross_validation: bool = False,
    n_folds: int = 5,
    # Concurrency
    max_workers: int = 8,
    # Checkpoint options
    checkpoint_interval: int = 5,
    resume_from: str = None,
    # Adaptive reward
    enable_adaptive_reward: bool = False,
    # Self-verification options (LLM self-critique, Constitutional AI patterns)
    enable_self_critique: bool = True,
    enable_consistency_check: bool = True,
    enable_cot_reasoning: bool = True,
    # Baseline option
    run_baseline: bool = True
):
    """
    Convenience function to run SIGR training.

    Args:
        task_name: Name of the downstream task
        kg_path: Path to knowledge graph
        n_iterations: Number of iterations
        genes_per_iter: Genes per iteration
        log_dir: Log directory
        results_dir: Results directory
        llm_client: Optional LLM client (creates real client if not provided)
        use_mock: Force use of mock LLM client
        api_base: API base URL for LLM
        api_key: API key for LLM
        model: Model name for LLM (single model mode)
        use_dual_model: Use dual model architecture (recommended)
        fast_model: Model for generation tasks (dual model mode)
        strong_model: Model for reflection/reasoning (dual model mode)
        use_cross_validation: Whether to use cross-validation
        n_folds: Number of folds for cross-validation
        max_workers: Maximum concurrent LLM requests
        checkpoint_interval: Save checkpoint every N iterations (0 to disable)
        resume_from: Path to checkpoint file to resume from
        run_baseline: Whether to run baseline (no-KG) evaluation first (default: True)

    Returns:
        Best strategy found
    """
    setup_logging()

    # Create LLM client
    if use_mock:
        logger.warning("Using MockLLMClient - results are for testing only")
        llm_client = MockLLMClient()
    elif llm_client is None:
        try:
            if use_dual_model:
                # Use dual model client (recommended)
                from configs.client import get_dual_model_client
                llm_client = get_dual_model_client(
                    base_url=api_base,
                    api_key=api_key,
                    fast_model=fast_model,
                    strong_model=strong_model,
                    max_connections=max_workers * 2
                )
                logger.info(
                    f"Using DualModelClient: fast={fast_model}, strong={strong_model}"
                )
            else:
                # Single model client
                from configs.client import get_llm_client
                llm_client = get_llm_client(
                    base_url=api_base,
                    api_key=api_key,
                    model=model
                )
                logger.info(f"Using single LLM client: {model} @ {api_base}")
        except Exception as e:
            logger.error(f"Failed to create LLM client: {e}")
            logger.warning("Falling back to MockLLMClient")
            llm_client = MockLLMClient()

    framework = SIGRFramework(
        kg_path=kg_path,
        llm_client=llm_client,
        task_name=task_name,
        log_dir=log_dir,
        results_dir=results_dir,
        use_cross_validation=use_cross_validation,
        n_folds=n_folds,
        max_workers=max_workers,
        enable_adaptive_reward=enable_adaptive_reward,
        enable_self_critique=enable_self_critique,
        enable_consistency_check=enable_consistency_check,
        enable_cot_reasoning=enable_cot_reasoning
    )

    # Use checkpoint-aware training if checkpoint/resume specified
    if checkpoint_interval > 0 or resume_from:
        return framework.train_with_checkpoint(
            n_iterations=n_iterations,
            genes_per_iter=genes_per_iter,
            checkpoint_interval=checkpoint_interval,
            resume_from=resume_from,
            run_baseline=run_baseline
        )
    else:
        return framework.train(
            n_iterations=n_iterations,
            genes_per_iter=genes_per_iter,
            run_baseline=run_baseline
        )


def run_all_tasks(
    kg_path: str = "data/kg/sigr_kg.pkl",
    n_iterations: int = 10,
    genes_per_iter: int = 100,
    llm_client: Optional[LLMClient] = None,
    tasks: Optional[List[str]] = None,
    api_base: str = "https://yunwu.ai/v1",
    api_key: Optional[str] = None,
    model: str = "gpt-4o-mini",
    use_mock: bool = False
) -> Dict[str, Dict[str, Any]]:
    """
    Run training for multiple downstream tasks.

    Args:
        kg_path: Path to knowledge graph
        n_iterations: Number of iterations per task
        genes_per_iter: Genes per iteration
        llm_client: LLM client to use
        tasks: List of tasks to run (default: all available)
        api_base: API base URL
        api_key: API key
        model: Model name
        use_mock: Use mock LLM client

    Returns:
        Dictionary mapping task names to best strategies
    """
    if tasks is None:
        tasks = AVAILABLE_TASKS

    results = {}

    for task_name in tasks:
        print(f"\n{'='*80}")
        print(f"Training for task: {task_name}")
        print(f"{'='*80}\n")

        try:
            best_strategy = run_training(
                task_name=task_name,
                kg_path=kg_path,
                n_iterations=n_iterations,
                genes_per_iter=genes_per_iter,
                llm_client=llm_client,
                api_base=api_base,
                api_key=api_key,
                model=model,
                use_mock=use_mock
            )
            results[task_name] = best_strategy
        except Exception as e:
            logger.error(f"Error training {task_name}: {e}")
            results[task_name] = None

    return results
