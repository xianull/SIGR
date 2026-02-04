"""
Task Evaluator for SIGR Framework

Evaluates gene embeddings on downstream tasks.
Supports multiple classifiers and cross-validation.
"""

import logging
from typing import Dict, Any, Optional, Callable, List

import numpy as np
import networkx as nx
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score

from .tasks.base_task import BaseTask
from .tasks.ppi_task import PPITask
from .tasks.genetype_task import GeneTypeTask
from .tasks.geneattribute_task import GeneAttributeTask
from .tasks.ggi_task import GGITask
from .tasks.cell_task import CellTask
from .classifiers import get_classifier, ClassifierFactory
from .metrics import compute_clustering_metrics
from ..mdp.reward import RewardComputer, RewardResult

# Import configurable reward weights
try:
    from configs.reward_config import get_reward_weights, AdaptiveRewardConfig
    HAS_REWARD_CONFIG = True
except ImportError:
    HAS_REWARD_CONFIG = False


logger = logging.getLogger(__name__)


class TaskEvaluator:
    """
    Evaluator for downstream tasks.

    Supports multiple tasks, classifiers, and evaluation methods.
    """

    # Mapping of task names to task classes or factory functions
    TASK_CLASSES: Dict[str, Any] = {
        'ppi': PPITask,
        'genetype': GeneTypeTask,
        'ggi': GGITask,
        'cell': CellTask,
        # GeneAttribute subtasks
        'geneattribute_dosage_sensitivity': lambda kg: GeneAttributeTask(kg, 'dosage_sensitivity'),
        'geneattribute_lys4_only': lambda kg: GeneAttributeTask(kg, 'lys4_only'),
        'geneattribute_no_methylation': lambda kg: GeneAttributeTask(kg, 'no_methylation'),
        'geneattribute_bivalent': lambda kg: GeneAttributeTask(kg, 'bivalent'),
    }

    # Primary metric for each task
    TASK_METRICS = {
        'ppi': 'auc',
        'genetype': 'f1',
        'ggi': 'auc',
        'cell': 'f1',
        'geneattribute_dosage_sensitivity': 'auc',
        'geneattribute_lys4_only': 'auc',
        'geneattribute_no_methylation': 'auc',
        'geneattribute_bivalent': 'auc',
    }

    def __init__(
        self,
        task_name: str,
        kg: nx.DiGraph,
        classifier_type: str = 'logistic',
        use_cross_validation: bool = False,
        n_folds: int = 5,
        enable_adaptive_reward: bool = False
    ):
        """
        Initialize the evaluator.

        Args:
            task_name: Name of the task to evaluate
            kg: Knowledge graph
            classifier_type: Type of classifier ('logistic' or 'random_forest')
            use_cross_validation: Whether to use cross-validation
            n_folds: Number of folds for cross-validation
            enable_adaptive_reward: Enable adaptive reward weight adjustment
        """
        self.task_name = task_name
        self.kg = kg
        self.classifier_type = classifier_type
        self.use_cross_validation = use_cross_validation
        self.n_folds = n_folds

        # Load the task
        if task_name not in self.TASK_CLASSES:
            raise ValueError(
                f"Unknown task: {task_name}. "
                f"Available: {list(self.TASK_CLASSES.keys())}"
            )

        task_factory = self.TASK_CLASSES[task_name]
        if callable(task_factory) and not isinstance(task_factory, type):
            # It's a lambda/function
            self.task = task_factory(kg)
        else:
            # It's a class
            self.task = task_factory(kg)

        self.primary_metric = self.TASK_METRICS.get(task_name, 'f1')

        # Initialize reward computer with task-specific weights
        if HAS_REWARD_CONFIG:
            task_weights = get_reward_weights(task_name)
            self.reward_computer = RewardComputer(
                weights=task_weights,
                task_name=task_name,
                enable_adaptive=enable_adaptive_reward
            )
            logger.info(f"RewardComputer initialized with task-specific weights for {task_name}")
        else:
            self.reward_computer = RewardComputer()
            logger.info("RewardComputer initialized with default weights")

    def evaluate(
        self,
        embeddings: Dict[str, np.ndarray],
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Dict[str, float]:
        """
        Evaluate embeddings on the task.

        Args:
            embeddings: Dictionary mapping gene_id to embedding
            test_size: Fraction of data to use for testing
            random_state: Random seed for reproducibility

        Returns:
            Dictionary of metrics
        """
        # Prepare data
        X, y = self.task.prepare_data(embeddings)

        if len(X) == 0:
            logger.warning(f"No data available for {self.task_name} task")
            return self._empty_metrics()

        # Check for invalid embeddings (NaN or Inf)
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            logger.error("Invalid embeddings detected (NaN or Inf values)")
            return self._empty_metrics()

        # Choose evaluation method
        if self.use_cross_validation:
            metrics = self._evaluate_with_cv(X, y, random_state)
        else:
            metrics = self._evaluate_with_split(X, y, test_size, random_state)

        # Add clustering metrics if required
        if self.task.requires_clustering_metrics():
            try:
                clustering_metrics = compute_clustering_metrics(X, y)
                metrics.update(clustering_metrics)
            except Exception as e:
                logger.warning(f"Error computing clustering metrics: {e}")

        logger.info(f"Evaluation results for {self.task_name}: "
                   f"primary={metrics.get(self.primary_metric, 0):.4f}")
        return metrics

    def _evaluate_with_cv(
        self,
        X: np.ndarray,
        y: np.ndarray,
        random_state: int
    ) -> Dict[str, float]:
        """Evaluate using stratified k-fold cross-validation."""
        skf = StratifiedKFold(
            n_splits=self.n_folds,
            shuffle=True,
            random_state=random_state
        )

        all_metrics = {
            'accuracy': [], 'f1': [], 'auc': [], 'ap': []
        }

        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            fold_metrics = self._train_and_evaluate(
                X_train, X_test, y_train, y_test, random_state
            )

            for key in all_metrics:
                if key in fold_metrics:
                    all_metrics[key].append(fold_metrics[key])

        # Compute mean and std
        metrics = {}
        for key, values in all_metrics.items():
            if values:
                metrics[key] = float(np.mean(values))
                metrics[f'{key}_std'] = float(np.std(values))

        return metrics

    def _evaluate_with_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float,
        random_state: int
    ) -> Dict[str, float]:
        """Evaluate using train/test split."""
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=test_size,
                random_state=random_state,
                stratify=y
            )
        except ValueError:
            # Fall back to non-stratified if stratification fails
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=test_size,
                random_state=random_state
            )

        return self._train_and_evaluate(X_train, X_test, y_train, y_test, random_state)

    def _train_and_evaluate(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        random_state: int
    ) -> Dict[str, float]:
        """Train classifier and compute metrics."""
        # Get classifier
        clf = get_classifier(self.classifier_type, random_state=random_state)

        try:
            clf.fit(X_train, y_train)
        except Exception as e:
            logger.warning(f"Error training classifier: {e}")
            return self._empty_metrics()

        # Predictions
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)

        # Compute metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred, average='macro', zero_division=0),
        }

        # AUC (handle binary vs multi-class)
        n_classes = len(np.unique(np.concatenate([y_train, y_test])))
        if n_classes == 2:
            try:
                metrics['auc'] = roc_auc_score(y_test, y_prob[:, 1])
                metrics['ap'] = average_precision_score(y_test, y_prob[:, 1])
            except Exception:
                metrics['auc'] = 0.5
                metrics['ap'] = 0.5
        else:
            try:
                metrics['auc'] = roc_auc_score(
                    y_test, y_prob, multi_class='ovr', average='macro'
                )
            except Exception:
                metrics['auc'] = 0.5

        return metrics

    def _empty_metrics(self) -> Dict[str, float]:
        """Return empty/default metrics."""
        return {'accuracy': 0.0, 'f1': 0.0, 'auc': 0.5}

    def evaluate_with_multiple_classifiers(
        self,
        embeddings: Dict[str, np.ndarray],
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate with both logistic regression and random forest.

        Args:
            embeddings: Dictionary mapping gene_id to embedding
            test_size: Fraction of data for testing
            random_state: Random seed

        Returns:
            Dictionary mapping classifier name to metrics
        """
        results = {}

        for clf_type in ['logistic', 'random_forest']:
            self.classifier_type = clf_type
            metrics = self.evaluate(embeddings, test_size, random_state)
            results[clf_type] = metrics

        return results

    def compute_reward(
        self,
        metrics: Dict[str, float],
        history: Optional[List[float]] = None
    ) -> float:
        """
        Compute reward from metrics using relative reward computation.

        Uses the primary metric for the task and computes relative
        improvement over history for clearer learning signals.

        Args:
            metrics: Dictionary of metrics
            history: List of previous primary metric values (oldest to newest)

        Returns:
            Reward value (normalized to [-1, 1])
        """
        current_metric = metrics.get(self.primary_metric, 0.0)

        # Always use RewardComputer for consistent normalization
        # For first iteration (no history), it returns absolute_reward in [-1, 1]
        result = self.reward_computer.compute(current_metric, history)
        logger.debug(
            f"Reward computed: total={result.total_reward:.4f}, "
            f"relative={result.relative_reward:.4f}, raw={result.raw_metric:.4f}"
        )
        return result.total_reward

    def compute_reward_detailed(
        self,
        metrics: Dict[str, float],
        history: Optional[List[float]] = None
    ) -> RewardResult:
        """
        Compute detailed reward breakdown.

        Args:
            metrics: Dictionary of metrics
            history: List of previous primary metric values

        Returns:
            RewardResult with full breakdown
        """
        current_metric = metrics.get(self.primary_metric, 0.0)
        return self.reward_computer.compute(current_metric, history)

    def generate_feedback(
        self,
        metrics: Dict[str, float],
        embeddings: Optional[Dict[str, np.ndarray]] = None,
        descriptions: Optional[Dict[str, str]] = None,
        best_metric: Optional[float] = None
    ) -> str:
        """
        Generate feedback for the Actor.

        Args:
            metrics: Evaluation metrics
            embeddings: Gene embeddings (for analysis)
            descriptions: Gene descriptions (for analysis)
            best_metric: Best metric achieved so far (for comparison)

        Returns:
            Feedback string
        """
        from .feedback import generate_feedback
        return generate_feedback(self.task_name, metrics, embeddings, descriptions, best_metric)

    @classmethod
    def get_available_tasks(cls) -> list:
        """Return list of available task names."""
        return list(cls.TASK_CLASSES.keys())

    def get_task_genes(self) -> list:
        """Return list of genes relevant to this task."""
        return self.task.get_task_genes()

    def update_reward_weights_for_trend(self, metric: float, trend: str = 'unknown') -> bool:
        """
        Update reward weights based on current trend (if adaptive is enabled).

        Args:
            metric: Latest evaluation metric
            trend: Current performance trend

        Returns:
            True if weights were updated
        """
        return self.reward_computer.update_weights_for_trend(metric, trend)

    def get_reward_weights(self) -> dict:
        """Get current reward weights."""
        return self.reward_computer.get_weights()

    def set_reward_weights(self, **kwargs):
        """Manually set reward weights."""
        self.reward_computer.set_weights(**kwargs)
