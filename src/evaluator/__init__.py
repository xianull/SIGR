"""Evaluator module for SIGR framework."""

from .evaluator import TaskEvaluator
from .feedback import generate_feedback
from .classifiers import get_classifier, ClassifierFactory
from .metrics import compute_clustering_metrics, compute_embedding_statistics

from .tasks import (
    BaseTask,
    PPITask,
    GeneTypeTask,
    GeneAttributeTask,
    GGITask,
    CellTask,
)

__all__ = [
    # Main evaluator
    "TaskEvaluator",
    "generate_feedback",
    # Classifiers
    "get_classifier",
    "ClassifierFactory",
    # Metrics
    "compute_clustering_metrics",
    "compute_embedding_statistics",
    # Tasks
    "BaseTask",
    "PPITask",
    "GeneTypeTask",
    "GeneAttributeTask",
    "GGITask",
    "CellTask",
]
