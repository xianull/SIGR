"""Evaluation tasks for SIGR framework."""

from .base_task import BaseTask
from .ppi_task import PPITask
from .genetype_task import GeneTypeTask
from .geneattribute_task import GeneAttributeTask
from .ggi_task import GGITask
from .cell_task import CellTask
from .perturbation_task import PerturbationTask

__all__ = [
    "BaseTask",
    "PPITask",
    "GeneTypeTask",
    "GeneAttributeTask",
    "GGITask",
    "CellTask",
    "PerturbationTask",
]
