"""
SIGR - Self-Iterative Gene Representation Framework

A framework for generating enhanced gene representations using
knowledge graph augmentation and LLM-based text generation.

MDP-based optimization:
- State: Gene info + KG neighborhood + history
- Action: Strategy (edge_types, hops, sampling, prompt)
- Reward: Downstream task performance
- Policy: LLM reflection for strategy updates
"""

__version__ = "0.1.0"

from .sigr_framework import (
    SIGRFramework,
    run_training,
    run_all_tasks,
    MockLLMClient,
    list_available_tasks,
    get_task_info,
    AVAILABLE_TASKS,
)

from .actor import ActorAgent, Strategy, StrategyConfig
from .generator import extract_subgraph, format_subgraph, TextGenerator
from .encoder import GeneEncoder
from .evaluator import TaskEvaluator
from .utils import load_kg, get_all_genes, get_gene_info
from .utils.logger import SIGRLogger, setup_logging

__all__ = [
    # Main framework
    "SIGRFramework",
    "run_training",
    "run_all_tasks",
    "MockLLMClient",
    "list_available_tasks",
    "get_task_info",
    "AVAILABLE_TASKS",
    # Actor
    "ActorAgent",
    "Strategy",
    "StrategyConfig",
    # Generator
    "extract_subgraph",
    "format_subgraph",
    "TextGenerator",
    # Encoder
    "GeneEncoder",
    # Evaluator
    "TaskEvaluator",
    # Utils
    "load_kg",
    "get_all_genes",
    "get_gene_info",
    "SIGRLogger",
    "setup_logging",
]
