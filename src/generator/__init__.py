"""Generator module for SIGR framework."""

from .subgraph_extractor import (
    extract_subgraph,
    expand_subgraph,
    extract_subgraph_with_scoring,
    extract_subgraphs_batch_with_scoring,
    get_neighbor_analysis_for_actor,
)
from .formatter import format_subgraph
from .text_generator import TextGenerator, MockTextGenerator
from .neighbor_scorer import NeighborScorer, NeighborScore
from .neighbor_presenter import NeighborPresenter, format_neighbor_analysis_section
from .neighbor_stats import (
    NeighborStatsCollector,
    IterationNeighborStats,
    EdgeTypeStats,
    HighFrequencyNeighbor,
    format_neighbor_stats_for_actor,
)

__all__ = [
    # Subgraph extraction
    "extract_subgraph",
    "expand_subgraph",
    "extract_subgraph_with_scoring",
    "extract_subgraphs_batch_with_scoring",
    "get_neighbor_analysis_for_actor",
    # Formatting
    "format_subgraph",
    # Text generation
    "TextGenerator",
    "MockTextGenerator",
    # Neighbor scoring
    "NeighborScorer",
    "NeighborScore",
    "NeighborPresenter",
    "format_neighbor_analysis_section",
    # Neighbor statistics
    "NeighborStatsCollector",
    "IterationNeighborStats",
    "EdgeTypeStats",
    "HighFrequencyNeighbor",
    "format_neighbor_stats_for_actor",
]
