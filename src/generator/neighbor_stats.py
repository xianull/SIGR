"""
Neighbor Statistics Collection for SIGR Framework

Collects and aggregates neighbor statistics across all genes in an iteration.
This information helps the Actor understand what actually happened during
subgraph extraction, not just what was configured.

Key statistics:
- Per-edge-type neighbor counts and scores
- Quality distribution (high/medium/low relevance)
- High-frequency neighbors (potential noise)
- Token estimation for context management
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Set, Optional, Any
from collections import defaultdict

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EdgeTypeStats:
    """Statistics for a single edge type across all genes."""
    edge_type: str
    neighbor_count: int = 0
    avg_score: float = 0.0
    high_relevance_count: int = 0  # score >= 0.7
    low_relevance_count: int = 0   # score < 0.4
    unique_neighbors: Set[str] = field(default_factory=set)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'edge_type': self.edge_type,
            'neighbor_count': self.neighbor_count,
            'avg_score': self.avg_score,
            'high_relevance_count': self.high_relevance_count,
            'low_relevance_count': self.low_relevance_count,
            'unique_neighbor_count': len(self.unique_neighbors),
        }


@dataclass
class HighFrequencyNeighbor:
    """A neighbor that appears frequently across genes."""
    neighbor_id: str
    occurrence_count: int
    primary_edge_type: str
    avg_score: float
    is_task_relevant: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            'neighbor_id': self.neighbor_id,
            'occurrence_count': self.occurrence_count,
            'primary_edge_type': self.primary_edge_type,
            'avg_score': self.avg_score,
            'is_task_relevant': self.is_task_relevant,
        }


@dataclass
class IterationNeighborStats:
    """
    Aggregate neighbor statistics for an entire iteration.

    This class collects information across all genes processed in an iteration,
    providing the Actor with visibility into what actually happened during
    subgraph extraction.

    Important distinction:
    - total_unique_neighbors: Number of distinct neighbor nodes (去重后)
    - total_edges: Number of edges/connections (可能重复计算同一邻居)
    """
    # Per-edge-type statistics
    per_edge_type: Dict[str, EdgeTypeStats] = field(default_factory=dict)

    # Overall counts - 区分 unique neighbors vs edges
    total_unique_neighbors: int = 0  # 唯一邻居节点数（去重后）
    total_edges: int = 0             # 总边数（原来的计数方式）
    total_neighbors: int = 0         # 保留向后兼容，等于 total_edges

    # Overall quality distribution (based on edges)
    high_relevance_count: int = 0    # score >= 0.7
    medium_relevance_count: int = 0  # 0.4 <= score < 0.7
    low_relevance_count: int = 0     # score < 0.4

    # High-frequency neighbors (potential noise)
    high_frequency_neighbors: List[HighFrequencyNeighbor] = field(default_factory=list)

    # Token estimation
    estimated_context_tokens: int = 0
    genes_processed: int = 0

    # Thresholds used
    HIGH_RELEVANCE_THRESHOLD: float = 0.7
    LOW_RELEVANCE_THRESHOLD: float = 0.4
    HIGH_FREQUENCY_THRESHOLD: int = 3

    def to_dict(self) -> Dict[str, Any]:
        return {
            'per_edge_type': {k: v.to_dict() for k, v in self.per_edge_type.items()},
            'total_unique_neighbors': self.total_unique_neighbors,
            'total_edges': self.total_edges,
            'total_neighbors': self.total_neighbors,  # 向后兼容
            'quality_distribution': {
                'high': self.high_relevance_count,
                'medium': self.medium_relevance_count,
                'low': self.low_relevance_count,
            },
            'high_frequency_neighbors': [n.to_dict() for n in self.high_frequency_neighbors],
            'estimated_context_tokens': self.estimated_context_tokens,
            'genes_processed': self.genes_processed,
        }


class NeighborStatsCollector:
    """
    Collects neighbor statistics during subgraph extraction.

    Usage:
        collector = NeighborStatsCollector(task_genes=set(...))

        # For each gene processed
        collector.record_gene_neighbors(gene_id, neighbors, scores)

        # After all genes
        stats = collector.get_stats()
    """

    # Tokens per neighbor (rough estimate)
    TOKENS_PER_NEIGHBOR = 30

    def __init__(
        self,
        task_genes: Optional[Set[str]] = None,
        high_threshold: float = 0.7,
        low_threshold: float = 0.4,
        high_freq_threshold: int = 3,
    ):
        """
        Initialize the collector.

        Args:
            task_genes: Set of task-relevant genes (for marking hub nodes)
            high_threshold: Threshold for high relevance
            low_threshold: Threshold for low relevance
            high_freq_threshold: Minimum occurrences to be considered high-frequency
        """
        self.task_genes = task_genes or set()
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.high_freq_threshold = high_freq_threshold

        # Internal tracking
        self._edge_type_stats: Dict[str, EdgeTypeStats] = {}
        self._neighbor_occurrences: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {'count': 0, 'scores': [], 'edge_types': defaultdict(int)}
        )
        self._genes_processed = 0
        self._total_edges = 0           # 总边数
        self._seen_neighbors: Set[str] = set()  # 用于去重的邻居集合
        self._high_count = 0
        self._medium_count = 0
        self._low_count = 0

    def record_gene_neighbors(
        self,
        gene_id: str,
        neighbors: List[Tuple[str, str, str]],  # (neighbor_id, edge_type, direction)
        scores: Optional[Dict[str, Any]] = None,  # neighbor_id -> NeighborScore or dict
    ) -> None:
        """
        Record neighbors extracted for a single gene.

        Args:
            gene_id: The center gene
            neighbors: List of (neighbor_id, edge_type, direction) tuples
            scores: Optional dictionary mapping neighbor_id to score info
        """
        self._genes_processed += 1
        scores = scores or {}

        for neighbor_id, edge_type, direction in neighbors:
            self._total_edges += 1  # 每条边计数
            self._seen_neighbors.add(neighbor_id)  # 记录唯一邻居

            # Get score
            score_info = scores.get(neighbor_id)
            if score_info:
                if hasattr(score_info, 'total_score'):
                    score = score_info.total_score
                elif isinstance(score_info, dict):
                    score = score_info.get('total_score', 0.5)
                else:
                    score = 0.5
            else:
                score = 0.5

            # Classify by quality
            if score >= self.high_threshold:
                self._high_count += 1
            elif score >= self.low_threshold:
                self._medium_count += 1
            else:
                self._low_count += 1

            # Update edge type stats
            if edge_type not in self._edge_type_stats:
                self._edge_type_stats[edge_type] = EdgeTypeStats(edge_type=edge_type)

            stats = self._edge_type_stats[edge_type]
            stats.neighbor_count += 1
            stats.unique_neighbors.add(neighbor_id)

            if score >= self.high_threshold:
                stats.high_relevance_count += 1
            elif score < self.low_threshold:
                stats.low_relevance_count += 1

            # Track for high-frequency analysis
            occ = self._neighbor_occurrences[neighbor_id]
            occ['count'] += 1
            occ['scores'].append(score)
            occ['edge_types'][edge_type] += 1

    def get_stats(self) -> IterationNeighborStats:
        """
        Compute and return the aggregated statistics.

        Returns:
            IterationNeighborStats object with all collected data
        """
        total_unique = len(self._seen_neighbors)

        stats = IterationNeighborStats(
            total_unique_neighbors=total_unique,
            total_edges=self._total_edges,
            total_neighbors=self._total_edges,  # 向后兼容
            high_relevance_count=self._high_count,
            medium_relevance_count=self._medium_count,
            low_relevance_count=self._low_count,
            genes_processed=self._genes_processed,
            # 使用唯一邻居数估算 tokens，更准确
            estimated_context_tokens=total_unique * self.TOKENS_PER_NEIGHBOR,
        )

        # Finalize per-edge-type stats
        for edge_type, edge_stats in self._edge_type_stats.items():
            # Compute average score for this edge type
            type_scores = []
            for neighbor_id in edge_stats.unique_neighbors:
                occ = self._neighbor_occurrences[neighbor_id]
                if occ['scores']:
                    type_scores.append(np.mean(occ['scores']))

            edge_stats.avg_score = np.mean(type_scores) if type_scores else 0.5
            stats.per_edge_type[edge_type] = edge_stats

        # Identify high-frequency neighbors
        high_freq = []
        for neighbor_id, occ in self._neighbor_occurrences.items():
            if occ['count'] >= self.high_freq_threshold:
                # Get primary edge type
                primary_edge = max(occ['edge_types'].items(), key=lambda x: x[1])[0]
                avg_score = np.mean(occ['scores']) if occ['scores'] else 0.5

                high_freq.append(HighFrequencyNeighbor(
                    neighbor_id=neighbor_id,
                    occurrence_count=occ['count'],
                    primary_edge_type=primary_edge,
                    avg_score=avg_score,
                    is_task_relevant=neighbor_id in self.task_genes,
                ))

        # Sort by occurrence count (descending)
        high_freq.sort(key=lambda x: (-x.occurrence_count, x.avg_score))
        stats.high_frequency_neighbors = high_freq[:10]  # Top 10

        return stats


def format_neighbor_stats_for_actor(stats: IterationNeighborStats) -> str:
    """
    Format neighbor statistics into a human-readable section for Bio-CoT prompt.

    Args:
        stats: IterationNeighborStats object

    Returns:
        Formatted string for inclusion in Actor prompt
    """
    if stats.total_edges == 0:
        return "## CURRENT ITERATION NEIGHBOR STATISTICS\n\nNo neighbors extracted in this iteration."

    lines = ["## CURRENT ITERATION NEIGHBOR STATISTICS\n"]

    # Per-edge-type distribution
    lines.append("### Per-Edge-Type Distribution:")
    for edge_type, edge_stats in sorted(
        stats.per_edge_type.items(),
        key=lambda x: x[1].neighbor_count,
        reverse=True
    ):
        lines.append(
            f"- **{edge_type}**: {edge_stats.neighbor_count} edges "
            f"(avg_score={edge_stats.avg_score:.2f}, "
            f"unique={len(edge_stats.unique_neighbors)})"
        )
    # 清晰区分 unique neighbors vs edges
    lines.append(f"- **Total**: {stats.total_unique_neighbors} unique neighbors ({stats.total_edges} edges) across {stats.genes_processed} genes\n")

    # Quality distribution
    total = max(stats.total_edges, 1)
    high_pct = stats.high_relevance_count / total * 100
    med_pct = stats.medium_relevance_count / total * 100
    low_pct = stats.low_relevance_count / total * 100

    lines.append("### Quality Distribution:")
    lines.append(f"- High relevance (score >= 0.7): {stats.high_relevance_count} ({high_pct:.0f}%)")
    lines.append(f"- Medium relevance (0.4 - 0.7): {stats.medium_relevance_count} ({med_pct:.0f}%)")
    lines.append(f"- Low relevance (< 0.4): {stats.low_relevance_count} ({low_pct:.0f}%)\n")

    # High-frequency neighbors
    if stats.high_frequency_neighbors:
        lines.append("### High-Frequency Neighbors (appeared in 3+ genes):")
        for hf in stats.high_frequency_neighbors[:5]:  # Top 5
            noise_flag = ""
            if not hf.is_task_relevant and hf.avg_score < 0.5:
                noise_flag = " <- **potential noise**"
            elif hf.is_task_relevant:
                noise_flag = " <- task-relevant hub"

            lines.append(
                f"- {hf.neighbor_id} ({hf.occurrence_count}x, {hf.primary_edge_type}, "
                f"avg_score={hf.avg_score:.2f}){noise_flag}"
            )
        lines.append("")

    # Context size warning
    lines.append("### Context Size Estimate:")
    lines.append(f"- Estimated tokens: ~{stats.estimated_context_tokens:,}")
    if stats.estimated_context_tokens > 4000:
        lines.append(
            "- **WARNING**: Context size is large. Consider reducing max_neighbors "
            "or removing low-relevance edge types."
        )
    lines.append("")

    # Recommendations based on analysis
    lines.append("### Automatic Recommendations:")
    recommendations = []

    # Check for noise-heavy edge types
    for edge_type, edge_stats in stats.per_edge_type.items():
        if edge_stats.neighbor_count > 0:
            low_ratio = edge_stats.low_relevance_count / edge_stats.neighbor_count
            if low_ratio > 0.5 and edge_stats.neighbor_count >= 10:
                recommendations.append(
                    f"- {edge_type} has {low_ratio*100:.0f}% low-relevance neighbors "
                    f"-> consider reducing or removing"
                )

    # Check for high-frequency low-score nodes
    noisy_hubs = [
        hf for hf in stats.high_frequency_neighbors
        if not hf.is_task_relevant and hf.avg_score < 0.4
    ]
    if noisy_hubs:
        names = ", ".join(hf.neighbor_id for hf in noisy_hubs[:3])
        recommendations.append(
            f"- High-frequency low-score neighbors detected ({names}) "
            f"-> consider excluding via exclude_neighbors"
        )

    # Check quality ratio
    if low_pct > 40:
        recommendations.append(
            f"- {low_pct:.0f}% of neighbors are low relevance "
            f"-> consider threshold filtering or reducing max_neighbors"
        )

    if recommendations:
        lines.extend(recommendations)
    else:
        lines.append("- No immediate concerns. Quality distribution looks reasonable.")

    return "\n".join(lines)
