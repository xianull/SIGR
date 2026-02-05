"""
Neighbor Score Presentation for SIGR Framework

Formats neighbor scoring information for Actor consumption.
Provides human-readable summaries that help the Actor make
informed decisions about neighbor selection.

Output modes:
- summary: High-level overview with recommendations
- detailed: Full breakdown per neighbor
- minimal: Just the key numbers
"""

import logging
from typing import Dict, List, Optional, Any

from .neighbor_scorer import NeighborScore


logger = logging.getLogger(__name__)


class NeighborPresenter:
    """
    Formats neighbor scores into Actor-readable text.

    Transforms raw NeighborScore objects into structured
    text that helps the Actor understand neighbor quality
    and make selection decisions.
    """

    # Default thresholds for categorization
    HIGH_RELEVANCE_THRESHOLD = 0.7
    LOW_RELEVANCE_THRESHOLD = 0.4

    # Maximum neighbors to show per category in summary mode
    MAX_NEIGHBORS_PER_CATEGORY = 5

    def __init__(
        self,
        high_threshold: float = 0.7,
        low_threshold: float = 0.4,
        max_per_category: int = 5,
    ):
        """
        Initialize the presenter.

        Args:
            high_threshold: Score threshold for high relevance
            low_threshold: Score threshold for low relevance
            max_per_category: Maximum neighbors to show per category
        """
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.max_per_category = max_per_category

    def format_for_actor(
        self,
        gene_id: str,
        scored_neighbors: Dict[str, NeighborScore],
        mode: str = 'summary',
    ) -> str:
        """
        Format neighbor scores for Actor consumption.

        Args:
            gene_id: The center gene ID
            scored_neighbors: Dictionary of neighbor scores
            mode: Output mode ('summary', 'detailed', 'minimal')

        Returns:
            Formatted text for inclusion in Actor prompt
        """
        if not scored_neighbors:
            return f"No neighbors found for {gene_id}."

        if mode == 'detailed':
            return self._format_detailed(gene_id, scored_neighbors)
        elif mode == 'minimal':
            return self._format_minimal(gene_id, scored_neighbors)
        else:
            return self._format_summary(gene_id, scored_neighbors)

    def _format_summary(
        self,
        gene_id: str,
        scored_neighbors: Dict[str, NeighborScore],
    ) -> str:
        """
        Format summary mode output.

        Groups neighbors into high/medium/low relevance categories
        and provides recommendations.

        Args:
            gene_id: Center gene ID
            scored_neighbors: Dictionary of scores

        Returns:
            Summary text
        """
        # Categorize neighbors
        high, medium, low = [], [], []
        for score in scored_neighbors.values():
            if score.total_score >= self.high_threshold:
                high.append(score)
            elif score.total_score >= self.low_threshold:
                medium.append(score)
            else:
                low.append(score)

        # Sort by score within each category
        high.sort(key=lambda x: x.total_score, reverse=True)
        medium.sort(key=lambda x: x.total_score, reverse=True)
        low.sort(key=lambda x: x.total_score, reverse=True)

        # Format output
        lines = [f"## Neighbor Analysis for {gene_id}"]
        lines.append(f"Total neighbors: {len(scored_neighbors)}")
        lines.append("")

        # High relevance
        if high:
            lines.append(f"### High relevance (score >= {self.high_threshold}): {len(high)} neighbors")
            top_high = high[:self.max_per_category]
            neighbor_strs = [f"{s.neighbor_id} ({s.total_score:.2f}, {s.edge_type})"
                          for s in top_high]
            lines.append("  " + ", ".join(neighbor_strs))
            if len(high) > self.max_per_category:
                lines.append(f"  ... and {len(high) - self.max_per_category} more")
            lines.append("")

        # Medium relevance
        if medium:
            lines.append(f"### Medium relevance ({self.low_threshold} - {self.high_threshold}): {len(medium)} neighbors")
            top_medium = medium[:self.max_per_category]
            neighbor_strs = [f"{s.neighbor_id} ({s.total_score:.2f}, {s.edge_type})"
                          for s in top_medium]
            lines.append("  " + ", ".join(neighbor_strs))
            if len(medium) > self.max_per_category:
                lines.append(f"  ... and {len(medium) - self.max_per_category} more")
            lines.append("")

        # Low relevance
        if low:
            lines.append(f"### Low relevance (score < {self.low_threshold}): {len(low)} neighbors")
            top_low = low[:self.max_per_category]
            neighbor_strs = [f"{s.neighbor_id} ({s.total_score:.2f}, {s.edge_type})"
                          for s in top_low]
            lines.append("  " + ", ".join(neighbor_strs))
            if len(low) > self.max_per_category:
                lines.append(f"  ... and {len(low) - self.max_per_category} more")
            lines.append("")

        # Recommendations
        lines.append("### Recommendations")
        if low and len(low) > 3:
            low_ids = [s.neighbor_id for s in low[:5]]
            lines.append(f"- Consider excluding low-relevance neighbors: {', '.join(low_ids)}")
        if not high and medium:
            lines.append("- No high-relevance neighbors found. Consider broadening edge types.")
        if high and len(high) > 10:
            lines.append(f"- Many high-relevance neighbors ({len(high)}). Consider using top_k selection.")

        return "\n".join(lines)

    def _format_detailed(
        self,
        gene_id: str,
        scored_neighbors: Dict[str, NeighborScore],
    ) -> str:
        """
        Format detailed mode output.

        Shows full score breakdown for each neighbor.

        Args:
            gene_id: Center gene ID
            scored_neighbors: Dictionary of scores

        Returns:
            Detailed text
        """
        lines = [f"## Detailed Neighbor Analysis for {gene_id}"]
        lines.append(f"Total neighbors: {len(scored_neighbors)}")
        lines.append("")

        # Sort by total score
        sorted_scores = sorted(
            scored_neighbors.values(),
            key=lambda x: x.total_score,
            reverse=True
        )

        for i, score in enumerate(sorted_scores, 1):
            lines.append(f"### {i}. {score.neighbor_id}")
            lines.append(f"  - Edge type: {score.edge_type} ({score.direction})")
            lines.append(f"  - Total score: {score.total_score:.3f} (confidence: {score.confidence:.2f})")
            lines.append(f"  - Breakdown:")
            lines.append(f"    * Structural: {score.structural_score:.3f}")
            lines.append(f"    * Task relevance: {score.task_relevance_score:.3f}")
            lines.append(f"    * Semantic: {score.semantic_score:.3f}")
            lines.append(f"    * Memory: {score.memory_score:.3f}")
            lines.append("")

        return "\n".join(lines)

    def _format_minimal(
        self,
        gene_id: str,
        scored_neighbors: Dict[str, NeighborScore],
    ) -> str:
        """
        Format minimal mode output.

        Just the key statistics.

        Args:
            gene_id: Center gene ID
            scored_neighbors: Dictionary of scores

        Returns:
            Minimal text
        """
        if not scored_neighbors:
            return f"{gene_id}: 0 neighbors"

        scores = [s.total_score for s in scored_neighbors.values()]
        avg_score = sum(scores) / len(scores)
        high_count = sum(1 for s in scores if s >= self.high_threshold)
        low_count = sum(1 for s in scores if s < self.low_threshold)

        return (
            f"{gene_id}: {len(scored_neighbors)} neighbors, "
            f"avg={avg_score:.2f}, high={high_count}, low={low_count}"
        )

    def format_multiple_genes(
        self,
        all_scores: Dict[str, Dict[str, NeighborScore]],
        mode: str = 'summary',
    ) -> str:
        """
        Format scores for multiple genes.

        Args:
            all_scores: Dictionary mapping gene_id to neighbor scores
            mode: Output mode

        Returns:
            Combined formatted text
        """
        if not all_scores:
            return "No neighbor analysis available."

        sections = []
        for gene_id, scores in all_scores.items():
            section = self.format_for_actor(gene_id, scores, mode)
            sections.append(section)

        if mode == 'minimal':
            return "\n".join(sections)
        else:
            return "\n\n---\n\n".join(sections)

    def get_selection_guidance(
        self,
        all_scores: Dict[str, Dict[str, NeighborScore]],
    ) -> str:
        """
        Generate selection guidance text for Actor.

        Provides actionable recommendations based on neighbor analysis.

        Args:
            all_scores: Dictionary mapping gene_id to neighbor scores

        Returns:
            Selection guidance text
        """
        if not all_scores:
            return "No neighbor analysis available for selection guidance."

        # Aggregate statistics
        total_neighbors = 0
        total_high = 0
        total_low = 0
        all_low_neighbors = []

        for gene_id, scores in all_scores.items():
            total_neighbors += len(scores)
            for neighbor_id, score in scores.items():
                if score.total_score >= self.high_threshold:
                    total_high += 1
                elif score.total_score < self.low_threshold:
                    total_low += 1
                    all_low_neighbors.append((neighbor_id, score.total_score))

        # Sort low neighbors by score
        all_low_neighbors.sort(key=lambda x: x[1])

        lines = ["## Neighbor Selection Guidance"]
        lines.append("")
        lines.append(f"Analyzed {total_neighbors} total neighbors across {len(all_scores)} genes.")
        lines.append(f"- High relevance: {total_high} ({total_high/max(total_neighbors,1)*100:.1f}%)")
        lines.append(f"- Low relevance: {total_low} ({total_low/max(total_neighbors,1)*100:.1f}%)")
        lines.append("")

        # Recommendations
        lines.append("### Selection Options")
        lines.append("")
        lines.append("You may optionally adjust neighbor selection in your strategy:")
        lines.append("")
        lines.append('1. **Keep all** (default): `"neighbor_selection_mode": "retain_all"`')
        lines.append('   - Use when: You want maximum context, or high-relevance ratio is high')
        lines.append("")
        lines.append('2. **Top-k selection**: `"neighbor_selection_mode": "top_k", "top_k": N`')
        lines.append('   - Use when: Too many neighbors are diluting signal')
        lines.append('   - Recommendation: N=10-20 for focused context')
        lines.append("")
        lines.append('3. **Threshold filtering**: `"neighbor_selection_mode": "threshold", "threshold": 0.5`')
        lines.append('   - Use when: Want to remove only low-quality neighbors')
        lines.append('   - Recommendation: threshold=0.4-0.5')
        lines.append("")
        lines.append('4. **Explicit exclusion**: `"neighbor_selection_mode": "exclude", "exclude_neighbors": ["ID1", "ID2"]`')
        lines.append('   - Use when: Specific neighbors are known to be noisy')

        if all_low_neighbors and len(all_low_neighbors) >= 3:
            lines.append("")
            lines.append("### Candidates for Exclusion")
            lines.append("Lowest scoring neighbors (potential noise):")
            for neighbor_id, score in all_low_neighbors[:5]:
                lines.append(f"  - {neighbor_id}: {score:.3f}")

        return "\n".join(lines)


def format_neighbor_analysis_section(
    all_scores: Dict[str, Dict[str, NeighborScore]],
    presenter: Optional[NeighborPresenter] = None,
    mode: str = 'summary',
    include_guidance: bool = True,
) -> str:
    """
    Convenience function to format complete neighbor analysis section.

    Args:
        all_scores: Dictionary mapping gene_id to neighbor scores
        presenter: Optional custom presenter
        mode: Output mode
        include_guidance: Whether to include selection guidance

    Returns:
        Complete formatted section for Actor prompt
    """
    if presenter is None:
        presenter = NeighborPresenter()

    sections = []

    # Main analysis
    analysis = presenter.format_multiple_genes(all_scores, mode)
    sections.append(analysis)

    # Selection guidance
    if include_guidance:
        guidance = presenter.get_selection_guidance(all_scores)
        sections.append(guidance)

    return "\n\n".join(sections)
