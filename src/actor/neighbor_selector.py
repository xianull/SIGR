"""
Neighbor Selection for SIGR Framework

Executes neighbor selection policies specified by the Actor.
Filters neighbors based on scores and selection criteria.

Selection modes:
- retain_all: Keep all neighbors (default)
- top_k: Keep top k neighbors by score
- threshold: Keep neighbors above score threshold
- exclude: Exclude specific neighbors by ID
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any

from ..generator.neighbor_scorer import NeighborScore


logger = logging.getLogger(__name__)


@dataclass
class NeighborSelectionPolicy:
    """
    Policy for neighbor selection.

    Attributes:
        mode: Selection mode ('retain_all', 'top_k', 'threshold', 'exclude')
        top_k: Number of neighbors to keep (for 'top_k' mode)
        threshold: Score threshold (for 'threshold' mode)
        exclude_neighbors: List of neighbor IDs to exclude (for 'exclude' mode)
        per_edge_type: Optional per-edge-type selection limits
        reason: Optional explanation for selection choice
    """
    mode: str = 'retain_all'
    top_k: Optional[int] = None
    threshold: Optional[float] = None
    exclude_neighbors: Optional[List[str]] = None
    per_edge_type: Optional[Dict[str, int]] = None
    reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'mode': self.mode,
            'top_k': self.top_k,
            'threshold': self.threshold,
            'exclude_neighbors': self.exclude_neighbors,
            'per_edge_type': self.per_edge_type,
            'reason': self.reason,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NeighborSelectionPolicy':
        """Create from dictionary."""
        return cls(
            mode=data.get('mode', 'retain_all'),
            top_k=data.get('top_k'),
            threshold=data.get('threshold'),
            exclude_neighbors=data.get('exclude_neighbors'),
            per_edge_type=data.get('per_edge_type'),
            reason=data.get('reason'),
        )

    @classmethod
    def from_strategy(cls, strategy: Dict[str, Any]) -> 'NeighborSelectionPolicy':
        """
        Extract neighbor selection policy from strategy dict.

        Args:
            strategy: Strategy dictionary from Actor

        Returns:
            NeighborSelectionPolicy instance
        """
        return cls(
            mode=strategy.get('neighbor_selection_mode', 'retain_all'),
            top_k=strategy.get('top_k'),
            threshold=strategy.get('threshold'),
            exclude_neighbors=strategy.get('exclude_neighbors'),
            per_edge_type=strategy.get('neighbors_per_type'),
            reason=strategy.get('neighbor_selection_reason'),
        )


@dataclass
class SelectionResult:
    """
    Result of neighbor selection.

    Attributes:
        selected: List of selected neighbor IDs
        excluded: List of excluded neighbor IDs
        policy_used: The policy that was applied
        stats: Statistics about the selection
    """
    selected: List[str] = field(default_factory=list)
    excluded: List[str] = field(default_factory=list)
    policy_used: Optional[NeighborSelectionPolicy] = None
    stats: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'selected': self.selected,
            'excluded': self.excluded,
            'policy': self.policy_used.to_dict() if self.policy_used else None,
            'stats': self.stats,
        }


class NeighborSelector:
    """
    Executes neighbor selection based on Actor's policy.

    Takes scored neighbors and applies selection criteria
    to determine which neighbors to include in context.
    """

    # Valid selection modes
    VALID_MODES = {'retain_all', 'top_k', 'threshold', 'exclude'}

    # Default values
    DEFAULT_TOP_K = 20
    DEFAULT_THRESHOLD = 0.4

    def __init__(
        self,
        default_mode: str = 'retain_all',
        default_top_k: int = 20,
        default_threshold: float = 0.4,
    ):
        """
        Initialize the selector.

        Args:
            default_mode: Default selection mode
            default_top_k: Default k value for top_k mode
            default_threshold: Default threshold for threshold mode
        """
        self.default_mode = default_mode
        self.default_top_k = default_top_k
        self.default_threshold = default_threshold

    def apply_selection(
        self,
        scored_neighbors: Dict[str, NeighborScore],
        policy: Optional[NeighborSelectionPolicy] = None,
    ) -> SelectionResult:
        """
        Apply selection policy to scored neighbors.

        Args:
            scored_neighbors: Dictionary mapping neighbor_id to NeighborScore
            policy: Selection policy to apply

        Returns:
            SelectionResult with selected and excluded neighbors
        """
        if not scored_neighbors:
            return SelectionResult(
                selected=[],
                excluded=[],
                policy_used=policy,
                stats={'total': 0, 'selected': 0, 'excluded': 0}
            )

        # Use default policy if none provided
        if policy is None:
            policy = NeighborSelectionPolicy(mode=self.default_mode)

        # Validate mode
        mode = policy.mode
        if mode not in self.VALID_MODES:
            logger.warning(f"Invalid selection mode '{mode}', using 'retain_all'")
            mode = 'retain_all'

        # Apply selection based on mode
        if mode == 'retain_all':
            selected, excluded = self._select_all(scored_neighbors)
        elif mode == 'top_k':
            k = policy.top_k or self.default_top_k
            selected, excluded = self._select_top_k(scored_neighbors, k)
        elif mode == 'threshold':
            threshold = policy.threshold or self.default_threshold
            selected, excluded = self._select_threshold(scored_neighbors, threshold)
        elif mode == 'exclude':
            exclude_set = set(policy.exclude_neighbors or [])
            selected, excluded = self._select_exclude(scored_neighbors, exclude_set)
        else:
            selected, excluded = self._select_all(scored_neighbors)

        # Apply per-edge-type limits if specified
        if policy.per_edge_type:
            selected, additional_excluded = self._apply_per_edge_limits(
                scored_neighbors, selected, policy.per_edge_type
            )
            excluded.extend(additional_excluded)

        # Compute statistics
        stats = {
            'total': len(scored_neighbors),
            'selected': len(selected),
            'excluded': len(excluded),
            'selection_rate': len(selected) / max(len(scored_neighbors), 1),
        }

        if selected:
            selected_scores = [scored_neighbors[n].total_score for n in selected]
            stats['avg_selected_score'] = sum(selected_scores) / len(selected_scores)
            stats['min_selected_score'] = min(selected_scores)

        if excluded:
            excluded_scores = [scored_neighbors[n].total_score for n in excluded]
            stats['avg_excluded_score'] = sum(excluded_scores) / len(excluded_scores)

        result = SelectionResult(
            selected=selected,
            excluded=excluded,
            policy_used=policy,
            stats=stats
        )

        logger.debug(
            f"Selection applied: mode={mode}, selected={len(selected)}, "
            f"excluded={len(excluded)}, rate={stats['selection_rate']:.2%}"
        )

        return result

    def _select_all(
        self,
        scored_neighbors: Dict[str, NeighborScore],
    ) -> tuple:
        """Select all neighbors."""
        return list(scored_neighbors.keys()), []

    def _select_top_k(
        self,
        scored_neighbors: Dict[str, NeighborScore],
        k: int,
    ) -> tuple:
        """Select top k neighbors by score."""
        sorted_neighbors = sorted(
            scored_neighbors.items(),
            key=lambda x: x[1].total_score,
            reverse=True
        )

        selected = [n for n, _ in sorted_neighbors[:k]]
        excluded = [n for n, _ in sorted_neighbors[k:]]

        return selected, excluded

    def _select_threshold(
        self,
        scored_neighbors: Dict[str, NeighborScore],
        threshold: float,
    ) -> tuple:
        """Select neighbors above threshold."""
        selected = []
        excluded = []

        for neighbor_id, score in scored_neighbors.items():
            if score.total_score >= threshold:
                selected.append(neighbor_id)
            else:
                excluded.append(neighbor_id)

        return selected, excluded

    def _select_exclude(
        self,
        scored_neighbors: Dict[str, NeighborScore],
        exclude_set: Set[str],
    ) -> tuple:
        """Exclude specific neighbors."""
        selected = []
        excluded = []

        for neighbor_id in scored_neighbors:
            if neighbor_id in exclude_set:
                excluded.append(neighbor_id)
            else:
                selected.append(neighbor_id)

        return selected, excluded

    def _apply_per_edge_limits(
        self,
        scored_neighbors: Dict[str, NeighborScore],
        selected: List[str],
        per_edge_limits: Dict[str, int],
    ) -> tuple:
        """Apply per-edge-type neighbor limits."""
        # Group selected by edge type
        by_edge_type: Dict[str, List[tuple]] = {}
        for neighbor_id in selected:
            score = scored_neighbors[neighbor_id]
            edge_type = score.edge_type
            if edge_type not in by_edge_type:
                by_edge_type[edge_type] = []
            by_edge_type[edge_type].append((neighbor_id, score.total_score))

        # Apply limits
        final_selected = []
        additional_excluded = []

        for edge_type, neighbors in by_edge_type.items():
            # Sort by score
            neighbors.sort(key=lambda x: x[1], reverse=True)

            # Get limit for this edge type
            limit = per_edge_limits.get(edge_type, len(neighbors))

            # Select up to limit
            for i, (neighbor_id, _) in enumerate(neighbors):
                if i < limit:
                    final_selected.append(neighbor_id)
                else:
                    additional_excluded.append(neighbor_id)

        return final_selected, additional_excluded

    def select_for_multiple_genes(
        self,
        all_scores: Dict[str, Dict[str, NeighborScore]],
        policies: Optional[Dict[str, NeighborSelectionPolicy]] = None,
        default_policy: Optional[NeighborSelectionPolicy] = None,
    ) -> Dict[str, SelectionResult]:
        """
        Apply selection for multiple genes.

        Args:
            all_scores: Dictionary mapping gene_id to neighbor scores
            policies: Optional per-gene policies
            default_policy: Default policy for genes without specific policy

        Returns:
            Dictionary mapping gene_id to SelectionResult
        """
        results = {}
        policies = policies or {}
        default_policy = default_policy or NeighborSelectionPolicy()

        for gene_id, scores in all_scores.items():
            policy = policies.get(gene_id, default_policy)
            results[gene_id] = self.apply_selection(scores, policy)

        return results

    def get_selected_neighbor_ids(
        self,
        results: Dict[str, SelectionResult],
    ) -> Set[str]:
        """
        Get all selected neighbor IDs across all genes.

        Args:
            results: Dictionary of selection results

        Returns:
            Set of all selected neighbor IDs
        """
        selected = set()
        for result in results.values():
            selected.update(result.selected)
        return selected

    def summarize_selections(
        self,
        results: Dict[str, SelectionResult],
    ) -> Dict[str, Any]:
        """
        Summarize selection results across all genes.

        Args:
            results: Dictionary of selection results

        Returns:
            Summary statistics
        """
        total_original = 0
        total_selected = 0
        total_excluded = 0

        for result in results.values():
            total_original += result.stats.get('total', 0)
            total_selected += result.stats.get('selected', 0)
            total_excluded += result.stats.get('excluded', 0)

        return {
            'num_genes': len(results),
            'total_original_neighbors': total_original,
            'total_selected': total_selected,
            'total_excluded': total_excluded,
            'overall_selection_rate': (
                total_selected / max(total_original, 1)
            ),
        }
