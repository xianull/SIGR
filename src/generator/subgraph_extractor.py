"""
Subgraph Extraction for SIGR Framework

Extracts subgraphs from the knowledge graph based on Actor's strategy.
Supports neighbor scoring and selection for noise filtering.
"""

import logging
from typing import Dict, List, Tuple, Any, Set, Optional

import numpy as np
import networkx as nx

from ..utils.kg_utils import get_neighbors, EDGE_LABEL_MAPPING

logger = logging.getLogger(__name__)


# =============================================================================
# Neighbor Scoring Integration
# =============================================================================

def extract_subgraph_with_scoring(
    gene_id: str,
    strategy: Dict[str, Any],
    kg: nx.DiGraph,
    edge_weights: Dict[str, float] = None,
    task_genes: Optional[Set[str]] = None,
    memory: Optional[Any] = None,
    embeddings: Optional[Dict[str, np.ndarray]] = None,
    enable_neighbor_selection: bool = False,
) -> Tuple[nx.DiGraph, Optional[Dict[str, Any]]]:
    """
    Extract a subgraph with optional neighbor scoring and selection.

    This is an enhanced version of extract_subgraph that supports:
    1. Neighbor relevance scoring
    2. Actor-guided neighbor selection
    3. Filtering of low-relevance neighbors

    Args:
        gene_id: The central gene symbol
        strategy: Strategy dictionary (see extract_subgraph for details)
        kg: The full knowledge graph
        edge_weights: Dynamic weights for each edge type from Memory
        task_genes: Set of task-relevant genes for scoring
        memory: Memory instance for historical effectiveness
        embeddings: Embeddings for semantic scoring
        enable_neighbor_selection: Whether to apply neighbor selection

    Returns:
        Tuple of (subgraph, neighbor_scores_info)
        - subgraph: NetworkX DiGraph representing the subgraph
        - neighbor_scores_info: Dictionary with scoring details (or None if disabled)
    """
    # First, extract the full subgraph
    subgraph = extract_subgraph(gene_id, strategy, kg, edge_weights)

    # If neighbor selection is not enabled, return as-is
    if not enable_neighbor_selection:
        return subgraph, None

    # Import scoring components
    try:
        from .neighbor_scorer import NeighborScorer
        from ..actor.neighbor_selector import NeighborSelector, NeighborSelectionPolicy
    except ImportError:
        logger.warning("Neighbor scoring modules not available, skipping selection")
        return subgraph, None

    # Get selection policy from strategy
    selection_mode = strategy.get('neighbor_selection_mode', 'retain_all')

    # If retain_all, no need to score/filter
    if selection_mode == 'retain_all':
        return subgraph, {'mode': 'retain_all', 'filtered': False}

    # Score neighbors
    scorer = NeighborScorer(kg)

    # Get center gene embedding if available
    center_embedding = embeddings.get(gene_id) if embeddings else None

    # Collect all neighbors from subgraph
    neighbors = []
    for _, neighbor, data in subgraph.out_edges(gene_id, data=True):
        edge_type = data.get('edge_type', 'unknown')
        neighbors.append((neighbor, edge_type, 'out'))

    # Score all neighbors
    scores = scorer.score_neighbors(
        center_gene=gene_id,
        neighbors=neighbors,
        task_genes=task_genes,
        memory=memory,
        embeddings=embeddings,
        center_embedding=center_embedding,
    )

    # Create selection policy
    policy = NeighborSelectionPolicy(
        mode=selection_mode,
        top_k=strategy.get('neighbor_selection_top_k'),
        threshold=strategy.get('neighbor_selection_threshold'),
        exclude_neighbors=strategy.get('exclude_neighbors'),
    )

    # Apply selection
    selector = NeighborSelector()
    result = selector.apply_selection(scores, policy)

    # Filter subgraph based on selection
    if result.excluded:
        excluded_set = set(result.excluded)
        nodes_to_remove = []

        # Identify nodes to remove (only direct neighbors, not multi-hop)
        for node in subgraph.nodes():
            if node != gene_id and node in excluded_set:
                nodes_to_remove.append(node)

        # Remove excluded nodes
        for node in nodes_to_remove:
            subgraph.remove_node(node)

        logger.debug(
            f"Neighbor selection for {gene_id}: removed {len(nodes_to_remove)} nodes, "
            f"kept {len(result.selected)}"
        )

    # Prepare scoring info
    scoring_info = {
        'mode': selection_mode,
        'filtered': True,
        'original_count': len(neighbors),
        'selected_count': len(result.selected),
        'excluded_count': len(result.excluded),
        'selection_rate': result.stats.get('selection_rate', 1.0),
        'avg_selected_score': result.stats.get('avg_selected_score', 0.5),
        'scores': {n: s.to_dict() for n, s in scores.items()},
    }

    return subgraph, scoring_info


def extract_subgraph(
    gene_id: str,
    strategy: Dict[str, Any],
    kg: nx.DiGraph,
    edge_weights: Dict[str, float] = None
) -> nx.DiGraph:
    """
    Extract a subgraph around a gene based on the strategy.

    Args:
        gene_id: The central gene symbol
        strategy: Strategy dictionary with:
            - edge_types: List of edge types to include (empty = baseline mode)
            - max_hops: Maximum depth of exploration
            - sampling: Sampling method ('top_k', 'random', 'weighted')
            - max_neighbors: Maximum neighbors per edge type
            - neighbors_per_type: Fine-grained neighbor limits per edge type (optional)
        kg: The full knowledge graph
        edge_weights: Dynamic weights for each edge type from Memory (0.1-1.0).
                     Higher weight = more neighbors sampled for that edge type.

    Returns:
        NetworkX DiGraph representing the subgraph
    """
    subgraph = nx.DiGraph()
    edge_weights = edge_weights or {}

    # Add the central gene node
    if gene_id in kg.nodes:
        subgraph.add_node(gene_id, **kg.nodes[gene_id])
    else:
        subgraph.add_node(gene_id, node_label='gene', symbol=gene_id)

    edge_types = strategy.get('edge_types', ['PPI', 'GO', 'HPO'])
    max_hops = strategy.get('max_hops', 1)
    sampling = strategy.get('sampling', 'top_k')
    max_neighbors = strategy.get('max_neighbors', 50)
    neighbors_per_type = strategy.get('neighbors_per_type', {})

    # BASELINE MODE: If edge_types is empty, return only the center node
    # This is used for iteration 0 to measure performance without KG info
    if not edge_types:
        logger.debug(f"Baseline mode for {gene_id}: returning empty subgraph (no edges)")
        return subgraph

    # Track visited nodes to avoid duplicates
    visited = {gene_id}

    # First hop: get direct neighbors
    for edge_type in edge_types:
        # Use per-type neighbor limit if specified, otherwise use global max_neighbors
        effective_max = neighbors_per_type.get(edge_type, max_neighbors)

        # Get dynamic weight for this edge type (default 1.0)
        weight = edge_weights.get(edge_type, 1.0)

        neighbors = get_neighbors(
            kg, gene_id, edge_type,
            max_count=effective_max,
            method=sampling,
            edge_type_weight=weight
        )

        for neighbor_id, edge_data in neighbors:
            # Add neighbor node
            if neighbor_id in kg.nodes:
                subgraph.add_node(neighbor_id, **kg.nodes[neighbor_id])
            else:
                subgraph.add_node(neighbor_id)

            # Add edge
            subgraph.add_edge(gene_id, neighbor_id, edge_type=edge_type, **edge_data)
            visited.add(neighbor_id)

    # Multi-hop expansion if needed
    if max_hops > 1:
        # Use gentler decay factor instead of halving
        HOP_DECAY_FACTOR = 0.7
        decayed_neighbors = max(int(max_neighbors * HOP_DECAY_FACTOR), 10)
        expand_subgraph(
            subgraph, kg, gene_id,
            edge_types=edge_types,
            max_hops=max_hops - 1,
            sampling=sampling,
            max_neighbors=decayed_neighbors,
            visited=visited,
            edge_weights=edge_weights
        )

    logger.debug(f"Extracted subgraph for {gene_id}: {subgraph.number_of_nodes()} nodes, {subgraph.number_of_edges()} edges")
    return subgraph


def expand_subgraph(
    subgraph: nx.DiGraph,
    kg: nx.DiGraph,
    center_gene: str,
    edge_types: List[str],
    max_hops: int,
    sampling: str,
    max_neighbors: int,
    visited: Set[str],
    edge_weights: Dict[str, float] = None
) -> None:
    """
    Expand the subgraph by exploring neighbors of neighbors.

    This function modifies the subgraph in-place.

    Args:
        subgraph: The subgraph to expand
        kg: The full knowledge graph
        center_gene: The original center gene
        edge_types: Edge types to follow
        max_hops: Remaining hops to explore
        sampling: Sampling method
        max_neighbors: Max neighbors per hop
        visited: Set of already visited nodes
        edge_weights: Dynamic weights for each edge type from Memory
    """
    if max_hops <= 0:
        return

    edge_weights = edge_weights or {}

    # Get current frontier (nodes at the edge of the subgraph)
    frontier = [n for n in subgraph.nodes() if n != center_gene and n not in visited]

    # Only expand from gene nodes (not GO terms, phenotypes, etc.)
    gene_frontier = []
    for node in frontier:
        node_data = subgraph.nodes.get(node, {})
        node_label = node_data.get('node_label', '')
        if node_label == 'gene':
            gene_frontier.append(node)

    # Limit frontier size
    gene_frontier = gene_frontier[:max_neighbors]

    for node in gene_frontier:
        visited.add(node)

        for edge_type in edge_types:
            # Edge types that can be expanded in multi-hop
            # PPI, TRRUST, and CORUM are chainable (gene-to-gene relationships)
            # GO/HPO/CellMarker/Reactome/OMIM/GTEx are terminal (gene-to-annotation)
            MULTI_HOP_EDGE_TYPES = {'PPI', 'TRRUST', 'CORUM'}
            if edge_type not in MULTI_HOP_EDGE_TYPES:
                continue

            # Use more conservative neighbor count division
            effective_max = max(max_neighbors // max(len(gene_frontier), 1), 5)

            # Get dynamic weight for this edge type
            weight = edge_weights.get(edge_type, 1.0)

            neighbors = get_neighbors(
                kg, node, edge_type,
                max_count=effective_max,
                method=sampling,
                edge_type_weight=weight
            )

            for neighbor_id, edge_data in neighbors:
                if neighbor_id in visited:
                    continue

                # Add neighbor node
                if neighbor_id in kg.nodes:
                    subgraph.add_node(neighbor_id, **kg.nodes[neighbor_id])
                else:
                    subgraph.add_node(neighbor_id)

                # Add edge
                subgraph.add_edge(node, neighbor_id, edge_type=edge_type, **edge_data)

    # Recurse for more hops with gentler decay
    if max_hops > 1:
        HOP_DECAY_FACTOR = 0.7
        decayed_neighbors = max(int(max_neighbors * HOP_DECAY_FACTOR), 10)
        expand_subgraph(
            subgraph, kg, center_gene,
            edge_types=edge_types,
            max_hops=max_hops - 1,
            sampling=sampling,
            max_neighbors=decayed_neighbors,
            visited=visited,
            edge_weights=edge_weights
        )


def get_subgraph_stats(subgraph: nx.DiGraph) -> Dict[str, Any]:
    """
    Get statistics about a subgraph.

    Args:
        subgraph: The subgraph

    Returns:
        Dictionary with statistics
    """
    # Count nodes by type
    node_types = {}
    for node, data in subgraph.nodes(data=True):
        label = data.get('node_label', 'unknown')
        node_types[label] = node_types.get(label, 0) + 1

    # Count edges by type
    edge_types = {}
    for _, _, data in subgraph.edges(data=True):
        etype = data.get('edge_type', 'unknown')
        edge_types[etype] = edge_types.get(etype, 0) + 1

    return {
        'num_nodes': subgraph.number_of_nodes(),
        'num_edges': subgraph.number_of_edges(),
        'node_types': node_types,
        'edge_types': edge_types,
    }


# =============================================================================
# Batch Processing with Scoring
# =============================================================================

def extract_subgraphs_batch_with_scoring(
    gene_ids: List[str],
    strategy: Dict[str, Any],
    kg: nx.DiGraph,
    edge_weights: Dict[str, float] = None,
    task_genes: Optional[Set[str]] = None,
    memory: Optional[Any] = None,
    embeddings: Optional[Dict[str, np.ndarray]] = None,
    enable_neighbor_selection: bool = False,
) -> Tuple[Dict[str, nx.DiGraph], Dict[str, Dict[str, Any]]]:
    """
    Extract subgraphs for multiple genes with scoring.

    Args:
        gene_ids: List of gene IDs to process
        strategy: Strategy dictionary
        kg: Knowledge graph
        edge_weights: Dynamic edge weights
        task_genes: Task-relevant genes
        memory: Memory instance
        embeddings: Embeddings dictionary
        enable_neighbor_selection: Whether to enable selection

    Returns:
        Tuple of (subgraphs_dict, scoring_info_dict)
    """
    subgraphs = {}
    scoring_info = {}

    for gene_id in gene_ids:
        subgraph, info = extract_subgraph_with_scoring(
            gene_id=gene_id,
            strategy=strategy,
            kg=kg,
            edge_weights=edge_weights,
            task_genes=task_genes,
            memory=memory,
            embeddings=embeddings,
            enable_neighbor_selection=enable_neighbor_selection,
        )
        subgraphs[gene_id] = subgraph
        if info:
            scoring_info[gene_id] = info

    return subgraphs, scoring_info


def get_neighbor_analysis_for_actor(
    gene_ids: List[str],
    strategy: Dict[str, Any],
    kg: nx.DiGraph,
    task_genes: Optional[Set[str]] = None,
    memory: Optional[Any] = None,
    embeddings: Optional[Dict[str, np.ndarray]] = None,
    presentation_mode: str = 'summary',
) -> str:
    """
    Get formatted neighbor analysis for Actor prompt.

    This is a convenience function that:
    1. Scores neighbors for given genes
    2. Formats the results for Actor consumption

    Args:
        gene_ids: List of gene IDs to analyze
        strategy: Current strategy
        kg: Knowledge graph
        task_genes: Task-relevant genes
        memory: Memory instance
        embeddings: Embeddings dictionary
        presentation_mode: How to format output ('summary', 'detailed', 'minimal')

    Returns:
        Formatted text for inclusion in Actor prompt
    """
    try:
        from .neighbor_scorer import NeighborScorer
        from .neighbor_presenter import NeighborPresenter, format_neighbor_analysis_section
    except ImportError:
        logger.warning("Neighbor analysis modules not available")
        return ""

    scorer = NeighborScorer(kg)
    all_scores = {}

    edge_types = strategy.get('edge_types', ['PPI', 'GO', 'HPO'])

    for gene_id in gene_ids:
        scores = scorer.score_all_neighbors_for_gene(
            gene_id=gene_id,
            edge_types=edge_types,
            task_genes=task_genes,
            memory=memory,
            embeddings=embeddings,
        )
        if scores:
            all_scores[gene_id] = scores

    if not all_scores:
        return "No neighbor analysis available."

    return format_neighbor_analysis_section(
        all_scores=all_scores,
        mode=presentation_mode,
        include_guidance=True,
    )
