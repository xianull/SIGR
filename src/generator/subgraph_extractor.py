"""
Subgraph Extraction for SIGR Framework

Extracts subgraphs from the knowledge graph based on Actor's strategy.
"""

import logging
from typing import Dict, List, Tuple, Any, Set

import networkx as nx

from ..utils.kg_utils import get_neighbors, EDGE_LABEL_MAPPING

logger = logging.getLogger(__name__)


def extract_subgraph(
    gene_id: str,
    strategy: Dict[str, Any],
    kg: nx.DiGraph
) -> nx.DiGraph:
    """
    Extract a subgraph around a gene based on the strategy.

    Args:
        gene_id: The central gene symbol
        strategy: Strategy dictionary with:
            - edge_types: List of edge types to include
            - max_hops: Maximum depth of exploration
            - sampling: Sampling method ('top_k', 'random', 'weighted')
            - max_neighbors: Maximum neighbors per edge type
        kg: The full knowledge graph

    Returns:
        NetworkX DiGraph representing the subgraph
    """
    subgraph = nx.DiGraph()

    # Add the central gene node
    if gene_id in kg.nodes:
        subgraph.add_node(gene_id, **kg.nodes[gene_id])
    else:
        subgraph.add_node(gene_id, node_label='gene', symbol=gene_id)

    edge_types = strategy.get('edge_types', ['PPI', 'GO', 'HPO'])
    max_hops = strategy.get('max_hops', 1)
    sampling = strategy.get('sampling', 'top_k')
    max_neighbors = strategy.get('max_neighbors', 50)

    # Track visited nodes to avoid duplicates
    visited = {gene_id}

    # First hop: get direct neighbors
    for edge_type in edge_types:
        neighbors = get_neighbors(
            kg, gene_id, edge_type,
            max_count=max_neighbors,
            method=sampling
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
        expand_subgraph(
            subgraph, kg, gene_id,
            edge_types=edge_types,
            max_hops=max_hops - 1,
            sampling=sampling,
            max_neighbors=max_neighbors // 2,  # Reduce neighbors for deeper hops
            visited=visited
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
    visited: Set[str]
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
    """
    if max_hops <= 0:
        return

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
            # Only expand PPI edges in multi-hop (GO/HPO don't chain well)
            if edge_type != 'PPI':
                continue

            neighbors = get_neighbors(
                kg, node, edge_type,
                max_count=max_neighbors // len(gene_frontier) if gene_frontier else max_neighbors,
                method=sampling
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

    # Recurse for more hops
    if max_hops > 1:
        expand_subgraph(
            subgraph, kg, center_gene,
            edge_types=edge_types,
            max_hops=max_hops - 1,
            sampling=sampling,
            max_neighbors=max_neighbors // 2,
            visited=visited
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
