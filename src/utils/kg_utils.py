"""
Knowledge Graph Utilities for SIGR Framework

Provides functions for loading and querying the SIGR knowledge graph.
"""

import pickle
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional, Any
from collections import defaultdict

import networkx as nx

logger = logging.getLogger(__name__)

# Edge type mapping from KG labels to simplified names
EDGE_TYPE_MAPPING = {
    'pairwise molecular interaction': 'PPI',
    'participates in': 'GO',
    'gene to phenotypic feature association': 'HPO',
    'regulates': 'TRRUST',
    'expressed in': 'CellMarker',
    'in pathway': 'Reactome',
}

# Reverse mapping
EDGE_LABEL_MAPPING = {v: k for k, v in EDGE_TYPE_MAPPING.items()}


def load_kg(kg_path: str) -> nx.DiGraph:
    """
    Load the knowledge graph from a pickle file.

    Args:
        kg_path: Path to the pickle file containing the KG

    Returns:
        NetworkX DiGraph
    """
    kg_path = Path(kg_path)
    if not kg_path.exists():
        raise FileNotFoundError(f"Knowledge graph not found: {kg_path}")

    logger.info(f"Loading knowledge graph from {kg_path}")
    with open(kg_path, 'rb') as f:
        kg = pickle.load(f)

    logger.info(f"Loaded KG with {kg.number_of_nodes()} nodes and {kg.number_of_edges()} edges")
    return kg


def get_all_genes(kg: nx.DiGraph) -> Set[str]:
    """
    Get all gene nodes from the knowledge graph.

    Args:
        kg: NetworkX DiGraph

    Returns:
        Set of gene IDs (symbols)
    """
    genes = set()
    for node, data in kg.nodes(data=True):
        label = data.get('node_label') or data.get('label', '')
        if label == 'gene':
            genes.add(node)
    return genes


def get_gene_info(kg: nx.DiGraph, gene_id: str) -> Dict[str, Any]:
    """
    Get information about a gene from the KG.

    Args:
        kg: NetworkX DiGraph
        gene_id: Gene symbol

    Returns:
        Dictionary with gene information
    """
    if gene_id not in kg.nodes:
        return {'symbol': gene_id, 'name': gene_id, 'found': False}

    data = kg.nodes[gene_id]
    return {
        'symbol': gene_id,
        'name': data.get('name', gene_id),
        'hgnc_id': data.get('hgnc_id', ''),
        'ensembl_id': data.get('ensembl_id', ''),
        'entrez_id': data.get('entrez_id', ''),
        'chromosome': data.get('chromosome', ''),
        'gene_type': data.get('gene_type', ''),
        'description': data.get('description', ''),
        'found': True
    }


def get_neighbors(
    kg: nx.DiGraph,
    gene_id: str,
    edge_type: str,
    max_count: int = 50,
    method: str = 'top_k',
    edge_type_weight: float = 1.0
) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Get neighbors of a gene by edge type.

    Args:
        kg: NetworkX DiGraph
        gene_id: Gene symbol
        edge_type: Type of edge ('PPI', 'GO', 'HPO', 'TRRUST', 'CellMarker', 'Reactome')
        max_count: Maximum number of neighbors to return
        method: Sampling method ('top_k', 'random', 'weighted')
        edge_type_weight: Dynamic weight for this edge type (0.1-1.0).
                         Higher weight means more neighbors will be returned.

    Returns:
        List of (neighbor_id, edge_data) tuples
    """
    if gene_id not in kg.nodes:
        return []

    # Get the edge label for this type
    edge_label = EDGE_LABEL_MAPPING.get(edge_type, edge_type)

    neighbors = []

    # Check outgoing edges
    for _, target, data in kg.out_edges(gene_id, data=True):
        label = data.get('relationship_label') or data.get('label', '')
        if label == edge_label:
            neighbors.append((target, data))

    # For PPI, also check incoming edges (interactions are bidirectional)
    if edge_type == 'PPI':
        for source, _, data in kg.in_edges(gene_id, data=True):
            label = data.get('relationship_label') or data.get('label', '')
            if label == edge_label:
                neighbors.append((source, data))

    # Apply sampling method with edge_type_weight
    # Higher weight = more neighbors returned
    effective_max = max(5, int(max_count * edge_type_weight))

    if method == 'top_k':
        # Sort by score if available (for PPI)
        if edge_type == 'PPI':
            neighbors.sort(key=lambda x: x[1].get('combined_score', 0), reverse=True)
        neighbors = neighbors[:effective_max]

    elif method == 'random':
        import random
        random.shuffle(neighbors)
        neighbors = neighbors[:effective_max]

    elif method == 'weighted':
        # Weight by score for PPI, otherwise use edge_type_weight as base
        import random
        if neighbors:
            if edge_type == 'PPI':
                # For PPI: combine combined_score with edge_type_weight
                weights = [
                    n[1].get('combined_score', 0.5) * edge_type_weight
                    for n in neighbors
                ]
            else:
                # For non-PPI: use edge_type_weight as uniform weight
                weights = [edge_type_weight] * len(neighbors)

            total = sum(weights)
            if total > 0:
                weights = [w / total for w in weights]
                indices = random.choices(
                    range(len(neighbors)),
                    weights=weights,
                    k=min(effective_max, len(neighbors))
                )
                neighbors = [neighbors[i] for i in indices]
            else:
                random.shuffle(neighbors)
                neighbors = neighbors[:effective_max]
        else:
            neighbors = []

    return neighbors


def get_kg_summary(kg: nx.DiGraph, gene_id: str) -> str:
    """
    Get a summary of a gene's KG neighborhood.

    Args:
        kg: NetworkX DiGraph
        gene_id: Gene symbol

    Returns:
        String summary of the gene's KG context
    """
    if gene_id not in kg.nodes:
        return f"Gene {gene_id} not found in knowledge graph."

    # Count neighbors by edge type
    edge_counts = defaultdict(int)

    for _, target, data in kg.out_edges(gene_id, data=True):
        label = data.get('relationship_label') or data.get('label', '')
        edge_type = EDGE_TYPE_MAPPING.get(label, label)
        edge_counts[edge_type] += 1

    for source, _, data in kg.in_edges(gene_id, data=True):
        label = data.get('relationship_label') or data.get('label', '')
        if label == 'pairwise molecular interaction':  # Count incoming PPI edges
            edge_counts['PPI'] += 1

    # Build summary
    gene_info = get_gene_info(kg, gene_id)
    summary_parts = [
        f"Gene: {gene_id} ({gene_info.get('name', 'N/A')})",
        f"Chromosome: {gene_info.get('chromosome', 'N/A')}",
        "KG Neighbors:"
    ]

    for edge_type in ['PPI', 'GO', 'HPO', 'TRRUST', 'CellMarker', 'Reactome']:
        count = edge_counts.get(edge_type, 0)
        if count > 0:
            summary_parts.append(f"  - {edge_type}: {count}")

    total = sum(edge_counts.values())
    summary_parts.append(f"Total connections: {total}")

    return "\n".join(summary_parts)


def get_edge_types_for_gene(kg: nx.DiGraph, gene_id: str) -> Set[str]:
    """
    Get all edge types available for a gene.

    Args:
        kg: NetworkX DiGraph
        gene_id: Gene symbol

    Returns:
        Set of edge type names
    """
    edge_types = set()

    for _, target, data in kg.out_edges(gene_id, data=True):
        label = data.get('relationship_label') or data.get('label', '')
        edge_type = EDGE_TYPE_MAPPING.get(label, label)
        if edge_type:
            edge_types.add(edge_type)

    for source, _, data in kg.in_edges(gene_id, data=True):
        label = data.get('relationship_label') or data.get('label', '')
        if label == 'pairwise molecular interaction':
            edge_types.add('PPI')

    return edge_types
