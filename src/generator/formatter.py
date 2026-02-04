"""
Subgraph Formatter for SIGR Framework

Converts extracted subgraphs into structured text for LLM processing.
"""

import logging
from typing import Dict, List, Any

import networkx as nx

logger = logging.getLogger(__name__)


def format_subgraph(subgraph: nx.DiGraph, center_gene: str = None) -> Dict[str, str]:
    """
    Format a subgraph into structured text for each edge type.

    Args:
        subgraph: The extracted subgraph
        center_gene: The central gene (optional, will be detected if not provided)

    Returns:
        Dictionary with formatted text for each category:
        - ppi_info: Protein-protein interactions
        - go_info: Gene Ontology (biological processes)
        - phenotype_info: Phenotype associations
        - tf_info: Transcription factor regulations
        - celltype_info: Cell type markers
        - pathway_info: Reactome pathway information
    """
    # Detect center gene if not provided
    if center_gene is None:
        # Assume the gene with most outgoing edges is the center
        out_degrees = [(n, subgraph.out_degree(n)) for n in subgraph.nodes()]
        if out_degrees:
            center_gene = max(out_degrees, key=lambda x: x[1])[0]

    # Collect edges by type
    ppi_neighbors = []
    go_terms = []
    phenotypes = []
    tf_targets = []
    cell_types = []
    pathways = []

    for source, target, data in subgraph.edges(data=True):
        edge_type = data.get('edge_type', '')

        if edge_type == 'PPI':
            # Get the other gene (could be source or target)
            other_gene = target if source == center_gene else source
            score = data.get('combined_score', 0)
            if isinstance(score, float) and score < 1:
                score = int(score * 1000)  # Convert to 0-1000 scale if needed
            ppi_neighbors.append({
                'gene': other_gene,
                'score': score,
                'source': data.get('source_database', 'STRING')
            })

        elif edge_type == 'GO':
            target_data = subgraph.nodes.get(target, {})
            go_terms.append({
                'id': target,
                'name': target_data.get('name', target),
                'evidence': data.get('evidence_code', 'N/A')
            })

        elif edge_type == 'HPO':
            target_data = subgraph.nodes.get(target, {})
            phenotypes.append({
                'id': target,
                'name': target_data.get('name', target),
                'evidence': data.get('evidence', 'N/A')
            })

        elif edge_type == 'TRRUST':
            other_gene = target if source == center_gene else source
            is_target = source == center_gene  # center_gene regulates other
            tf_targets.append({
                'gene': other_gene,
                'regulation_type': data.get('regulation_type', 'unknown'),
                'direction': 'target' if is_target else 'regulator'
            })

        elif edge_type == 'CellMarker':
            target_data = subgraph.nodes.get(target, {})
            cell_name = target_data.get('name', target)
            if target.startswith('CellMarker:'):
                cell_name = target.replace('CellMarker:', '').replace('_', ' ')
            cell_types.append({
                'cell_type': cell_name,
                'tissue': target_data.get('tissue_origin', data.get('tissue_type', 'N/A'))
            })

        elif edge_type == 'Reactome':
            target_data = subgraph.nodes.get(target, {})
            pathways.append({
                'id': target,
                'name': target_data.get('name', target),
                'evidence': data.get('evidence', 'N/A')
            })

    # Format each category
    result = {
        'ppi_info': _format_ppi(ppi_neighbors),
        'go_info': _format_go(go_terms),
        'phenotype_info': _format_phenotypes(phenotypes),
        'tf_info': _format_tf(tf_targets),
        'celltype_info': _format_celltype(cell_types),
        'pathway_info': _format_pathways(pathways),
    }

    return result


def _format_ppi(neighbors: List[Dict]) -> str:
    """Format PPI interactions."""
    if not neighbors:
        return "No protein-protein interactions found."

    # Sort by score
    neighbors = sorted(neighbors, key=lambda x: x['score'], reverse=True)

    lines = [f"Found {len(neighbors)} protein interaction partners:"]

    # Top interactions
    top_n = min(10, len(neighbors))
    for i, n in enumerate(neighbors[:top_n], 1):
        lines.append(f"  {i}. {n['gene']} (confidence score: {n['score']})")

    if len(neighbors) > top_n:
        lines.append(f"  ... and {len(neighbors) - top_n} more")

    return "\n".join(lines)


def _format_go(terms: List[Dict]) -> str:
    """Format GO biological processes."""
    if not terms:
        return "No biological process annotations found."

    lines = [f"Participates in {len(terms)} biological processes:"]

    for i, term in enumerate(terms[:10], 1):
        name = term.get('name') or term.get('id', 'Unknown')
        if name and name.startswith('GO:'):
            name = term.get('id', name)  # Use ID if name is actually the ID
        lines.append(f"  {i}. {name}")

    if len(terms) > 10:
        lines.append(f"  ... and {len(terms) - 10} more")

    return "\n".join(lines)


def _format_phenotypes(phenotypes: List[Dict]) -> str:
    """Format phenotype associations."""
    if not phenotypes:
        return "No phenotype associations found."

    lines = [f"Associated with {len(phenotypes)} phenotypes:"]

    for i, p in enumerate(phenotypes[:10], 1):
        name = p.get('name') or p.get('id', 'Unknown')
        if name and name.startswith('HP:'):
            name = p.get('id', name)
        lines.append(f"  {i}. {name}")

    if len(phenotypes) > 10:
        lines.append(f"  ... and {len(phenotypes) - 10} more")

    return "\n".join(lines)


def _format_tf(targets: List[Dict]) -> str:
    """Format TF regulation information."""
    if not targets:
        return "No transcription factor regulation information found."

    # Separate targets and regulators
    regulated = [t for t in targets if t['direction'] == 'target']
    regulators = [t for t in targets if t['direction'] == 'regulator']

    lines = []

    if regulated:
        lines.append(f"Regulates {len(regulated)} target genes:")
        for i, t in enumerate(regulated[:5], 1):
            reg_type = t['regulation_type']
            lines.append(f"  {i}. {t['gene']} ({reg_type})")
        if len(regulated) > 5:
            lines.append(f"  ... and {len(regulated) - 5} more")

    if regulators:
        lines.append(f"Regulated by {len(regulators)} transcription factors:")
        for i, t in enumerate(regulators[:5], 1):
            reg_type = t['regulation_type']
            lines.append(f"  {i}. {t['gene']} ({reg_type})")
        if len(regulators) > 5:
            lines.append(f"  ... and {len(regulators) - 5} more")

    return "\n".join(lines) if lines else "No transcription factor regulation information found."


def _format_celltype(cell_types: List[Dict]) -> str:
    """Format cell type marker information."""
    if not cell_types:
        return "No cell type marker information found."

    lines = [f"Marker for {len(cell_types)} cell types:"]

    for i, ct in enumerate(cell_types[:10], 1):
        tissue = ct['tissue']
        tissue_str = f" (tissue: {tissue})" if tissue and tissue != 'N/A' else ""
        lines.append(f"  {i}. {ct['cell_type']}{tissue_str}")

    if len(cell_types) > 10:
        lines.append(f"  ... and {len(cell_types) - 10} more")

    return "\n".join(lines)


def _format_pathways(pathways: List[Dict]) -> str:
    """Format Reactome pathway information."""
    if not pathways:
        return "No pathway information found."

    lines = [f"Participates in {len(pathways)} biological pathways:"]

    for i, p in enumerate(pathways[:10], 1):
        name = p.get('name') or p.get('id', 'Unknown')
        # Clean up pathway ID prefix if name is actually ID
        if name and name.startswith('R-HSA-'):
            name = p.get('id', name)
        lines.append(f"  {i}. {name}")

    if len(pathways) > 10:
        lines.append(f"  ... and {len(pathways) - 10} more")

    return "\n".join(lines)
