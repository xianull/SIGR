"""
Triplet Textualizer for KG-RAG

Converts KG triplets (edges) into natural language descriptions
suitable for embedding and semantic retrieval.
"""

import logging
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass

import networkx as nx

logger = logging.getLogger(__name__)


# Edge type to relationship label mapping (from kg_utils.py)
EDGE_LABEL_MAPPING = {
    'PPI': 'pairwise molecular interaction',
    'GO': 'participates in',
    'HPO': 'gene to phenotypic feature association',
    'TRRUST': 'regulates',
    'CellMarker': 'expressed in',
    'Reactome': 'in pathway',
    'OMIM': 'gene to disease association',
    'GTEx': 'expressed in tissue',
    'CORUM': 'part of complex',
}

# Reverse mapping
LABEL_TO_EDGE_TYPE = {v: k for k, v in EDGE_LABEL_MAPPING.items()}


@dataclass
class Triplet:
    """Represents a KG triplet (edge)."""
    triplet_id: str
    source_node: str
    target_node: str
    edge_type: str
    text: str
    metadata: Dict[str, Any]


class TripletTextualizer:
    """
    Converts KG triplets to natural language text.

    Each edge in the KG is converted to a human-readable sentence
    that captures the relationship and relevant properties.
    """

    def __init__(self, kg: nx.DiGraph):
        """
        Initialize the textualizer.

        Args:
            kg: NetworkX DiGraph containing the knowledge graph
        """
        self.kg = kg
        self._node_cache: Dict[str, Dict] = {}

    def _get_node_info(self, node_id: str) -> Dict[str, Any]:
        """Get node information with caching."""
        if node_id not in self._node_cache:
            if node_id in self.kg.nodes:
                self._node_cache[node_id] = dict(self.kg.nodes[node_id])
            else:
                self._node_cache[node_id] = {}
        return self._node_cache[node_id]

    def _get_node_name(self, node_id: str) -> str:
        """Get display name for a node."""
        info = self._get_node_info(node_id)
        # Try different name fields
        name = info.get('name') or info.get('symbol') or node_id
        return str(name)

    def _get_node_description(self, node_id: str) -> Optional[str]:
        """Get description for a node if available."""
        info = self._get_node_info(node_id)
        # Try different description fields
        desc = info.get('definition') or info.get('ncbi_summary') or info.get('description')
        if desc and len(str(desc)) > 10:
            # Truncate long descriptions
            desc = str(desc)[:200]
            if len(desc) == 200:
                desc = desc.rsplit(' ', 1)[0] + '...'
        return desc

    def _get_node_label(self, node_id: str) -> str:
        """Get the node type label."""
        info = self._get_node_info(node_id)
        return info.get('node_label', 'unknown')

    def triplet_to_text(
        self,
        source: str,
        target: str,
        edge_data: Dict[str, Any]
    ) -> str:
        """
        Convert a single triplet to natural language text.

        Args:
            source: Source node ID
            target: Target node ID
            edge_data: Edge attributes

        Returns:
            Natural language description of the triplet
        """
        # Determine edge type
        rel_label = edge_data.get('relationship_label') or edge_data.get('label', '')
        edge_type = LABEL_TO_EDGE_TYPE.get(rel_label, 'unknown')

        # Get node information
        source_name = self._get_node_name(source)
        target_name = self._get_node_name(target)
        source_label = self._get_node_label(source)
        target_label = self._get_node_label(target)
        target_desc = self._get_node_description(target)

        # Generate text based on edge type
        if edge_type == 'PPI':
            score = edge_data.get('combined_score', 0)
            if isinstance(score, (int, float)) and score > 1:
                score = score / 1000  # Normalize if raw STRING score
            text = f"{source_name} physically interacts with {target_name} (confidence: {score:.2f})"

        elif edge_type == 'GO':
            # GO term association
            evidence = edge_data.get('evidence_code', 'unknown')
            if target_desc:
                text = f"{source_name} participates in {target_name}: {target_desc} (evidence: {evidence})"
            else:
                text = f"{source_name} participates in biological process {target_name} (evidence: {evidence})"

        elif edge_type == 'HPO':
            # Phenotype association
            frequency = edge_data.get('frequency', '')
            if frequency:
                text = f"{source_name} is associated with phenotype {target_name} (frequency: {frequency})"
            else:
                text = f"{source_name} is associated with phenotype {target_name}"

        elif edge_type == 'TRRUST':
            # Transcription factor regulation
            reg_type = edge_data.get('regulation_type', 'unknown')
            direction = edge_data.get('direction', '')
            if reg_type == 'Activation':
                text = f"{source_name} activates expression of {target_name}"
            elif reg_type == 'Repression':
                text = f"{source_name} represses expression of {target_name}"
            else:
                text = f"{source_name} regulates {target_name}"

        elif edge_type == 'Reactome':
            # Pathway membership
            text = f"{source_name} is involved in pathway {target_name}"

        elif edge_type == 'CellMarker':
            # Cell type marker
            tissue = edge_data.get('tissue_type', '')
            if tissue:
                text = f"{source_name} is a marker for {target_name} in {tissue}"
            else:
                text = f"{source_name} is a marker for cell type {target_name}"

        elif edge_type == 'OMIM':
            # Disease association
            text = f"{source_name} is associated with disease {target_name}"

        elif edge_type == 'GTEx':
            # Tissue expression
            tpm = edge_data.get('median_tpm', 0)
            if tpm:
                text = f"{source_name} is expressed in {target_name} (TPM: {tpm:.1f})"
            else:
                text = f"{source_name} is expressed in tissue {target_name}"

        elif edge_type == 'CORUM':
            # Protein complex membership
            text = f"{source_name} is part of protein complex {target_name}"

        else:
            # Generic fallback
            text = f"{source_name} is related to {target_name} via {rel_label}"

        return text

    def extract_all_triplets(
        self,
        include_edge_types: Optional[List[str]] = None,
        exclude_edge_types: Optional[List[str]] = None,
    ) -> List[Triplet]:
        """
        Extract all triplets from the KG and convert to text.

        Args:
            include_edge_types: Only include these edge types (None = all)
            exclude_edge_types: Exclude these edge types

        Returns:
            List of Triplet objects with text descriptions
        """
        triplets = []
        triplet_counter = 0

        # Default exclusions if not specified
        if exclude_edge_types is None:
            exclude_edge_types = []

        for source, target, edge_data in self.kg.edges(data=True):
            # Determine edge type
            rel_label = edge_data.get('relationship_label') or edge_data.get('label', '')
            edge_type = LABEL_TO_EDGE_TYPE.get(rel_label, 'unknown')

            # Filter by edge type
            if include_edge_types and edge_type not in include_edge_types:
                continue
            if edge_type in exclude_edge_types:
                continue

            # Generate triplet ID
            triplet_id = f"t_{triplet_counter}"
            triplet_counter += 1

            # Convert to text
            text = self.triplet_to_text(source, target, edge_data)

            # Create triplet object
            triplet = Triplet(
                triplet_id=triplet_id,
                source_node=source,
                target_node=target,
                edge_type=edge_type,
                text=text,
                metadata={
                    'relationship_label': rel_label,
                    **{k: v for k, v in edge_data.items() if k != 'relationship_label'}
                }
            )
            triplets.append(triplet)

        logger.info(f"Extracted {len(triplets)} triplets from KG")
        return triplets

    def get_gene_triplets(
        self,
        gene_id: str,
        include_edge_types: Optional[List[str]] = None,
    ) -> List[Triplet]:
        """
        Get all triplets involving a specific gene.

        Args:
            gene_id: Gene symbol
            include_edge_types: Only include these edge types

        Returns:
            List of triplets involving the gene
        """
        triplets = []

        if gene_id not in self.kg.nodes:
            return triplets

        # Check outgoing edges
        for _, target, edge_data in self.kg.out_edges(gene_id, data=True):
            rel_label = edge_data.get('relationship_label') or edge_data.get('label', '')
            edge_type = LABEL_TO_EDGE_TYPE.get(rel_label, 'unknown')

            if include_edge_types and edge_type not in include_edge_types:
                continue

            text = self.triplet_to_text(gene_id, target, edge_data)
            triplet = Triplet(
                triplet_id=f"{gene_id}_out_{target}",
                source_node=gene_id,
                target_node=target,
                edge_type=edge_type,
                text=text,
                metadata=dict(edge_data)
            )
            triplets.append(triplet)

        # Check incoming edges (for bidirectional relationships like PPI)
        for source, _, edge_data in self.kg.in_edges(gene_id, data=True):
            rel_label = edge_data.get('relationship_label') or edge_data.get('label', '')
            edge_type = LABEL_TO_EDGE_TYPE.get(rel_label, 'unknown')

            # Only include PPI for incoming (to avoid duplicates)
            if edge_type != 'PPI':
                continue
            if include_edge_types and edge_type not in include_edge_types:
                continue

            text = self.triplet_to_text(source, gene_id, edge_data)
            triplet = Triplet(
                triplet_id=f"{source}_in_{gene_id}",
                source_node=source,
                target_node=gene_id,
                edge_type=edge_type,
                text=text,
                metadata=dict(edge_data)
            )
            triplets.append(triplet)

        return triplets
