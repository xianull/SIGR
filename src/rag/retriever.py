"""
RAG Retriever for KG-RAG

Handles query formulation and semantic retrieval of
relevant KG triplets for a given gene.
"""

import logging
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass

import numpy as np
import networkx as nx

from .index_manager import IndexManager, TripletIndex

logger = logging.getLogger(__name__)


@dataclass
class RetrievedTriplet:
    """A single retrieved triplet with score."""
    triplet_id: str
    text: str
    edge_type: str
    source_node: str
    target_node: str
    score: float
    metadata: Dict[str, Any]

    def __repr__(self):
        return f"RetrievedTriplet({self.edge_type}: {self.source_node} -> {self.target_node}, score={self.score:.3f})"


# Task-specific context for query augmentation
TASK_CONTEXTS = {
    'geneattribute_dosage_sensitivity': (
        "Identify biological features relevant to dosage sensitivity prediction, "
        "including haploinsufficiency, network centrality, phenotype associations, "
        "and regulatory complexity."
    ),
    'ppi': (
        "Identify features relevant to protein-protein interaction prediction, "
        "including functional similarity, pathway co-membership, and network topology."
    ),
    'cell': (
        "Identify features relevant to cell type classification, "
        "including marker gene expression, tissue specificity, and functional annotations."
    ),
    'perturbation': (
        "Identify features relevant to gene perturbation response prediction, "
        "including regulatory relationships, pathway membership, and expression patterns."
    ),
}


class RAGRetriever:
    """
    Semantic retriever for KG triplets.

    Given a query gene, retrieves the most semantically relevant
    triplets from the pre-built index.
    """

    def __init__(
        self,
        index: TripletIndex,
        encoder,
        kg: Optional[nx.DiGraph] = None,
        task_name: Optional[str] = None,
    ):
        """
        Initialize the retriever.

        Args:
            index: Pre-built TripletIndex
            encoder: GeneEncoder for query encoding
            kg: Knowledge graph (for query formulation)
            task_name: Task name for context augmentation
        """
        self.index = index
        self.encoder = encoder
        self.kg = kg
        self.task_name = task_name
        self.task_context = TASK_CONTEXTS.get(task_name, '')

        # Initialize index manager for search
        self._index_manager = IndexManager()
        self._index_manager._faiss_index = None

        # Build gene-to-triplet mapping for filtering
        self._gene_triplets: Dict[str, Set[str]] = {}
        self._build_gene_triplet_mapping()

    def _build_gene_triplet_mapping(self):
        """Build mapping from genes to their triplet IDs."""
        for triplet_id, metadata in self.index.triplet_metadata.items():
            source = metadata.get('source_node', '')
            target = metadata.get('target_node', '')

            if source:
                if source not in self._gene_triplets:
                    self._gene_triplets[source] = set()
                self._gene_triplets[source].add(triplet_id)

            if target:
                if target not in self._gene_triplets:
                    self._gene_triplets[target] = set()
                self._gene_triplets[target].add(triplet_id)

    def _get_gene_info(self, gene_id: str) -> Dict[str, Any]:
        """Get gene information from KG."""
        if self.kg is None or gene_id not in self.kg.nodes:
            return {'symbol': gene_id, 'name': gene_id}

        data = self.kg.nodes[gene_id]
        return {
            'symbol': gene_id,
            'name': data.get('name', gene_id),
            'ncbi_summary': data.get('ncbi_summary', ''),
            'description': data.get('description', ''),
        }

    def formulate_query(
        self,
        gene_id: str,
        include_task_context: bool = True,
        include_gene_summary: bool = True,
    ) -> str:
        """
        Formulate a query for the given gene.

        Args:
            gene_id: Gene symbol
            include_task_context: Whether to include task-specific context
            include_gene_summary: Whether to include gene summary

        Returns:
            Query string for embedding
        """
        gene_info = self._get_gene_info(gene_id)

        # Build query parts
        parts = [f"Gene {gene_info['symbol']}"]

        if gene_info.get('name') and gene_info['name'] != gene_id:
            parts[0] += f" ({gene_info['name']})"

        # Add gene summary if available
        if include_gene_summary:
            summary = gene_info.get('ncbi_summary') or gene_info.get('description', '')
            if summary:
                # Take first sentence
                first_sentence = summary.split('.')[0].strip()
                if len(first_sentence) > 20:
                    parts.append(first_sentence + '.')

        # Add task context
        if include_task_context and self.task_context:
            parts.append(f"Task: {self.task_context}")

        query = ' '.join(parts)
        return query

    def retrieve(
        self,
        gene_id: str,
        top_k: int = 50,
        filter_connected: bool = False,
        include_edge_types: Optional[List[str]] = None,
        exclude_edge_types: Optional[List[str]] = None,
        min_score: float = 0.0,
    ) -> List[RetrievedTriplet]:
        """
        Retrieve relevant triplets for a gene.

        Args:
            gene_id: Gene symbol
            top_k: Number of triplets to retrieve
            filter_connected: Only return triplets connected to the gene
            include_edge_types: Only include these edge types
            exclude_edge_types: Exclude these edge types
            min_score: Minimum similarity score threshold

        Returns:
            List of RetrievedTriplet sorted by relevance
        """
        # 1. Formulate query
        query = self.formulate_query(gene_id)

        # 2. Encode query
        query_embedding = self.encoder.encode_batch([query])[0]

        # 3. Search index (retrieve more for filtering)
        retrieve_count = top_k * 3 if filter_connected else top_k
        results = self._index_manager.search(
            query_embedding,
            self.index,
            top_k=retrieve_count
        )

        # 4. Filter and convert to RetrievedTriplet
        triplets = []
        gene_triplet_ids = self._gene_triplets.get(gene_id, set())

        for result in results:
            triplet_id = result['triplet_id']
            metadata = result.get('metadata', {})
            score = result.get('score', 0.0)

            # Score filter
            if score < min_score:
                continue

            # Edge type filter
            edge_type = metadata.get('edge_type', 'unknown')
            if include_edge_types and edge_type not in include_edge_types:
                continue
            if exclude_edge_types and edge_type in exclude_edge_types:
                continue

            # Connected filter
            if filter_connected and triplet_id not in gene_triplet_ids:
                continue

            triplet = RetrievedTriplet(
                triplet_id=triplet_id,
                text=result.get('text', ''),
                edge_type=edge_type,
                source_node=metadata.get('source_node', ''),
                target_node=metadata.get('target_node', ''),
                score=score,
                metadata=metadata,
            )
            triplets.append(triplet)

            if len(triplets) >= top_k:
                break

        logger.debug(f"Retrieved {len(triplets)} triplets for {gene_id}")
        return triplets

    def retrieve_batch(
        self,
        gene_ids: List[str],
        top_k: int = 50,
        **kwargs
    ) -> Dict[str, List[RetrievedTriplet]]:
        """
        Batch retrieval for multiple genes.

        Args:
            gene_ids: List of gene symbols
            top_k: Number of triplets per gene
            **kwargs: Additional arguments passed to retrieve()

        Returns:
            Dict mapping gene_id to list of RetrievedTriplet
        """
        results = {}
        for gene_id in gene_ids:
            results[gene_id] = self.retrieve(gene_id, top_k=top_k, **kwargs)
        return results

    def retrieve_with_diversity(
        self,
        gene_id: str,
        top_k: int = 50,
        diversity_weight: float = 0.3,
        **kwargs
    ) -> List[RetrievedTriplet]:
        """
        Retrieve with edge type diversity.

        Ensures a balanced mix of different edge types
        rather than being dominated by one type.

        Args:
            gene_id: Gene symbol
            top_k: Total number of triplets
            diversity_weight: Weight for diversity vs. pure relevance
            **kwargs: Additional arguments

        Returns:
            List of RetrievedTriplet with edge type diversity
        """
        # Get more candidates
        candidates = self.retrieve(gene_id, top_k=top_k * 3, **kwargs)

        if not candidates:
            return []

        # Group by edge type
        by_type: Dict[str, List[RetrievedTriplet]] = {}
        for t in candidates:
            if t.edge_type not in by_type:
                by_type[t.edge_type] = []
            by_type[t.edge_type].append(t)

        # Interleave selection for diversity
        selected = []
        type_indices = {et: 0 for et in by_type}
        edge_types = list(by_type.keys())

        while len(selected) < top_k and edge_types:
            # Round-robin through edge types
            for et in edge_types[:]:
                if type_indices[et] < len(by_type[et]):
                    selected.append(by_type[et][type_indices[et]])
                    type_indices[et] += 1
                    if len(selected) >= top_k:
                        break
                else:
                    edge_types.remove(et)

        # Sort final selection by score
        selected.sort(key=lambda t: t.score, reverse=True)

        return selected[:top_k]


class HybridRetriever(RAGRetriever):
    """
    Hybrid retriever combining semantic and structural scores.

    Balances semantic relevance (from embeddings) with
    structural relevance (graph distance, edge confidence).
    """

    def __init__(
        self,
        index: TripletIndex,
        encoder,
        kg: nx.DiGraph,
        task_name: Optional[str] = None,
        semantic_weight: float = 0.7,
    ):
        """
        Initialize hybrid retriever.

        Args:
            index: Pre-built TripletIndex
            encoder: GeneEncoder
            kg: Knowledge graph (required for structural scoring)
            task_name: Task name
            semantic_weight: Weight for semantic vs. structural (0-1)
        """
        super().__init__(index, encoder, kg, task_name)
        self.semantic_weight = semantic_weight
        self.structural_weight = 1.0 - semantic_weight

    def _compute_structural_score(
        self,
        gene_id: str,
        triplet: RetrievedTriplet,
    ) -> float:
        """
        Compute structural relevance score.

        Based on:
        - Graph distance (is gene directly connected?)
        - Edge confidence (for PPI: combined_score)
        - Edge type importance for task
        """
        score = 0.0

        # 1. Connected bonus
        if triplet.source_node == gene_id or triplet.target_node == gene_id:
            score += 0.5

        # 2. Edge confidence
        if triplet.edge_type == 'PPI':
            confidence = triplet.metadata.get('combined_score', 0)
            if isinstance(confidence, (int, float)):
                if confidence > 1:
                    confidence = confidence / 1000
                score += 0.3 * confidence

        # 3. Edge type importance (task-specific)
        edge_importance = {
            'geneattribute_dosage_sensitivity': {
                'HPO': 0.3, 'TRRUST': 0.25, 'GO': 0.2, 'PPI': 0.15, 'Reactome': 0.1
            },
        }
        task_weights = edge_importance.get(self.task_name, {})
        score += task_weights.get(triplet.edge_type, 0.1)

        return min(score, 1.0)

    def retrieve(
        self,
        gene_id: str,
        top_k: int = 50,
        **kwargs
    ) -> List[RetrievedTriplet]:
        """
        Retrieve with hybrid scoring.
        """
        # Get semantic candidates
        candidates = super().retrieve(gene_id, top_k=top_k * 2, **kwargs)

        if not candidates:
            return []

        # Compute hybrid scores
        for triplet in candidates:
            semantic_score = triplet.score
            structural_score = self._compute_structural_score(gene_id, triplet)
            triplet.score = (
                self.semantic_weight * semantic_score +
                self.structural_weight * structural_score
            )

        # Re-sort by hybrid score
        candidates.sort(key=lambda t: t.score, reverse=True)

        return candidates[:top_k]
