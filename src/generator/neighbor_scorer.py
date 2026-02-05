"""
Neighbor Relevance Scoring for SIGR Framework

Multi-dimensional scoring of KG neighbors to help Actor identify
useful context vs noise. Scores are used for:
1. Informing Actor about neighbor quality
2. Filtering/selecting neighbors based on policies
3. Learning from historical effectiveness

Scoring dimensions:
- Structural: Graph topology features (degree, centrality)
- Task relevance: Whether neighbor is task-related
- Semantic: Embedding similarity (if available)
- Memory: Historical effectiveness from Memory system
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Set, Optional, Any

import numpy as np
import networkx as nx


logger = logging.getLogger(__name__)


@dataclass
class NeighborScore:
    """
    Multi-dimensional score for a neighbor node.

    Attributes:
        neighbor_id: The neighbor node identifier
        edge_type: Type of edge connecting to center
        direction: 'out' (center -> neighbor) or 'in' (neighbor -> center)
        structural_score: Score based on graph structure [0, 1]
        task_relevance_score: Score based on task relevance [0, 1]
        semantic_score: Score based on embedding similarity [0, 1]
        memory_score: Score based on historical effectiveness [0, 1]
        total_score: Weighted combination of all scores [0, 1]
        confidence: Confidence in the score (based on available info) [0, 1]
    """
    neighbor_id: str
    edge_type: str
    direction: str = 'out'
    structural_score: float = 0.5
    task_relevance_score: float = 0.5
    semantic_score: float = 0.5
    memory_score: float = 0.5
    total_score: float = 0.5
    confidence: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'neighbor_id': self.neighbor_id,
            'edge_type': self.edge_type,
            'direction': self.direction,
            'structural_score': self.structural_score,
            'task_relevance_score': self.task_relevance_score,
            'semantic_score': self.semantic_score,
            'memory_score': self.memory_score,
            'total_score': self.total_score,
            'confidence': self.confidence,
        }


class NeighborScorer:
    """
    Multi-dimensional neighbor relevance scorer.

    Evaluates neighbors based on:
    1. Structural features (graph topology)
    2. Task relevance (overlap with task genes)
    3. Semantic similarity (embedding distance)
    4. Historical effectiveness (from Memory)
    """

    # Scoring dimension weights
    DIMENSION_WEIGHTS = {
        'structural': 0.25,
        'task_relevance': 0.30,
        'semantic': 0.25,
        'memory': 0.20,
    }

    # Structural scoring parameters
    MAX_DEGREE_NORMALIZATION = 500  # Degrees above this are capped at 1.0

    def __init__(
        self,
        kg: nx.DiGraph,
        dimension_weights: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize the neighbor scorer.

        Args:
            kg: Knowledge graph
            dimension_weights: Optional custom weights for scoring dimensions
        """
        self.kg = kg
        self.weights = dimension_weights or self.DIMENSION_WEIGHTS

        # Pre-compute graph statistics for structural scoring
        self._precompute_graph_stats()

        logger.info(f"NeighborScorer initialized with weights: {self.weights}")

    def _precompute_graph_stats(self):
        """Pre-compute graph statistics for efficient scoring."""
        # Degree statistics
        self._degree_cache: Dict[str, int] = {}
        self._max_degree = 1

        for node in self.kg.nodes():
            degree = self.kg.degree(node)
            self._degree_cache[node] = degree
            if degree > self._max_degree:
                self._max_degree = degree

        logger.debug(f"Pre-computed stats: max_degree={self._max_degree}")

    def score_neighbors(
        self,
        center_gene: str,
        neighbors: List[Tuple[str, str, str]],  # (neighbor_id, edge_type, direction)
        task_genes: Optional[Set[str]] = None,
        memory: Optional[Any] = None,  # Memory instance
        embeddings: Optional[Dict[str, np.ndarray]] = None,
        center_embedding: Optional[np.ndarray] = None,
    ) -> Dict[str, NeighborScore]:
        """
        Score a list of neighbors for a center gene.

        Args:
            center_gene: The center gene ID
            neighbors: List of (neighbor_id, edge_type, direction) tuples
            task_genes: Set of genes relevant to the current task
            memory: Memory instance for historical effectiveness
            embeddings: Dictionary mapping gene_id to embedding vector
            center_embedding: Pre-computed embedding for center gene

        Returns:
            Dictionary mapping neighbor_id to NeighborScore
        """
        if not neighbors:
            return {}

        task_genes = task_genes or set()
        scores: Dict[str, NeighborScore] = {}

        # Get center embedding if not provided
        if center_embedding is None and embeddings and center_gene in embeddings:
            center_embedding = embeddings[center_gene]

        for neighbor_id, edge_type, direction in neighbors:
            # 1. Structural score
            structural = self._compute_structural_score(
                center_gene, neighbor_id, edge_type
            )

            # 2. Task relevance score
            task_rel = self._compute_task_relevance_score(
                neighbor_id, task_genes
            )

            # 3. Semantic score
            semantic = self._compute_semantic_score(
                center_embedding, neighbor_id, embeddings
            )

            # 4. Memory score
            mem_score = self._compute_memory_score(
                neighbor_id, edge_type, memory
            )

            # Compute weighted total
            total = (
                self.weights['structural'] * structural +
                self.weights['task_relevance'] * task_rel +
                self.weights['semantic'] * semantic +
                self.weights['memory'] * mem_score
            )

            # Compute confidence based on available information
            confidence = self._compute_confidence(
                has_task_genes=bool(task_genes),
                has_embedding=(center_embedding is not None and
                              embeddings is not None and
                              neighbor_id in embeddings),
                has_memory=(memory is not None),
            )

            scores[neighbor_id] = NeighborScore(
                neighbor_id=neighbor_id,
                edge_type=edge_type,
                direction=direction,
                structural_score=structural,
                task_relevance_score=task_rel,
                semantic_score=semantic,
                memory_score=mem_score,
                total_score=total,
                confidence=confidence,
            )

        logger.debug(
            f"Scored {len(scores)} neighbors for {center_gene}: "
            f"avg_score={np.mean([s.total_score for s in scores.values()]):.3f}"
        )

        return scores

    def _compute_structural_score(
        self,
        center_gene: str,
        neighbor_id: str,
        edge_type: str,
    ) -> float:
        """
        Compute structural score based on graph topology.

        Components:
        - Neighbor degree (higher degree = more connected = potentially hub)
        - Edge type importance (some edge types are more informative)
        - Local clustering coefficient

        Args:
            center_gene: Center gene ID
            neighbor_id: Neighbor gene ID
            edge_type: Type of connecting edge

        Returns:
            Structural score in [0, 1]
        """
        score = 0.5  # Default

        # 1. Degree-based score (normalized)
        neighbor_degree = self._degree_cache.get(neighbor_id, 0)
        # Use log-normalization to handle high-degree hubs
        degree_score = min(
            np.log1p(neighbor_degree) / np.log1p(self.MAX_DEGREE_NORMALIZATION),
            1.0
        )

        # 2. Edge type importance (some edges are more reliable)
        edge_importance = {
            'PPI': 0.7,      # Protein interactions - moderate importance
            'GO': 0.8,       # Gene Ontology - high importance
            'HPO': 0.8,      # Phenotype - high importance
            'TRRUST': 0.75,  # Regulatory - moderate-high
            'Reactome': 0.8, # Pathway - high importance
            'CellMarker': 0.7,  # Cell type - moderate
        }
        edge_score = edge_importance.get(edge_type, 0.5)

        # 3. Local connectivity (does neighbor also connect to other neighbors?)
        # This is expensive, so we use a simplified heuristic
        local_score = 0.5  # Placeholder - could be computed if needed

        # Weighted combination
        score = 0.4 * degree_score + 0.4 * edge_score + 0.2 * local_score

        return float(np.clip(score, 0.0, 1.0))

    def _compute_task_relevance_score(
        self,
        neighbor_id: str,
        task_genes: Set[str],
    ) -> float:
        """
        Compute task relevance score.

        A neighbor is more relevant if:
        - It is in the task gene set
        - It shares annotations with task genes (future extension)

        Args:
            neighbor_id: Neighbor gene ID
            task_genes: Set of task-relevant genes

        Returns:
            Task relevance score in [0, 1]
        """
        if not task_genes:
            return 0.5  # Neutral if no task genes provided

        # Direct membership in task genes
        if neighbor_id in task_genes:
            return 1.0

        # Check if neighbor is in KG and has same node type as task genes
        # This gives partial credit for being in the same "space"
        if neighbor_id in self.kg.nodes:
            node_data = self.kg.nodes[neighbor_id]
            node_type = node_data.get('node_label', '')
            if node_type == 'gene':
                # Gene but not in task set - moderate relevance
                return 0.4

        # Not directly relevant
        return 0.2

    def _compute_semantic_score(
        self,
        center_embedding: Optional[np.ndarray],
        neighbor_id: str,
        embeddings: Optional[Dict[str, np.ndarray]],
    ) -> float:
        """
        Compute semantic similarity score using embeddings.

        Args:
            center_embedding: Embedding of center gene
            neighbor_id: Neighbor gene ID
            embeddings: Dictionary of all embeddings

        Returns:
            Semantic similarity score in [0, 1]
        """
        if center_embedding is None or embeddings is None:
            return 0.5  # Neutral if no embeddings

        if neighbor_id not in embeddings:
            return 0.3  # Low score if neighbor has no embedding

        neighbor_embedding = embeddings[neighbor_id]

        # Cosine similarity
        center_norm = np.linalg.norm(center_embedding)
        neighbor_norm = np.linalg.norm(neighbor_embedding)

        if center_norm < 1e-10 or neighbor_norm < 1e-10:
            return 0.5

        cosine_sim = np.dot(center_embedding, neighbor_embedding) / (
            center_norm * neighbor_norm
        )

        # Transform from [-1, 1] to [0, 1]
        score = (cosine_sim + 1) / 2

        return float(np.clip(score, 0.0, 1.0))

    def _compute_memory_score(
        self,
        neighbor_id: str,
        edge_type: str,
        memory: Optional[Any],
    ) -> float:
        """
        Compute score based on historical effectiveness from Memory.

        Args:
            neighbor_id: Neighbor gene ID
            edge_type: Type of connecting edge
            memory: Memory instance

        Returns:
            Memory-based score in [0, 1]
        """
        if memory is None:
            return 0.5  # Neutral if no memory

        score = 0.5

        # Get edge type effectiveness from memory
        if hasattr(memory, 'get_edge_effectiveness'):
            edge_effects = memory.get_edge_effectiveness()
            if edge_type in edge_effects:
                effect = edge_effects[edge_type]
                ema = effect.get('ema_effect', 0)
                # Transform EMA from [-1, 1] to [0, 1]
                score = (ema + 1) / 2

        # Get neighbor-specific suggestions if available
        if hasattr(memory, 'get_neighbor_selection_suggestions'):
            suggestions = memory.get_neighbor_selection_suggestions(
                '', [neighbor_id]
            )
            if neighbor_id in suggestions:
                # Combine with edge type score
                neighbor_score = suggestions[neighbor_id]
                score = 0.6 * score + 0.4 * neighbor_score

        return float(np.clip(score, 0.0, 1.0))

    def _compute_confidence(
        self,
        has_task_genes: bool,
        has_embedding: bool,
        has_memory: bool,
    ) -> float:
        """
        Compute confidence in the score based on available information.

        Args:
            has_task_genes: Whether task genes were provided
            has_embedding: Whether embedding was available
            has_memory: Whether memory was available

        Returns:
            Confidence score in [0, 1]
        """
        # Base confidence from structural (always available)
        confidence = 0.4

        # Add confidence from other sources
        if has_task_genes:
            confidence += 0.2
        if has_embedding:
            confidence += 0.25
        if has_memory:
            confidence += 0.15

        return float(np.clip(confidence, 0.0, 1.0))

    def score_all_neighbors_for_gene(
        self,
        gene_id: str,
        edge_types: Optional[List[str]] = None,
        task_genes: Optional[Set[str]] = None,
        memory: Optional[Any] = None,
        embeddings: Optional[Dict[str, np.ndarray]] = None,
    ) -> Dict[str, NeighborScore]:
        """
        Score all neighbors of a gene in the KG.

        Convenience method that extracts neighbors from the KG.

        Args:
            gene_id: Gene to score neighbors for
            edge_types: Optional filter for edge types
            task_genes: Task-relevant genes
            memory: Memory instance
            embeddings: Embeddings dictionary

        Returns:
            Dictionary mapping neighbor_id to NeighborScore
        """
        if gene_id not in self.kg:
            logger.warning(f"Gene {gene_id} not in KG")
            return {}

        neighbors = []

        # Get outgoing edges
        for _, neighbor, data in self.kg.out_edges(gene_id, data=True):
            edge_type = data.get('edge_type', data.get('label', 'unknown'))
            if edge_types is None or edge_type in edge_types:
                neighbors.append((neighbor, edge_type, 'out'))

        # Get incoming edges
        for neighbor, _, data in self.kg.in_edges(gene_id, data=True):
            edge_type = data.get('edge_type', data.get('label', 'unknown'))
            if edge_types is None or edge_type in edge_types:
                neighbors.append((neighbor, edge_type, 'in'))

        return self.score_neighbors(
            center_gene=gene_id,
            neighbors=neighbors,
            task_genes=task_genes,
            memory=memory,
            embeddings=embeddings,
        )

    def get_top_neighbors(
        self,
        scores: Dict[str, NeighborScore],
        k: int = 10,
        min_score: float = 0.0,
    ) -> List[NeighborScore]:
        """
        Get top-k neighbors by score.

        Args:
            scores: Dictionary of neighbor scores
            k: Number of neighbors to return
            min_score: Minimum score threshold

        Returns:
            List of top-k NeighborScore objects
        """
        filtered = [s for s in scores.values() if s.total_score >= min_score]
        sorted_scores = sorted(
            filtered,
            key=lambda x: x.total_score,
            reverse=True
        )
        return sorted_scores[:k]

    def categorize_by_relevance(
        self,
        scores: Dict[str, NeighborScore],
        high_threshold: float = 0.7,
        low_threshold: float = 0.4,
    ) -> Dict[str, List[NeighborScore]]:
        """
        Categorize neighbors into high/medium/low relevance groups.

        Args:
            scores: Dictionary of neighbor scores
            high_threshold: Threshold for high relevance
            low_threshold: Threshold for low relevance

        Returns:
            Dictionary with 'high', 'medium', 'low' keys
        """
        categories = {
            'high': [],
            'medium': [],
            'low': [],
        }

        for score in scores.values():
            if score.total_score >= high_threshold:
                categories['high'].append(score)
            elif score.total_score >= low_threshold:
                categories['medium'].append(score)
            else:
                categories['low'].append(score)

        # Sort each category by score
        for key in categories:
            categories[key] = sorted(
                categories[key],
                key=lambda x: x.total_score,
                reverse=True
            )

        return categories
