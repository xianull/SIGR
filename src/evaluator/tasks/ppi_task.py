"""
PPI Prediction Task for SIGR Evaluation

Predicts protein-protein interactions using gene embeddings.
"""

import logging
import random
from typing import Dict, Tuple, List, Set

import numpy as np
import networkx as nx

from .base_task import BaseTask

logger = logging.getLogger(__name__)


class PPITask(BaseTask):
    """
    Protein-Protein Interaction prediction task.

    Given embeddings for genes, predict whether two genes interact.
    Uses a binary classification approach with concatenated embeddings.
    """

    def __init__(self, kg: nx.DiGraph, negative_ratio: float = 1.0):
        """
        Initialize the PPI task.

        Args:
            kg: Knowledge graph
            negative_ratio: Ratio of negative to positive samples
        """
        super().__init__('ppi', kg)
        self.negative_ratio = negative_ratio

        # Cache PPI edges
        self._positive_pairs = None
        self._all_genes = None

    def _get_ppi_pairs(self) -> Set[Tuple[str, str]]:
        """Get all PPI pairs from the KG."""
        if self._positive_pairs is not None:
            return self._positive_pairs

        pairs = set()
        for source, target, data in self.kg.edges(data=True):
            label = data.get('relationship_label') or data.get('label', '')
            if label == 'pairwise molecular interaction':
                # Store as sorted tuple to avoid duplicates
                pair = tuple(sorted([source, target]))
                pairs.add(pair)

        self._positive_pairs = pairs
        logger.info(f"Found {len(pairs)} PPI pairs in KG")
        return pairs

    def _get_all_genes(self) -> List[str]:
        """Get all genes from the KG."""
        if self._all_genes is not None:
            return self._all_genes

        genes = []
        for node, data in self.kg.nodes(data=True):
            label = data.get('node_label') or data.get('label', '')
            if label == 'gene':
                genes.append(node)

        self._all_genes = genes
        return genes

    def prepare_data(
        self,
        embeddings: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data for PPI prediction.

        Creates feature vectors by concatenating or combining embeddings
        of gene pairs, with positive labels for known interactions and
        negative labels for random non-interacting pairs.

        Args:
            embeddings: Dictionary mapping gene_id to embedding

        Returns:
            Tuple of (X, y) where:
            - X: Feature matrix (num_pairs, 2 * embedding_dim)
            - y: Binary labels (1 for interaction, 0 for no interaction)
        """
        # Get available genes (those with embeddings)
        available_genes = set(embeddings.keys())
        all_ppi_pairs = self._get_ppi_pairs()

        # Filter to pairs where both genes have embeddings
        positive_pairs = [
            (g1, g2) for g1, g2 in all_ppi_pairs
            if g1 in available_genes and g2 in available_genes
        ]

        if not positive_pairs:
            logger.warning("No positive pairs found with available embeddings")
            # Return empty data
            return np.array([]), np.array([])

        # Sample negative pairs
        num_negatives = int(len(positive_pairs) * self.negative_ratio)
        negative_pairs = self._sample_negative_pairs(
            available_genes, all_ppi_pairs, num_negatives
        )

        # Create feature vectors
        X_positive = []
        for g1, g2 in positive_pairs:
            # Concatenate embeddings (order-independent: use sorted)
            g1, g2 = sorted([g1, g2])
            feature = np.concatenate([embeddings[g1], embeddings[g2]])
            X_positive.append(feature)

        X_negative = []
        for g1, g2 in negative_pairs:
            g1, g2 = sorted([g1, g2])
            feature = np.concatenate([embeddings[g1], embeddings[g2]])
            X_negative.append(feature)

        # Combine
        X = np.vstack(X_positive + X_negative)
        y = np.array([1] * len(X_positive) + [0] * len(X_negative))

        logger.info(f"PPI task: {len(positive_pairs)} positive, {len(negative_pairs)} negative pairs")
        return X, y

    def _sample_negative_pairs(
        self,
        available_genes: Set[str],
        positive_pairs: Set[Tuple[str, str]],
        num_samples: int
    ) -> List[Tuple[str, str]]:
        """Sample negative (non-interacting) gene pairs."""
        genes = list(available_genes)
        negative_pairs = []
        max_attempts = num_samples * 10

        attempts = 0
        while len(negative_pairs) < num_samples and attempts < max_attempts:
            g1, g2 = random.sample(genes, 2)
            pair = tuple(sorted([g1, g2]))

            if pair not in positive_pairs and pair not in negative_pairs:
                negative_pairs.append((g1, g2))

            attempts += 1

        return negative_pairs

    def get_metric_name(self) -> str:
        """Return the primary metric for PPI task."""
        return 'auc'

    def get_task_genes(self) -> List[str]:
        """
        Get list of genes relevant to this task.

        Returns all unique genes from PPI pairs.
        """
        pairs = self._get_ppi_pairs()
        genes = set()
        for g1, g2 in pairs:
            genes.add(g1)
            genes.add(g2)
        return list(genes)
