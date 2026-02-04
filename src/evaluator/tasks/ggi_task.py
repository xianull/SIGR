"""
Gene-Gene Interaction (GGI) Prediction Task for SIGR Evaluation

Binary classification of gene pair interactions.
"""

import os
import logging
from typing import Dict, Tuple, Optional, List

import numpy as np
import networkx as nx

from .base_task import BaseTask


logger = logging.getLogger(__name__)


class GGITask(BaseTask):
    """
    Gene-Gene Interaction prediction task.

    Predicts whether two genes interact based on their concatenated embeddings.
    Uses pre-defined train/valid/test splits.

    Data source: data/downstreams/ggi/
    Format: gene1 gene2 (text) + 0/1 (label)
    """

    DEFAULT_DATA_DIR = "data/downstreams/ggi"

    def __init__(
        self,
        kg: nx.DiGraph,
        data_dir: Optional[str] = None,
        split: str = 'all'
    ):
        """
        Initialize the GGI task.

        Args:
            kg: Knowledge graph
            data_dir: Directory containing GGI data
            split: Which split to use ('train', 'valid', 'test', or 'all')
        """
        self.data_dir = data_dir or self.DEFAULT_DATA_DIR
        self.split = split

        super().__init__('ggi', kg, self.data_dir)

        self._pairs: List[Tuple[str, str]] = []
        self._labels: List[int] = []

    def _load_split(self, split_name: str) -> Tuple[List[Tuple[str, str]], List[int]]:
        """
        Load a single data split.

        Args:
            split_name: 'train', 'valid', or 'test'

        Returns:
            Tuple of (pairs, labels)
        """
        text_file = os.path.join(self.data_dir, f'{split_name}_text.txt')
        label_file = os.path.join(self.data_dir, f'{split_name}_label.txt')

        if not os.path.exists(text_file):
            logger.warning(f"Text file not found: {text_file}")
            return [], []

        pairs = []
        with open(text_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    pairs.append((parts[0], parts[1]))

        labels = []
        if os.path.exists(label_file):
            with open(label_file, 'r') as f:
                for line in f:
                    try:
                        labels.append(int(line.strip()))
                    except ValueError:
                        continue

        # Ensure same length
        min_len = min(len(pairs), len(labels))
        pairs = pairs[:min_len]
        labels = labels[:min_len]

        logger.info(f"Loaded {split_name}: {len(pairs)} pairs")
        return pairs, labels

    def _load_data(self) -> Tuple[List[Tuple[str, str]], List[int]]:
        """Load all data based on split setting."""
        if self._pairs:
            return self._pairs, self._labels

        if self.split == 'all':
            splits = ['train', 'valid', 'test']
        else:
            splits = [self.split]

        all_pairs, all_labels = [], []
        for s in splits:
            pairs, labels = self._load_split(s)
            all_pairs.extend(pairs)
            all_labels.extend(labels)

        self._pairs = all_pairs
        self._labels = all_labels

        logger.info(f"Total GGI data: {len(self._pairs)} pairs")
        return self._pairs, self._labels

    def prepare_data(
        self,
        embeddings: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare gene pair features.

        Concatenates embeddings of gene pairs in sorted order for consistency.

        Args:
            embeddings: Dictionary mapping gene_id to embedding

        Returns:
            Tuple of (X, y) feature matrix and binary labels
        """
        pairs, labels = self._load_data()

        X_list, y_list = [], []
        missing_count = 0

        for (g1, g2), label in zip(pairs, labels):
            # GGI data uses gene symbols
            key1 = self.symbol_to_embedding_key(g1, embeddings)
            key2 = self.symbol_to_embedding_key(g2, embeddings)

            if key1 is None or key2 is None:
                missing_count += 1
                continue

            # Concatenate embeddings in sorted order for consistency
            if g1 <= g2:
                feature = np.concatenate([embeddings[key1], embeddings[key2]])
            else:
                feature = np.concatenate([embeddings[key2], embeddings[key1]])

            X_list.append(feature)
            y_list.append(label)

        if not X_list:
            logger.warning("No gene pairs found with embeddings")
            return np.array([]), np.array([])

        X = np.vstack(X_list)
        y = np.array(y_list)

        logger.info(f"GGI task ({self.split}): {len(X)} pairs, "
                   f"{sum(y)} positive, {len(y) - sum(y)} negative")
        logger.info(f"Missing pairs: {missing_count}")

        return X, y

    def prepare_data_with_splits(
        self,
        embeddings: Dict[str, np.ndarray]
    ) -> Tuple[Tuple[np.ndarray, np.ndarray],
               Tuple[np.ndarray, np.ndarray],
               Tuple[np.ndarray, np.ndarray]]:
        """
        Prepare data with original train/valid/test splits.

        Args:
            embeddings: Dictionary mapping gene_id to embedding

        Returns:
            Tuple of (train_data, valid_data, test_data)
            Each is (X, y) tuple
        """
        results = []
        for split in ['train', 'valid', 'test']:
            task = GGITask(self.kg, self.data_dir, split=split)
            X, y = task.prepare_data(embeddings)
            results.append((X, y))

        return tuple(results)

    def get_metric_name(self) -> str:
        """Return primary metric for GGI task."""
        return 'auc'

    def get_task_type(self) -> str:
        """Return task type."""
        return 'pair'

    def get_split_sizes(self) -> Dict[str, int]:
        """Return sizes of each split."""
        sizes = {}
        for split in ['train', 'valid', 'test']:
            pairs, _ = self._load_split(split)
            sizes[split] = len(pairs)
        return sizes

    def get_task_genes(self) -> List[str]:
        """
        Get list of genes relevant to this task.

        Returns all unique genes from gene pairs.
        """
        pairs, _ = self._load_data()

        genes = set()
        for g1, g2 in pairs:
            genes.add(g1)
            genes.add(g2)

        return list(genes)
