"""
Base Task for SIGR Evaluation

Abstract base class for downstream evaluation tasks.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Any, Optional, List

import numpy as np
import networkx as nx


logger = logging.getLogger(__name__)


class BaseTask(ABC):
    """
    Abstract base class for evaluation tasks.

    Each task must implement:
    - prepare_data(): Convert embeddings to training data (X, y)
    - get_metric_name(): Return the primary metric name

    Extended features:
    - Gene ID mapping (symbol <-> ensembl_id)
    - Task type classification
    - Clustering metrics flag
    """

    def __init__(
        self,
        task_name: str,
        kg: nx.DiGraph,
        data_path: Optional[str] = None
    ):
        """
        Initialize the task.

        Args:
            task_name: Name of the task
            kg: Knowledge graph
            data_path: Optional path to task-specific data
        """
        self.task_name = task_name
        self.kg = kg
        self.data_path = data_path

        # Lazy-loaded ID mapper
        self._id_mapper = None
        self._symbol_to_ensembl_cache: Dict[str, str] = {}
        self._ensembl_to_symbol_cache: Dict[str, str] = {}

    def _build_id_cache(self):
        """Build gene ID mapping cache from KG."""
        if self._symbol_to_ensembl_cache:
            return

        for node, data in self.kg.nodes(data=True):
            node_label = data.get('node_label') or data.get('label', '')
            if node_label == 'gene':
                symbol = data.get('symbol') or data.get('name', '')
                ensembl_id = data.get('ensembl_id', '')

                # Node ID is typically the gene symbol
                if symbol:
                    self._symbol_to_ensembl_cache[symbol] = node
                    self._ensembl_to_symbol_cache[node] = symbol

                # Also map ensembl_id to symbol (for evaluation data lookup)
                if ensembl_id:
                    self._ensembl_to_symbol_cache[ensembl_id] = symbol

        logger.debug(f"Built ID cache with {len(self._symbol_to_ensembl_cache)} genes")

    def symbol_to_embedding_key(
        self,
        symbol: str,
        embeddings: Dict[str, np.ndarray]
    ) -> Optional[str]:
        """
        Convert gene symbol to embedding dictionary key.

        Tries multiple strategies:
        1. Direct match with symbol
        2. Convert to ensembl ID and match
        3. Case-insensitive search

        Args:
            symbol: Gene symbol (e.g., 'TP53')
            embeddings: Embedding dictionary

        Returns:
            Key in embeddings dict, or None if not found
        """
        # Direct match
        if symbol in embeddings:
            return symbol

        # Build cache if needed
        self._build_id_cache()

        # Try ensembl conversion
        ensembl = self._symbol_to_ensembl_cache.get(symbol)
        if ensembl and ensembl in embeddings:
            return ensembl

        # Case-insensitive search (slower)
        symbol_upper = symbol.upper()
        for key in embeddings:
            if key.upper() == symbol_upper:
                return key

        return None

    def ensembl_to_embedding_key(
        self,
        ensembl_id: str,
        embeddings: Dict[str, np.ndarray]
    ) -> Optional[str]:
        """
        Convert Ensembl ID to embedding dictionary key.

        Tries multiple strategies:
        1. Direct match with ensembl ID
        2. Convert to symbol and match

        Args:
            ensembl_id: Ensembl gene ID (e.g., 'ENSG00000141510')
            embeddings: Embedding dictionary

        Returns:
            Key in embeddings dict, or None if not found
        """
        # Direct match (if embeddings use ensembl ID as key)
        if ensembl_id in embeddings:
            return ensembl_id

        # Build cache if needed
        self._build_id_cache()

        # Convert ensembl to symbol, then find in embeddings
        symbol = self._ensembl_to_symbol_cache.get(ensembl_id)
        if symbol:
            # Try symbol directly
            if symbol in embeddings:
                return symbol
            # Try case-insensitive
            for key in embeddings:
                if key.upper() == symbol.upper():
                    return key

        return None

    def get_available_genes(
        self,
        embeddings: Dict[str, np.ndarray]
    ) -> List[str]:
        """
        Get list of gene IDs that have embeddings.

        Args:
            embeddings: Embedding dictionary

        Returns:
            List of available gene IDs (keys in embeddings)
        """
        return list(embeddings.keys())

    def get_task_genes(self) -> List[str]:
        """
        Get list of genes relevant to this task.

        Returns gene symbols that are part of the task dataset.
        Override in subclasses for task-specific implementation.

        Returns:
            List of gene symbols
        """
        return []

    @abstractmethod
    def prepare_data(
        self,
        embeddings: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data from embeddings.

        Args:
            embeddings: Dictionary mapping gene_id to embedding

        Returns:
            Tuple of (X, y) where:
            - X: Feature matrix (n_samples, n_features)
            - y: Labels (n_samples,)
        """
        pass

    @abstractmethod
    def get_metric_name(self) -> str:
        """
        Get the primary metric name for this task.

        Returns:
            Metric name (e.g., 'auc', 'f1', 'accuracy')
        """
        pass

    def get_task_type(self) -> str:
        """
        Get the type of task.

        Returns:
            One of: 'binary', 'multiclass', 'multilabel', 'pair', 'regression'
        """
        return 'binary'

    def get_num_classes(self) -> int:
        """
        Get the number of classes for classification tasks.

        Returns:
            Number of classes
        """
        return 2

    def requires_clustering_metrics(self) -> bool:
        """
        Whether this task should compute clustering metrics.

        Returns:
            True if clustering metrics should be computed
        """
        return False

    def get_task_info(self) -> Dict[str, Any]:
        """
        Get information about the task.

        Returns:
            Dictionary with task information
        """
        return {
            'task_name': self.task_name,
            'metric_name': self.get_metric_name(),
            'task_type': self.get_task_type(),
            'num_classes': self.get_num_classes(),
            'requires_clustering': self.requires_clustering_metrics(),
        }
