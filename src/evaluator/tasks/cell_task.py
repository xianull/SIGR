"""
Cell Type Classification Task for SIGR Evaluation

Multi-class classification of cell types with clustering evaluation.
Uses single-cell RNA-seq data in AnnData format.
"""

import logging
from typing import Dict, Tuple, Optional, List

import numpy as np
import networkx as nx
from sklearn.preprocessing import LabelEncoder

from .base_task import BaseTask


logger = logging.getLogger(__name__)


class CellTask(BaseTask):
    """
    Cell type classification and clustering task.

    Predicts cell types based on aggregated gene embeddings.
    Each cell's embedding is computed as the weighted average of its
    expressed genes' embeddings.

    Also computes clustering metrics (silhouette, ARI, NMI).

    Data source: data/downstreams/cell/sample_aorta_data_updated.h5ad
    Format: AnnData (cells × genes)
    """

    DEFAULT_DATA_PATH = "data/downstreams/cell/sample_aorta_data_updated.h5ad"

    def __init__(
        self,
        kg: nx.DiGraph,
        data_path: Optional[str] = None,
        aggregation: str = 'weighted_mean',
        top_k_genes: int = 500,
        filter_unknown: bool = True,
        min_genes_per_cell: int = 10
    ):
        """
        Initialize the cell type task.

        Args:
            kg: Knowledge graph
            data_path: Path to AnnData file
            aggregation: Method to aggregate gene embeddings
                         ('mean', 'weighted_mean', 'max')
            top_k_genes: Use top k expressed genes per cell
            filter_unknown: Filter out 'Unknown' cell type
            min_genes_per_cell: Minimum genes required for valid embedding
        """
        super().__init__('cell', kg, data_path or self.DEFAULT_DATA_PATH)
        self.aggregation = aggregation
        self.top_k_genes = top_k_genes
        self.filter_unknown = filter_unknown
        self.min_genes_per_cell = min_genes_per_cell

        self._adata = None
        self._label_encoder: Optional[LabelEncoder] = None

    def _load_adata(self):
        """Load AnnData file."""
        if self._adata is not None:
            return self._adata

        try:
            import scanpy as sc
        except ImportError:
            raise ImportError(
                "scanpy is required for cell task. "
                "Install with: pip install scanpy"
            )

        logger.info(f"Loading AnnData from {self.data_path}")
        self._adata = sc.read_h5ad(self.data_path)

        # Filter Unknown cell type
        if self.filter_unknown and 'celltype' in self._adata.obs.columns:
            mask = self._adata.obs['celltype'] != 'Unknown'
            self._adata = self._adata[mask].copy()

        logger.info(f"Loaded AnnData: {self._adata.shape[0]} cells, "
                   f"{self._adata.shape[1]} genes")

        if 'celltype' in self._adata.obs.columns:
            logger.info(f"Cell types: {self._adata.obs['celltype'].nunique()}")

        return self._adata

    def _get_cell_embedding(
        self,
        cell_idx: int,
        gene_embeddings: Dict[str, np.ndarray]
    ) -> Optional[np.ndarray]:
        """
        Compute embedding for a single cell.

        Aggregates embeddings of expressed genes weighted by expression.

        Args:
            cell_idx: Index of cell in AnnData
            gene_embeddings: Dictionary mapping gene_id to embedding

        Returns:
            Cell embedding or None if insufficient genes
        """
        adata = self._load_adata()

        # Get expression vector for this cell
        cell_expr = adata.X[cell_idx]
        if hasattr(cell_expr, 'toarray'):
            cell_expr = cell_expr.toarray().flatten()
        else:
            cell_expr = np.array(cell_expr).flatten()

        gene_names = adata.var_names.tolist()

        # Collect genes with embeddings and positive expression
        valid_embeddings = []
        valid_weights = []

        for i, gene in enumerate(gene_names):
            if cell_expr[i] <= 0:
                continue

            key = self.symbol_to_embedding_key(gene, gene_embeddings)
            if key is not None:
                valid_embeddings.append(gene_embeddings[key])
                valid_weights.append(cell_expr[i])

        if len(valid_embeddings) < self.min_genes_per_cell:
            return None

        # Select top-k genes by expression
        if len(valid_embeddings) > self.top_k_genes:
            sorted_indices = np.argsort(valid_weights)[::-1][:self.top_k_genes]
            valid_embeddings = [valid_embeddings[i] for i in sorted_indices]
            valid_weights = [valid_weights[i] for i in sorted_indices]

        embeddings_array = np.vstack(valid_embeddings)
        weights_array = np.array(valid_weights)

        # Aggregate
        if self.aggregation == 'mean':
            cell_emb = np.mean(embeddings_array, axis=0)
        elif self.aggregation == 'weighted_mean':
            weights_norm = weights_array / np.sum(weights_array)
            cell_emb = np.average(embeddings_array, axis=0, weights=weights_norm)
        elif self.aggregation == 'max':
            cell_emb = np.max(embeddings_array, axis=0)
        else:
            cell_emb = np.mean(embeddings_array, axis=0)

        return cell_emb

    def prepare_data(
        self,
        embeddings: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare cell classification data.

        Args:
            embeddings: Dictionary mapping gene_id to embedding

        Returns:
            Tuple of (X, y) cell embeddings and encoded labels
        """
        adata = self._load_adata()

        if 'celltype' not in adata.obs.columns:
            raise ValueError("AnnData must have 'celltype' in obs")

        X_list, y_list = [], []
        skipped_cells = 0

        logger.info(f"Computing cell embeddings for {adata.shape[0]} cells...")

        for i in range(adata.shape[0]):
            cell_emb = self._get_cell_embedding(i, embeddings)
            if cell_emb is not None:
                X_list.append(cell_emb)
                y_list.append(adata.obs['celltype'].iloc[i])
            else:
                skipped_cells += 1

            # Progress logging
            if (i + 1) % 1000 == 0:
                logger.info(f"Processed {i + 1}/{adata.shape[0]} cells")

        if not X_list:
            logger.warning("No cells with sufficient gene embeddings")
            return np.array([]), np.array([])

        # Encode labels
        self._label_encoder = LabelEncoder()
        y_encoded = self._label_encoder.fit_transform(y_list)

        X = np.vstack(X_list)
        y = np.array(y_encoded)

        logger.info(f"Cell task: {len(X)} cells, "
                   f"{len(self._label_encoder.classes_)} cell types")
        logger.info(f"Skipped cells (insufficient genes): {skipped_cells}")

        return X, y

    def get_metric_name(self) -> str:
        """Return primary metric for cell task."""
        return 'f1'

    def get_task_type(self) -> str:
        """Return task type."""
        return 'multiclass'

    def get_num_classes(self) -> int:
        """Return number of cell types."""
        if self._label_encoder is not None:
            return len(self._label_encoder.classes_)
        adata = self._load_adata()
        return adata.obs['celltype'].nunique()

    def requires_clustering_metrics(self) -> bool:
        """Cell task requires clustering metrics."""
        return True

    def get_cell_types(self) -> List[str]:
        """Return list of cell types."""
        if self._label_encoder is not None:
            return list(self._label_encoder.classes_)
        adata = self._load_adata()
        return adata.obs['celltype'].unique().tolist()

    def get_cell_type_distribution(self) -> Dict[str, int]:
        """Return cell type distribution."""
        adata = self._load_adata()
        return adata.obs['celltype'].value_counts().to_dict()

    def get_data_info(self) -> Dict:
        """Return information about the dataset."""
        adata = self._load_adata()
        return {
            'n_cells': adata.shape[0],
            'n_genes': adata.shape[1],
            'n_cell_types': adata.obs['celltype'].nunique(),
            'cell_types': self.get_cell_types(),
            'aggregation': self.aggregation,
            'top_k_genes': self.top_k_genes,
        }

    def get_task_genes(self) -> List[str]:
        """
        Get list of genes relevant to this task.

        Returns all gene names from the AnnData matrix.
        """
        adata = self._load_adata()
        return adata.var_names.tolist()
