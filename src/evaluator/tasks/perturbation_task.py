"""
Gene Perturbation Prediction Task for SIGR Framework

Predicts expression changes after gene knockout/knockdown.
This is a REGRESSION task (not classification).

Based on Perturb-seq data:
- Input: Embedding of perturbed gene
- Output: Expression change vector (delta from control)
- Metric: Pearson correlation of predicted vs actual expression changes

Reference: GenePT perturbation experiments
"""

import logging
from typing import Dict, Tuple, Optional, List, Any

import numpy as np
import networkx as nx

from .base_task import BaseTask


logger = logging.getLogger(__name__)


class PerturbationTask(BaseTask):
    """
    Gene perturbation prediction task.

    Predicts expression changes when a gene is perturbed (knocked out/down).

    Data format (h5ad):
    - adata.obs['condition']: Perturbation condition (gene name or 'control')
    - adata.X: Expression matrix

    Workflow:
    1. Load h5ad data
    2. Compute control baseline expression
    3. For each perturbation condition, compute mean expression
    4. Calculate delta (perturbation - control)
    5. Use regression to predict delta from gene embedding
    6. Evaluate using Pearson correlation

    Key difference from classification tasks:
    - y is a continuous vector (expression profile), not discrete labels
    - Uses regressors instead of classifiers
    - Evaluation uses correlation instead of AUC/F1
    """

    # Control condition identifiers
    CONTROL_NAMES = {'control', 'ctrl', 'non-targeting', 'Control', 'CTRL', 'NT'}

    # Common column names for condition
    CONDITION_COLUMNS = ['condition', 'perturbation_name', 'gene', 'perturbation',
                         'target_gene', 'guide']

    def __init__(
        self,
        kg: nx.DiGraph,
        data_path: str,
        min_cells_per_condition: int = 3,
        exclude_double_perturbation: bool = True,
    ):
        """
        Initialize the perturbation task.

        Args:
            kg: Knowledge graph
            data_path: Path to h5ad file
            min_cells_per_condition: Minimum cells required for a condition
            exclude_double_perturbation: Whether to exclude combined perturbations (gene1+gene2)
        """
        super().__init__('perturbation', kg, data_path)

        self.min_cells_per_condition = min_cells_per_condition
        self.exclude_double_perturbation = exclude_double_perturbation

        # Data containers (lazy-loaded)
        self._loaded = False
        self.adata = None
        self.mean_ctrl: Optional[np.ndarray] = None  # Control baseline
        self.delta_expression: Dict[str, np.ndarray] = {}  # gene -> delta vector
        self.conditions: List[str] = []  # Valid perturbation conditions
        self.n_output_genes: int = 0  # Output dimension

    def _load_h5ad(self):
        """Load and preprocess h5ad data."""
        if self._loaded:
            return

        try:
            import scanpy as sc
        except ImportError:
            raise ImportError(
                "scanpy is required for PerturbationTask. "
                "Install with: pip install scanpy"
            )

        logger.info(f"Loading perturbation data from {self.data_path}")
        self.adata = sc.read_h5ad(self.data_path)

        # Find condition column
        condition_col = None
        for col in self.CONDITION_COLUMNS:
            if col in self.adata.obs.columns:
                condition_col = col
                break

        if condition_col is None:
            raise ValueError(
                f"No condition column found in h5ad. "
                f"Available columns: {list(self.adata.obs.columns)}"
            )

        # Standardize to 'condition'
        self.adata.obs['condition'] = self.adata.obs[condition_col].astype(str)

        # Compute control baseline
        ctrl_mask = self.adata.obs['condition'].isin(self.CONTROL_NAMES)
        n_ctrl = ctrl_mask.sum()

        if n_ctrl > 0:
            ctrl_expr = self.adata[ctrl_mask].X
            if hasattr(ctrl_expr, 'toarray'):
                ctrl_expr = ctrl_expr.toarray()
            self.mean_ctrl = np.array(ctrl_expr).mean(axis=0).flatten()
            logger.info(f"Control cells: {n_ctrl}")
        else:
            logger.warning("No control samples found! Using zero baseline.")
            self.mean_ctrl = np.zeros(self.adata.n_vars)

        self.n_output_genes = len(self.mean_ctrl)

        # Get valid perturbation conditions
        all_conditions = self.adata.obs['condition'].unique()

        for cond in all_conditions:
            # Skip control
            if cond in self.CONTROL_NAMES:
                continue

            # Skip NaN
            if cond == 'nan' or cond is None:
                continue

            # Skip double perturbations if requested
            if self.exclude_double_perturbation and '+' in str(cond):
                continue

            # Check minimum cells
            mask = self.adata.obs['condition'] == cond
            if mask.sum() < self.min_cells_per_condition:
                continue

            # Compute mean expression for this condition
            expr = self.adata[mask].X
            if hasattr(expr, 'toarray'):
                expr = expr.toarray()
            mean_expr = np.array(expr).mean(axis=0).flatten()

            # Store delta (perturbation - control)
            delta = mean_expr - self.mean_ctrl
            self.delta_expression[cond] = delta
            self.conditions.append(cond)

        logger.info(f"Loaded {len(self.conditions)} valid perturbation conditions")
        logger.info(f"Output dimension: {self.n_output_genes} genes")

        self._loaded = True

    def prepare_data(
        self,
        embeddings: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare regression data from embeddings.

        Args:
            embeddings: Dictionary mapping gene_id to embedding

        Returns:
            Tuple of (X, Y) where:
            - X: Feature matrix (n_conditions, embedding_dim)
            - Y: Target matrix (n_conditions, n_output_genes)
        """
        if not self._loaded:
            self._load_h5ad()

        X_list, Y_list = [], []
        matched_conditions = []

        for condition in self.conditions:
            # Find embedding for this gene
            emb_key = self.symbol_to_embedding_key(condition, embeddings)

            if emb_key is None:
                # Try case variations
                for key in embeddings:
                    if key.upper() == condition.upper():
                        emb_key = key
                        break

            if emb_key is None:
                continue

            X_list.append(embeddings[emb_key])
            Y_list.append(self.delta_expression[condition])
            matched_conditions.append(condition)

        if not X_list:
            logger.warning(f"No genes matched! Conditions: {self.conditions[:5]}...")
            return np.array([]), np.array([])

        X = np.vstack(X_list)
        Y = np.vstack(Y_list)

        logger.info(f"Perturbation task: {len(matched_conditions)} conditions matched, "
                    f"X shape: {X.shape}, Y shape: {Y.shape}")

        return X, Y

    def get_metric_name(self) -> str:
        """Return primary metric for perturbation task."""
        return 'delta_correlation'

    def get_task_type(self) -> str:
        """Return task type (regression for perturbation)."""
        return 'regression'

    def get_num_classes(self) -> int:
        """Not applicable for regression, return 0."""
        return 0

    def requires_clustering_metrics(self) -> bool:
        """Clustering metrics not applicable for regression."""
        return False

    def get_task_genes(self) -> List[str]:
        """Get list of perturbed genes."""
        if not self._loaded:
            self._load_h5ad()
        return list(self.conditions)

    def get_output_dim(self) -> int:
        """Get output dimension (number of genes in expression profile)."""
        if not self._loaded:
            self._load_h5ad()
        return self.n_output_genes

    def get_task_info(self) -> Dict[str, Any]:
        """Get task information."""
        if not self._loaded:
            try:
                self._load_h5ad()
            except Exception as e:
                logger.warning(f"Could not load data for task info: {e}")
                return {
                    'task_name': self.task_name,
                    'metric_name': self.get_metric_name(),
                    'task_type': self.get_task_type(),
                    'num_classes': self.get_num_classes(),
                    'requires_clustering': self.requires_clustering_metrics(),
                }

        return {
            'task_name': self.task_name,
            'metric_name': self.get_metric_name(),
            'task_type': self.get_task_type(),
            'num_classes': self.get_num_classes(),
            'requires_clustering': self.requires_clustering_metrics(),
            'num_conditions': len(self.conditions),
            'output_dim': self.n_output_genes,
        }
