"""
Gene Type Classification Task for SIGR Evaluation

Multi-class classification of gene functional types.
"""

import logging
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import networkx as nx
from sklearn.preprocessing import LabelEncoder

from .base_task import BaseTask


logger = logging.getLogger(__name__)


class GeneTypeTask(BaseTask):
    """
    Gene type multi-class classification task.

    Predicts the functional type of genes (protein_coding, pseudogene, lincRNA, etc.)
    based on their embeddings.

    Data source: data/downstreams/genetype/gene_info_table.csv
    Format: ensembl_id, gene_name, gene_type
    """

    DEFAULT_DATA_PATH = "data/downstreams/genetype/gene_info_table.csv"

    # Main gene types to include (filter out rare types)
    MAIN_GENE_TYPES = [
        'protein_coding',
        'pseudogene',
        'lincRNA',
        'lncRNA',
        'miRNA',
        'antisense',
        'misc_RNA',
        'snRNA',
        'snoRNA',
        'processed_transcript',
        'rRNA',
    ]

    def __init__(
        self,
        kg: nx.DiGraph,
        data_path: Optional[str] = None,
        min_class_size: int = 100,
        filter_main_types: bool = True
    ):
        """
        Initialize the gene type task.

        Args:
            kg: Knowledge graph
            data_path: Path to gene type data file
            min_class_size: Minimum number of samples per class
            filter_main_types: If True, only keep main gene types
        """
        super().__init__('genetype', kg, data_path or self.DEFAULT_DATA_PATH)
        self.min_class_size = min_class_size
        self.filter_main_types = filter_main_types

        self._data_df: Optional[pd.DataFrame] = None
        self._label_encoder: Optional[LabelEncoder] = None

    def _load_data(self) -> pd.DataFrame:
        """Load and preprocess gene type data."""
        if self._data_df is not None:
            return self._data_df

        logger.info(f"Loading gene type data from {self.data_path}")
        df = pd.read_csv(self.data_path)

        # Check columns
        expected_cols = ['ensembl_id', 'gene_name', 'gene_type']
        if not all(col in df.columns for col in expected_cols):
            # Try to infer columns
            if len(df.columns) >= 4:
                df.columns = ['index', 'ensembl_id', 'gene_name', 'gene_type']
            else:
                raise ValueError(f"Expected columns {expected_cols}, got {df.columns.tolist()}")

        # Filter missing values
        df = df.dropna(subset=['gene_type', 'ensembl_id'])
        df = df[~df['gene_type'].isin(['nan', 'None', ''])]

        # Filter gene types
        if self.filter_main_types:
            df = df[df['gene_type'].isin(self.MAIN_GENE_TYPES)]
            logger.info(f"Filtered to main gene types: {df['gene_type'].nunique()} types")
        else:
            # Filter by minimum class size
            type_counts = df['gene_type'].value_counts()
            valid_types = type_counts[type_counts >= self.min_class_size].index
            df = df[df['gene_type'].isin(valid_types)]
            logger.info(f"Filtered by min_class_size={self.min_class_size}: {len(valid_types)} types")

        self._data_df = df
        logger.info(f"Loaded {len(df)} genes with {df['gene_type'].nunique()} types")
        return df

    def prepare_data(
        self,
        embeddings: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare multi-class classification data.

        Args:
            embeddings: Dictionary mapping gene_id to embedding

        Returns:
            Tuple of (X, y) feature matrix and encoded labels
        """
        df = self._load_data()

        X_list, y_list = [], []
        missing_count = 0

        for _, row in df.iterrows():
            ensembl_id = row['ensembl_id']
            gene_type = row['gene_type']

            # Try to find embedding
            key = self.ensembl_to_embedding_key(ensembl_id, embeddings)
            if key is None:
                # Also try gene name
                gene_name = row.get('gene_name', '')
                if gene_name:
                    key = self.symbol_to_embedding_key(gene_name, embeddings)

            if key is None:
                missing_count += 1
                continue

            X_list.append(embeddings[key])
            y_list.append(gene_type)

        if not X_list:
            logger.warning("No genes found with embeddings")
            return np.array([]), np.array([])

        # Encode labels
        self._label_encoder = LabelEncoder()
        y_encoded = self._label_encoder.fit_transform(y_list)

        X = np.vstack(X_list)
        y = np.array(y_encoded)

        logger.info(f"GeneType task: {len(X)} samples, {len(self._label_encoder.classes_)} classes")
        logger.info(f"Missing embeddings: {missing_count}")

        return X, y

    def get_metric_name(self) -> str:
        """Return primary metric for gene type task."""
        return 'f1'

    def get_task_type(self) -> str:
        """Return task type."""
        return 'multiclass'

    def get_num_classes(self) -> int:
        """Return number of classes."""
        if self._label_encoder is not None:
            return len(self._label_encoder.classes_)
        return len(self.MAIN_GENE_TYPES)

    def get_class_names(self) -> List[str]:
        """Return class names."""
        if self._label_encoder is not None:
            return list(self._label_encoder.classes_)
        return self.MAIN_GENE_TYPES

    def get_class_distribution(self) -> Dict[str, int]:
        """Return class distribution in the dataset."""
        df = self._load_data()
        return df['gene_type'].value_counts().to_dict()

    def get_task_genes(self) -> List[str]:
        """
        Get list of genes relevant to this task.

        Returns gene symbols for genes in the task dataset.
        """
        df = self._load_data()

        # Build ensembl to symbol cache
        self._build_id_cache()

        # Convert Ensembl IDs to symbols
        symbols = []
        for _, row in df.iterrows():
            ensembl_id = row['ensembl_id']
            symbol = self._ensembl_to_symbol_cache.get(ensembl_id)
            if symbol:
                symbols.append(symbol)

        return symbols
