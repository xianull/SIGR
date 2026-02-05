"""
Gene Attribute Classification Tasks for SIGR Evaluation

Binary classification tasks for various gene attributes:
- Dosage sensitivity
- H3K4me1 (Lys4) methylation
- No methylation
- Bivalent chromatin mark
"""

import os
import random
import logging
from typing import Dict, Tuple, Optional, Set, List

import numpy as np
import pandas as pd
import networkx as nx

from .base_task import BaseTask


logger = logging.getLogger(__name__)


class GeneAttributeTask(BaseTask):
    """
    Gene attribute binary classification task.

    Supports multiple subtasks:
    - dosage_sensitivity: Dosage-sensitive vs insensitive genes
    - lys4_only: Genes with H3K4me1 mark only
    - no_methylation: Genes with no methylation
    - bivalent: Genes with bivalent chromatin marks

    Data source: data/downstreams/geneattribute/
    """

    SUBTASKS = {
        'dosage_sensitivity': {
            'file': 'dosage_sens_tf_labels.csv',
            'format': 'csv_two_column',
            'metric': 'auc',
            'description': 'Dosage sensitive vs insensitive transcription factors'
        },
        'lys4_only': {
            'file': 'lys4_only_gene_labels.txt',
            'format': 'positive_list',
            'metric': 'auc',
            'description': 'Genes with H3K4me1 mark only'
        },
        'no_methylation': {
            'file': 'no_methylation_gene_labels.txt',
            'format': 'positive_list',
            'metric': 'auc',
            'description': 'Genes with no methylation'
        },
        'bivalent': {
            'file': 'bivalent_gene_labels.txt',
            'format': 'positive_list',
            'metric': 'auc',
            'description': 'Genes with bivalent chromatin marks'
        },
        # Cross-category comparisons (GenePT benchmark tasks)
        'bivalent_vs_no_methylation': {
            'positive_file': 'bivalent_gene_labels.txt',
            'negative_file': 'no_methylation_gene_labels.txt',
            'format': 'two_file_comparison',
            'metric': 'auc',
            'description': 'Bivalent vs non-methylated genes (GenePT benchmark)'
        },
        'bivalent_vs_lys4_only': {
            'positive_file': 'bivalent_gene_labels.txt',
            'negative_file': 'lys4_only_gene_labels.txt',
            'format': 'two_file_comparison',
            'metric': 'auc',
            'description': 'Bivalent vs H3K4me1-only genes (GenePT benchmark)'
        },
        'tf_range': {
            'file': 'dosage_sens_tf_labels.csv',
            'format': 'tf_range',
            'metric': 'auc',
            'description': 'TF range classification (dosage sensitive TFs)'
        }
    }

    DEFAULT_DATA_DIR = "data/downstreams/geneattribute"

    def __init__(
        self,
        kg: nx.DiGraph,
        subtask: str = 'dosage_sensitivity',
        data_dir: Optional[str] = None,
        negative_ratio: float = 1.0
    ):
        """
        Initialize the gene attribute task.

        Args:
            kg: Knowledge graph
            subtask: Name of the subtask
            data_dir: Directory containing task data
            negative_ratio: Ratio of negative to positive samples
        """
        if subtask not in self.SUBTASKS:
            raise ValueError(
                f"Unknown subtask: {subtask}. "
                f"Available: {list(self.SUBTASKS.keys())}"
            )

        self.subtask = subtask
        self.subtask_config = self.SUBTASKS[subtask]
        self.data_dir = data_dir or self.DEFAULT_DATA_DIR
        self.negative_ratio = negative_ratio

        # Determine data path based on format
        fmt = self.subtask_config.get('format', '')
        if fmt == 'two_file_comparison':
            # Use positive file as primary data path
            data_path = os.path.join(self.data_dir, self.subtask_config['positive_file'])
        else:
            data_path = os.path.join(self.data_dir, self.subtask_config['file'])

        super().__init__(f'geneattribute_{subtask}', kg, data_path)

        self._positive_genes: Set[str] = set()
        self._negative_genes: Set[str] = set()

    def _load_data(self) -> Tuple[Set[str], Set[str]]:
        """Load positive and negative gene sets."""
        if self._positive_genes:
            return self._positive_genes, self._negative_genes

        logger.info(f"Loading {self.subtask} data from {self.data_path}")
        fmt = self.subtask_config['format']

        if fmt == 'csv_two_column':
            self._load_csv_two_column()
        elif fmt == 'positive_list':
            self._load_positive_list()
        elif fmt == 'two_file_comparison':
            self._load_two_file_comparison()
        elif fmt == 'tf_range':
            self._load_tf_range()
        else:
            raise ValueError(f"Unknown format: {fmt}")

        logger.info(f"Loaded {len(self._positive_genes)} positive, "
                   f"{len(self._negative_genes)} negative genes")

        return self._positive_genes, self._negative_genes

    def _load_csv_two_column(self):
        """Load CSV with two columns (positive and negative genes)."""
        df = pd.read_csv(self.data_path)
        col_names = df.columns.tolist()

        # First column is typically positive (dosage_sensitive)
        # Second column is negative (dosage_insensitive)
        if len(col_names) >= 2:
            positive_col = col_names[0]
            negative_col = col_names[1]

            self._positive_genes = set(
                df[positive_col].dropna().astype(str).str.strip()
            )
            self._negative_genes = set(
                df[negative_col].dropna().astype(str).str.strip()
            )

            # Remove empty strings
            self._positive_genes.discard('')
            self._negative_genes.discard('')

    def _load_positive_list(self):
        """Load text file with one gene per line (positive only)."""
        with open(self.data_path, 'r') as f:
            self._positive_genes = set(
                line.strip() for line in f
                if line.strip() and not line.startswith('#')
            )
        # Negative genes will be sampled later
        self._negative_genes = set()

    def _load_two_file_comparison(self):
        """Load two files for cross-category comparison (e.g., bivalent vs no_methylation)."""
        # Load positive genes from positive_file
        positive_file = os.path.join(self.data_dir, self.subtask_config['positive_file'])
        with open(positive_file, 'r') as f:
            self._positive_genes = set(
                line.strip() for line in f
                if line.strip() and not line.startswith('#')
            )

        # Load negative genes from negative_file
        negative_file = os.path.join(self.data_dir, self.subtask_config['negative_file'])
        with open(negative_file, 'r') as f:
            self._negative_genes = set(
                line.strip() for line in f
                if line.strip() and not line.startswith('#')
            )

        # Remove overlap (genes in both categories are ambiguous)
        overlap = self._positive_genes & self._negative_genes
        if overlap:
            logger.warning(f"Removing {len(overlap)} overlapping genes from both classes")
            self._positive_genes -= overlap
            self._negative_genes -= overlap

    def _load_tf_range(self):
        """Load TF range data (same as dosage sensitivity but may filter differently)."""
        # For TF range, we use the same dosage sensitivity CSV
        # This task is essentially the same as dosage_sensitivity
        self._load_csv_two_column()

    def _sample_negatives(
        self,
        embeddings: Dict[str, np.ndarray],
        available_genes: Set[str]
    ) -> Set[str]:
        """
        Sample negative genes from available genes.

        Args:
            embeddings: Embedding dictionary
            available_genes: Set of genes with embeddings

        Returns:
            Set of negative gene IDs
        """
        # Convert positive genes to embedding keys
        positive_keys = set()
        for gene in self._positive_genes:
            key = self.ensembl_to_embedding_key(gene, embeddings)
            if key:
                positive_keys.add(key)

        # Candidates are all available genes minus positives
        candidates = available_genes - positive_keys

        # Sample negatives
        num_negatives = int(len(positive_keys) * self.negative_ratio)
        if len(candidates) < num_negatives:
            sampled = candidates
        else:
            sampled = set(random.sample(list(candidates), num_negatives))

        logger.info(f"Sampled {len(sampled)} negative genes")
        return sampled

    def prepare_data(
        self,
        embeddings: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare binary classification data.

        Args:
            embeddings: Dictionary mapping gene_id to embedding

        Returns:
            Tuple of (X, y) feature matrix and binary labels
        """
        positive_genes, negative_genes = self._load_data()

        # Get available genes with embeddings
        available_genes = set(embeddings.keys())

        # If no negatives loaded, sample them
        if not negative_genes:
            negative_genes = self._sample_negatives(embeddings, available_genes)

        X_list, y_list = [], []

        # Process positive genes
        for gene in positive_genes:
            key = self.ensembl_to_embedding_key(gene, embeddings)
            if key:
                X_list.append(embeddings[key])
                y_list.append(1)

        # Process negative genes
        for gene in negative_genes:
            # For sampled negatives, gene is already the key
            if gene in embeddings:
                X_list.append(embeddings[gene])
                y_list.append(0)
            else:
                # For loaded negatives (Ensembl IDs)
                key = self.ensembl_to_embedding_key(gene, embeddings)
                if key:
                    X_list.append(embeddings[key])
                    y_list.append(0)

        if not X_list:
            logger.warning(f"No genes found for {self.subtask}")
            return np.array([]), np.array([])

        X = np.vstack(X_list)
        y = np.array(y_list)

        logger.info(f"GeneAttribute-{self.subtask}: {sum(y)} positive, "
                   f"{len(y) - sum(y)} negative samples")

        return X, y

    def get_metric_name(self) -> str:
        """Return primary metric for this subtask."""
        return self.subtask_config['metric']

    def get_task_type(self) -> str:
        """Return task type."""
        return 'binary'

    def get_subtask_info(self) -> Dict:
        """Return information about this subtask."""
        return {
            'subtask': self.subtask,
            'description': self.subtask_config['description'],
            'metric': self.subtask_config['metric'],
        }

    @classmethod
    def get_available_subtasks(cls) -> List[str]:
        """Return list of available subtasks."""
        return list(cls.SUBTASKS.keys())

    def get_task_genes(self) -> List[str]:
        """
        Get list of genes relevant to this task.

        Returns gene symbols for genes in the task dataset.
        """
        positive_genes, negative_genes = self._load_data()

        # Build ensembl to symbol cache
        self._build_id_cache()

        # Convert Ensembl IDs to symbols
        symbols = []
        for gene in positive_genes | negative_genes:
            symbol = self._ensembl_to_symbol_cache.get(gene)
            if symbol:
                symbols.append(symbol)

        return symbols
