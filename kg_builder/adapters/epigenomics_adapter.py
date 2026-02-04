"""
Epigenomics Adapter

Loads chromatin state annotations (bivalent marks) from
processed Roadmap Epigenomics data.

Used for: Bivalent vs non-methylated vs Lys4-methylated classification tasks.

Bivalent genes: Have both H3K4me3 (active) and H3K27me3 (repressive) marks
Lys4-only: Have H3K4me3 but not H3K27me3
Non-methylated: Have neither mark
"""

import logging
from pathlib import Path
from typing import Dict, Any, Set, Optional, List

logger = logging.getLogger(__name__)


class EpigenomicsAdapter:
    """
    Adapter for chromatin state annotations.

    Processes H3K4me3 and H3K27me3 gene lists to determine
    chromatin state for each gene:
    - bivalent: Both H3K4me3 and H3K27me3
    - lys4_methylated: Only H3K4me3
    - lys27_methylated: Only H3K27me3
    - no_methylation: Neither mark

    This adapter provides gene attributes rather than edges.
    """

    def __init__(
        self,
        h3k4me3_path: Optional[str] = None,
        h3k27me3_path: Optional[str] = None,
        bivalent_genes_path: Optional[str] = None,
        lys4_only_path: Optional[str] = None,
        no_methylation_path: Optional[str] = None,
    ):
        """
        Initialize Epigenomics adapter.

        Can be initialized in two ways:
        1. Provide h3k4me3_path and h3k27me3_path to compute states
        2. Provide pre-computed gene lists (bivalent, lys4_only, no_methylation)

        Args:
            h3k4me3_path: Path to H3K4me3 marked genes list
            h3k27me3_path: Path to H3K27me3 marked genes list
            bivalent_genes_path: Path to pre-computed bivalent genes
            lys4_only_path: Path to pre-computed lys4-only genes
            no_methylation_path: Path to pre-computed non-methylated genes
        """
        self.h3k4me3_path = Path(h3k4me3_path) if h3k4me3_path else None
        self.h3k27me3_path = Path(h3k27me3_path) if h3k27me3_path else None
        self.bivalent_genes_path = Path(bivalent_genes_path) if bivalent_genes_path else None
        self.lys4_only_path = Path(lys4_only_path) if lys4_only_path else None
        self.no_methylation_path = Path(no_methylation_path) if no_methylation_path else None

        self._h3k4me3_genes: Set[str] = set()
        self._h3k27me3_genes: Set[str] = set()
        self._bivalent_genes: Set[str] = set()
        self._lys4_only_genes: Set[str] = set()
        self._no_methylation_genes: Set[str] = set()
        self._loaded = False

    def _load_gene_list(self, path: Path) -> Set[str]:
        """Load a gene list from file (one gene per line)."""
        if not path.exists():
            logger.warning(f"Gene list file not found: {path}")
            return set()

        genes = set()
        with open(path, 'r') as f:
            for line in f:
                gene = line.strip().upper()
                if gene and not gene.startswith('#'):
                    genes.add(gene)
        return genes

    def _load_data(self) -> None:
        """Load chromatin state data."""
        if self._loaded:
            return

        logger.info("Loading epigenomics chromatin state data")

        # Option 1: Load from histone mark files and compute states
        if self.h3k4me3_path and self.h3k27me3_path:
            self._h3k4me3_genes = self._load_gene_list(self.h3k4me3_path)
            self._h3k27me3_genes = self._load_gene_list(self.h3k27me3_path)

            # Compute chromatin states
            self._bivalent_genes = self._h3k4me3_genes & self._h3k27me3_genes
            self._lys4_only_genes = self._h3k4me3_genes - self._h3k27me3_genes
            self._lys27_only_genes = self._h3k27me3_genes - self._h3k4me3_genes

            logger.info(
                f"Computed: {len(self._bivalent_genes)} bivalent, "
                f"{len(self._lys4_only_genes)} lys4-only genes"
            )

        # Option 2: Load pre-computed state files
        else:
            if self.bivalent_genes_path:
                self._bivalent_genes = self._load_gene_list(self.bivalent_genes_path)
            if self.lys4_only_path:
                self._lys4_only_genes = self._load_gene_list(self.lys4_only_path)
            if self.no_methylation_path:
                self._no_methylation_genes = self._load_gene_list(self.no_methylation_path)

            logger.info(
                f"Loaded: {len(self._bivalent_genes)} bivalent, "
                f"{len(self._lys4_only_genes)} lys4-only, "
                f"{len(self._no_methylation_genes)} non-methylated genes"
            )

        self._loaded = True

    def get_gene_chromatin_state(self, gene_symbol: str) -> Optional[str]:
        """
        Get chromatin state for a gene.

        Args:
            gene_symbol: HGNC gene symbol

        Returns:
            Chromatin state: 'bivalent', 'lys4_methylated',
            'lys27_methylated', 'no_methylation', or None if unknown
        """
        self._load_data()
        gene = gene_symbol.upper()

        if gene in self._bivalent_genes:
            return 'bivalent'
        elif gene in self._lys4_only_genes:
            return 'lys4_methylated'
        elif gene in self._no_methylation_genes:
            return 'no_methylation'
        elif hasattr(self, '_lys27_only_genes') and gene in self._lys27_only_genes:
            return 'lys27_methylated'
        else:
            return None

    def get_gene_annotations(self) -> Dict[str, Dict[str, Any]]:
        """
        Get chromatin state annotations for all genes.

        Returns:
            Dict mapping gene symbol to annotation dict
        """
        self._load_data()

        annotations = {}

        for gene in self._bivalent_genes:
            annotations[gene] = {'chromatin_state': 'bivalent'}

        for gene in self._lys4_only_genes:
            annotations[gene] = {'chromatin_state': 'lys4_methylated'}

        for gene in self._no_methylation_genes:
            annotations[gene] = {'chromatin_state': 'no_methylation'}

        if hasattr(self, '_lys27_only_genes'):
            for gene in self._lys27_only_genes:
                annotations[gene] = {'chromatin_state': 'lys27_methylated'}

        return annotations

    def get_genes_by_state(self, state: str) -> Set[str]:
        """
        Get genes with a specific chromatin state.

        Args:
            state: One of 'bivalent', 'lys4_methylated', 'no_methylation'

        Returns:
            Set of gene symbols
        """
        self._load_data()

        if state == 'bivalent':
            return self._bivalent_genes.copy()
        elif state == 'lys4_methylated':
            return self._lys4_only_genes.copy()
        elif state == 'no_methylation':
            return self._no_methylation_genes.copy()
        elif state == 'lys27_methylated' and hasattr(self, '_lys27_only_genes'):
            return self._lys27_only_genes.copy()
        else:
            return set()

    def get_all_annotated_genes(self) -> Set[str]:
        """
        Get all genes with chromatin state annotations.

        Returns:
            Set of gene symbols
        """
        self._load_data()
        return (
            self._bivalent_genes |
            self._lys4_only_genes |
            self._no_methylation_genes
        )

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the loaded data.

        Returns:
            Dict with counts
        """
        self._load_data()

        stats = {
            'bivalent_genes': len(self._bivalent_genes),
            'lys4_methylated_genes': len(self._lys4_only_genes),
            'no_methylation_genes': len(self._no_methylation_genes),
            'total_annotated': len(self.get_all_annotated_genes()),
        }

        if hasattr(self, '_lys27_only_genes'):
            stats['lys27_methylated_genes'] = len(self._lys27_only_genes)

        return stats

    def get_task_labels(self) -> Dict[str, List[tuple]]:
        """
        Get labeled data for downstream classification tasks.

        Returns:
            Dict with task names mapping to (gene, label) tuples
        """
        self._load_data()

        tasks = {}

        # Binary: Bivalent vs Non-methylated
        bivalent_vs_no_meth = []
        for gene in self._bivalent_genes:
            bivalent_vs_no_meth.append((gene, 1))
        for gene in self._no_methylation_genes:
            bivalent_vs_no_meth.append((gene, 0))
        tasks['bivalent_vs_non_methylated'] = bivalent_vs_no_meth

        # Binary: Bivalent vs Lys4-methylated
        bivalent_vs_lys4 = []
        for gene in self._bivalent_genes:
            bivalent_vs_lys4.append((gene, 1))
        for gene in self._lys4_only_genes:
            bivalent_vs_lys4.append((gene, 0))
        tasks['bivalent_vs_lys4_methylated'] = bivalent_vs_lys4

        return tasks
