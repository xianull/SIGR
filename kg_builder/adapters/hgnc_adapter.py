"""
HGNC Adapter for BioCypher

Loads gene information from HGNC (HUGO Gene Nomenclature Committee).
Provides standardized gene naming and cross-references.

Data source: https://www.genenames.org/
Output: Gene nodes with HGNC Symbol as primary ID
"""

import logging
from pathlib import Path
from typing import Generator, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class HGNCAdapter:
    """
    BioCypher adapter for HGNC gene nomenclature.

    Yields Gene nodes following Biolink model.
    Node type: biolink:Gene
    """

    def __init__(self, data_path: str):
        """
        Initialize HGNC adapter.

        Args:
            data_path: Path to hgnc_complete_set.txt
        """
        self.data_path = Path(data_path)
        self._df: Optional[pd.DataFrame] = None

    def _load_data(self) -> None:
        """Load HGNC data file."""
        if self._df is not None:
            return

        logger.info(f"Loading HGNC data from {self.data_path}")

        if not self.data_path.exists():
            raise FileNotFoundError(f"HGNC file not found: {self.data_path}")

        self._df = pd.read_csv(self.data_path, sep='\t', low_memory=False)
        logger.info(f"Loaded {len(self._df)} genes from HGNC")

    def get_nodes(self) -> Generator[tuple, None, None]:
        """
        Yield gene nodes in BioCypher format.

        Yields:
            Tuple of (node_id, node_label, properties)
            - node_id: HGNC Symbol (uppercase)
            - node_label: 'Gene'
            - properties: dict with gene attributes
        """
        self._load_data()

        for _, row in self._df.iterrows():
            symbol = row.get('symbol')
            if pd.isna(symbol):
                continue

            symbol = str(symbol).strip().upper()

            # Extract properties
            properties = {
                'symbol': symbol,
                'name': row.get('name') if pd.notna(row.get('name')) else None,
                'hgnc_id': row.get('hgnc_id') if pd.notna(row.get('hgnc_id')) else None,
                'ensembl_id': row.get('ensembl_gene_id') if pd.notna(row.get('ensembl_gene_id')) else None,
                'entrez_id': str(int(row.get('entrez_id'))) if pd.notna(row.get('entrez_id')) else None,
                'chromosome': row.get('location') if pd.notna(row.get('location')) else None,
                'gene_type': row.get('locus_type') if pd.notna(row.get('locus_type')) else None,
                'source_database': 'HGNC',
            }

            # Remove None values
            properties = {k: v for k, v in properties.items() if v is not None}

            yield (symbol, 'Gene', properties)

    def get_node_count(self) -> int:
        """Get total number of genes."""
        self._load_data()
        return len(self._df)
