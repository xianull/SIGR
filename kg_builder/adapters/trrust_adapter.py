"""
TRRUST Adapter for BioCypher

Loads transcription factor (TF) - target gene regulatory relationships
from the TRRUST database.

Data source: https://www.grnpedia.org/trrust/
Output: REGULATES edges (TF -> target gene)
"""

import logging
from pathlib import Path
from typing import Generator, Set, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class TRRUSTAdapter:
    """
    BioCypher adapter for TRRUST TF-target regulation.

    Edge type: biolink:Regulates
    Relationship label: REGULATES
    """

    def __init__(self, data_path: str):
        """
        Initialize TRRUST adapter.

        Args:
            data_path: Path to trrust_rawdata.human.tsv
        """
        self.data_path = Path(data_path)
        self._df: Optional[pd.DataFrame] = None
        self._tf_set: Set[str] = set()

    def _load_data(self) -> None:
        """Load TRRUST data file."""
        if self._df is not None:
            return

        logger.info(f"Loading TRRUST data from {self.data_path}")

        if not self.data_path.exists():
            raise FileNotFoundError(f"TRRUST file not found: {self.data_path}")

        self._df = pd.read_csv(
            self.data_path,
            sep='\t',
            names=['tf', 'target', 'regulation_type', 'pmid'],
            dtype=str
        )

        # Normalize gene symbols
        self._df['tf'] = self._df['tf'].str.strip().str.upper()
        self._df['target'] = self._df['target'].str.strip().str.upper()

        self._tf_set = set(self._df['tf'].dropna().unique())

        logger.info(
            f"Loaded {len(self._df)} regulations, "
            f"{len(self._tf_set)} unique TFs"
        )

    def get_edges(self) -> Generator[tuple, None, None]:
        """
        Yield TF-target regulatory edges in BioCypher format.

        Yields:
            Tuple of (source_id, target_id, relationship_label, properties)
            - source_id: TF gene symbol
            - target_id: Target gene symbol
            - relationship_label: 'REGULATES'
        """
        self._load_data()

        for _, row in self._df.iterrows():
            tf = row['tf']
            target = row['target']
            reg_type = row['regulation_type']
            pmid = row['pmid']

            if pd.isna(tf) or pd.isna(target):
                continue

            # Normalize regulation type
            if reg_type == 'Activation':
                direction = 'positive'
            elif reg_type == 'Repression':
                direction = 'negative'
            else:
                direction = 'unknown'

            properties = {
                'regulation_type': reg_type if pd.notna(reg_type) else 'Unknown',
                'direction': direction,
                'pubmed_id': pmid if pd.notna(pmid) else None,
                'source_database': 'TRRUST',
            }

            # Remove None values
            properties = {k: v for k, v in properties.items() if v is not None}

            yield (tf, target, 'REGULATES', properties)

    def get_tf_set(self) -> Set[str]:
        """Get set of all transcription factors."""
        self._load_data()
        return self._tf_set.copy()

    def get_edge_count(self) -> int:
        """Get total number of regulations."""
        self._load_data()
        return len(self._df)
