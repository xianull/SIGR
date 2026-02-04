"""
STRING Adapter for BioCypher

Loads protein-protein interaction data from STRING database.

Data source: https://string-db.org/
Output: INTERACTS_WITH edges between genes
"""

import gzip
import logging
from pathlib import Path
from typing import Generator, Dict, Set, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class STRINGAdapter:
    """
    BioCypher adapter for STRING protein-protein interactions.

    Edge type: biolink:PairwiseMolecularInteraction
    Relationship label: INTERACTS_WITH
    """

    def __init__(
        self,
        links_path: str,
        aliases_path: str,
        score_threshold: int = 700,
    ):
        """
        Initialize STRING adapter.

        Args:
            links_path: Path to 9606.protein.links.v12.0.txt.gz
            aliases_path: Path to 9606.protein.aliases.v12.0.txt.gz
            score_threshold: Minimum combined score (0-1000), default 700 (high confidence)
        """
        self.links_path = Path(links_path)
        self.aliases_path = Path(aliases_path)
        self.score_threshold = score_threshold

        self._ensp_to_symbol: Dict[str, str] = {}
        self._links_df: Optional[pd.DataFrame] = None

    def _load_aliases(self) -> None:
        """Load ENSP to gene symbol mapping from aliases file."""
        if self._ensp_to_symbol:
            return

        logger.info(f"Loading STRING aliases from {self.aliases_path}")

        if not self.aliases_path.exists():
            raise FileNotFoundError(f"STRING aliases file not found: {self.aliases_path}")

        # Priority sources for gene symbol mapping
        priority_sources = [
            'BioMart_HUGO',
            'Ensembl_HGNC_symbol',
            'Ensembl_HGNC',
            'BLAST_KEGG_NAME',
        ]

        with gzip.open(self.aliases_path, 'rt') as f:
            next(f)  # Skip header
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    ensp = parts[0].replace('9606.', '')
                    alias = parts[1]
                    source = parts[2]

                    # Only use high-quality mappings
                    if source in priority_sources:
                        if ensp not in self._ensp_to_symbol:
                            self._ensp_to_symbol[ensp] = alias.upper()

        logger.info(f"Loaded {len(self._ensp_to_symbol)} ENSP to symbol mappings")

    def _load_links(self) -> None:
        """Load protein links with score filtering."""
        if self._links_df is not None:
            return

        logger.info(f"Loading STRING links from {self.links_path}")

        if not self.links_path.exists():
            raise FileNotFoundError(f"STRING links file not found: {self.links_path}")

        self._links_df = pd.read_csv(
            self.links_path,
            sep=' ',
            compression='gzip'
        )

        # Filter by score
        original_count = len(self._links_df)
        self._links_df = self._links_df[
            self._links_df['combined_score'] >= self.score_threshold
        ]

        logger.info(
            f"Loaded {len(self._links_df):,} interactions "
            f"(filtered from {original_count:,} with score >= {self.score_threshold})"
        )

    def get_edges(self) -> Generator[tuple, None, None]:
        """
        Yield PPI edges in BioCypher format.

        Yields:
            Tuple of (source_id, target_id, relationship_label, properties)
            - Uses gene symbols (not ENSP IDs)
            - Filters to mapped genes only
        """
        self._load_aliases()
        self._load_links()

        seen_edges: Set[tuple] = set()
        mapped = 0
        unmapped = 0

        for _, row in self._links_df.iterrows():
            ensp1 = row['protein1'].replace('9606.', '')
            ensp2 = row['protein2'].replace('9606.', '')
            score = row['combined_score']

            # Map to gene symbols
            gene1 = self._ensp_to_symbol.get(ensp1)
            gene2 = self._ensp_to_symbol.get(ensp2)

            if not gene1 or not gene2:
                unmapped += 1
                continue

            # Skip self-loops
            if gene1 == gene2:
                continue

            # Deduplicate (treat A-B same as B-A for undirected PPI)
            edge_key = tuple(sorted([gene1, gene2]))
            if edge_key in seen_edges:
                continue
            seen_edges.add(edge_key)

            properties = {
                'combined_score': score / 1000.0,  # Normalize to 0-1
                'source_database': 'STRING',
                'ensp1': ensp1,
                'ensp2': ensp2,
            }

            mapped += 1
            yield (gene1, gene2, 'INTERACTS_WITH', properties)

        logger.info(f"Yielded {mapped:,} edges, skipped {unmapped:,} unmapped")

    def get_edge_count(self) -> int:
        """Get approximate edge count."""
        self._load_links()
        return len(self._links_df)
