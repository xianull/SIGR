"""
HPO Adapter for BioCypher

Loads gene-phenotype associations from Human Phenotype Ontology.

Data source: https://hpo.jax.org/
Output: Phenotype nodes + HAS_PHENOTYPE edges
"""

import logging
from pathlib import Path
from typing import Generator, Set, Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class HPOAdapter:
    """
    BioCypher adapter for Human Phenotype Ontology.

    Node type: biolink:PhenotypicFeature
    Edge type: biolink:GeneToPhenotypicFeatureAssociation
    Relationship label: HAS_PHENOTYPE
    """

    def __init__(self, data_path: str):
        """
        Initialize HPO adapter.

        Args:
            data_path: Path to genes_to_phenotype.txt
        """
        self.data_path = Path(data_path)

        self._phenotypes: Dict[str, Dict] = {}
        self._associations: list = []
        self._loaded = False

    def _load_data(self) -> None:
        """Load HPO gene-phenotype associations."""
        if self._loaded:
            return

        logger.info(f"Loading HPO data from {self.data_path}")

        if not self.data_path.exists():
            raise FileNotFoundError(f"HPO file not found: {self.data_path}")

        # Read file - format varies by version
        # Try to detect columns
        df = pd.read_csv(self.data_path, sep='\t', comment='#')

        # Normalize column names
        df.columns = df.columns.str.lower().str.replace('-', '_').str.replace(' ', '_')

        # Find gene and HPO columns
        gene_col = None
        hpo_col = None
        hpo_name_col = None
        freq_col = None

        for col in df.columns:
            if 'gene_symbol' in col or col == 'gene':
                gene_col = col
            elif 'hpo_term_id' in col or 'hpo_id' in col:
                hpo_col = col
            elif 'hpo_term_name' in col or 'hpo_name' in col:
                hpo_name_col = col
            elif 'frequency' in col:
                freq_col = col

        if gene_col is None or hpo_col is None:
            # Try positional
            gene_col = df.columns[1] if len(df.columns) > 1 else None
            hpo_col = df.columns[3] if len(df.columns) > 3 else None
            hpo_name_col = df.columns[4] if len(df.columns) > 4 else None

        logger.info(f"Using columns: gene={gene_col}, hpo={hpo_col}")

        for _, row in df.iterrows():
            gene = row.get(gene_col)
            hpo_id = row.get(hpo_col)

            if pd.isna(gene) or pd.isna(hpo_id):
                continue

            gene = str(gene).strip().upper()
            hpo_id = str(hpo_id).strip()

            # Store phenotype
            if hpo_id not in self._phenotypes:
                hpo_name = row.get(hpo_name_col) if hpo_name_col else None
                self._phenotypes[hpo_id] = {
                    'hpo_id': hpo_id,
                    'name': hpo_name if pd.notna(hpo_name) else None,
                }

            # Store association
            frequency = row.get(freq_col) if freq_col else None
            self._associations.append({
                'gene': gene,
                'hpo_id': hpo_id,
                'frequency': frequency if pd.notna(frequency) else None,
            })

        self._loaded = True
        logger.info(
            f"Loaded {len(self._phenotypes)} phenotypes, "
            f"{len(self._associations)} associations"
        )

    def get_nodes(self) -> Generator[tuple, None, None]:
        """
        Yield phenotype nodes in BioCypher format.

        Yields:
            Tuple of (node_id, node_label, properties)
        """
        self._load_data()

        for hpo_id, info in self._phenotypes.items():
            properties = {
                'hpo_id': hpo_id,
                'source_database': 'HPO',
            }

            if info.get('name'):
                properties['name'] = info['name']

            yield (hpo_id, 'Phenotype', properties)

    def get_edges(self) -> Generator[tuple, None, None]:
        """
        Yield gene-phenotype association edges in BioCypher format.

        Yields:
            Tuple of (source_id, target_id, relationship_label, properties)
        """
        self._load_data()

        # Deduplicate
        seen: Set[tuple] = set()

        for assoc in self._associations:
            gene = assoc['gene']
            hpo_id = assoc['hpo_id']

            edge_key = (gene, hpo_id)
            if edge_key in seen:
                continue
            seen.add(edge_key)

            properties = {
                'source_database': 'HPO',
            }

            if assoc.get('frequency'):
                properties['frequency'] = assoc['frequency']

            yield (gene, hpo_id, 'HAS_PHENOTYPE', properties)

    def get_node_count(self) -> int:
        """Get number of phenotypes."""
        self._load_data()
        return len(self._phenotypes)

    def get_edge_count(self) -> int:
        """Get number of associations."""
        self._load_data()
        return len(self._associations)
