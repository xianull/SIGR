"""
CORUM Adapter for BioCypher

Loads protein complex data from CORUM database.

Data source: https://mips.helmholtz-muenchen.de/corum/
Output: Complex nodes + IN_COMPLEX edges
"""

import logging
from pathlib import Path
from typing import Generator, Set, Dict, Optional, List

import pandas as pd

logger = logging.getLogger(__name__)


class CORUMAdapter:
    """
    BioCypher adapter for CORUM protein complex data.

    Node type: biolink:MacromolecularComplex
    Edge type: biolink:GeneToMacromolecularComplexAssociation
    Relationship label: IN_COMPLEX

    Expected data file: coreComplexes.txt or allComplexes.txt from CORUM
    """

    def __init__(self, data_path: str, organism: str = 'Human'):
        """
        Initialize CORUM adapter.

        Args:
            data_path: Path to CORUM data file
            organism: Organism to filter for (default: Human)
        """
        self.data_path = Path(data_path)
        self.organism = organism

        self._complexes: Dict[str, Dict] = {}
        self._memberships: list = []
        self._loaded = False

    def _load_data(self) -> None:
        """Load CORUM protein complex data."""
        if self._loaded:
            return

        logger.info(f"Loading CORUM data from {self.data_path}")

        if not self.data_path.exists():
            raise FileNotFoundError(f"CORUM file not found: {self.data_path}")

        # CORUM files are tab-separated
        df = pd.read_csv(self.data_path, sep='\t', low_memory=False)

        # Normalize column names
        df.columns = df.columns.str.strip()

        # Find relevant columns
        complex_id_col = None
        complex_name_col = None
        subunits_col = None
        organism_col = None
        function_col = None
        go_col = None
        pubmed_col = None

        for col in df.columns:
            col_lower = col.lower()
            if 'complexid' in col_lower or 'complex_id' in col_lower or col_lower == 'id':
                complex_id_col = col
            elif 'complexname' in col_lower or 'complex_name' in col_lower or col_lower == 'name':
                complex_name_col = col
            elif 'subunits' in col_lower and 'gene' in col_lower:
                subunits_col = col
            elif 'organism' in col_lower:
                organism_col = col
            elif 'go_description' in col_lower or 'function' in col_lower:
                function_col = col
            elif 'go_id' in col_lower:
                go_col = col
            elif 'pubmed' in col_lower:
                pubmed_col = col

        # If subunits column not found, try alternative names
        if subunits_col is None:
            for col in df.columns:
                col_lower = col.lower()
                if 'subunits' in col_lower or 'components' in col_lower or 'genes' in col_lower:
                    subunits_col = col
                    break

        if complex_id_col is None or subunits_col is None:
            logger.error(f"Required columns not found. Available: {list(df.columns)}")
            raise ValueError("CORUM file missing required columns (ComplexID, subunits)")

        logger.info(f"Using columns: id={complex_id_col}, name={complex_name_col}, "
                   f"subunits={subunits_col}, organism={organism_col}")

        for _, row in df.iterrows():
            # Filter by organism
            if organism_col and pd.notna(row.get(organism_col)):
                org = str(row.get(organism_col)).strip()
                if self.organism.lower() not in org.lower():
                    continue

            complex_id = row.get(complex_id_col)
            if pd.isna(complex_id):
                continue

            complex_id = f"CORUM:{int(complex_id)}" if isinstance(complex_id, (int, float)) else f"CORUM:{complex_id}"

            # Parse subunits
            subunits_str = row.get(subunits_col)
            if pd.isna(subunits_str):
                continue

            subunits = self._parse_subunits(str(subunits_str))
            if not subunits:
                continue

            # Store complex info
            complex_name = row.get(complex_name_col) if complex_name_col else None
            function_desc = row.get(function_col) if function_col else None
            go_ids = row.get(go_col) if go_col else None
            pubmed_ids = row.get(pubmed_col) if pubmed_col else None

            self._complexes[complex_id] = {
                'complex_id': complex_id,
                'name': complex_name if pd.notna(complex_name) else None,
                'function': function_desc if pd.notna(function_desc) else None,
                'go_ids': self._parse_list(go_ids) if pd.notna(go_ids) else [],
                'pubmed_ids': self._parse_list(pubmed_ids) if pd.notna(pubmed_ids) else [],
                'subunit_count': len(subunits),
            }

            # Store memberships
            for gene in subunits:
                self._memberships.append({
                    'gene': gene,
                    'complex_id': complex_id,
                    'complex_name': complex_name if pd.notna(complex_name) else complex_id,
                    'subunit_count': len(subunits),
                })

        self._loaded = True
        logger.info(
            f"Loaded {len(self._complexes)} complexes, "
            f"{len(self._memberships)} memberships"
        )

    def _parse_subunits(self, subunits_str: str) -> List[str]:
        """
        Parse subunits string to list of gene symbols.

        Args:
            subunits_str: String containing gene symbols

        Returns:
            List of gene symbols
        """
        # CORUM uses various separators: ;, ,, or spaces
        # Also may have format like "GENE1(UniprotID);GENE2(UniprotID)"

        # First try semicolon
        if ';' in subunits_str:
            parts = subunits_str.split(';')
        elif ',' in subunits_str:
            parts = subunits_str.split(',')
        else:
            parts = subunits_str.split()

        genes = []
        for part in parts:
            part = part.strip()
            if not part:
                continue

            # Remove UniProt ID in parentheses
            if '(' in part:
                part = part.split('(')[0].strip()

            # Clean up
            gene = part.upper()

            # Skip if it looks like a UniProt ID (starts with letters, has numbers)
            if gene and not gene[0].isdigit():
                genes.append(gene)

        return genes

    def _parse_list(self, value) -> List[str]:
        """Parse a list value that may be separated by ; or ,."""
        if pd.isna(value):
            return []

        value_str = str(value)
        if ';' in value_str:
            return [v.strip() for v in value_str.split(';') if v.strip()]
        elif ',' in value_str:
            return [v.strip() for v in value_str.split(',') if v.strip()]
        else:
            return [value_str.strip()] if value_str.strip() else []

    def get_nodes(self) -> Generator[tuple, None, None]:
        """
        Yield complex nodes in BioCypher format.

        Yields:
            Tuple of (node_id, node_label, properties)
        """
        self._load_data()

        for complex_id, info in self._complexes.items():
            properties = {
                'complex_id': complex_id,
                'source_database': 'CORUM',
                'subunit_count': info['subunit_count'],
            }

            if info.get('name'):
                properties['name'] = info['name']
            if info.get('function'):
                properties['function'] = info['function']
            if info.get('go_ids'):
                properties['go_ids'] = ';'.join(info['go_ids'][:5])  # Limit to avoid too long

            yield (complex_id, 'ProteinComplex', properties)

    def get_edges(self) -> Generator[tuple, None, None]:
        """
        Yield gene-complex membership edges in BioCypher format.

        Yields:
            Tuple of (source_id, target_id, relationship_label, properties)
        """
        self._load_data()

        # Deduplicate
        seen: Set[tuple] = set()

        for membership in self._memberships:
            gene = membership['gene']
            complex_id = membership['complex_id']

            edge_key = (gene, complex_id)
            if edge_key in seen:
                continue
            seen.add(edge_key)

            properties = {
                'source_database': 'CORUM',
                'complex_name': membership.get('complex_name', complex_id),
                'subunit_count': membership.get('subunit_count', 0),
            }

            yield (gene, complex_id, 'IN_COMPLEX', properties)

    def get_node_count(self) -> int:
        """Get number of complexes."""
        self._load_data()
        return len(self._complexes)

    def get_edge_count(self) -> int:
        """Get number of memberships."""
        self._load_data()
        return len(self._memberships)

    def get_complex_genes(self, complex_id: str) -> List[str]:
        """
        Get all genes in a complex.

        Args:
            complex_id: CORUM complex ID

        Returns:
            List of gene symbols
        """
        self._load_data()

        return [
            m['gene'] for m in self._memberships
            if m['complex_id'] == complex_id
        ]

    def get_gene_complexes(self, gene: str) -> List[str]:
        """
        Get all complexes containing a gene.

        Args:
            gene: Gene symbol

        Returns:
            List of complex IDs
        """
        self._load_data()

        gene = gene.upper()
        return list(set(
            m['complex_id'] for m in self._memberships
            if m['gene'] == gene
        ))
