"""
OMIM Adapter for BioCypher

Loads gene-disease associations from Online Mendelian Inheritance in Man.

Data source: https://omim.org/downloads/
Output: Disease nodes + ASSOCIATED_WITH_DISEASE edges
"""

import logging
from pathlib import Path
from typing import Generator, Set, Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class OMIMAdapter:
    """
    BioCypher adapter for OMIM gene-disease associations.

    Node type: biolink:Disease
    Edge type: biolink:GeneToDiseaseAssociation
    Relationship label: ASSOCIATED_WITH_DISEASE

    Expected data file: morbidmap.txt or genemap2.txt from OMIM
    """

    def __init__(self, data_path: str):
        """
        Initialize OMIM adapter.

        Args:
            data_path: Path to OMIM data file (morbidmap.txt or genemap2.txt)
        """
        self.data_path = Path(data_path)

        self._diseases: Dict[str, Dict] = {}
        self._associations: list = []
        self._loaded = False

    def _load_data(self) -> None:
        """Load OMIM gene-disease associations."""
        if self._loaded:
            return

        logger.info(f"Loading OMIM data from {self.data_path}")

        if not self.data_path.exists():
            raise FileNotFoundError(f"OMIM file not found: {self.data_path}")

        # Determine file type based on name
        filename = self.data_path.name.lower()

        if 'morbidmap' in filename:
            self._load_morbidmap()
        elif 'genemap' in filename:
            self._load_genemap()
        else:
            # Try to auto-detect format
            self._load_generic()

        self._loaded = True
        logger.info(
            f"Loaded {len(self._diseases)} diseases, "
            f"{len(self._associations)} associations"
        )

    def _load_morbidmap(self) -> None:
        """Load from morbidmap.txt format."""
        # Format: Phenotype | Gene Symbols | MIM Number | Cyto Location
        with open(self.data_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                parts = line.split('\t')
                if len(parts) < 3:
                    continue

                phenotype = parts[0].strip()
                genes_str = parts[1].strip()
                mim_number = parts[2].strip()

                # Parse phenotype name and inheritance
                phenotype_name, inheritance = self._parse_phenotype(phenotype)

                # Store disease
                disease_id = f"OMIM:{mim_number}"
                if disease_id not in self._diseases:
                    self._diseases[disease_id] = {
                        'omim_id': mim_number,
                        'name': phenotype_name,
                        'inheritance': inheritance,
                    }

                # Parse genes
                genes = [g.strip().upper() for g in genes_str.split(',') if g.strip()]

                for gene in genes:
                    self._associations.append({
                        'gene': gene,
                        'disease_id': disease_id,
                        'inheritance': inheritance,
                        'confidence': 'curated',
                    })

    def _load_genemap(self) -> None:
        """Load from genemap2.txt format."""
        df = pd.read_csv(self.data_path, sep='\t', comment='#', header=None,
                         names=['chromosome', 'genomic_position_start', 'genomic_position_end',
                                'cyto_location', 'computed_cyto_location', 'mim_number',
                                'gene_symbols', 'gene_name', 'approved_gene_symbol',
                                'entrez_gene_id', 'ensembl_gene_id', 'comments',
                                'phenotypes', 'mouse_gene_symbol_id'])

        for _, row in df.iterrows():
            phenotypes_str = row.get('phenotypes')
            gene_symbol = row.get('approved_gene_symbol')

            if pd.isna(phenotypes_str) or pd.isna(gene_symbol):
                continue

            gene = str(gene_symbol).strip().upper()

            # Parse multiple phenotypes
            for pheno_entry in str(phenotypes_str).split(';'):
                pheno_entry = pheno_entry.strip()
                if not pheno_entry:
                    continue

                phenotype_name, inheritance = self._parse_phenotype(pheno_entry)

                # Extract MIM number from phenotype entry
                mim_match = None
                parts = pheno_entry.split(',')
                for part in reversed(parts):
                    part = part.strip()
                    if part.isdigit() and len(part) == 6:
                        mim_match = part
                        break

                if mim_match:
                    disease_id = f"OMIM:{mim_match}"
                else:
                    # Use hash of phenotype name as ID
                    disease_id = f"OMIM:PHENO_{hash(phenotype_name) % 1000000}"

                if disease_id not in self._diseases:
                    self._diseases[disease_id] = {
                        'omim_id': mim_match or 'unknown',
                        'name': phenotype_name,
                        'inheritance': inheritance,
                    }

                self._associations.append({
                    'gene': gene,
                    'disease_id': disease_id,
                    'inheritance': inheritance,
                    'confidence': 'curated',
                })

    def _load_generic(self) -> None:
        """Try to load from generic TSV format."""
        df = pd.read_csv(self.data_path, sep='\t', comment='#')
        df.columns = df.columns.str.lower().str.replace('-', '_').str.replace(' ', '_')

        # Find relevant columns
        gene_col = None
        disease_col = None
        mim_col = None

        for col in df.columns:
            if 'gene' in col and gene_col is None:
                gene_col = col
            elif 'phenotype' in col or 'disease' in col:
                disease_col = col
            elif 'mim' in col or 'omim' in col:
                mim_col = col

        if gene_col is None:
            logger.warning("Could not find gene column in OMIM file")
            return

        for _, row in df.iterrows():
            gene = row.get(gene_col)
            if pd.isna(gene):
                continue

            gene = str(gene).strip().upper()

            disease_name = str(row.get(disease_col, 'Unknown')) if disease_col else 'Unknown'
            mim_id = str(row.get(mim_col, '')) if mim_col else ''

            disease_id = f"OMIM:{mim_id}" if mim_id else f"OMIM:PHENO_{hash(disease_name) % 1000000}"

            if disease_id not in self._diseases:
                self._diseases[disease_id] = {
                    'omim_id': mim_id or 'unknown',
                    'name': disease_name,
                    'inheritance': None,
                }

            self._associations.append({
                'gene': gene,
                'disease_id': disease_id,
                'inheritance': None,
                'confidence': 'curated',
            })

    def _parse_phenotype(self, phenotype_str: str) -> tuple:
        """
        Parse phenotype string to extract name and inheritance pattern.

        Args:
            phenotype_str: Raw phenotype string

        Returns:
            Tuple of (phenotype_name, inheritance)
        """
        inheritance = None

        # Extract inheritance pattern (AD, AR, XL, etc.)
        inheritance_patterns = {
            'AD': 'Autosomal dominant',
            'AR': 'Autosomal recessive',
            'XL': 'X-linked',
            'XLR': 'X-linked recessive',
            'XLD': 'X-linked dominant',
            'YL': 'Y-linked',
            'Mi': 'Mitochondrial',
        }

        for code, full_name in inheritance_patterns.items():
            if f'({code})' in phenotype_str or f', {code}' in phenotype_str:
                inheritance = full_name
                break

        # Clean up phenotype name
        name = phenotype_str.strip()
        # Remove mapping key prefixes like [, {, ?
        for prefix in ['[', '{', '?']:
            if name.startswith(prefix):
                name = name[1:]
        for suffix in [']', '}']:
            if name.endswith(suffix):
                name = name[:-1]

        # Remove trailing MIM number
        parts = name.rsplit(',', 1)
        if len(parts) > 1 and parts[1].strip().isdigit():
            name = parts[0].strip()

        return name.strip(), inheritance

    def get_nodes(self) -> Generator[tuple, None, None]:
        """
        Yield disease nodes in BioCypher format.

        Yields:
            Tuple of (node_id, node_label, properties)
        """
        self._load_data()

        for disease_id, info in self._diseases.items():
            properties = {
                'omim_id': info['omim_id'],
                'source_database': 'OMIM',
            }

            if info.get('name'):
                properties['name'] = info['name']
            if info.get('inheritance'):
                properties['inheritance'] = info['inheritance']

            yield (disease_id, 'Disease', properties)

    def get_edges(self) -> Generator[tuple, None, None]:
        """
        Yield gene-disease association edges in BioCypher format.

        Yields:
            Tuple of (source_id, target_id, relationship_label, properties)
        """
        self._load_data()

        # Deduplicate
        seen: Set[tuple] = set()

        for assoc in self._associations:
            gene = assoc['gene']
            disease_id = assoc['disease_id']

            edge_key = (gene, disease_id)
            if edge_key in seen:
                continue
            seen.add(edge_key)

            properties = {
                'source_database': 'OMIM',
                'confidence': assoc.get('confidence', 'curated'),
            }

            if assoc.get('inheritance'):
                properties['inheritance'] = assoc['inheritance']

            yield (gene, disease_id, 'ASSOCIATED_WITH_DISEASE', properties)

    def get_node_count(self) -> int:
        """Get number of diseases."""
        self._load_data()
        return len(self._diseases)

    def get_edge_count(self) -> int:
        """Get number of associations."""
        self._load_data()
        return len(self._associations)
