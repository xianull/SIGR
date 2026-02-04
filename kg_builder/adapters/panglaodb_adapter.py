"""
PanglaoDB Adapter

Loads cell type marker gene associations from PanglaoDB database.

Data source: https://panglaodb.se/
Format: TSV file with marker information
"""

import logging
from pathlib import Path
from typing import Generator, Dict, Any, Set, Optional
import gzip

import pandas as pd

logger = logging.getLogger(__name__)


class PanglaoDBAdapter:
    """
    Adapter for PanglaoDB database.

    PanglaoDB provides single-cell RNA sequencing derived
    cell type marker genes.

    Node type: CellType
    Edge type: marker_of_cell_type
    - Source: Gene (HGNC Symbol)
    - Target: CellType
    - Properties: organ, species, sensitivity, specificity
    """

    def __init__(self, data_path: str, species: str = "Hs"):
        """
        Initialize PanglaoDB adapter.

        Args:
            data_path: Path to PanglaoDB TSV file (can be gzipped)
            species: Filter by species code (Hs=Human, Mm=Mouse)
        """
        self.data_path = Path(data_path)
        self.species = species
        self._df: Optional[pd.DataFrame] = None
        self._cell_types: Set[str] = set()
        self._genes: Set[str] = set()

    def _load_data(self) -> None:
        """Load PanglaoDB data file."""
        if self._df is not None:
            return

        logger.info(f"Loading PanglaoDB data from {self.data_path}")

        if not self.data_path.exists():
            raise FileNotFoundError(
                f"PanglaoDB data file not found: {self.data_path}\n"
                f"Download from: https://panglaodb.se/markers.html"
            )

        # Handle gzipped files
        if str(self.data_path).endswith('.gz'):
            self._df = pd.read_csv(self.data_path, sep='\t', compression='gzip')
        else:
            self._df = pd.read_csv(self.data_path, sep='\t')

        # Standardize column names
        column_map = {
            'species': 'species',
            'official gene symbol': 'gene_symbol',
            'cell type': 'cell_type',
            'organ': 'organ',
            'sensitivity_human': 'sensitivity',
            'specificity_human': 'specificity',
            'sensitivity_mouse': 'sensitivity_mouse',
            'specificity_mouse': 'specificity_mouse',
        }

        # Rename columns (case-insensitive matching)
        rename_map = {}
        for col in self._df.columns:
            col_lower = col.lower()
            for orig, new in column_map.items():
                if col_lower == orig.lower():
                    rename_map[col] = new
                    break
        self._df = self._df.rename(columns=rename_map)

        # Filter by species
        if 'species' in self._df.columns:
            # Species column may contain values like "Hs Mm" for both
            self._df = self._df[
                self._df['species'].str.contains(self.species, case=False, na=False)
            ]

        # Clean gene symbols
        if 'gene_symbol' in self._df.columns:
            self._df['gene_symbol'] = self._df['gene_symbol'].str.strip().str.upper()

        # Extract unique cell types and genes
        if 'cell_type' in self._df.columns:
            self._cell_types = set(self._df['cell_type'].dropna().unique())
        if 'gene_symbol' in self._df.columns:
            self._genes = set(self._df['gene_symbol'].dropna().unique())

        logger.info(
            f"Loaded {len(self._df)} markers, "
            f"{len(self._cell_types)} cell types, {len(self._genes)} genes"
        )

    def get_cell_type_nodes(self) -> Generator[Dict[str, Any], None, None]:
        """
        Yield CellType node data.

        Yields:
            Dict with node_id, node_type, and properties
        """
        self._load_data()

        for cell_type in self._cell_types:
            subset = self._df[self._df['cell_type'] == cell_type]

            # Get associated organs
            organs = subset['organ'].dropna().unique().tolist() if 'organ' in subset.columns else []
            primary_organ = organs[0] if organs else None

            yield {
                'node_id': f"PanglaoDB:{cell_type}",
                'node_type': 'CellType',
                'properties': {
                    'name': cell_type,
                    'tissue_origin': primary_organ,
                    'source': 'PanglaoDB',
                }
            }

    def get_edges(self) -> Generator[Dict[str, Any], None, None]:
        """
        Yield marker-cell type association edges.

        Yields:
            Dict with source, target, edge_type, and properties
        """
        self._load_data()

        # Track seen edges to avoid duplicates
        seen = set()

        for _, row in self._df.iterrows():
            gene = row.get('gene_symbol')
            cell_type = row.get('cell_type')

            if pd.isna(gene) or pd.isna(cell_type):
                continue

            edge_key = (gene, cell_type)
            if edge_key in seen:
                continue
            seen.add(edge_key)

            # Get sensitivity and specificity scores
            sensitivity = row.get('sensitivity')
            specificity = row.get('specificity')
            organ = row.get('organ')

            properties = {
                'source_database': 'PanglaoDB',
            }

            if pd.notna(sensitivity):
                properties['sensitivity'] = float(sensitivity)
            if pd.notna(specificity):
                properties['specificity'] = float(specificity)
            if pd.notna(organ):
                properties['organ'] = organ

            yield {
                'source': gene,
                'target': f"PanglaoDB:{cell_type}",
                'edge_type': 'marker_of_cell_type',
                'properties': properties
            }

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the loaded data.

        Returns:
            Dict with counts and distributions
        """
        self._load_data()

        organ_counts = {}
        if 'organ' in self._df.columns:
            organ_counts = self._df['organ'].value_counts().head(20).to_dict()

        return {
            'total_markers': len(self._df),
            'unique_cell_types': len(self._cell_types),
            'unique_genes': len(self._genes),
            'top_organs': organ_counts,
        }

    def get_all_genes(self) -> Set[str]:
        """
        Get all marker gene symbols.

        Returns:
            Set of gene symbols
        """
        self._load_data()
        return self._genes

    def get_cell_type_markers(self, cell_type: str) -> Set[str]:
        """
        Get marker genes for a cell type.

        Args:
            cell_type: Cell type name

        Returns:
            Set of marker gene symbols
        """
        self._load_data()
        genes = self._df[self._df['cell_type'] == cell_type]['gene_symbol']
        return set(genes.dropna().unique())
