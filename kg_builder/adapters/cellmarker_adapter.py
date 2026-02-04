"""
CellMarker Adapter for BioCypher

Loads cell type marker gene associations from CellMarker 2.0 database.

Data source: http://biocc.hrbmu.edu.cn/CellMarker/
Output: CellType nodes + MARKER_OF edges
"""

import logging
from pathlib import Path
from typing import Generator, Set, Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class CellMarkerAdapter:
    """
    BioCypher adapter for CellMarker cell type markers.

    Node type: biolink:Cell
    Edge type: biolink:ExpressedIn
    Relationship label: MARKER_OF
    """

    def __init__(
        self,
        data_path: str,
        species: str = "Human",
        cancer_type: Optional[str] = None,
    ):
        """
        Initialize CellMarker adapter.

        Args:
            data_path: Path to Cell_marker_Human.xlsx
            species: Filter by species (default: Human)
            cancer_type: Filter by cancer type (default: None = all)
        """
        self.data_path = Path(data_path)
        self.species = species
        self.cancer_type = cancer_type

        self._cell_types: Dict[str, Dict] = {}
        self._markers: list = []
        self._loaded = False

    def _load_data(self) -> None:
        """Load CellMarker data from Excel file."""
        if self._loaded:
            return

        logger.info(f"Loading CellMarker data from {self.data_path}")

        if not self.data_path.exists():
            raise FileNotFoundError(f"CellMarker file not found: {self.data_path}")

        # Read Excel file
        df = pd.read_excel(self.data_path)

        # Filter by species
        if 'species' in df.columns:
            df = df[df['species'] == self.species]

        # Filter by cancer type
        if self.cancer_type and 'cancer_type' in df.columns:
            df = df[df['cancer_type'] == self.cancer_type]

        # Process each row
        for _, row in df.iterrows():
            # Get gene symbol
            gene_symbol = row.get('Symbol') or row.get('marker')
            if pd.isna(gene_symbol):
                continue
            gene_symbol = str(gene_symbol).strip().upper()

            # Get cell type
            cell_name = row.get('cell_name') or row.get('cellName')
            if pd.isna(cell_name):
                continue
            cell_name = str(cell_name).strip()

            # Get tissue type
            tissue_type = row.get('tissue_type') or row.get('tissueType')
            tissue_type = str(tissue_type).strip() if pd.notna(tissue_type) else None

            # Get cell ontology ID
            cell_ontology_id = row.get('cellontology_id') or row.get('CellOntologyID')
            cell_ontology_id = str(cell_ontology_id).strip() if pd.notna(cell_ontology_id) else None

            # Get PMID
            pmid = row.get('PMID')
            pmid = str(int(pmid)) if pd.notna(pmid) else None

            # Store cell type
            if cell_name not in self._cell_types:
                self._cell_types[cell_name] = {
                    'name': cell_name,
                    'tissue_origin': tissue_type,
                    'cell_ontology_id': cell_ontology_id,
                }

            # Store marker association
            self._markers.append({
                'gene': gene_symbol,
                'cell_type': cell_name,
                'tissue_type': tissue_type,
                'pmid': pmid,
            })

        self._loaded = True
        logger.info(
            f"Loaded {len(self._cell_types)} cell types, "
            f"{len(self._markers)} marker associations"
        )

    def get_nodes(self) -> Generator[tuple, None, None]:
        """
        Yield cell type nodes in BioCypher format.

        Yields:
            Tuple of (node_id, node_label, properties)
        """
        self._load_data()

        for cell_name, info in self._cell_types.items():
            properties = {
                'name': cell_name,
                'source_database': 'CellMarker',
            }

            if info.get('tissue_origin'):
                properties['tissue_origin'] = info['tissue_origin']
            if info.get('cell_ontology_id'):
                properties['cell_ontology_id'] = info['cell_ontology_id']

            # Create a clean node ID
            node_id = f"CellMarker:{cell_name.replace(' ', '_')}"

            yield (node_id, 'CellType', properties)

    def get_edges(self) -> Generator[tuple, None, None]:
        """
        Yield gene-cell type marker edges in BioCypher format.

        Yields:
            Tuple of (source_id, target_id, relationship_label, properties)
        """
        self._load_data()

        # Deduplicate
        seen: Set[tuple] = set()

        for marker in self._markers:
            gene = marker['gene']
            cell_type = marker['cell_type']
            cell_type_id = f"CellMarker:{cell_type.replace(' ', '_')}"

            edge_key = (gene, cell_type_id)
            if edge_key in seen:
                continue
            seen.add(edge_key)

            properties = {
                'source_database': 'CellMarker',
            }

            if marker.get('tissue_type'):
                properties['tissue_type'] = marker['tissue_type']
            if marker.get('pmid'):
                properties['pubmed_id'] = marker['pmid']

            yield (gene, cell_type_id, 'MARKER_OF', properties)

    def get_node_count(self) -> int:
        """Get number of cell types."""
        self._load_data()
        return len(self._cell_types)

    def get_edge_count(self) -> int:
        """Get number of marker associations."""
        self._load_data()
        return len(self._markers)
