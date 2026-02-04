"""
Reactome Adapter for BioCypher

Loads pathway information from Reactome database.

Data source: https://reactome.org/
Files:
- ReactomePathways.txt: Pathway definitions
- NCBI2Reactome.txt: Gene-pathway associations (via NCBI/Entrez ID)
- ReactomePathwaysRelation.txt: Pathway hierarchy

Output:
- Pathway nodes (biolink:Pathway)
- IN_PATHWAY edges (biolink:InPathway) - Gene participates in pathway
- PATHWAY_CONTAINS edges - Pathway hierarchy (parent contains child)
"""

import logging
from pathlib import Path
from typing import Generator, Set, Dict, Optional, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)


class ReactomeAdapter:
    """
    BioCypher adapter for Reactome pathway database.

    Node type: biolink:Pathway
    Edge types:
        - biolink:InPathway (gene to pathway)
        - PATHWAY_CONTAINS (pathway hierarchy)
    """

    SPECIES_PREFIX = 'R-HSA'  # Human pathways

    def __init__(
        self,
        pathways_path: str,
        gene_pathway_path: str,
        pathway_relations_path: Optional[str] = None,
        hgnc_path: Optional[str] = None,
        include_hierarchy: bool = True,
    ):
        """
        Initialize Reactome adapter.

        Args:
            pathways_path: Path to ReactomePathways.txt
            gene_pathway_path: Path to NCBI2Reactome.txt
            pathway_relations_path: Path to ReactomePathwaysRelation.txt (optional)
            hgnc_path: Path to HGNC file for Entrez->Symbol mapping (optional)
            include_hierarchy: Whether to include pathway hierarchy edges
        """
        self.pathways_path = Path(pathways_path)
        self.gene_pathway_path = Path(gene_pathway_path)
        self.pathway_relations_path = Path(pathway_relations_path) if pathway_relations_path else None
        self.hgnc_path = Path(hgnc_path) if hgnc_path else None
        self.include_hierarchy = include_hierarchy

        # Data storage
        self._pathways: Dict[str, Dict] = {}  # pathway_id -> info
        self._gene_pathway_edges: list = []   # (gene_symbol, pathway_id, evidence)
        self._pathway_hierarchy: list = []     # (parent_id, child_id)
        self._entrez_to_symbol: Dict[str, str] = {}

        self._loaded = False

    def _load_entrez_mapping(self) -> None:
        """Load Entrez ID to HGNC Symbol mapping from HGNC file."""
        if self.hgnc_path is None or not self.hgnc_path.exists():
            logger.warning("HGNC file not provided or not found, will skip unmapped genes")
            return

        import pandas as pd
        logger.info(f"Loading Entrez->Symbol mapping from {self.hgnc_path}")

        df = pd.read_csv(self.hgnc_path, sep='\t', low_memory=False)
        for _, row in df.iterrows():
            symbol = row.get('symbol')
            entrez = row.get('entrez_id')
            if pd.notna(symbol) and pd.notna(entrez):
                entrez_str = str(int(entrez))
                self._entrez_to_symbol[entrez_str] = str(symbol).strip().upper()

        logger.info(f"Loaded {len(self._entrez_to_symbol)} Entrez->Symbol mappings")

    def _load_pathways(self) -> None:
        """Load pathway definitions from ReactomePathways.txt."""
        if not self.pathways_path.exists():
            raise FileNotFoundError(f"Reactome pathways file not found: {self.pathways_path}")

        logger.info(f"Loading pathways from {self.pathways_path}")

        with open(self.pathways_path, 'r', encoding='utf-8') as f:
            for line in f:
                fields = line.strip().split('\t')
                if len(fields) < 3:
                    continue

                pathway_id = fields[0]
                pathway_name = fields[1]
                species = fields[2]

                # Only include human pathways
                if not pathway_id.startswith(self.SPECIES_PREFIX):
                    continue

                self._pathways[pathway_id] = {
                    'reactome_id': pathway_id,
                    'name': pathway_name,
                    'species': species,
                }

        logger.info(f"Loaded {len(self._pathways)} human pathways")

    def _load_gene_pathway_associations(self) -> None:
        """Load gene-pathway associations from NCBI2Reactome.txt."""
        if not self.gene_pathway_path.exists():
            raise FileNotFoundError(f"Gene-pathway file not found: {self.gene_pathway_path}")

        logger.info(f"Loading gene-pathway associations from {self.gene_pathway_path}")

        # Track associations for deduplication
        seen_edges: Set[Tuple[str, str]] = set()

        with open(self.gene_pathway_path, 'r', encoding='utf-8') as f:
            for line in f:
                fields = line.strip().split('\t')
                if len(fields) < 6:
                    continue

                entrez_id = fields[0]
                pathway_id = fields[1]
                # fields[2] is URL
                pathway_name = fields[3]
                evidence = fields[4]  # TAS or IEA
                species = fields[5]

                # Only include human pathways
                if species != 'Homo sapiens':
                    continue

                if not pathway_id.startswith(self.SPECIES_PREFIX):
                    continue

                # Convert Entrez ID to HGNC Symbol
                gene_symbol = self._entrez_to_symbol.get(entrez_id)
                if gene_symbol is None:
                    continue

                # Deduplicate
                edge_key = (gene_symbol, pathway_id)
                if edge_key in seen_edges:
                    continue
                seen_edges.add(edge_key)

                self._gene_pathway_edges.append({
                    'gene': gene_symbol,
                    'pathway_id': pathway_id,
                    'evidence': evidence,
                })

                # Ensure pathway exists in our dictionary
                if pathway_id not in self._pathways:
                    self._pathways[pathway_id] = {
                        'reactome_id': pathway_id,
                        'name': pathway_name,
                        'species': 'Homo sapiens',
                    }

        logger.info(f"Loaded {len(self._gene_pathway_edges)} gene-pathway associations")

    def _load_pathway_hierarchy(self) -> None:
        """Load pathway hierarchy from ReactomePathwaysRelation.txt."""
        if self.pathway_relations_path is None or not self.pathway_relations_path.exists():
            logger.info("Pathway relations file not found, skipping hierarchy")
            return

        logger.info(f"Loading pathway hierarchy from {self.pathway_relations_path}")

        with open(self.pathway_relations_path, 'r', encoding='utf-8') as f:
            for line in f:
                fields = line.strip().split('\t')
                if len(fields) < 2:
                    continue

                parent_id = fields[0]
                child_id = fields[1]

                # Only include human pathways
                if not parent_id.startswith(self.SPECIES_PREFIX):
                    continue
                if not child_id.startswith(self.SPECIES_PREFIX):
                    continue

                self._pathway_hierarchy.append({
                    'parent': parent_id,
                    'child': child_id,
                })

        logger.info(f"Loaded {len(self._pathway_hierarchy)} pathway hierarchy relations")

    def _load_data(self) -> None:
        """Load all Reactome data."""
        if self._loaded:
            return

        # Load Entrez mapping first
        self._load_entrez_mapping()

        # Load pathways
        self._load_pathways()

        # Load gene-pathway associations
        self._load_gene_pathway_associations()

        # Load hierarchy if requested
        if self.include_hierarchy:
            self._load_pathway_hierarchy()

        self._loaded = True

    def get_nodes(self) -> Generator[tuple, None, None]:
        """
        Yield Pathway nodes in BioCypher format.

        Yields:
            Tuple of (node_id, node_label, properties)
        """
        self._load_data()

        for pathway_id, info in self._pathways.items():
            properties = {
                'reactome_id': pathway_id,
                'name': info['name'],
                'species': info.get('species', 'Homo sapiens'),
                'source_database': 'Reactome',
            }

            yield (pathway_id, 'Pathway', properties)

    def get_edges(self) -> Generator[tuple, None, None]:
        """
        Yield edges in BioCypher format.

        Yields:
            Tuple of (source_id, target_id, relationship_label, properties)
        """
        self._load_data()

        # Gene-Pathway edges (gene IN_PATHWAY pathway)
        for edge in self._gene_pathway_edges:
            properties = {
                'evidence': edge['evidence'],
                'source_database': 'Reactome',
            }

            yield (edge['gene'], edge['pathway_id'], 'IN_PATHWAY', properties)

        # Pathway hierarchy edges (parent CONTAINS child)
        if self.include_hierarchy:
            for rel in self._pathway_hierarchy:
                properties = {
                    'source_database': 'Reactome',
                }

                yield (rel['parent'], rel['child'], 'PATHWAY_CONTAINS', properties)

    def get_node_count(self) -> int:
        """Get number of pathways."""
        self._load_data()
        return len(self._pathways)

    def get_gene_pathway_edge_count(self) -> int:
        """Get number of gene-pathway associations."""
        self._load_data()
        return len(self._gene_pathway_edges)

    def get_hierarchy_edge_count(self) -> int:
        """Get number of pathway hierarchy relations."""
        self._load_data()
        return len(self._pathway_hierarchy)

    def get_stats(self) -> Dict:
        """Get statistics about loaded data."""
        self._load_data()

        # Count genes per pathway
        genes_per_pathway = defaultdict(int)
        for edge in self._gene_pathway_edges:
            genes_per_pathway[edge['pathway_id']] += 1

        # Count unique genes
        unique_genes = set(edge['gene'] for edge in self._gene_pathway_edges)

        return {
            'num_pathways': len(self._pathways),
            'num_gene_pathway_edges': len(self._gene_pathway_edges),
            'num_hierarchy_edges': len(self._pathway_hierarchy),
            'num_unique_genes': len(unique_genes),
            'avg_genes_per_pathway': sum(genes_per_pathway.values()) / len(genes_per_pathway) if genes_per_pathway else 0,
            'max_genes_per_pathway': max(genes_per_pathway.values()) if genes_per_pathway else 0,
        }
