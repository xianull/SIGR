"""
GO Adapter for BioCypher

Loads Gene Ontology annotations from GAF (Gene Association Format) files.

Data source: http://geneontology.org/
Output: BiologicalProcess nodes + PARTICIPATES_IN edges
"""

import gzip
import logging
from pathlib import Path
from typing import Generator, Set, Dict, Optional

logger = logging.getLogger(__name__)


class GOAdapter:
    """
    BioCypher adapter for Gene Ontology annotations.

    Node type: biolink:BiologicalProcess
    Edge type: biolink:ParticipatesIn
    Relationship label: PARTICIPATES_IN
    """

    # GO namespace mapping
    NAMESPACE_MAP = {
        'P': 'biological_process',
        'F': 'molecular_function',
        'C': 'cellular_component',
    }

    def __init__(
        self,
        gaf_path: str,
        namespaces: Optional[list] = None,
    ):
        """
        Initialize GO adapter.

        Args:
            gaf_path: Path to goa_human.gaf.gz
            namespaces: List of namespaces to include ('P', 'F', 'C')
                       Default: ['P'] (biological process only)
        """
        self.gaf_path = Path(gaf_path)
        self.namespaces = namespaces or ['P']

        self._go_terms: Dict[str, Dict] = {}
        self._annotations: list = []
        self._loaded = False

    def _load_data(self) -> None:
        """Load and parse GAF file."""
        if self._loaded:
            return

        logger.info(f"Loading GO annotations from {self.gaf_path}")

        if not self.gaf_path.exists():
            raise FileNotFoundError(f"GO GAF file not found: {self.gaf_path}")

        # Parse GAF format
        # Columns: DB, DB_ID, Symbol, Qualifier, GO_ID, Reference, Evidence, With, Aspect, Name, Synonym, Type, Taxon, Date, Assigned_by
        with gzip.open(self.gaf_path, 'rt') as f:
            for line in f:
                if line.startswith('!'):
                    continue

                fields = line.strip().split('\t')
                if len(fields) < 15:
                    continue

                gene_symbol = fields[2].upper()
                go_id = fields[4]
                evidence_code = fields[6]
                aspect = fields[8]  # P, F, or C
                qualifier = fields[3]

                # Filter by namespace
                if aspect not in self.namespaces:
                    continue

                # Store GO term
                if go_id not in self._go_terms:
                    self._go_terms[go_id] = {
                        'go_id': go_id,
                        'namespace': self.NAMESPACE_MAP.get(aspect, aspect),
                    }

                # Store annotation
                self._annotations.append({
                    'gene': gene_symbol,
                    'go_id': go_id,
                    'evidence_code': evidence_code,
                    'qualifier': qualifier,
                })

        self._loaded = True
        logger.info(
            f"Loaded {len(self._go_terms)} GO terms, "
            f"{len(self._annotations)} annotations"
        )

    # Node label mapping based on namespace
    NODE_LABEL_MAP = {
        'biological_process': 'BiologicalProcess',
        'molecular_function': 'MolecularFunction',
        'cellular_component': 'CellularComponent',
    }

    def get_nodes(self) -> Generator[tuple, None, None]:
        """
        Yield GO term nodes in BioCypher format.

        Yields:
            Tuple of (node_id, node_label, properties)
        """
        self._load_data()

        for go_id, info in self._go_terms.items():
            namespace = info['namespace']
            node_label = self.NODE_LABEL_MAP.get(namespace, 'GOTerm')

            properties = {
                'go_id': go_id,
                'namespace': namespace,
                'source_database': 'GeneOntology',
            }

            yield (go_id, node_label, properties)

    def get_edges(self) -> Generator[tuple, None, None]:
        """
        Yield gene-GO annotation edges in BioCypher format.

        Yields:
            Tuple of (source_id, target_id, relationship_label, properties)
        """
        self._load_data()

        # Deduplicate gene-GO pairs
        seen: Set[tuple] = set()

        for annot in self._annotations:
            gene = annot['gene']
            go_id = annot['go_id']

            edge_key = (gene, go_id)
            if edge_key in seen:
                continue
            seen.add(edge_key)

            properties = {
                'evidence_code': annot['evidence_code'],
                'qualifier': annot['qualifier'] if annot['qualifier'] else None,
                'source_database': 'GeneOntology',
            }

            # Remove None values
            properties = {k: v for k, v in properties.items() if v is not None}

            yield (gene, go_id, 'PARTICIPATES_IN', properties)

    def get_node_count(self) -> int:
        """Get number of GO terms."""
        self._load_data()
        return len(self._go_terms)

    def get_edge_count(self) -> int:
        """Get number of annotations."""
        self._load_data()
        return len(self._annotations)
