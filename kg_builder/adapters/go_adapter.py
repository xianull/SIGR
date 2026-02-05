"""
GO Adapter for BioCypher

Loads Gene Ontology annotations from GAF (Gene Association Format) files.
Enriches GO terms with names and definitions from OBO ontology file.

Data source: http://geneontology.org/
Output: BiologicalProcess/MolecularFunction/CellularComponent nodes + PARTICIPATES_IN edges
"""

import gzip
import logging
from pathlib import Path
from typing import Generator, Set, Dict, Optional

logger = logging.getLogger(__name__)


def parse_obo_file(obo_path: str) -> Dict[str, Dict]:
    """
    Parse GO OBO file to extract term names and definitions.

    Args:
        obo_path: Path to go-basic.obo file

    Returns:
        Dictionary mapping GO ID to {name, namespace, definition}
    """
    obo_path = Path(obo_path)
    if not obo_path.exists():
        logger.warning(f"OBO file not found: {obo_path}")
        return {}

    logger.info(f"Parsing GO ontology from {obo_path}")

    go_terms = {}
    current_term = None

    with open(obo_path, 'r') as f:
        for line in f:
            line = line.strip()

            if line == '[Term]':
                current_term = {}
            elif line == '' or line.startswith('['):
                # End of term or start of non-Term stanza
                if current_term and 'id' in current_term:
                    go_id = current_term['id']
                    go_terms[go_id] = {
                        'name': current_term.get('name'),
                        'namespace': current_term.get('namespace'),
                        'definition': current_term.get('def'),
                    }
                current_term = None
            elif current_term is not None:
                if line.startswith('id: '):
                    current_term['id'] = line[4:]
                elif line.startswith('name: '):
                    current_term['name'] = line[6:]
                elif line.startswith('namespace: '):
                    current_term['namespace'] = line[11:]
                elif line.startswith('def: '):
                    # Extract definition text (between first pair of quotes)
                    def_text = line[5:]
                    if def_text.startswith('"'):
                        end_quote = def_text.find('"', 1)
                        if end_quote > 0:
                            current_term['def'] = def_text[1:end_quote]
                elif line.startswith('is_obsolete: true'):
                    current_term['obsolete'] = True

    # Handle last term
    if current_term and 'id' in current_term:
        go_id = current_term['id']
        go_terms[go_id] = {
            'name': current_term.get('name'),
            'namespace': current_term.get('namespace'),
            'definition': current_term.get('def'),
        }

    logger.info(f"Parsed {len(go_terms)} GO terms from OBO file")
    return go_terms


class GOAdapter:
    """
    BioCypher adapter for Gene Ontology annotations.

    Node type: biolink:BiologicalProcess / MolecularFunction / CellularComponent
    Edge type: biolink:ParticipatesIn
    Relationship label: PARTICIPATES_IN
    """

    # GO namespace mapping (GAF aspect code to namespace)
    NAMESPACE_MAP = {
        'P': 'biological_process',
        'F': 'molecular_function',
        'C': 'cellular_component',
    }

    # Node label mapping based on namespace
    NODE_LABEL_MAP = {
        'biological_process': 'BiologicalProcess',
        'molecular_function': 'MolecularFunction',
        'cellular_component': 'CellularComponent',
    }

    def __init__(
        self,
        gaf_path: str,
        namespaces: Optional[list] = None,
        obo_path: Optional[str] = None,
    ):
        """
        Initialize GO adapter.

        Args:
            gaf_path: Path to goa_human.gaf.gz
            namespaces: List of namespaces to include ('P', 'F', 'C')
                       Default: ['P'] (biological process only)
            obo_path: Path to go-basic.obo for GO term names/definitions
                     If not provided, will try to find it in same directory as gaf_path
        """
        self.gaf_path = Path(gaf_path)
        self.namespaces = namespaces or ['P']

        # Auto-detect OBO path if not provided
        if obo_path:
            self.obo_path = Path(obo_path)
        else:
            # Try common locations
            parent_dir = self.gaf_path.parent
            for candidate in ['go-basic.obo', 'go.obo']:
                candidate_path = parent_dir / candidate
                if candidate_path.exists():
                    self.obo_path = candidate_path
                    break
            else:
                self.obo_path = None

        self._go_terms: Dict[str, Dict] = {}
        self._go_ontology: Dict[str, Dict] = {}  # From OBO file
        self._annotations: list = []
        self._loaded = False

    def _load_ontology(self) -> None:
        """Load GO ontology from OBO file."""
        if self.obo_path and self.obo_path.exists():
            self._go_ontology = parse_obo_file(str(self.obo_path))
        else:
            logger.warning("No OBO file available - GO terms will not have names")
            self._go_ontology = {}

    def _load_data(self) -> None:
        """Load and parse GAF file."""
        if self._loaded:
            return

        # First load ontology for names
        self._load_ontology()

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

                # Store GO term with ontology info
                if go_id not in self._go_terms:
                    ontology_info = self._go_ontology.get(go_id, {})
                    self._go_terms[go_id] = {
                        'go_id': go_id,
                        'namespace': self.NAMESPACE_MAP.get(aspect, aspect),
                        'name': ontology_info.get('name'),
                        'definition': ontology_info.get('definition'),
                    }

                # Store annotation
                self._annotations.append({
                    'gene': gene_symbol,
                    'go_id': go_id,
                    'evidence_code': evidence_code,
                    'qualifier': qualifier,
                })

        self._loaded = True

        # Count terms with names
        terms_with_names = sum(1 for t in self._go_terms.values() if t.get('name'))
        logger.info(
            f"Loaded {len(self._go_terms)} GO terms ({terms_with_names} with names), "
            f"{len(self._annotations)} annotations"
        )

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
                'name': info.get('name'),
                'definition': info.get('definition'),
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
