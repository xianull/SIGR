"""
BioCypher Adapters for SIGR Knowledge Graph

All adapters follow the official BioCypher format:
- Nodes: yield (node_id, node_label, properties)
- Edges: yield (source_id, target_id, relationship_label, properties)

Adapters:
- HGNCAdapter: Gene nomenclature and basic info
- STRINGAdapter: Protein-protein interactions
- TRRUSTAdapter: TF-target gene regulation
- GOAdapter: Gene Ontology annotations
- HPOAdapter: Gene-phenotype associations
- CellMarkerAdapter: Cell type marker genes
- ReactomeAdapter: Pathway annotations and hierarchy
- OMIMAdapter: Gene-disease associations (new)
- GTExAdapter: Tissue-specific expression (new)
- CORUMAdapter: Protein complex membership (new)
"""

from .hgnc_adapter import HGNCAdapter
from .string_adapter import STRINGAdapter
from .trrust_adapter import TRRUSTAdapter
from .go_adapter import GOAdapter
from .hpo_adapter import HPOAdapter
from .cellmarker_adapter import CellMarkerAdapter
from .reactome_adapter import ReactomeAdapter
from .omim_adapter import OMIMAdapter
from .gtex_adapter import GTExAdapter
from .corum_adapter import CORUMAdapter

__all__ = [
    "HGNCAdapter",
    "STRINGAdapter",
    "TRRUSTAdapter",
    "GOAdapter",
    "HPOAdapter",
    "CellMarkerAdapter",
    "ReactomeAdapter",
    "OMIMAdapter",
    "GTExAdapter",
    "CORUMAdapter",
]
