"""
SIGR Knowledge Graph Builder using Official BioCypher Framework

Constructs a biomedical knowledge graph using BioCypher with Biolink Model ontology.

Usage:
    python build_kg.py [--string-threshold 700] [--output data/kg/sigr_kg.pkl]
    python build_kg.py --include-omim --include-gtex --include-corum
"""

import os
import sys
import logging
import pickle
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Generator

import networkx as nx
from tqdm import tqdm

# BioCypher imports
from biocypher import BioCypher

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import adapters
from kg_builder.adapters import (
    HGNCAdapter,
    STRINGAdapter,
    TRRUSTAdapter,
    GOAdapter,
    HPOAdapter,
    CellMarkerAdapter,
    ReactomeAdapter,
    OMIMAdapter,
    GTExAdapter,
    CORUMAdapter,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SIGRKnowledgeGraph:
    """
    SIGR Knowledge Graph Builder using BioCypher.

    Integrates multiple biological databases following Biolink model:
    - HGNC: Gene nomenclature (biolink:Gene)
    - STRING: Protein interactions (biolink:PairwiseMolecularInteraction)
    - TRRUST: TF regulation (biolink:Regulates)
    - GO: Biological processes (biolink:ParticipatesIn)
    - HPO: Phenotypes (biolink:GeneToPhenotypicFeatureAssociation)
    - CellMarker: Cell type markers (biolink:ExpressedIn)
    - Reactome: Pathway annotations (biolink:InPathway)
    - OMIM: Disease associations (biolink:GeneToDiseaseAssociation) [optional]
    - GTEx: Tissue expression (biolink:GeneToExpressionSiteAssociation) [optional]
    - CORUM: Protein complex membership (biolink:GeneToMacromolecularComplexAssociation) [optional]
    """

    def __init__(
        self,
        data_dir: str = "data/raw",
        output_dir: str = "data/kg",
        string_threshold: int = 700,
        include_omim: bool = False,
        include_gtex: bool = False,
        include_corum: bool = False,
        gtex_tpm_threshold: float = 1.0,
    ):
        """
        Initialize the KG builder.

        Args:
            data_dir: Directory containing raw data files
            output_dir: Directory for output files
            string_threshold: STRING score threshold (0-1000)
            include_omim: Include OMIM disease associations
            include_gtex: Include GTEx tissue expression
            include_corum: Include CORUM protein complex data
            gtex_tpm_threshold: TPM threshold for GTEx expression
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.string_threshold = string_threshold
        self.include_omim = include_omim
        self.include_gtex = include_gtex
        self.include_corum = include_corum
        self.gtex_tpm_threshold = gtex_tpm_threshold

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize BioCypher
        config_dir = Path(__file__).parent / "config"
        self.bc = BioCypher(
            biocypher_config_path=str(config_dir / "biocypher_config.yaml"),
            schema_config_path=str(config_dir / "schema_config.yaml"),
        )

        # Statistics
        self.stats = defaultdict(int)

        # Gene set for validation
        self.valid_genes = set()

    def _get_data_path(self, filename: str) -> Path:
        """Get full path to a data file."""
        return self.data_dir / filename

    def _load_valid_genes(self) -> None:
        """Load the set of valid HGNC gene symbols."""
        logger.info("Loading valid gene symbols from HGNC...")
        hgnc_adapter = HGNCAdapter(str(self._get_data_path("hgnc_complete_set.txt")))

        for node_id, _, _ in hgnc_adapter.get_nodes():
            self.valid_genes.add(node_id)

        logger.info(f"Loaded {len(self.valid_genes)} valid gene symbols")

    def _add_gene_nodes(self) -> None:
        """Add gene nodes from HGNC."""
        logger.info("Adding gene nodes from HGNC...")

        hgnc_adapter = HGNCAdapter(str(self._get_data_path("hgnc_complete_set.txt")))

        nodes = list(tqdm(hgnc_adapter.get_nodes(), desc="Gene nodes"))
        self.bc.add(nodes)

        self.stats['gene_nodes'] = len(nodes)
        logger.info(f"Added {len(nodes)} gene nodes")

    def _add_string_edges(self) -> None:
        """Add protein-protein interaction edges from STRING."""
        logger.info(f"Adding STRING edges (threshold={self.string_threshold})...")

        try:
            string_adapter = STRINGAdapter(
                links_path=str(self._get_data_path("9606.protein.links.v12.0.txt.gz")),
                aliases_path=str(self._get_data_path("9606.protein.aliases.v12.0.txt.gz")),
                score_threshold=self.string_threshold,
            )

            edges = list(tqdm(string_adapter.get_edges(), desc="STRING edges"))
            self.bc.add(edges)

            self.stats['ppi_edges'] = len(edges)
            logger.info(f"Added {len(edges)} PPI edges")

        except FileNotFoundError as e:
            logger.warning(f"STRING data not found: {e}")

    def _add_trrust_edges(self) -> None:
        """Add TF-target regulation edges from TRRUST."""
        logger.info("Adding TRRUST TF-target edges...")

        try:
            trrust_adapter = TRRUSTAdapter(
                str(self._get_data_path("trrust_rawdata.human.tsv"))
            )

            edges = list(tqdm(trrust_adapter.get_edges(), desc="TRRUST edges"))
            self.bc.add(edges)

            self.stats['regulation_edges'] = len(edges)
            logger.info(f"Added {len(edges)} TF-target edges")

        except FileNotFoundError as e:
            logger.warning(f"TRRUST data not found: {e}")

    def _add_go_data(self) -> None:
        """Add GO terms and gene-GO edges."""
        logger.info("Adding Gene Ontology data...")

        try:
            go_adapter = GOAdapter(
                str(self._get_data_path("goa_human.gaf.gz")),
                namespaces=['P'],  # Biological Process only
            )

            # Add GO term nodes
            nodes = list(tqdm(go_adapter.get_nodes(), desc="GO nodes"))
            self.bc.add(nodes)
            self.stats['go_nodes'] = len(nodes)

            # Add gene-GO edges
            edges = list(tqdm(go_adapter.get_edges(), desc="GO edges"))
            self.bc.add(edges)
            self.stats['go_edges'] = len(edges)

            logger.info(f"Added {len(nodes)} GO terms, {len(edges)} annotations")

        except FileNotFoundError as e:
            logger.warning(f"GO data not found: {e}")

    def _add_hpo_data(self) -> None:
        """Add phenotype nodes and gene-phenotype edges."""
        logger.info("Adding HPO phenotype data...")

        try:
            hpo_adapter = HPOAdapter(
                str(self._get_data_path("genes_to_phenotype.txt"))
            )

            # Add phenotype nodes
            nodes = list(tqdm(hpo_adapter.get_nodes(), desc="HPO nodes"))
            self.bc.add(nodes)
            self.stats['phenotype_nodes'] = len(nodes)

            # Add gene-phenotype edges
            edges = list(tqdm(hpo_adapter.get_edges(), desc="HPO edges"))
            self.bc.add(edges)
            self.stats['phenotype_edges'] = len(edges)

            logger.info(f"Added {len(nodes)} phenotypes, {len(edges)} associations")

        except FileNotFoundError as e:
            logger.warning(f"HPO data not found: {e}")

    def _add_cellmarker_data(self) -> None:
        """Add cell type nodes and marker edges."""
        logger.info("Adding CellMarker data...")

        try:
            cellmarker_adapter = CellMarkerAdapter(
                str(self._get_data_path("Cell_marker_Human.xlsx")),
                species="Human",
            )

            # Add cell type nodes
            nodes = list(tqdm(cellmarker_adapter.get_nodes(), desc="CellType nodes"))
            self.bc.add(nodes)
            self.stats['celltype_nodes'] = len(nodes)

            # Add marker edges
            edges = list(tqdm(cellmarker_adapter.get_edges(), desc="Marker edges"))
            self.bc.add(edges)
            self.stats['marker_edges'] = len(edges)

            logger.info(f"Added {len(nodes)} cell types, {len(edges)} marker relations")

        except FileNotFoundError as e:
            logger.warning(f"CellMarker data not found: {e}")

    def _add_reactome_data(self) -> None:
        """Add pathway nodes and gene-pathway edges from Reactome."""
        logger.info("Adding Reactome pathway data...")

        try:
            reactome_adapter = ReactomeAdapter(
                pathways_path=str(self._get_data_path("Reactome/ReactomePathways.txt")),
                gene_pathway_path=str(self._get_data_path("Reactome/NCBI2Reactome.txt")),
                pathway_relations_path=str(self._get_data_path("Reactome/ReactomePathwaysRelation.txt")),
                hgnc_path=str(self._get_data_path("hgnc_complete_set.txt")),
                include_hierarchy=True,
            )

            # Add pathway nodes
            nodes = list(tqdm(reactome_adapter.get_nodes(), desc="Pathway nodes"))
            self.bc.add(nodes)
            self.stats['pathway_nodes'] = len(nodes)

            # Add gene-pathway and hierarchy edges
            edges = list(tqdm(reactome_adapter.get_edges(), desc="Reactome edges"))
            self.bc.add(edges)

            # Count edge types
            gene_pathway_edges = sum(1 for e in edges if e[2] == 'IN_PATHWAY')
            hierarchy_edges = sum(1 for e in edges if e[2] == 'PATHWAY_CONTAINS')

            self.stats['pathway_edges'] = gene_pathway_edges
            self.stats['pathway_hierarchy_edges'] = hierarchy_edges

            logger.info(
                f"Added {len(nodes)} pathways, "
                f"{gene_pathway_edges} gene-pathway edges, "
                f"{hierarchy_edges} hierarchy edges"
            )

            # Print Reactome stats
            stats = reactome_adapter.get_stats()
            logger.info(f"Reactome stats: {stats['num_unique_genes']} unique genes in pathways")

        except FileNotFoundError as e:
            logger.warning(f"Reactome data not found: {e}")

    def build(self) -> nx.DiGraph:
        """
        Build the complete knowledge graph.

        Returns:
            NetworkX DiGraph
        """
        logger.info("=" * 60)
        logger.info("Starting SIGR Knowledge Graph Construction")
        logger.info("Using BioCypher with Biolink Model")
        logger.info("=" * 60)

        start_time = datetime.now()

        # 1. Add gene nodes (foundation)
        self._add_gene_nodes()

        # 2. Add edges from various sources
        self._add_string_edges()
        self._add_trrust_edges()
        self._add_go_data()
        self._add_hpo_data()
        self._add_cellmarker_data()
        self._add_reactome_data()

        # 3. Get the NetworkX graph from in-memory KG
        logger.info("Retrieving NetworkX graph from BioCypher...")
        graph = self.bc.get_kg()

        # 4. Print statistics
        elapsed = datetime.now() - start_time
        self._print_statistics(graph, elapsed)

        return graph

    def _print_statistics(self, graph: nx.DiGraph, elapsed) -> None:
        """Print comprehensive graph statistics."""
        logger.info("\n" + "=" * 60)
        logger.info("SIGR Knowledge Graph - Build Complete")
        logger.info("=" * 60)
        logger.info(f"Build time: {elapsed}")
        logger.info(f"\nGraph Statistics:")
        logger.info(f"  Total nodes: {graph.number_of_nodes():,}")
        logger.info(f"  Total edges: {graph.number_of_edges():,}")

        # Node type distribution
        node_types = defaultdict(int)
        for _, data in graph.nodes(data=True):
            label = data.get('label') or data.get('node_label') or 'Unknown'
            node_types[label] += 1

        logger.info("\nNode Types:")
        for ntype, count in sorted(node_types.items(), key=lambda x: -x[1]):
            logger.info(f"  {ntype:25s}: {count:,}")

        # Edge type distribution
        edge_types = defaultdict(int)
        for _, _, data in graph.edges(data=True):
            label = data.get('label') or data.get('relationship_label') or 'Unknown'
            edge_types[label] += 1

        logger.info("\nEdge Types (Biolink):")
        for etype, count in sorted(edge_types.items(), key=lambda x: -x[1]):
            logger.info(f"  {etype:25s}: {count:,}")

        logger.info("=" * 60)

    def save(self, graph: nx.DiGraph, output_path: str) -> None:
        """
        Save the knowledge graph.

        Args:
            graph: NetworkX graph to save
            output_path: Output file path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'wb') as f:
            pickle.dump(graph, f, protocol=4)

        # Also save statistics
        stats_path = output_path.with_suffix('.stats.txt')
        with open(stats_path, 'w') as f:
            f.write(f"SIGR Knowledge Graph Statistics\n")
            f.write(f"Generated: {datetime.now()}\n")
            f.write(f"Nodes: {graph.number_of_nodes()}\n")
            f.write(f"Edges: {graph.number_of_edges()}\n")
            f.write(f"\nBuild Statistics:\n")
            for key, value in self.stats.items():
                f.write(f"  {key}: {value}\n")

        logger.info(f"Saved graph to: {output_path}")
        logger.info(f"Saved stats to: {stats_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Build SIGR Knowledge Graph using BioCypher"
    )
    parser.add_argument(
        '--data-dir',
        default='data/raw',
        help='Directory containing raw data files'
    )
    parser.add_argument(
        '--output',
        default='data/kg/sigr_kg.pkl',
        help='Output path for the graph'
    )
    parser.add_argument(
        '--string-threshold',
        type=int,
        default=700,
        help='STRING score threshold (0-1000, default: 700)'
    )

    args = parser.parse_args()

    # Build graph
    builder = SIGRKnowledgeGraph(
        data_dir=args.data_dir,
        string_threshold=args.string_threshold,
    )

    graph = builder.build()
    builder.save(graph, args.output)

    logger.info("Done!")


if __name__ == "__main__":
    main()
