"""
Text Generator for SIGR Framework

Uses LLM to convert KG subgraphs into natural language gene descriptions.
"""

import logging
from typing import Dict, Any, Optional, Protocol

import networkx as nx

from .formatter import format_subgraph
from ..utils.kg_utils import get_gene_info
from ..actor.strategy import DESCRIPTION_LENGTH_WORDS


logger = logging.getLogger(__name__)


# Word count targets for each description length
DESCRIPTION_LENGTH_TARGETS = {
    'short': (50, 100),
    'medium': (100, 150),
    'long': (150, 250),
}


class LLMClient(Protocol):
    """Protocol for LLM client interface."""
    def generate(self, prompt: str) -> str:
        """Generate text from prompt."""
        ...


class TextGenerator:
    """
    Generates natural language gene descriptions using LLM.

    Takes a KG subgraph and converts it to text using:
    1. Formatter to structure the graph information
    2. LLM to generate coherent descriptions
    """

    def __init__(self, llm_client: LLMClient):
        """
        Initialize the text generator.

        Args:
            llm_client: LLM client for text generation
        """
        self.llm = llm_client
        logger.info("TextGenerator initialized")

    def generate(
        self,
        gene_id: str,
        subgraph: nx.DiGraph,
        prompt_template: str,
        kg: Optional[nx.DiGraph] = None,
        strategy: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate a description for a gene based on its subgraph.

        Args:
            gene_id: Gene identifier
            subgraph: Extracted KG subgraph for the gene
            prompt_template: Template for LLM prompt
            kg: Full KG for additional gene info (optional)
            strategy: Strategy dict with optional description_length, focus_keywords, etc.

        Returns:
            Generated gene description
        """
        # Get gene info
        if kg is not None:
            gene_info = get_gene_info(kg, gene_id)
            gene_name = gene_info.get('name', gene_id)
        else:
            gene_name = gene_id

        # Extract strategy parameters
        strategy = strategy or {}
        description_length = strategy.get('description_length', 'medium')
        focus_keywords = strategy.get('focus_keywords', [])
        include_statistics = strategy.get('include_statistics', True)

        # Get word count target
        min_words, max_words = DESCRIPTION_LENGTH_TARGETS.get(description_length, (100, 150))

        # Check for BASELINE MODE (no graph edges)
        is_baseline = subgraph.number_of_edges() == 0

        if is_baseline:
            # Baseline mode: generate description without graph context
            logger.debug(f"Baseline mode for {gene_id}: generating without graph context")
            # Diagnostic logging for debugging high AUC issue (DEBUG level to avoid console spam)
            logger.debug(f"BASELINE MODE for {gene_id}")
            logger.debug(f"  Prompt template length: {len(prompt_template)}")
            if prompt_template:
                logger.debug(f"  First 200 chars: {prompt_template[:200]}")
            try:
                prompt = prompt_template.format(
                    gene_id=gene_id,
                    gene_name=gene_name,
                    ppi_info="Not available (baseline mode)",
                    go_info="Not available (baseline mode)",
                    phenotype_info="Not available (baseline mode)",
                    tf_info="Not available (baseline mode)",
                    celltype_info="Not available (baseline mode)",
                    pathway_info="Not available (baseline mode)",
                    function_info="Not available (baseline mode)",
                    disease_info="Not available (baseline mode)",
                    tissue_info="Not available (baseline mode)",
                )
            except KeyError as e:
                logger.warning(f"Missing placeholder in baseline prompt template: {e}")
                prompt = f"""Generate a biological description for gene {gene_id} ({gene_name}).
Based only on the gene name and your general biological knowledge, describe what is typically known about this gene.
Do not make specific predictions or classification judgments.
Target length: {min_words}-{max_words} words.
"""
        else:
            # Normal mode: use graph information
            graph_info = format_subgraph(subgraph, gene_id, include_statistics=include_statistics)

            # Build prompt
            try:
                prompt = prompt_template.format(
                    gene_id=gene_id,
                    gene_name=gene_name,
                    ppi_info=graph_info.get('ppi_info', 'No PPI information available'),
                    go_info=graph_info.get('go_info', 'No GO information available'),
                    phenotype_info=graph_info.get('phenotype_info', 'No phenotype information available'),
                    tf_info=graph_info.get('tf_info', 'No regulatory information available'),
                    celltype_info=graph_info.get('celltype_info', 'No cell type information available'),
                    pathway_info=graph_info.get('pathway_info', 'No pathway information available'),
                    # Additional optional placeholders
                    function_info=graph_info.get('go_info', 'N/A'),
                    disease_info=graph_info.get('phenotype_info', 'N/A'),
                    tissue_info=graph_info.get('celltype_info', 'N/A'),
                )
            except KeyError as e:
                logger.warning(f"Missing placeholder in prompt template: {e}")
                # Use a fallback minimal prompt
                prompt = f"""Generate a biological description for gene {gene_id} ({gene_name}).

Available information:
- Protein Interactions: {graph_info.get('ppi_info', 'N/A')}
- Biological Processes: {graph_info.get('go_info', 'N/A')}
- Phenotypes: {graph_info.get('phenotype_info', 'N/A')}
- Regulatory Relations: {graph_info.get('tf_info', 'N/A')}
- Cell Type Markers: {graph_info.get('celltype_info', 'N/A')}
- Pathways: {graph_info.get('pathway_info', 'N/A')}

Target length: {min_words}-{max_words} words.
Provide a factual description without predictions or labels.
"""

            # Add focus keywords guidance if provided
            if focus_keywords:
                keywords_str = ', '.join(focus_keywords[:5])
                prompt += f"\n\nEmphasize these aspects if relevant: {keywords_str}"

        # Generate description using LLM
        try:
            description = self.llm.generate(prompt)
            mode_str = "baseline" if is_baseline else "normal"
            logger.debug(f"Generated {mode_str} description for {gene_id}: {len(description)} chars")
        except Exception as e:
            logger.error(f"Error generating description for {gene_id}: {e}")
            description = f"Gene {gene_id} ({gene_name})"

        # Apply anti-leakage filter
        from ..utils.filter import filter_description
        filtered_description = filter_description(description)

        return filtered_description

    def generate_batch(
        self,
        gene_subgraphs: Dict[str, nx.DiGraph],
        prompt_template: str,
        kg: Optional[nx.DiGraph] = None,
        strategy: Optional[Dict[str, Any]] = None
    ) -> Dict[str, str]:
        """
        Generate descriptions for multiple genes.

        Args:
            gene_subgraphs: Dictionary mapping gene_id to subgraph
            prompt_template: Template for LLM prompt
            kg: Full KG for additional gene info (optional)
            strategy: Strategy dict with optional description_length, focus_keywords, etc.

        Returns:
            Dictionary mapping gene_id to description
        """
        descriptions = {}

        for gene_id, subgraph in gene_subgraphs.items():
            try:
                description = self.generate(
                    gene_id=gene_id,
                    subgraph=subgraph,
                    prompt_template=prompt_template,
                    kg=kg,
                    strategy=strategy
                )
                descriptions[gene_id] = description
            except Exception as e:
                logger.error(f"Error generating description for {gene_id}: {e}")
                descriptions[gene_id] = f"Gene {gene_id}"

        return descriptions


class MockTextGenerator:
    """
    Mock text generator for testing without LLM calls.

    Generates deterministic descriptions based on subgraph content.
    """

    def generate(
        self,
        gene_id: str,
        subgraph: nx.DiGraph,
        prompt_template: str,
        kg: Optional[nx.DiGraph] = None
    ) -> str:
        """Generate mock description."""
        # Get basic info from subgraph
        num_nodes = subgraph.number_of_nodes()
        num_edges = subgraph.number_of_edges()

        # Get gene name if available
        gene_name = gene_id
        if kg is not None:
            gene_info = get_gene_info(kg, gene_id)
            gene_name = gene_info.get('name', gene_id)

        # Build description from subgraph
        graph_info = format_subgraph(subgraph, gene_id)

        description_parts = [f"{gene_name} is a gene"]

        if graph_info.get('ppi_info') and graph_info['ppi_info'] != 'No protein interactions found':
            description_parts.append(f"with protein interactions")

        if graph_info.get('go_info') and graph_info['go_info'] != 'No GO terms found':
            description_parts.append(f"involved in biological processes")

        if graph_info.get('phenotype_info') and graph_info['phenotype_info'] != 'No phenotype associations found':
            description_parts.append(f"associated with phenotypes")

        description = ' '.join(description_parts) + '.'
        description += f" [Subgraph: {num_nodes} nodes, {num_edges} edges]"

        return description

    def generate_batch(
        self,
        gene_subgraphs: Dict[str, nx.DiGraph],
        prompt_template: str,
        kg: Optional[nx.DiGraph] = None
    ) -> Dict[str, str]:
        """Generate mock descriptions for multiple genes."""
        return {
            gene_id: self.generate(gene_id, subgraph, prompt_template, kg)
            for gene_id, subgraph in gene_subgraphs.items()
        }
