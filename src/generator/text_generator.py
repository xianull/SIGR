"""
Text Generator for SIGR Framework

Uses LLM to convert KG subgraphs into natural language gene descriptions.
"""

import logging
from typing import Dict, Any, Optional, Protocol, List

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

# Focus mode guidance for description generation
FOCUS_MODE_GUIDANCE = {
    'balanced': "",  # No additional guidance
    'network': "Focus primarily on network properties: hub gene status, interaction count, network centrality, protein complex membership, and connectivity patterns.",
    'function': "Focus primarily on biological function: molecular activities, biological processes, pathway involvement, and functional annotations.",
    'phenotype': "Focus primarily on phenotype associations: disease links, clinical manifestations, phenotypic outcomes, and pathological features.",
}

# Prompt style templates
PROMPT_STYLE_TEMPLATES = {
    'analytical': "{base_prompt}",  # Default analytical style
    'narrative': "Tell the story of gene {gene_id} ({gene_name}). Describe its biological journey from molecular function to phenotypic impact.\n\n{context}\n\nProvide a narrative description that weaves together the available information into a cohesive story.",
    'structured': "Provide a structured summary of gene {gene_id} ({gene_name}) using the following format:\n\n{context}\n\n- Molecular Function:\n- Biological Processes:\n- Protein Interactions:\n- Associated Phenotypes:\n- Key Features:",
    'comparative': "Describe gene {gene_id} ({gene_name}) by highlighting what makes it distinctive compared to typical genes.\n\n{context}\n\nEmphasize unique or notable characteristics.",
}

# Context window configurations
CONTEXT_WINDOW_CONFIG = {
    'minimal': {'max_items_per_type': 5, 'max_hops': 1},
    'local': {'max_items_per_type': 15, 'max_hops': 1},
    'extended': {'max_items_per_type': 30, 'max_hops': 2},
    'full': {'max_items_per_type': None, 'max_hops': None},  # No limits
}

# Feature selection priorities
FEATURE_SELECTION_CONFIG = {
    'all': None,  # Include all features
    'essential': ['ppi_info', 'go_info', 'phenotype_info'],  # Core features only
    'diverse': ['ppi_info', 'go_info', 'phenotype_info', 'tf_info', 'pathway_info', 'celltype_info'],
    'task_specific': None,  # Determined by task
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

    Supports extended strategy parameters:
    - description_focus: 'balanced', 'network', 'function', 'phenotype'
    - context_window: 'minimal', 'local', 'extended', 'full'
    - prompt_style: 'analytical', 'narrative', 'structured', 'comparative'
    - feature_selection: 'all', 'essential', 'diverse', 'task_specific'
    - generation_passes: 1-3 (multi-pass refinement)
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

        # New strategy parameters (v2)
        description_focus = strategy.get('description_focus', 'balanced')
        context_window = strategy.get('context_window', 'full')
        prompt_style = strategy.get('prompt_style', 'analytical')
        feature_selection = strategy.get('feature_selection', 'all')
        generation_passes = strategy.get('generation_passes', 1)

        # Get word count target
        min_words, max_words = DESCRIPTION_LENGTH_TARGETS.get(description_length, (100, 150))

        # Check for BASELINE MODE (no graph edges)
        is_baseline = subgraph.number_of_edges() == 0

        if is_baseline:
            # Baseline mode: generate description without graph context
            logger.debug(f"Baseline mode for {gene_id}: generating without graph context")
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
            # Apply context window filtering
            graph_info = self._format_with_context_window(
                subgraph, gene_id, include_statistics, context_window
            )

            # Apply feature selection
            graph_info = self._apply_feature_selection(graph_info, feature_selection, strategy)

            # Build prompt based on style
            prompt = self._build_styled_prompt(
                gene_id=gene_id,
                gene_name=gene_name,
                graph_info=graph_info,
                prompt_template=prompt_template,
                prompt_style=prompt_style,
                description_focus=description_focus,
                focus_keywords=focus_keywords,
                min_words=min_words,
                max_words=max_words
            )

        # Generate description using LLM (with multi-pass if enabled)
        try:
            description = self._generate_with_passes(
                prompt=prompt,
                gene_id=gene_id,
                gene_name=gene_name,
                generation_passes=generation_passes,
                min_words=min_words,
                max_words=max_words,
                is_baseline=is_baseline
            )
            mode_str = "baseline" if is_baseline else "normal"
            logger.debug(f"Generated {mode_str} description for {gene_id}: {len(description)} chars")
        except Exception as e:
            logger.error(f"Error generating description for {gene_id}: {e}")
            description = f"Gene {gene_id} ({gene_name})"

        # Apply anti-leakage filter
        from ..utils.filter import filter_description
        filtered_description = filter_description(description)

        return filtered_description

    def _format_with_context_window(
        self,
        subgraph: nx.DiGraph,
        gene_id: str,
        include_statistics: bool,
        context_window: str
    ) -> Dict[str, str]:
        """
        Format subgraph with context window constraints.

        Args:
            subgraph: Gene subgraph
            gene_id: Gene identifier
            include_statistics: Whether to include statistics
            context_window: Context window mode ('minimal', 'local', 'extended', 'full')

        Returns:
            Formatted graph information
        """
        config = CONTEXT_WINDOW_CONFIG.get(context_window, CONTEXT_WINDOW_CONFIG['full'])
        max_items = config.get('max_items_per_type')

        # Get base formatted info
        graph_info = format_subgraph(subgraph, gene_id, include_statistics=include_statistics)

        # If context window has limits, truncate each info type
        if max_items is not None:
            for key, value in graph_info.items():
                if isinstance(value, str) and value:
                    # Truncate by limiting comma-separated items or lines
                    lines = value.split('\n')
                    if len(lines) > max_items:
                        graph_info[key] = '\n'.join(lines[:max_items]) + f'\n... (truncated to {max_items} items)'
                    else:
                        # Try comma-separated
                        items = value.split(', ')
                        if len(items) > max_items:
                            graph_info[key] = ', '.join(items[:max_items]) + f', ... ({len(items) - max_items} more)'

        logger.debug(f"Context window '{context_window}' applied: max_items={max_items}")
        return graph_info

    def _apply_feature_selection(
        self,
        graph_info: Dict[str, str],
        feature_selection: str,
        strategy: Dict[str, Any]
    ) -> Dict[str, str]:
        """
        Apply feature selection to graph info.

        Args:
            graph_info: Formatted graph information
            feature_selection: Feature selection mode
            strategy: Full strategy dict for task-specific selection

        Returns:
            Filtered graph information
        """
        if feature_selection == 'all':
            return graph_info

        # Get features to keep
        features_to_keep = FEATURE_SELECTION_CONFIG.get(feature_selection)

        if feature_selection == 'task_specific':
            # Infer from focus_keywords or task context
            focus_keywords = strategy.get('focus_keywords', [])
            features_to_keep = self._infer_features_from_keywords(focus_keywords)

        if features_to_keep is None:
            return graph_info

        # Filter graph info
        filtered_info = {}
        for key, value in graph_info.items():
            if key in features_to_keep:
                filtered_info[key] = value
            else:
                filtered_info[key] = 'Not included (feature selection)'

        logger.debug(f"Feature selection '{feature_selection}': kept {len(features_to_keep)} features")
        return filtered_info

    def _infer_features_from_keywords(self, focus_keywords: List[str]) -> List[str]:
        """
        Infer which features to include based on focus keywords.

        Args:
            focus_keywords: List of focus keywords from strategy

        Returns:
            List of feature keys to include
        """
        features = ['ppi_info', 'go_info', 'phenotype_info']  # Base features

        keyword_str = ' '.join(focus_keywords).lower()

        # Add features based on keyword presence
        if any(kw in keyword_str for kw in ['regulat', 'transcript', 'tf', 'expression']):
            features.append('tf_info')
        if any(kw in keyword_str for kw in ['pathway', 'reactome', 'signal']):
            features.append('pathway_info')
        if any(kw in keyword_str for kw in ['cell', 'tissue', 'expression', 'marker']):
            features.append('celltype_info')

        return features

    def _build_styled_prompt(
        self,
        gene_id: str,
        gene_name: str,
        graph_info: Dict[str, str],
        prompt_template: str,
        prompt_style: str,
        description_focus: str,
        focus_keywords: List[str],
        min_words: int,
        max_words: int
    ) -> str:
        """
        Build prompt with appropriate style and focus.

        Args:
            gene_id: Gene identifier
            gene_name: Gene name
            graph_info: Formatted graph information
            prompt_template: Base prompt template
            prompt_style: Style of prompt ('analytical', 'narrative', etc.)
            description_focus: Focus mode for description
            focus_keywords: Keywords to emphasize
            min_words: Minimum word count
            max_words: Maximum word count

        Returns:
            Complete prompt string
        """
        # First, try to fill the base template
        try:
            base_prompt = prompt_template.format(
                gene_id=gene_id,
                gene_name=gene_name,
                ppi_info=graph_info.get('ppi_info', 'No PPI information available'),
                go_info=graph_info.get('go_info', 'No GO information available'),
                phenotype_info=graph_info.get('phenotype_info', 'No phenotype information available'),
                tf_info=graph_info.get('tf_info', 'No regulatory information available'),
                celltype_info=graph_info.get('celltype_info', 'No cell type information available'),
                pathway_info=graph_info.get('pathway_info', 'No pathway information available'),
                function_info=graph_info.get('go_info', 'N/A'),
                disease_info=graph_info.get('phenotype_info', 'N/A'),
                tissue_info=graph_info.get('celltype_info', 'N/A'),
            )
        except KeyError as e:
            logger.warning(f"Missing placeholder in prompt template: {e}")
            # Use a fallback minimal prompt
            base_prompt = f"""Generate a biological description for gene {gene_id} ({gene_name}).

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

        # Apply prompt style transformation
        if prompt_style != 'analytical':
            style_template = PROMPT_STYLE_TEMPLATES.get(prompt_style, '{base_prompt}')
            context_str = self._format_context_for_style(graph_info)
            prompt = style_template.format(
                gene_id=gene_id,
                gene_name=gene_name,
                base_prompt=base_prompt,
                context=context_str
            )
        else:
            prompt = base_prompt

        # Add focus mode guidance
        focus_guidance = FOCUS_MODE_GUIDANCE.get(description_focus, '')
        if focus_guidance:
            prompt += f"\n\n{focus_guidance}"

        # Add focus keywords guidance if provided
        if focus_keywords:
            keywords_str = ', '.join(focus_keywords[:5])
            prompt += f"\n\nEmphasize these aspects if relevant: {keywords_str}"

        return prompt

    def _format_context_for_style(self, graph_info: Dict[str, str]) -> str:
        """
        Format graph info as context string for styled prompts.

        Args:
            graph_info: Formatted graph information

        Returns:
            Context string
        """
        context_parts = []
        info_labels = {
            'ppi_info': 'Protein Interactions',
            'go_info': 'Biological Processes',
            'phenotype_info': 'Phenotypes',
            'tf_info': 'Regulatory Relations',
            'pathway_info': 'Pathways',
            'celltype_info': 'Cell Type Markers',
        }

        for key, label in info_labels.items():
            value = graph_info.get(key)
            if value and 'Not included' not in value and 'No ' not in value[:10]:
                context_parts.append(f"- {label}: {value}")

        return '\n'.join(context_parts) if context_parts else "Limited information available."

    def _generate_with_passes(
        self,
        prompt: str,
        gene_id: str,
        gene_name: str,
        generation_passes: int,
        min_words: int,
        max_words: int,
        is_baseline: bool
    ) -> str:
        """
        Generate description with optional multi-pass refinement.

        Args:
            prompt: Initial prompt
            gene_id: Gene identifier
            gene_name: Gene name
            generation_passes: Number of generation passes (1-3)
            min_words: Minimum word count
            max_words: Maximum word count
            is_baseline: Whether in baseline mode

        Returns:
            Generated (and optionally refined) description
        """
        # Pass 1: Generate base description
        description = self.llm.generate(prompt)
        logger.debug(f"Pass 1 for {gene_id}: {len(description.split())} words")

        if generation_passes < 2 or is_baseline:
            return description

        # Pass 2: Refine description for clarity and relevance
        refine_prompt = f"""Review and improve this gene description for {gene_id} ({gene_name}):

Original description:
{description}

Instructions:
1. Ensure factual accuracy and biological coherence
2. Remove any redundant or unclear statements
3. Strengthen the focus on key biological features
4. Maintain target length of {min_words}-{max_words} words
5. Do NOT add predictions or classification labels

Provide the improved description only, without explanations.
"""
        description = self.llm.generate(refine_prompt)
        logger.debug(f"Pass 2 for {gene_id}: {len(description.split())} words")

        if generation_passes < 3:
            return description

        # Pass 3: Compress and polish
        compress_prompt = f"""Distill this gene description into a highly information-dense summary:

Current description:
{description}

Requirements:
1. Maximum {max_words} words
2. Every sentence must convey essential biological information
3. Prioritize features that distinguish this gene
4. Remove filler words and generic statements
5. Maintain scientific accuracy

Provide the final polished description only.
"""
        description = self.llm.generate(compress_prompt)
        logger.debug(f"Pass 3 for {gene_id}: {len(description.split())} words")

        return description

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
