"""
RAG Generator for KG-RAG

Generates gene descriptions using retrieved KG triplets
as context for the LLM.
"""

import logging
from typing import Dict, List, Optional, Any
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from .retriever import RAGRetriever, RetrievedTriplet

logger = logging.getLogger(__name__)


# RAG prompt template
RAG_PROMPT_TEMPLATE = """You are a computational biologist. Generate a concise description (100-150 words)
for gene {gene_symbol} based on the following semantically retrieved knowledge.

## Retrieved Context (most relevant to {task_description}):

{retrieved_context}

Based on this retrieved information, synthesize a description focusing on:
- Key biological functions and processes
- Network importance and connectivity
- Phenotypic associations
- Regulatory relationships

IMPORTANT CONSTRAINTS:
- Do NOT predict whether this gene is dosage-sensitive or not
- Do NOT include any classification labels
- Focus on features that indicate biological essentiality

Generate a description for gene {gene_symbol}:"""


# Task descriptions for prompts
TASK_DESCRIPTIONS = {
    'geneattribute_dosage_sensitivity': 'dosage sensitivity prediction',
    'ppi': 'protein-protein interaction prediction',
    'cell': 'cell type classification',
    'perturbation': 'perturbation response prediction',
}


class RAGGenerator:
    """
    RAG-based gene description generator.

    Uses retrieved triplets as context for LLM generation.
    """

    def __init__(
        self,
        retriever: RAGRetriever,
        llm_client,
        task_name: str = 'geneattribute_dosage_sensitivity',
    ):
        """
        Initialize the RAG generator.

        Args:
            retriever: RAGRetriever instance
            llm_client: LLM client with generate() method
            task_name: Task name for context
        """
        self.retriever = retriever
        self.llm = llm_client
        self.task_name = task_name
        self.task_description = TASK_DESCRIPTIONS.get(task_name, task_name)

    def format_retrieved_context(
        self,
        triplets: List[RetrievedTriplet],
        max_per_type: int = 10,
        include_scores: bool = True,
    ) -> str:
        """
        Format retrieved triplets as context string.

        Groups triplets by edge type for better organization.

        Args:
            triplets: List of retrieved triplets
            max_per_type: Max triplets per edge type
            include_scores: Whether to include relevance scores

        Returns:
            Formatted context string
        """
        if not triplets:
            return "No relevant knowledge found."

        # Group by edge type
        by_type: Dict[str, List[RetrievedTriplet]] = defaultdict(list)
        for t in triplets:
            by_type[t.edge_type].append(t)

        # Format each group
        sections = []

        # Order edge types by importance
        edge_type_order = ['HPO', 'TRRUST', 'GO', 'PPI', 'Reactome', 'CellMarker', 'OMIM', 'GTEx', 'CORUM']
        sorted_types = sorted(
            by_type.keys(),
            key=lambda x: edge_type_order.index(x) if x in edge_type_order else 100
        )

        for edge_type in sorted_types:
            group = by_type[edge_type][:max_per_type]
            if not group:
                continue

            # Edge type header
            header = self._get_edge_type_header(edge_type)
            section_lines = [f"### {header} ({len(group)} items):"]

            for t in group:
                if include_scores:
                    section_lines.append(f"- {t.text} (relevance: {t.score:.2f})")
                else:
                    section_lines.append(f"- {t.text}")

            sections.append('\n'.join(section_lines))

        return '\n\n'.join(sections)

    def _get_edge_type_header(self, edge_type: str) -> str:
        """Get human-readable header for edge type."""
        headers = {
            'PPI': 'Protein-Protein Interactions',
            'GO': 'Gene Ontology Annotations',
            'HPO': 'Phenotype Associations',
            'TRRUST': 'Regulatory Relationships',
            'Reactome': 'Pathway Membership',
            'CellMarker': 'Cell Type Markers',
            'OMIM': 'Disease Associations',
            'GTEx': 'Tissue Expression',
            'CORUM': 'Protein Complex Membership',
        }
        return headers.get(edge_type, edge_type)

    def generate(
        self,
        gene_id: str,
        top_k: int = 30,
        filter_connected: bool = False,
        use_diversity: bool = True,
    ) -> str:
        """
        Generate description for a single gene.

        Args:
            gene_id: Gene symbol
            top_k: Number of triplets to retrieve
            filter_connected: Only use triplets connected to gene
            use_diversity: Ensure edge type diversity

        Returns:
            Generated description
        """
        # 1. Retrieve relevant triplets
        if use_diversity:
            triplets = self.retriever.retrieve_with_diversity(
                gene_id,
                top_k=top_k,
                filter_connected=filter_connected,
            )
        else:
            triplets = self.retriever.retrieve(
                gene_id,
                top_k=top_k,
                filter_connected=filter_connected,
            )

        # 2. Format context
        context = self.format_retrieved_context(triplets)

        # 3. Build prompt
        prompt = RAG_PROMPT_TEMPLATE.format(
            gene_symbol=gene_id,
            task_description=self.task_description,
            retrieved_context=context,
        )

        # 4. Generate description
        description = self.llm.generate(prompt)

        return description

    def generate_batch(
        self,
        gene_ids: List[str],
        top_k: int = 30,
        max_workers: int = 8,
        show_progress: bool = True,
        **kwargs
    ) -> Dict[str, str]:
        """
        Generate descriptions for multiple genes.

        Args:
            gene_ids: List of gene symbols
            top_k: Triplets per gene
            max_workers: Concurrent LLM requests
            show_progress: Show progress bar
            **kwargs: Additional arguments

        Returns:
            Dict mapping gene_id to description
        """
        descriptions = {}

        def _generate_one(gene_id: str) -> tuple:
            try:
                desc = self.generate(gene_id, top_k=top_k, **kwargs)
                return (gene_id, desc, None)
            except Exception as e:
                logger.debug(f"Error generating for {gene_id}: {e}")
                return (gene_id, None, str(e))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_generate_one, gid): gid
                for gid in gene_ids
            }

            iterator = as_completed(futures)
            if show_progress:
                iterator = tqdm(iterator, total=len(gene_ids), desc="Generating (RAG)")

            for future in iterator:
                gene_id, desc, error = future.result()
                if desc:
                    descriptions[gene_id] = desc
                else:
                    logger.warning(f"Failed for {gene_id}: {error}")

        logger.info(f"Generated {len(descriptions)}/{len(gene_ids)} descriptions")
        return descriptions


class RAGBaseline:
    """
    Complete RAG baseline for experiments.

    Combines retriever, generator, and encoder for
    end-to-end KG-RAG pipeline.
    """

    def __init__(
        self,
        kg,
        index,
        encoder,
        llm_client,
        task_name: str = 'geneattribute_dosage_sensitivity',
    ):
        """
        Initialize RAG baseline.

        Args:
            kg: Knowledge graph
            index: Pre-built TripletIndex
            encoder: GeneEncoder
            llm_client: LLM client
            task_name: Task name
        """
        self.kg = kg
        self.index = index
        self.encoder = encoder
        self.task_name = task_name

        # Initialize retriever
        self.retriever = RAGRetriever(
            index=index,
            encoder=encoder,
            kg=kg,
            task_name=task_name,
        )

        # Initialize generator
        self.generator = RAGGenerator(
            retriever=self.retriever,
            llm_client=llm_client,
            task_name=task_name,
        )

    def run(
        self,
        gene_ids: List[str],
        top_k: int = 30,
        max_workers: int = 8,
        show_progress: bool = True,
    ) -> Dict[str, Any]:
        """
        Run RAG pipeline for genes.

        Args:
            gene_ids: Genes to process
            top_k: Triplets per gene
            max_workers: Concurrent workers
            show_progress: Show progress

        Returns:
            Dict with descriptions and embeddings
        """
        # 1. Generate descriptions
        descriptions = self.generator.generate_batch(
            gene_ids,
            top_k=top_k,
            max_workers=max_workers,
            show_progress=show_progress,
        )

        # 2. Encode descriptions
        if descriptions:
            logger.info("Encoding descriptions...")
            texts = list(descriptions.values())
            gids = list(descriptions.keys())
            emb_array = self.encoder.encode_batch(texts, show_progress=show_progress)

            # Post-process: center and normalize
            mean = emb_array.mean(axis=0)
            emb_array = emb_array - mean
            norms = np.linalg.norm(emb_array, axis=1, keepdims=True)
            emb_array = emb_array / np.maximum(norms, 1e-8)

            embeddings = {gid: emb for gid, emb in zip(gids, emb_array)}
        else:
            embeddings = {}

        return {
            'descriptions': descriptions,
            'embeddings': embeddings,
        }


# Import numpy for RAGBaseline
import numpy as np
