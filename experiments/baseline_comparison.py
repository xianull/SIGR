#!/usr/bin/env python3
"""
Baseline Comparison Experiment for Dosage Sensitivity Task

Compares three approaches for gene description generation:
1. No-KG: Pure LLM generation without graph knowledge
2. Full-KG: Use all available graph knowledge
3. KG-RAG: Retrieval-augmented generation with relevant KG neighbors

This experiment evaluates how different levels of KG information
affect downstream task performance.
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import load_kg, get_all_genes
from src.utils.logger import setup_logging
from src.encoder import GeneEncoder
from src.evaluator import TaskEvaluator
from src.generator import extract_subgraph, format_subgraph


logger = logging.getLogger(__name__)


# =============================================================================
# Prompts for Dosage Sensitivity Task
# =============================================================================

# Method 1: No-KG - Pure LLM without graph knowledge
NO_KG_PROMPT = """You are a computational biologist. Generate a concise description (100-150 words)
for gene {gene_symbol} that would be useful for predicting dosage sensitivity.

Focus on:
- Gene function and biological role
- Network importance (if known)
- Evolutionary conservation
- Disease associations
- Regulatory complexity

IMPORTANT CONSTRAINTS:
- Do NOT predict whether this gene is dosage-sensitive or not
- Do NOT include any classification labels
- Only describe objective biological features
- Be factual and avoid speculation

Generate a description for gene {gene_symbol}:"""


# Method 2: Full-KG - Use all available graph information
FULL_KG_PROMPT = """You are a computational biologist. Generate a concise description (100-150 words)
for gene {gene_symbol} based on the following knowledge graph information.

## Knowledge Graph Context:

### Protein-Protein Interactions:
{ppi_info}

### Gene Ontology (Biological Processes):
{go_info}

### Human Phenotype Ontology:
{phenotype_info}

### Regulatory Network (TRRUST):
{tf_info}

### Pathway Membership (Reactome):
{pathway_info}

Focus on features relevant to dosage sensitivity:
- Network centrality and hub status
- Essential pathway membership
- Phenotype associations
- Regulatory complexity

IMPORTANT CONSTRAINTS:
- Do NOT predict whether this gene is dosage-sensitive or not
- Do NOT include any classification labels
- Only describe objective biological features derived from the KG

Generate a description for gene {gene_symbol}:"""


# Method 3: KG-RAG - Retrieval-augmented with selective KG information
KG_RAG_PROMPT = """You are a computational biologist. Generate a concise description (100-150 words)
for gene {gene_symbol} using the following retrieved knowledge.

## Retrieved Context (most relevant to dosage sensitivity):

### Key Phenotype Associations:
{phenotype_info}

### Regulatory Network Position:
{tf_info}

### Functional Annotations:
{go_info}

### Interaction Partners (top by relevance):
{ppi_info}

Synthesize this information to describe the gene's:
- Network importance and connectivity
- Phenotypic impact
- Regulatory role
- Biological significance

IMPORTANT CONSTRAINTS:
- Do NOT predict whether this gene is dosage-sensitive or not
- Do NOT include any classification labels
- Focus on features that indicate biological essentiality

Generate a description for gene {gene_symbol}:"""


# =============================================================================
# Experiment Classes
# =============================================================================

class BaselineExperiment:
    """Base class for baseline experiments."""

    def __init__(
        self,
        kg_path: str,
        llm_client,
        task_name: str = 'geneattribute_dosage_sensitivity',
        max_workers: int = 8,
        output_dir: str = 'results/baseline_comparison',
    ):
        """
        Initialize the experiment.

        Args:
            kg_path: Path to knowledge graph
            llm_client: LLM client for text generation
            task_name: Downstream task name
            max_workers: Max concurrent LLM requests
            output_dir: Directory for results
        """
        self.kg_path = kg_path
        self.llm = llm_client
        self.task_name = task_name
        self.max_workers = max_workers
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load KG
        logger.info(f"Loading knowledge graph from {kg_path}")
        self.kg = load_kg(kg_path)
        logger.info(f"KG loaded: {self.kg.number_of_nodes()} nodes, {self.kg.number_of_edges()} edges")

        # Initialize encoder
        self.encoder = GeneEncoder(encoder_type='local')

        # Initialize evaluator
        self.evaluator = TaskEvaluator(
            task_name=task_name,
            kg=self.kg,
            use_cross_validation=True,
            n_folds=5,
            use_multiple_classifiers=True,
        )

        # Get task genes
        self.task_genes = self.evaluator.get_task_genes()
        kg_genes = set(get_all_genes(self.kg))
        self.task_genes = [g for g in self.task_genes if g in kg_genes]
        logger.info(f"Task genes: {len(self.task_genes)}")

    def run_method(
        self,
        method_name: str,
        prompt_template: str,
        use_kg: bool = False,
        kg_mode: str = 'full',
    ) -> Dict[str, Any]:
        """
        Run a single method and return results.

        Args:
            method_name: Name of the method
            prompt_template: Prompt template to use
            use_kg: Whether to use KG information
            kg_mode: 'full' or 'rag' mode for KG extraction

        Returns:
            Dictionary with metrics and descriptions
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Running method: {method_name}")
        logger.info(f"{'='*60}")

        descriptions = {}

        def generate_description(gene_id: str) -> tuple:
            """Generate description for a single gene."""
            try:
                if use_kg:
                    # Extract subgraph
                    if kg_mode == 'rag':
                        # RAG mode: prioritize phenotype and regulatory edges
                        strategy = {
                            'edge_types': ['HPO', 'TRRUST', 'GO', 'PPI'],
                            'max_neighbors': 30,  # Limited neighbors
                            'sampling': 'top_k',
                            'max_hops': 1,
                        }
                    else:
                        # Full mode: use all edge types with more neighbors
                        strategy = {
                            'edge_types': ['PPI', 'GO', 'HPO', 'TRRUST', 'Reactome', 'CellMarker'],
                            'max_neighbors': 100,
                            'sampling': 'top_k',
                            'max_hops': 2,
                        }

                    subgraph = extract_subgraph(
                        gene_id=gene_id,
                        strategy=strategy,
                        kg=self.kg,
                    )

                    # Format subgraph
                    formatted = format_subgraph(subgraph, center_gene=gene_id)

                    # Fill prompt
                    prompt = prompt_template.format(
                        gene_symbol=gene_id,
                        ppi_info=formatted.get('ppi_info', 'No interactions found.'),
                        go_info=formatted.get('go_info', 'No GO terms found.'),
                        phenotype_info=formatted.get('phenotype_info', 'No phenotypes found.'),
                        tf_info=formatted.get('tf_info', 'No regulatory information found.'),
                        pathway_info=formatted.get('pathway_info', 'No pathway information found.'),
                    )
                else:
                    # No KG mode
                    prompt = prompt_template.format(gene_symbol=gene_id)

                # Generate description
                description = self.llm.generate(prompt)

                return (gene_id, description, None)

            except Exception as e:
                logger.debug(f"Error generating description for {gene_id}: {e}")
                return (gene_id, None, str(e))

        # Generate descriptions with progress bar
        logger.info(f"Generating descriptions for {len(self.task_genes)} genes...")

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(generate_description, gene_id): gene_id
                for gene_id in self.task_genes
            }

            for future in tqdm(as_completed(futures), total=len(self.task_genes),
                              desc=f"Generating ({method_name})"):
                gene_id, desc, error = future.result()
                if desc:
                    descriptions[gene_id] = desc
                else:
                    logger.warning(f"Failed for {gene_id}: {error}")

        if not descriptions:
            logger.error("No descriptions generated!")
            return {'error': 'No descriptions generated'}

        logger.info(f"Generated {len(descriptions)} descriptions")

        # Encode descriptions
        logger.info("Encoding descriptions...")
        embeddings = self.encoder.encode_genes(descriptions, batch_size=64, show_progress=True)
        logger.info(f"Encoded {len(embeddings)} embeddings")

        # Evaluate
        logger.info("Evaluating on downstream task...")
        metrics = self.evaluator.evaluate(embeddings)

        # Log results
        if 'combined' in metrics:
            logger.info(f"Results for {method_name}:")
            logger.info(f"  LR:  AUC={metrics['logistic'].get('auc', 0):.4f}, F1={metrics['logistic'].get('f1', 0):.4f}")
            logger.info(f"  RF:  AUC={metrics['random_forest'].get('auc', 0):.4f}, F1={metrics['random_forest'].get('f1', 0):.4f}")
            logger.info(f"  Avg: AUC={metrics['combined'].get('auc', 0):.4f}, F1={metrics['combined'].get('f1', 0):.4f}")
        else:
            logger.info(f"Results for {method_name}:")
            logger.info(f"  AUC={metrics.get('auc', 0):.4f}, F1={metrics.get('f1', 0):.4f}")

        return {
            'method': method_name,
            'metrics': metrics,
            'num_descriptions': len(descriptions),
            'num_embeddings': len(embeddings),
        }

    def run_all_methods(self) -> Dict[str, Any]:
        """Run all three baseline methods and compare."""
        results = {}
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Method 1: No-KG
        logger.info("\n" + "="*70)
        logger.info("Method 1: NO-KG (Pure LLM without graph knowledge)")
        logger.info("="*70)
        results['no_kg'] = self.run_method(
            method_name='No-KG',
            prompt_template=NO_KG_PROMPT,
            use_kg=False,
        )

        # Method 2: Full-KG
        logger.info("\n" + "="*70)
        logger.info("Method 2: FULL-KG (All available graph knowledge)")
        logger.info("="*70)
        results['full_kg'] = self.run_method(
            method_name='Full-KG',
            prompt_template=FULL_KG_PROMPT,
            use_kg=True,
            kg_mode='full',
        )

        # Method 3: KG-RAG
        logger.info("\n" + "="*70)
        logger.info("Method 3: KG-RAG (Retrieval-augmented with selective KG)")
        logger.info("="*70)
        results['kg_rag'] = self.run_method(
            method_name='KG-RAG',
            prompt_template=KG_RAG_PROMPT,
            use_kg=True,
            kg_mode='rag',
        )

        # Summary comparison
        self._print_comparison(results)

        # Save results
        results_file = self.output_dir / f'baseline_comparison_{timestamp}.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"\nResults saved to: {results_file}")

        return results

    def _print_comparison(self, results: Dict[str, Any]):
        """Print comparison table."""
        print("\n" + "="*70)
        print("BASELINE COMPARISON RESULTS")
        print("="*70)
        print(f"Task: {self.task_name}")
        print(f"Genes: {len(self.task_genes)}")
        print("-"*70)
        print(f"{'Method':<15} {'AUC (LR)':<12} {'AUC (RF)':<12} {'AUC (Avg)':<12} {'F1 (Avg)':<12}")
        print("-"*70)

        for method_key in ['no_kg', 'full_kg', 'kg_rag']:
            if method_key not in results or 'error' in results[method_key]:
                print(f"{method_key:<15} {'ERROR':<12}")
                continue

            metrics = results[method_key]['metrics']
            method_name = results[method_key]['method']

            if 'combined' in metrics:
                auc_lr = metrics['logistic'].get('auc', 0)
                auc_rf = metrics['random_forest'].get('auc', 0)
                auc_avg = metrics['combined'].get('auc', 0)
                f1_avg = metrics['combined'].get('f1', 0)
            else:
                auc_lr = auc_rf = auc_avg = metrics.get('auc', 0)
                f1_avg = metrics.get('f1', 0)

            print(f"{method_name:<15} {auc_lr:<12.4f} {auc_rf:<12.4f} {auc_avg:<12.4f} {f1_avg:<12.4f}")

        print("="*70)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run baseline comparison experiment for dosage sensitivity"
    )

    parser.add_argument(
        '--kg-path',
        type=str,
        default='data/kg/sigr_kg_v2.pkl',
        help='Path to knowledge graph'
    )

    parser.add_argument(
        '--api-base',
        type=str,
        default='https://yunwu.ai/v1',
        help='LLM API base URL'
    )

    parser.add_argument(
        '--api-key',
        type=str,
        default=None,
        help='LLM API key'
    )

    parser.add_argument(
        '--model',
        type=str,
        default='gpt-4o-mini',
        help='LLM model name (fast model)'
    )

    parser.add_argument(
        '--mock',
        action='store_true',
        help='Use mock LLM for testing'
    )

    parser.add_argument(
        '--max-workers',
        type=int,
        default=8,
        help='Max concurrent LLM requests'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/baseline_comparison',
        help='Output directory for results'
    )

    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(level=log_level)

    logger.info("="*70)
    logger.info("Baseline Comparison Experiment")
    logger.info("="*70)
    logger.info(f"Task: geneattribute_dosage_sensitivity")
    logger.info(f"KG: {args.kg_path}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Mock: {args.mock}")

    # Create LLM client
    if args.mock:
        from src.sigr_framework import MockLLMClient
        llm_client = MockLLMClient()
        logger.warning("Using MockLLMClient - results are for testing only")
    else:
        try:
            from configs.client import get_llm_client
            llm_client = get_llm_client(
                base_url=args.api_base,
                api_key=args.api_key,
                model=args.model,
            )
            logger.info(f"Using LLM client: {args.model} @ {args.api_base}")
        except Exception as e:
            logger.error(f"Failed to create LLM client: {e}")
            from src.sigr_framework import MockLLMClient
            llm_client = MockLLMClient()
            logger.warning("Falling back to MockLLMClient")

    # Run experiment
    experiment = BaselineExperiment(
        kg_path=args.kg_path,
        llm_client=llm_client,
        task_name='geneattribute_dosage_sensitivity',
        max_workers=args.max_workers,
        output_dir=args.output_dir,
    )

    results = experiment.run_all_methods()

    logger.info("\nExperiment completed!")


if __name__ == '__main__':
    main()
