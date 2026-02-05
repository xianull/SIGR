#!/usr/bin/env python3
"""
True RAG Baseline Comparison Experiment

Compares four approaches for gene description generation:
1. No-KG: Pure LLM generation without graph knowledge
2. Full-KG: Use all available graph knowledge (rule-based)
3. Rule-RAG: Prioritized edge selection (current "KG-RAG")
4. True-RAG: Semantic retrieval-based KG-RAG (NEW)

This experiment evaluates how different retrieval strategies
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
from src.rag import IndexManager, RAGRetriever, RAGGenerator

logger = logging.getLogger(__name__)


# =============================================================================
# Prompts for Different Methods
# =============================================================================

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


RULE_RAG_PROMPT = """You are a computational biologist. Generate a concise description (100-150 words)
for gene {gene_symbol} using the following retrieved knowledge.

## Retrieved Context (prioritized by biological relevance):

### Key Phenotype Associations:
{phenotype_info}

### Regulatory Network Position:
{tf_info}

### Functional Annotations:
{go_info}

### Interaction Partners (top by score):
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

class TrueRAGBaselineExperiment:
    """
    Baseline comparison experiment with True KG-RAG.

    Compares No-KG, Full-KG, Rule-RAG, and True-RAG methods.
    """

    def __init__(
        self,
        kg_path: str,
        index_path: str,
        llm_client,
        task_name: str = 'geneattribute_dosage_sensitivity',
        max_workers: int = 8,
        output_dir: str = 'results/true_rag_baseline',
    ):
        """
        Initialize the experiment.

        Args:
            kg_path: Path to knowledge graph
            index_path: Path to pre-built RAG index
            llm_client: LLM client for text generation
            task_name: Downstream task name
            max_workers: Max concurrent LLM requests
            output_dir: Directory for results
        """
        self.llm = llm_client
        self.task_name = task_name
        self.max_workers = max_workers
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load KG
        logger.info(f"Loading knowledge graph from {kg_path}")
        self.kg = load_kg(kg_path)
        logger.info(f"KG: {self.kg.number_of_nodes()} nodes, {self.kg.number_of_edges()} edges")

        # Initialize encoder
        self.encoder = GeneEncoder(encoder_type='local')

        # Load RAG index
        logger.info(f"Loading RAG index from {index_path}")
        self.index_manager = IndexManager()
        self.index = self.index_manager.load_index(index_path)
        logger.info(f"Index: {self.index.num_triplets} triplets")

        # Initialize RAG retriever
        self.retriever = RAGRetriever(
            index=self.index,
            encoder=self.encoder,
            kg=self.kg,
            task_name=task_name,
        )

        # Initialize RAG generator
        self.rag_generator = RAGGenerator(
            retriever=self.retriever,
            llm_client=llm_client,
            task_name=task_name,
        )

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

    def _generate_descriptions_rule_based(
        self,
        prompt_template: str,
        use_kg: bool,
        kg_mode: str = 'full',
    ) -> Dict[str, str]:
        """Generate descriptions using rule-based methods."""
        descriptions = {}

        def generate_one(gene_id: str) -> tuple:
            try:
                if use_kg:
                    if kg_mode == 'rule_rag':
                        strategy = {
                            'edge_types': ['HPO', 'TRRUST', 'GO', 'PPI'],
                            'max_neighbors': 30,
                            'sampling': 'top_k',
                            'max_hops': 1,
                        }
                    else:  # full
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
                    formatted = format_subgraph(subgraph, center_gene=gene_id)

                    prompt = prompt_template.format(
                        gene_symbol=gene_id,
                        ppi_info=formatted.get('ppi_info', 'No interactions found.'),
                        go_info=formatted.get('go_info', 'No GO terms found.'),
                        phenotype_info=formatted.get('phenotype_info', 'No phenotypes found.'),
                        tf_info=formatted.get('tf_info', 'No regulatory information found.'),
                        pathway_info=formatted.get('pathway_info', 'No pathway information found.'),
                    )
                else:
                    prompt = prompt_template.format(gene_symbol=gene_id)

                description = self.llm.generate(prompt)
                return (gene_id, description, None)

            except Exception as e:
                return (gene_id, None, str(e))

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(generate_one, gid): gid
                for gid in self.task_genes
            }

            for future in tqdm(as_completed(futures), total=len(self.task_genes)):
                gene_id, desc, error = future.result()
                if desc:
                    descriptions[gene_id] = desc

        return descriptions

    def _encode_and_evaluate(
        self,
        descriptions: Dict[str, str],
        method_name: str,
    ) -> Dict[str, Any]:
        """Encode descriptions and evaluate."""
        if not descriptions:
            return {'error': 'No descriptions'}

        logger.info(f"Encoding {len(descriptions)} descriptions...")
        embeddings = self.encoder.encode_genes(
            descriptions,
            batch_size=64,
            show_progress=True
        )

        logger.info("Evaluating on downstream task...")
        metrics = self.evaluator.evaluate(embeddings)

        if 'combined' in metrics:
            logger.info(f"Results for {method_name}:")
            logger.info(f"  LR:  AUC={metrics['logistic'].get('auc', 0):.4f}")
            logger.info(f"  RF:  AUC={metrics['random_forest'].get('auc', 0):.4f}")
            logger.info(f"  Avg: AUC={metrics['combined'].get('auc', 0):.4f}")

        return {
            'method': method_name,
            'metrics': metrics,
            'num_descriptions': len(descriptions),
            'num_embeddings': len(embeddings),
        }

    def run_no_kg(self) -> Dict[str, Any]:
        """Run No-KG baseline."""
        logger.info("\n" + "=" * 60)
        logger.info("Method 1: NO-KG (Pure LLM)")
        logger.info("=" * 60)

        descriptions = self._generate_descriptions_rule_based(
            NO_KG_PROMPT,
            use_kg=False,
        )
        return self._encode_and_evaluate(descriptions, 'No-KG')

    def run_full_kg(self) -> Dict[str, Any]:
        """Run Full-KG baseline."""
        logger.info("\n" + "=" * 60)
        logger.info("Method 2: FULL-KG (All graph knowledge)")
        logger.info("=" * 60)

        descriptions = self._generate_descriptions_rule_based(
            FULL_KG_PROMPT,
            use_kg=True,
            kg_mode='full',
        )
        return self._encode_and_evaluate(descriptions, 'Full-KG')

    def run_rule_rag(self) -> Dict[str, Any]:
        """Run Rule-RAG baseline (current KG-RAG)."""
        logger.info("\n" + "=" * 60)
        logger.info("Method 3: RULE-RAG (Prioritized edges)")
        logger.info("=" * 60)

        descriptions = self._generate_descriptions_rule_based(
            RULE_RAG_PROMPT,
            use_kg=True,
            kg_mode='rule_rag',
        )
        return self._encode_and_evaluate(descriptions, 'Rule-RAG')

    def run_true_rag(self, top_k: int = 30) -> Dict[str, Any]:
        """
        Run True KG-RAG baseline (semantic retrieval).

        Args:
            top_k: Number of triplets to retrieve per gene
        """
        logger.info("\n" + "=" * 60)
        logger.info("Method 4: TRUE-RAG (Semantic retrieval)")
        logger.info("=" * 60)

        logger.info(f"Generating with top-{top_k} retrieved triplets...")
        descriptions = self.rag_generator.generate_batch(
            self.task_genes,
            top_k=top_k,
            max_workers=self.max_workers,
            show_progress=True,
            use_diversity=True,
        )

        return self._encode_and_evaluate(descriptions, 'True-RAG')

    def run_all_methods(self, run_true_rag: bool = True) -> Dict[str, Any]:
        """
        Run all baseline methods.

        Args:
            run_true_rag: Whether to run True-RAG (requires index)
        """
        results = {}
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Method 1: No-KG
        results['no_kg'] = self.run_no_kg()

        # Method 2: Full-KG
        results['full_kg'] = self.run_full_kg()

        # Method 3: Rule-RAG
        results['rule_rag'] = self.run_rule_rag()

        # Method 4: True-RAG
        if run_true_rag:
            results['true_rag'] = self.run_true_rag()

        # Print comparison
        self._print_comparison(results)

        # Save results
        results_file = self.output_dir / f'baseline_comparison_{timestamp}.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"\nResults saved to: {results_file}")

        return results

    def _print_comparison(self, results: Dict[str, Any]):
        """Print comparison table."""
        print("\n" + "=" * 80)
        print("BASELINE COMPARISON RESULTS (True KG-RAG)")
        print("=" * 80)
        print(f"Task: {self.task_name}")
        print(f"Genes: {len(self.task_genes)}")
        print("-" * 80)
        print(f"{'Method':<15} {'AUC (LR)':<12} {'AUC (RF)':<12} {'AUC (Avg)':<12} {'F1 (Avg)':<12}")
        print("-" * 80)

        for method_key in ['no_kg', 'full_kg', 'rule_rag', 'true_rag']:
            if method_key not in results or 'error' in results[method_key]:
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

        print("=" * 80)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run True RAG baseline comparison experiment"
    )

    parser.add_argument(
        '--kg-path',
        type=str,
        default='data/kg/sigr_kg_v3.pkl',
        help='Path to knowledge graph'
    )

    parser.add_argument(
        '--index-path',
        type=str,
        default='data/rag/triplet_index.pkl',
        help='Path to RAG index'
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
        help='LLM model name'
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
        default='results/true_rag_baseline',
        help='Output directory'
    )

    parser.add_argument(
        '--skip-true-rag',
        action='store_true',
        help='Skip True-RAG method'
    )

    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )

    return parser.parse_args()


def main():
    args = parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(level=log_level)

    logger.info("=" * 70)
    logger.info("True RAG Baseline Comparison Experiment")
    logger.info("=" * 70)
    logger.info(f"Task: geneattribute_dosage_sensitivity")
    logger.info(f"KG: {args.kg_path}")
    logger.info(f"Index: {args.index_path}")
    logger.info(f"Model: {args.model}")

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
            logger.info(f"Using LLM: {args.model} @ {args.api_base}")
        except Exception as e:
            logger.error(f"Failed to create LLM client: {e}")
            from src.sigr_framework import MockLLMClient
            llm_client = MockLLMClient()
            logger.warning("Falling back to MockLLMClient")

    # Run experiment
    experiment = TrueRAGBaselineExperiment(
        kg_path=args.kg_path,
        index_path=args.index_path,
        llm_client=llm_client,
        task_name='geneattribute_dosage_sensitivity',
        max_workers=args.max_workers,
        output_dir=args.output_dir,
    )

    results = experiment.run_all_methods(run_true_rag=not args.skip_true_rag)

    logger.info("\nExperiment completed!")


if __name__ == '__main__':
    main()
