#!/usr/bin/env python3
"""
Build RAG Index for KG-RAG

This script builds a FAISS index from KG triplets for
semantic retrieval in the True KG-RAG baseline.

Usage:
    python scripts/build_rag_index.py --kg-path data/kg/sigr_kg_v3.pkl

Output:
    data/rag/triplet_index.pkl - FAISS index with triplet embeddings
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import load_kg
from src.encoder import GeneEncoder
from src.rag import IndexManager, get_index_stats

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build FAISS index for KG-RAG"
    )

    parser.add_argument(
        '--kg-path',
        type=str,
        default='data/kg/sigr_kg_v3.pkl',
        help='Path to knowledge graph pickle file'
    )

    parser.add_argument(
        '--output-path',
        type=str,
        default='data/rag/triplet_index.pkl',
        help='Output path for index'
    )

    parser.add_argument(
        '--encoder-type',
        type=str,
        default='local',
        choices=['local', 'api'],
        help='Encoder type (local=SentenceTransformer, api=OpenAI)'
    )

    parser.add_argument(
        '--api-base',
        type=str,
        default='https://yunwu.ai/v1',
        help='API base URL (for api mode)'
    )

    parser.add_argument(
        '--api-key',
        type=str,
        default=None,
        help='API key (for api mode)'
    )

    parser.add_argument(
        '--api-model',
        type=str,
        default='text-embedding-ada-002',
        help='Embedding model (for api mode)'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=256,
        help='Batch size for encoding'
    )

    parser.add_argument(
        '--edge-types',
        type=str,
        nargs='+',
        default=None,
        help='Edge types to include (default: all)'
    )

    parser.add_argument(
        '--use-gpu',
        action='store_true',
        help='Use GPU for FAISS (requires faiss-gpu)'
    )

    return parser.parse_args()


def main():
    args = parse_args()

    logger.info("=" * 60)
    logger.info("Building RAG Index")
    logger.info("=" * 60)
    logger.info(f"KG path: {args.kg_path}")
    logger.info(f"Output path: {args.output_path}")
    logger.info(f"Encoder: {args.encoder_type}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Edge types: {args.edge_types or 'all'}")

    # 1. Load KG
    logger.info("\n[1/4] Loading knowledge graph...")
    kg = load_kg(args.kg_path)
    logger.info(f"Loaded KG: {kg.number_of_nodes()} nodes, {kg.number_of_edges()} edges")

    # 2. Initialize encoder
    logger.info("\n[2/4] Initializing encoder...")
    if args.encoder_type == 'api':
        encoder = GeneEncoder(
            encoder_type='api',
            api_base=args.api_base,
            api_key=args.api_key,
            api_model=args.api_model,
        )
    else:
        encoder = GeneEncoder(encoder_type='local')

    logger.info(f"Encoder ready: {args.encoder_type}, dim={encoder.embedding_dim}")

    # 3. Build index
    logger.info("\n[3/4] Building FAISS index...")
    start_time = datetime.now()

    index_manager = IndexManager(use_gpu=args.use_gpu)
    index = index_manager.build_index(
        kg=kg,
        encoder=encoder,
        batch_size=args.batch_size,
        include_edge_types=args.edge_types,
        show_progress=True,
    )

    build_time = (datetime.now() - start_time).total_seconds()
    logger.info(f"Index built in {build_time:.1f} seconds")

    # 4. Save index
    logger.info("\n[4/4] Saving index...")
    index_manager.save_index(index, args.output_path)

    # Print stats
    stats = get_index_stats(index)
    logger.info("\n" + "=" * 60)
    logger.info("Index Statistics")
    logger.info("=" * 60)
    logger.info(f"Total triplets: {stats['num_triplets']:,}")
    logger.info(f"Embedding dimension: {stats['embedding_dim']}")
    logger.info(f"Memory usage: {stats['memory_mb']:.1f} MB")
    logger.info("\nEdge type distribution:")
    for edge_type, count in sorted(stats['edge_type_counts'].items(), key=lambda x: -x[1]):
        logger.info(f"  {edge_type}: {count:,}")

    logger.info("\n" + "=" * 60)
    logger.info("Index building complete!")
    logger.info(f"Output: {args.output_path}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
