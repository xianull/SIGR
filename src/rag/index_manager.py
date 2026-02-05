"""
FAISS Index Manager for KG-RAG

Builds, saves, and loads FAISS indices for efficient
semantic retrieval of KG triplets.
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

import numpy as np

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    faiss = None

from .triplet_textualizer import TripletTextualizer, Triplet

logger = logging.getLogger(__name__)


@dataclass
class TripletIndex:
    """
    FAISS index with triplet metadata.

    Attributes:
        embeddings: Numpy array of shape (num_triplets, embedding_dim)
        triplet_ids: List mapping index position to triplet ID
        triplet_texts: Dict mapping triplet_id to text
        triplet_metadata: Dict mapping triplet_id to metadata
        embedding_dim: Dimension of embeddings
        num_triplets: Number of triplets in index
    """
    embeddings: np.ndarray
    triplet_ids: List[str]
    triplet_texts: Dict[str, str]
    triplet_metadata: Dict[str, Dict[str, Any]]
    embedding_dim: int
    num_triplets: int = field(init=False)

    def __post_init__(self):
        self.num_triplets = len(self.triplet_ids)

    def get_triplet_by_idx(self, idx: int) -> Dict[str, Any]:
        """Get triplet info by index position."""
        if idx < 0 or idx >= self.num_triplets:
            return {}
        triplet_id = self.triplet_ids[idx]
        return {
            'triplet_id': triplet_id,
            'text': self.triplet_texts.get(triplet_id, ''),
            'metadata': self.triplet_metadata.get(triplet_id, {}),
        }


class IndexManager:
    """
    Manages FAISS index for KG triplets.

    Supports building indices from KG, saving/loading,
    and efficient similarity search.
    """

    def __init__(self, use_gpu: bool = False):
        """
        Initialize the index manager.

        Args:
            use_gpu: Whether to use GPU for FAISS operations
        """
        if not FAISS_AVAILABLE:
            raise ImportError(
                "FAISS is required for KG-RAG indexing. "
                "Install with: pip install faiss-cpu or pip install faiss-gpu"
            )
        self.use_gpu = use_gpu
        self._faiss_index: Optional[faiss.Index] = None

    def build_index(
        self,
        kg,
        encoder,
        batch_size: int = 256,
        include_edge_types: Optional[List[str]] = None,
        show_progress: bool = True,
    ) -> TripletIndex:
        """
        Build FAISS index from KG triplets.

        Args:
            kg: NetworkX DiGraph knowledge graph
            encoder: GeneEncoder instance for embedding
            batch_size: Batch size for encoding
            include_edge_types: Edge types to include (None = all)
            show_progress: Whether to show progress bar

        Returns:
            TripletIndex with embeddings and metadata
        """
        logger.info("Starting index building...")

        # 1. Extract triplets
        textualizer = TripletTextualizer(kg)
        triplets = textualizer.extract_all_triplets(
            include_edge_types=include_edge_types
        )
        logger.info(f"Extracted {len(triplets)} triplets")

        if not triplets:
            raise ValueError("No triplets extracted from KG")

        # 2. Prepare texts for encoding
        triplet_ids = [t.triplet_id for t in triplets]
        triplet_texts = {t.triplet_id: t.text for t in triplets}
        triplet_metadata = {
            t.triplet_id: {
                'source_node': t.source_node,
                'target_node': t.target_node,
                'edge_type': t.edge_type,
                **t.metadata
            }
            for t in triplets
        }
        texts = [t.text for t in triplets]

        # 3. Encode triplets
        logger.info(f"Encoding {len(texts)} triplets...")
        embeddings = encoder.encode_batch(
            texts,
            batch_size=batch_size,
            show_progress=show_progress
        )

        # Normalize embeddings for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        embeddings = embeddings / norms
        embeddings = embeddings.astype(np.float32)

        embedding_dim = embeddings.shape[1]
        logger.info(f"Embeddings shape: {embeddings.shape}")

        # 4. Build FAISS index
        logger.info("Building FAISS index...")
        self._faiss_index = faiss.IndexFlatIP(embedding_dim)

        if self.use_gpu:
            try:
                res = faiss.StandardGpuResources()
                self._faiss_index = faiss.index_cpu_to_gpu(res, 0, self._faiss_index)
                logger.info("Using GPU for FAISS index")
            except Exception as e:
                logger.warning(f"GPU not available, using CPU: {e}")

        self._faiss_index.add(embeddings)
        logger.info(f"Index built with {self._faiss_index.ntotal} vectors")

        # 5. Create TripletIndex
        index = TripletIndex(
            embeddings=embeddings,
            triplet_ids=triplet_ids,
            triplet_texts=triplet_texts,
            triplet_metadata=triplet_metadata,
            embedding_dim=embedding_dim,
        )

        return index

    def search(
        self,
        query_embedding: np.ndarray,
        index: TripletIndex,
        top_k: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar triplets.

        Args:
            query_embedding: Query embedding (1D or 2D array)
            index: TripletIndex to search
            top_k: Number of results to return

        Returns:
            List of dicts with triplet info and scores
        """
        # Ensure query is 2D and float32
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        query_embedding = query_embedding.astype(np.float32)

        # Normalize query
        norm = np.linalg.norm(query_embedding)
        if norm > 0:
            query_embedding = query_embedding / norm

        # Build temporary index if not cached
        if self._faiss_index is None or self._faiss_index.ntotal != index.num_triplets:
            self._faiss_index = faiss.IndexFlatIP(index.embedding_dim)
            self._faiss_index.add(index.embeddings)

        # Search
        scores, indices = self._faiss_index.search(query_embedding, min(top_k, index.num_triplets))

        # Build results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:  # Invalid index
                continue
            triplet_info = index.get_triplet_by_idx(idx)
            if triplet_info:
                triplet_info['score'] = float(score)
                results.append(triplet_info)

        return results

    def save_index(self, index: TripletIndex, path: str) -> None:
        """
        Save index to disk.

        Args:
            index: TripletIndex to save
            path: Output file path
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save as pickle (includes embeddings, metadata)
        with open(path, 'wb') as f:
            pickle.dump({
                'embeddings': index.embeddings,
                'triplet_ids': index.triplet_ids,
                'triplet_texts': index.triplet_texts,
                'triplet_metadata': index.triplet_metadata,
                'embedding_dim': index.embedding_dim,
            }, f)

        logger.info(f"Index saved to {path}")
        logger.info(f"  Triplets: {index.num_triplets}")
        logger.info(f"  Embedding dim: {index.embedding_dim}")
        logger.info(f"  File size: {path.stat().st_size / 1024 / 1024:.1f} MB")

    def load_index(self, path: str) -> TripletIndex:
        """
        Load index from disk.

        Args:
            path: Path to saved index

        Returns:
            TripletIndex
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Index file not found: {path}")

        with open(path, 'rb') as f:
            data = pickle.load(f)

        index = TripletIndex(
            embeddings=data['embeddings'],
            triplet_ids=data['triplet_ids'],
            triplet_texts=data['triplet_texts'],
            triplet_metadata=data['triplet_metadata'],
            embedding_dim=data['embedding_dim'],
        )

        # Rebuild FAISS index
        self._faiss_index = faiss.IndexFlatIP(index.embedding_dim)
        self._faiss_index.add(index.embeddings)

        logger.info(f"Index loaded from {path}")
        logger.info(f"  Triplets: {index.num_triplets}")
        logger.info(f"  Embedding dim: {index.embedding_dim}")

        return index


def get_index_stats(index: TripletIndex) -> Dict[str, Any]:
    """
    Get statistics about the index.

    Args:
        index: TripletIndex

    Returns:
        Dictionary with statistics
    """
    # Count by edge type
    edge_type_counts = {}
    for triplet_id, metadata in index.triplet_metadata.items():
        edge_type = metadata.get('edge_type', 'unknown')
        edge_type_counts[edge_type] = edge_type_counts.get(edge_type, 0) + 1

    # Embedding stats
    emb_mean = np.mean(index.embeddings)
    emb_std = np.std(index.embeddings)
    emb_norm_mean = np.mean(np.linalg.norm(index.embeddings, axis=1))

    return {
        'num_triplets': index.num_triplets,
        'embedding_dim': index.embedding_dim,
        'edge_type_counts': edge_type_counts,
        'embedding_stats': {
            'mean': float(emb_mean),
            'std': float(emb_std),
            'norm_mean': float(emb_norm_mean),
        },
        'memory_mb': index.embeddings.nbytes / 1024 / 1024,
    }
