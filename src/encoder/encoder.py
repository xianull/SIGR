"""
Gene Encoder for SIGR Framework

Encodes gene descriptions into vector representations using Sentence Transformers.
"""

import logging
from typing import List, Dict, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


class GeneEncoder:
    """
    Encoder for gene descriptions using Sentence Transformers.

    Uses the all-MiniLM-L6-v2 model by default, which produces 384-dimensional
    embeddings and is optimized for semantic similarity tasks.
    """

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the encoder.

        Args:
            model_name: Name of the Sentence Transformer model to use
        """
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        """Lazy load the model on first use."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info(f"Loading Sentence Transformer model: {self.model_name}")
                self._model = SentenceTransformer(self.model_name)
                logger.info(f"Model loaded. Embedding dimension: {self._model.get_sentence_embedding_dimension()}")
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required for encoding. "
                    "Install it with: pip install sentence-transformers"
                )
        return self._model

    @property
    def embedding_dim(self) -> int:
        """Get the embedding dimension."""
        return self.model.get_sentence_embedding_dimension()

    def encode(
        self,
        gene_id: str,
        generated_context: str,
        original_desc: Optional[str] = None
    ) -> np.ndarray:
        """
        Encode a gene's text description into a vector.

        Args:
            gene_id: Gene symbol (used for logging)
            generated_context: The LLM-generated description
            original_desc: Optional original gene description to prepend

        Returns:
            Numpy array of shape (embedding_dim,)
        """
        # Combine original description with generated context
        if original_desc:
            combined_text = f"{original_desc} {generated_context}"
        else:
            combined_text = generated_context

        # Encode
        embedding = self.model.encode(combined_text, convert_to_numpy=True)
        return embedding

    def encode_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = False
    ) -> np.ndarray:
        """
        Encode multiple texts in a batch.

        Args:
            texts: List of text strings to encode
            batch_size: Batch size for encoding
            show_progress: Whether to show a progress bar

        Returns:
            Numpy array of shape (num_texts, embedding_dim)
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        return embeddings

    def postprocess_embeddings(
        self,
        embeddings: np.ndarray,
        normalize: bool = True,
        add_noise: bool = False,
        noise_scale: float = 0.01
    ) -> np.ndarray:
        """
        Post-process embeddings to improve discriminability.

        Args:
            embeddings: Raw embeddings (N, D)
            normalize: Whether to L2 normalize
            add_noise: Whether to add small noise for diversity
            noise_scale: Noise magnitude

        Returns:
            Processed embeddings
        """
        processed = embeddings.copy()

        # 1. Center (subtract mean)
        mean = np.mean(processed, axis=0)
        processed = processed - mean

        # 2. Optional: add small noise for diversity
        if add_noise:
            noise = np.random.normal(0, noise_scale, processed.shape)
            processed = processed + noise

        # 3. L2 normalize
        if normalize:
            norms = np.linalg.norm(processed, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-8)  # Avoid division by zero
            processed = processed / norms

        return processed

    def encode_genes(
        self,
        gene_descriptions: Dict[str, str],
        batch_size: int = 32,
        show_progress: bool = True,
        postprocess: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Encode descriptions for multiple genes.

        Args:
            gene_descriptions: Dictionary mapping gene_id to description
            batch_size: Batch size for encoding
            show_progress: Whether to show a progress bar
            postprocess: Whether to apply post-processing (centering + normalization)

        Returns:
            Dictionary mapping gene_id to embedding
        """
        if not gene_descriptions:
            return {}

        gene_ids = list(gene_descriptions.keys())
        texts = [gene_descriptions[gid] for gid in gene_ids]

        embeddings = self.encode_batch(texts, batch_size=batch_size, show_progress=show_progress)

        # Apply post-processing to improve discriminability
        if postprocess:
            embeddings = self.postprocess_embeddings(embeddings, normalize=True)

        return {gid: emb for gid, emb in zip(gene_ids, embeddings)}

    def similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray
    ) -> float:
        """
        Compute cosine similarity between two embeddings.

        Args:
            embedding1: First embedding
            embedding2: Second embedding

        Returns:
            Cosine similarity score
        """
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(embedding1, embedding2) / (norm1 * norm2))
