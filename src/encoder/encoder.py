"""
Gene Encoder for SIGR Framework

Encodes gene descriptions into vector representations.
Supports two modes:
- local: Uses Sentence Transformers (default, 384-dim)
- api: Uses OpenAI-compatible embedding API (e.g., text-embedding-ada-002, 1536-dim)
"""

import os
import logging
from typing import List, Dict, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)

# Embedding dimensions for known models
EMBEDDING_DIMS = {
    'text-embedding-ada-002': 1536,
    'text-embedding-3-small': 1536,
    'text-embedding-3-large': 3072,
}


class GeneEncoder:
    """
    Encoder for gene descriptions.

    Supports two modes:
    - 'local': Uses Sentence Transformers (default)
    - 'api': Uses OpenAI-compatible embedding API
    """

    def __init__(
        self,
        model_name: str = 'all-MiniLM-L6-v2',
        encoder_type: str = 'local',
        api_base: str = 'https://yunwu.ai/v1',
        api_key: Optional[str] = None,
        api_model: str = 'text-embedding-ada-002',
        max_connections: int = 20,
        timeout: float = 120.0
    ):
        """
        Initialize the encoder.

        Args:
            model_name: Name of the Sentence Transformer model (for local mode)
            encoder_type: 'local' for SentenceTransformer, 'api' for OpenAI API
            api_base: API base URL (for api mode)
            api_key: API key (falls back to OPENAI_API_KEY env var)
            api_model: Embedding model name (for api mode)
            max_connections: Maximum concurrent connections (for api mode)
            timeout: Request timeout in seconds (for api mode)
        """
        self.encoder_type = encoder_type

        if encoder_type == 'api':
            self._init_api_client(api_base, api_key, api_model, max_connections, timeout)
        else:
            self.model_name = model_name
            self._model = None

    def _init_api_client(
        self,
        api_base: str,
        api_key: Optional[str],
        api_model: str,
        max_connections: int,
        timeout: float
    ):
        """Initialize OpenAI embedding API client."""
        try:
            import httpx
        except ImportError:
            raise ImportError(
                "httpx is required for API mode. Install with: pip install httpx>=0.24.0"
            )

        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai package is required for API mode. Install with: pip install openai>=1.0.0"
            )

        self.api_model = api_model
        self._api_key = api_key or os.getenv("OPENAI_API_KEY")

        if not self._api_key:
            raise ValueError(
                "API key required for API mode. Set OPENAI_API_KEY env var or pass api_key parameter."
            )

        # Create HTTP client with connection pooling
        self._http_client = httpx.Client(
            limits=httpx.Limits(
                max_connections=max_connections,
                max_keepalive_connections=max_connections
            ),
            timeout=httpx.Timeout(timeout)
        )

        self._openai_client = OpenAI(
            base_url=api_base,
            api_key=self._api_key,
            http_client=self._http_client
        )

        # Set embedding dimension based on model
        self._embedding_dim = EMBEDDING_DIMS.get(api_model, 1536)

        logger.info(f"GeneEncoder initialized in API mode")
        logger.info(f"  API base: {api_base}")
        logger.info(f"  Model: {api_model}")
        logger.info(f"  Embedding dimension: {self._embedding_dim}")

    @property
    def model(self):
        """Lazy load the local model on first use."""
        if self.encoder_type == 'api':
            raise RuntimeError("Local model not available in API mode")

        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info(f"Loading Sentence Transformer model: {self.model_name}")
                self._model = SentenceTransformer(self.model_name)
                logger.info(f"Model loaded. Embedding dimension: {self._model.get_sentence_embedding_dimension()}")
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required for local mode. "
                    "Install it with: pip install sentence-transformers"
                )
        return self._model

    @property
    def embedding_dim(self) -> int:
        """Get the embedding dimension."""
        if self.encoder_type == 'api':
            return self._embedding_dim
        else:
            return self.model.get_sentence_embedding_dimension()

    def _encode_via_api(
        self,
        texts: List[str],
        batch_size: int = 100,
        show_progress: bool = False
    ) -> np.ndarray:
        """
        Encode texts via OpenAI embedding API.

        Args:
            texts: List of texts to encode
            batch_size: Batch size for API calls (max 2048 for OpenAI)
            show_progress: Whether to show progress bar

        Returns:
            Numpy array of shape (num_texts, embedding_dim)
        """
        all_embeddings = []
        total_batches = (len(texts) + batch_size - 1) // batch_size

        if show_progress:
            try:
                from tqdm import tqdm
                batch_iter = tqdm(range(0, len(texts), batch_size), total=total_batches, desc="Encoding (API)")
            except ImportError:
                batch_iter = range(0, len(texts), batch_size)
        else:
            batch_iter = range(0, len(texts), batch_size)

        for i in batch_iter:
            batch = texts[i:i+batch_size]

            try:
                response = self._openai_client.embeddings.create(
                    model=self.api_model,
                    input=batch
                )

                # Extract embeddings in correct order (API may return out of order)
                batch_embeddings = [None] * len(batch)
                for item in response.data:
                    batch_embeddings[item.index] = item.embedding

                all_embeddings.extend(batch_embeddings)

            except Exception as e:
                logger.error(f"Embedding API error: {e}")
                raise

        return np.array(all_embeddings)

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

        # Encode based on mode
        if self.encoder_type == 'api':
            embeddings = self._encode_via_api([combined_text], show_progress=False)
            return embeddings[0]
        else:
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
        if self.encoder_type == 'api':
            # API mode: use API with appropriate batch size
            # OpenAI recommends max 2048 inputs per request, but we limit to 100 for safety
            api_batch_size = min(batch_size, 100)
            return self._encode_via_api(texts, batch_size=api_batch_size, show_progress=show_progress)
        else:
            # Local mode: use SentenceTransformer
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

    def close(self):
        """Close HTTP client and release resources."""
        if self.encoder_type == 'api' and hasattr(self, '_http_client'):
            try:
                self._http_client.close()
            except Exception as e:
                logger.debug(f"Error closing HTTP client: {e}")

    def __del__(self):
        """Destructor to close connections."""
        self.close()

    def __enter__(self):
        """Support context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Close connections on exit."""
        self.close()
        return False
