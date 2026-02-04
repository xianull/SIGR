"""
Clustering Metrics for SIGR Evaluator

Computes clustering quality metrics for cell type and other clustering tasks.
"""

import logging
from typing import Dict, Optional

import numpy as np
from sklearn.metrics import (
    silhouette_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    homogeneity_score,
    completeness_score,
    v_measure_score
)
from sklearn.cluster import KMeans


logger = logging.getLogger(__name__)


def compute_clustering_metrics(
    X: np.ndarray,
    y_true: np.ndarray,
    n_clusters: Optional[int] = None,
    random_state: int = 42
) -> Dict[str, float]:
    """
    Compute clustering quality metrics.

    Performs KMeans clustering and compares results with true labels.

    Args:
        X: Feature matrix (n_samples, n_features)
        y_true: True labels
        n_clusters: Number of clusters (defaults to number of unique labels)
        random_state: Random seed for KMeans

    Returns:
        Dictionary of clustering metrics
    """
    if len(X) == 0 or len(y_true) == 0:
        return _empty_clustering_metrics()

    if n_clusters is None:
        n_clusters = len(np.unique(y_true))

    # Perform KMeans clustering
    try:
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=10,
            max_iter=300
        )
        y_pred = kmeans.fit_predict(X)
    except Exception as e:
        logger.warning(f"KMeans clustering failed: {e}")
        return _empty_clustering_metrics()

    metrics = {}

    # External metrics (compare with true labels)
    try:
        metrics['adjusted_rand_index'] = adjusted_rand_score(y_true, y_pred)
        metrics['normalized_mutual_info'] = normalized_mutual_info_score(
            y_true, y_pred, average_method='arithmetic'
        )
        metrics['homogeneity'] = homogeneity_score(y_true, y_pred)
        metrics['completeness'] = completeness_score(y_true, y_pred)
        metrics['v_measure'] = v_measure_score(y_true, y_pred)
    except Exception as e:
        logger.warning(f"Error computing external metrics: {e}")
        metrics['adjusted_rand_index'] = 0.0
        metrics['normalized_mutual_info'] = 0.0
        metrics['homogeneity'] = 0.0
        metrics['completeness'] = 0.0
        metrics['v_measure'] = 0.0

    # Internal metrics (based on cluster quality)
    n_unique_pred = len(np.unique(y_pred))
    if n_unique_pred > 1 and n_unique_pred < len(X):
        try:
            metrics['silhouette_score'] = silhouette_score(X, y_pred)
            metrics['calinski_harabasz'] = calinski_harabasz_score(X, y_pred)
            metrics['davies_bouldin'] = davies_bouldin_score(X, y_pred)
        except Exception as e:
            logger.warning(f"Error computing internal metrics: {e}")
            metrics['silhouette_score'] = 0.0
            metrics['calinski_harabasz'] = 0.0
            metrics['davies_bouldin'] = float('inf')
    else:
        metrics['silhouette_score'] = 0.0
        metrics['calinski_harabasz'] = 0.0
        metrics['davies_bouldin'] = float('inf')

    logger.info(f"Clustering metrics: ARI={metrics['adjusted_rand_index']:.4f}, "
                f"NMI={metrics['normalized_mutual_info']:.4f}, "
                f"Silhouette={metrics['silhouette_score']:.4f}")

    return metrics


def _empty_clustering_metrics() -> Dict[str, float]:
    """Return empty clustering metrics."""
    return {
        'adjusted_rand_index': 0.0,
        'normalized_mutual_info': 0.0,
        'homogeneity': 0.0,
        'completeness': 0.0,
        'v_measure': 0.0,
        'silhouette_score': 0.0,
        'calinski_harabasz': 0.0,
        'davies_bouldin': float('inf'),
    }


def compute_embedding_statistics(embeddings: np.ndarray) -> Dict[str, float]:
    """
    Compute statistics about embeddings.

    Args:
        embeddings: Embedding matrix (n_samples, n_features)

    Returns:
        Dictionary of embedding statistics
    """
    if len(embeddings) == 0:
        return {
            'num_embeddings': 0,
            'embedding_dim': 0,
            'mean_norm': 0.0,
            'std_norm': 0.0,
            'mean_dim_variance': 0.0,
        }

    norms = np.linalg.norm(embeddings, axis=1)
    dim_variance = np.var(embeddings, axis=0)

    return {
        'num_embeddings': len(embeddings),
        'embedding_dim': embeddings.shape[1],
        'mean_norm': float(np.mean(norms)),
        'std_norm': float(np.std(norms)),
        'mean_dim_variance': float(np.mean(dim_variance)),
        'max_dim_variance': float(np.max(dim_variance)),
        'min_dim_variance': float(np.min(dim_variance)),
    }


def format_clustering_metrics(metrics: Dict[str, float]) -> str:
    """
    Format clustering metrics for display.

    Args:
        metrics: Dictionary of clustering metrics

    Returns:
        Formatted string
    """
    lines = ["Clustering Metrics:"]

    # External metrics
    lines.append("  External (vs true labels):")
    lines.append(f"    - Adjusted Rand Index: {metrics.get('adjusted_rand_index', 0):.4f}")
    lines.append(f"    - Normalized Mutual Info: {metrics.get('normalized_mutual_info', 0):.4f}")
    lines.append(f"    - V-measure: {metrics.get('v_measure', 0):.4f}")

    # Internal metrics
    lines.append("  Internal (cluster quality):")
    lines.append(f"    - Silhouette Score: {metrics.get('silhouette_score', 0):.4f}")
    lines.append(f"    - Calinski-Harabasz: {metrics.get('calinski_harabasz', 0):.2f}")
    lines.append(f"    - Davies-Bouldin: {metrics.get('davies_bouldin', float('inf')):.4f}")

    return "\n".join(lines)
