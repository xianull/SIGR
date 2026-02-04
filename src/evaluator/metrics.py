"""
Clustering and Regression Metrics for SIGR Evaluator

Computes clustering quality metrics for cell type and other clustering tasks.
Also provides regression metrics for perturbation prediction task.
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
    v_measure_score,
    mean_squared_error,
)
from sklearn.cluster import KMeans
from scipy.stats import pearsonr


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


# ============================================================================
# Regression Metrics for Perturbation Task
# ============================================================================

def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    top_k_de: int = 50,
) -> Dict[str, float]:
    """
    Compute regression metrics for perturbation prediction.

    For perturbation task, y_true and y_pred are (n_samples, n_genes) matrices
    representing expression change profiles (delta from control).

    Metrics computed:
    - delta_correlation: Mean Pearson correlation of predicted vs true delta (PRIMARY)
    - mse: Mean squared error across all samples
    - rmse: Root mean squared error
    - de_correlation: Correlation for top differentially expressed genes
    - de_mse: MSE for top differentially expressed genes

    Args:
        y_true: True expression change matrix (n_samples, n_genes)
        y_pred: Predicted expression change matrix (n_samples, n_genes)
        top_k_de: Number of top DE genes to consider for DE metrics

    Returns:
        Dictionary of regression metrics
    """
    if len(y_true) == 0 or len(y_pred) == 0:
        return _empty_regression_metrics()

    # Ensure 2D arrays
    if y_true.ndim == 1:
        y_true = y_true.reshape(1, -1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(1, -1)

    n_samples = len(y_true)
    n_genes = y_true.shape[1]

    all_corrs = []
    all_mses = []
    all_de_corrs = []
    all_de_mses = []

    for i in range(n_samples):
        true_row = y_true[i]
        pred_row = y_pred[i]

        # 1. MSE for this sample
        mse = mean_squared_error(true_row, pred_row)
        all_mses.append(mse)

        # 2. Delta correlation (expression change correlation)
        try:
            corr, _ = pearsonr(true_row, pred_row)
            if np.isnan(corr):
                corr = 0.0
        except Exception:
            corr = 0.0
        all_corrs.append(corr)

        # 3. DE genes metrics (top genes with largest true changes)
        if n_genes > top_k_de:
            # Find top DE genes by absolute change
            top_indices = np.argsort(np.abs(true_row))[-top_k_de:]

            true_de = true_row[top_indices]
            pred_de = pred_row[top_indices]

            de_mse = mean_squared_error(true_de, pred_de)
            try:
                de_corr, _ = pearsonr(true_de, pred_de)
                if np.isnan(de_corr):
                    de_corr = 0.0
            except Exception:
                de_corr = 0.0
        else:
            de_mse = mse
            de_corr = corr

        all_de_mses.append(de_mse)
        all_de_corrs.append(de_corr)

    # Compute means
    metrics = {
        'delta_correlation': float(np.mean(all_corrs)),
        'mse': float(np.mean(all_mses)),
        'rmse': float(np.sqrt(np.mean(all_mses))),
        'de_correlation': float(np.mean(all_de_corrs)),
        'de_mse': float(np.mean(all_de_mses)),
    }

    # Add std for correlation
    if n_samples > 1:
        metrics['delta_correlation_std'] = float(np.std(all_corrs))
        metrics['de_correlation_std'] = float(np.std(all_de_corrs))

    logger.info(
        f"Regression metrics: delta_corr={metrics['delta_correlation']:.4f}, "
        f"mse={metrics['mse']:.6f}, de_corr={metrics['de_correlation']:.4f}"
    )

    return metrics


def _empty_regression_metrics() -> Dict[str, float]:
    """Return empty regression metrics."""
    return {
        'delta_correlation': 0.0,
        'mse': float('inf'),
        'rmse': float('inf'),
        'de_correlation': 0.0,
        'de_mse': float('inf'),
    }


def compute_per_gene_correlation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    gene_names: Optional[list] = None,
) -> Dict[str, float]:
    """
    Compute correlation for each perturbation condition.

    Useful for analyzing which genes are well-predicted.

    Args:
        y_true: True expression change matrix (n_samples, n_genes)
        y_pred: Predicted expression change matrix (n_samples, n_genes)
        gene_names: Optional list of gene names for each sample

    Returns:
        Dictionary mapping gene name (or index) to correlation
    """
    if len(y_true) == 0:
        return {}

    n_samples = len(y_true)
    correlations = {}

    for i in range(n_samples):
        name = gene_names[i] if gene_names else str(i)

        try:
            corr, _ = pearsonr(y_true[i], y_pred[i])
            if np.isnan(corr):
                corr = 0.0
        except Exception:
            corr = 0.0

        correlations[name] = corr

    return correlations


def format_regression_metrics(metrics: Dict[str, float]) -> str:
    """
    Format regression metrics for display.

    Args:
        metrics: Dictionary of regression metrics

    Returns:
        Formatted string
    """
    lines = ["Regression Metrics (Perturbation):"]
    lines.append(f"  - Delta Correlation: {metrics.get('delta_correlation', 0):.4f}")
    if 'delta_correlation_std' in metrics:
        lines.append(f"    (std: {metrics['delta_correlation_std']:.4f})")
    lines.append(f"  - MSE: {metrics.get('mse', float('inf')):.6f}")
    lines.append(f"  - RMSE: {metrics.get('rmse', float('inf')):.6f}")
    lines.append(f"  - DE Correlation (top50): {metrics.get('de_correlation', 0):.4f}")
    lines.append(f"  - DE MSE (top50): {metrics.get('de_mse', float('inf')):.6f}")

    return "\n".join(lines)
