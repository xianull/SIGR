"""
Feedback Generation for SIGR Framework

Generates natural language feedback for the Actor based on evaluation results.
"""

from typing import Dict, Optional

import numpy as np


def generate_feedback(
    task_name: str,
    metrics: Dict[str, float],
    embeddings: Optional[Dict[str, np.ndarray]] = None,
    descriptions: Optional[Dict[str, str]] = None,
    best_metric: Optional[float] = None
) -> str:
    """
    Generate feedback for the Actor based on evaluation results.

    Args:
        task_name: Name of the task
        metrics: Dictionary of evaluation metrics
        embeddings: Optional gene embeddings for analysis
        descriptions: Optional gene descriptions for analysis
        best_metric: Optional best metric achieved so far (for comparison)

    Returns:
        Feedback string
    """
    feedback_parts = []

    # Overall performance summary
    display_name = _get_display_name(task_name)
    feedback_parts.append(f"=== Evaluation Results for {display_name} Task ===\n")

    # Report classification metrics
    feedback_parts.append("Classification Metrics:")
    for metric_name in ['accuracy', 'f1', 'auc', 'ap']:
        if metric_name in metrics:
            value = metrics[metric_name]
            std_key = f'{metric_name}_std'
            if std_key in metrics:
                feedback_parts.append(f"  - {metric_name}: {value:.4f} ± {metrics[std_key]:.4f}")
            else:
                feedback_parts.append(f"  - {metric_name}: {value:.4f}")

    # Report clustering metrics if present
    clustering_metrics = ['adjusted_rand_index', 'normalized_mutual_info',
                          'silhouette_score', 'v_measure']
    has_clustering = any(m in metrics for m in clustering_metrics)
    if has_clustering:
        feedback_parts.append("\nClustering Metrics:")
        for metric_name in clustering_metrics:
            if metric_name in metrics:
                feedback_parts.append(f"  - {metric_name}: {metrics[metric_name]:.4f}")

    # Performance assessment
    primary_metric = _get_primary_metric(task_name)
    primary_value = metrics.get(primary_metric, 0.0)

    feedback_parts.append(f"\nPrimary metric ({primary_metric}): {primary_value:.4f}")

    # More granular performance levels for better differentiation
    if primary_value >= 0.95:
        perf_feedback = "Performance: EXCEPTIONAL - Outstanding strategy performance."
    elif primary_value >= 0.92:
        perf_feedback = "Performance: EXCELLENT - The strategy is working very well."
    elif primary_value >= 0.88:
        perf_feedback = "Performance: VERY GOOD - Strong performance with minor room for improvement."
    elif primary_value >= 0.85:
        perf_feedback = "Performance: GOOD - The strategy is effective."
    elif primary_value >= 0.75:
        perf_feedback = "Performance: MODERATE - There is room for improvement."
    elif primary_value >= 0.65:
        perf_feedback = "Performance: BELOW AVERAGE - The strategy needs changes."
    elif primary_value >= 0.55:
        perf_feedback = "Performance: POOR - The strategy needs significant changes."
    else:
        perf_feedback = "Performance: VERY POOR - Consider a major strategy revision."

    # Add comparison with best metric if available
    if best_metric is not None:
        delta = primary_value - best_metric
        if delta > 0.001:
            perf_feedback += f" (↑{delta:.4f} from best, NEW BEST!)"
        elif delta < -0.01:
            perf_feedback += f" (↓{abs(delta):.4f} from best)"
        elif delta < -0.001:
            perf_feedback += f" (↓{abs(delta):.4f} from best, minor decline)"
        else:
            perf_feedback += " (≈ best performance)"

    feedback_parts.append(perf_feedback)

    # Task-specific suggestions
    suggestions = _get_task_suggestions(task_name, metrics)
    if suggestions:
        feedback_parts.append("\nSuggestions:")
        for suggestion in suggestions:
            feedback_parts.append(f"  - {suggestion}")

    # Embedding analysis if available
    if embeddings:
        embedding_analysis = _analyze_embeddings(embeddings)
        if embedding_analysis:
            feedback_parts.append(f"\nEmbedding Analysis:")
            feedback_parts.append(embedding_analysis)

    return "\n".join(feedback_parts)


def _get_display_name(task_name: str) -> str:
    """Get display name for task."""
    display_names = {
        'ppi': 'PPI Prediction',
        'genetype': 'Gene Type Classification',
        'ggi': 'Gene-Gene Interaction',
        'cell': 'Cell Type Classification',
        'geneattribute_dosage_sensitivity': 'Dosage Sensitivity',
        'geneattribute_lys4_only': 'H3K4me1 Methylation',
        'geneattribute_no_methylation': 'No Methylation',
        'geneattribute_bivalent': 'Bivalent Chromatin',
    }
    return display_names.get(task_name, task_name.upper())


def _get_primary_metric(task_name: str) -> str:
    """Get the primary metric for a task."""
    metrics = {
        'ppi': 'auc',
        'genetype': 'f1',
        'ggi': 'auc',
        'cell': 'f1',
        'phenotype': 'auc',
        'celltype': 'f1',
        'dosage': 'accuracy',
        'geneattribute_dosage_sensitivity': 'auc',
        'geneattribute_lys4_only': 'auc',
        'geneattribute_no_methylation': 'auc',
        'geneattribute_bivalent': 'auc',
    }
    return metrics.get(task_name, 'f1')


def _get_task_suggestions(task_name: str, metrics: Dict[str, float]) -> list:
    """Generate task-specific suggestions based on metrics."""
    suggestions = []
    primary_metric = _get_primary_metric(task_name)
    primary_value = metrics.get(primary_metric, 0.0)

    if task_name == 'ppi':
        if primary_value < 0.7:
            suggestions.append("Consider increasing PPI edge weight in the strategy")
            suggestions.append("Try including more protein interaction partners in descriptions")
            suggestions.append("Focus prompt on protein domains and binding sites")
        if metrics.get('f1', 0) < metrics.get('auc', 0) - 0.1:
            suggestions.append("Class imbalance detected - consider adjusting negative sampling")

    elif task_name == 'genetype':
        if primary_value < 0.7:
            suggestions.append("Prioritize GO (Gene Ontology) edges for functional annotation")
            suggestions.append("Focus prompt on molecular function and biological processes")
            suggestions.append("Consider using balanced class weights for minority gene types")

    elif task_name == 'ggi':
        if primary_value < 0.7:
            suggestions.append("Include both PPI and regulatory edges for interaction context")
            suggestions.append("Focus prompt on pathway membership and co-expression")
            suggestions.append("Consider using pre-trained gene2vec features as initialization")

    elif task_name == 'cell':
        if primary_value < 0.7:
            suggestions.append("Prioritize CellMarker edges for cell-type specificity")
            suggestions.append("Focus prompt on tissue-specific expression patterns")
            suggestions.append("Consider using top-expressed genes only (reduce noise)")
        # Clustering-specific suggestions
        ari = metrics.get('adjusted_rand_index', 0)
        if ari < 0.5:
            suggestions.append("Low clustering agreement - embeddings may lack cell-type specificity")

    elif task_name.startswith('geneattribute'):
        if primary_value < 0.7:
            suggestions.append("Focus on regulatory network information (TRRUST edges)")
            suggestions.append("Include GO biological process annotations")
        if 'dosage' in task_name:
            suggestions.append("Prioritize gene essentiality and constraint information")
        elif 'methylation' in task_name or 'lys4' in task_name or 'bivalent' in task_name:
            suggestions.append("Focus on chromatin state and epigenetic context")

    elif task_name == 'phenotype':
        if primary_value < 0.7:
            suggestions.append("Consider prioritizing HPO edges")
            suggestions.append("Focus prompt on disease associations and clinical features")

    elif task_name == 'celltype':
        if primary_value < 0.7:
            suggestions.append("Consider prioritizing CellMarker edges")
            suggestions.append("Focus prompt on tissue-specific expression patterns")

    elif task_name == 'dosage':
        if primary_value < 0.7:
            suggestions.append("Consider including both TRRUST and GO edges")
            suggestions.append("Focus prompt on gene essentiality and regulatory networks")

    # General suggestions
    if primary_value < 0.6:
        suggestions.append("Consider increasing max_neighbors to capture more context")
        suggestions.append("Try different sampling methods (e.g., weighted vs top_k)")

    return suggestions


def _analyze_embeddings(embeddings: Dict[str, np.ndarray]) -> str:
    """Analyze embedding statistics."""
    if not embeddings:
        return ""

    # Convert to array
    emb_array = np.stack(list(embeddings.values()))

    # Statistics
    mean_norm = np.mean(np.linalg.norm(emb_array, axis=1))
    std_norm = np.std(np.linalg.norm(emb_array, axis=1))

    # Variance per dimension
    dim_variance = np.var(emb_array, axis=0)
    mean_dim_var = np.mean(dim_variance)

    analysis = f"  - Number of embeddings: {len(embeddings)}\n"
    analysis += f"  - Mean embedding norm: {mean_norm:.4f}\n"
    analysis += f"  - Std embedding norm: {std_norm:.4f}\n"
    analysis += f"  - Mean dimension variance: {mean_dim_var:.6f}"

    return analysis
