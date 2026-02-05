"""
Memory - Knowledge Graph Memory System

Evolved from KGBOOK to support:
1. Successful strategy storage (original KGBook functionality)
2. Edge type effect tracking with EMA
3. Dynamic edge weight calculation

When optimization plateaus, Actor can reference Memory for inspiration.
Edge weights are dynamically adjusted based on historical effectiveness.
"""

import json
import logging
import os
import shutil
import tempfile
import threading
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

# All supported edge types
ALL_EDGE_TYPES = ['PPI', 'GO', 'HPO', 'TRRUST', 'CellMarker', 'Reactome']


@dataclass
class EdgeEffect:
    """Tracks the effect of an edge type on task performance."""
    usage_count: int = 0
    success_count: int = 0  # Number of times using this edge led to improvement
    total_improvement: float = 0.0
    ema_effect: float = 0.0  # Exponential moving average of improvement
    best_metric_with: float = 0.0  # Best metric achieved when using this edge type
    last_updated: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EdgeEffect':
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class Memory:
    """
    Knowledge Graph Memory System (evolved from KGBook).

    Stores and retrieves successful strategies across different tasks.
    Tracks edge type effectiveness and computes dynamic edge weights.

    Structure:
    {
        "version": "2.0",
        "global_edge_effects": {
            "PPI": {"usage_count": 10, "success_count": 7, "ema_effect": 0.023, ...},
            ...
        },
        "task_edge_effects": {
            "task_name": {
                "PPI": {"usage_count": 5, "success_count": 4, "ema_effect": 0.018, ...},
                ...
            }
        },
        "tasks": {
            "task_name": {
                "successful_strategies": [...],
                "best_strategy": {...},
                "best_metric": 0.93
            }
        }
    }
    """

    # EMA decay factor (higher = more weight to history)
    # 0.7 means 30% weight to new data (was 0.9 = 10%)
    EMA_DECAY = 0.7

    # Minimum samples before using learned weights
    # Lower threshold enables faster adaptation
    MIN_SAMPLES = 2

    # Learning rate for mixing prior and learned weights
    # 0.5 gives equal weight to prior and learned (was 0.3)
    LEARNING_RATE = 0.5

    # Base prior weights for edge types
    BASE_PRIOR_WEIGHTS = {
        'PPI': 0.7,
        'GO': 0.6,
        'HPO': 0.7,
        'TRRUST': 0.5,
        'CellMarker': 0.4,
        'Reactome': 0.5
    }

    # Task-specific prior weights
    TASK_PRIOR_WEIGHTS = {
        'geneattribute_dosage_sensitivity': {
            'HPO': 0.9, 'GO': 0.7, 'PPI': 0.6, 'TRRUST': 0.5, 'Reactome': 0.5, 'CellMarker': 0.4
        },
        'ppi': {
            'PPI': 0.9, 'GO': 0.5, 'HPO': 0.3, 'TRRUST': 0.4, 'Reactome': 0.4, 'CellMarker': 0.3
        },
        'genetype': {
            'GO': 0.8, 'HPO': 0.6, 'PPI': 0.5, 'TRRUST': 0.5, 'Reactome': 0.5, 'CellMarker': 0.4
        },
    }

    def __init__(self, memory_path: str = "data/memory.json"):
        """
        Initialize Memory.

        Args:
            memory_path: Path to the Memory JSON file
        """
        self.memory_path = Path(memory_path)
        self.data: Dict[str, Any] = self._get_empty_structure()
        self._file_lock = threading.Lock()
        self._load()

    def _get_empty_structure(self) -> Dict[str, Any]:
        """Return empty Memory structure."""
        return {
            "version": "2.0",
            "global_edge_effects": {},
            "task_edge_effects": {},
            "tasks": {}
        }

    def _load(self):
        """Load Memory from disk, with migration from old KGBook format.

        Note: Uses _file_lock to prevent race conditions during initialization.
        """
        with self._file_lock:
            if self.memory_path.exists():
                try:
                    with open(self.memory_path, 'r') as f:
                        loaded = json.load(f)

                    # Check if this is old KGBook format (no version field)
                    if "version" not in loaded:
                        logger.info("Migrating from old KGBook format to Memory v2.0")
                        self.data = self._migrate_from_kgbook(loaded)
                        self._save_unlocked()
                    else:
                        self.data = loaded

                    task_count = len(self.data.get("tasks", {}))
                    logger.info(f"Loaded Memory v{self.data.get('version', '1.0')} with {task_count} tasks")
                except Exception as e:
                    logger.warning(f"Error loading Memory: {e}, starting fresh")
                    self.data = self._get_empty_structure()
            else:
                # Try to load from old kgbook.json location
                old_path = Path("data/kgbook.json")
                if old_path.exists():
                    try:
                        with open(old_path, 'r') as f:
                            old_data = json.load(f)
                        logger.info("Migrating from kgbook.json to memory.json")
                        self.data = self._migrate_from_kgbook(old_data)
                        self._save_unlocked()
                    except Exception as e:
                        logger.warning(f"Error migrating from kgbook.json: {e}")
                        self.data = self._get_empty_structure()
                else:
                    logger.info("Memory not found, creating new one")
                    self.data = self._get_empty_structure()

    def _migrate_from_kgbook(self, old_data: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate old KGBook format to new Memory format."""
        new_data = self._get_empty_structure()
        new_data["tasks"] = old_data  # Old format is just the tasks dict
        return new_data

    def _save_unlocked(self):
        """Save without acquiring lock (caller must hold _file_lock)."""
        self.memory_path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            dir=self.memory_path.parent,
            suffix='.tmp'
        )
        try:
            with os.fdopen(fd, 'w') as f:
                json.dump(self.data, f, indent=2, default=str)
            shutil.move(tmp_path, self.memory_path)
            logger.debug(f"Memory saved to {self.memory_path}")
        except Exception as e:
            if os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
            logger.error(f"Error saving Memory: {e}")
            raise

    def _save(self):
        """原子性保存 Memory（write-then-rename）"""
        with self._file_lock:
            self._save_unlocked()

    # ========== Edge Effect Tracking (NEW) ==========

    def record_edge_effects(
        self,
        task_name: str,
        edge_types: List[str],
        metric_before: float,
        metric_after: float
    ):
        """
        Record the effect of edge types used in an iteration.

        Args:
            task_name: Name of the task
            edge_types: List of edge types used
            metric_before: Metric before this iteration
            metric_after: Metric after this iteration
        """
        improvement = metric_after - metric_before
        is_improvement = improvement > 0

        # Update global effects
        if "global_edge_effects" not in self.data:
            self.data["global_edge_effects"] = {}

        for edge_type in edge_types:
            self._update_edge_effect(
                self.data["global_edge_effects"],
                edge_type,
                improvement,
                metric_after,
                is_improvement
            )

        # Update task-specific effects
        if "task_edge_effects" not in self.data:
            self.data["task_edge_effects"] = {}
        if task_name not in self.data["task_edge_effects"]:
            self.data["task_edge_effects"][task_name] = {}

        for edge_type in edge_types:
            self._update_edge_effect(
                self.data["task_edge_effects"][task_name],
                edge_type,
                improvement,
                metric_after,
                is_improvement
            )

        self._save()

        # Log edge effects
        effect_strs = []
        for et in edge_types:
            effect = self.data["task_edge_effects"][task_name].get(et, {})
            usage = effect.get("usage_count", 0)
            success = effect.get("success_count", 0)
            ema = effect.get("ema_effect", 0)
            effect_strs.append(f"{et}(n={usage}, s={success}, ema={ema:+.4f})")
        logger.info(f"Memory: Recorded edge effects for {task_name}: {', '.join(effect_strs)}")

    def _update_edge_effect(
        self,
        effects: Dict[str, Dict],
        edge_type: str,
        improvement: float,
        metric: float,
        is_improvement: bool
    ):
        """Update effect statistics for a single edge type."""
        if edge_type not in effects:
            effects[edge_type] = EdgeEffect().to_dict()

        effect = effects[edge_type]
        effect["usage_count"] = effect.get("usage_count", 0) + 1
        if is_improvement:
            effect["success_count"] = effect.get("success_count", 0) + 1
        effect["total_improvement"] = effect.get("total_improvement", 0.0) + improvement

        # EMA update
        old_ema = effect.get("ema_effect", 0.0)
        if effect["usage_count"] == 1:
            effect["ema_effect"] = improvement
        else:
            effect["ema_effect"] = self.EMA_DECAY * old_ema + (1 - self.EMA_DECAY) * improvement

        # Update best metric
        if metric > effect.get("best_metric_with", 0.0):
            effect["best_metric_with"] = metric

        effect["last_updated"] = datetime.now().isoformat()

    def get_edge_weights(self, task_name: str) -> Dict[str, float]:
        """
        Compute dynamic edge weights for a task based on historical effectiveness.

        Args:
            task_name: Name of the task

        Returns:
            Dictionary mapping edge types to weights (0.1 to 1.0)
        """
        weights = {}

        for edge_type in ALL_EDGE_TYPES:
            weights[edge_type] = self._compute_single_weight(task_name, edge_type)

        # Normalize to [0.05, 1.0] range with non-linear scaling
        # This preserves more differentiation between edge types
        if weights:
            max_w = max(weights.values())
            min_w = min(weights.values())
            if max_w > min_w:
                for et in weights:
                    # Linear normalization to [0, 1]
                    normalized = (weights[et] - min_w) / (max_w - min_w)
                    # Apply power scaling to expand middle range differences
                    # 0.7 power makes differences more pronounced
                    scaled = normalized ** 0.7
                    # Map to [0.05, 1.0] - lower floor allows more differentiation
                    weights[et] = round(0.05 + 0.95 * scaled, 3)
            else:
                for et in weights:
                    weights[et] = 0.5

        # Log weights
        weight_str = ", ".join([f"{k}={v:.2f}" for k, v in sorted(weights.items(), key=lambda x: -x[1])])
        logger.info(f"Memory: Edge weights for {task_name}: {weight_str}")

        return weights

    def _compute_single_weight(self, task_name: str, edge_type: str) -> float:
        """Compute weight for a single edge type."""
        # Get prior weight
        task_priors = self.TASK_PRIOR_WEIGHTS.get(task_name, {})
        prior_weight = task_priors.get(edge_type, self.BASE_PRIOR_WEIGHTS.get(edge_type, 0.5))

        # Get learned effect (prefer task-specific, fallback to global)
        task_effects = self.data.get("task_edge_effects", {}).get(task_name, {})
        global_effects = self.data.get("global_edge_effects", {})

        effect = task_effects.get(edge_type) or global_effects.get(edge_type)

        if not effect or effect.get("usage_count", 0) < self.MIN_SAMPLES:
            # Not enough samples, use prior
            return prior_weight

        # Compute learned weight from success rate and EMA
        usage_count = effect.get("usage_count", 1)
        success_count = effect.get("success_count", 0)
        ema_effect = effect.get("ema_effect", 0.0)

        success_rate = success_count / usage_count

        # Normalize EMA to [0, 1] assuming improvement range is [-0.1, 0.1]
        normalized_ema = max(0, min(1, (ema_effect + 0.1) / 0.2))

        learned_weight = 0.6 * success_rate + 0.4 * normalized_ema

        # Mix prior and learned weights
        final_weight = (1 - self.LEARNING_RATE) * prior_weight + self.LEARNING_RATE * learned_weight

        return final_weight

    def get_task_edge_effect(self, task_name: str, edge_type: str) -> Optional[EdgeEffect]:
        """Get edge effect for a specific task."""
        task_effects = self.data.get("task_edge_effects", {}).get(task_name, {})
        effect_dict = task_effects.get(edge_type)
        return EdgeEffect.from_dict(effect_dict) if effect_dict else None

    def get_global_edge_effect(self, edge_type: str) -> Optional[EdgeEffect]:
        """Get global edge effect."""
        effect_dict = self.data.get("global_edge_effects", {}).get(edge_type)
        return EdgeEffect.from_dict(effect_dict) if effect_dict else None

    # ========== Original KGBook Methods (preserved for compatibility) ==========

    @property
    def book(self) -> Dict[str, Any]:
        """Compatibility property: access tasks as 'book'."""
        return self.data.get("tasks", {})

    def _init_task(self, task_name: str):
        """Initialize task entry if not exists."""
        if "tasks" not in self.data:
            self.data["tasks"] = {}
        if task_name not in self.data["tasks"]:
            self.data["tasks"][task_name] = {
                "successful_strategies": [],
                "best_strategy": None,
                "best_metric": 0.0,
                "last_updated": None
            }

    def record_improvement(
        self,
        task_name: str,
        strategy: Dict[str, Any],
        from_metric: float,
        to_metric: float,
        iteration: int,
        context: str = ""
    ):
        """
        Record a successful strategy that improved performance.

        Args:
            task_name: Name of the task
            strategy: Strategy configuration that led to improvement
            from_metric: Metric before this strategy
            to_metric: Metric after this strategy
            iteration: Iteration number
            context: Optional description of what improved
        """
        self._init_task(task_name)

        improvement = to_metric - from_metric

        # Only record if there's actual improvement
        if improvement <= 0:
            return

        entry = {
            "strategy": self._clean_strategy(strategy),
            "improvement": round(improvement, 4),
            "improvement_pct": round(improvement / max(from_metric, 0.001) * 100, 2),
            "from_metric": round(from_metric, 4),
            "to_metric": round(to_metric, 4),
            "iteration": iteration,
            "timestamp": datetime.now().isoformat(),
            "context": context
        }

        self.data["tasks"][task_name]["successful_strategies"].append(entry)

        # Update best strategy if this is the best
        if to_metric > self.data["tasks"][task_name]["best_metric"]:
            self.data["tasks"][task_name]["best_strategy"] = self._clean_strategy(strategy)
            self.data["tasks"][task_name]["best_metric"] = round(to_metric, 4)

        self.data["tasks"][task_name]["last_updated"] = datetime.now().isoformat()

        # Keep only top 10 strategies per task (by improvement)
        self.data["tasks"][task_name]["successful_strategies"] = sorted(
            self.data["tasks"][task_name]["successful_strategies"],
            key=lambda x: x["improvement"],
            reverse=True
        )[:10]

        self._save()
        logger.info(f"Memory: Recorded improvement +{improvement:.4f} for {task_name}")

    def _clean_strategy(self, strategy: Dict[str, Any]) -> Dict[str, Any]:
        """
        Clean strategy for storage (remove non-serializable fields).

        Args:
            strategy: Raw strategy dict

        Returns:
            Cleaned strategy dict
        """
        # Keep only the essential fields
        clean = {}
        essential_keys = [
            'edge_types', 'max_neighbors', 'max_hops', 'sampling',
            'description_length', 'description_focus', 'context_window',
            'prompt_style', 'feature_selection', 'focus_keywords',
            'include_statistics'
        ]

        for key in essential_keys:
            if key in strategy:
                value = strategy[key]
                # Convert to JSON-serializable
                if isinstance(value, (list, tuple)):
                    clean[key] = list(value)
                else:
                    clean[key] = value

        return clean

    def get_suggestions(
        self,
        task_name: str,
        current_strategy: Dict[str, Any] = None,
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Get strategy suggestions for a task (when plateaued).

        Args:
            task_name: Name of the task
            current_strategy: Current strategy (to avoid duplicates)
            top_k: Number of suggestions to return

        Returns:
            List of suggested strategies with their context
        """
        suggestions = []

        # 1. Get from same task
        if task_name in self.book:
            task_data = self.book[task_name]
            for entry in task_data.get("successful_strategies", [])[:top_k]:
                if not self._is_similar_strategy(entry["strategy"], current_strategy):
                    suggestions.append({
                        "source": "same_task",
                        "strategy": entry["strategy"],
                        "improvement": entry["improvement"],
                        "improvement_pct": entry.get("improvement_pct", 0),
                        "context": entry.get("context", "")
                    })

        # 2. Get from similar tasks (same task type prefix)
        task_prefix = task_name.split('_')[0]  # e.g., "geneattribute"
        for other_task, task_data in self.book.items():
            if other_task != task_name and other_task.startswith(task_prefix):
                for entry in task_data.get("successful_strategies", [])[:2]:
                    if not self._is_similar_strategy(entry["strategy"], current_strategy):
                        suggestions.append({
                            "source": f"similar_task:{other_task}",
                            "strategy": entry["strategy"],
                            "improvement": entry["improvement"],
                            "improvement_pct": entry.get("improvement_pct", 0),
                            "context": entry.get("context", "")
                        })

        # 3. Get best strategies from other tasks
        for other_task, task_data in self.book.items():
            if other_task != task_name and task_data.get("best_strategy"):
                if not self._is_similar_strategy(task_data["best_strategy"], current_strategy):
                    suggestions.append({
                        "source": f"best_from:{other_task}",
                        "strategy": task_data["best_strategy"],
                        "improvement": task_data["best_metric"],
                        "context": "Best strategy from another task"
                    })

        # Sort by improvement and return top_k
        suggestions = sorted(
            suggestions,
            key=lambda x: x.get("improvement", 0),
            reverse=True
        )[:top_k]

        return suggestions

    def _is_similar_strategy(
        self,
        strategy1: Dict[str, Any],
        strategy2: Dict[str, Any]
    ) -> bool:
        """
        Check if two strategies are similar (to avoid duplicates).

        Args:
            strategy1: First strategy
            strategy2: Second strategy

        Returns:
            True if strategies are similar
        """
        if strategy1 is None or strategy2 is None:
            return False

        # Compare edge_types (main differentiator)
        edges1 = set(strategy1.get('edge_types', []))
        edges2 = set(strategy2.get('edge_types', []))

        return edges1 == edges2

    def format_suggestions_for_prompt(
        self,
        suggestions: List[Dict[str, Any]]
    ) -> str:
        """
        Format suggestions as text for LLM prompt.

        Args:
            suggestions: List of suggestion dicts

        Returns:
            Formatted text for prompt
        """
        if not suggestions:
            return "No previous successful strategies found in KGBOOK."

        lines = ["## KGBOOK Suggestions (from previous successful runs):\n"]

        for i, sugg in enumerate(suggestions, 1):
            strategy = sugg["strategy"]
            edge_types = strategy.get("edge_types", [])
            improvement = sugg.get("improvement", 0)
            improvement_pct = sugg.get("improvement_pct", 0)
            source = sugg.get("source", "unknown")
            context = sugg.get("context", "")

            lines.append(f"### Suggestion {i} (from {source}):")
            lines.append(f"- Edge types: {edge_types}")
            lines.append(f"- Improvement: +{improvement:.4f} ({improvement_pct:.1f}%)")
            if context:
                lines.append(f"- Context: {context}")

            # Add other relevant params
            if strategy.get("max_neighbors"):
                lines.append(f"- Max neighbors: {strategy['max_neighbors']}")
            if strategy.get("description_focus"):
                lines.append(f"- Focus: {strategy['description_focus']}")
            lines.append("")

        return "\n".join(lines)

    def get_stats(self) -> Dict[str, Any]:
        """Get Memory statistics."""
        stats = {
            "version": self.data.get("version", "1.0"),
            "total_tasks": len(self.book),
            "total_strategies": sum(
                len(task.get("successful_strategies", []))
                for task in self.book.values()
            ),
            "global_edge_effects": {
                et: {
                    "usage": effect.get("usage_count", 0),
                    "success_rate": effect.get("success_count", 0) / max(effect.get("usage_count", 1), 1),
                    "ema_effect": effect.get("ema_effect", 0)
                }
                for et, effect in self.data.get("global_edge_effects", {}).items()
            },
            "tasks": {}
        }

        for task_name, task_data in self.book.items():
            stats["tasks"][task_name] = {
                "num_strategies": len(task_data.get("successful_strategies", [])),
                "best_metric": task_data.get("best_metric", 0),
                "last_updated": task_data.get("last_updated")
            }

        return stats

    # ========== Neighbor Selection Tracking (v3) ==========

    def record_neighbor_selection_effect(
        self,
        task_name: str,
        gene: str,
        selected_neighbors: List[str],
        excluded_neighbors: List[str],
        reward_delta: float
    ):
        """
        Record the effect of neighbor selection on performance.

        This helps the system learn which neighbors are typically noise
        vs useful for specific tasks.

        Args:
            task_name: Name of the task
            gene: Center gene ID
            selected_neighbors: List of selected neighbor IDs
            excluded_neighbors: List of excluded neighbor IDs
            reward_delta: Change in reward after this selection
        """
        # Initialize neighbor selection tracking if needed
        if "neighbor_selection_effects" not in self.data:
            self.data["neighbor_selection_effects"] = {}
        if task_name not in self.data["neighbor_selection_effects"]:
            self.data["neighbor_selection_effects"][task_name] = {
                "exclusion_effects": {},  # neighbor_id -> effect stats
                "selection_count": 0
            }

        task_effects = self.data["neighbor_selection_effects"][task_name]
        task_effects["selection_count"] = task_effects.get("selection_count", 0) + 1

        is_improvement = reward_delta > 0

        # Track effect of exclusions
        for neighbor_id in excluded_neighbors:
            if neighbor_id not in task_effects["exclusion_effects"]:
                task_effects["exclusion_effects"][neighbor_id] = {
                    "exclusion_count": 0,
                    "improvement_count": 0,
                    "total_improvement": 0.0,
                    "ema_effect": 0.0
                }

            effect = task_effects["exclusion_effects"][neighbor_id]
            effect["exclusion_count"] += 1

            if is_improvement:
                effect["improvement_count"] += 1

            effect["total_improvement"] += reward_delta

            # EMA update
            old_ema = effect.get("ema_effect", 0.0)
            if effect["exclusion_count"] == 1:
                effect["ema_effect"] = reward_delta
            else:
                effect["ema_effect"] = self.EMA_DECAY * old_ema + (1 - self.EMA_DECAY) * reward_delta

        self._save()

        if excluded_neighbors:
            logger.debug(
                f"Memory: Recorded neighbor selection for {task_name}: "
                f"excluded {len(excluded_neighbors)}, reward_delta={reward_delta:+.4f}"
            )

    def get_neighbor_selection_suggestions(
        self,
        task_name: str,
        neighbors: List[str]
    ) -> Dict[str, float]:
        """
        Get suggestions for neighbor selection based on history.

        Returns a score for each neighbor based on historical exclusion effects.
        Higher score = more likely to be beneficial to include (i.e., NOT exclude).

        Args:
            task_name: Name of the task
            neighbors: List of neighbor IDs to evaluate

        Returns:
            Dictionary mapping neighbor_id to score [0, 1]
        """
        suggestions = {}

        task_effects = self.data.get("neighbor_selection_effects", {}).get(task_name, {})
        exclusion_effects = task_effects.get("exclusion_effects", {})

        for neighbor_id in neighbors:
            if neighbor_id in exclusion_effects:
                effect = exclusion_effects[neighbor_id]
                exclusion_count = effect.get("exclusion_count", 0)
                improvement_count = effect.get("improvement_count", 0)
                ema = effect.get("ema_effect", 0.0)

                if exclusion_count > 0:
                    # If excluding this neighbor usually improves performance,
                    # it's likely noise (lower score = more likely to exclude)
                    exclusion_success_rate = improvement_count / exclusion_count

                    # Transform: if exclusion improves performance, neighbor is noise
                    # So we want to return a LOW score for noisy neighbors
                    # and a HIGH score for useful neighbors
                    # exclusion_success_rate high -> neighbor is noise -> low score
                    base_score = 1 - exclusion_success_rate

                    # Adjust based on EMA (positive EMA when excluded = noise)
                    # Normalize EMA to [-0.1, 0.1] range
                    ema_adjustment = -ema * 5  # Scale and negate
                    ema_adjustment = max(-0.3, min(0.3, ema_adjustment))

                    score = base_score + ema_adjustment
                    suggestions[neighbor_id] = max(0.0, min(1.0, score))
                else:
                    suggestions[neighbor_id] = 0.5  # Unknown
            else:
                suggestions[neighbor_id] = 0.5  # No history, neutral

        return suggestions

    def get_commonly_excluded_neighbors(
        self,
        task_name: str,
        min_exclusions: int = 3,
        min_success_rate: float = 0.6
    ) -> List[str]:
        """
        Get neighbors that are commonly excluded with positive effect.

        These are likely housekeeping genes or other noise sources.

        Args:
            task_name: Name of the task
            min_exclusions: Minimum times excluded to be considered
            min_success_rate: Minimum success rate when excluded

        Returns:
            List of neighbor IDs that are likely noise
        """
        noisy_neighbors = []

        task_effects = self.data.get("neighbor_selection_effects", {}).get(task_name, {})
        exclusion_effects = task_effects.get("exclusion_effects", {})

        for neighbor_id, effect in exclusion_effects.items():
            exclusion_count = effect.get("exclusion_count", 0)
            improvement_count = effect.get("improvement_count", 0)

            if exclusion_count >= min_exclusions:
                success_rate = improvement_count / exclusion_count
                if success_rate >= min_success_rate:
                    noisy_neighbors.append(neighbor_id)

        return noisy_neighbors

    def get_edge_effectiveness(self) -> Dict[str, Dict[str, Any]]:
        """
        Get edge type effectiveness data for neighbor scoring.

        Returns:
            Dictionary mapping edge_type to effectiveness stats
        """
        return self.data.get("global_edge_effects", {})


# Alias for backward compatibility
KGBook = Memory


# Global Memory instance with thread-safe initialization
_memory_instance: Optional[Memory] = None
_memory_lock = threading.Lock()


def get_memory(memory_path: str = "data/memory.json") -> Memory:
    """
    Get or create global Memory instance (thread-safe).

    Uses double-checked locking pattern for efficient thread-safe singleton.

    Args:
        memory_path: Path to Memory file

    Returns:
        Memory instance
    """
    global _memory_instance
    if _memory_instance is None:
        with _memory_lock:
            # Double-check after acquiring lock
            if _memory_instance is None:
                _memory_instance = Memory(memory_path)
    return _memory_instance


# Backward compatibility alias
def get_kgbook(book_path: str = "data/memory.json") -> Memory:
    """Backward compatibility wrapper for get_memory."""
    return get_memory(book_path)
