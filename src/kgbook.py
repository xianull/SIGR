"""
KGBOOK - Knowledge Graph Strategy Book

Stores successful strategies across tasks for experience replay.
When optimization plateaus, Actor can reference KGBOOK for inspiration.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class KGBook:
    """
    Knowledge Graph Strategy Book.

    Stores and retrieves successful strategies across different tasks.
    Enables cross-task learning and plateau recovery.

    Structure:
    {
        "task_name": {
            "successful_strategies": [
                {
                    "strategy": {...},
                    "improvement": 0.05,
                    "from_metric": 0.85,
                    "to_metric": 0.90,
                    "iteration": 3,
                    "timestamp": "...",
                    "context": "description of what improved"
                }
            ],
            "best_strategy": {...},
            "best_metric": 0.93
        }
    }
    """

    def __init__(self, book_path: str = "data/kgbook.json"):
        """
        Initialize KGBOOK.

        Args:
            book_path: Path to the KGBOOK JSON file
        """
        self.book_path = Path(book_path)
        self.book: Dict[str, Any] = {}
        self._load()

    def _load(self):
        """Load KGBOOK from disk."""
        if self.book_path.exists():
            try:
                with open(self.book_path, 'r') as f:
                    self.book = json.load(f)
                logger.info(f"Loaded KGBOOK with {len(self.book)} tasks")
            except Exception as e:
                logger.warning(f"Error loading KGBOOK: {e}, starting fresh")
                self.book = {}
        else:
            logger.info("KGBOOK not found, creating new one")
            self.book = {}

    def _save(self):
        """Save KGBOOK to disk."""
        self.book_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.book_path, 'w') as f:
            json.dump(self.book, f, indent=2, default=str)
        logger.debug(f"KGBOOK saved to {self.book_path}")

    def _init_task(self, task_name: str):
        """Initialize task entry if not exists."""
        if task_name not in self.book:
            self.book[task_name] = {
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

        self.book[task_name]["successful_strategies"].append(entry)

        # Update best strategy if this is the best
        if to_metric > self.book[task_name]["best_metric"]:
            self.book[task_name]["best_strategy"] = self._clean_strategy(strategy)
            self.book[task_name]["best_metric"] = round(to_metric, 4)

        self.book[task_name]["last_updated"] = datetime.now().isoformat()

        # Keep only top 10 strategies per task (by improvement)
        self.book[task_name]["successful_strategies"] = sorted(
            self.book[task_name]["successful_strategies"],
            key=lambda x: x["improvement"],
            reverse=True
        )[:10]

        self._save()
        logger.info(f"KGBOOK: Recorded improvement +{improvement:.4f} for {task_name}")

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
                        "improvement_pct": entry["improvement_pct"],
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
                            "improvement_pct": entry["improvement_pct"],
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
        """Get KGBOOK statistics."""
        stats = {
            "total_tasks": len(self.book),
            "total_strategies": sum(
                len(task.get("successful_strategies", []))
                for task in self.book.values()
            ),
            "tasks": {}
        }

        for task_name, task_data in self.book.items():
            stats["tasks"][task_name] = {
                "num_strategies": len(task_data.get("successful_strategies", [])),
                "best_metric": task_data.get("best_metric", 0),
                "last_updated": task_data.get("last_updated")
            }

        return stats


# Global KGBOOK instance
_kgbook_instance: Optional[KGBook] = None


def get_kgbook(book_path: str = "data/kgbook.json") -> KGBook:
    """
    Get or create global KGBOOK instance.

    Args:
        book_path: Path to KGBOOK file

    Returns:
        KGBook instance
    """
    global _kgbook_instance
    if _kgbook_instance is None:
        _kgbook_instance = KGBook(book_path)
    return _kgbook_instance
