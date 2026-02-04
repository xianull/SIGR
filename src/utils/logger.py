"""
Logger for SIGR Framework

Comprehensive logging system for tracking training iterations,
strategies, descriptions, evaluations, and reflections.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List


logger = logging.getLogger(__name__)


class SIGRLogger:
    """
    Logger for SIGR training process.

    Creates structured logs for each iteration including:
    - Strategy configurations
    - Gene descriptions
    - Evaluation metrics
    - Feedback from evaluator
    - Actor reflections

    Each training run is saved with a timestamp to avoid overwriting.
    """

    def __init__(self, task_name: str, log_dir: str = "logs"):
        """
        Initialize the logger.

        Args:
            task_name: Name of the downstream task
            log_dir: Root directory for logs
        """
        self.task_name = task_name

        # Add timestamp to create unique run directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = f"{task_name}_{timestamp}"
        self.log_dir = Path(log_dir) / self.run_id
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.current_iteration = 0
        self.iter_dir: Optional[Path] = None

        # Summary tracking
        self.summary: List[Dict[str, Any]] = []

        logger.info(f"SIGRLogger initialized for task: {task_name}")
        logger.info(f"Run ID: {self.run_id}")
        logger.info(f"Log directory: {self.log_dir}")

    def start_iteration(self, iteration: int):
        """
        Start a new iteration.

        Creates iteration directory and prepares for logging.

        Args:
            iteration: Iteration number (0-indexed)
        """
        self.current_iteration = iteration
        self.iter_dir = self.log_dir / f"iteration_{iteration:03d}"
        self.iter_dir.mkdir(exist_ok=True)

        # Create gene descriptions subdirectory
        (self.iter_dir / "gene_descriptions").mkdir(exist_ok=True)

        logger.info(f"Started iteration {iteration}")

    def log_strategy(self, strategy: Dict[str, Any]):
        """
        Log the current strategy.

        Args:
            strategy: Strategy configuration dictionary
        """
        if self.iter_dir is None:
            raise RuntimeError("Must call start_iteration first")

        strategy_file = self.iter_dir / "strategy.json"
        with open(strategy_file, 'w') as f:
            json.dump(strategy, f, indent=2)

        logger.debug(f"Logged strategy to {strategy_file}")

    def log_gene_description(self, gene_id: str, description: str):
        """
        Log a gene description.

        Args:
            gene_id: Gene identifier
            description: Generated description
        """
        if self.iter_dir is None:
            raise RuntimeError("Must call start_iteration first")

        desc_dir = self.iter_dir / "gene_descriptions"
        # Sanitize gene_id for filename
        safe_gene_id = gene_id.replace('/', '_').replace('\\', '_')
        desc_file = desc_dir / f"{safe_gene_id}.txt"

        with open(desc_file, 'w') as f:
            f.write(description)

    def log_evaluation(self, metrics: Dict[str, float], reward: float):
        """
        Log evaluation results.

        Args:
            metrics: Dictionary of evaluation metrics
            reward: Computed reward value
        """
        if self.iter_dir is None:
            raise RuntimeError("Must call start_iteration first")

        eval_file = self.iter_dir / "evaluation.json"
        with open(eval_file, 'w') as f:
            json.dump({
                'metrics': metrics,
                'reward': reward,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)

        logger.info(f"Iteration {self.current_iteration}: reward = {reward:.4f}")

    def log_feedback(self, feedback: str):
        """
        Log feedback from evaluator.

        Args:
            feedback: Feedback text
        """
        if self.iter_dir is None:
            raise RuntimeError("Must call start_iteration first")

        feedback_file = self.iter_dir / "feedback.txt"
        with open(feedback_file, 'w') as f:
            f.write(feedback)

    def log_reflection(self, reflection: str):
        """
        Log Actor's reflection.

        Args:
            reflection: Reflection text from LLM
        """
        if self.iter_dir is None:
            raise RuntimeError("Must call start_iteration first")

        reflection_file = self.iter_dir / "reflection.txt"
        with open(reflection_file, 'w') as f:
            f.write(reflection)

    def save_iteration_summary(self, additional_info: Optional[Dict[str, Any]] = None):
        """
        Save summary for this iteration.

        Appends to the training summary file.

        Args:
            additional_info: Optional additional information to include
        """
        summary_entry = {
            'iteration': self.current_iteration,
            'task': self.task_name,
            'timestamp': datetime.now().isoformat()
        }

        if additional_info:
            summary_entry.update(additional_info)

        self.summary.append(summary_entry)

        # Append to summary file (JSONL format)
        summary_file = self.log_dir / "training_summary.jsonl"
        with open(summary_file, 'a') as f:
            f.write(json.dumps(summary_entry) + '\n')

    def save_best_strategy(self, strategy: Dict[str, Any], reward: float):
        """
        Save the best strategy found.

        Args:
            strategy: Best strategy configuration
            reward: Reward achieved by the strategy
        """
        best_file = self.log_dir / "best_strategy.json"
        with open(best_file, 'w') as f:
            json.dump({
                'strategy': strategy,
                'reward': reward,
                'task': self.task_name,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)

        logger.info(f"Saved best strategy with reward {reward:.4f}")

    def save_final_report(
        self,
        best_strategy: Dict[str, Any],
        best_reward: float,
        total_iterations: int
    ):
        """
        Save final training report.

        Args:
            best_strategy: Best strategy found
            best_reward: Best reward achieved
            total_iterations: Total number of iterations
        """
        report = {
            'task': self.task_name,
            'total_iterations': total_iterations,
            'best_reward': best_reward,
            'best_strategy': best_strategy,
            'summary': self.summary,
            'completed_at': datetime.now().isoformat()
        }

        report_file = self.log_dir / "final_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"Final report saved to {report_file}")

    def get_log_path(self) -> Path:
        """Get the log directory path."""
        return self.log_dir


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None
):
    """
    Set up logging configuration.

    Args:
        level: Logging level
        log_file: Optional file to write logs to
    """
    handlers = [logging.StreamHandler()]

    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
