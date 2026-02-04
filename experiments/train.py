#!/usr/bin/env python3
"""
SIGR Training Script

Command-line interface for running SIGR training on downstream tasks.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import run_training, run_all_tasks, setup_logging
from src.sigr_framework import list_available_tasks, get_task_info, AVAILABLE_TASKS


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train SIGR framework on downstream tasks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available tasks
  python train.py --list-tasks

  # Run PPI task with real LLM (default)
  python train.py --task ppi --iterations 3 --genes-per-iter 50 --api-key YOUR_KEY

  # Run gene type classification
  python train.py --task genetype --iterations 5 --api-key YOUR_KEY

  # Run with mock LLM for testing
  python train.py --task ppi --iterations 2 --mock

  # Run all tasks
  python train.py --task all --iterations 3 --api-key YOUR_KEY

  # Run specific tasks
  python train.py --tasks ppi genetype ggi --iterations 3 --api-key YOUR_KEY
"""
    )

    # Task selection
    parser.add_argument(
        '--task',
        type=str,
        choices=AVAILABLE_TASKS + ['all'],
        default='ppi',
        help='Task to train on (default: ppi)'
    )

    parser.add_argument(
        '--tasks',
        type=str,
        nargs='+',
        choices=AVAILABLE_TASKS,
        help='Run multiple specific tasks'
    )

    parser.add_argument(
        '--list-tasks',
        action='store_true',
        help='List all available tasks and exit'
    )

    # Knowledge graph
    parser.add_argument(
        '--kg-path',
        type=str,
        default='data/kg/sigr_kg_v2.pkl',
        help='Path to knowledge graph pickle file (v2 includes Reactome)'
    )

    # Training parameters
    parser.add_argument(
        '--iterations',
        type=int,
        default=10,
        help='Number of training iterations (default: 10)'
    )

    parser.add_argument(
        '--genes-per-iter',
        type=int,
        default=0,
        help='Number of genes per iteration (default: 0 = use all task genes)'
    )

    # Output directories
    parser.add_argument(
        '--log-dir',
        type=str,
        default='logs',
        help='Directory for logs (default: logs)'
    )

    parser.add_argument(
        '--results-dir',
        type=str,
        default='results',
        help='Directory for results (default: results)'
    )

    # LLM API configuration
    parser.add_argument(
        '--api-base',
        type=str,
        default='https://yunwu.ai/v1',
        help='LLM API base URL (default: https://yunwu.ai/v1)'
    )

    parser.add_argument(
        '--api-key',
        type=str,
        default=None,
        help='LLM API key (or set OPENAI_API_KEY env var)'
    )

    parser.add_argument(
        '--model',
        type=str,
        default='gpt-4o-mini',
        help='LLM model name for single-model mode (default: gpt-4o-mini)'
    )

    # Dual model configuration (default: enabled for better Actor reasoning)
    parser.add_argument(
        '--use-dual-model',
        action='store_true',
        default=True,
        help='Use dual-model architecture (fast model for generation, strong model for Actor reflection). Enabled by default.'
    )

    parser.add_argument(
        '--single-model',
        action='store_true',
        help='Use single model for all tasks (disables dual-model mode)'
    )

    parser.add_argument(
        '--fast-model',
        type=str,
        default='gpt-4o-mini',
        help='Fast model for generation tasks (default: gpt-4o-mini)'
    )

    parser.add_argument(
        '--strong-model',
        type=str,
        default='gemini-3-pro-preview',
        help='Strong model for Actor reflection/reasoning (default: gemini-2.5-pro-preview-05-06)'
    )

    # Mock mode
    parser.add_argument(
        '--mock',
        action='store_true',
        help='Use mock LLM client for testing (no API calls)'
    )

    # Concurrency
    parser.add_argument(
        '--max-workers',
        type=int,
        default=8,
        help='Maximum concurrent LLM requests (default: 8)'
    )

    # Evaluation options
    parser.add_argument(
        '--cross-validation',
        action='store_true',
        help='Use 5-fold cross-validation for evaluation'
    )

    parser.add_argument(
        '--n-folds',
        type=int,
        default=5,
        help='Number of folds for cross-validation (default: 5)'
    )

    # Checkpoint/Resume
    parser.add_argument(
        '--checkpoint-interval',
        type=int,
        default=5,
        help='Save checkpoint every N iterations (0 to disable, default: 5)'
    )

    parser.add_argument(
        '--resume-from',
        type=str,
        default=None,
        help='Path to checkpoint file to resume training from'
    )

    # Debugging
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )

    # Self-verification options (LLM self-critique, Constitutional AI patterns)
    parser.add_argument(
        '--no-self-critique',
        action='store_true',
        help='Disable Actor self-critique (two-stage reasoning validation)'
    )

    parser.add_argument(
        '--no-consistency-check',
        action='store_true',
        help='Disable strategy consistency check (prevents undoing successful changes)'
    )

    parser.add_argument(
        '--no-cot',
        action='store_true',
        help='Disable Chain-of-Thought enhanced reasoning'
    )

    return parser.parse_args()


def print_task_list():
    """Print list of available tasks with descriptions."""
    print("\n" + "="*70)
    print("Available Downstream Tasks")
    print("="*70 + "\n")

    for task_name in AVAILABLE_TASKS:
        info = get_task_info(task_name)
        print(f"  {task_name}")
        print(f"    Name: {info.get('name', 'N/A')}")
        print(f"    Description: {info.get('description', 'N/A')}")
        print(f"    Primary Metric: {info.get('metric', 'N/A')}")
        print(f"    Data: {info.get('data', 'N/A')}")
        print()

    print("="*70)
    print("\nUsage examples:")
    print("  python train.py --task ppi --iterations 5 --api-key YOUR_KEY")
    print("  python train.py --task genetype --mock")
    print("  python train.py --task all --iterations 3")
    print()


def main():
    """Main entry point."""
    args = parse_args()

    # List tasks and exit
    if args.list_tasks:
        print_task_list()
        return

    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(level=log_level)

    logger = logging.getLogger(__name__)
    logger.info(f"Starting SIGR training")
    logger.info(f"Task: {args.task}")
    logger.info(f"Iterations: {args.iterations}")
    logger.info(f"Genes per iteration: {args.genes_per_iter}")

    if not args.mock:
        logger.info(f"LLM API: {args.api_base}")
        # Determine model mode
        use_dual = args.use_dual_model and not args.single_model
        if use_dual:
            logger.info(f"Dual-model mode: fast={args.fast_model}, strong={args.strong_model}")
        else:
            logger.info(f"Single-model mode: {args.model}")

    # Determine which tasks to run
    if args.tasks:
        # Multiple specific tasks
        tasks_to_run = args.tasks
    elif args.task == 'all':
        tasks_to_run = AVAILABLE_TASKS
    else:
        tasks_to_run = [args.task]

    # Run training
    if len(tasks_to_run) > 1:
        # Run multiple tasks
        results = run_all_tasks(
            kg_path=args.kg_path,
            n_iterations=args.iterations,
            genes_per_iter=args.genes_per_iter,
            tasks=tasks_to_run,
            api_base=args.api_base,
            api_key=args.api_key,
            model=args.model,
            use_mock=args.mock
        )

        print("\n" + "="*70)
        print("Training Complete - Results Summary")
        print("="*70)
        for task, strategy in results.items():
            print(f"\n{task}:")
            if strategy:
                print(f"  Edge types: {strategy.get('edge_types', [])}")
                print(f"  Max neighbors: {strategy.get('max_neighbors', 0)}")
                print(f"  Max hops: {strategy.get('max_hops', 2)}")
                print(f"  Sampling: {strategy.get('sampling', 'top_k')}")
            else:
                print("  Failed to train")

    else:
        # Run single task
        task_name = tasks_to_run[0]
        # Determine model mode (dual is default unless --single-model is specified)
        use_dual = args.use_dual_model and not args.single_model
        best_strategy = run_training(
            task_name=task_name,
            kg_path=args.kg_path,
            n_iterations=args.iterations,
            genes_per_iter=args.genes_per_iter,
            log_dir=args.log_dir,
            results_dir=args.results_dir,
            use_mock=args.mock,
            api_base=args.api_base,
            api_key=args.api_key,
            model=args.model,
            use_dual_model=use_dual,
            fast_model=args.fast_model,
            strong_model=args.strong_model,
            use_cross_validation=args.cross_validation,
            n_folds=args.n_folds,
            max_workers=args.max_workers,
            checkpoint_interval=args.checkpoint_interval,
            resume_from=args.resume_from,
            # Self-verification options
            enable_self_critique=not args.no_self_critique,
            enable_consistency_check=not args.no_consistency_check,
            enable_cot_reasoning=not args.no_cot
        )

        print("\n" + "="*70)
        print(f"Training Complete - Best Strategy for {task_name}")
        print("="*70)
        print(f"Edge types: {best_strategy.get('edge_types', [])}")
        print(f"Max hops: {best_strategy.get('max_hops', 2)}")
        print(f"Sampling: {best_strategy.get('sampling', 'top_k')}")
        print(f"Max neighbors: {best_strategy.get('max_neighbors', 50)}")
        print(f"\nResults saved to: {args.results_dir}/strategies/{task_name}_best.json")
        print(f"Logs saved to: {args.log_dir}/{task_name}_task/")


if __name__ == '__main__':
    main()
