#!/usr/bin/env python3
"""
Prepare dataset splits for fact-checking experiments.

This script:
1. Loads the full Politifact dataset
2. Creates train-val-test splits (60-20-20)
3. Generates 5 different test iterations for robustness
4. Saves all configurations for reproducibility

Usage:
    python prepare_dataset.py [--seed SEED] [--n-iterations N]
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data_loaders.politifact import load_politifact
from src.utils.dataset_split import (
    split_dataset,
    create_test_iterations,
    save_split_config,
    print_split_summary,
)
from src.config import settings


def main():
    parser = argparse.ArgumentParser(
        description="Prepare train-val-test splits with multiple test iterations"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--n-iterations",
        type=int,
        default=5,
        help="Number of test iterations (default: 5)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/splits"),
        help="Output directory for splits (default: data/splits)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.6,
        help="Training set ratio (default: 0.6)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Validation set ratio (default: 0.2)",
    )
    parser.add_argument(
        "--test-ratio", type=float, default=0.2, help="Test set ratio (default: 0.2)"
    )

    args = parser.parse_args()

    # Validate ratios
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        print(f"Error: Ratios must sum to 1.0 (got {total_ratio})")
        sys.exit(1)

    print("Loading Politifact dataset...")
    try:
        # Load full dataset
        data = load_politifact(settings.data_dir, split="all")
        print(f"✓ Loaded {len(data)} samples")

        # Filter out None labels
        data = [(id_, claim, label) for id_, claim, label in data if label is not None]
        print(f"✓ Filtered to {len(data)} samples with valid labels")

    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)

    print(
        f"\nCreating splits with ratios: {args.train_ratio:.0%} / {args.val_ratio:.0%} / {args.test_ratio:.0%}"
    )
    print(f"Random seed: {args.seed}")

    # Create train-val-test split
    train_data, val_data, test_data = split_dataset(
        data,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    print(f"\nCreating {args.n_iterations} test iterations...")

    # Create test iterations
    # Note: We're creating iterations by shuffling the test set with different seeds
    # This maintains the same test samples but in different orders
    train_val_combined = train_data + val_data
    test_iterations = create_test_iterations(
        train_val_combined,
        test_data,
        n_iterations=args.n_iterations,
        base_seed=args.seed,
    )

    # Print summary
    print_split_summary(train_data, val_data, test_data, test_iterations)

    # Save configuration
    config_info = {
        "split_ratios": {
            "train": args.train_ratio,
            "val": args.val_ratio,
            "test": args.test_ratio,
        },
        "base_seed": args.seed,
        "n_iterations": args.n_iterations,
        "created_at": datetime.now().isoformat(),
    }

    save_split_config(
        args.output_dir, train_data, val_data, test_data, test_iterations, config_info
    )

    print(f"\n✓ Dataset preparation complete!")
    print(f"\nNext steps:")
    print(f"  1. Review the splits in {args.output_dir}")
    print(f"  2. Run inference with: python run_experiments.py")
    print(f"  3. Evaluate results with: python evaluate_experiments.py")


if __name__ == "__main__":
    main()
