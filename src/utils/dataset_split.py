"""
Dataset splitting utility for creating train-validation-test splits with multiple test iterations.
"""

import json
import random
from pathlib import Path
from typing import List, Tuple, Dict, Any
import hashlib


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)


def create_deterministic_seed(base_seed: int, iteration: int) -> int:
    """Create a deterministic seed for a given iteration."""
    # Use hash to create deterministic but different seeds
    combined = f"{base_seed}_{iteration}"
    hash_value = int(hashlib.md5(combined.encode()).hexdigest(), 16)
    return hash_value % (2**32)


def split_dataset(
    data: List[Tuple[str, str, str]],
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[List[Tuple], List[Tuple], List[Tuple]]:
    """
    Split dataset into train, validation, and test sets.

    Args:
        data: List of tuples (id, claim, label)
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_data, val_data, test_data)
    """
    assert (
        abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    ), "Ratios must sum to 1.0"

    set_seed(seed)

    # Shuffle data
    data_copy = data.copy()
    random.shuffle(data_copy)

    # Calculate split sizes
    total_size = len(data_copy)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)

    # Split data
    train_data = data_copy[:train_size]
    val_data = data_copy[train_size : train_size + val_size]
    test_data = data_copy[train_size + val_size :]

    return train_data, val_data, test_data


def create_test_iterations(
    train_val_data: List[Tuple],
    test_data: List[Tuple],
    n_iterations: int = 5,
    base_seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Create multiple test set partitions for robustness testing.

    This keeps train+val data fixed but creates different test set partitions
    by shuffling with different seeds.

    Args:
        train_val_data: Combined train and validation data (fixed)
        test_data: Test data to be partitioned
        n_iterations: Number of different test partitions
        base_seed: Base seed for generating iteration seeds

    Returns:
        List of iteration configurations, each containing:
        - iteration_id
        - seed
        - test_indices (indices into original test_data)
        - test_size
    """
    iterations = []

    for i in range(n_iterations):
        iteration_seed = create_deterministic_seed(base_seed, i)
        set_seed(iteration_seed)

        # Create shuffled indices for this iteration
        test_indices = list(range(len(test_data)))
        random.shuffle(test_indices)

        iterations.append(
            {
                "iteration_id": i,
                "seed": iteration_seed,
                "test_indices": test_indices,
                "test_size": len(test_data),
                "description": f"Test iteration {i} with seed {iteration_seed}",
            }
        )

    return iterations


def save_split_config(
    output_dir: Path,
    train_data: List[Tuple],
    val_data: List[Tuple],
    test_data: List[Tuple],
    test_iterations: List[Dict],
    config_info: Dict[str, Any],
):
    """
    Save dataset split configuration to files.

    Args:
        output_dir: Directory to save configuration
        train_data: Training data
        val_data: Validation data
        test_data: Test data
        test_iterations: Test iteration configurations
        config_info: Additional configuration metadata
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save main configuration
    main_config = {
        "split_ratios": config_info.get(
            "split_ratios", {"train": 0.6, "val": 0.2, "test": 0.2}
        ),
        "base_seed": config_info.get("base_seed", 42),
        "n_iterations": len(test_iterations),
        "dataset_sizes": {
            "train": len(train_data),
            "val": len(val_data),
            "test": len(test_data),
            "total": len(train_data) + len(val_data) + len(test_data),
        },
        "label_distribution": {
            "train": get_label_distribution(train_data),
            "val": get_label_distribution(val_data),
            "test": get_label_distribution(test_data),
        },
        "created_at": config_info.get("created_at", None),
    }

    with open(output_dir / "split_config.json", "w") as f:
        json.dump(main_config, f, indent=2)

    # Save test iterations config
    with open(output_dir / "test_iterations.json", "w") as f:
        json.dump(test_iterations, f, indent=2)

    # Save actual data splits
    save_data_to_jsonl(train_data, output_dir / "train.jsonl")
    save_data_to_jsonl(val_data, output_dir / "val.jsonl")
    save_data_to_jsonl(test_data, output_dir / "test.jsonl")

    print(f"✓ Split configuration saved to {output_dir}")
    print(f"  - Train: {len(train_data)} samples")
    print(f"  - Val: {len(val_data)} samples")
    print(f"  - Test: {len(test_data)} samples")
    print(f"  - Test iterations: {len(test_iterations)}")


def load_split_config(config_dir: Path) -> Dict[str, Any]:
    """Load split configuration from directory."""
    config_dir = Path(config_dir)

    with open(config_dir / "split_config.json", "r") as f:
        main_config = json.load(f)

    with open(config_dir / "test_iterations.json", "r") as f:
        test_iterations = json.load(f)

    train_data = load_data_from_jsonl(config_dir / "train.jsonl")
    val_data = load_data_from_jsonl(config_dir / "val.jsonl")
    test_data = load_data_from_jsonl(config_dir / "test.jsonl")

    return {
        "config": main_config,
        "test_iterations": test_iterations,
        "train_data": train_data,
        "val_data": val_data,
        "test_data": test_data,
    }


def get_test_data_for_iteration(
    test_data: List[Tuple], iteration_config: Dict
) -> List[Tuple]:
    """
    Get test data for a specific iteration using its configuration.

    Args:
        test_data: Original test data
        iteration_config: Configuration for this iteration (from test_iterations.json)

    Returns:
        Test data shuffled according to iteration config
    """
    indices = iteration_config["test_indices"]
    return [test_data[i] for i in indices]


def save_data_to_jsonl(data: List[Tuple], filepath: Path):
    """Save data tuples to JSONL file."""
    with open(filepath, "w") as f:
        for item_id, claim, label in data:
            record = {"id": item_id, "claim": claim, "label": label}
            f.write(json.dumps(record) + "\n")


def load_data_from_jsonl(filepath: Path) -> List[Tuple]:
    """Load data from JSONL file."""
    data = []
    with open(filepath, "r") as f:
        for line in f:
            record = json.loads(line.strip())
            data.append((record["id"], record["claim"], record["label"]))
    return data


def get_label_distribution(data: List[Tuple]) -> Dict[str, int]:
    """Get distribution of labels in dataset."""
    distribution = {}
    for _, _, label in data:
        distribution[label] = distribution.get(label, 0) + 1
    return distribution


def print_split_summary(train_data, val_data, test_data, test_iterations):
    """Print summary of dataset split."""
    print("\n" + "=" * 60)
    print("DATASET SPLIT SUMMARY")
    print("=" * 60)

    total = len(train_data) + len(val_data) + len(test_data)

    print(f"\nTotal samples: {total}")
    print(f"\nSplit sizes:")
    print(f"  Train: {len(train_data)} ({len(train_data)/total*100:.1f}%)")
    print(f"  Val:   {len(val_data)} ({len(val_data)/total*100:.1f}%)")
    print(f"  Test:  {len(test_data)} ({len(test_data)/total*100:.1f}%)")

    print(f"\nLabel distribution:")
    for split_name, split_data in [
        ("Train", train_data),
        ("Val", val_data),
        ("Test", test_data),
    ]:
        dist = get_label_distribution(split_data)
        print(f"  {split_name}:")
        for label, count in sorted(dist.items()):
            print(f"    {label}: {count} ({count/len(split_data)*100:.1f}%)")

    print(f"\nTest iterations: {len(test_iterations)}")
    for iteration in test_iterations:
        print(f"  Iteration {iteration['iteration_id']}: seed={iteration['seed']}")

    print("\n" + "=" * 60)
