#!/usr/bin/env python3
"""
Vectorize training data and store in Weaviate for RAG.

This script:
1. Loads training data for each iteration
2. Vectorizes claims using OpenAI embeddings
3. Stores in iteration-specific Weaviate classes

Usage:
    python -m src.weaviate.vectorize
    python -m src.weaviate.vectorize --iteration 0  # Only one iteration
    python -m src.weaviate.vectorize --max-samples 100  # Limit for testing
"""

import argparse
import sys
from pathlib import Path
from tqdm import tqdm
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.weaviate.client import get_weaviate_client, check_weaviate_ready
from src.utils.dataset_split import load_split_config
from src.config import settings


def vectorize_iteration(
    client,
    iteration_id: int,
    train_data: list,
    max_samples: int = 0,
    batch_size: int = 100,
):
    """
    Vectorize and store training data for a specific iteration.

    Args:
        client: Weaviate client instance
        iteration_id: Test iteration number (0-4)
        train_data: Training data as list of (claim_id, claim, label) tuples
        max_samples: Maximum number of samples to vectorize (0 = all)
        batch_size: Batch size for Weaviate ingestion
    """
    class_name = f"FactCheckKB_Iter{iteration_id}"

    # Limit samples if requested
    if max_samples > 0:
        train_data = train_data[:max_samples]

    print(f"\nVectorizing {len(train_data)} samples for {class_name}...")

    # Get collection
    collection = client.collections.get(class_name)

    # Check current count
    try:
        agg = collection.aggregate.over_all(total_count=True)
        current_count = agg.total_count

        if current_count > 0:
            print(f"⚠ Warning: {class_name} already has {current_count} objects")
            response = input("  Delete existing data? [y/N]: ")
            if response.lower() == "y":
                print(f"  Deleting all objects from {class_name}...")
                collection.data.delete_many(
                    where={
                        "path": ["source"],
                        "operator": "Equal",
                        "valueText": "politifact",
                    }
                )
                print("  ✓ Deleted")
            else:
                print("  Skipping this iteration")
                return
    except Exception as e:
        print(f"  Could not check existing count: {e}")

    # Batch import
    successful = 0
    failed = 0

    with collection.batch.dynamic() as batch:
        for claim_id, claim, label in tqdm(train_data, desc=f"Iter {iteration_id}"):
            try:
                properties = {
                    "claim_id": str(claim_id),
                    "claim": claim,
                    "label": label,
                    "verdict": label,  # Store label as verdict for now
                    "source": "politifact",
                }

                batch.add_object(properties=properties)
                successful += 1

            except Exception as e:
                print(f"\n✗ Error adding claim {claim_id}: {e}")
                failed += 1

    print(f"✓ Completed: {successful} successful, {failed} failed")

    # Verify
    try:
        agg = collection.aggregate.over_all(total_count=True)
        final_count = agg.total_count
        print(f"✓ Final count in {class_name}: {final_count}")
    except Exception as e:
        print(f"⚠ Could not verify count: {e}")


def main():
    parser = argparse.ArgumentParser(description="Vectorize training data for RAG")
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:8080",
        help="Weaviate URL (default: http://localhost:8080)",
    )
    parser.add_argument(
        "--iteration",
        type=int,
        default=None,
        help="Specific iteration to vectorize (0-4). If not set, all iterations are vectorized.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Maximum samples per iteration (0 = all, default: 0)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for ingestion (default: 100)",
    )
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=Path("data/splits"),
        help="Directory containing dataset splits",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("WEAVIATE VECTORIZATION")
    print("=" * 60)

    # Load dataset
    print("\nLoading dataset splits...")
    try:
        split_data = load_split_config(args.splits_dir)
        train_data = split_data["train_data"]
        print(f"✓ Loaded {len(train_data)} training samples")
    except Exception as e:
        print(f"✗ Failed to load dataset: {e}")
        sys.exit(1)

    # Connect to Weaviate
    print(f"\nConnecting to Weaviate at {args.url}...")
    try:
        client = get_weaviate_client(
            url=args.url, openai_api_key=settings.openai_api_key
        )
    except Exception as e:
        print(f"✗ Failed to create client: {e}")
        sys.exit(1)

    if not check_weaviate_ready(client):
        print("✗ Weaviate is not ready")
        print("\nStart Weaviate with:")
        print("  docker-compose -f docker-compose.weaviate.yml up -d")
        sys.exit(1)

    print("✓ Connected to Weaviate")

    # Determine which iterations to vectorize
    if args.iteration is not None:
        if not (0 <= args.iteration <= 4):
            print(f"✗ Invalid iteration: {args.iteration}. Must be 0-4.")
            sys.exit(1)
        iterations = [args.iteration]
    else:
        iterations = range(5)

    print(f"\nVectorizing iterations: {list(iterations)}")
    if args.max_samples > 0:
        print(f"Limited to {args.max_samples} samples per iteration")

    # Vectorize each iteration
    for iteration_id in iterations:
        try:
            vectorize_iteration(
                client=client,
                iteration_id=iteration_id,
                train_data=train_data,
                max_samples=args.max_samples,
                batch_size=args.batch_size,
            )
        except Exception as e:
            print(f"✗ Error vectorizing iteration {iteration_id}: {e}")
            continue

    print("\n" + "=" * 60)
    print("VECTORIZATION COMPLETE")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Test RAG: python test_single_sample.py --strategies rag --sample-idx 0")
    print(
        "  2. Run experiments: python run_experiments.py --strategies rag --iterations 0"
    )

    # Close client
    client.close()


if __name__ == "__main__":
    main()
