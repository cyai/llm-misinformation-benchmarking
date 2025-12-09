#!/usr/bin/env python3
"""
Deploy Weaviate instance and create schemas.

This script:
1. Checks if Weaviate is running (or starts it via docker-compose)
2. Creates schemas for all test iterations
3. Verifies setup

Usage:
    python -m src.weaviate.deploy
    python -m src.weaviate.deploy --reset  # Delete existing schemas
"""

import argparse
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.weaviate.client import get_weaviate_client, check_weaviate_ready
from src.weaviate.schema import get_all_iteration_schemas
from src.config import settings


def deploy_schemas(client, reset: bool = False):
    """
    Deploy schemas for all iterations.

    Args:
        client: Weaviate client instance
        reset: If True, delete existing schemas first
    """
    schemas = get_all_iteration_schemas()

    for schema_config in schemas:
        class_name = schema_config["name"]

        # Check if collection already exists
        try:
            if client.collections.exists(class_name):
                if reset:
                    print(f"Deleting existing collection: {class_name}")
                    client.collections.delete(class_name)
                else:
                    print(
                        f"Collection already exists: {class_name} (use --reset to recreate)"
                    )
                    continue
        except Exception as e:
            print(f"Error checking collection {class_name}: {e}")

        # Create collection
        try:
            print(f"Creating collection: {class_name}")
            client.collections.create(
                name=schema_config["name"],
                description=schema_config["description"],
                properties=schema_config["properties"],
                vectorizer_config=schema_config["vectorizer_config"],
            )
            print(f"✓ Created: {class_name}")
        except Exception as e:
            print(f"✗ Error creating {class_name}: {e}")
            raise


def main():
    parser = argparse.ArgumentParser(
        description="Deploy Weaviate schemas for fact-checking"
    )
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:8080",
        help="Weaviate URL (default: http://localhost:8080)",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete existing schemas before creating new ones",
    )
    parser.add_argument(
        "--wait",
        type=int,
        default=30,
        help="Seconds to wait for Weaviate to be ready (default: 30)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("WEAVIATE DEPLOYMENT")
    print("=" * 60)

    # Create client
    print(f"\nConnecting to Weaviate at {args.url}...")
    try:
        client = get_weaviate_client(
            url=args.url, openai_api_key=settings.openai_api_key
        )
    except Exception as e:
        print(f"✗ Failed to create client: {e}")
        print("\nMake sure:")
        print(
            "  1. Weaviate is running: docker-compose -f docker-compose.weaviate.yml up -d"
        )
        print("  2. OPENAI_API_KEY is set in .env")
        sys.exit(1)

    # Wait for Weaviate to be ready
    print(f"Waiting for Weaviate to be ready (max {args.wait}s)...")
    for i in range(args.wait):
        if check_weaviate_ready(client):
            print("✓ Weaviate is ready")
            break
        time.sleep(1)
        if (i + 1) % 5 == 0:
            print(f"  Still waiting... ({i + 1}s)")
    else:
        print("✗ Weaviate did not become ready in time")
        print("\nTry:")
        print("  docker-compose -f docker-compose.weaviate.yml up -d")
        print("  docker-compose -f docker-compose.weaviate.yml logs")
        sys.exit(1)

    # Deploy schemas
    print("\nDeploying schemas...")
    try:
        deploy_schemas(client, reset=args.reset)
    except Exception as e:
        print(f"\n✗ Deployment failed: {e}")
        sys.exit(1)

    # Verify
    print("\nVerifying deployment...")
    try:
        all_collections = client.collections.list_all()
        factcheck_collections = [
            name
            for name in all_collections.keys()
            if name.startswith("FactCheckKB_Iter")
        ]

        print(f"✓ Found {len(factcheck_collections)} FactCheckKB collections:")
        for collection_name in sorted(factcheck_collections):
            print(f"  - {collection_name}")
    except Exception as e:
        print(f"✗ Verification failed: {e}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("DEPLOYMENT COMPLETE")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Vectorize data: python -m src.weaviate.vectorize")
    print("  2. Test RAG: python test_single_sample.py --strategies rag --sample-idx 0")
    print("  3. Run experiments: python run_experiments.py --strategies rag")

    # Close client connection
    client.close()


if __name__ == "__main__":
    main()
