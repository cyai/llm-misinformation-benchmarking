#!/usr/bin/env python3
"""
Test all fact-checking strategies on a single sample.

This script runs all available strategies (zero_shot, one_shot, few_shot, cot, search)
on a single claim and saves detailed outputs to results/tests/

Usage:
    # Use a custom claim
    python test_single_sample.py --claim "The Earth is flat."

    # Use a claim from the dataset by ID
    python test_single_sample.py --claim-id 12345

    # Use first sample from test data
    python test_single_sample.py --sample-idx 0

    # Test specific strategies only
    python test_single_sample.py --claim "Water boils at 100°C" --strategies zero_shot,cot
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.chains.fact_check import build_fact_check_chain
from src.chains.fact_check_with_search import build_fact_check_search_chain
from src.chains.fact_check_rag import build_fact_check_rag_chain
from src.tools.search import SearchTool
from src.weaviate.client import get_weaviate_client
from src.weaviate.retriever import FactCheckRetriever
from src.models.llm import make_chat_model
from src.config import settings
from src.utils.dataset_split import load_split_config


# Prompt strategies to test
PROMPT_STRATEGIES = {
    "zero_shot": "src/prompts/fact_check.txt",
    "one_shot": "src/prompts/fact_check_oneshot.txt",
    "few_shot": "src/prompts/fact_check_fewshot.txt",
    "cot": "src/prompts/fact_check_cot.txt",
    "search": "src/prompts/fact_check_search.txt",
    "rag": "src/prompts/fact_check_rag.txt",
}

SEARCH_STRATEGIES = {"search"}
RAG_STRATEGIES = {"rag"}


def run_single_sample(
    claim,
    gold_label,
    claim_id,
    strategies_to_run,
    llm,
    search_tool,
    weaviate_client,
    iteration_id=0,
):
    """Run all strategies on a single claim and return results."""

    results = {
        "claim": claim,
        "claim_id": claim_id,
        "gold_label": gold_label,
        "timestamp": datetime.now().isoformat(),
        "strategies": {},
    }

    for strategy in strategies_to_run:
        print(f"\n{'='*60}")
        print(f"Running: {strategy.upper()}")
        print(f"{'='*60}")

        prompt_file = Path(PROMPT_STRATEGIES[strategy])

        if not prompt_file.exists():
            print(f"⚠ Prompt file not found: {prompt_file}")
            results["strategies"][strategy] = {
                "error": f"Prompt file not found: {prompt_file}"
            }
            continue

        # Skip search if tool unavailable
        if strategy in SEARCH_STRATEGIES and not search_tool:
            print(f"⚠ Search tool not available, skipping {strategy}")
            results["strategies"][strategy] = {"error": "Search tool not available"}
            continue

        # Skip RAG if Weaviate unavailable
        if strategy in RAG_STRATEGIES and not weaviate_client:
            print(f"⚠ Weaviate not available, skipping {strategy}")
            results["strategies"][strategy] = {"error": "Weaviate not available"}
            continue

        try:
            # Build chain
            if strategy in RAG_STRATEGIES:
                retriever = FactCheckRetriever(
                    client=weaviate_client, iteration_id=iteration_id
                )
                chain = build_fact_check_rag_chain(
                    llm=llm,
                    prompt_path=prompt_file,
                    retriever=retriever,
                    top_k=5,
                    certainty=0.7,
                )
            elif strategy in SEARCH_STRATEGIES:
                chain = build_fact_check_search_chain(
                    llm=llm, prompt_path=prompt_file, search_tool=search_tool
                )
            else:
                chain = build_fact_check_chain(llm=llm, prompt_path=prompt_file)

            # Run inference
            print(f"Claim: {claim[:100]}...")
            result = chain.invoke({"claim": claim})

            # Extract result
            if isinstance(result, dict) and "result" in result:
                result_data = result["result"]
            else:
                result_data = result

            # Store full result
            results["strategies"][strategy] = {
                "verdict": result_data.get("verdict", "UNKNOWN"),
                "confidence": result_data.get("confidence", 0.5),
                "rationale": result_data.get("rationale", ""),
                "cited_knowledge": result_data.get("cited_knowledge", ""),
                "safety_notes": result_data.get("safety_notes", ""),
            }

            # Add strategy-specific fields
            if "reasoning_steps" in result_data:
                results["strategies"][strategy]["reasoning_steps"] = result_data[
                    "reasoning_steps"
                ]
            if "search_relevance" in result_data:
                results["strategies"][strategy]["search_relevance"] = result_data[
                    "search_relevance"
                ]

            # Print summary
            print(f"✓ Verdict: {results['strategies'][strategy]['verdict']}")
            print(f"  Confidence: {results['strategies'][strategy]['confidence']}")
            print(
                f"  Rationale: {results['strategies'][strategy]['rationale'][:150]}..."
            )

        except Exception as e:
            print(f"✗ Error: {str(e)}")
            results["strategies"][strategy] = {"error": str(e)}

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Test all strategies on a single claim"
    )
    parser.add_argument("--claim", type=str, default=None, help="Custom claim to test")
    parser.add_argument(
        "--claim-id", type=str, default=None, help="Claim ID from dataset"
    )
    parser.add_argument(
        "--sample-idx",
        type=int,
        default=None,
        help="Sample index from test data (0-based)",
    )
    parser.add_argument(
        "--strategies",
        type=str,
        default="all",
        help='Which strategies to test: "all" or comma-separated list (default: all)',
    )
    parser.add_argument(
        "--model", type=str, default=None, help="Model name (default: from .env)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/tests"),
        help="Directory to save test results (default: results/tests)",
    )

    args = parser.parse_args()

    # Determine which strategies to run
    if args.strategies == "all":
        strategies_to_run = list(PROMPT_STRATEGIES.keys())
    else:
        strategies_to_run = [s.strip() for s in args.strategies.split(",")]

    # Validate strategies
    for strategy in strategies_to_run:
        if strategy not in PROMPT_STRATEGIES:
            print(
                f"Error: Unknown strategy '{strategy}'. Valid options: {list(PROMPT_STRATEGIES.keys())}"
            )
            sys.exit(1)

    # Get claim and label
    claim = None
    gold_label = "UNKNOWN"
    claim_id = "custom"

    if args.claim:
        # Custom claim provided
        claim = args.claim
        print(f"Using custom claim: {claim}")

    elif args.claim_id or args.sample_idx is not None:
        # Load from dataset
        print("Loading dataset...")
        try:
            split_data = load_split_config(Path("data/splits"))
            test_data = split_data["test_data"]

            if args.claim_id:
                # Find by claim_id
                for cid, c, label in test_data:
                    if cid == args.claim_id:
                        claim_id = cid
                        claim = c
                        gold_label = label
                        break

                if not claim:
                    print(f"Error: Claim ID '{args.claim_id}' not found in test data")
                    sys.exit(1)

            elif args.sample_idx is not None:
                # Get by index
                if 0 <= args.sample_idx < len(test_data):
                    claim_id, claim, gold_label = test_data[args.sample_idx]
                else:
                    print(
                        f"Error: Sample index {args.sample_idx} out of range (0-{len(test_data)-1})"
                    )
                    sys.exit(1)

            print(f"Loaded claim from dataset:")
            print(f"  ID: {claim_id}")
            print(f"  Gold label: {gold_label}")
            print(f"  Claim: {claim[:100]}...")

        except Exception as e:
            print(f"Error loading dataset: {e}")
            sys.exit(1)

    else:
        # No claim provided, use default
        claim = "The Earth is flat."
        print(f"No claim provided, using default: {claim}")

    # Initialize LLM
    model_name = args.model or settings.openai_model
    print(f"\nInitializing LLM: {model_name}")
    llm = make_chat_model(
        provider="openai", model_name=model_name, api_key=settings.openai_api_key
    )

    # Initialize search tool if needed
    search_tool = None
    if any(s in SEARCH_STRATEGIES for s in strategies_to_run):
        try:
            search_tool = SearchTool(max_results=5)
            print("✓ Search tool initialized")
        except Exception as e:
            print(f"⚠ Warning: Could not initialize search tool: {e}")
            print("  Search strategy will be skipped")

    # Initialize Weaviate client if needed
    weaviate_client = None
    if any(s in RAG_STRATEGIES for s in strategies_to_run):
        try:
            weaviate_client = get_weaviate_client(
                url="http://localhost:8080", openai_api_key=settings.openai_api_key
            )
            if weaviate_client.is_ready():
                print("✓ Weaviate client initialized")
            else:
                print("⚠ Warning: Weaviate not ready")
                weaviate_client = None
        except Exception as e:
            print(f"⚠ Warning: Could not initialize Weaviate: {e}")
            print("  RAG strategy will be skipped")

    print(f"\nTesting strategies: {', '.join(strategies_to_run)}")
    print(f"Output directory: {args.output_dir}")

    # Run test
    results = run_single_sample(
        claim=claim,
        gold_label=gold_label,
        claim_id=claim_id,
        strategies_to_run=strategies_to_run,
        llm=llm,
        search_tool=search_tool,
        weaviate_client=weaviate_client,
        iteration_id=0,  # Default to iteration 0 for single sample tests
    )

    # Save results
    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = args.output_dir / f"test_{timestamp}.json"

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print("TEST COMPLETE")
    print(f"{'='*60}")
    print(f"\nResults saved to: {output_file}")

    # Print summary
    print("\nSummary:")
    print(f"  Claim: {claim[:80]}...")
    print(f"  Gold Label: {gold_label}")
    print(f"\n  Strategy Results:")

    for strategy, data in results["strategies"].items():
        if "error" in data:
            print(f"    {strategy:12} - ERROR: {data['error']}")
        else:
            verdict = data.get("verdict", "N/A")
            confidence = data.get("confidence", 0)
            match = "✓" if verdict == gold_label else "✗"
            print(f"    {strategy:12} - {verdict:5} (conf: {confidence:.2f}) {match}")

    print(f"\nDetailed results: {output_file}")


if __name__ == "__main__":
    main()
