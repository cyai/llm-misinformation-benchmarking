#!/usr/bin/env python3
"""
Run fact-checking experiments across all prompt strategies and test iterations.

This script:
1. Loads the prepared dataset splits
2. Runs inference for zero-shot, one-shot, and few-shot prompts
3. Evaluates only on test data across all 5 iterations
4. Saves results for reproducibility

Usage:
    python run_experiments.py [--model MODEL] [--provider PROVIDER] [--iterations ITERATIONS]
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.dataset_split import load_split_config, get_test_data_for_iteration
from src.chains.fact_check import build_fact_check_chain
from src.chains.fact_check_with_search import build_fact_check_search_chain
from src.chains.fact_check_rag import build_fact_check_rag_chain
from src.tools.search import SearchTool
from src.weaviate.client import get_weaviate_client
from src.weaviate.retriever import FactCheckRetriever
from src.models.llm import make_chat_model
from src.config import settings


# Prompt strategies to test
PROMPT_STRATEGIES = {
    "zero_shot": "src/prompts/fact_check.txt",
    "one_shot": "src/prompts/fact_check_oneshot.txt",
    "few_shot": "src/prompts/fact_check_fewshot.txt",
    "cot": "src/prompts/fact_check_cot.txt",
    "search": "src/prompts/fact_check_search.txt",
    "rag": "src/prompts/fact_check_rag.txt",
}

# Strategies that require search tool
SEARCH_STRATEGIES = {"search"}

# Strategies that require RAG (Weaviate)
RAG_STRATEGIES = {"rag"}


def run_inference(chain, test_data, strategy, iteration, output_file, max_samples=None):
    """Run inference with parallel API calls (10 concurrent)."""
    print(f"\nRunning {strategy} inference on iteration {iteration}")
    print(f"Output: {output_file}")

    # Limit samples if specified
    samples = test_data[:max_samples] if max_samples else test_data
    print(f"Processing {len(samples)} samples")

    results = []
    lock = threading.Lock()

    def process_sample(idx_sample):
        idx, sample = idx_sample
        claim_id, claim, label = sample
        try:
            result = chain.invoke({"claim": claim})

            # Extract result from the chain output
            if isinstance(result, dict) and "result" in result:
                result_data = result["result"]
            else:
                result_data = result

            output = {
                "iteration": iteration,
                "strategy": strategy,
                "sample_idx": idx,
                "claim_id": claim_id,
                "claim": claim,
                "gold_label": label,
                "verdict": result_data.get("verdict", "UNKNOWN"),
                "confidence": result_data.get("confidence", 0.5),
                "rationale": result_data.get("rationale", ""),
                "timestamp": datetime.now().isoformat(),
            }

            return output
        except Exception as e:
            print(f"\nError processing sample {idx}: {str(e)}")
            return {
                "iteration": iteration,
                "strategy": strategy,
                "sample_idx": idx,
                "claim_id": claim_id,
                "claim": claim,
                "gold_label": label,
                "verdict": "ERROR",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    # Process samples in parallel with 10 workers
    with ThreadPoolExecutor(max_workers=10) as executor:
        # Submit all tasks
        futures = {
            executor.submit(process_sample, (idx, sample)): idx
            for idx, sample in enumerate(samples)
        }

        # Process completed tasks with progress bar
        with tqdm(total=len(samples), desc=f"{strategy} (iter {iteration})") as pbar:
            for future in as_completed(futures):
                result = future.result()
                with lock:
                    results.append(result)
                pbar.update(1)

    # Sort results by sample_idx to maintain order
    results.sort(key=lambda x: x["sample_idx"])

    # Write all results to file
    with open(output_file, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

    print(f"Completed {len(results)} samples")


def get_few_shot_examples(train_data, n_examples: int = 3):
    """
    Select few-shot examples from training data.
    Tries to get balanced examples across labels.
    """
    from collections import defaultdict

    # Group by label
    by_label = defaultdict(list)
    for item_id, claim, label in train_data:
        by_label[label].append((item_id, claim, label))

    # Sample from each label
    examples = []
    labels = list(by_label.keys())
    examples_per_label = n_examples // len(labels)

    for label in labels:
        label_examples = by_label[label][:examples_per_label]
        examples.extend(label_examples)

    # If we need more, add from first label
    while len(examples) < n_examples and by_label[labels[0]]:
        if len(by_label[labels[0]]) > len(examples):
            examples.append(by_label[labels[0]][len(examples)])
        else:
            break

    return examples[:n_examples]


def main():
    parser = argparse.ArgumentParser(
        description="Run fact-checking experiments across all prompt strategies and test iterations"
    )
    parser.add_argument(
        "--provider", type=str, default="openai", help="LLM provider (default: openai)"
    )
    parser.add_argument(
        "--model", type=str, default=None, help="Model name (default: from .env)"
    )
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=Path("data/splits"),
        help="Directory containing dataset splits (default: data/splits)",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/experiments"),
        help="Directory to save results (default: results/experiments)",
    )
    parser.add_argument(
        "--iterations",
        type=str,
        default="all",
        help='Which iterations to run: "all" or comma-separated list like "0,1,2" (default: all)',
    )
    parser.add_argument(
        "--strategies",
        type=str,
        default="all",
        help='Which strategies to run: "all" or comma-separated list like "zero_shot,one_shot" (default: all)',
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Limit number of samples per iteration (0 = no limit, default: 0)",
    )

    args = parser.parse_args()

    # Load dataset splits
    print("Loading dataset splits...")
    try:
        split_data = load_split_config(args.splits_dir)
        train_data = split_data["train_data"]
        val_data = split_data["val_data"]
        test_data = split_data["test_data"]
        test_iterations = split_data["test_iterations"]
        config = split_data["config"]

        print(f"✓ Loaded splits:")
        print(f"  Train: {len(train_data)} samples")
        print(f"  Val: {len(val_data)} samples")
        print(f"  Test: {len(test_data)} samples")
        print(f"  Test iterations: {len(test_iterations)}")

    except Exception as e:
        print(f"Error loading splits: {e}")
        print(f"Please run 'python prepare_dataset.py' first to create splits.")
        sys.exit(1)

    # Determine which iterations to run
    if args.iterations == "all":
        iterations_to_run = list(range(len(test_iterations)))
    else:
        iterations_to_run = [int(i.strip()) for i in args.iterations.split(",")]

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

    print(f"\nExperiment configuration:")
    print(f"  Provider: {args.provider}")
    print(f"  Model: {args.model or settings.openai_model}")
    print(f"  Strategies: {strategies_to_run}")
    print(f"  Test iterations: {iterations_to_run}")
    print(f"  Max samples per iteration: {args.max_samples or 'unlimited'}")

    # Initialize LLM
    model_name = args.model or settings.openai_model
    llm = make_chat_model(
        provider=args.provider, model_name=model_name, api_key=settings.openai_api_key
    )

    # Initialize search tool if needed
    search_tool = None
    if any(s in SEARCH_STRATEGIES for s in strategies_to_run):
        try:
            search_tool = SearchTool(max_results=5)
            print("✓ Search tool initialized")
        except Exception as e:
            print(f"⚠ Warning: Could not initialize search tool: {e}")
            print("  Search-based strategies will be skipped.")
            print(
                "  To enable: Set SERPAPI_API_KEY or (GOOGLE_API_KEY + GOOGLE_CSE_ID) in .env"
            )

    # Initialize Weaviate client if needed for RAG
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
            print("  RAG strategies will be skipped.")
            print(
                "  To enable: Run 'docker-compose -f docker-compose.weaviate.yml up -d'"
            )
            print(
                "  Then: python -m src.weaviate.deploy && python -m src.weaviate.vectorize"
            )

    # Create experiment metadata
    experiment_meta = {
        "timestamp": datetime.now().isoformat(),
        "provider": args.provider,
        "model": model_name,
        "strategies": strategies_to_run,
        "iterations": iterations_to_run,
        "max_samples": args.max_samples,
        "split_config": config,
    }

    # Save experiment metadata
    meta_file = args.results_dir / "experiment_metadata.json"
    args.results_dir.mkdir(parents=True, exist_ok=True)
    with open(meta_file, "w") as f:
        json.dump(experiment_meta, f, indent=2)
    print(f"\n✓ Experiment metadata saved to {meta_file}")

    # Get few-shot examples (for few-shot strategy)
    few_shot_examples = get_few_shot_examples(train_data, n_examples=3)

    # Run experiments
    total_runs = len(strategies_to_run) * len(iterations_to_run)
    current_run = 0

    print(f"\n{'='*60}")
    print(f"STARTING EXPERIMENTS ({total_runs} total runs)")
    print(f"{'='*60}\n")

    all_results = {}

    for strategy in strategies_to_run:
        prompt_file = Path(PROMPT_STRATEGIES[strategy])

        if not prompt_file.exists():
            print(
                f"Warning: Prompt file not found: {prompt_file}. Skipping {strategy}."
            )
            continue

        # Skip search strategies if search tool unavailable
        if strategy in SEARCH_STRATEGIES and not search_tool:
            print(f"\nSkipping {strategy}: Search tool not available")
            continue

        # Skip RAG strategies if Weaviate unavailable
        if strategy in RAG_STRATEGIES and not weaviate_client:
            print(f"\nSkipping {strategy}: Weaviate not available")
            continue

        print(f"\n{'='*60}")
        print(f"STRATEGY: {strategy.upper()}")
        print(f"{'='*60}")

        strategy_results = {}

        for iter_idx in iterations_to_run:
            current_run += 1
            iteration_config = test_iterations[iter_idx]

            print(
                f"\n[{current_run}/{total_runs}] Running iteration {iter_idx} (seed: {iteration_config['seed']})"
            )

            # Initialize chain for this strategy (iteration-specific for RAG)
            if strategy in RAG_STRATEGIES:
                # Create retriever for this specific iteration
                retriever = FactCheckRetriever(
                    client=weaviate_client, iteration_id=iter_idx
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

            # Get test data for this iteration
            iter_test_data = get_test_data_for_iteration(test_data, iteration_config)

            # Limit samples if specified
            if args.max_samples > 0:
                iter_test_data = iter_test_data[: args.max_samples]
                print(f"  Limited to {len(iter_test_data)} samples")

            # Create run name
            run_name = f"{args.provider}_{model_name.replace('/', '_')}_{strategy}_iter{iter_idx}"

            # Output file
            output_file = args.results_dir / strategy / f"iteration_{iter_idx}.jsonl"
            output_file.parent.mkdir(parents=True, exist_ok=True)

            # Run inference
            run_inference(
                chain=chain,
                test_data=iter_test_data,
                strategy=strategy,
                iteration=iter_idx,
                output_file=output_file,
                max_samples=args.max_samples if args.max_samples > 0 else None,
            )

            strategy_results[f"iteration_{iter_idx}"] = {
                "output_file": str(output_file),
                "run_name": run_name,
            }

        all_results[strategy] = strategy_results

    # Save results summary
    summary_file = args.results_dir / "results_summary.json"
    with open(summary_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"ALL EXPERIMENTS COMPLETE")
    print(f"{'='*60}")
    print(f"\nResults saved to: {args.results_dir}")
    print(f"Summary: {summary_file}")
    print(f"\nNext step: Run evaluation with 'python evaluate_experiments.py'")


if __name__ == "__main__":
    main()
