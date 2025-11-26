#!/usr/bin/env python3
"""
Identify and re-run only the samples that failed with errors in the experiment results.
Replaces error entries in the original JSONL files with corrected results.
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.chains.fact_check import build_fact_check_chain
from src.models.llm import make_chat_model
from src.config import settings


PROMPT_STRATEGIES = {
    "zero_shot": "src/prompts/fact_check.txt",
    "one_shot": "src/prompts/fact_check_oneshot.txt",
    "few_shot": "src/prompts/fact_check_fewshot.txt",
}


def find_error_samples(results_dir: Path):
    """
    Scan all result files and identify samples with errors.
    Returns dict: {strategy: {iteration: [error_samples]}}
    """
    error_samples = {}
    
    for strategy_dir in results_dir.iterdir():
        if not strategy_dir.is_dir():
            continue
            
        strategy = strategy_dir.name
        error_samples[strategy] = {}
        
        for result_file in sorted(strategy_dir.glob("iteration_*.jsonl")):
            iteration = int(result_file.stem.split("_")[1])
            errors = []
            
            with open(result_file, "r") as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        result = json.loads(line)
                        if result.get("verdict") == "ERROR" or "error" in result:
                            errors.append(result)
                    except json.JSONDecodeError:
                        print(f"Warning: Invalid JSON at {result_file}:{line_num}")
            
            if errors:
                error_samples[strategy][iteration] = errors
                
    return error_samples


def rerun_sample(chain, sample, strategy, iteration):
    """Re-run inference for a single error sample."""
    claim_id = sample["claim_id"]
    claim = sample["claim"]
    gold_label = sample["gold_label"]
    
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
            "sample_idx": sample["sample_idx"],
            "claim_id": claim_id,
            "claim": claim,
            "gold_label": gold_label,
            "verdict": result_data.get("verdict", "UNKNOWN"),
            "confidence": result_data.get("confidence", 0.5),
            "rationale": result_data.get("rationale", ""),
            "cited_knowledge": result_data.get("cited_knowledge", ""),
            "safety_notes": result_data.get("safety_notes", ""),
            "timestamp": datetime.now().isoformat(),
            "rerun": True,  # Mark as re-run
        }
        
        return output
        
    except Exception as e:
        print(f"\nError re-running sample {claim_id}: {str(e)}")
        return {
            "iteration": iteration,
            "strategy": strategy,
            "sample_idx": sample["sample_idx"],
            "claim_id": claim_id,
            "claim": claim,
            "gold_label": gold_label,
            "verdict": "ERROR",
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
            "rerun": True,
            "rerun_failed": True,
        }


def update_results_file(result_file: Path, fixed_results: dict):
    """
    Update the JSONL file by replacing error entries with fixed results.
    fixed_results: dict mapping sample_idx to new result
    """
    # Read all results
    all_results = []
    with open(result_file, "r") as f:
        for line in f:
            result = json.loads(line)
            sample_idx = result["sample_idx"]
            
            # Replace if we have a fix
            if sample_idx in fixed_results:
                all_results.append(fixed_results[sample_idx])
            else:
                all_results.append(result)
    
    # Write back
    with open(result_file, "w") as f:
        for result in all_results:
            f.write(json.dumps(result) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Re-run only error samples and fix result files"
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("uku_expirements"),
        help="Directory containing experiment results (default: uku_expirements)",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="openai",
        help="LLM provider (default: openai)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name (default: from .env)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=10,
        help="Number of parallel workers (default: 10)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only show what would be fixed without making changes",
    )
    
    args = parser.parse_args()
    
    # Find all error samples
    print("Scanning for error samples...")
    error_samples = find_error_samples(args.results_dir)
    
    # Count total errors
    total_errors = sum(
        len(errors)
        for strategy_errors in error_samples.values()
        for errors in strategy_errors.values()
    )
    
    if total_errors == 0:
        print("✓ No error samples found! All experiments completed successfully.")
        return
    
    print(f"\nFound {total_errors} error samples:")
    for strategy, iterations in error_samples.items():
        for iteration, errors in iterations.items():
            print(f"  {strategy}/iteration_{iteration}: {len(errors)} errors")
    
    if args.dry_run:
        print("\n[DRY RUN] Would re-run these samples. Use without --dry-run to fix.")
        return
    
    # Initialize LLM
    model_name = args.model or settings.openai_model
    llm = make_chat_model(
        provider=args.provider,
        model_name=model_name,
        api_key=settings.openai_api_key,
    )
    
    print(f"\nRe-running {total_errors} error samples...")
    print(f"Provider: {args.provider}")
    print(f"Model: {model_name}")
    print(f"Workers: {args.max_workers}\n")
    
    # Process each strategy
    for strategy, iterations in error_samples.items():
        if not iterations:
            continue
            
        print(f"\n{'='*60}")
        print(f"STRATEGY: {strategy.upper()}")
        print(f"{'='*60}")
        
        # Build chain for this strategy
        prompt_file = Path(PROMPT_STRATEGIES[strategy])
        chain = build_fact_check_chain(llm=llm, prompt_path=prompt_file)
        
        # Process each iteration
        for iteration, errors in iterations.items():
            print(f"\nIteration {iteration}: Re-running {len(errors)} samples...")
            
            fixed_results = {}
            lock = threading.Lock()
            
            # Re-run samples in parallel
            with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
                futures = {
                    executor.submit(rerun_sample, chain, sample, strategy, iteration): sample
                    for sample in errors
                }
                
                with tqdm(total=len(errors), desc=f"{strategy} iter {iteration}") as pbar:
                    for future in as_completed(futures):
                        result = future.result()
                        with lock:
                            fixed_results[result["sample_idx"]] = result
                        pbar.update(1)
            
            # Update the results file
            result_file = args.results_dir / strategy / f"iteration_{iteration}.jsonl"
            update_results_file(result_file, fixed_results)
            
            # Count successful fixes
            successful_fixes = sum(
                1 for r in fixed_results.values()
                if r.get("verdict") != "ERROR"
            )
            still_errors = len(fixed_results) - successful_fixes
            
            print(f"  ✓ Fixed: {successful_fixes}/{len(errors)}")
            if still_errors > 0:
                print(f"  ⚠ Still errors: {still_errors}")
    
    print(f"\n{'='*60}")
    print(f"ERROR FIXING COMPLETE")
    print(f"{'='*60}")
    print(f"\nUpdated result files in: {args.results_dir}")
    
    # Re-scan to show final status
    print("\nFinal status:")
    final_errors = find_error_samples(args.results_dir)
    final_total = sum(
        len(errors)
        for strategy_errors in final_errors.values()
        for errors in strategy_errors.values()
    )
    
    if final_total == 0:
        print("✓ All errors fixed!")
    else:
        print(f"⚠ {final_total} errors remaining:")
        for strategy, iterations in final_errors.items():
            for iteration, errors in iterations.items():
                print(f"  {strategy}/iteration_{iteration}: {len(errors)} errors")


if __name__ == "__main__":
    main()
