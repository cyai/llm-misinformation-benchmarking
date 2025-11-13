#!/usr/bin/env python3
"""
Evaluate fact-checking experiments across all strategies and iterations.

This script:
1. Loads results from all experiments
2. Evaluates each iteration separately
3. Computes aggregate statistics across iterations
4. Generates comprehensive comparison reports

Usage:
    python evaluate_experiments.py [--results-dir DIR]
"""

import argparse
import sys
import json
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import csv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))


def load_results(results_file: Path):
    """Load results from JSONL file."""
    gold_labels = []
    predicted_labels = []
    confidence_scores = []

    with open(results_file, "r") as f:
        for line in f:
            data = json.loads(line.strip())
            # Skip error records
            if data.get("verdict") == "ERROR":
                continue
            gold_labels.append(data["gold_label"])
            predicted_labels.append(data["verdict"])
            confidence_scores.append(data.get("confidence", 0.5))

    return gold_labels, predicted_labels, confidence_scores


def calculate_metrics(y_true, y_pred, confidence_scores):
    """Calculate comprehensive metrics."""
    # Overall accuracy
    correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
    accuracy = correct / len(y_true) if len(y_true) > 0 else 0.0

    # Binary classification metrics (FACT vs FALSE)
    labels = ["FACT", "FALSE"]
    metrics = {"accuracy": accuracy, "n_samples": len(y_true)}

    # Per-class metrics
    for label in labels:
        tp = sum(
            1 for true, pred in zip(y_true, y_pred) if true == label and pred == label
        )
        fp = sum(
            1 for true, pred in zip(y_true, y_pred) if true != label and pred == label
        )
        fn = sum(
            1 for true, pred in zip(y_true, y_pred) if true == label and pred != label
        )
        tn = sum(
            1 for true, pred in zip(y_true, y_pred) if true != label and pred != label
        )

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        metrics[f"precision_{label.lower()}"] = precision
        metrics[f"recall_{label.lower()}"] = recall
        metrics[f"f1_{label.lower()}"] = f1
        metrics[f"specificity_{label.lower()}"] = specificity

    # Macro averages
    metrics["precision_macro"] = (
        metrics["precision_fact"] + metrics["precision_false"]
    ) / 2
    metrics["recall_macro"] = (metrics["recall_fact"] + metrics["recall_false"]) / 2
    metrics["f1_macro"] = (metrics["f1_fact"] + metrics["f1_false"]) / 2

    # Confidence stats
    if confidence_scores:
        metrics["mean_confidence"] = sum(confidence_scores) / len(confidence_scores)
        metrics["std_confidence"] = (
            (
                sum((x - metrics["mean_confidence"]) ** 2 for x in confidence_scores)
                / len(confidence_scores)
            )
            ** 0.5
            if len(confidence_scores) > 1
            else 0.0
        )
    else:
        metrics["mean_confidence"] = 0.0
        metrics["std_confidence"] = 0.0

    # Confusion matrix
    metrics["confusion_matrix"] = {
        "tp_fact": sum(
            1 for true, pred in zip(y_true, y_pred) if true == "FACT" and pred == "FACT"
        ),
        "fp_fact": sum(
            1
            for true, pred in zip(y_true, y_pred)
            if true == "FALSE" and pred == "FACT"
        ),
        "fn_fact": sum(
            1
            for true, pred in zip(y_true, y_pred)
            if true == "FACT" and pred == "FALSE"
        ),
        "tn_fact": sum(
            1
            for true, pred in zip(y_true, y_pred)
            if true == "FALSE" and pred == "FALSE"
        ),
    }

    return metrics


def aggregate_iteration_metrics(iteration_metrics):
    """Aggregate metrics across iterations."""
    if not iteration_metrics:
        return {}

    # Get all metric keys (excluding confusion matrix)
    metric_keys = [
        k
        for k in iteration_metrics[0].keys()
        if k != "confusion_matrix" and k != "n_samples"
    ]

    aggregated = {}

    for key in metric_keys:
        values = [m[key] for m in iteration_metrics if key in m]
        if values:
            aggregated[f"{key}_mean"] = sum(values) / len(values)
            aggregated[f"{key}_std"] = (
                (
                    sum((x - aggregated[f"{key}_mean"]) ** 2 for x in values)
                    / len(values)
                )
                ** 0.5
                if len(values) > 1
                else 0.0
            )
            aggregated[f"{key}_min"] = min(values)
            aggregated[f"{key}_max"] = max(values)

    # Total samples
    aggregated["total_samples"] = sum(m.get("n_samples", 0) for m in iteration_metrics)
    aggregated["n_iterations"] = len(iteration_metrics)

    return aggregated


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate all fact-checking experiments"
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/experiments"),
        help="Directory containing experiment results (default: results/experiments)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/evaluations"),
        help="Directory to save evaluation results (default: results/evaluations)",
    )

    args = parser.parse_args()

    if not args.results_dir.exists():
        print(f"Error: Results directory not found: {args.results_dir}")
        print("Please run 'python run_experiments.py' first.")
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load experiment metadata
    meta_file = args.results_dir / "experiment_metadata.json"
    if meta_file.exists():
        with open(meta_file, "r") as f:
            experiment_meta = json.load(f)
        print("Loaded experiment metadata")
    else:
        print("Warning: No experiment metadata found")
        experiment_meta = {}

    # Find all result files
    print(f"\nScanning {args.results_dir} for results...")

    all_evaluations = {}
    strategy_dirs = [d for d in args.results_dir.iterdir() if d.is_dir()]

    for strategy_dir in strategy_dirs:
        strategy_name = strategy_dir.name
        print(f"\nEvaluating strategy: {strategy_name}")

        iteration_files = sorted(strategy_dir.glob("iteration_*.jsonl"))

        if not iteration_files:
            print(f"  No iteration files found in {strategy_dir}")
            continue

        strategy_metrics = []
        iteration_details = {}

        for iter_file in iteration_files:
            iter_name = iter_file.stem  # e.g., 'iteration_0'

            try:
                y_true, y_pred, confidence_scores = load_results(iter_file)

                if not y_true:
                    print(f"  {iter_name}: No valid results")
                    continue

                metrics = calculate_metrics(y_true, y_pred, confidence_scores)
                strategy_metrics.append(metrics)
                iteration_details[iter_name] = metrics

                print(
                    f"  {iter_name}: Acc={metrics['accuracy']:.4f}, F1-Macro={metrics['f1_macro']:.4f}, F1-FACT={metrics['f1_fact']:.4f}"
                )

            except Exception as e:
                print(f"  {iter_name}: Error - {e}")

        if strategy_metrics:
            # Aggregate across iterations
            aggregated = aggregate_iteration_metrics(strategy_metrics)

            all_evaluations[strategy_name] = {
                "iterations": iteration_details,
                "aggregated": aggregated,
            }

            print(f"\n  Aggregated results:")
            print(
                f"    Accuracy: {aggregated['accuracy_mean']:.4f} ± {aggregated['accuracy_std']:.4f}"
            )
            print(
                f"    F1-Macro: {aggregated['f1_macro_mean']:.4f} ± {aggregated['f1_macro_std']:.4f}"
            )
            print(
                f"    F1-FACT: {aggregated['f1_fact_mean']:.4f} ± {aggregated['f1_fact_std']:.4f}"
            )

    if not all_evaluations:
        print("\nNo results found to evaluate.")
        sys.exit(1)

    # Save detailed evaluation
    eval_file = args.output_dir / "detailed_evaluation.json"
    with open(eval_file, "w") as f:
        json.dump(
            {
                "metadata": experiment_meta,
                "timestamp": datetime.now().isoformat(),
                "evaluations": all_evaluations,
            },
            f,
            indent=2,
        )

    print(f"\n✓ Detailed evaluation saved to {eval_file}")

    # Create comparison CSV
    csv_file = args.output_dir / "comparison_summary.csv"

    with open(csv_file, "w", newline="") as f:
        fieldnames = [
            "strategy",
            "accuracy_mean",
            "accuracy_std",
            "f1_macro_mean",
            "f1_macro_std",
            "f1_fact_mean",
            "f1_fact_std",
            "f1_false_mean",
            "f1_false_std",
            "precision_fact_mean",
            "recall_fact_mean",
            "precision_false_mean",
            "recall_false_mean",
            "mean_confidence_mean",
            "n_iterations",
            "total_samples",
        ]

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for strategy, results in sorted(all_evaluations.items()):
            agg = results["aggregated"]
            row = {"strategy": strategy}
            for field in fieldnames[1:]:
                row[field] = agg.get(field, 0.0)
            writer.writerow(row)

    print(f"✓ Comparison summary saved to {csv_file}")

    # Create markdown report
    md_file = args.output_dir / "evaluation_report.md"

    with open(md_file, "w") as f:
        f.write("# Fact-Checking Experiment Evaluation Report\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        if experiment_meta:
            f.write("## Experiment Configuration\n\n")
            f.write(f"- **Model**: {experiment_meta.get('model', 'N/A')}\n")
            f.write(f"- **Provider**: {experiment_meta.get('provider', 'N/A')}\n")
            f.write(
                f"- **Strategies**: {', '.join(experiment_meta.get('strategies', []))}\n"
            )
            f.write(
                f"- **Test iterations**: {len(experiment_meta.get('iterations', []))}\n\n"
            )

        f.write("## Overall Results\n\n")
        f.write("| Strategy | Accuracy | F1-Macro | F1-FACT | F1-FALSE | Samples |\n")
        f.write("|----------|----------|----------|---------|----------|----------|\n")

        # Sort by F1-macro
        sorted_strategies = sorted(
            all_evaluations.items(),
            key=lambda x: x[1]["aggregated"]["f1_macro_mean"],
            reverse=True,
        )

        for strategy, results in sorted_strategies:
            agg = results["aggregated"]
            f.write(
                f"| {strategy} | "
                f"{agg['accuracy_mean']:.4f} ± {agg['accuracy_std']:.4f} | "
                f"{agg['f1_macro_mean']:.4f} ± {agg['f1_macro_std']:.4f} | "
                f"{agg['f1_fact_mean']:.4f} ± {agg['f1_fact_std']:.4f} | "
                f"{agg['f1_false_mean']:.4f} ± {agg['f1_false_std']:.4f} | "
                f"{agg['total_samples']} |\n"
            )

        f.write("\n## Detailed Per-Strategy Results\n\n")

        for strategy, results in sorted_strategies:
            f.write(f"### {strategy.upper()}\n\n")

            agg = results["aggregated"]

            f.write("**Aggregated Metrics (Mean ± Std):**\n\n")
            f.write(
                f"- Accuracy: {agg['accuracy_mean']:.4f} ± {agg['accuracy_std']:.4f}\n"
            )
            f.write(
                f"- F1-Macro: {agg['f1_macro_mean']:.4f} ± {agg['f1_macro_std']:.4f}\n"
            )
            f.write(
                f"- Precision (FACT): {agg['precision_fact_mean']:.4f} ± {agg['precision_fact_std']:.4f}\n"
            )
            f.write(
                f"- Recall (FACT): {agg['recall_fact_mean']:.4f} ± {agg['recall_fact_std']:.4f}\n"
            )
            f.write(
                f"- F1-Score (FACT): {agg['f1_fact_mean']:.4f} ± {agg['f1_fact_std']:.4f}\n"
            )
            f.write(
                f"- Precision (FALSE): {agg['precision_false_mean']:.4f} ± {agg['precision_false_std']:.4f}\n"
            )
            f.write(
                f"- Recall (FALSE): {agg['recall_false_mean']:.4f} ± {agg['recall_false_std']:.4f}\n"
            )
            f.write(
                f"- F1-Score (FALSE): {agg['f1_false_mean']:.4f} ± {agg['f1_false_std']:.4f}\n"
            )
            f.write(
                f"- Mean Confidence: {agg['mean_confidence_mean']:.4f} ± {agg['mean_confidence_std']:.4f}\n"
            )
            f.write(f"- Total Samples: {agg['total_samples']}\n")
            f.write(f"- Iterations: {agg['n_iterations']}\n\n")

            # Per-iteration breakdown
            f.write("**Per-Iteration Results:**\n\n")
            f.write("| Iteration | Accuracy | F1-Macro | F1-FACT | F1-FALSE |\n")
            f.write("|-----------|----------|----------|---------|----------|\n")

            for iter_name, metrics in sorted(results["iterations"].items()):
                f.write(
                    f"| {iter_name} | "
                    f"{metrics['accuracy']:.4f} | "
                    f"{metrics['f1_macro']:.4f} | "
                    f"{metrics['f1_fact']:.4f} | "
                    f"{metrics['f1_false']:.4f} |\n"
                )

            f.write("\n")

        # Best performing strategy
        best_strategy, best_results = sorted_strategies[0]
        f.write("## Key Findings\n\n")
        f.write(f"### Best Performing Strategy: **{best_strategy}**\n\n")
        f.write(
            f"- **Accuracy**: {best_results['aggregated']['accuracy_mean']:.4f} ± {best_results['aggregated']['accuracy_std']:.4f}\n"
        )
        f.write(
            f"- **F1-Macro**: {best_results['aggregated']['f1_macro_mean']:.4f} ± {best_results['aggregated']['f1_macro_std']:.4f}\n"
        )
        f.write(
            f"- **Consistency**: {'High' if best_results['aggregated']['f1_macro_std'] < 0.02 else 'Moderate' if best_results['aggregated']['f1_macro_std'] < 0.05 else 'Low'} (std: {best_results['aggregated']['f1_macro_std']:.4f})\n\n"
        )

        # Recommendations
        f.write("### Recommendations\n\n")
        best_f1 = best_results["aggregated"]["f1_macro_mean"]

        if best_f1 > 0.8:
            f.write(
                "✅ **Excellent performance**: The model shows strong fact-checking capabilities.\n"
            )
        elif best_f1 > 0.7:
            f.write(
                "👍 **Good performance**: The model performs well but has room for improvement.\n"
            )
        elif best_f1 > 0.6:
            f.write(
                "⚠️ **Fair performance**: Consider model improvements or additional training data.\n"
            )
        else:
            f.write("❌ **Poor performance**: Significant improvements needed.\n")

        f.write("\n")

        # Compare strategies
        if len(sorted_strategies) > 1:
            f.write("### Strategy Comparison\n\n")
            best_f1 = sorted_strategies[0][1]["aggregated"]["f1_macro_mean"]
            worst_f1 = sorted_strategies[-1][1]["aggregated"]["f1_macro_mean"]
            gap = best_f1 - worst_f1

            f.write(f"- **Performance gap** (best vs worst): {gap:.4f}\n")

            if gap > 0.1:
                f.write(
                    "- **Finding**: Significant difference between strategies suggests prompt engineering matters.\n"
                )
            elif gap > 0.05:
                f.write("- **Finding**: Moderate difference between strategies.\n")
            else:
                f.write("- **Finding**: Strategies perform similarly.\n")

    print(f"✓ Evaluation report saved to {md_file}")

    # Print summary
    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")

    for strategy, results in sorted_strategies[:3]:  # Top 3
        agg = results["aggregated"]
        print(f"\n{strategy.upper()}:")
        print(f"  Accuracy: {agg['accuracy_mean']:.4f} ± {agg['accuracy_std']:.4f}")
        print(f"  F1-Macro: {agg['f1_macro_mean']:.4f} ± {agg['f1_macro_std']:.4f}")
        print(f"  F1-FACT:  {agg['f1_fact_mean']:.4f} ± {agg['f1_fact_std']:.4f}")

    print(f"\n✓ Evaluation complete! Check {args.output_dir} for detailed reports.")


if __name__ == "__main__":
    main()
