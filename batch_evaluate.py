#!/usr/bin/env python3
"""
Batch evaluation script to compare multiple fact-checking runs.

Usage:
    python batch_evaluate.py [results_dir]

This script will find all .jsonl files in the results directory and evaluate each one,
then create a comparison report.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List
import csv
from datetime import datetime


def load_results(results_file: Path):
    """Load results from JSONL file."""
    gold_labels = []
    predicted_labels = []
    confidence_scores = []

    with open(results_file, "r") as f:
        for line in f:
            data = json.loads(line.strip())
            gold_labels.append(data["gold_label"])
            predicted_labels.append(data["verdict"])
            confidence_scores.append(data.get("confidence", 0.5))

    return gold_labels, predicted_labels, confidence_scores


def calculate_metrics(
    y_true: List[str], y_pred: List[str], confidence_scores: List[float]
) -> Dict:
    """Calculate key metrics for comparison."""
    # Overall accuracy
    correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
    accuracy = correct / len(y_true)

    # Per-class metrics
    labels = ["FALSE", "MIXED", "FACT"]
    metrics = {"accuracy": accuracy, "total_samples": len(y_true)}

    f1_scores = []
    precisions = []
    recalls = []

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

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        metrics[f"precision_{label.lower()}"] = precision
        metrics[f"recall_{label.lower()}"] = recall
        metrics[f"f1_{label.lower()}"] = f1

        f1_scores.append(f1)
        precisions.append(precision)
        recalls.append(recall)

    # Macro averages
    metrics["precision_macro"] = sum(precisions) / len(precisions)
    metrics["recall_macro"] = sum(recalls) / len(recalls)
    metrics["f1_macro"] = sum(f1_scores) / len(f1_scores)

    # Binary classification (FACT vs non-FACT)
    y_true_binary = [1 if label == "FACT" else 0 for label in y_true]
    y_pred_binary = [1 if label == "FACT" else 0 for label in y_pred]

    tp_bin = sum(
        1 for true, pred in zip(y_true_binary, y_pred_binary) if true == 1 and pred == 1
    )
    fp_bin = sum(
        1 for true, pred in zip(y_true_binary, y_pred_binary) if true == 0 and pred == 1
    )
    fn_bin = sum(
        1 for true, pred in zip(y_true_binary, y_pred_binary) if true == 1 and pred == 0
    )

    precision_bin = tp_bin / (tp_bin + fp_bin) if (tp_bin + fp_bin) > 0 else 0.0
    recall_bin = tp_bin / (tp_bin + fn_bin) if (tp_bin + fn_bin) > 0 else 0.0
    f1_bin = (
        2 * precision_bin * recall_bin / (precision_bin + recall_bin)
        if (precision_bin + recall_bin) > 0
        else 0.0
    )

    metrics["binary_f1_fact"] = f1_bin
    metrics["binary_precision_fact"] = precision_bin
    metrics["binary_recall_fact"] = recall_bin

    # Confidence stats
    metrics["mean_confidence"] = sum(confidence_scores) / len(confidence_scores)
    metrics["std_confidence"] = (
        sum((x - metrics["mean_confidence"]) ** 2 for x in confidence_scores)
        / len(confidence_scores)
    ) ** 0.5

    return metrics


def find_result_files(results_dir: Path) -> List[Path]:
    """Find all JSONL result files in the directory."""
    return list(results_dir.glob("*.jsonl"))


def create_comparison_report(all_metrics: Dict[str, Dict], output_dir: Path):
    """Create comparison report and CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Sort runs by F1-macro score
    sorted_runs = sorted(
        all_metrics.items(), key=lambda x: x[1]["f1_macro"], reverse=True
    )

    # Create comparison CSV
    csv_file = (
        output_dir / f"comparison_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )

    if sorted_runs:
        fieldnames = ["run_name"] + list(sorted_runs[0][1].keys())

        with open(csv_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for run_name, metrics in sorted_runs:
                row = {"run_name": run_name, **metrics}
                writer.writerow(row)

    # Create markdown report
    md_file = (
        output_dir / f"comparison_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    )

    with open(md_file, "w") as f:
        f.write("# Fact-Checking Model Comparison Report\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        if not sorted_runs:
            f.write("No results found.\n")
            return

        f.write(f"Evaluated {len(sorted_runs)} model runs.\n\n")

        # Summary table
        f.write("## Performance Summary\n\n")
        f.write(
            "| Rank | Model | Accuracy | F1-Macro | F1-FACT | Binary F1 | Mean Confidence |\n"
        )
        f.write(
            "|------|-------|----------|----------|---------|-----------|----------------|\n"
        )

        for i, (run_name, metrics) in enumerate(sorted_runs, 1):
            f.write(
                f"| {i} | {run_name} | {metrics['accuracy']:.4f} | {metrics['f1_macro']:.4f} | "
                f"{metrics['f1_fact']:.4f} | {metrics['binary_f1_fact']:.4f} | {metrics['mean_confidence']:.4f} |\n"
            )

        f.write("\n## Detailed Analysis\n\n")

        # Best performing model
        best_run, best_metrics = sorted_runs[0]
        f.write(f"### Best Performing Model: {best_run}\n")
        f.write(f"- **Accuracy**: {best_metrics['accuracy']:.4f}\n")
        f.write(f"- **F1-Macro**: {best_metrics['f1_macro']:.4f}\n")
        f.write(f"- **Binary F1 (FACT)**: {best_metrics['binary_f1_fact']:.4f}\n")
        f.write(f"- **Mean Confidence**: {best_metrics['mean_confidence']:.4f}\n\n")

        # Per-class performance for best model
        f.write("**Per-class performance (best model):**\n")
        for label in ["FALSE", "MIXED", "FACT"]:
            label_lower = label.lower()
            f.write(
                f"- **{label}**: P={best_metrics[f'precision_{label_lower}']:.3f}, "
                f"R={best_metrics[f'recall_{label_lower}']:.3f}, "
                f"F1={best_metrics[f'f1_{label_lower}']:.3f}\n"
            )
        f.write("\n")

        # Performance comparison
        if len(sorted_runs) > 1:
            f.write("### Performance Gaps\n")
            worst_run, worst_metrics = sorted_runs[-1]

            acc_gap = best_metrics["accuracy"] - worst_metrics["accuracy"]
            f1_gap = best_metrics["f1_macro"] - worst_metrics["f1_macro"]

            f.write(f"- **Accuracy gap** (best vs worst): {acc_gap:.4f}\n")
            f.write(f"- **F1-Macro gap** (best vs worst): {f1_gap:.4f}\n\n")

        # Recommendations
        f.write("### Recommendations\n\n")

        if best_metrics["f1_macro"] > 0.8:
            f.write(
                "✅ **Excellent performance**: The best model shows strong fact-checking capabilities.\n"
            )
        elif best_metrics["f1_macro"] > 0.7:
            f.write(
                "👍 **Good performance**: The best model performs well but has room for improvement.\n"
            )
        elif best_metrics["f1_macro"] > 0.6:
            f.write(
                "⚠️ **Fair performance**: Consider model improvements or additional training data.\n"
            )
        else:
            f.write(
                "❌ **Poor performance**: Significant improvements needed in model architecture or training.\n"
            )

        f.write("\n")

        # Class-specific recommendations
        worst_class_f1 = min(
            best_metrics["f1_false"], best_metrics["f1_mixed"], best_metrics["f1_fact"]
        )
        if worst_class_f1 < 0.6:
            if best_metrics["f1_mixed"] == worst_class_f1:
                f.write(
                    "- **Focus on MIXED class**: This appears to be the most challenging category.\n"
                )
            elif best_metrics["f1_false"] == worst_class_f1:
                f.write(
                    "- **Improve FALSE detection**: Consider better negative examples in training.\n"
                )
            else:
                f.write(
                    "- **Enhance FACT recognition**: May need more positive examples or better features.\n"
                )

        confidence_assessment = (
            "high"
            if best_metrics["mean_confidence"] > 0.8
            else "moderate" if best_metrics["mean_confidence"] > 0.6 else "low"
        )
        f.write(
            f"- **Confidence calibration**: Model shows {confidence_assessment} confidence levels.\n"
        )

    print(f"\nComparison report saved to:")
    print(f"  CSV: {csv_file}")
    print(f"  Markdown: {md_file}")


def main():
    results_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results")

    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        sys.exit(1)

    # Find all result files
    result_files = find_result_files(results_dir)

    if not result_files:
        print(f"No JSONL result files found in {results_dir}")
        sys.exit(1)

    print(f"Found {len(result_files)} result files:")
    for f in result_files:
        print(f"  {f.name}")
    print()

    # Evaluate each file
    all_metrics = {}

    for result_file in result_files:
        try:
            print(f"Evaluating {result_file.name}...")
            y_true, y_pred, confidence_scores = load_results(result_file)
            metrics = calculate_metrics(y_true, y_pred, confidence_scores)
            all_metrics[result_file.stem] = metrics
            print(
                f"  ✓ Accuracy: {metrics['accuracy']:.4f}, F1-Macro: {metrics['f1_macro']:.4f}"
            )
        except Exception as e:
            print(f"  ✗ Error: {e}")

    if not all_metrics:
        print("No results could be processed.")
        sys.exit(1)

    print(f"\nProcessed {len(all_metrics)} runs successfully.")

    # Create comparison report
    output_dir = Path("results/summaries")
    create_comparison_report(all_metrics, output_dir)

    # Print quick summary
    sorted_runs = sorted(
        all_metrics.items(), key=lambda x: x[1]["f1_macro"], reverse=True
    )

    print("\n" + "=" * 60)
    print("QUICK COMPARISON")
    print("=" * 60)
    print(
        f"{'Rank':<4} {'Model':<20} {'Accuracy':<10} {'F1-Macro':<10} {'F1-FACT':<10}"
    )
    print("-" * 60)

    for i, (run_name, metrics) in enumerate(sorted_runs[:5], 1):  # Top 5
        print(
            f"{i:<4} {run_name:<20} {metrics['accuracy']:<10.4f} {metrics['f1_macro']:<10.4f} {metrics['f1_fact']:<10.4f}"
        )

    if len(sorted_runs) > 5:
        print(f"... and {len(sorted_runs) - 5} more")

    print(f"\n✓ Complete comparison report saved to results/summaries/")


if __name__ == "__main__":
    main()
