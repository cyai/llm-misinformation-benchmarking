import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class EvaluationSummaryGenerator:
    """Generate comprehensive summary reports and visualizations."""

    def __init__(self, output_dir: Path = None):
        """
        Initialize summary generator.

        Args:
            output_dir: Directory to save summary files. Defaults to results/summaries/
        """
        self.output_dir = output_dir or Path("results/summaries")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_text_summary(self, evaluation: Dict[str, Any], run_name: str) -> str:
        """
        Generate a comprehensive text summary of evaluation results.
        """
        summary = []
        summary.append(f"# Fact-Checking Evaluation Summary: {run_name}")
        summary.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        summary.append("=" * 60)
        summary.append("")

        # Dataset info
        dataset_info = evaluation["dataset_info"]
        summary.append("## Dataset Information")
        summary.append(f"Total samples: {dataset_info['total_samples']}")
        summary.append("")
        summary.append("True label distribution:")
        for label, count in dataset_info["label_distribution_true"].items():
            percentage = (count / dataset_info["total_samples"]) * 100
            summary.append(f"  {label}: {count} ({percentage:.1f}%)")
        summary.append("")
        summary.append("Predicted label distribution:")
        for label, count in dataset_info["label_distribution_pred"].items():
            percentage = (count / dataset_info["total_samples"]) * 100
            summary.append(f"  {label}: {count} ({percentage:.1f}%)")
        summary.append("")

        # Main metrics
        multi_metrics = evaluation["multiclass_metrics"]
        binary_metrics = evaluation["binary_metrics"]
        auc_metrics = evaluation["auc_metrics"]

        summary.append("## Key Performance Metrics")
        summary.append("")
        summary.append("### Overall Performance")
        summary.append(f"Accuracy: {multi_metrics['accuracy']:.4f}")
        summary.append(f"F1-Score (Macro): {multi_metrics['f1_macro']:.4f}")
        summary.append(f"F1-Score (Micro): {multi_metrics['f1_micro']:.4f}")
        summary.append(f"F1-Score (Weighted): {multi_metrics['f1_weighted']:.4f}")
        summary.append("")

        summary.append("### Binary Classification (FACT vs non-FACT)")
        summary.append(f"Accuracy: {binary_metrics['binary_accuracy']:.4f}")
        summary.append(
            f"Precision (FACT): {binary_metrics['binary_precision_fact']:.4f}"
        )
        summary.append(f"Recall (FACT): {binary_metrics['binary_recall_fact']:.4f}")
        summary.append(f"F1-Score (FACT): {binary_metrics['binary_f1_fact']:.4f}")
        summary.append(
            f"Precision (non-FACT): {binary_metrics['binary_precision_nonfact']:.4f}"
        )
        summary.append(
            f"Recall (non-FACT): {binary_metrics['binary_recall_nonfact']:.4f}"
        )
        summary.append(
            f"F1-Score (non-FACT): {binary_metrics['binary_f1_nonfact']:.4f}"
        )
        summary.append("")

        summary.append("### AUC Scores")
        summary.append(
            f"Binary AUC (FACT vs non-FACT): {auc_metrics.get('auc_binary', 'N/A'):.4f}"
        )
        summary.append(
            f"Multiclass AUC (One-vs-Rest): {auc_metrics.get('auc_multiclass_ovr', 'N/A'):.4f}"
        )
        summary.append("")

        # Per-class metrics
        summary.append("### Per-Class Performance")
        for label in ["FALSE", "MIXED", "FACT"]:
            label_lower = label.lower()
            if f"precision_{label_lower}" in multi_metrics:
                summary.append(f"{label}:")
                summary.append(
                    f"  Precision: {multi_metrics[f'precision_{label_lower}']:.4f}"
                )
                summary.append(
                    f"  Recall: {multi_metrics[f'recall_{label_lower}']:.4f}"
                )
                summary.append(f"  F1-Score: {multi_metrics[f'f1_{label_lower}']:.4f}")
                summary.append("")

        # Confidence analysis
        analysis = evaluation["analysis"]
        summary.append("### Confidence Analysis")
        summary.append(f"Mean confidence: {analysis['mean_confidence']:.4f}")
        summary.append(f"Std confidence: {analysis['std_confidence']:.4f}")
        summary.append("")

        summary.append("Confidence by true label:")
        for label, stats in analysis["confidence_by_label_true"].items():
            summary.append(f"  {label}: {stats['mean']:.4f} ± {stats['std']:.4f}")
        summary.append("")

        summary.append("Confidence by predicted label:")
        for label, stats in analysis["confidence_by_label_pred"].items():
            summary.append(f"  {label}: {stats['mean']:.4f} ± {stats['std']:.4f}")
        summary.append("")

        # Confusion matrix
        summary.append("### Confusion Matrix")
        cm = evaluation["confusion_matrix"]
        labels = ["FALSE", "MIXED", "FACT"]

        # Header
        summary.append("Predicted →")
        header = "True ↓     " + "".join(f"{label:>8}" for label in labels)
        summary.append(header)

        # Rows
        for i, true_label in enumerate(labels):
            row = f"{true_label:>8}   " + "".join(
                f"{cm[i][j]:>8}" for j in range(len(labels))
            )
            summary.append(row)
        summary.append("")

        # Classification report
        summary.append("### Detailed Classification Report")
        summary.append("```")
        summary.append(evaluation["classification_report"])
        summary.append("```")

        return "\n".join(summary)

    def generate_visualizations(self, evaluation: Dict[str, Any], run_name: str):
        """
        Generate visualization plots for the evaluation results.
        """
        # Set up the plotting style
        plt.style.use("default")
        sns.set_palette("husl")

        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(
            f"Fact-Checking Evaluation Results: {run_name}",
            fontsize=16,
            fontweight="bold",
        )

        # 1. Label Distribution Comparison
        ax = axes[0, 0]
        labels = ["FALSE", "MIXED", "FACT"]
        true_counts = [
            evaluation["dataset_info"]["label_distribution_true"][label]
            for label in labels
        ]
        pred_counts = [
            evaluation["dataset_info"]["label_distribution_pred"][label]
            for label in labels
        ]

        x = np.arange(len(labels))
        width = 0.35

        ax.bar(x - width / 2, true_counts, width, label="True", alpha=0.8)
        ax.bar(x + width / 2, pred_counts, width, label="Predicted", alpha=0.8)
        ax.set_xlabel("Labels")
        ax.set_ylabel("Count")
        ax.set_title("Label Distribution: True vs Predicted")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. Confusion Matrix Heatmap
        ax = axes[0, 1]
        cm = np.array(evaluation["confusion_matrix"])
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            ax=ax,
        )
        ax.set_title("Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

        # 3. Performance Metrics Comparison
        ax = axes[0, 2]
        metrics = ["Precision", "Recall", "F1-Score"]
        false_scores = [
            evaluation["multiclass_metrics"]["precision_false"],
            evaluation["multiclass_metrics"]["recall_false"],
            evaluation["multiclass_metrics"]["f1_false"],
        ]
        mixed_scores = [
            evaluation["multiclass_metrics"]["precision_mixed"],
            evaluation["multiclass_metrics"]["recall_mixed"],
            evaluation["multiclass_metrics"]["f1_mixed"],
        ]
        fact_scores = [
            evaluation["multiclass_metrics"]["precision_fact"],
            evaluation["multiclass_metrics"]["recall_fact"],
            evaluation["multiclass_metrics"]["f1_fact"],
        ]

        x = np.arange(len(metrics))
        width = 0.25

        ax.bar(x - width, false_scores, width, label="FALSE", alpha=0.8)
        ax.bar(x, mixed_scores, width, label="MIXED", alpha=0.8)
        ax.bar(x + width, fact_scores, width, label="FACT", alpha=0.8)
        ax.set_xlabel("Metrics")
        ax.set_ylabel("Score")
        ax.set_title("Per-Class Performance Metrics")
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)

        # 4. Overall Metrics Bar Chart
        ax = axes[1, 0]
        overall_metrics = ["Accuracy", "F1-Macro", "F1-Micro", "F1-Weighted"]
        overall_scores = [
            evaluation["multiclass_metrics"]["accuracy"],
            evaluation["multiclass_metrics"]["f1_macro"],
            evaluation["multiclass_metrics"]["f1_micro"],
            evaluation["multiclass_metrics"]["f1_weighted"],
        ]

        bars = ax.bar(overall_metrics, overall_scores, alpha=0.8, color="skyblue")
        ax.set_ylabel("Score")
        ax.set_title("Overall Performance Metrics")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)

        # Add value labels on bars
        for bar, score in zip(bars, overall_scores):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.01,
                f"{score:.3f}",
                ha="center",
                va="bottom",
            )

        # 5. Confidence Distribution
        ax = axes[1, 1]
        confidence_true = evaluation["analysis"]["confidence_by_label_true"]
        confidence_pred = evaluation["analysis"]["confidence_by_label_pred"]

        labels_conf = list(confidence_true.keys())
        means_true = [confidence_true[label]["mean"] for label in labels_conf]
        stds_true = [confidence_true[label]["std"] for label in labels_conf]
        means_pred = [confidence_pred[label]["mean"] for label in labels_conf]
        stds_pred = [confidence_pred[label]["std"] for label in labels_conf]

        x = np.arange(len(labels_conf))
        width = 0.35

        ax.bar(
            x - width / 2,
            means_true,
            width,
            yerr=stds_true,
            label="By True Label",
            alpha=0.8,
            capsize=5,
        )
        ax.bar(
            x + width / 2,
            means_pred,
            width,
            yerr=stds_pred,
            label="By Predicted Label",
            alpha=0.8,
            capsize=5,
        )
        ax.set_xlabel("Labels")
        ax.set_ylabel("Mean Confidence")
        ax.set_title("Confidence Scores by Label")
        ax.set_xticks(x)
        ax.set_xticklabels(labels_conf)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)

        # 6. Binary vs Multiclass Comparison
        ax = axes[1, 2]
        binary_metrics = evaluation["binary_metrics"]
        multi_metrics = evaluation["multiclass_metrics"]

        comparison_metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
        binary_scores = [
            binary_metrics["binary_accuracy"],
            binary_metrics["binary_precision_fact"],
            binary_metrics["binary_recall_fact"],
            binary_metrics["binary_f1_fact"],
        ]
        multi_scores = [
            multi_metrics["accuracy"],
            multi_metrics["precision_macro"],
            multi_metrics["recall_macro"],
            multi_metrics["f1_macro"],
        ]

        x = np.arange(len(comparison_metrics))
        width = 0.35

        ax.bar(
            x - width / 2,
            binary_scores,
            width,
            label="Binary (FACT vs non-FACT)",
            alpha=0.8,
        )
        ax.bar(
            x + width / 2,
            multi_scores,
            width,
            label="Multiclass (Macro Avg)",
            alpha=0.8,
        )
        ax.set_xlabel("Metrics")
        ax.set_ylabel("Score")
        ax.set_title("Binary vs Multiclass Performance")
        ax.set_xticks(x)
        ax.set_xticklabels(comparison_metrics)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)

        plt.tight_layout()

        # Save the plot
        plot_path = self.output_dir / f"{run_name}_evaluation_plots.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        return plot_path

    def generate_csv_summary(self, evaluation: Dict[str, Any], run_name: str) -> Path:
        """
        Generate a CSV summary of key metrics for easy comparison.
        """
        # Flatten metrics into a single row
        summary_data = {
            "run_name": run_name,
            "timestamp": datetime.now().isoformat(),
            "total_samples": evaluation["dataset_info"]["total_samples"],
        }

        # Add label distribution
        for label, count in evaluation["dataset_info"][
            "label_distribution_true"
        ].items():
            summary_data[f"true_{label.lower()}_count"] = count
            summary_data[f"true_{label.lower()}_percent"] = (
                count / evaluation["dataset_info"]["total_samples"]
            ) * 100

        for label, count in evaluation["dataset_info"][
            "label_distribution_pred"
        ].items():
            summary_data[f"pred_{label.lower()}_count"] = count
            summary_data[f"pred_{label.lower()}_percent"] = (
                count / evaluation["dataset_info"]["total_samples"]
            ) * 100

        # Add all metrics
        summary_data.update(evaluation["multiclass_metrics"])
        summary_data.update(evaluation["binary_metrics"])
        summary_data.update(evaluation["auc_metrics"])

        # Add confidence stats
        summary_data["mean_confidence"] = evaluation["analysis"]["mean_confidence"]
        summary_data["std_confidence"] = evaluation["analysis"]["std_confidence"]

        # Create DataFrame and save
        df = pd.DataFrame([summary_data])
        csv_path = self.output_dir / f"{run_name}_summary.csv"
        df.to_csv(csv_path, index=False)

        return csv_path

    def generate_full_summary(
        self, evaluation: Dict[str, Any], run_name: str
    ) -> Dict[str, Path]:
        """
        Generate all summary formats: text, visualizations, and CSV.

        Returns:
            Dictionary mapping format names to file paths
        """
        output_files = {}

        # Generate text summary
        text_summary = self.generate_text_summary(evaluation, run_name)
        text_path = self.output_dir / f"{run_name}_summary.md"
        with open(text_path, "w") as f:
            f.write(text_summary)
        output_files["text_summary"] = text_path

        # Generate visualizations
        try:
            plot_path = self.generate_visualizations(evaluation, run_name)
            output_files["visualizations"] = plot_path
        except Exception as e:
            print(f"Warning: Could not generate visualizations: {e}")

        # Generate CSV summary
        csv_path = self.generate_csv_summary(evaluation, run_name)
        output_files["csv_summary"] = csv_path

        # Save full evaluation as JSON
        json_path = self.output_dir / f"{run_name}_full_evaluation.json"
        with open(json_path, "w") as f:
            json.dump(evaluation, f, indent=2)
        output_files["full_evaluation"] = json_path

        return output_files
