import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize
import pandas as pd


class FactCheckEvaluator:
    """Comprehensive evaluator for fact-checking results."""

    def __init__(self, label_order: List[str] = None):
        """
        Initialize evaluator.

        Args:
            label_order: Order of labels for consistent reporting.
                        Defaults to ['FALSE', 'MIXED', 'FACT']
        """
        self.label_order = label_order or ["FALSE", "MIXED", "FACT"]

    def load_results(
        self, results_file: Path
    ) -> Tuple[List[str], List[str], List[float]]:
        """
        Load results from JSONL file.

        Returns:
            Tuple of (gold_labels, predicted_labels, confidence_scores)
        """
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

    def calculate_binary_metrics(
        self, y_true: List[str], y_pred: List[str]
    ) -> Dict[str, float]:
        """
        Calculate binary classification metrics (FACT vs non-FACT).
        """
        # Convert to binary: FACT vs non-FACT
        y_true_binary = [1 if label == "FACT" else 0 for label in y_true]
        y_pred_binary = [1 if label == "FACT" else 0 for label in y_pred]

        return {
            "binary_accuracy": accuracy_score(y_true_binary, y_pred_binary),
            "binary_precision_fact": precision_score(
                y_true_binary, y_pred_binary, pos_label=1
            ),
            "binary_recall_fact": recall_score(
                y_true_binary, y_pred_binary, pos_label=1
            ),
            "binary_f1_fact": f1_score(y_true_binary, y_pred_binary, pos_label=1),
            "binary_precision_nonfact": precision_score(
                y_true_binary, y_pred_binary, pos_label=0
            ),
            "binary_recall_nonfact": recall_score(
                y_true_binary, y_pred_binary, pos_label=0
            ),
            "binary_f1_nonfact": f1_score(y_true_binary, y_pred_binary, pos_label=0),
        }

    def calculate_multiclass_metrics(
        self, y_true: List[str], y_pred: List[str]
    ) -> Dict[str, float]:
        """
        Calculate multiclass classification metrics.
        """
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "f1_micro": f1_score(y_true, y_pred, average="micro"),
            "f1_macro": f1_score(y_true, y_pred, average="macro"),
            "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
            "precision_micro": precision_score(y_true, y_pred, average="micro"),
            "precision_macro": precision_score(y_true, y_pred, average="macro"),
            "precision_weighted": precision_score(y_true, y_pred, average="weighted"),
            "recall_micro": recall_score(y_true, y_pred, average="micro"),
            "recall_macro": recall_score(y_true, y_pred, average="macro"),
            "recall_weighted": recall_score(y_true, y_pred, average="weighted"),
        }

        # Per-class metrics
        for label in self.label_order:
            if label in y_true and label in y_pred:
                metrics[f"precision_{label.lower()}"] = (
                    precision_score(y_true, y_pred, labels=[label], average=None)[0]
                    if label in y_pred
                    else 0.0
                )
                metrics[f"recall_{label.lower()}"] = (
                    recall_score(y_true, y_pred, labels=[label], average=None)[0]
                    if label in y_true
                    else 0.0
                )
                metrics[f"f1_{label.lower()}"] = (
                    f1_score(y_true, y_pred, labels=[label], average=None)[0]
                    if label in y_true and label in y_pred
                    else 0.0
                )

        return metrics

    def calculate_auc_metrics(
        self, y_true: List[str], y_pred: List[str], confidence_scores: List[float]
    ) -> Dict[str, float]:
        """
        Calculate AUC metrics using confidence scores.
        """
        metrics = {}

        # Binary AUC (FACT vs non-FACT)
        try:
            y_true_binary = [1 if label == "FACT" else 0 for label in y_true]
            fact_confidences = []

            for pred, conf in zip(y_pred, confidence_scores):
                if pred == "FACT":
                    fact_confidences.append(conf)
                else:
                    fact_confidences.append(1 - conf)

            metrics["auc_binary"] = roc_auc_score(y_true_binary, fact_confidences)
        except Exception as e:
            metrics["auc_binary"] = 0.0
            print(f"Warning: Could not calculate binary AUC: {e}")

        # Multiclass AUC (one-vs-rest)
        try:
            # Convert to numeric labels
            label_to_num = {label: i for i, label in enumerate(self.label_order)}
            y_true_numeric = [label_to_num[label] for label in y_true]

            # Create probability matrix (simplified using confidence scores)
            n_classes = len(self.label_order)
            y_prob = np.zeros((len(y_pred), n_classes))

            for i, (pred, conf) in enumerate(zip(y_pred, confidence_scores)):
                pred_idx = label_to_num[pred]
                # Distribute confidence: high for predicted class, low for others
                y_prob[i, pred_idx] = conf
                remaining_prob = (1 - conf) / (n_classes - 1)
                for j in range(n_classes):
                    if j != pred_idx:
                        y_prob[i, j] = remaining_prob

            # Binarize labels for multiclass AUC
            y_true_binarized = label_binarize(
                y_true_numeric, classes=list(range(n_classes))
            )

            if y_true_binarized.shape[1] > 1:
                metrics["auc_multiclass_ovr"] = roc_auc_score(
                    y_true_binarized, y_prob, multi_class="ovr", average="macro"
                )
            else:
                metrics["auc_multiclass_ovr"] = metrics.get("auc_binary", 0.0)

        except Exception as e:
            metrics["auc_multiclass_ovr"] = 0.0
            print(f"Warning: Could not calculate multiclass AUC: {e}")

        return metrics

    def get_confusion_matrix(self, y_true: List[str], y_pred: List[str]) -> np.ndarray:
        """
        Get confusion matrix with consistent label ordering.
        """
        return confusion_matrix(y_true, y_pred, labels=self.label_order)

    def get_classification_report(self, y_true: List[str], y_pred: List[str]) -> str:
        """
        Get detailed classification report.
        """
        return classification_report(
            y_true, y_pred, labels=self.label_order, target_names=self.label_order
        )

    def evaluate(self, results_file: Path) -> Dict[str, Any]:
        """
        Comprehensive evaluation of fact-checking results.

        Returns:
            Dictionary containing all metrics and analysis
        """
        # Load data
        y_true, y_pred, confidence_scores = self.load_results(results_file)

        # Calculate all metrics
        evaluation = {
            "dataset_info": {
                "total_samples": len(y_true),
                "label_distribution_true": {
                    label: y_true.count(label) for label in self.label_order
                },
                "label_distribution_pred": {
                    label: y_pred.count(label) for label in self.label_order
                },
            }
        }

        # Multiclass metrics
        evaluation["multiclass_metrics"] = self.calculate_multiclass_metrics(
            y_true, y_pred
        )

        # Binary metrics
        evaluation["binary_metrics"] = self.calculate_binary_metrics(y_true, y_pred)

        # AUC metrics
        evaluation["auc_metrics"] = self.calculate_auc_metrics(
            y_true, y_pred, confidence_scores
        )

        # Confusion matrix
        evaluation["confusion_matrix"] = self.get_confusion_matrix(
            y_true, y_pred
        ).tolist()

        # Classification report
        evaluation["classification_report"] = self.get_classification_report(
            y_true, y_pred
        )

        # Additional analysis
        evaluation["analysis"] = {
            "mean_confidence": np.mean(confidence_scores),
            "std_confidence": np.std(confidence_scores),
            "confidence_by_label_true": {},
            "confidence_by_label_pred": {},
        }

        # Confidence analysis by true and predicted labels
        for label in self.label_order:
            true_indices = [i for i, l in enumerate(y_true) if l == label]
            pred_indices = [i for i, l in enumerate(y_pred) if l == label]

            if true_indices:
                evaluation["analysis"]["confidence_by_label_true"][label] = {
                    "mean": np.mean([confidence_scores[i] for i in true_indices]),
                    "std": np.std([confidence_scores[i] for i in true_indices]),
                }

            if pred_indices:
                evaluation["analysis"]["confidence_by_label_pred"][label] = {
                    "mean": np.mean([confidence_scores[i] for i in pred_indices]),
                    "std": np.std([confidence_scores[i] for i in pred_indices]),
                }

        return evaluation
