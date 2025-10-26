#!/usr/bin/env python3
"""
Comprehensive evaluation script for fact-checking results.

Usage:
    python evaluate_results.py <results_file> [--run-name <name>] [--output-dir <dir>]

Example:
    python evaluate_results.py results/openai_zero_shot.jsonl --run-name openai_zero_shot
"""

import argparse
import sys
from pathlib import Path

from src.evaluation.evaluator import FactCheckEvaluator
from src.evaluation.summary_generator import EvaluationSummaryGenerator


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate fact-checking results and generate comprehensive summaries"
    )
    parser.add_argument(
        "results_file", type=Path, help="Path to the JSONL results file"
    )
    parser.add_argument(
        "--run-name",
        type=str,
        help="Name for this evaluation run (defaults to filename without extension)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/summaries"),
        help="Directory to save evaluation summaries (default: results/summaries)",
    )
    parser.add_argument(
        "--no-plots", action="store_true", help="Skip generating visualization plots"
    )
    parser.add_argument("--quiet", action="store_true", help="Reduce output verbosity")

    args = parser.parse_args()

    # Validate input file
    if not args.results_file.exists():
        print(f"Error: Results file not found: {args.results_file}")
        sys.exit(1)

    # Determine run name
    run_name = args.run_name or args.results_file.stem

    if not args.quiet:
        print(f"Evaluating results from: {args.results_file}")
        print(f"Run name: {run_name}")
        print(f"Output directory: {args.output_dir}")
        print("-" * 50)

    try:
        # Initialize evaluator
        evaluator = FactCheckEvaluator()

        if not args.quiet:
            print("Loading results and calculating metrics...")

        # Run comprehensive evaluation
        evaluation = evaluator.evaluate(args.results_file)

        if not args.quiet:
            print("✓ Evaluation complete")

            # Print quick summary to console
            print(f"\nQuick Summary for {run_name}:")
            print(f"  Total samples: {evaluation['dataset_info']['total_samples']}")
            print(f"  Accuracy: {evaluation['multiclass_metrics']['accuracy']:.4f}")
            print(f"  F1-Macro: {evaluation['multiclass_metrics']['f1_macro']:.4f}")
            print(f"  F1-Micro: {evaluation['multiclass_metrics']['f1_micro']:.4f}")
            print(
                f"  Binary F1 (FACT): {evaluation['binary_metrics']['binary_f1_fact']:.4f}"
            )

            if "auc_binary" in evaluation["auc_metrics"]:
                print(f"  Binary AUC: {evaluation['auc_metrics']['auc_binary']:.4f}")

        # Generate summaries
        if not args.quiet:
            print("\nGenerating summary reports...")

        summary_generator = EvaluationSummaryGenerator(args.output_dir)

        if args.no_plots:
            # Generate without plots
            output_files = {}

            # Text summary
            text_summary = summary_generator.generate_text_summary(evaluation, run_name)
            text_path = args.output_dir / f"{run_name}_summary.md"
            args.output_dir.mkdir(parents=True, exist_ok=True)
            with open(text_path, "w") as f:
                f.write(text_summary)
            output_files["text_summary"] = text_path

            # CSV summary
            csv_path = summary_generator.generate_csv_summary(evaluation, run_name)
            output_files["csv_summary"] = csv_path

            # JSON evaluation
            import json

            json_path = args.output_dir / f"{run_name}_full_evaluation.json"
            with open(json_path, "w") as f:
                json.dump(evaluation, f, indent=2)
            output_files["full_evaluation"] = json_path
        else:
            # Generate full summary with plots
            output_files = summary_generator.generate_full_summary(evaluation, run_name)

        if not args.quiet:
            print("✓ Summary generation complete")
            print("\nGenerated files:")
            for format_name, file_path in output_files.items():
                print(f"  {format_name}: {file_path}")

        # Print key findings
        if not args.quiet:
            print(f"\n" + "=" * 60)
            print("KEY FINDINGS")
            print("=" * 60)

            # Performance tier
            accuracy = evaluation["multiclass_metrics"]["accuracy"]
            f1_macro = evaluation["multiclass_metrics"]["f1_macro"]

            if accuracy > 0.8 and f1_macro > 0.8:
                performance_tier = "Excellent"
            elif accuracy > 0.7 and f1_macro > 0.7:
                performance_tier = "Good"
            elif accuracy > 0.6 and f1_macro > 0.6:
                performance_tier = "Fair"
            else:
                performance_tier = "Needs Improvement"

            print(f"Overall Performance: {performance_tier}")
            print(f"  Accuracy: {accuracy:.1%}")
            print(f"  F1-Macro: {f1_macro:.1%}")

            # Best and worst performing classes
            class_f1s = {
                "FALSE": evaluation["multiclass_metrics"]["f1_false"],
                "MIXED": evaluation["multiclass_metrics"]["f1_mixed"],
                "FACT": evaluation["multiclass_metrics"]["f1_fact"],
            }

            best_class = max(class_f1s, key=class_f1s.get)
            worst_class = min(class_f1s, key=class_f1s.get)

            print(
                f"\nBest performing class: {best_class} (F1: {class_f1s[best_class]:.3f})"
            )
            print(
                f"Worst performing class: {worst_class} (F1: {class_f1s[worst_class]:.3f})"
            )

            # Confidence analysis
            mean_conf = evaluation["analysis"]["mean_confidence"]
            print(f"\nMean confidence: {mean_conf:.3f}")
            if mean_conf > 0.8:
                conf_assessment = "High (model is confident)"
            elif mean_conf > 0.6:
                conf_assessment = "Moderate"
            else:
                conf_assessment = "Low (model is uncertain)"
            print(f"Confidence assessment: {conf_assessment}")

        print(f"\n✓ Evaluation complete! Check {args.output_dir} for detailed reports.")

    except Exception as e:
        print(f"Error during evaluation: {e}")
        if not args.quiet:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
