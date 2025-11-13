# LLM Misinfo Benchmarking

A comprehensive benchmarking framework for evaluating Large Language Models on fact-checking tasks using the Politifact dataset.

## Features

-   **Binary Classification**: FACT vs FALSE detection
-   **Multiple Prompting Strategies**: Zero-shot, One-shot, Few-shot
-   **Robust Evaluation**: 5 test iterations with different random seeds
-   **Comprehensive Metrics**: Accuracy, F1-scores, Precision, Recall, Confidence analysis
-   **Reproducibility**: All configurations saved for future comparison

---

## Setup

```bash
# Create virtual environment and install dependencies
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Configure API keys
cp .env.example .env   # fill OPENAI_API_KEY & OPENAI_MODEL
```

## Downloading the Politifact Dataset

```bash
curl -L -o politifact-fact-check-dataset.zip https://www.kaggle.com/api/v1/datasets/download/rmisra/politifact-fact-check-dataset
unzip politifact-fact-check-dataset.zip -d ./data/politifact
```

---

## Complete Workflow

### 1. Prepare Dataset Splits (60-20-20)

Create train-validation-test splits with 5 test iterations for robustness:

```bash
python prepare_dataset.py
```

**Options:**

-   `--seed`: Random seed (default: 42)
-   `--n-iterations`: Number of test iterations (default: 5)
-   `--train-ratio`: Training ratio (default: 0.6)
-   `--val-ratio`: Validation ratio (default: 0.2)
-   `--test-ratio`: Test ratio (default: 0.2)

**Output:**

-   `data/splits/train.jsonl` - Training data
-   `data/splits/val.jsonl` - Validation data
-   `data/splits/test.jsonl` - Test data
-   `data/splits/split_config.json` - Split configuration
-   `data/splits/test_iterations.json` - Test iteration seeds (for reproducibility)

### 2. Run Experiments (Zero-shot, One-shot, Few-shot)

Run fact-checking experiments on test data across all iterations:

**Recommended: Background execution** (for long-running experiments):

```bash
# Quick test in background
nohup python run_experiments.py > logs/experiment.log 2>&1 &
```

**Example - Run specific strategies:**

```bash
python run_experiments.py --strategies zero_shot,few_shot --iterations 0,1,2
```

**Example - Quick test:**

```bash
python run_experiments.py --max-samples 50 --iterations 0
```

**Output:**

-   `results/experiments/zero_shot/iteration_*.jsonl` - Zero-shot results
-   `results/experiments/one_shot/iteration_*.jsonl` - One-shot results
-   `results/experiments/few_shot/iteration_*.jsonl` - Few-shot results
-   `results/experiments/experiment_metadata.json` - Experiment configuration
-   `results/experiments/results_summary.json` - Results summary

### 3. Evaluate Experiments

Evaluate all experiments and generate comparison reports:

```bash
python evaluate_experiments.py
```

**Options:**

-   `--results-dir`: Experiments directory (default: results/experiments)
-   `--output-dir`: Evaluation output directory (default: results/evaluations)

**Output:**

-   `results/evaluations/detailed_evaluation.json` - Detailed metrics per iteration
-   `results/evaluations/comparison_summary.csv` - CSV comparison table
-   `results/evaluations/evaluation_report.md` - Comprehensive markdown report with:
    -   Aggregated metrics (mean ± std) across iterations
    -   Per-iteration breakdown
    -   Strategy comparison
    -   Best performing strategy analysis
    -   Recommendations

---

## Legacy: Single Run Mode

For quick single runs without the full experiment pipeline:

```bash
python -m src.run_factcheck [OPTIONS]
```

### Arguments

| Argument        | Type   | Default                            | Description                                                                                                                         |
| --------------- | ------ | ---------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| `--provider`    | `str`  | `"openai"`                         | Which model provider to use. Currently supports `"openai"`. Easily extendable for `"anthropic"`, `"azure_openai"`, `"ollama"`, etc. |
| `--model`       | `str`  | value from `.env` (`OPENAI_MODEL`) | The specific model name to use, e.g. `gpt-4o`, `gpt-4o-mini`, `gpt-4-turbo`, etc.                                                   |
| `--max_records` | `int`  | `100`                              | Limits how many rows from the dataset to process (useful for testing). Set `0` or omit to process all rows.                         |
| `--split`       | `str`  | `"train"`                          | Which dataset split to load. Options: `train`, `valid`, `test`, or `all`.                                                           |
| `--prompt`      | `path` | `src/prompts/fact_check.txt`       | Path to the prompt file used by the model. You can easily swap this to test different prompting strategies (few-shot, CoT, etc.).   |
| `--results`     | `path` | `./results/openai_zero_shot.jsonl` | Output file path for the JSONL results. Each line is a single `FactCheckRecord`.                                                    |

---

## Evaluation Metrics

The evaluation framework provides:

### Binary Classification Metrics (FACT vs FALSE)

-   **Accuracy**: Overall correctness
-   **Precision**: True positives / (True positives + False positives)
-   **Recall**: True positives / (True positives + False negatives)
-   **F1-Score**: Harmonic mean of precision and recall
-   **Specificity**: True negatives / (True negatives + False positives)

### Aggregate Statistics (Across 5 Iterations)

-   **Mean ± Std**: Average performance with standard deviation
-   **Min/Max**: Range of performance
-   **Consistency**: Low std indicates robust performance

### Per-Iteration Analysis

-   Individual metrics for each test partition
-   Confidence score analysis
-   Confusion matrices

---

## Project Structure

```
.
├── data/
│   ├── politifact/              # Raw dataset
│   └── splits/                  # Prepared train-val-test splits
├── results/
│   ├── experiments/             # Experiment results (per strategy/iteration)
│   └── evaluations/             # Evaluation reports and metrics
├── src/
│   ├── chains/                  # LangChain fact-checking logic
│   ├── data_loaders/            # Dataset loaders
│   ├── evaluation/              # Evaluation utilities
│   ├── models/                  # LLM wrappers
│   ├── prompts/                 # Prompt templates
│   │   ├── fact_check.txt       # Zero-shot prompt
│   │   ├── fact_check_oneshot.txt  # One-shot prompt
│   │   └── fact_check_fewshot.txt  # Few-shot prompt
│   └── utils/                   # Helper utilities
├── prepare_dataset.py           # Dataset splitting script
├── run_experiments.py           # Main experiment runner
├── evaluate_experiments.py      # Evaluation script
└── README.md
```

---

## Reproducibility

All experiment configurations are saved to ensure reproducibility:

1. **Dataset splits**: Saved with random seeds in `data/splits/`
2. **Test iterations**: Seeds for each iteration in `test_iterations.json`
3. **Experiment metadata**: Model, strategies, timestamps in `experiment_metadata.json`
4. **Results**: Complete JSONL records with all predictions and metadata

To reproduce experiments:

1. Use the same split configuration from `data/splits/`
2. Use the same iteration seeds from `test_iterations.json`
3. Run with the same model and prompts
4. Results will be identical (deterministic)
