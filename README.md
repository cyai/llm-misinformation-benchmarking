# LLM Misinfo Benchmarkking

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill OPENAI_API_KEY & OPENAI_MODEL
```

## Downloading the Politifact Dataset
```bash
curl -L -o politifact-fact-check-dataset.zip https://www.kaggle.com/api/v1/datasets/download/rmisra/politifact-fact-check-dataset
```

```bash
unzip politifact-fact-check-dataset.zip -d ./data/politifact
```

---

## Running `run_factcheck.py`

The main entry point for this project is:

```bash
python -m src.run_factcheck [OPTIONS]
```

This script loads the Politifact dataset, runs your chosen LLM (via LangChain), and writes JSONL results for later metric computation.

### Arguments

| Argument        | Type   | Default                            | Description                                                                                                                         |
| --------------- | ------ | ---------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| `--provider`    | `str`  | `"openai"`                         | Which model provider to use. Currently supports `"openai"`. Easily extendable for `"anthropic"`, `"azure_openai"`, `"ollama"`, etc. |
| `--model`       | `str`  | value from `.env` (`OPENAI_MODEL`) | The specific model name to use, e.g. `gpt-4o`, `gpt-4o-mini`, `gpt-4-turbo`, etc.                                                   |
| `--max_records` | `int`  | `100`                              | Limits how many rows from the dataset to process (useful for testing). Set `0` or omit to process all rows.                         |
| `--split`       | `str`  | `"train"`                          | Which dataset split to load. Options: `train`, `valid`, `test`, or `all`.                                                           |
| `--prompt`      | `path` | `src/prompts/fact_check.txt`       | Path to the prompt file used by the model. You can easily swap this to test different prompting strategies (few-shot, CoT, etc.).   |
| `--results`     | `path` | `./results/openai_zero_shot.jsonl` | Output file path for the JSONL results. Each line is a single `FactCheckRecord`.                                                    |
