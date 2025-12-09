# RAG (Retrieval-Augmented Generation) Setup Guide

This guide explains how to set up and use RAG with Weaviate for fact-checking experiments.

## Overview

The RAG strategy retrieves similar claims from a vector database (Weaviate) to inform fact-checking decisions. Each test iteration has its own knowledge base created from the training data.

## Architecture

```
Training Data → Weaviate (per iteration) → RAG Chain → Fact-Check Result
                   ↓
            Vector Embeddings
           (OpenAI text-embedding-3-small)
```

**Components:**

-   **Weaviate**: Vector database running in Docker
-   **Schema**: 5 iteration-specific classes (`FactCheckKB_Iter0` - `FactCheckKB_Iter4`)
-   **Vectorization**: OpenAI embeddings (text-embedding-3-small)
-   **Retrieval**: Top-K similar claims with certainty threshold
-   **Chain**: RAG-augmented fact-checking prompt

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

This installs `weaviate-client>=3.25.0` and other required packages.

### 2. Start Weaviate with Docker

```bash
# Start Weaviate
docker-compose -f docker-compose.weaviate.yml up -d

# Check status
docker-compose -f docker-compose.weaviate.yml ps

# View logs
docker-compose -f docker-compose.weaviate.yml logs -f
```

Weaviate will be available at `http://localhost:8080`.

**Environment Requirements:**

-   Docker installed and running
-   `OPENAI_API_KEY` set in `.env` (for vectorization)

### 3. Deploy Schemas

Create the database schema for all 5 test iterations:

```bash
python -m src.weaviate.deploy
```

**Options:**

```bash
# Reset existing schemas
python -m src.weaviate.deploy --reset

# Use different URL
python -m src.weaviate.deploy --url http://your-weaviate:8080

# Wait longer for startup
python -m src.weaviate.deploy --wait 60
```

**Output:**

```
✓ Weaviate is ready
Creating schema: FactCheckKB_Iter0
✓ Created: FactCheckKB_Iter0
...
✓ Found 5 FactCheckKB schemas
```

### 4. Vectorize Training Data

Load training data and create embeddings:

```bash
python -m src.weaviate.vectorize
```

**Options:**

```bash
# Vectorize specific iteration
python -m src.weaviate.vectorize --iteration 0

# Limit samples for testing
python -m src.weaviate.vectorize --max-samples 100

# Adjust batch size
python -m src.weaviate.vectorize --batch-size 200
```

**What happens:**

-   Loads training data from `data/splits/`
-   Creates embeddings via OpenAI API
-   Stores in Weaviate with metadata (claim, label, ID)
-   Progress bar shows vectorization status

**Cost estimate:** ~$0.01 per 1000 samples with text-embedding-3-small

## Running RAG Experiments

### Test Single Sample

```bash
# Test RAG on one claim
python test_single_sample.py --strategies rag --sample-idx 0

# Custom claim with RAG
python test_single_sample.py --strategies rag --claim "Water boils at 100°C"

# Compare RAG with other strategies
python test_single_sample.py --strategies rag,zero_shot,cot --sample-idx 10
```

### Run Full Experiments

```bash
# Small test batch (iteration 0 only, 50 samples)
python run_experiments.py --strategies rag --iterations 0 --max-samples 50

# Full RAG experiment (all iterations)
python run_experiments.py --strategies rag

# Compare RAG with all strategies
python run_experiments.py --strategies all
```

**Output location:** `results/experiments/rag/iteration_X.jsonl`

## Schema Structure

Each iteration class has this schema:

```python
{
  "class": "FactCheckKB_Iter0",
  "vectorizer": "text2vec-openai",
  "properties": [
    {"name": "claim_id", "dataType": ["text"]},
    {"name": "claim", "dataType": ["text"]},  # Vectorized
    {"name": "label", "dataType": ["text"]},
    {"name": "verdict", "dataType": ["text"]},
    {"name": "source", "dataType": ["text"]}
  ]
}
```

## RAG Configuration

**Retrieval parameters** (in `run_experiments.py` and `test_single_sample.py`):

```python
retriever = FactCheckRetriever(
    client=weaviate_client,
    iteration_id=0  # Which iteration's KB to use
)

chain = build_fact_check_rag_chain(
    llm=llm,
    prompt_path="src/prompts/fact_check_rag.txt",
    retriever=retriever,
    top_k=5,           # Number of similar claims to retrieve
    certainty=0.7      # Minimum similarity threshold (0-1)
)
```

**Tuning:**

-   `top_k`: Higher = more context, but noisier
-   `certainty`: Higher = more relevant, but fewer results

## Prompt Template

The RAG prompt (`src/prompts/fact_check_rag.txt`) provides:

1. The claim to verify
2. Similar claims from the knowledge base
3. Instructions to use them as reference points
4. Expected JSON output format

**Output fields:**

-   `verdict`: FACT or FALSE
-   `confidence`: 0.0 to 1.0
-   `rationale`: Explanation referencing similar claims
-   `retrieved_claims_used`: Count of helpful similar claims
-   `cited_knowledge`: List of facts/patterns from retrieved examples

## Troubleshooting

### Weaviate not ready

```bash
# Check if Docker is running
docker ps

# Restart Weaviate
docker-compose -f docker-compose.weaviate.yml restart

# Check logs for errors
docker-compose -f docker-compose.weaviate.yml logs
```

### No similar claims retrieved

**Possible causes:**

-   Knowledge base not vectorized (run `python -m src.weaviate.vectorize`)
-   `certainty` threshold too high (try 0.6 or 0.5)
-   Wrong iteration specified

**Debug:**

```python
# Check object count
from src.weaviate.client import get_weaviate_client

client = get_weaviate_client()
result = client.query.aggregate("FactCheckKB_Iter0").with_meta_count().do()
print(result)  # Should show count > 0
```

### Import errors

```bash
# Make sure Weaviate client is installed
pip install weaviate-client>=3.25.0

# Verify installation
python -c "import weaviate; print(weaviate.__version__)"
```

### OpenAI API errors during vectorization

**Rate limits:** Weaviate calls OpenAI for embeddings. If you hit rate limits:

```bash
# Reduce batch size
python -m src.weaviate.vectorize --batch-size 50

# Vectorize one iteration at a time
python -m src.weaviate.vectorize --iteration 0
```

## Data Management

### Clear all data

```bash
# Stop Weaviate (preserves volume)
docker-compose -f docker-compose.weaviate.yml down

# Stop and delete volume (deletes all data)
docker-compose -f docker-compose.weaviate.yml down -v

# Restart and re-deploy
docker-compose -f docker-compose.weaviate.yml up -d
python -m src.weaviate.deploy
python -m src.weaviate.vectorize
```

### Re-vectorize specific iteration

```bash
# Schemas must exist first
python -m src.weaviate.vectorize --iteration 2
```

The script will prompt to delete existing data before re-vectorizing.

## Cost Considerations

**Vectorization costs (one-time per setup):**

-   ~$0.01 per 1000 samples (text-embedding-3-small)
-   Full training set (~4,200 samples) × 5 iterations = ~$0.21

**Inference costs (per RAG experiment):**

-   Same LLM costs as other strategies
-   Additional embedding cost for query vectorization: ~$0.00001 per claim
-   Full test set (4,231 samples × 5 iterations) adds ~$0.21 in embedding costs

**Total:** ~$0.42 for full RAG setup + experiments

## Performance Tips

1. **Batch vectorization:** Default 100 works well, increase for faster ingestion
2. **Parallel queries:** Weaviate handles concurrent reads well
3. **Certainty tuning:** Start at 0.7, lower if too few results
4. **Top-K:** 5 is good balance, increase for more context

## Files Overview

```
src/weaviate/
├── __init__.py           # Module init
├── client.py             # Weaviate client utilities
├── schema.py             # Schema definitions
├── deploy.py             # Deploy schemas script
├── vectorize.py          # Vectorization script
└── retriever.py          # RAG retrieval class

src/chains/
└── fact_check_rag.py     # RAG chain builder

src/prompts/
└── fact_check_rag.txt    # RAG prompt template

docker-compose.weaviate.yml  # Weaviate Docker config
```

## Next Steps

After setup:

1. ✅ Test on single sample: `python test_single_sample.py --strategies rag --sample-idx 0`
2. ✅ Run small batch: `python run_experiments.py --strategies rag --iterations 0 --max-samples 50`
3. ✅ Full experiment: `python run_experiments.py --strategies rag`
4. ✅ Evaluate: `python evaluate_experiments.py --results-dir results/experiments`
5. ✅ Compare with other strategies in evaluation report

## References

-   [Weaviate Docker Setup](https://weaviate.io/blog/docker-and-containers-with-weaviate)
-   [Weaviate Python Client](https://weaviate.io/developers/weaviate/client-libraries/python)
-   [OpenAI Embeddings](https://platform.openai.com/docs/guides/embeddings)
