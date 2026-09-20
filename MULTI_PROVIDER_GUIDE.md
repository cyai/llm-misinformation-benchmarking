# Multi-Provider LLM Support Guide

## Overview

The fact-checking framework now supports **10+ LLM providers** including commercial APIs and open-source models:

### Commercial APIs:

1. **OpenAI** - GPT-4, GPT-5, GPT-4o-mini
2. **Anthropic** - Claude 3 (Opus, Sonnet, Haiku)
3. **Google** - Gemini Pro, Gemini 1.5
4. **xAI** - Grok 2, Grok 4
5. **DeepSeek** - DeepSeek-v2

### Open Source Models (via Ollama - local):

6. **Llama** - Llama 3.1, Llama 3.2
7. **Mistral** - Mistral 7B
8. **Phi** - Phi3, Phi3.5, Phi4
9. **Gemma** - Gemma2, Gemma3
10. **Qwen** - Qwen 2.5
11. **Vicuna** - Vicuna 7B

### Via HuggingFace:

-   Any model available on HuggingFace Hub

---

## Quick Start

### 1. Install Dependencies

```bash
# Install all provider packages
pip install -r requirements.txt

# Or install individually:
pip install langchain-anthropic        # Claude
pip install langchain-google-genai     # Gemini
pip install langchain-huggingface      # HuggingFace models
pip install langchain-community        # Ollama support
```

### 2. Setup API Keys

Add to your `.env` file:

```bash
# OpenAI (GPT-4, GPT-5)
OPENAI_API_KEY=sk-...

# Anthropic (Claude)
ANTHROPIC_API_KEY=sk-ant-...

# Google (Gemini)
GOOGLE_API_KEY=...

# xAI (Grok)
XAI_API_KEY=xai-...

# DeepSeek
DEEPSEEK_API_KEY=sk-...

# HuggingFace
HUGGINGFACE_API_KEY=hf_...
```

### 3. For Local Models (Ollama)

```bash
# Install Ollama
curl https://ollama.ai/install.sh | sh

# Pull models
ollama pull llama3.2
ollama pull mistral
ollama pull phi3
ollama pull gemma2
ollama pull deepseek-coder-v2
ollama pull qwen2.5
ollama pull vicuna
```

---

## Usage Examples

### Running Experiments

**OpenAI (GPT-4):**

```bash
python run_experiments.py --provider openai --model gpt-4
```

**Claude 3 Opus:**

```bash
python run_experiments.py --provider anthropic --model claude-3-opus-20240229
```

**Google Gemini Pro:**

```bash
python run_experiments.py --provider google --model gemini-pro
```

**Grok 4:**

```bash
python run_experiments.py --provider xai --model grok-4
```

**Llama 3.2 (local via Ollama):**

```bash
python run_experiments.py --provider ollama --model llama3.2
```

**Mistral (local):**

```bash
python run_experiments.py --provider ollama --model mistral
```

**Phi3 (local):**

```bash
python run_experiments.py --provider ollama --model phi3
```

**Gemma3 (local):**

```bash
python run_experiments.py --provider ollama --model gemma3
```

**DeepSeek-v2 (local):**

```bash
python run_experiments.py --provider ollama --model deepseek-coder-v2
```

**Qwen 2.5 (local):**

```bash
python run_experiments.py --provider ollama --model qwen2.5
```

**Vicuna (local):**

```bash
python run_experiments.py --provider ollama --model vicuna
```

**HuggingFace Model:**

```bash
python run_experiments.py --provider huggingface --model meta-llama/Llama-2-7b-chat-hf
```

### Testing Single Samples

```bash
# Test with Claude
python test_single_sample.py --claim "The Earth is flat" --model claude-3-opus --provider anthropic

# Test with Gemini
python test_single_sample.py --claim "Water boils at 100°C" --model gemini-pro --provider google

# Test with local Llama
python test_single_sample.py --sample-idx 0 --model llama3.2 --provider ollama
```

---

## Model Presets

Use preset names for convenience:

```python
from src.models.llm import get_model_preset

# Commercial APIs
llm = get_model_preset("gpt-4")
llm = get_model_preset("claude-3-opus")
llm = get_model_preset("gemini-pro")
llm = get_model_preset("grok-4")

# Local models
llm = get_model_preset("llama3.2")
llm = get_model_preset("mistral")
llm = get_model_preset("phi3")
llm = get_model_preset("gemma3")
llm = get_model_preset("qwen2.5")
llm = get_model_preset("vicuna")
```

---

## Provider-Specific Notes

### OpenAI

-   **Models**: gpt-4, gpt-4-turbo, gpt-5, gpt-5-mini
-   **API**: https://platform.openai.com/api-keys
-   **Cost**: Pay per token

### Anthropic (Claude)

-   **Models**: claude-3-opus, claude-3-sonnet, claude-3-haiku
-   **API**: https://console.anthropic.com/
-   **Cost**: Pay per token
-   **Note**: Claude excels at detailed reasoning

### Google (Gemini)

-   **Models**: gemini-pro, gemini-1.5-pro, gemini-1.5-flash
-   **API**: https://makersuite.google.com/app/apikey
-   **Cost**: Free tier available, then pay per token
-   **Note**: Good for multimodal tasks

### xAI (Grok)

-   **Models**: grok-2, grok-4
-   **API**: https://console.x.ai/
-   **Cost**: Pay per token
-   **Note**: Uses OpenAI-compatible API

### DeepSeek

-   **Models**: deepseek-chat, deepseek-coder
-   **API**: https://platform.deepseek.com/
-   **Cost**: Competitive pricing
-   **Note**: Strong at coding tasks

### Ollama (Local Models)

-   **Installation**: https://ollama.ai/
-   **Cost**: FREE (runs locally)
-   **Requirements**:
    -   8GB+ RAM for 7B models
    -   16GB+ RAM for 13B models
    -   GPU recommended for speed
-   **Models**: llama3.2, mistral, phi3, phi4, gemma2, gemma3, deepseek-coder-v2, qwen2.5, vicuna

### HuggingFace

-   **Models**: 100,000+ models available
-   **API**: https://huggingface.co/settings/tokens
-   **Cost**: Free tier available, paid inference endpoints
-   **Note**: Access to cutting-edge research models

---

## Performance Comparison

### Speed (approximate):

-   **GPT-4**: 20-30 tokens/sec
-   **Claude-3**: 30-40 tokens/sec
-   **Gemini**: 40-50 tokens/sec
-   **Local 7B models**: 10-30 tokens/sec (GPU dependent)
-   **Local 13B models**: 5-15 tokens/sec (GPU dependent)

### Quality (subjective):

-   **Best reasoning**: GPT-4, Claude-3-Opus
-   **Best value**: GPT-4o-mini, Claude-3-Haiku, Gemini-1.5-Flash
-   **Best local**: Llama-3.2, Mistral, Qwen-2.5

---

## Cost Estimates (per 1M tokens)

### Commercial:

-   **GPT-4**: ~$30-60
-   **GPT-4o-mini**: ~$0.15-0.60
-   **Claude-3-Opus**: ~$15-75
-   **Claude-3-Sonnet**: ~$3-15
-   **Claude-3-Haiku**: ~$0.25-1.25
-   **Gemini-Pro**: ~$0.50-1.50
-   **Grok**: TBD
-   **DeepSeek**: ~$0.14-0.28

### Local (Ollama):

-   **All models**: $0 (electricity only)
-   **Full test set (4,231 samples)**: FREE

---

## Troubleshooting

### Ollama Connection Error

```bash
# Start Ollama service
ollama serve

# Or check if running
curl http://localhost:11434/api/tags
```

### Model Not Found (Ollama)

```bash
# List available models
ollama list

# Pull missing model
ollama pull llama3.2
```

### HuggingFace Rate Limit

-   Sign up for pro account: https://huggingface.co/pricing
-   Or use dedicated inference endpoints

### API Key Not Working

-   Check `.env` file is in project root
-   Ensure no spaces around `=` in `.env`
-   Restart terminal/IDE after adding keys

---

## Advanced Usage

### Custom Temperature

```bash
# More creative (higher temperature)
python run_experiments.py --provider anthropic --model claude-3-opus --temperature 0.7

# More deterministic (lower temperature)
python run_experiments.py --provider openai --model gpt-4 --temperature 0.1
```

### Programmatic Usage

```python
from src.models.llm import make_chat_model

# Create any model
llm = make_chat_model(
    provider="anthropic",
    model_name="claude-3-opus-20240229",
    api_key="sk-ant-...",
    temperature=0.0
)

# Use in chain
from src.chains.fact_check import build_fact_check_chain
chain = build_fact_check_chain(llm, prompt_path="src/prompts/fact_check.txt")
result = chain.invoke({"claim": "The Earth is flat"})
```

---

## Recommended Setup

### For Development/Testing:

-   Use **Ollama** with Llama 3.2 or Mistral (free, fast enough)

### For Production/Research:

-   Use **GPT-4o-mini** (best value)
-   Or **Claude-3-Haiku** (fast and cheap)
-   Or **Gemini-1.5-Flash** (free tier)

### For Best Quality:

-   **GPT-4** or **Claude-3-Opus**
-   Worth the cost for critical applications

---

## Summary

✅ **10+ providers supported**
✅ **Free local models via Ollama**  
✅ **Simple API key setup**
✅ **Same interface for all providers**
✅ **Easy model switching**

Get started:

```bash
# Quick test with local model (FREE)
ollama pull llama3.2
python test_single_sample.py --model llama3.2 --provider ollama

# Or use GPT-4o-mini (cheap)
python test_single_sample.py --model gpt-4o-mini --provider openai
```
