"""
LLM model factory for various providers.

Supported providers:
- OpenAI (GPT-4, GPT-5, etc.)
- Anthropic (Claude)
- Google (Gemini)
- Ollama (local models: Llama, Mistral, Phi, Gemma, DeepSeek, Qwen, Vicuna)
- HuggingFace (any model via HF Hub)
- xAI (Grok)
"""

import os
from typing import Optional
from langchain_core.language_models.chat_models import BaseChatModel


def make_chat_model(
    provider: str,
    model_name: str,
    api_key: Optional[str] = None,
    temperature: float = 0.0,
    **kwargs,
) -> BaseChatModel:
    """
    Create a chat model from various providers.

    Args:
        provider: Provider name (openai, anthropic, google, ollama, huggingface, xai)
        model_name: Model name/identifier
        api_key: API key (optional for local models like Ollama)
        temperature: Model temperature (0.0-1.0)
        **kwargs: Additional provider-specific arguments

    Returns:
        BaseChatModel instance

    Examples:
        # OpenAI
        llm = make_chat_model("openai", "gpt-4", api_key="sk-...")

        # Claude
        llm = make_chat_model("anthropic", "claude-3-opus-20240229", api_key="sk-ant-...")

        # Google Gemini
        llm = make_chat_model("google", "gemini-pro", api_key="...")

        # Ollama (local)
        llm = make_chat_model("ollama", "llama3.2")
        llm = make_chat_model("ollama", "mistral")

        # HuggingFace
        llm = make_chat_model("huggingface", "meta-llama/Llama-2-7b-chat-hf", api_key="hf_...")
    """
    provider = provider.lower().strip()

    # OpenAI (GPT-4, GPT-5, etc.)
    if provider in ("openai", "oai", "gpt"):
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model_name,
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            temperature=temperature,
            timeout=kwargs.get("timeout", 60),
            max_retries=kwargs.get("max_retries", 2),
        )

    # Anthropic (Claude)
    elif provider in ("anthropic", "claude"):
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError:
            raise ImportError(
                "langchain-anthropic not installed. "
                "Install with: pip install langchain-anthropic"
            )
        return ChatAnthropic(
            model=model_name,
            anthropic_api_key=api_key or os.getenv("ANTHROPIC_API_KEY"),
            temperature=temperature,
            timeout=kwargs.get("timeout", 60),
            max_retries=kwargs.get("max_retries", 2),
        )

    # Google (Gemini)
    elif provider in ("google", "gemini", "vertex"):
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError:
            raise ImportError(
                "langchain-google-genai not installed. "
                "Install with: pip install langchain-google-genai"
            )
        return ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key or os.getenv("GOOGLE_API_KEY"),
            temperature=temperature,
            timeout=kwargs.get("timeout", 60),
        )

    # xAI (Grok)
    elif provider in ("xai", "grok"):
        # xAI uses OpenAI-compatible API
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model_name,
            api_key=api_key or os.getenv("XAI_API_KEY"),
            base_url="https://api.x.ai/v1",
            temperature=temperature,
            timeout=kwargs.get("timeout", 60),
        )

    # Ollama (local models: Llama, Mistral, Phi, Gemma, DeepSeek, Qwen, Vicuna)
    elif provider in ("ollama", "local"):
        try:
            from langchain_community.chat_models import ChatOllama
        except ImportError:
            raise ImportError(
                "langchain-community not installed. "
                "Install with: pip install langchain-community"
            )
        return ChatOllama(
            model=model_name,
            base_url=kwargs.get("base_url", "http://localhost:11434"),
            temperature=temperature,
            timeout=kwargs.get("timeout", 120),
        )

    # HuggingFace Hub (any model)
    elif provider in ("huggingface", "hf", "hugging_face"):
        try:
            from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
        except ImportError:
            raise ImportError(
                "langchain-huggingface not installed. "
                "Install with: pip install langchain-huggingface"
            )

        # Create endpoint
        endpoint = HuggingFaceEndpoint(
            repo_id=model_name,
            huggingfacehub_api_token=api_key or os.getenv("HUGGINGFACE_API_KEY"),
            temperature=temperature,
            timeout=kwargs.get("timeout", 120),
        )

        # Wrap in ChatHuggingFace for chat interface
        return ChatHuggingFace(llm=endpoint)

    # DeepSeek (has its own API)
    elif provider in ("deepseek", "deepseek-ai"):
        # DeepSeek uses OpenAI-compatible API
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model_name,
            api_key=api_key or os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com/v1",
            temperature=temperature,
            timeout=kwargs.get("timeout", 60),
        )

    else:
        raise ValueError(
            f"Unsupported provider: {provider}\n"
            f"Supported providers: openai, anthropic, google, xai, ollama, huggingface, deepseek"
        )


# Model presets for easy access
MODEL_PRESETS = {
    # OpenAI
    "gpt-4": ("openai", "gpt-4"),
    "gpt-4-turbo": ("openai", "gpt-4-turbo-preview"),
    "gpt-5": ("openai", "gpt-5"),
    "gpt-5-mini": ("openai", "gpt-5-mini-2025-08-07"),
    # Claude
    "claude-3-opus": ("anthropic", "claude-3-opus-20240229"),
    "claude-3-sonnet": ("anthropic", "claude-3-sonnet-20240229"),
    "claude-3-haiku": ("anthropic", "claude-3-haiku-20240307"),
    # Gemini
    "gemini-pro": ("google", "gemini-pro"),
    "gemini-1.5-pro": ("google", "gemini-1.5-pro"),
    "gemini-1.5-flash": ("google", "gemini-1.5-flash"),
    # Grok
    "grok-2": ("xai", "grok-2-latest"),
    "grok-4": ("xai", "grok-4"),
    # Ollama models (local)
    "llama3.2": ("ollama", "llama3.2"),
    "llama3.1": ("ollama", "llama3.1"),
    "mistral": ("ollama", "mistral"),
    "mistral-7b": ("ollama", "mistral:7b"),
    "phi3": ("ollama", "phi3"),
    "phi3.5": ("ollama", "phi3.5"),
    "phi4": ("ollama", "phi4"),
    "gemma2": ("ollama", "gemma2"),
    "gemma3": ("ollama", "gemma3"),
    "deepseek-v2": ("ollama", "deepseek-coder-v2"),
    "qwen2.5": ("ollama", "qwen2.5"),
    "qwen2.5-7b": ("ollama", "qwen2.5:7b"),
    "vicuna": ("ollama", "vicuna"),
    "vicuna-7b": ("ollama", "vicuna:7b"),
    # HuggingFace models
    "llama2-7b": ("huggingface", "meta-llama/Llama-2-7b-chat-hf"),
    "llama2-13b": ("huggingface", "meta-llama/Llama-2-13b-chat-hf"),
    "mistral-hf": ("huggingface", "mistralai/Mistral-7B-Instruct-v0.2"),
    "my-hf-model": ("huggingface", "username/my-custom-model"),
}


def get_model_preset(
    preset_name: str, api_key: Optional[str] = None, **kwargs
) -> BaseChatModel:
    """
    Get a model using a preset name.

    Args:
        preset_name: Preset name (e.g., "gpt-4", "claude-3-opus", "llama3.2")
        api_key: Optional API key
        **kwargs: Additional arguments

    Returns:
        BaseChatModel instance

    Example:
        llm = get_model_preset("claude-3-opus", api_key="sk-ant-...")
    """
    if preset_name not in MODEL_PRESETS:
        raise ValueError(
            f"Unknown preset: {preset_name}\n"
            f"Available presets: {', '.join(MODEL_PRESETS.keys())}"
        )

    provider, model_name = MODEL_PRESETS[preset_name]
    return make_chat_model(provider, model_name, api_key, **kwargs)
