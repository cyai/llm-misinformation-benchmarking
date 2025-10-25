from langchain_openai import ChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel


def make_chat_model(provider: str, model_name: str, api_key: str) -> BaseChatModel:
    provider = provider.lower()
    if provider in ("openai", "oai"):
        return ChatOpenAI(model=model_name, api_key=api_key, temperature=0, timeout=60)
    # Future: elif provider == "deepseek": ...
    # Future: elif provider == "bert": ...
    # Future: elif provider == "ollama": ...
    raise ValueError(f"Unsupported provider: {provider}")
