from pydantic import BaseModel
import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(override=True)


class Settings(BaseModel):
    # OpenAI
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # Anthropic (Claude)
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")

    # Google (Gemini)
    google_api_key: str = os.getenv("GOOGLE_API_KEY", "")

    # xAI (Grok)
    xai_api_key: str = os.getenv("XAI_API_KEY", "")

    # HuggingFace
    huggingface_api_key: str = os.getenv("HUGGINGFACE_API_KEY", "")

    # DeepSeek
    deepseek_api_key: str = os.getenv("DEEPSEEK_API_KEY", "")

    # Search API configuration
    google_cse_id: str = os.getenv("GOOGLE_CSE_ID", "")
    serpapi_key: str = os.getenv("SERPAPI_API_KEY", "")
    data_dir: Path = Path(os.getenv("DATA_DIR", "./data")).resolve()
    results_dir: Path = Path(os.getenv("RESULTS_DIR", "./results")).resolve()
    run_name: str = os.getenv("RUN_NAME", "openai_zero_shot")


settings = Settings()

# Ensure dirs exist
settings.data_dir.mkdir(parents=True, exist_ok=True)
settings.results_dir.mkdir(parents=True, exist_ok=True)
