from pydantic import BaseModel
import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()


class Settings(BaseModel):
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    data_dir: Path = Path(os.getenv("DATA_DIR", "./data")).resolve()
    results_dir: Path = Path(os.getenv("RESULTS_DIR", "./results")).resolve()
    run_name: str = os.getenv("RUN_NAME", "openai_zero_shot")


settings = Settings()

# Ensure dirs exist
settings.data_dir.mkdir(parents=True, exist_ok=True)
settings.results_dir.mkdir(parents=True, exist_ok=True)
