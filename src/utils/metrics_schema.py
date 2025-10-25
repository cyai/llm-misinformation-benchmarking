from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from datetime import datetime


class FactCheckRecord(BaseModel):
    run_name: str
    model_name: str
    provider: str
    claim_id: str
    claim_text: str
    gold_label: Optional[str] = None  # if available
    verdict: str
    confidence: float
    rationale: str
    cited_knowledge: str
    safety_notes: str
    latency_ms: int
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    extra: Dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
