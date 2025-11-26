from pathlib import Path
from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
import json
import re


def _load_prompt(path: Path) -> ChatPromptTemplate:
    text = Path(path).read_text(encoding="utf-8")
    # User content only (minimal systeming). We keep it simple to honor "zero prompting".
    return ChatPromptTemplate.from_messages(
        [
            ("system", "You respond with a single JSON object and nothing else."),
            ("user", text + "\n\nClaim:\n{claim}"),
        ]
    )


def _parse_json_robust(s: str) -> Dict[str, Any]:
    """
    Parse JSON with multiple fallback strategies to handle LLM output issues.
    """
    # Strategy 1: Direct parse
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    
    # Strategy 2: Extract JSON from markdown code blocks or other text
    try:
        # Look for JSON between ```json and ``` or just between { and }
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', s, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(1))
        
        # Try to find the first complete JSON object
        json_match = re.search(r'\{.*\}', s, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(0))
    except json.JSONDecodeError:
        pass
    
    # Strategy 3: Try fixing common JSON issues
    try:
        # Remove potential BOM and leading/trailing whitespace
        cleaned = s.strip().lstrip('\ufeff')
        
        # Try to fix common issues like trailing commas
        cleaned = re.sub(r',(\s*[}\]])', r'\1', cleaned)
        
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    
    # Strategy 4: Extract fields manually with regex
    try:
        verdict_match = re.search(r'"verdict"\s*:\s*"([^"]+)"', s)
        confidence_match = re.search(r'"confidence"\s*:\s*([\d.]+)', s)
        rationale_match = re.search(r'"rationale"\s*:\s*"([^"]*(?:"[^"]*)*)"', s)
        
        return {
            "verdict": verdict_match.group(1) if verdict_match else "UNKNOWN",
            "confidence": float(confidence_match.group(1)) if confidence_match else 0.5,
            "rationale": rationale_match.group(1) if rationale_match else "Failed to parse LLM response",
            "cited_knowledge": "",
            "safety_notes": f"Warning: JSON parsing failed, extracted fields manually from: {s[:200]}",
        }
    except Exception as e:
        # Last resort: return error structure
        return {
            "verdict": "ERROR",
            "confidence": 0.0,
            "rationale": f"Failed to parse LLM output: {str(e)}",
            "cited_knowledge": "",
            "safety_notes": f"Raw output: {s[:500]}",
        }


def build_fact_check_chain(llm, prompt_path: Path):
    prompt = _load_prompt(prompt_path)
    to_json = RunnableLambda(_parse_json_robust)
    chain = prompt | llm | StrOutputParser() | to_json
    # Wrap with input passthrough to preserve the raw claim too
    return RunnablePassthrough.assign(result=chain)
