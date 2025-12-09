"""Fact-checking chain with RAG (Retrieval-Augmented Generation)."""

from pathlib import Path
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from typing import Dict, Any
import json
import re

from src.weaviate.retriever import FactCheckRetriever


def _parse_json_robust(text: str) -> Dict[str, Any]:
    """
    Robustly parse JSON from LLM output with multiple fallback strategies.

    Same parsing logic as in fact_check.py for consistency.
    """
    # Strategy 1: Try direct JSON parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strategy 2: Extract JSON from markdown code block
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    # Strategy 3: Find first { to last }
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        json_str = text[start:end]
        return json.loads(json_str)
    except (ValueError, json.JSONDecodeError):
        pass

    # Strategy 4: Return error structure
    return {
        "verdict": "ERROR",
        "confidence": 0.5,
        "rationale": f"Failed to parse JSON from response: {text[:200]}",
        "retrieved_claims_used": 0,
        "cited_knowledge": [],
    }


def build_fact_check_rag_chain(
    llm: ChatOpenAI,
    prompt_path: Path,
    retriever: FactCheckRetriever,
    top_k: int = 5,
    certainty: float = 0.7,
):
    """
    Build a fact-checking chain with RAG.

    Args:
        llm: Language model to use
        prompt_path: Path to prompt template file
        retriever: Weaviate retriever instance
        top_k: Number of similar claims to retrieve
        certainty: Minimum similarity threshold

    Returns:
        Runnable chain
    """
    # Load prompt template
    with open(prompt_path, "r") as f:
        template = f.read()

    prompt = PromptTemplate(
        input_variables=["claim", "retrieved_context"], template=template
    )

    def _retrieve_and_format(inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Retrieve similar claims and format as context."""
        claim = inputs["claim"]

        # Retrieve similar claims
        retrieved = retriever.retrieve(query=claim, top_k=top_k, certainty=certainty)

        # Format context
        context = retriever.format_context(retrieved)

        # Add to inputs
        return {
            "claim": claim,
            "retrieved_context": context,
            "retrieved_claims": retrieved,  # Keep for potential analysis
        }

    # Build chain
    chain = (
        RunnablePassthrough()
        | _retrieve_and_format
        | prompt
        | llm
        | StrOutputParser()
        | (lambda x: {"raw_output": x, "result": _parse_json_robust(x)})
        | RunnablePassthrough.assign(result=lambda x: x["result"])
    )

    return chain
