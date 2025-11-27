"""Fact-checking chain with Google Search integration."""

from pathlib import Path
from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from src.chains.fact_check import _parse_json_robust


def _load_search_prompt(path: Path) -> ChatPromptTemplate:
    """Load prompt template with search results placeholder."""
    text = Path(path).read_text(encoding="utf-8")
    return ChatPromptTemplate.from_messages(
        [
            ("system", "You respond with a single JSON object and nothing else."),
            ("user", text),
        ]
    )


def _perform_search(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run Google search and add results to inputs.

    Args:
        inputs: Dict with 'claim' and 'search_tool'

    Returns:
        Dict with 'claim' and 'search_results' added
    """
    claim = inputs["claim"]
    search_tool = inputs.get("search_tool")

    if search_tool:
        try:
            search_results = search_tool.search_claim(claim)
        except Exception as e:
            search_results = f"Search failed: {str(e)}"
    else:
        search_results = "Search unavailable: No search tool provided"

    return {"claim": claim, "search_results": search_results}


def build_fact_check_search_chain(llm, prompt_path: Path, search_tool):
    """
    Build a fact-checking chain with Google Search integration.

    Flow:
    1. Receive claim as input
    2. Search Google for information about the claim
    3. Pass claim + search results to LLM prompt
    4. Parse JSON response

    Args:
        llm: Language model instance
        prompt_path: Path to prompt template file
        search_tool: SearchTool instance for querying Google

    Returns:
        Runnable chain that accepts {"claim": str} and returns parsed result
    """
    prompt = _load_search_prompt(prompt_path)
    to_json = RunnableLambda(_parse_json_robust)

    # Chain: add search tool → perform search → prompt → llm → parse
    chain = (
        RunnablePassthrough.assign(search_tool=lambda _: search_tool)
        | RunnableLambda(_perform_search)
        | prompt
        | llm
        | StrOutputParser()
        | to_json
    )

    return RunnablePassthrough.assign(result=chain)
