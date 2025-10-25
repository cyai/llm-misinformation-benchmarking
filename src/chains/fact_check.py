from pathlib import Path
from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
import json


def _load_prompt(path: Path) -> ChatPromptTemplate:
    text = Path(path).read_text(encoding="utf-8")
    # User content only (minimal systeming). We keep it simple to honor "zero prompting".
    return ChatPromptTemplate.from_messages(
        [
            ("system", "You respond with a single JSON object and nothing else."),
            ("user", text + "\n\nClaim:\n{claim}"),
        ]
    )


def build_fact_check_chain(llm, prompt_path: Path):
    prompt = _load_prompt(prompt_path)
    to_json = RunnableLambda(lambda s: json.loads(s))
    chain = prompt | llm | StrOutputParser() | to_json
    # Wrap with input passthrough to preserve the raw claim too
    return RunnablePassthrough.assign(result=chain)
