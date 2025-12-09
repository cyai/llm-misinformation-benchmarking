"""Weaviate schema definitions for fact-checking knowledge base."""

from typing import Dict, Any
from weaviate.classes.config import Configure, Property, DataType


def get_factcheck_schema(iteration_id: int) -> Dict[str, Any]:
    """
    Get schema for fact-checking knowledge base for a specific iteration.

    Each iteration gets its own collection to keep data separate and allow
    iteration-specific retrieval.

    Args:
        iteration_id: Test iteration number (0-4)

    Returns:
        Dictionary with collection configuration for v4 API
    """
    class_name = f"FactCheckKB_Iter{iteration_id}"

    # Return configuration for v4 API
    return {
        "name": class_name,
        "description": f"Fact-checking knowledge base for test iteration {iteration_id}",
        "properties": [
            Property(
                name="claim_id",
                data_type=DataType.TEXT,
                description="Unique identifier for the claim",
                skip_vectorization=True,
            ),
            Property(
                name="claim",
                data_type=DataType.TEXT,
                description="The claim text to fact-check",
                skip_vectorization=False,
            ),
            Property(
                name="label",
                data_type=DataType.TEXT,
                description="Ground truth label (FACT or FALSE)",
                skip_vectorization=True,
            ),
            Property(
                name="verdict",
                data_type=DataType.TEXT,
                description="Verdict reasoning or context",
                skip_vectorization=True,
            ),
            Property(
                name="source",
                data_type=DataType.TEXT,
                description="Source of the claim (e.g., 'politifact')",
                skip_vectorization=True,
            ),
        ],
        "vectorizer_config": Configure.Vectorizer.text2vec_openai(
            model="text-embedding-3-small"
        ),
    }


def get_all_iteration_schemas() -> list:
    """
    Get schemas for all test iterations (0-4).

    Returns:
        List of schema definitions
    """
    return [get_factcheck_schema(i) for i in range(5)]
