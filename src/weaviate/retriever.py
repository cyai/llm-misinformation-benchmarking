"""RAG retrieval utilities for fact-checking."""

from typing import List, Dict, Any, Optional
import weaviate
from weaviate.classes.query import MetadataQuery


class FactCheckRetriever:
    """Retrieve similar claims from Weaviate for RAG."""

    def __init__(self, client: weaviate.WeaviateClient, iteration_id: int):
        """
        Initialize retriever for a specific iteration.

        Args:
            client: Weaviate client instance
            iteration_id: Test iteration number (0-4)
        """
        self.client = client
        self.iteration_id = iteration_id
        self.class_name = f"FactCheckKB_Iter{iteration_id}"
        self.collection = client.collections.get(self.class_name)

    def retrieve(
        self, query: str, top_k: int = 5, certainty: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Retrieve similar claims from the knowledge base.

        Args:
            query: Query claim to search for
            top_k: Number of results to return
            certainty: Minimum similarity threshold (0-1)

        Returns:
            List of similar claims with metadata
        """
        try:
            result = self.collection.query.near_text(
                query=query,
                limit=top_k,
                certainty=certainty,
                return_metadata=MetadataQuery(certainty=True, distance=True),
            )

            # Format results
            formatted = []
            for obj in result.objects:
                formatted.append(
                    {
                        "claim_id": obj.properties.get("claim_id"),
                        "claim": obj.properties.get("claim"),
                        "label": obj.properties.get("label"),
                        "verdict": obj.properties.get("verdict"),
                        "certainty": obj.metadata.certainty if obj.metadata else None,
                        "distance": obj.metadata.distance if obj.metadata else None,
                    }
                )

            return formatted

        except Exception as e:
            print(f"Error retrieving from {self.class_name}: {e}")
            return []
        finally:
            self.client.close()

    def format_context(self, retrieved: List[Dict[str, Any]]) -> str:
        """
        Format retrieved claims as context for the prompt.

        Args:
            retrieved: List of retrieved claims

        Returns:
            Formatted context string
        """
        if not retrieved:
            return "No similar claims found in knowledge base."

        context_parts = []
        for i, item in enumerate(retrieved, 1):
            certainty = item.get("certainty", 0)
            claim = item.get("claim", "N/A")
            label = item.get("label", "UNKNOWN")

            context_parts.append(
                f"[{i}] Claim: {claim}\n"
                f"    Verdict: {label}\n"
                f"    Similarity: {certainty:.2f}"
            )

        return "\n\n".join(context_parts)
