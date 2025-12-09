"""Weaviate client configuration and utilities."""

import weaviate
from weaviate.classes.init import Auth, AdditionalConfig, Timeout
import os
from typing import Optional


def get_weaviate_client(
    url: str = "http://localhost:8080",
    api_key: Optional[str] = None,
    openai_api_key: Optional[str] = None,
) -> weaviate.WeaviateClient:
    """
    Create and return a Weaviate client instance (v4 API).

    Args:
        url: Weaviate instance URL (default: http://localhost:8080)
        api_key: Weaviate API key (if authentication enabled)
        openai_api_key: OpenAI API key for vectorization

    Returns:
        Configured Weaviate client
    """
    # Get OpenAI API key from env if not provided
    if not openai_api_key:
        openai_api_key = os.getenv("OPENAI_API_KEY")

    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY is required for vectorization")

    # Parse host and port from URL
    if url.startswith("http://"):
        host = url.replace("http://", "")
    elif url.startswith("https://"):
        host = url.replace("https://", "")
    else:
        host = url

    # Split host:port
    if ":" in host:
        host, port = host.split(":")
        port = int(port)
    else:
        port = 8080

    # Create client with OpenAI API key in headers
    client = weaviate.connect_to_local(
        host=host,
        port=port,
        headers={"X-OpenAI-Api-Key": openai_api_key},
        additional_config=AdditionalConfig(
            timeout=Timeout(init=30, query=60, insert=120)
        ),
    )

    return client


def check_weaviate_ready(client: weaviate.WeaviateClient) -> bool:
    """
    Check if Weaviate instance is ready.

    Args:
        client: Weaviate client instance

    Returns:
        True if ready, False otherwise
    """
    try:
        return client.is_ready()
    except Exception as e:
        print(f"Weaviate not ready: {e}")
        return False
