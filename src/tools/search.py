"""Google Search tool for fact-checking."""

from typing import List, Dict, Optional
import os


class SearchTool:
    """Simple Google search wrapper for fact-checking."""

    def __init__(self, api_key: Optional[str] = None, max_results: int = 5):
        """
        Initialize search tool.

        Args:
            api_key: API key (SERPAPI_API_KEY or GOOGLE_API_KEY)
            max_results: Maximum number of search results to return
        """
        self.max_results = max_results
        self._search = None

        # Try SerpAPI first (simpler, more reliable)
        serpapi_key = api_key or os.getenv("SERPAPI_API_KEY")
        if serpapi_key:
            try:
                from langchain_community.utilities import SerpAPIWrapper

                self._search = SerpAPIWrapper(serpapi_api_key=serpapi_key)
                self._search_type = "serpapi"
                return
            except ImportError:
                print(
                    "Warning: SerpAPI not available. Install: pip install google-search-results"
                )
            except Exception as e:
                print(f"Warning: SerpAPI initialization failed: {e}")

        # Fallback to Google Custom Search API
        google_api_key = os.getenv("GOOGLE_API_KEY")
        google_cse_id = os.getenv("GOOGLE_CSE_ID")
        if google_api_key and google_cse_id:
            try:
                from langchain_community.utilities import GoogleSearchAPIWrapper

                self._search = GoogleSearchAPIWrapper(
                    google_api_key=google_api_key,
                    google_cse_id=google_cse_id,
                )
                self._search_type = "google"
                return
            except ImportError:
                print(
                    "Warning: Google Search API not available. Install: pip install google-api-python-client"
                )
            except Exception as e:
                print(f"Warning: Google Search API initialization failed: {e}")

        raise ValueError(
            "No search API configured. Please set either:\n"
            "  - SERPAPI_API_KEY (recommended: https://serpapi.com)\n"
            "  - GOOGLE_API_KEY and GOOGLE_CSE_ID\n"
            "Then install: pip install google-search-results or pip install google-api-python-client"
        )

    def search_claim(self, claim: str) -> str:
        """
        Search for information about a claim.

        Args:
            claim: The claim to search for

        Returns:
            Formatted search results as a string
        """
        if not self._search:
            return "Search unavailable: No API configured"

        try:
            # Get search results
            if self._search_type == "serpapi":
                results = self._search.results(claim)
                organic_results = results.get("organic_results", [])[: self.max_results]

                # Format results
                formatted = []
                for i, result in enumerate(organic_results, 1):
                    title = result.get("title", "No title")
                    snippet = result.get("snippet", "No snippet")
                    link = result.get("link", "No link")

                    formatted.append(
                        f"[{i}] {title}\n" f"    {snippet}\n" f"    Source: {link}"
                    )

                return (
                    "\n\n".join(formatted) if formatted else "No search results found."
                )

            elif self._search_type == "google":
                # Google Custom Search returns formatted text
                result_text = self._search.run(claim)
                return result_text if result_text else "No search results found."

            else:
                return "Search error: Unknown search type"

        except Exception as e:
            return f"Search error: {str(e)}"

    def quick_search(self, claim: str) -> str:
        """
        Simpler interface - just get snippet text.

        Args:
            claim: The claim to search for

        Returns:
            Search result snippets as text
        """
        if not self._search:
            return "Search unavailable"

        try:
            return self._search.run(claim)
        except Exception as e:
            return f"Search error: {str(e)}"
