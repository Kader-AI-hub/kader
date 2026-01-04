"""
Web Tools for Agentic Operations.

Provides web search and fetch capabilities using Ollama's native API.
"""

import json
from typing import Any

from .base import (
    BaseTool,
    ParameterSchema,
    ToolCategory,
)

try:
    import ollama
except ImportError:
    ollama = None


class WebSearchTool(BaseTool[list[dict[str, Any]]]):
    """
    Tool to search the web using Ollama's search capability.
    """

    def __init__(self) -> None:
        """Initialize the web search tool."""
        super().__init__(
            name="web_search",
            description=(
                "Search the web for information. Returns a list of relevant "
                "results with titles, descriptions, and URLs."
            ),
            parameters=[
                ParameterSchema(
                    name="query",
                    type="string",
                    description="Search query",
                ),
                ParameterSchema(
                    name="limit",
                    type="integer",
                    description="Number of results to return (max 10)",
                    required=False,
                    default=5,
                    minimum=1,
                    maximum=10,
                ),
            ],
            category=ToolCategory.WEB,
        )

    def execute(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """
        Execute web search.

        Args:
            query: Search query
            limit: Number of results (default 5)

        Returns:
        """
        if ollama is None:
            raise ImportError(
                "ollama is required for web tools. Install it with: uv add ollama"
            )

        try:
            # Note: Ollama python lib uses 'limit' or just implicitly returns results
            # The exact signature might vary as it's experimental, but based on docs:
            # results = ollama.web_search(query=query)
            # We'll try to slice manually if limit is not supported directly

            # Using getattr to avoid static analysis errors if method is missing in older stub
            search_func = getattr(ollama, "web_search", None)

            if not search_func:
                # Fallback or error if not available
                # Assuming duckduckgo-search as fallback if allowed, but plan said Ollama first
                # Let's check if we can import duckduckgo search as a robust fallback
                # since "ollama.web_search" might be a hosted-only feature in some contexts.
                # But per instruction, we use Ollama.

                # If function is missing, raise clear error
                raise NotImplementedError(
                    "ollama.web_search is not available in the installed library version. "
                    "Ensure 'ollama>=0.6.0' is installed."
                )

            # Execute search
            response = search_func(query=query)

            # Parse results
            # Response might be a dict, an object with .results, or a list
            results = []

            if isinstance(response, dict):
                results = response.get("results", [])
            elif hasattr(response, "results"):
                results = response.results
            elif hasattr(response, "__dict__") and "results" in response.__dict__:
                # Handle WebSearchResult object specifically
                results = response.results
            elif isinstance(response, list):
                results = response
            else:
                # If it's some other object that behaves like a dict but failed above checks
                try:
                    results = response["results"]
                except (TypeError, KeyError, AttributeError):
                    # As a last resort, assume the response itself might be the iterable result
                    results = response

            # Type check results to ensure it's a list (or iterable)
            if not isinstance(results, list):
                # Try to convert to list if it's iterable
                try:
                    results = list(results)
                except TypeError:
                    # If not iterable, wrap it or return empty
                    results = []

            # Ensure results are JSON serializable
            serializable_results = []
            for item in results[:limit]:
                if isinstance(item, (str, int, float, bool, type(None))):
                    serializable_results.append(item)
                elif isinstance(item, dict):
                    # Ensure all dict values are serializable
                    serializable_dict = {}
                    for k, v in item.items():
                        try:
                            json.dumps(v)  # Test if serializable
                            serializable_dict[k] = v
                        except TypeError:
                            serializable_dict[k] = str(
                                v
                            )  # Convert non-serializable to string
                    serializable_results.append(serializable_dict)
                else:
                    # Convert any other type to string representation
                    serializable_results.append(str(item))

            return serializable_results

        except NotImplementedError:
            raise
        except Exception as e:
            # Handle specific Ollama errors
            raise RuntimeError(f"Web search failed: {str(e)}")

    async def aexecute(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """Async version of execute."""
        import asyncio

        return await asyncio.to_thread(self.execute, query, limit)

    def get_interruption_message(self, query: str, **kwargs) -> str:
        """Get interruption message for user confirmation."""
        return f"execute web_search: {query}"


class WebFetchTool(BaseTool[str]):
    """
    Tool to fetch and extract text content from a web page.
    """

    def __init__(self) -> None:
        """Initialize the web fetch tool."""
        super().__init__(
            name="web_fetch",
            description=(
                "Fetch and extract readable text content from a URL. "
                "Useful for reading documentation, articles, or web pages found via search."
            ),
            parameters=[
                ParameterSchema(
                    name="url",
                    type="string",
                    description="URL of the page to fetch",
                ),
            ],
            category=ToolCategory.WEB,
        )

    def execute(self, url: str) -> str:
        """
        Fetch web page content.

        Args:
            url: URL to fetch

        Returns:
            Extracted text content
        """
        if ollama is None:
            raise ImportError(
                "ollama is required for web tools. Install it with: uv add ollama"
            )

        try:
            fetch_func = getattr(ollama, "web_fetch", None)

            if not fetch_func:
                raise NotImplementedError(
                    "ollama.web_fetch is not available in the installed library version."
                )

            response = fetch_func(url=url)

            # Return content (assuming response implies content string or dict with content)
            if isinstance(response, dict):
                content = response.get("content", str(response))
            else:
                content = str(response)

            # Ensure the content is JSON serializable
            try:
                json.dumps(content)  # Test if serializable
                return content
            except TypeError:
                return str(content)  # Convert non-serializable to string

        except NotImplementedError:
            raise
        except Exception as e:
            raise RuntimeError(f"Web fetch failed: {str(e)}")

    async def aexecute(self, url: str) -> str:
        """Async version of execute."""
        import asyncio

        return await asyncio.to_thread(self.execute, url)

    def get_interruption_message(self, url: str, **kwargs) -> str:
        """Get interruption message for user confirmation."""
        return f"execute web_fetch: {url}"


def get_web_tools() -> list[BaseTool]:
    """Get all web tools."""
    return [
        WebSearchTool(),
        WebFetchTool(),
    ]
