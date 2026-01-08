"""
Unit tests for the web tools functionality.
"""

from unittest.mock import Mock, patch

import pytest

from kader.tools.web import WebFetchTool, WebSearchTool, get_web_tools


class TestWebSearchTool:
    """Test cases for WebSearchTool."""

    def test_initialization(self):
        """Test WebSearchTool initialization."""
        tool = WebSearchTool()

        assert tool.name == "web_search"
        assert "Search the web for information" in tool.description
        assert tool.schema.category == "web"

        # Check parameters
        params = {param.name: param for param in tool.schema.parameters}
        assert "query" in params
        assert "limit" in params
        assert params["query"].type == "string"
        assert params["limit"].type == "integer"
        assert params["limit"].default == 5

    @patch("kader.tools.web.ollama")
    def test_execute_success(self, mock_ollama):
        """Test successful web search execution."""
        # Mock the ollama.web_search function
        mock_result = Mock()
        mock_result.results = [
            {
                "title": "Test Result 1",
                "url": "http://example1.com",
                "content": "Content 1",
            },
            {
                "title": "Test Result 2",
                "url": "http://example2.com",
                "content": "Content 2",
            },
        ]
        mock_ollama.web_search.return_value = mock_result

        tool = WebSearchTool()
        result = tool.execute("test query", limit=2)

        # Verify the ollama function was called correctly
        mock_ollama.web_search.assert_called_once_with(query="test query")

        # Verify the result
        assert len(result) == 2
        assert result[0]["title"] == "Test Result 1"
        assert result[1]["url"] == "http://example2.com"

    @patch("kader.tools.web.ollama")
    def test_execute_dict_response(self, mock_ollama):
        """Test web search with dictionary response."""
        mock_ollama.web_search.return_value = {
            "results": [
                {
                    "title": "Dict Result",
                    "url": "http://dict.com",
                    "content": "Dict content",
                }
            ]
        }

        tool = WebSearchTool()
        result = tool.execute("dict query")

        assert len(result) == 1
        assert result[0]["title"] == "Dict Result"

    @patch("kader.tools.web.ollama")
    def test_execute_list_response(self, mock_ollama):
        """Test web search with list response."""
        mock_ollama.web_search.return_value = [
            {
                "title": "List Result",
                "url": "http://list.com",
                "content": "List content",
            }
        ]

        tool = WebSearchTool()
        result = tool.execute("list query")

        assert len(result) == 1
        assert result[0]["title"] == "List Result"

    @patch("kader.tools.web.ollama")
    def test_execute_limit_results(self, mock_ollama):
        """Test limiting the number of results."""
        mock_ollama.web_search.return_value = {
            "results": [
                {
                    "title": f"Result {i}",
                    "url": f"http://example{i}.com",
                    "content": f"Content {i}",
                }
                for i in range(10)
            ]
        }

        tool = WebSearchTool()
        result = tool.execute("test query", limit=3)

        assert len(result) == 3
        assert result[0]["title"] == "Result 0"
        assert result[2]["title"] == "Result 2"

    @patch("kader.tools.web.ollama")
    def test_execute_no_web_search_function(self, mock_ollama):
        """Test when ollama.web_search is not available."""
        # Remove the web_search attribute to simulate it not being available
        del mock_ollama.web_search

        tool = WebSearchTool()

        with pytest.raises(
            NotImplementedError, match="ollama.web_search is not available"
        ):
            tool.execute("test query")

    @patch("kader.tools.web.ollama")
    def test_execute_error_handling(self, mock_ollama):
        """Test error handling in web search."""
        mock_ollama.web_search.side_effect = Exception("API Error")

        tool = WebSearchTool()

        with pytest.raises(RuntimeError, match="Web search failed"):
            tool.execute("test query")

    @patch("kader.tools.web.ollama")
    @pytest.mark.asyncio
    async def test_aexecute(self, mock_ollama):
        """Test asynchronous web search execution."""
        mock_result = Mock()
        mock_result.results = [
            {
                "title": "Async Result",
                "url": "http://async.com",
                "content": "Async content",
            }
        ]
        mock_ollama.web_search.return_value = mock_result

        tool = WebSearchTool()
        result = await tool.aexecute("async query")

        assert len(result) == 1
        assert result[0]["title"] == "Async Result"


class TestWebFetchTool:
    """Test cases for WebFetchTool."""

    def test_initialization(self):
        """Test WebFetchTool initialization."""
        tool = WebFetchTool()

        assert tool.name == "web_fetch"
        assert "Fetch and extract readable text content from a URL" in tool.description
        assert tool.schema.category == "web"

        # Check parameters
        params = {param.name: param for param in tool.schema.parameters}
        assert "url" in params
        assert params["url"].type == "string"
        assert params["url"].required is True

    @patch("kader.tools.web.ollama")
    def test_execute_success(self, mock_ollama):
        """Test successful web fetch execution."""
        mock_ollama.web_fetch.return_value = {"content": "Fetched web page content"}

        tool = WebFetchTool()
        result = tool.execute("http://example.com")

        # Verify the ollama function was called correctly
        mock_ollama.web_fetch.assert_called_once_with(url="http://example.com")

        # Verify the result
        assert result == "Fetched web page content"

    @patch("kader.tools.web.ollama")
    def test_execute_string_response(self, mock_ollama):
        """Test web fetch with string response."""
        mock_ollama.web_fetch.return_value = "Direct string content"

        tool = WebFetchTool()
        result = tool.execute("http://example.com")

        assert result == "Direct string content"

    @patch("kader.tools.web.ollama")
    def test_execute_no_web_fetch_function(self, mock_ollama):
        """Test when ollama.web_fetch is not available."""
        # Remove the web_fetch attribute to simulate it not being available
        del mock_ollama.web_fetch

        tool = WebFetchTool()

        with pytest.raises(
            NotImplementedError, match="ollama.web_fetch is not available"
        ):
            tool.execute("http://example.com")

    @patch("kader.tools.web.ollama")
    def test_execute_error_handling(self, mock_ollama):
        """Test error handling in web fetch."""
        mock_ollama.web_fetch.side_effect = Exception("Fetch Error")

        tool = WebFetchTool()

        with pytest.raises(RuntimeError, match="Web fetch failed"):
            tool.execute("http://example.com")

    @patch("kader.tools.web.ollama")
    @pytest.mark.asyncio
    async def test_aexecute(self, mock_ollama):
        """Test asynchronous web fetch execution."""
        mock_ollama.web_fetch.return_value = {"content": "Async fetch content"}

        tool = WebFetchTool()
        result = await tool.aexecute("http://async.com")

        assert result == "Async fetch content"


class TestGetWebTools:
    """Test cases for get_web_tools function."""

    def test_get_web_tools(self):
        """Test getting all web tools."""
        tools = get_web_tools()

        assert len(tools) == 2
        tool_names = [tool.name for tool in tools]
        assert "web_search" in tool_names
        assert "web_fetch" in tool_names

        # Verify they are the correct types
        search_tool = next(tool for tool in tools if tool.name == "web_search")
        fetch_tool = next(tool for tool in tools if tool.name == "web_fetch")

        assert isinstance(search_tool, WebSearchTool)
        assert isinstance(fetch_tool, WebFetchTool)
