"""
Unit tests for the RAG tools functionality.
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from kader.tools.rag import (
    DEFAULT_EMBEDDING_MODEL,
    DocumentChunk,
    RAGIndex,
    RAGSearchTool,
    SearchResult,
)


class TestDocumentChunk:
    """Test cases for DocumentChunk."""

    def test_document_chunk_creation(self):
        """Test creating a document chunk."""
        chunk = DocumentChunk(
            content="Test content",
            file_path="test.py",
            start_line=1,
            end_line=10,
            chunk_index=0,
            embedding=[0.1, 0.2, 0.3],
        )

        assert chunk.content == "Test content"
        assert chunk.file_path == "test.py"
        assert chunk.start_line == 1
        assert chunk.end_line == 10
        assert chunk.chunk_index == 0
        assert chunk.embedding == [0.1, 0.2, 0.3]

    def test_document_chunk_to_dict(self):
        """Test converting document chunk to dictionary."""
        chunk = DocumentChunk(
            content="Test content",
            file_path="test.py",
            start_line=1,
            end_line=10,
            chunk_index=0,
        )

        result = chunk.to_dict()

        expected = {
            "content": "Test content",
            "file_path": "test.py",
            "start_line": 1,
            "end_line": 10,
            "chunk_index": 0,
        }

        assert result == expected


class TestSearchResult:
    """Test cases for SearchResult."""

    def test_search_result_creation(self):
        """Test creating a search result."""
        result = SearchResult(
            content="Test result",
            file_path="test.py",
            start_line=1,
            end_line=10,
            score=0.8,
        )

        assert result.content == "Test result"
        assert result.file_path == "test.py"
        assert result.start_line == 1
        assert result.end_line == 10
        assert result.score == 0.8

    def test_search_result_to_dict(self):
        """Test converting search result to dictionary."""
        result = SearchResult(
            content="Test result",
            file_path="test.py",
            start_line=1,
            end_line=10,
            score=0.8,
        )

        result_dict = result.to_dict()

        expected = {
            "content": "Test result",
            "file_path": "test.py",
            "start_line": 1,
            "end_line": 10,
            "score": 0.8,
        }

        assert result_dict == expected


class TestRAGIndex:
    """Test cases for RAGIndex."""

    def test_initialization(self):
        """Test RAGIndex initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            base_path = Path(temp_dir)
            index = RAGIndex(base_path=base_path)

            assert index.base_path == base_path
            assert index.embedding_model == DEFAULT_EMBEDDING_MODEL
            assert index._is_built is False
            assert index._chunks == []
            assert index._index is None

    def test_initialization_with_custom_params(self):
        """Test RAGIndex initialization with custom parameters."""
        with tempfile.TemporaryDirectory() as temp_dir:
            base_path = Path(temp_dir)
            index = RAGIndex(
                base_path=base_path,
                embedding_model="custom-model",
                include_extensions={".py", ".js"},
                exclude_dirs={"node_modules", "__pycache__"},
            )

            assert index.base_path == base_path
            assert index.embedding_model == "custom-model"
            assert index.include_extensions == {".py", ".js"}
            assert index.exclude_dirs == {"node_modules", "__pycache__"}

    @patch("kader.tools.rag.Client")
    def test_get_ollama_client(self, mock_client_class):
        """Test getting Ollama client."""
        with tempfile.TemporaryDirectory() as temp_dir:
            index = RAGIndex(base_path=Path(temp_dir))
            _ = index._get_ollama_client()

            # Verify Client was instantiated
            mock_client_class.assert_called_once()

    @patch("kader.tools.rag.Client")
    def test_embed_text(self, mock_client_class):
        """Test embedding a single text."""
        mock_client_instance = Mock()
        mock_client_class.return_value = mock_client_instance

        mock_response = Mock()
        mock_response.embeddings = [[0.1, 0.2, 0.3]]
        mock_client_instance.embed.return_value = mock_response

        with tempfile.TemporaryDirectory() as temp_dir:
            index = RAGIndex(base_path=Path(temp_dir))
            embedding = index._embed_text("test text")

            # Verify the embed method was called correctly
            mock_client_instance.embed.assert_called_once_with(
                model=DEFAULT_EMBEDDING_MODEL, input="test text"
            )

            assert embedding == [0.1, 0.2, 0.3]

    @patch("kader.tools.rag.Client")
    def test_embed_texts(self, mock_client_class):
        """Test embedding multiple texts."""
        mock_client_instance = Mock()
        mock_client_class.return_value = mock_client_instance

        mock_response = Mock()
        mock_response.embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        mock_client_instance.embed.return_value = mock_response

        with tempfile.TemporaryDirectory() as temp_dir:
            index = RAGIndex(base_path=Path(temp_dir))
            embeddings = index._embed_texts(["text1", "text2"])

            # Verify the embed method was called correctly
            mock_client_instance.embed.assert_called_once_with(
                model=DEFAULT_EMBEDDING_MODEL, input=["text1", "text2"]
            )

            assert embeddings == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]

    def test_chunk_text(self):
        """Test chunking text."""
        with tempfile.TemporaryDirectory() as temp_dir:
            index = RAGIndex(base_path=Path(temp_dir))

            # Create a text with more than 500 characters to trigger chunking
            long_text = (
                "This is a line.\n" * 50
            )  # 50 lines, each ~15 chars = ~750 chars

            chunks = index._chunk_text(long_text, "test.py")

            # Should have multiple chunks due to size
            assert len(chunks) > 1
            assert all(isinstance(chunk, DocumentChunk) for chunk in chunks)
            assert all(chunk.file_path == "test.py" for chunk in chunks)

    def test_chunk_text_small(self):
        """Test chunking small text."""
        with tempfile.TemporaryDirectory() as temp_dir:
            index = RAGIndex(base_path=Path(temp_dir))

            text = "This is a short text."
            chunks = index._chunk_text(text, "test.py")

            # Should have just one chunk
            assert len(chunks) == 1
            assert chunks[0].content == text
            assert chunks[0].file_path == "test.py"
            assert chunks[0].start_line == 1

    def test_collect_files(self):
        """Test collecting files to index."""
        with tempfile.TemporaryDirectory() as temp_dir:
            base_path = Path(temp_dir)

            # Create test files
            (base_path / "test.py").write_text("print('hello')")
            (base_path / "test.js").write_text("console.log('hello')")
            (base_path / "test.txt").write_text(
                "just text"
            )  # Should be excluded by default

            # Create excluded directory
            excluded_dir = base_path / "__pycache__"
            excluded_dir.mkdir()
            (excluded_dir / "cached.py").write_text("cached content")

            index = RAGIndex(base_path=base_path)
            files = index._collect_files()

            # Should include .py, .js, and .txt files by default
            file_names = [f.name for f in files]
            assert "test.py" in file_names
            assert "test.js" in file_names
            assert "test.txt" in file_names  # Included by default extension
            assert "cached.py" not in file_names  # Excluded by directory

    def test_index_paths(self):
        """Test index path properties."""
        with tempfile.TemporaryDirectory() as temp_dir:
            base_path = Path(temp_dir) / "project"
            base_path.mkdir()

            index = RAGIndex(base_path=base_path)

            # Test index path
            index_path = index._index_path
            assert index_path.name.startswith("faiss_")
            assert index_path.suffix == ".index"

            # Test chunks path
            chunks_path = index._chunks_path
            assert chunks_path.name.startswith("chunks_")
            assert chunks_path.suffix == ".pkl"

    @patch("kader.tools.rag.faiss")
    @patch("kader.tools.rag.pickle")
    def test_save_and_load(self, mock_pickle, mock_faiss):
        """Test saving and loading the index."""
        with tempfile.TemporaryDirectory() as temp_dir:
            base_path = Path(temp_dir)
            index = RAGIndex(base_path=base_path)

            # Mock the index and chunks
            mock_faiss_index = Mock()
            index._index = mock_faiss_index
            index._chunks = [DocumentChunk("test", "file.py", 1, 10, 0)]
            index._is_built = True

            # Test save
            index.save()

            # Verify FAISS index was saved
            mock_faiss.write_index.assert_called_once()

            # Verify chunks were saved
            mock_pickle.dump.assert_called_once()

            # Reset mocks for load test
            mock_faiss.reset_mock()
            mock_pickle.reset_mock()

            # Mock the loading process
            mock_faiss.read_index.return_value = mock_faiss_index
            mock_pickle.load.return_value = [
                DocumentChunk("loaded", "file.py", 1, 10, 0)
            ]

            # Ensure index files "exist" for load() check
            index._index_path.parent.mkdir(parents=True, exist_ok=True)
            index._index_path.touch()
            index._chunks_path.touch()

            # Test load
            result = index.load()

            # Verify FAISS index was loaded
            mock_faiss.read_index.assert_called_once()

            # Verify chunks were loaded
            mock_pickle.load.assert_called_once()

            assert result is True
            assert index._is_built is True

    @patch("kader.tools.rag.faiss")
    @patch("kader.tools.rag.pickle")
    def test_load_nonexistent(self, mock_pickle, mock_faiss):
        """Test loading when index files don't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            base_path = Path(temp_dir)
            index = RAGIndex(base_path=base_path)

            # Test load when files don't exist
            result = index.load()

            assert result is False

    @patch("kader.tools.rag.faiss")
    @patch("kader.tools.rag.pickle")
    @patch("kader.tools.rag.Client")
    def test_build(self, mock_client_class, mock_pickle, mock_faiss):
        """Test building the index."""
        # Mock Ollama client
        mock_client_instance = Mock()
        mock_client_class.return_value = mock_client_instance

        # Mock embedding response
        mock_response = Mock()
        mock_response.embeddings = [[0.1, 0.2, 0.3]] * 2  # Two chunks
        mock_client_instance.embed.return_value = mock_response

        with tempfile.TemporaryDirectory() as temp_dir:
            base_path = Path(temp_dir)

            # Create a test file
            test_file = base_path / "test.py"
            test_file.write_text(
                "def hello():\n    print('Hello, World!')\n" * 20
            )  # Make it large enough to chunk

            index = RAGIndex(base_path=base_path, include_extensions={".py"})

            # Mock FAISS index
            mock_index_instance = Mock()
            mock_faiss.IndexFlatL2.return_value = mock_index_instance

            # Build the index
            chunk_count = index.build()

            # Verify the index was built
            assert chunk_count > 0
            assert index._is_built is True
            assert index._index is not None
            assert len(index._chunks) > 0

    @patch("kader.tools.rag.faiss")
    @patch("kader.tools.rag.Client")
    def test_search(self, mock_client_class, mock_faiss):
        """Test searching the index."""
        # Mock Ollama client
        mock_client_instance = Mock()
        mock_client_class.return_value = mock_client_instance

        # Mock embedding responses
        def mock_embed(model, input):
            if isinstance(input, list):
                # For building the index
                return Mock(embeddings=[[0.1, 0.2, 0.3]] * len(input))
            else:
                # For query embedding
                return Mock(embeddings=[[0.4, 0.5, 0.6]])

        mock_client_instance.embed.side_effect = mock_embed

        with tempfile.TemporaryDirectory() as temp_dir:
            base_path = Path(temp_dir)

            # Create a test file
            test_file = base_path / "test.py"
            test_file.write_text(
                "def search_term():\n    print('This contains the search term')\n" * 5
            )

            index = RAGIndex(base_path=base_path, include_extensions={".py"})

            # Mock FAISS index behavior
            mock_index_instance = Mock()
            mock_index_instance.d = 3  # embedding dimension
            mock_index_instance.search.return_value = (
                Mock(),
                Mock(),
            )  # distances, indices
            mock_faiss.IndexFlatL2.return_value = mock_index_instance

            # Build the index first
            index.build()

            # Mock the search result
            mock_index_instance.search.return_value = (
                [[0.1]],  # distances
                [[0]],  # indices
            )

            # Perform search
            results = index.search("search query", top_k=1)

            # Verify results are returned
            assert len(results) == 1
            assert isinstance(results[0], SearchResult)

    def test_clear(self):
        """Test clearing the index."""
        with tempfile.TemporaryDirectory() as temp_dir:
            index = RAGIndex(base_path=Path(temp_dir))

            # Simulate a built index
            index._chunks = [DocumentChunk("test", "file.py", 1, 10, 0)]
            index._index = Mock()
            index._is_built = True

            # Clear the index
            index.clear()

            # Verify everything was cleared
            assert index._chunks == []
            assert index._index is None
            assert index._is_built is False


class TestRAGSearchTool:
    """Test cases for RAGSearchTool."""

    def test_initialization(self):
        """Test RAGSearchTool initialization."""
        tool = RAGSearchTool()

        assert tool.name == "rag_search"
        assert "Search for code and text using semantic similarity" in tool.description
        assert tool.schema.category == "search"

        # Check parameters
        params = {param.name: param for param in tool.schema.parameters}
        assert "query" in params
        assert "top_k" in params
        assert "rebuild" in params
        assert params["query"].type == "string"
        assert params["top_k"].type == "integer"
        assert params["rebuild"].type == "boolean"

    def test_initialization_with_custom_params(self):
        """Test RAGSearchTool initialization with custom parameters."""
        custom_path = Path("/custom/path")
        tool = RAGSearchTool(base_path=custom_path, embedding_model="custom-model")

        assert tool._base_path == custom_path
        assert tool._embedding_model == "custom-model"
        assert tool._index is None

    @patch("kader.tools.rag.RAGIndex")
    def test_get_or_build_index_new(self, mock_rag_index_class):
        """Test getting or building a new index."""
        mock_index_instance = Mock()
        mock_rag_index_class.return_value = mock_index_instance

        tool = RAGSearchTool()
        index = tool._get_or_build_index()

        # Verify RAGIndex was created with correct parameters
        mock_rag_index_class.assert_called_once_with(
            base_path=Path.cwd(),
            embedding_model=DEFAULT_EMBEDDING_MODEL,
        )

        assert tool._index == mock_index_instance
        assert index == mock_index_instance

    @patch("kader.tools.rag.RAGIndex")
    def test_get_or_build_index_existing(self, mock_rag_index_class):
        """Test getting an existing index."""
        mock_index_instance = Mock()
        mock_index_instance._is_built = True

        tool = RAGSearchTool()
        tool._index = mock_index_instance

        index = tool._get_or_build_index()

        # RAGIndex should not be called again
        mock_rag_index_class.assert_not_called()

        assert index == mock_index_instance

    @patch("kader.tools.rag.RAGIndex")
    def test_get_or_build_index_rebuild(self, mock_rag_index_class):
        """Test getting or building index with rebuild."""
        mock_index_instance = Mock()
        mock_index_instance._is_built = True
        mock_rag_index_class.return_value = mock_index_instance

        tool = RAGSearchTool()
        index = tool._get_or_build_index(rebuild=True)

        # Verify build was called due to rebuild flag
        mock_index_instance.build.assert_called_once()
        assert index == mock_index_instance

    @patch("kader.tools.rag.RAGIndex")
    def test_execute(self, mock_rag_index_class):
        """Test executing the RAG search."""
        mock_index_instance = Mock()
        mock_index_instance._is_built = True
        mock_rag_index_class.return_value = mock_index_instance

        # Mock search results
        mock_search_result = [
            SearchResult("content1", "file1.py", 1, 10, 0.9),
            SearchResult("content2", "file2.py", 5, 15, 0.8),
        ]
        mock_index_instance.search.return_value = mock_search_result

        tool = RAGSearchTool()
        result = tool.execute("test query", top_k=2)

        # Verify search was called with correct parameters
        mock_index_instance.search.assert_called_once_with("test query", 2)

        # Verify the result format
        assert len(result) == 2
        assert result[0]["content"] == "content1"
        assert result[0]["file_path"] == "file1.py"
        assert result[1]["score"] == 0.8

    @patch("kader.tools.rag.RAGIndex")
    @pytest.mark.asyncio
    async def test_aexecute(self, mock_rag_index_class):
        """Test asynchronous execution of RAG search."""
        mock_index_instance = Mock()
        mock_index_instance._is_built = True
        mock_rag_index_class.return_value = mock_index_instance

        # Mock search results
        mock_search_result = [SearchResult("async content", "async.py", 1, 5, 0.95)]
        mock_index_instance.search.return_value = mock_search_result

        tool = RAGSearchTool()
        result = await tool.aexecute("async query", top_k=1, rebuild=False)

        # Verify the result
        assert len(result) == 1
        assert result[0]["content"] == "async content"
