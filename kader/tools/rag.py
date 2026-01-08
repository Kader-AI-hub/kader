"""
RAG (Retrieval Augmented Generation) Tool.

Provides semantic search capabilities using Ollama embeddings and FAISS indexing.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import ollama
    from ollama import Client
except ImportError:
    ollama = None
    Client = None

import hashlib
import pickle

try:
    import faiss
    import numpy as np
except ImportError:
    faiss = None
    np = None

from .base import (
    BaseTool,
    ParameterSchema,
    ToolCategory,
)

# Default embedding model
DEFAULT_EMBEDDING_MODEL = "all-minilm:22m"

# File extensions to index by default
DEFAULT_CODE_EXTENSIONS = {
    ".py",
    ".js",
    ".ts",
    ".jsx",
    ".tsx",
    ".java",
    ".cpp",
    ".c",
    ".h",
    ".go",
    ".rs",
    ".rb",
    ".php",
    ".cs",
    ".swift",
    ".kt",
    ".scala",
    ".html",
    ".css",
    ".scss",
    ".less",
    ".json",
    ".yaml",
    ".yml",
    ".md",
    ".txt",
    ".rst",
    ".toml",
    ".ini",
    ".cfg",
    ".sh",
    ".bash",
}

# Directories to exclude by default
DEFAULT_EXCLUDE_DIRS = {
    ".git",
    ".svn",
    ".hg",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    "env",
    ".env",
    "dist",
    "build",
    ".tox",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "target",
    "bin",
    "obj",
    ".idea",
    ".vscode",
}

# Maximum file size to index (1MB)
MAX_FILE_SIZE = 1_000_000

# Chunk size for text splitting
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


@dataclass
class DocumentChunk:
    """A chunk of text with metadata for indexing."""

    content: str
    file_path: str
    start_line: int
    end_line: int
    chunk_index: int

    # Embedding vector (populated after embedding)
    embedding: list[float] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "content": self.content,
            "file_path": self.file_path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "chunk_index": self.chunk_index,
        }


@dataclass
class SearchResult:
    """A search result from RAG search."""

    content: str
    file_path: str
    start_line: int
    end_line: int
    score: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "content": self.content,
            "file_path": self.file_path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "score": self.score,
        }


@dataclass
class RAGIndex:
    """
    Manages FAISS index with Ollama embeddings for semantic search.

    Example:
        index = RAGIndex(base_path=Path.cwd())
        index.build()
        results = index.search("function to read file", top_k=5)
    """

    base_path: Path
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    include_extensions: set[str] = field(
        default_factory=lambda: DEFAULT_CODE_EXTENSIONS.copy()
    )
    exclude_dirs: set[str] = field(default_factory=lambda: DEFAULT_EXCLUDE_DIRS.copy())
    index_dir: Path | None = None

    # Internal state
    _chunks: list[DocumentChunk] = field(default_factory=list, repr=False)
    _index: Any = field(default=None, repr=False)  # FAISS index
    _is_built: bool = field(default=False, repr=False)
    _embedding_dim: int = field(default=768, repr=False)  # Default for embeddinggemma

    def _get_ollama_client(self):
        """Get Ollama client for embeddings."""
        if Client is None:
            raise ImportError("ollama is required for embeddings.")
        return Client()

    def _embed_text(self, text: str) -> list[float]:
        """Generate embedding for text using Ollama."""
        client = self._get_ollama_client()
        response = client.embed(model=self.embedding_model, input=text)
        return response.embeddings[0]

    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        client = self._get_ollama_client()
        response = client.embed(model=self.embedding_model, input=texts)
        return response.embeddings

    def _chunk_text(self, content: str, file_path: str) -> list[DocumentChunk]:
        """Split text into overlapping chunks."""
        lines = content.split("\n")
        chunks = []

        current_chunk_lines = []
        current_start_line = 1
        current_char_count = 0
        chunk_index = 0

        for i, line in enumerate(lines, start=1):
            line_len = len(line) + 1  # +1 for newline

            if current_char_count + line_len > CHUNK_SIZE and current_chunk_lines:
                # Save current chunk
                chunk_content = "\n".join(current_chunk_lines)
                chunks.append(
                    DocumentChunk(
                        content=chunk_content,
                        file_path=file_path,
                        start_line=current_start_line,
                        end_line=i - 1,
                        chunk_index=chunk_index,
                    )
                )
                chunk_index += 1

                # Start new chunk with overlap
                overlap_lines = max(1, len(current_chunk_lines) // 4)
                current_chunk_lines = current_chunk_lines[-overlap_lines:]
                current_start_line = i - overlap_lines
                current_char_count = sum(len(line) + 1 for line in current_chunk_lines)

            current_chunk_lines.append(line)
            current_char_count += line_len

        # Don't forget the last chunk
        if current_chunk_lines:
            chunk_content = "\n".join(current_chunk_lines)
            chunks.append(
                DocumentChunk(
                    content=chunk_content,
                    file_path=file_path,
                    start_line=current_start_line,
                    end_line=len(lines),
                    chunk_index=chunk_index,
                )
            )

        return chunks

    def _collect_files(self) -> list[Path]:
        """Collect all files to index."""
        files = []

        for root, dirs, filenames in os.walk(self.base_path):
            # Filter out excluded directories
            dirs[:] = [d for d in dirs if d not in self.exclude_dirs]

            for filename in filenames:
                file_path = Path(root) / filename

                # Check extension
                if file_path.suffix.lower() not in self.include_extensions:
                    continue

                # Check file size
                try:
                    if file_path.stat().st_size > MAX_FILE_SIZE:
                        continue
                except OSError:
                    continue

                files.append(file_path)

        return files

    @property
    def _index_dir(self) -> Path:
        """Get the directory where the index is stored."""
        if self.index_dir:
            dir_path = self.index_dir
        else:
            dir_path = self.base_path / ".kader" / "index"

        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path

    @property
    def _index_path(self) -> Path:
        """Get path to the FAISS index file."""
        # Create a unique name based on the base path hash to avoid collisions
        path_hash = hashlib.md5(str(self.base_path.absolute()).encode()).hexdigest()
        return self._index_dir / f"faiss_{path_hash}.index"

    @property
    def _chunks_path(self) -> Path:
        """Get path to the chunks metadata file."""
        path_hash = hashlib.md5(str(self.base_path.absolute()).encode()).hexdigest()
        return self._index_dir / f"chunks_{path_hash}.pkl"

    def save(self) -> None:
        """Save the index and chunks to disk."""
        if faiss is None:
            return

        if not self._is_built or self._index is None:
            return

        # Save FAISS index
        faiss.write_index(self._index, str(self._index_path))

        # Save chunks metadata
        with open(self._chunks_path, "wb") as f:
            pickle.dump(self._chunks, f)

    def load(self) -> bool:
        """
        Load the index and chunks from disk.

        Returns:
            True if loaded successfully, False otherwise
        """
        if faiss is None:
            return False

        if not self._index_path.exists() or not self._chunks_path.exists():
            return False

        try:
            # Load FAISS index
            self._index = faiss.read_index(str(self._index_path))

            # Load chunks metadata
            with open(self._chunks_path, "rb") as f:
                self._chunks = pickle.load(f)

            self._is_built = True
            self._embedding_dim = self._index.d
            return True
        except Exception:
            return False

    def build(self) -> int:
        """
        Build the index by scanning and embedding all files.

        Returns:
            Number of chunks indexed
        """
        if faiss is None:
            raise ImportError(
                "faiss-cpu is required for RAG search. "
                "Install it with: uv add faiss-cpu"
            )

        self._chunks = []
        files = self._collect_files()

        # Collect all chunks
        for file_path in files:
            try:
                content = file_path.read_text(encoding="utf-8", errors="ignore")
                rel_path = str(file_path.relative_to(self.base_path))
                chunks = self._chunk_text(content, rel_path)
                self._chunks.extend(chunks)
            except Exception:
                continue

        if not self._chunks:
            self._is_built = True
            return 0

        # Generate embeddings in batches
        batch_size = 32
        all_embeddings = []

        for i in range(0, len(self._chunks), batch_size):
            batch = self._chunks[i : i + batch_size]
            texts = [chunk.content for chunk in batch]
            embeddings = self._embed_texts(texts)
            all_embeddings.extend(embeddings)

            # Store embeddings in chunks
            for chunk, emb in zip(batch, embeddings):
                chunk.embedding = emb

        # Build FAISS index
        embeddings_array = np.array(all_embeddings, dtype=np.float32)
        self._embedding_dim = embeddings_array.shape[1]

        self._index = faiss.IndexFlatL2(self._embedding_dim)
        self._index.add(embeddings_array)

        self._is_built = True
        self.save()  # Auto-save after build
        return len(self._chunks)

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """
        Search the index for similar content.

        Args:
            query: Search query text
            top_k: Number of results to return

        Returns:
            List of SearchResult objects
        """
        import numpy as np

        if not self._is_built:
            self.build()

        if not self._chunks or self._index is None:
            return []

        # Embed the query
        query_embedding = self._embed_text(query)
        query_array = np.array([query_embedding], dtype=np.float32)

        # Search
        k = min(top_k, len(self._chunks))
        distances, indices = self._index.search(query_array, k)

        # Convert to results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self._chunks):
                continue

            chunk = self._chunks[idx]
            # Convert L2 distance to similarity score (lower distance = higher score)
            score = 1.0 / (1.0 + float(dist))

            results.append(
                SearchResult(
                    content=chunk.content,
                    file_path=chunk.file_path,
                    start_line=chunk.start_line,
                    end_line=chunk.end_line,
                    score=score,
                )
            )

        return results

    def clear(self) -> None:
        """Clear the index."""
        self._chunks = []
        self._index = None
        self._is_built = False


class RAGSearchTool(BaseTool[list[dict[str, Any]]]):
    """
    Tool for semantic search using RAG (Retrieval Augmented Generation).

    Uses Ollama embeddings and FAISS for fast similarity search across
    the codebase in the current working directory.
    """

    def __init__(
        self,
        base_path: Path | None = None,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    ) -> None:
        """
        Initialize the RAG search tool.

        Args:
            base_path: Base path to search in (defaults to CWD)
            embedding_model: Ollama embedding model to use
        """
        super().__init__(
            name="rag_search",
            description=(
                "Search for code and text using semantic similarity. "
                "Finds relevant files and code snippets based on meaning, not just keywords."
            ),
            parameters=[
                ParameterSchema(
                    name="query",
                    type="string",
                    description="Natural language search query",
                ),
                ParameterSchema(
                    name="top_k",
                    type="integer",
                    description="Number of results to return",
                    required=False,
                    default=5,
                    minimum=1,
                    maximum=20,
                ),
                ParameterSchema(
                    name="rebuild",
                    type="boolean",
                    description="Force rebuild the index before searching",
                    required=False,
                    default=False,
                ),
            ],
            category=ToolCategory.SEARCH,
        )

        self._base_path = base_path or Path.cwd()
        self._embedding_model = embedding_model
        self._index: RAGIndex | None = None

    def _get_or_build_index(self, rebuild: bool = False) -> RAGIndex:
        """Get existing index or build a new one."""
        if self._index is None:
            self._index = RAGIndex(
                base_path=self._base_path,
                embedding_model=self._embedding_model,
            )

        if rebuild:
            self._index.build()
        elif not self._index._is_built:
            # Try loading first, otherwise build
            if not self._index.load():
                self._index.build()

        return self._index

    def execute(
        self,
        query: str,
        top_k: int = 5,
        rebuild: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Execute semantic search.

        Args:
            query: Natural language search query
            top_k: Number of results to return
            rebuild: Force rebuild the index

        Returns:
            List of search result dictionaries
        """
        index = self._get_or_build_index(rebuild)
        results = index.search(query, top_k)
        return [r.to_dict() for r in results]

    async def aexecute(
        self,
        query: str,
        top_k: int = 5,
        rebuild: bool = False,
    ) -> list[dict[str, Any]]:
        """Async version of execute."""
        import asyncio

        return await asyncio.to_thread(self.execute, query, top_k, rebuild)

    def get_interruption_message(self, query: str, **kwargs) -> str:
        """Get interruption message for user confirmation."""
        return f"execute rag_search: {query}"
