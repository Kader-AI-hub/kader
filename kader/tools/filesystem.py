"""FilesystemBackend: Read and write files directly from the filesystem.

Security and search upgrades:
- Secure path resolution with root containment when in virtual_mode (sandboxed to cwd)
- Prevent symlink-following on file I/O using O_NOFOLLOW when available
- Ripgrep-powered grep with JSON parsing, plus Python fallback with regex
  and optional glob include filtering, while preserving virtual path behavior
"""

import contextlib
import json
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path

try:
    import fcntl
except ImportError:
    fcntl = None  # type: ignore

import wcmatch.glob as wcglob

from kader.tools.protocol import (
    BackendProtocol,
    EditResult,
    FileDownloadResponse,
    FileInfo,
    FileUploadResponse,
    GrepMatch,
    WriteResult,
)
from kader.tools.utils import (
    check_empty_content,
    format_content_with_line_numbers,
    perform_string_replacement,
)


class FilesystemBackend(BackendProtocol):
    """Backend that reads and writes files directly from the filesystem.

    Files are accessed using their actual filesystem paths. Relative paths are
    resolved relative to the current working directory. Content is read/written
    as plain text, and metadata (timestamps) are derived from filesystem stats.
    """

    def __init__(
        self,
        root_dir: str | Path | None = None,
        virtual_mode: bool = False,
        max_file_size_mb: int = 10,
        use_file_lock: bool = True,
        include_mtime: bool = False,
    ) -> None:
        """Initialize filesystem backend.

        Args:
            root_dir: Optional root directory for file operations. If provided,
                     all file paths will be resolved relative to this directory.
                     If not provided, uses the current working directory.
            virtual_mode: Whether to treat paths as virtual absolute paths under cwd.
            max_file_size_mb: Maximum file size in MB for search/read guards.
            use_file_lock: Acquire an advisory file lock around edit() to serialize
                concurrent modifications. Ignored on platforms without fcntl.
            include_mtime: When True, read() appends a footer containing the file's
                st_mtime_ns token, enabling edit() freshness checks.
        """
        self.cwd = Path(root_dir).resolve() if root_dir else Path.cwd()
        self.virtual_mode = virtual_mode
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024
        self.use_file_lock = use_file_lock and fcntl is not None
        self.include_mtime = include_mtime

    def _resolve_path(self, key: str) -> Path:
        """Resolve a file path with security checks.

        When virtual_mode=True, treat incoming paths as virtual absolute paths under
        self.cwd, disallow traversal (.., ~) and ensure resolved path stays within root.
        When virtual_mode=False, preserve legacy behavior: absolute paths are allowed
        as-is; relative paths resolve under cwd.

        Args:
            key: File path (absolute, relative, or virtual when virtual_mode=True)

        Returns:
            Resolved absolute Path object
        """
        if self.virtual_mode:
            vpath = key if key.startswith("/") else "/" + key
            if ".." in vpath or vpath.startswith("~"):
                raise ValueError("Path traversal not allowed")
            full = (self.cwd / vpath.lstrip("/")).resolve()
            try:
                full.relative_to(self.cwd)
            except ValueError:
                raise ValueError(
                    f"Path:{full} outside root directory: {self.cwd}"
                ) from None
            return full

        path = Path(key)
        if path.is_absolute():
            return path
        return (self.cwd / path).resolve()

    @contextlib.contextmanager
    def _file_lock(self, resolved_path: Path):
        """Acquire an advisory exclusive lock for a file path.

        The lock is implemented with a sibling lock file so that O_NOFOLLOW
        semantics on the target are not compromised. The lock file is removed
        after the lock is released to avoid leaving stale files in the workspace.
        On platforms without fcntl or when use_file_lock is False, this is a
        no-op context manager.
        """
        if not self.use_file_lock or fcntl is None:
            yield
            return

        lock_path = resolved_path.parent / f".kader-lock.{resolved_path.name}"
        fd = -1
        try:
            flags = os.O_RDWR | os.O_CREAT
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            fd = os.open(lock_path, flags, 0o644)
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            if fd >= 0:
                try:
                    fcntl.flock(fd, fcntl.LOCK_UN)
                except OSError:
                    pass
                try:
                    os.close(fd)
                except OSError:
                    pass
            try:
                lock_path.unlink(missing_ok=True)
            except OSError:
                pass

    def _atomic_write(
        self,
        resolved_path: Path,
        content: str | bytes,
        mode: str = "w",
        encoding: str = "utf-8",
    ) -> None:
        """Atomically write content to resolved_path.

        Writes to a sibling temp file, fsyncs data and parent directory, then
        uses os.replace() so readers always see either the old or the new
        content, never a partial write. Temp files are cleaned up on failure.
        """
        tmp_path = resolved_path.with_suffix(resolved_path.suffix + ".kader-tmp")
        try:
            flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            fd = os.open(tmp_path, flags, 0o644)
            try:
                if "b" in mode:
                    fobj = os.fdopen(fd, mode)
                else:
                    fobj = os.fdopen(fd, mode, encoding=encoding)
                with fobj as f:
                    f.write(content)
                    f.flush()
                    os.fsync(fd)
            except Exception:
                try:
                    os.close(fd)
                except OSError:
                    pass
                raise

            # fsync parent directory to ensure the replacement is durable
            try:
                parent_fd = os.open(
                    resolved_path.parent,
                    os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
                )
                try:
                    os.fsync(parent_fd)
                finally:
                    os.close(parent_fd)
            except OSError:
                # Best-effort durability on platforms where fsync on a dir fails
                pass

            os.replace(tmp_path, resolved_path)
        except OSError:
            # Best-effort cleanup; re-raise so caller can report the error
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
            raise
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass

    def ls_info(self, path: str) -> list[FileInfo]:
        """List files and directories in the specified directory (non-recursive).

        Args:
            path: Absolute directory path to list files from.

        Returns:
            List of FileInfo-like dicts for files and directories directly in the directory.
            Directories have a trailing / in their path and is_dir=True.
        """
        try:
            dir_path = self._resolve_path(path)
        except ValueError:
            return []

        if not dir_path.exists() or not dir_path.is_dir():
            return []

        results: list[FileInfo] = []

        # Convert cwd to string for comparison
        cwd_str = str(self.cwd)
        if not cwd_str.endswith("/"):
            cwd_str += "/"

        # List only direct children (non-recursive)
        try:
            for child_path in dir_path.iterdir():
                try:
                    is_file = child_path.is_file()
                    is_dir = child_path.is_dir()
                except OSError:
                    continue

                abs_path = str(child_path)

                if not self.virtual_mode:
                    # Non-virtual mode: use absolute paths
                    if is_file:
                        try:
                            st = child_path.stat()
                            results.append(
                                {
                                    "path": abs_path,
                                    "is_dir": False,
                                    "size": int(st.st_size),
                                    "modified_at": datetime.fromtimestamp(
                                        st.st_mtime
                                    ).isoformat(),
                                }
                            )
                        except OSError:
                            results.append({"path": abs_path, "is_dir": False})
                    elif is_dir:
                        try:
                            st = child_path.stat()
                            results.append(
                                {
                                    "path": abs_path + "/",
                                    "is_dir": True,
                                    "size": 0,
                                    "modified_at": datetime.fromtimestamp(
                                        st.st_mtime
                                    ).isoformat(),
                                }
                            )
                        except OSError:
                            results.append({"path": abs_path + "/", "is_dir": True})
                else:
                    # Virtual mode: strip cwd prefix
                    if abs_path.startswith(cwd_str):
                        relative_path = abs_path[len(cwd_str) :]
                    elif abs_path.startswith(str(self.cwd)):
                        # Handle case where cwd doesn't end with /
                        relative_path = abs_path[len(str(self.cwd)) :].lstrip("/")
                    else:
                        # Path is outside cwd, return as-is or skip
                        relative_path = abs_path

                    virt_path = "/" + relative_path

                    if is_file:
                        try:
                            st = child_path.stat()
                            results.append(
                                {
                                    "path": virt_path,
                                    "is_dir": False,
                                    "size": int(st.st_size),
                                    "modified_at": datetime.fromtimestamp(
                                        st.st_mtime
                                    ).isoformat(),
                                }
                            )
                        except OSError:
                            results.append({"path": virt_path, "is_dir": False})
                    elif is_dir:
                        try:
                            st = child_path.stat()
                            results.append(
                                {
                                    "path": virt_path + "/",
                                    "is_dir": True,
                                    "size": 0,
                                    "modified_at": datetime.fromtimestamp(
                                        st.st_mtime
                                    ).isoformat(),
                                }
                            )
                        except OSError:
                            results.append({"path": virt_path + "/", "is_dir": True})
        except (OSError, PermissionError):
            pass

        # Keep deterministic order by path
        results.sort(key=lambda x: x.get("path", ""))
        return results

    def read(
        self,
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
    ) -> str:
        """Read file content with line numbers.

        Args:
            file_path: Absolute or relative file path.
            offset: Line offset to start reading from (0-indexed).
            limit: Maximum number of lines to read.

        Returns:
            Formatted file content with line numbers, or error message.
        """
        try:
            resolved_path = self._resolve_path(file_path)
        except ValueError as e:
            return f"Error: {str(e)}"

        if not resolved_path.exists() or not resolved_path.is_file():
            return f"Error: File '{file_path}' not found"

        try:
            mtime_ns: int | None = None
            if self.include_mtime:
                try:
                    mtime_ns = resolved_path.stat().st_mtime_ns
                except OSError:
                    mtime_ns = None

            # Open with O_NOFOLLOW where available to avoid symlink traversal
            fd = os.open(resolved_path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
            with os.fdopen(fd, "r", encoding="utf-8") as f:
                content = f.read()

            empty_msg = check_empty_content(content)
            if empty_msg:
                if self.include_mtime and mtime_ns is not None:
                    return f"{empty_msg}\n# kader-mtime: {mtime_ns}"
                return empty_msg

            lines = content.splitlines()
            start_idx = offset
            end_idx = min(start_idx + limit, len(lines))

            if start_idx >= len(lines):
                return f"Error: Line offset {offset} exceeds file length ({len(lines)} lines)"

            selected_lines = lines[start_idx:end_idx]
            result = format_content_with_line_numbers(
                selected_lines, start_line=start_idx + 1
            )
            if self.include_mtime and mtime_ns is not None:
                result = f"{result}\n# kader-mtime: {mtime_ns}"
            return result
        except (OSError, UnicodeDecodeError) as e:
            return f"Error reading file '{file_path}': {e}"

    def write(
        self,
        file_path: str,
        content: str,
    ) -> WriteResult:
        """Create a new file with content.
        Returns WriteResult. External storage sets files_update=None.
        """
        resolved_path = self._resolve_path(file_path)

        if resolved_path.exists():
            return WriteResult(
                error=f"Cannot write to {file_path} because it already exists. Read and then make an edit, or write to a new path."
            )

        try:
            # Create parent directories if needed
            resolved_path.parent.mkdir(parents=True, exist_ok=True)

            self._atomic_write(resolved_path, content, mode="w")
            return WriteResult(path=file_path, files_update=None)
        except (OSError, UnicodeEncodeError) as e:
            return WriteResult(error=f"Error writing file '{file_path}': {e}")

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
        expected_mtime: int | None = None,
    ) -> EditResult:
        """Edit a file by replacing string occurrences.

        Args:
            file_path: Path to the file to edit.
            old_string: Exact string to search for and replace.
            new_string: Replacement string.
            replace_all: If True, replace all occurrences.
            expected_mtime: Optional st_mtime_ns token from a previous read.
                If the file's current mtime differs, the edit is rejected as stale.

        Returns:
            EditResult. External storage sets files_update=None.
        """
        resolved_path = self._resolve_path(file_path)

        if not resolved_path.exists() or not resolved_path.is_file():
            return EditResult(error=f"Error: File '{file_path}' not found")

        try:
            with self._file_lock(resolved_path):
                # Read securely
                fd = os.open(resolved_path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
                with os.fdopen(fd, "r", encoding="utf-8") as f:
                    content = f.read()

                if expected_mtime is not None:
                    current_mtime = resolved_path.stat().st_mtime_ns
                    if current_mtime != expected_mtime:
                        return EditResult(
                            error=(
                                f"stale_file: file modified since last read "
                                f"(expected {expected_mtime}, got {current_mtime})"
                            )
                        )

                result = perform_string_replacement(
                    content, old_string, new_string, replace_all
                )

                if isinstance(result, str):
                    return EditResult(error=result)

                new_content, occurrences = result
                self._atomic_write(resolved_path, new_content, mode="w")

                return EditResult(
                    path=file_path, files_update=None, occurrences=int(occurrences)
                )
        except (OSError, UnicodeDecodeError, UnicodeEncodeError) as e:
            return EditResult(error=f"Error editing file '{file_path}': {e}")

    def grep_raw(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
    ) -> list[GrepMatch] | str:
        # Validate regex
        try:
            re.compile(pattern)
        except re.error as e:
            return f"Invalid regex pattern: {e}"

        # Resolve base path
        try:
            base_full = self._resolve_path(path or ".")
        except ValueError:
            return []

        if not base_full.exists():
            return []

        # Try ripgrep first
        results = self._ripgrep_search(pattern, base_full, glob)
        if results is None:
            results = self._python_search(pattern, base_full, glob)

        matches: list[GrepMatch] = []
        for fpath, items in results.items():
            for line_num, line_text in items:
                matches.append(
                    {"path": fpath, "line": int(line_num), "text": line_text}
                )
        return matches

    def _ripgrep_search(
        self, pattern: str, base_full: Path, include_glob: str | None
    ) -> dict[str, list[tuple[int, str]]] | None:
        cmd = ["rg", "--json"]
        if include_glob:
            cmd.extend(["--glob", include_glob])
        cmd.extend(["--", pattern, str(base_full)])

        try:
            proc = subprocess.run(  # noqa: S603
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return None

        results: dict[str, list[tuple[int, str]]] = {}
        for line in proc.stdout.splitlines():
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if data.get("type") != "match":
                continue
            pdata = data.get("data", {})
            ftext = pdata.get("path", {}).get("text")
            if not ftext:
                continue
            p = Path(ftext)
            if self.virtual_mode:
                try:
                    virt = "/" + str(p.resolve().relative_to(self.cwd))
                except Exception:
                    continue
            else:
                virt = str(p)
            ln = pdata.get("line_number")
            lt = pdata.get("lines", {}).get("text", "").rstrip("\n")
            if ln is None:
                continue
            results.setdefault(virt, []).append((int(ln), lt))

        return results

    def _python_search(
        self, pattern: str, base_full: Path, include_glob: str | None
    ) -> dict[str, list[tuple[int, str]]]:
        try:
            regex = re.compile(pattern)
        except re.error:
            return {}

        results: dict[str, list[tuple[int, str]]] = {}
        root = base_full if base_full.is_dir() else base_full.parent

        for fp in root.rglob("*"):
            if not fp.is_file():
                continue
            if include_glob and not wcglob.globmatch(
                fp.name, include_glob, flags=wcglob.BRACE
            ):
                continue
            try:
                if fp.stat().st_size > self.max_file_size_bytes:
                    continue
            except OSError:
                continue
            try:
                content = fp.read_text()
            except (UnicodeDecodeError, PermissionError, OSError):
                continue
            for line_num, line in enumerate(content.splitlines(), 1):
                if regex.search(line):
                    if self.virtual_mode:
                        try:
                            virt_path = "/" + str(fp.resolve().relative_to(self.cwd))
                        except Exception:
                            continue
                    else:
                        virt_path = str(fp)
                    results.setdefault(virt_path, []).append((line_num, line))

        return results

    def glob_info(self, pattern: str, path: str = "/") -> list[FileInfo]:
        if pattern.startswith("/"):
            pattern = pattern.lstrip("/")

        search_path = self.cwd if path == "/" else self._resolve_path(path)
        if not search_path.exists() or not search_path.is_dir():
            return []

        results: list[FileInfo] = []
        try:
            # Use recursive globbing to match files in subdirectories as tests expect
            for matched_path in search_path.rglob(pattern):
                try:
                    is_file = matched_path.is_file()
                except OSError:
                    continue
                if not is_file:
                    continue
                abs_path = str(matched_path)
                if not self.virtual_mode:
                    try:
                        st = matched_path.stat()
                        results.append(
                            {
                                "path": abs_path,
                                "is_dir": False,
                                "size": int(st.st_size),
                                "modified_at": datetime.fromtimestamp(
                                    st.st_mtime
                                ).isoformat(),
                            }
                        )
                    except OSError:
                        results.append({"path": abs_path, "is_dir": False})
                else:
                    cwd_str = str(self.cwd)
                    if not cwd_str.endswith("/"):
                        cwd_str += "/"
                    if abs_path.startswith(cwd_str):
                        relative_path = abs_path[len(cwd_str) :]
                    elif abs_path.startswith(str(self.cwd)):
                        relative_path = abs_path[len(str(self.cwd)) :].lstrip("/")
                    else:
                        relative_path = abs_path
                    virt = "/" + relative_path
                    try:
                        st = matched_path.stat()
                        results.append(
                            {
                                "path": virt,
                                "is_dir": False,
                                "size": int(st.st_size),
                                "modified_at": datetime.fromtimestamp(
                                    st.st_mtime
                                ).isoformat(),
                            }
                        )
                    except OSError:
                        results.append({"path": virt, "is_dir": False})
        except (OSError, ValueError):
            pass

        results.sort(key=lambda x: x.get("path", ""))
        return results

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """Upload multiple files to the filesystem.

        Args:
            files: List of (path, content) tuples where content is bytes.

        Returns:
            List of FileUploadResponse objects, one per input file.
            Response order matches input order.
        """
        responses: list[FileUploadResponse] = []
        for path, content in files:
            try:
                resolved_path = self._resolve_path(path)

                # Create parent directories if needed
                resolved_path.parent.mkdir(parents=True, exist_ok=True)

                self._atomic_write(resolved_path, content, mode="wb")

                responses.append(FileUploadResponse(path=path, error=None))
            except FileNotFoundError:
                responses.append(FileUploadResponse(path=path, error="file_not_found"))
            except PermissionError:
                responses.append(
                    FileUploadResponse(path=path, error="permission_denied")
                )
            except (ValueError, OSError) as e:
                # ValueError from _resolve_path for path traversal, OSError for other file errors
                if isinstance(e, ValueError) or "invalid" in str(e).lower():
                    responses.append(
                        FileUploadResponse(path=path, error="invalid_path")
                    )
                else:
                    # Generic error fallback
                    responses.append(
                        FileUploadResponse(path=path, error="invalid_path")
                    )

        return responses

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Download multiple files from the filesystem.

        Args:
            paths: List of file paths to download.

        Returns:
            List of FileDownloadResponse objects, one per input path.
        """
        responses: list[FileDownloadResponse] = []
        for path in paths:
            try:
                resolved_path = self._resolve_path(path)
                # Use flags to optionally prevent symlink following if
                # supported by the OS
                fd = os.open(resolved_path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
                with os.fdopen(fd, "rb") as f:
                    content = f.read()
                responses.append(
                    FileDownloadResponse(path=path, content=content, error=None)
                )
            except FileNotFoundError:
                responses.append(
                    FileDownloadResponse(
                        path=path, content=None, error="file_not_found"
                    )
                )
            except PermissionError:
                responses.append(
                    FileDownloadResponse(
                        path=path, content=None, error="permission_denied"
                    )
                )
            except IsADirectoryError:
                responses.append(
                    FileDownloadResponse(path=path, content=None, error="is_directory")
                )
            except ValueError:
                responses.append(
                    FileDownloadResponse(path=path, content=None, error="invalid_path")
                )
            # Let other errors propagate
        return responses
