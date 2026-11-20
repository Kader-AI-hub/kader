"""
Core types and dataclasses for the Memory module.

Provides common types used across state, session, and conversation management.
"""

import asyncio
import errno
import json
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any


class SessionType(str, Enum):
    """Enumeration of session types."""

    AGENT = "AGENT"
    MULTI_AGENT = "MULTI_AGENT"


def get_default_memory_dir() -> Path:
    """Get the default memory directory path ($HOME/.kader/memory)."""
    home = Path.home()
    return home / ".kader" / "memory"


@dataclass
class MemoryConfig:
    """Configuration for memory management.

    Attributes:
        memory_dir: Root directory for memory storage
        auto_save: Whether to automatically save state changes
        max_conversation_length: Maximum messages to keep in conversation history
    """

    memory_dir: Path = field(default_factory=get_default_memory_dir)
    auto_save: bool = True
    max_conversation_length: int = 100

    def __post_init__(self) -> None:
        """Ensure memory_dir is a Path object."""
        if isinstance(self.memory_dir, str):
            self.memory_dir = Path(self.memory_dir)

    def ensure_directories(self) -> None:
        """Create memory directories if they don't exist."""
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        (self.memory_dir / "sessions").mkdir(exist_ok=True)


def encode_bytes_values(obj: Any) -> Any:
    """Recursively encode any bytes values in an object to base64.

    Handles dictionaries, lists, and nested structures.
    """
    import base64

    if isinstance(obj, bytes):
        return {"__bytes_encoded__": True, "data": base64.b64encode(obj).decode()}
    elif isinstance(obj, dict):
        return {k: encode_bytes_values(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [encode_bytes_values(item) for item in obj]
    else:
        return obj


def decode_bytes_values(obj: Any) -> Any:
    """Recursively decode any base64-encoded bytes values in an object.

    Handles dictionaries, lists, and nested structures.
    """
    import base64

    if isinstance(obj, dict):
        if obj.get("__bytes_encoded__") is True and "data" in obj:
            return base64.b64decode(obj["data"])
        return {k: decode_bytes_values(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [decode_bytes_values(item) for item in obj]
    else:
        return obj


def get_timestamp() -> str:
    """Get current UTC timestamp in ISO format."""
    return datetime.now(timezone.utc).isoformat()


class DurabilityUnconfirmedError(OSError):
    """os.replace() succeeded but the containing directory could not be synced.

    The destination may already hold the NEW content, so this must not be
    treated as an ordinary failed write that is safe to retry blindly. Re-read
    the stored file before deciding whether to write again.
    """

    status = "REPLACED_DURABILITY_UNCONFIRMED"
    #: Set by a revision-checked owner after re-reading the stored file: True
    #: if this write is currently visible (still NOT proven crash-durable),
    #: False if the stored file is not this write, None if not re-read.
    write_visible: bool | None = None


_UNSUPPORTED_DIR_FSYNC_ERRNOS = frozenset(
    {
        errno.EINVAL,
        errno.ENOTSUP,
        getattr(errno, "EOPNOTSUPP", errno.ENOTSUP),
        getattr(errno, "ENOSYS", errno.EINVAL),
    }
)


def _fsync_directory(directory: Path) -> None:
    """Flush a directory entry table so a completed rename survives a crash.

    Best-effort by contract: Windows cannot open a directory as a file
    descriptor, and some filesystems report directory fsync as unsupported.
    In both cases this returns without error and durability stays
    best-effort. Any other failure is raised.
    """
    if os.name == "nt":
        return
    try:
        dir_fd = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except OSError as exc:
        if exc.errno in _UNSUPPORTED_DIR_FSYNC_ERRNOS:
            return
        raise
    try:
        os.fsync(dir_fd)
    except OSError as exc:
        if exc.errno not in _UNSUPPORTED_DIR_FSYNC_ERRNOS:
            raise
    finally:
        os.close(dir_fd)


def atomic_write_text(path: Path, content: str, encoding: str = "utf-8") -> None:
    """Atomically replace ``path`` with ``content``.

    Writes to a unique sibling temporary file opened with O_EXCL, flushes and
    fsyncs it, then os.replace()s it over the destination, so a reader (or a
    process restarted after a crash) sees either the old or the new content,
    never a partial file. The containing directory is then synced
    (best-effort, see ``_fsync_directory``).

    Raises:
        DurabilityUnconfirmedError: the replacement happened but the
            directory sync failed; the destination may hold the new content.
        OSError: the write failed before the replacement; the destination is
            unchanged.
    """
    tmp_path = path.with_name(f"{path.name}.{uuid.uuid4().hex}.kader-tmp")
    try:
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_BINARY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        fd = os.open(tmp_path, flags, 0o644)
        try:
            fobj = os.fdopen(fd, "w", encoding=encoding, newline="")
        except Exception:
            # fdopen failed, so the raw descriptor is still ours to close.
            try:
                os.close(fd)
            except OSError:
                pass
            raise
        with fobj as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())

        # Windows can refuse a replace while another process briefly holds the
        # destination open (sharing/lock violation); retry those only.
        transient_windows_errors = frozenset({5, 32, 33})
        delays = (0.005, 0.010, 0.020, 0.040)
        for attempt in range(len(delays) + 1):
            try:
                os.replace(tmp_path, path)
                break
            except OSError as exc:
                retryable = (
                    os.name == "nt"
                    and getattr(exc, "winerror", None) in transient_windows_errors
                )
                if not retryable or attempt == len(delays):
                    raise
                time.sleep(delays[attempt])

        # The rename is only durable once the directory entry itself is
        # flushed, so the directory is synced AFTER os.replace().
        try:
            _fsync_directory(path.parent)
        except OSError as exc:
            raise DurabilityUnconfirmedError(
                exc.errno,
                f"{DurabilityUnconfirmedError.status}: '{path.name}' now holds "
                f"the new content, but the containing directory could not be "
                f"synced ({exc.strerror or exc}); do not retry blindly",
            ) from exc
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass


def dump_json_text(data: dict[str, Any]) -> str:
    """Serialize data (bytes values base64-encoded) the way save_json does."""
    return json.dumps(encode_bytes_values(data), indent=2, ensure_ascii=False)


def save_json(path: Path, data: dict[str, Any]) -> None:
    """Atomically save data to a JSON file.

    The file is replaced as a whole (see ``atomic_write_text``): an
    interrupted save leaves the previous content in place.

    Args:
        path: Path to the JSON file
        data: Data to save

    Raises:
        DurabilityUnconfirmedError: replaced, but the directory sync failed.
        OSError: the save failed and the previous content is unchanged.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, dump_json_text(data))


def load_json(path: Path) -> dict[str, Any]:
    """Load data from a JSON file.

    Args:
        path: Path to the JSON file

    Returns:
        Loaded data, or empty dict if file doesn't exist
    """
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return decode_bytes_values(json.load(f))


# --- Async File I/O Utilities ---


async def aread_text(path: Path, encoding: str = "utf-8") -> str:
    """Asynchronously read text from a file.

    Args:
        path: Path to the file
        encoding: File encoding (default: utf-8)

    Returns:
        File contents as string
    """
    import aiofiles

    async with aiofiles.open(path, "r", encoding=encoding) as f:
        return await f.read()


async def awrite_text(path: Path, content: str, encoding: str = "utf-8") -> None:
    """Asynchronously write text to a file.

    Args:
        path: Path to the file
        content: Text content to write
        encoding: File encoding (default: utf-8)
    """
    import aiofiles

    path.parent.mkdir(parents=True, exist_ok=True)
    async with aiofiles.open(path, "w", encoding=encoding) as f:
        await f.write(content)


async def asave_json(path: Path, data: dict[str, Any]) -> None:
    """Asynchronously and atomically save data to a JSON file.

    Runs ``save_json`` in a worker thread, so the async path gets the same
    atomic-replace guarantee as the sync path.

    Args:
        path: Path to the JSON file
        data: Data to save
    """
    await asyncio.to_thread(save_json, path, data)


async def aload_json(path: Path) -> dict[str, Any]:
    """Asynchronously load data from a JSON file.

    Args:
        path: Path to the JSON file

    Returns:
        Loaded data, or empty dict if file doesn't exist
    """
    if not path.exists():
        return {}
    content = await aread_text(path)
    return decode_bytes_values(json.loads(content))
