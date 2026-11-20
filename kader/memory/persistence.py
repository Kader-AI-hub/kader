"""
Revision-checked persistence for conversation history files.

A history file is a JSON object that carries a ``revision`` counter and the
``write_id`` of the save that produced it. ``RevisionedJsonFile`` is the one
owner of a history file inside a process. It remembers the exact stored
identity it last read or wrote, and every save compares that identity with
the stored file and writes under one cross-process lock.

Admission rules (fail closed):

* A missing file is admitted as a new history (revision 0) only if this
  owner has never observed the file as present. A history that was observed
  and has since disappeared is never silently recreated.
* A file that is present but unreadable, not UTF-8, not JSON, not a JSON
  object, or carries an invalid revision is refused and never overwritten.
* An existing history is identified by ``(revision, write_id, sha256)``,
  where the digest covers the exact stored bytes. A file restored from an
  older snapshot that happens to carry the same revision number therefore
  does not match (no ABA). Legacy files without revision metadata are
  ``(0, None, sha256)``: two different legacy histories never compare equal,
  and the first save upgrades the file to revision 1 in one transition.
* A conflict clears this owner's standing: it must ``load()`` again.

When a save cannot confirm durability (``DurabilityUnconfirmedError``), the
stored file is re-read under the same lock before anything else happens. If
the stored ``write_id`` is this save's, the write is visible and the owner
advances, but the error is still raised with ``write_visible = True``:
visible is not the same as crash-durable, and the save is not recorded as
clean, so an unchanged follow-up save is written again (revision-checked).
Otherwise the owner's standing is cleared. Nothing here retries on its own.
"""

from __future__ import annotations

import contextlib
import errno
import hashlib
import json
import math
import os
import re
import stat
import threading
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import types as _types
from .types import DurabilityUnconfirmedError, decode_bytes_values

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows
    fcntl = None  # type: ignore[assignment]

try:
    import msvcrt
except ImportError:  # pragma: no cover - POSIX
    msvcrt = None  # type: ignore[assignment]


REVISION_KEY = "revision"
WRITE_ID_KEY = "write_id"
_METADATA_KEYS = (REVISION_KEY, WRITE_ID_KEY)
_WRITE_ID_RE = re.compile(r"[0-9a-f]{32}")  # uuid4().hex


class RevisionConflictError(RuntimeError):
    """A save was refused because the stored history is not the one observed.

    Nothing was written. The owner's standing has been cleared; call
    ``load()`` before saving again.
    """

    def __init__(self, path: Path, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"Refusing to overwrite '{path.name}': {reason}")


class LockTimeoutError(TimeoutError):
    """The history lock was not acquired before the deadline."""


class CorruptHistoryError(ValueError):
    """A history file is present but is not an admissible JSON history."""


@dataclass(frozen=True)
class StoredIdentity:
    """Identity of a stored history file at the moment it was read."""

    revision: int
    write_id: str | None
    digest: str

    def describe(self) -> str:
        return f"revision {self.revision} (write_id={self.write_id})"


def _refuse_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """json object hook: a duplicated key is ambiguous, so refuse it.

    The default decoder silently keeps the last value, which would let a
    history carry two different revisions.
    """
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CorruptHistoryError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _refuse_constant(name: str) -> Any:
    """json constant hook: NaN/Infinity are not JSON; refuse them."""
    raise CorruptHistoryError(f"non-standard JSON constant {name}")


def _read_stored(path: Path) -> tuple[StoredIdentity, dict[str, Any]] | None:
    """Read a history file. ``None`` means positively absent.

    Raises:
        OSError: present but unreadable.
        CorruptHistoryError: present but not an admissible history.
    """
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        return None
    try:
        data = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_refuse_duplicates,
            parse_constant=_refuse_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CorruptHistoryError(f"'{path.name}' is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise CorruptHistoryError(f"'{path.name}' is not a JSON object")
    # Closed metadata grammar: legacy (neither key) or revisioned (a positive
    # int revision AND a canonical write_id). Anything else is refused.
    has_revision, has_write_id = REVISION_KEY in data, WRITE_ID_KEY in data
    if not has_revision and not has_write_id:
        revision, write_id = 0, None
    elif has_revision and has_write_id:
        revision, write_id = data[REVISION_KEY], data[WRITE_ID_KEY]
        if isinstance(revision, bool) or not isinstance(revision, int):
            raise CorruptHistoryError(
                f"'{path.name}' has invalid revision {revision!r}"
            )
        if revision < 1:
            raise CorruptHistoryError(
                f"'{path.name}' has non-positive revision {revision!r}"
            )
        if not isinstance(write_id, str) or not _WRITE_ID_RE.fullmatch(write_id):
            raise CorruptHistoryError(
                f"'{path.name}' has invalid write_id {write_id!r}"
            )
    else:
        raise CorruptHistoryError(
            f"'{path.name}' has mixed revision metadata (needs both or neither)"
        )
    identity = StoredIdentity(revision, write_id, hashlib.sha256(raw).hexdigest())
    return identity, decode_bytes_values(data)


_thread_locks_guard = threading.Lock()
_thread_locks: dict[str, threading.RLock] = {}

#: Upper bound on waiting for a history lock (thread and OS lock together).
LOCK_TIMEOUT_SECONDS = 30.0
_LOCK_POLL_SECONDS = 0.005
# Errors meaning "another holder has it", per adapter; retried until the
# deadline. Everything else (EBADF, ENOLCK, EDEADLOCK, EIO, ...) is raised at
# once.
# - POSIX flock(LOCK_NB): EWOULDBLOCK (== EAGAIN on Linux and macOS).
# - Windows msvcrt.locking(LK_NBLCK): observed as PermissionError with
#   errno EACCES and winerror None (CPython 3.13, Windows 10.0.26300);
#   the CRT maps the lock violation to EACCES.
_POSIX_CONTENTION_ERRNOS = frozenset({errno.EAGAIN, errno.EWOULDBLOCK})
_WINDOWS_CONTENTION_ERRNOS = frozenset({errno.EACCES})


def _try_os_lock(fd: int) -> bool:
    """Try once to take the exclusive OS lock. False means contended."""
    if fcntl is not None:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno in _POSIX_CONTENTION_ERRNOS:
                return False
            raise
        return True
    if msvcrt is not None:
        # Windows permits locking a byte past end-of-file, so the lock file
        # is never written. msvcrt locks at the current position.
        os.lseek(fd, 0, os.SEEK_SET)
        try:
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            if exc.errno in _WINDOWS_CONTENTION_ERRNOS:
                return False
            raise
        return True
    raise RuntimeError(  # pragma: no cover - no supported locking primitive
        "Cross-process file locking is unsupported here"
    )


def _os_unlock(fd: int) -> None:
    if fcntl is not None:
        fcntl.flock(fd, fcntl.LOCK_UN)
    elif msvcrt is not None:
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)


@contextlib.contextmanager
def file_lock(path: Path, timeout: float | None = None) -> Iterator[None]:
    """Hold an exclusive lock for ``path`` across threads and processes.

    Uses a stable sibling lock file (``.<name>.kader-lock``) that is never
    deleted or replaced, so every process locks the same file while the
    target itself is replaced atomically: fcntl.flock on POSIX,
    msvcrt.locking on byte 0 on Windows. The lock file is only created
    (O_CREAT), never written, so concurrent first use by several processes
    is safe. A process-local lock is held as well, because those OS locks do
    not exclude threads that share a process.

    ``timeout`` (default ``LOCK_TIMEOUT_SECONDS``) is one acquisition budget
    for the process-local lock and the OS lock together; it must be finite
    and >= 0. ``timeout=0`` makes exactly one non-blocking attempt at each.
    With a positive budget, no OS attempt starts after the deadline. Only
    contention errors are retried (polling every few milliseconds); any other
    locking error is raised at once. The budget bounds waiting, not the
    filesystem calls themselves.

    If the protected block raises and releasing the OS lock also fails, the
    block's exception propagates with the release failure attached as a
    note; if only the release fails, that failure is raised. The descriptor
    and the process-local lock are released on every path.
    """
    limit = LOCK_TIMEOUT_SECONDS if timeout is None else float(timeout)
    if not math.isfinite(limit) or limit < 0:
        raise ValueError(f"timeout must be finite and >= 0, got {timeout!r}")
    deadline = time.monotonic() + limit
    timeout_error = f"Timed out after {limit}s waiting for '{path.name}'"
    key = os.path.normcase(os.path.abspath(path))
    with _thread_locks_guard:
        thread_lock = _thread_locks.setdefault(key, threading.RLock())
    acquired = (
        thread_lock.acquire(blocking=False)
        if limit == 0
        else thread_lock.acquire(timeout=limit)
    )
    if not acquired:
        raise LockTimeoutError(timeout_error)
    try:
        if limit > 0 and time.monotonic() >= deadline:
            raise LockTimeoutError(timeout_error)
        path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = path.with_name(f".{path.name}.kader-lock")
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_BINARY", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
        fd = os.open(lock_path, flags, 0o600)
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise RuntimeError(f"Lock path is not a regular file: {lock_path}")
            while not _try_os_lock(fd):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise LockTimeoutError(timeout_error)
                time.sleep(min(_LOCK_POLL_SECONDS, remaining))
                if time.monotonic() >= deadline:
                    raise LockTimeoutError(timeout_error)
            body_error: BaseException | None = None
            try:
                yield
            except BaseException as exc:
                body_error = exc
                raise
            finally:
                try:
                    _os_unlock(fd)
                except OSError as unlock_error:
                    if body_error is None:
                        raise
                    # Keep the operation's failure; record the release one.
                    # Closing the descriptor below still drops the OS lock.
                    body_error.add_note(f"lock release also failed: {unlock_error!r}")
        finally:
            os.close(fd)
    finally:
        thread_lock.release()


_ABSENT = "absent"


class RevisionedJsonFile:
    """Owner of one revision-checked JSON history file.

    The target path is canonicalized at construction and cannot be changed.
    All state changes happen under the per-path lock, so concurrent use of
    one instance is serialized.
    """

    def __init__(self, path: Path) -> None:
        self._path = Path(os.path.abspath(path))
        # None: no standing (never observed, or invalidated by a conflict).
        # _ABSENT: observed as never-created. StoredIdentity: observed present.
        self._known: StoredIdentity | str | None = None
        # True once this owner has seen the file present; a later absence is
        # then a disappearance, never a new history.
        self._seen_present = False
        # JSON text of the last save confirmed durable (or as durable as the
        # platform reports), used only to skip unchanged saves.
        self._clean_text: str | None = None
        self.durability_confirmed = True
        #: True when the last save() checked the stored file under the lock;
        #: False when it took the unchanged fast path, whose returned revision
        #: is cached local standing, NOT a freshly verified stored revision.
        self.storage_verified = False

    @property
    def path(self) -> Path:
        return self._path

    @property
    def revision(self) -> int | None:
        """Revision of the stored file this owner stands on, if any."""
        if self._known is None:
            return None
        if self._known == _ABSENT:
            return 0
        return self._known.revision  # type: ignore[union-attr]

    def load(self) -> dict[str, Any]:
        """Read the file and take standing on exactly what was read.

        Returns:
            The stored data without revision metadata ({} if absent).

        Raises:
            CorruptHistoryError, OSError: the file is present but not an
                admissible history; standing is cleared.
            RevisionConflictError: a previously observed history is gone.
        """
        with file_lock(self._path):
            self._known = None
            self._clean_text = None
            stored = _read_stored(self._path)
            if stored is None:
                if self._seen_present:
                    raise RevisionConflictError(
                        self._path, "a previously observed history disappeared"
                    )
                self._known = _ABSENT
                return {}
            identity, data = stored
            self._known = identity
            self._seen_present = True
            return {k: v for k, v in data.items() if k not in _METADATA_KEYS}

    def save(self, data: dict[str, Any]) -> int:
        """Compare-and-write ``data`` as the next revision.

        ``data`` is snapshotted to JSON text once, at entry, so the file holds
        exactly one serialization; it must be JSON-compatible (bytes values
        are base64-encoded). Taking a coherent snapshot of live messages is
        the conversation manager's job; this method does not lock the
        caller's objects, so they must not be mutated concurrently.

        Returns:
            The new revision. If ``data`` equals this owner's last clean save,
            nothing is read or written and the CACHED revision is returned
            with ``storage_verified = False``: the content is unchanged
            locally, but the stored file was not checked. Anything that needs
            current storage standing must ``load()`` or save a change.

        Raises:
            RevisionConflictError: the stored file is not the one this owner
                last observed; nothing was written and standing is cleared.
            CorruptHistoryError, OSError (before replacement): nothing was
                written.
            DurabilityUnconfirmedError: the file was replaced but the
                directory sync failed; see the module docstring.
        """
        payload = {k: v for k, v in data.items() if k not in _METADATA_KEYS}
        # Strict JSON only (no NaN/Infinity), so the reader can admit it later.
        snapshot = json.loads(
            json.dumps(_types.encode_bytes_values(payload), allow_nan=False)
        )

        with file_lock(self._path):
            snapshot_text = json.dumps(snapshot, sort_keys=True)
            if (
                self._clean_text is not None
                and snapshot_text == self._clean_text
                and isinstance(self._known, StoredIdentity)
            ):
                self.storage_verified = False
                return self._known.revision

            self.storage_verified = True
            self._admit()
            base = 0 if self._known == _ABSENT else self._known.revision  # type: ignore[union-attr]
            new_revision = base + 1
            write_id = uuid.uuid4().hex
            text = _types.dump_json_text(
                {**snapshot, REVISION_KEY: new_revision, WRITE_ID_KEY: write_id}
            )
            new_identity = StoredIdentity(
                new_revision, write_id, hashlib.sha256(text.encode("utf-8")).hexdigest()
            )
            self._clean_text = None
            try:
                _types.atomic_write_text(self._path, text)
            except DurabilityUnconfirmedError as exc:
                self.durability_confirmed = False
                exc.write_visible = self._reconcile(new_identity)  # type: ignore[attr-defined]
                raise
            self._known = new_identity
            self._seen_present = True
            self._clean_text = snapshot_text
            self.durability_confirmed = True
            return new_revision

    def _admit(self) -> None:
        """Check the stored file against this owner's standing (under lock)."""
        try:
            stored = _read_stored(self._path)
        except (OSError, CorruptHistoryError):
            self._known = None
            raise
        known = self._known
        if stored is None:
            if known is None and not self._seen_present:
                self._known = _ABSENT  # never-created history: admit
                return
            if known == _ABSENT:
                return
            self._known = None
            raise RevisionConflictError(
                self._path, "a previously observed history disappeared"
            )
        identity, _ = stored
        if known is None:
            raise RevisionConflictError(
                self._path, f"stored {identity.describe()} was never loaded"
            )
        if known == _ABSENT:
            self._known = None
            raise RevisionConflictError(
                self._path, f"history was created meanwhile: {identity.describe()}"
            )
        if identity != known:
            self._known = None
            raise RevisionConflictError(
                self._path,
                f"expected {known.describe()}, stored is {identity.describe()}",  # type: ignore[union-attr]
            )

    def _reconcile(self, new_identity: StoredIdentity) -> bool:
        """Re-read the stored file after an unconfirmed-durability save."""
        try:
            stored = _read_stored(self._path)
        except (OSError, CorruptHistoryError):
            stored = None
        if stored is not None and stored[0] == new_identity:
            self._known = new_identity
            self._seen_present = True
            return True
        self._known = None
        return False
