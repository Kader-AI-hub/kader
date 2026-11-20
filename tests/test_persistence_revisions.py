"""Tests for atomic, revision-checked conversation persistence."""

import contextlib
import errno
import json
import os
import re
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from kader.memory import (
    AsyncFileSessionManager,
    DurabilityUnconfirmedError,
    FileSessionManager,
    MemoryConfig,
    RevisionConflictError,
    RevisionedJsonFile,
)
from kader.memory import types as memory_types
from kader.memory.persistence import CorruptHistoryError
from kader.memory.types import load_json, save_json
from kader.tools.agent import PersistentSlidingWindowConversationManager

VALID_ID = b"0123456789abcdef0123456789abcdef"  # canonical uuid4().hex shape
MEMORY_DIR = Path(__file__).resolve().parents[1] / "kader" / "memory"

# Subprocesses load kader/memory/{types,persistence}.py as a bare package so
# they do not pay for importing the whole kader package (providers, tools).
SUBPROCESS_PREAMBLE = f"""
import importlib, os, sys, time, types as _t
from pathlib import Path
_pkg = _t.ModuleType("kmem")
_pkg.__path__ = [{str(MEMORY_DIR)!r}]
sys.modules["kmem"] = _pkg
types = importlib.import_module("kmem.types")
persistence = importlib.import_module("kmem.persistence")
"""


def _run_python(body: str, *args: str) -> subprocess.CompletedProcess:
    script = SUBPROCESS_PREAMBLE + textwrap.dedent(body)
    return subprocess.run(
        [sys.executable, "-c", script, *args],
        capture_output=True,
        text=True,
        timeout=120,
    )


@pytest.fixture(autouse=True)
def _no_leaked_threads():
    """Fail any test here that leaves a thread of its own running."""
    before = set(threading.enumerate())
    yield
    leaked = [
        t
        for t in threading.enumerate()
        if t not in before and t.is_alive() and not t.name.startswith("asyncio")
    ]
    for thread in leaked:
        thread.join(5)
    still_alive = [t.name for t in leaked if t.is_alive()]
    assert still_alive == [], f"test left threads running: {still_alive}"


@contextlib.contextmanager
def _reaped_children(cleanup_timeout: float = 10.0):
    """Yield a list to register child processes in; reap them all on exit.

    Every registered child gets its own bounded kill-and-reap attempt, so one
    failure cannot skip the others. Incomplete cleanup is a verification
    failure: raised as RuntimeError if the block succeeded, or attached as a
    note to the block's own failure, which keeps precedence.
    """
    procs: list[subprocess.Popen] = []
    body_error: BaseException | None = None
    try:
        yield procs
    except BaseException as exc:
        body_error = exc
        raise
    finally:
        cleanup_errors: list[str] = []
        for proc in procs:
            try:
                if proc.poll() is None:
                    proc.kill()
                proc.communicate(timeout=cleanup_timeout)
            except subprocess.TimeoutExpired:
                cleanup_errors.append(f"PID {proc.pid}: cleanup timed out")
            except OSError as exc:
                cleanup_errors.append(f"PID {proc.pid}: {type(exc).__name__}: {exc}")
        if cleanup_errors:
            message = "Subprocess cleanup incomplete: " + "; ".join(cleanup_errors)
            if body_error is None:
                raise RuntimeError(message)
            body_error.add_note(message)


def _leftover_temp_files(directory: Path) -> list[Path]:
    return sorted(directory.glob("*.kader-tmp"))


def _failing_dir_sync(directory: Path) -> None:
    raise OSError(errno.EIO, "simulated directory fsync failure")


# --- Atomic save_json -------------------------------------------------------


def test_save_json_replaces_whole_file_and_leaves_no_temp(tmp_path: Path) -> None:
    target = tmp_path / "conversation.json"
    save_json(target, {"messages": ["old"]})
    save_json(target, {"messages": ["new"]})

    assert load_json(target) == {"messages": ["new"]}
    assert _leftover_temp_files(tmp_path) == []


def test_failed_write_before_replace_keeps_old_content(tmp_path: Path) -> None:
    target = tmp_path / "conversation.json"
    save_json(target, {"messages": ["old"]})

    with patch.object(memory_types.os, "replace", side_effect=OSError("boom")):
        with pytest.raises(OSError, match="boom"):
            save_json(target, {"messages": ["new"]})

    assert load_json(target) == {"messages": ["old"]}
    assert _leftover_temp_files(tmp_path) == []


@pytest.mark.parametrize("crash_at", ["fsync", "replace"])
def test_simulated_crash_leaves_no_partial_file(tmp_path: Path, crash_at: str) -> None:
    """A process killed mid-save leaves the previous, complete file in place."""
    target = tmp_path / "conversation.json"
    old = {"messages": [{"role": "user", "content": "old " * 200}]}
    save_json(target, old)

    result = _run_python(
        f"""
        def crash(*args, **kwargs):
            os._exit(17)  # no cleanup, no finally blocks: a hard crash

        setattr(types.os, {crash_at!r}, crash)
        big = {{"messages": [{{"role": "user", "content": "new " * 50000}}]}}
        types.save_json(Path(sys.argv[1]), big)
        """,
        str(target),
    )
    assert result.returncode == 17, result.stderr

    # The destination is still the complete old document, never a prefix of
    # the new one. A crash may strand the uniquely named temp file, which no
    # reader ever opens.
    assert json.loads(target.read_text(encoding="utf-8")) == old
    stranded = _leftover_temp_files(tmp_path)
    assert all(p.name.startswith("conversation.json.") for p in stranded)


# --- Admission: absent, valid, legacy, malformed, unreadable ---------------


def test_missing_history_is_admitted_as_revision_zero(tmp_path: Path) -> None:
    owner = RevisionedJsonFile(tmp_path / "conversation.json")
    assert owner.load() == {}
    assert owner.revision == 0
    assert owner.save({"messages": ["first"]}) == 1


def test_existing_valid_history_loads_its_revision(tmp_path: Path) -> None:
    target = tmp_path / "conversation.json"
    writer = RevisionedJsonFile(target)
    writer.save({"messages": ["a"]})
    writer.save({"messages": ["a", "b"]})

    reader = RevisionedJsonFile(target)
    assert reader.load() == {"messages": ["a", "b"]}
    assert reader.revision == 2


@pytest.mark.parametrize(
    "content",
    [
        b"",
        b'{"messages": ["trunc',
        b"\xff\xfe not utf-8",
        b'["a", "list"]',
        b'{"messages": [], "revision": -1, "write_id": "' + VALID_ID + b'"}',
        b'{"messages": [], "revision": 0, "write_id": "' + VALID_ID + b'"}',
        b'{"messages": [], "revision": "3", "write_id": "' + VALID_ID + b'"}',
        b'{"messages": [], "revision": true, "write_id": "' + VALID_ID + b'"}',
        b'{"messages": [], "revision": 2}',
        b'{"messages": [], "write_id": "' + VALID_ID + b'"}',
        b'{"messages": [], "revision": 2, "write_id": ""}',
        b'{"messages": [], "revision": 2, "write_id": "not-a-uuid-hex"}',
        b'{"messages": [], "revision": 2, "write_id": 7}',
        b'{"messages": [], "revision": 2, "write_id": "' + VALID_ID.upper() + b'"}',
        b'{"messages": [], "revision": 1, "revision": 2, "write_id": "'
        + VALID_ID
        + b'"}',
        b'{"messages": [{"role": "user", "role": "system"}]}',
        b'{"messages": [{"score": NaN}]}',
        b'{"messages": [{"score": -Infinity}]}',
    ],
    ids=[
        "empty",
        "truncated",
        "not-utf8",
        "not-object",
        "neg-rev",
        "zero-rev-with-id",
        "str-rev",
        "bool-rev",
        "rev-without-id",
        "id-without-rev",
        "empty-id",
        "non-canonical-id",
        "int-id",
        "uppercase-id",
        "duplicate-revision-key",
        "duplicate-nested-key",
        "nan-constant",
        "infinity-constant",
    ],
)
def test_malformed_history_is_refused_and_never_overwritten(
    tmp_path: Path, content: bytes
) -> None:
    target = tmp_path / "conversation.json"
    target.write_bytes(content)

    owner = RevisionedJsonFile(target)
    with pytest.raises(CorruptHistoryError):
        owner.load()
    assert owner.revision is None
    with pytest.raises(CorruptHistoryError):
        owner.save({"messages": ["replacement"]})
    assert target.read_bytes() == content


def test_unreadable_history_is_refused_and_never_overwritten(tmp_path: Path) -> None:
    target = tmp_path / "conversation.json"
    save_json(target, {"messages": ["kept"]})
    before = target.read_bytes()

    owner = RevisionedJsonFile(target)
    denied = PermissionError(errno.EACCES, "simulated unreadable file")
    with patch.object(Path, "read_bytes", side_effect=denied):
        with pytest.raises(PermissionError):
            owner.load()
        with pytest.raises(PermissionError):
            owner.save({"messages": ["replacement"]})
    assert owner.revision is None
    assert target.read_bytes() == before


def test_legacy_history_is_upgraded_in_one_transition(tmp_path: Path) -> None:
    target = tmp_path / "conversation.json"
    save_json(target, {"messages": ["legacy"]})  # pre-revision format

    owner = RevisionedJsonFile(target)
    assert owner.load() == {"messages": ["legacy"]}
    assert owner.revision == 0
    assert owner.save({"messages": ["legacy", "next"]}) == 1

    stored = load_json(target)
    assert stored["revision"] == 1
    # The writer emits exactly the canonical shape the reader admits.
    assert re.fullmatch(r"[0-9a-f]{32}", stored["write_id"])
    assert re.fullmatch(r"[0-9a-f]{32}", VALID_ID.decode())


def test_two_different_legacy_histories_never_compare_equal(tmp_path: Path) -> None:
    """Both are revision 0 without write_id; the byte digest tells them apart."""
    target = tmp_path / "conversation.json"
    save_json(target, {"messages": ["legacy A"]})
    owner = RevisionedJsonFile(target)
    owner.load()

    save_json(target, {"messages": ["legacy B"]})  # another legacy history
    with pytest.raises(RevisionConflictError):
        owner.save({"messages": ["legacy A", "next"]})
    assert load_json(target) == {"messages": ["legacy B"]}


def test_existing_file_never_read_is_not_overwritten(tmp_path: Path) -> None:
    target = tmp_path / "conversation.json"
    save_json(target, {"messages": ["history"]})

    with pytest.raises(RevisionConflictError):
        RevisionedJsonFile(target).save({"messages": []})
    assert load_json(target) == {"messages": ["history"]}


# --- Identity: stale revisions, ABA, disappearance --------------------------


def test_stale_revision_is_refused_and_requires_reload(tmp_path: Path) -> None:
    target = tmp_path / "conversation.json"
    RevisionedJsonFile(target).save({"messages": ["a"]})

    writer_a = RevisionedJsonFile(target)
    writer_b = RevisionedJsonFile(target)
    writer_a.load()
    writer_b.load()
    assert writer_a.save({"messages": ["a", "b"]}) == 2

    with pytest.raises(RevisionConflictError):
        writer_b.save({"messages": ["a", "stale"]})
    assert load_json(target)["messages"] == ["a", "b"]

    # Standing is cleared: a second attempt is refused until load().
    assert writer_b.revision is None
    with pytest.raises(RevisionConflictError):
        writer_b.save({"messages": ["a", "stale"]})
    assert load_json(target)["messages"] == ["a", "b"]

    assert writer_b.load() == {"messages": ["a", "b"]}
    assert writer_b.save({"messages": ["a", "b", "c"]}) == 3


def test_same_revision_different_write_id_is_a_conflict(tmp_path: Path) -> None:
    """ABA: a restored snapshot with the same revision number is not admitted."""
    target = tmp_path / "conversation.json"
    owner = RevisionedJsonFile(target)
    owner.save({"messages": ["a"]})
    owner.save({"messages": ["a", "b"]})  # revision 2

    stored = load_json(target)
    restored = {"messages": ["x", "y"], "revision": 2, "write_id": "ab" * 16}
    save_json(target, restored)
    assert stored["revision"] == restored["revision"]

    with pytest.raises(RevisionConflictError):
        owner.save({"messages": ["a", "b", "c"]})
    assert load_json(target) == restored


def test_same_revision_and_write_id_but_different_bytes_is_a_conflict(
    tmp_path: Path,
) -> None:
    target = tmp_path / "conversation.json"
    owner = RevisionedJsonFile(target)
    owner.save({"messages": ["a"]})

    tampered = load_json(target)
    tampered["messages"] = ["edited"]
    save_json(target, tampered)  # same revision and write_id, other content

    with pytest.raises(RevisionConflictError):
        owner.save({"messages": ["a", "b"]})
    assert load_json(target) == tampered


def test_previously_observed_history_that_disappears_is_not_recreated(
    tmp_path: Path,
) -> None:
    target = tmp_path / "conversation.json"
    owner = RevisionedJsonFile(target)
    owner.save({"messages": ["a"]})

    target.unlink()  # external deletion
    with pytest.raises(RevisionConflictError, match="disappeared"):
        owner.save({"messages": ["a", "b"]})
    assert not target.exists()
    with pytest.raises(RevisionConflictError, match="disappeared"):
        owner.load()
    assert not target.exists()


def test_history_created_after_observed_absence_is_a_conflict(tmp_path: Path) -> None:
    target = tmp_path / "conversation.json"
    owner = RevisionedJsonFile(target)
    owner.load()  # observed: never created

    RevisionedJsonFile(target).save({"messages": ["someone else"]})
    with pytest.raises(RevisionConflictError):
        owner.save({"messages": ["mine"]})
    assert load_json(target)["messages"] == ["someone else"]


def test_target_path_is_canonical_and_immutable(tmp_path: Path) -> None:
    relative = os.path.relpath(tmp_path / "conversation.json")
    owner = RevisionedJsonFile(Path(relative))
    assert owner.path.is_absolute()
    assert owner.path == Path(os.path.abspath(relative))
    with pytest.raises(AttributeError):
        owner.path = tmp_path / "other.json"  # type: ignore[misc]


# --- Snapshots and no-op saves ---------------------------------------------


def test_unchanged_save_performs_no_write(tmp_path: Path) -> None:
    target = tmp_path / "conversation.json"
    owner = RevisionedJsonFile(target)
    assert owner.save({"messages": ["a"]}) == 1

    with (
        patch.object(memory_types, "atomic_write_text") as write,
        patch("kader.memory.persistence._read_stored") as read,
    ):
        assert owner.save({"messages": ["a"]}) == 1
        assert owner.save({"messages": ["a"]}) == 1
    write.assert_not_called()
    read.assert_not_called()
    assert owner.storage_verified is False  # cached standing, not verified

    assert owner.save({"messages": ["a", "b"]}) == 2
    assert owner.storage_verified is True


def test_unchanged_save_after_external_change_never_overwrites(
    tmp_path: Path,
) -> None:
    """The fast path returns cached standing; it never claims fresh storage."""
    target = tmp_path / "conversation.json"
    owner_a = RevisionedJsonFile(target)
    assert owner_a.save({"messages": ["a"]}) == 1

    owner_b = RevisionedJsonFile(target)
    owner_b.load()
    assert owner_b.save({"messages": ["a", "b"]}) == 2

    # Unchanged locally: cached revision 1, flagged unverified, nothing written.
    assert owner_a.save({"messages": ["a"]}) == 1
    assert owner_a.storage_verified is False
    assert load_json(target)["messages"] == ["a", "b"]

    # A changed save checks storage and is refused.
    with pytest.raises(RevisionConflictError):
        owner_a.save({"messages": ["a", "c"]})
    assert load_json(target)["messages"] == ["a", "b"]


# --- Lock acquisition -------------------------------------------------------


def _lock_primitive():
    """(module, attribute, contention errno) for this platform's OS lock."""
    from kader.memory import persistence

    if persistence.fcntl is not None:
        return persistence.fcntl, "flock", errno.EWOULDBLOCK
    if persistence.msvcrt is not None:
        return persistence.msvcrt, "locking", errno.EACCES
    pytest.skip("no supported OS lock adapter (fcntl or msvcrt) on this platform")


@pytest.fixture
def lock_held_by_other_process(tmp_path: Path):
    """Yield a target path whose history lock another process holds.

    The holder is killed and reaped on every exit path.
    """
    target = tmp_path / "conversation.json"
    ready = tmp_path / "holder.ready"
    script = SUBPROCESS_PREAMBLE + textwrap.dedent(
        """
        with persistence.file_lock(Path(sys.argv[1])):
            Path(sys.argv[2]).write_text("ready")
            time.sleep(60)
        """
    )
    with _reaped_children() as procs:
        procs.append(
            subprocess.Popen(
                [sys.executable, "-c", script, str(target), str(ready)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        )
        holder = procs[0]
        deadline = time.monotonic() + 60
        while not ready.exists():
            assert holder.poll() is None, "holder exited before taking the lock"
            assert time.monotonic() < deadline, "holder never took the lock"
            time.sleep(0.05)
        assert holder.poll() is None, "holder exited after readiness"
        yield target


def test_os_lock_wait_is_bounded(lock_held_by_other_process: Path) -> None:
    from kader.memory.persistence import LockTimeoutError, file_lock

    target = lock_held_by_other_process
    started = time.monotonic()
    with pytest.raises(LockTimeoutError):
        with file_lock(target, timeout=0.5):
            pytest.fail("acquired a lock held by another process")
    waited = time.monotonic() - started
    assert 0.5 <= waited < 5.0


def test_immediate_timeout_tries_once(lock_held_by_other_process: Path) -> None:
    from kader.memory.persistence import LockTimeoutError, file_lock

    target = lock_held_by_other_process
    module, attribute, _ = _lock_primitive()
    real = getattr(module, attribute)
    attempts: list[int] = []

    def counting(*args: object) -> None:
        attempts.append(1)
        real(*args)

    started = time.monotonic()
    with patch.object(module, attribute, counting):
        with pytest.raises(LockTimeoutError):
            with file_lock(target, timeout=0):
                pytest.fail("acquired a lock held by another process")
    assert attempts == [1]
    assert time.monotonic() - started < 1.0


def test_immediate_timeout_succeeds_when_free(tmp_path: Path) -> None:
    from kader.memory.persistence import file_lock

    with file_lock(tmp_path / "conversation.json", timeout=0):
        pass


def test_lock_released_after_timeout_path(tmp_path: Path) -> None:
    """After a timed-out wait, the fd and thread lock were released."""
    from kader.memory.persistence import LockTimeoutError, file_lock

    target = tmp_path / "conversation.json"
    module, attribute, contention = _lock_primitive()

    def always_busy(*args: object) -> None:
        raise OSError(contention, "simulated contention")

    with patch.object(module, attribute, always_busy):
        with pytest.raises(LockTimeoutError):
            with file_lock(target, timeout=0.1):
                pass
    with file_lock(target, timeout=5):
        pass


def test_thread_lock_wait_is_bounded_and_skips_os_lock(tmp_path: Path) -> None:
    from kader.memory.persistence import LockTimeoutError, file_lock

    target = tmp_path / "conversation.json"
    held = threading.Event()
    release = threading.Event()

    def hold() -> None:
        with file_lock(target):
            held.set()
            release.wait(30)

    holder = threading.Thread(target=hold)
    holder.start()
    try:
        assert held.wait(10)
        module, attribute, _ = _lock_primitive()
        real = getattr(module, attribute)
        attempts: list[int] = []

        def counting(*args: object) -> None:
            attempts.append(1)
            real(*args)

        started = time.monotonic()
        with patch.object(module, attribute, counting):
            with pytest.raises(LockTimeoutError):
                with file_lock(target, timeout=0.3):
                    pytest.fail("acquired a lock held by another thread")
        assert 0.3 <= time.monotonic() - started < 5.0
        assert attempts == [], "no OS attempt once the budget is spent"
    finally:
        release.set()
        holder.join(10)
    assert not holder.is_alive(), "lock-holder thread did not terminate"


@pytest.mark.parametrize("bad", [-1, -0.001, float("nan"), float("inf")])
def test_invalid_timeout_is_refused(tmp_path: Path, bad: float) -> None:
    from kader.memory.persistence import file_lock

    with pytest.raises(ValueError):
        with file_lock(tmp_path / "conversation.json", timeout=bad):
            pass


def test_permanent_lock_error_fails_immediately(tmp_path: Path) -> None:
    from kader.memory.persistence import LockTimeoutError, file_lock

    target = tmp_path / "conversation.json"
    module, attribute, _ = _lock_primitive()
    calls: list[int] = []

    def broken(*args: object) -> None:
        calls.append(1)
        raise OSError(errno.EBADF, "simulated permanent lock failure")

    started = time.monotonic()
    with patch.object(module, attribute, broken):
        with pytest.raises(OSError) as excinfo:
            with file_lock(target, timeout=30):
                pytest.fail("lock acquired despite a permanent error")
    assert excinfo.value.errno == errno.EBADF
    assert not isinstance(excinfo.value, LockTimeoutError)
    assert calls == [1], "a permanent error must not be retried"
    assert time.monotonic() - started < 5.0

    # Failed acquisition released the thread lock and the descriptor.
    owner = RevisionedJsonFile(target)
    assert owner.save({"messages": ["ok"]}) == 1


def test_deadlock_errno_is_not_treated_as_contention(tmp_path: Path) -> None:
    deadlock = getattr(errno, "EDEADLOCK", None) or getattr(errno, "EDEADLK", None)
    if deadlock is None:
        pytest.skip("platform defines no deadlock errno")
    from kader.memory.persistence import file_lock

    module, attribute, _ = _lock_primitive()
    calls: list[int] = []

    def deadlocked(*args: object) -> None:
        calls.append(1)
        raise OSError(deadlock, "simulated deadlock")

    with patch.object(module, attribute, deadlocked):
        with pytest.raises(OSError) as excinfo:
            with file_lock(tmp_path / "conversation.json", timeout=30):
                pass
    assert excinfo.value.errno == deadlock
    assert calls == [1]


def test_contention_errors_are_retried_until_free(tmp_path: Path) -> None:
    from kader.memory.persistence import file_lock

    target = tmp_path / "conversation.json"
    module, attribute, contention = _lock_primitive()
    real = getattr(module, attribute)
    attempts: list[int] = []

    unlock_flag = module.LOCK_UN if attribute == "flock" else module.LK_UNLCK
    unlocks: list[int] = []

    def busy_twice(fd: int, op: int, *args: object) -> None:
        if op == unlock_flag:
            unlocks.append(1)
            real(fd, op, *args)
            return
        attempts.append(1)  # acquisition attempts only
        if len(attempts) <= 2:
            raise OSError(contention, "simulated contention")
        real(fd, op, *args)

    with patch.object(module, attribute, busy_twice):
        with file_lock(target, timeout=5):
            pass
    assert len(attempts) == 3  # two contended, one successful
    assert len(unlocks) == 1


def _break_unlock(module: object, attribute: str):
    """Patch the OS primitive so that lock works and unlock fails."""
    real = getattr(module, attribute)
    unlock_flag = module.LOCK_UN if attribute == "flock" else module.LK_UNLCK

    def lock_ok_unlock_fails(fd: int, op: int, *args: object) -> None:
        if op == unlock_flag:
            raise OSError(errno.EIO, "simulated unlock failure")
        real(fd, op, *args)

    return patch.object(module, attribute, lock_ok_unlock_fails)


def test_body_error_keeps_precedence_over_unlock_failure(tmp_path: Path) -> None:
    from kader.memory.persistence import file_lock

    target = tmp_path / "conversation.json"
    module, attribute, _ = _lock_primitive()
    with _break_unlock(module, attribute):
        with pytest.raises(KeyError) as excinfo:
            with file_lock(target, timeout=5):
                raise KeyError("operation failed")
    notes = getattr(excinfo.value, "__notes__", [])
    assert any("lock release also failed" in note for note in notes)

    # Closing the descriptor dropped the OS lock; the thread lock is free.
    with file_lock(target, timeout=5):
        pass


def test_unlock_failure_is_raised_when_body_succeeds(tmp_path: Path) -> None:
    from kader.memory.persistence import file_lock

    target = tmp_path / "conversation.json"
    module, attribute, _ = _lock_primitive()
    with _break_unlock(module, attribute):
        with pytest.raises(OSError) as excinfo:
            with file_lock(target, timeout=5):
                pass
    assert excinfo.value.errno == errno.EIO

    with file_lock(target, timeout=5):
        pass


def test_snapshot_is_taken_at_save_entry(tmp_path: Path) -> None:
    """Mutating the caller's data after save() starts cannot leak into the file."""
    target = tmp_path / "conversation.json"
    messages = [{"role": "user", "content": "original"}]
    owner = RevisionedJsonFile(target)

    real_write = memory_types.atomic_write_text

    def mutate_then_write(path: Path, content: str, encoding: str = "utf-8") -> None:
        messages[0]["content"] = "mutated during save"
        messages.append({"role": "user", "content": "appended during save"})
        real_write(path, content, encoding)

    with patch.object(memory_types, "atomic_write_text", mutate_then_write):
        owner.save({"messages": messages})

    assert load_json(target)["messages"] == [{"role": "user", "content": "original"}]


def test_non_json_payload_is_refused_without_writing(tmp_path: Path) -> None:
    target = tmp_path / "conversation.json"
    owner = RevisionedJsonFile(target)
    owner.save({"messages": ["a"]})
    before = target.read_bytes()

    with pytest.raises(TypeError):
        owner.save({"messages": [object()]})
    assert target.read_bytes() == before
    # NaN/Infinity would make a history the strict reader refuses: refuse
    # them at the writer instead, before anything is written.
    for value in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError):
            owner.save({"messages": [{"score": value}]})
        assert target.read_bytes() == before
    assert owner.save({"messages": ["a", "b"]}) == 2


# --- Concurrency ------------------------------------------------------------


def test_concurrent_writers_at_same_revision_exactly_one_wins(tmp_path: Path) -> None:
    target = tmp_path / "conversation.json"
    owners = [RevisionedJsonFile(target) for _ in range(8)]
    for owner in owners:
        owner.load()

    barrier = threading.Barrier(len(owners))
    outcomes: list[str] = []
    outcomes_lock = threading.Lock()

    def run(index: int) -> None:
        barrier.wait()
        try:
            owners[index].save({"messages": [f"writer-{index}"]})
            result = "saved"
        except RevisionConflictError:
            result = "refused"
        with outcomes_lock:
            outcomes.append(result)

    threads = [threading.Thread(target=run, args=(i,)) for i in range(len(owners))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert outcomes.count("saved") == 1
    assert outcomes.count("refused") == len(owners) - 1
    assert load_json(target)["revision"] == 1


def test_concurrent_use_of_one_owner_is_serialized(tmp_path: Path) -> None:
    target = tmp_path / "conversation.json"
    owner = RevisionedJsonFile(target)
    errors: list[BaseException] = []

    def run(index: int) -> None:
        for step in range(10):
            try:
                owner.save({"messages": [f"thread-{index}-step-{step}"]})
            except BaseException as exc:  # noqa: BLE001 - recorded for assert
                errors.append(exc)

    threads = [threading.Thread(target=run, args=(i,)) for i in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert owner.revision == 40
    assert load_json(target)["revision"] == 40


def test_independent_processes_lose_no_update(tmp_path: Path) -> None:
    """Real multi-process contention, starting before the lock file exists.

    Each process appends one message per round: load, append, compare-and-
    write, reloading after a refusal. Without the lock two processes could
    both replace from the same revision and drop an update; without the
    revision check a stale owner could overwrite a newer history. Either
    failure shows up as fewer than workers * rounds messages.
    """
    target = tmp_path / "conversation.json"
    workers, rounds = 4, 10
    start_at = time.time() + 3.0
    body = """
        target, name, rounds, start_at = (
            Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]), float(sys.argv[4])
        )
        owner = persistence.RevisionedJsonFile(target)
        while time.time() < start_at:
            time.sleep(0.001)
        done = conflicts = 0
        while done < rounds:
            data = owner.load()
            messages = data.get("messages", []) + [f"{name}-{done}"]
            try:
                owner.save({"messages": messages})
                done += 1
            except persistence.RevisionConflictError:
                conflicts += 1
        print(conflicts)
    """
    script = SUBPROCESS_PREAMBLE + textwrap.dedent(body)
    results: list[tuple[int, str]] = []
    with _reaped_children() as procs:
        for i in range(workers):
            procs.append(
                subprocess.Popen(
                    [sys.executable, "-c", script, str(target), f"w{i}", str(rounds)]
                    + [str(start_at)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
            )
        for proc in procs:
            _, err = proc.communicate(timeout=120)
            results.append((proc.returncode, err))
    for returncode, err in results:
        assert returncode == 0, err

    stored = load_json(target)
    assert stored["revision"] == workers * rounds
    assert len(stored["messages"]) == workers * rounds
    assert len(set(stored["messages"])) == workers * rounds
    for i in range(workers):
        mine = [m for m in stored["messages"] if m.startswith(f"w{i}-")]
        assert mine == [f"w{i}-{n}" for n in range(rounds)]


def test_cross_process_stale_writer_is_refused(tmp_path: Path) -> None:
    target = tmp_path / "conversation.json"
    owner = RevisionedJsonFile(target)
    owner.save({"messages": ["a"]})

    result = _run_python(
        """
        other = persistence.RevisionedJsonFile(Path(sys.argv[1]))
        data = other.load()
        other.save({"messages": data["messages"] + ["from another process"]})
        """,
        str(target),
    )
    assert result.returncode == 0, result.stderr

    with pytest.raises(RevisionConflictError):
        owner.save({"messages": ["a", "stale"]})
    assert load_json(target)["messages"] == ["a", "from another process"]


# --- Unconfirmed durability -------------------------------------------------


def test_unconfirmed_durability_rereads_revision_and_does_not_retry(
    tmp_path: Path,
) -> None:
    target = tmp_path / "conversation.json"
    owner = RevisionedJsonFile(target)

    real_write = memory_types.atomic_write_text
    calls: list[Path] = []

    def counting_write(path: Path, content: str, encoding: str = "utf-8") -> None:
        calls.append(path)
        real_write(path, content, encoding)

    with (
        patch.object(memory_types, "_fsync_directory", _failing_dir_sync),
        patch.object(memory_types, "atomic_write_text", counting_write),
    ):
        with pytest.raises(DurabilityUnconfirmedError) as excinfo:
            owner.save({"messages": ["a"]})

    assert DurabilityUnconfirmedError.status in str(excinfo.value)
    assert len(calls) == 1, "an unconfirmed write must never be retried blindly"
    # Re-read found this save's identity: visible, but still not durable.
    assert excinfo.value.write_visible is True
    assert owner.durability_confirmed is False
    assert owner.revision == 1
    assert load_json(target)["messages"] == ["a"]

    # Not settled as clean: an unchanged follow-up save is written again,
    # revision-checked, rather than skipped as if it were durable.
    with patch.object(memory_types, "atomic_write_text", counting_write):
        assert owner.save({"messages": ["a"]}) == 2
    assert len(calls) == 2
    assert owner.durability_confirmed is True


def test_unconfirmed_durability_with_foreign_write_invalidates_owner(
    tmp_path: Path,
) -> None:
    """FAULT INJECTION: foreign stored state during reconciliation.

    The hook bypasses the lock to plant a foreign state; this forces the
    reconciliation branch to meet a foreign write_id. It is not evidence
    about cooperating writers, which always take the lock; that is
    test_independent_processes_lose_no_update.
    """
    target = tmp_path / "conversation.json"
    owner = RevisionedJsonFile(target)
    owner.load()

    foreign = {"messages": ["other"], "revision": 1, "write_id": "cd" * 16}
    foreign_bytes = json.dumps(foreign).encode("utf-8")
    fired: list[bool] = []

    def one_shot_foreign_write_then_fail(directory: Path) -> None:
        # One-shot hook: plant a foreign stored state with plain file I/O
        # (no save_json, so no recursion through this patch), then report
        # the directory sync as failed.
        assert not fired, "hook must fire exactly once"
        fired.append(True)
        target.write_bytes(foreign_bytes)
        _failing_dir_sync(directory)

    with patch.object(
        memory_types, "_fsync_directory", one_shot_foreign_write_then_fail
    ):
        with pytest.raises(DurabilityUnconfirmedError) as excinfo:
            owner.save({"messages": ["mine"]})

    assert fired == [True]
    assert excinfo.value.write_visible is False
    assert owner.revision is None
    with pytest.raises(RevisionConflictError):
        owner.save({"messages": ["mine", "again"]})
    assert target.read_bytes() == foreign_bytes

    assert owner.load() == {"messages": ["other"]}
    assert owner.save({"messages": ["other", "mine"]}) == 2


# --- Persistence owners -----------------------------------------------------


def test_session_manager_refuses_stale_conversation_save(tmp_path: Path) -> None:
    config = MemoryConfig(memory_dir=tmp_path)
    first = FileSessionManager(config)
    session = first.create_session("agent")
    first.save_conversation(session.session_id, [{"role": "user", "content": "1"}])

    second = FileSessionManager(config)
    assert len(second.load_conversation(session.session_id)) == 1

    first.save_conversation(
        session.session_id,
        [{"role": "user", "content": "1"}, {"role": "user", "content": "2"}],
    )
    with pytest.raises(RevisionConflictError):
        second.save_conversation(
            session.session_id, [{"role": "user", "content": "stale"}]
        )

    assert [m["content"] for m in first.load_conversation(session.session_id)] == [
        "1",
        "2",
    ]


def test_session_manager_does_not_overwrite_unloaded_history(tmp_path: Path) -> None:
    config = MemoryConfig(memory_dir=tmp_path)
    first = FileSessionManager(config)
    session = first.create_session("agent")
    first.save_conversation(session.session_id, [{"role": "user", "content": "1"}])

    with pytest.raises(RevisionConflictError):
        FileSessionManager(config).save_conversation(session.session_id, [])
    assert len(first.load_conversation(session.session_id)) == 1


def test_conversation_survives_restart_and_stale_writer(tmp_path: Path) -> None:
    """End to end: add, save, restart (new manager), recover, refuse stale."""
    config = MemoryConfig(memory_dir=tmp_path)
    before = FileSessionManager(config)
    session = before.create_session("agent")
    history = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]
    before.save_conversation(session.session_id, history)

    after = FileSessionManager(config)  # simulated process restart
    recovered = after.load_conversation(session.session_id)
    assert recovered == history
    after.save_conversation(
        session.session_id, recovered + [{"role": "user", "content": "more"}]
    )

    # The pre-restart manager is now stale and cannot roll history back.
    # Re-saving its unchanged history takes the local fast path: nothing is
    # written, and the result is flagged as not verified against storage.
    before.save_conversation(session.session_id, history)
    store = before._conversation_store(session.session_id)
    assert store.storage_verified is False
    assert len(FileSessionManager(config).load_conversation(session.session_id)) == 3
    # Any changed save from the stale manager is checked and refused.
    with pytest.raises(RevisionConflictError):
        before.save_conversation(session.session_id, history[:1])
    assert len(FileSessionManager(config).load_conversation(session.session_id)) == 3


async def test_async_session_manager_is_revision_checked(tmp_path: Path) -> None:
    config = MemoryConfig(memory_dir=tmp_path)
    manager = AsyncFileSessionManager(config)
    session = await manager.async_create_session("agent")
    await manager.async_save_conversation(session.session_id, [{"content": "1"}])

    other = AsyncFileSessionManager(config)
    assert await other.async_load_conversation(session.session_id) == [{"content": "1"}]
    await manager.async_save_conversation(
        session.session_id, [{"content": "1"}, {"content": "2"}]
    )
    with pytest.raises(RevisionConflictError):
        await other.async_save_conversation(session.session_id, [{"content": "x"}])

    stored = load_json(tmp_path / "sessions" / session.session_id / "conversation.json")
    assert stored["revision"] == 2
    assert len(stored["messages"]) == 2


def test_persistent_memory_manager_writes_revisioned_history(tmp_path: Path) -> None:
    """On main these managers persist on every add (there is no deferred mode).

    add_messages() persists per message and once more at the end; that last
    save is unchanged and is skipped, so the revision counts real changes.
    """
    target = tmp_path / "executor" / "conversation.json"
    memory = PersistentSlidingWindowConversationManager(file_path=target)
    memory.add_message({"role": "user", "content": "hi"})
    memory.add_messages([{"role": "assistant", "content": "hello"}])

    stored = load_json(target)
    assert [m["content"] for m in stored["messages"]] == ["hi", "hello"]
    assert stored["revision"] == 2

    # A second manager pointed at the same file never wipes that history.
    intruder = PersistentSlidingWindowConversationManager(file_path=target)
    intruder.add_message({"role": "user", "content": "overwrite?"})
    assert load_json(target) == stored


def test_permission_error_opening_lock_file_is_not_contention(tmp_path: Path) -> None:
    """EACCES counts as contention only from the native lock call itself."""
    from kader.memory import persistence

    target = tmp_path / "conversation.json"
    real_open = os.open
    calls: list[str] = []

    def deny_lock_file(path: object, *args: object, **kwargs: object) -> int:
        if str(path).endswith(".kader-lock"):
            calls.append(str(path))
            raise PermissionError(errno.EACCES, "simulated permission denied")
        return real_open(path, *args, **kwargs)

    started = time.monotonic()
    with patch.object(persistence.os, "open", deny_lock_file):
        with pytest.raises(PermissionError):
            with persistence.file_lock(target, timeout=30):
                pass
    assert len(calls) == 1
    assert time.monotonic() - started < 5.0
    with persistence.file_lock(target, timeout=5):
        pass
