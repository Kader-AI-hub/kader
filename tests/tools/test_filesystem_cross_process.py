"""
Cross-process mutual exclusion for FilesystemBackend._file_lock.

These tests start real, independent Python processes (not threads), so they
exercise the platform adapter actually in use: msvcrt.locking on Windows,
fcntl.flock on POSIX. Every spawned process is reaped on every exit path.
"""

import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from kader.tools import filesystem as fs_module
from kader.tools.filesystem import FilesystemBackend

REPO_ROOT = Path(__file__).resolve().parents[2]

# Signals READY immediately before trying to take the lock, records enter/exit
# in a shared append-only log (one os.write per record), optionally drops a
# marker once it holds the lock, and reports an acquisition failure explicitly.
WORKER = r"""
import os, sys, time
from pathlib import Path
from kader.tools.filesystem import FilesystemBackend

root, name, log, tag, hold, held_marker, ready_marker = sys.argv[1:8]
backend = FilesystemBackend(root_dir=root)
target = (Path(root) / name).resolve()

def record(event):
    fd = os.open(log, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(fd, f"{event} {tag}\n".encode())
    finally:
        os.close(fd)

if ready_marker != "-":
    Path(ready_marker).write_text("ready")
try:
    with backend._file_lock(target):
        record("enter")
        if held_marker != "-":
            Path(held_marker).write_text("held")
        time.sleep(float(hold))
        record("exit")
except OSError as exc:
    record("acquire-failed")
    print(f"ACQUIRE_FAILED errno={exc.errno}", file=sys.stderr)
    sys.exit(3)
"""


class _Procs:
    """Tracks children; kills and reaps any still running at teardown."""

    def __init__(self) -> None:
        self._procs: list[subprocess.Popen] = []

    def spawn(
        self,
        root: Path,
        log: Path,
        tag: str,
        hold: float,
        held_marker: str = "-",
        ready_marker: str = "-",
    ) -> subprocess.Popen:
        env = dict(os.environ)
        env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        proc = subprocess.Popen(
            [
                sys.executable,
                "-c",
                WORKER,
                str(root),
                "shared.txt",
                str(log),
                tag,
                str(hold),
                held_marker,
                ready_marker,
            ],
            cwd=str(REPO_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self._procs.append(proc)
        return proc

    def reap_all(self) -> None:
        for proc in self._procs:
            if proc.poll() is None:
                proc.kill()
            try:
                proc.communicate(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                try:
                    proc.communicate(timeout=10)
                except subprocess.TimeoutExpired as exc:
                    raise RuntimeError(
                        f"could not reap test child pid={proc.pid}"
                    ) from exc


@pytest.fixture
def procs(tmp_path):
    # Depends on tmp_path so that children (which hold the lock file open) are
    # reaped before any directory cleanup; Windows refuses to delete an open
    # file (WinError 32).
    tracker = _Procs()
    try:
        yield tracker
    finally:
        tracker.reap_all()


def _events(log: Path) -> list[tuple[str, str]]:
    if not log.exists():
        return []
    return [tuple(line.split()) for line in log.read_text().splitlines()]


def _wait_for(predicate, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


def _assert_never_overlapping(events: list[tuple[str, str]]) -> None:
    """Every 'enter X' is immediately followed by 'exit X'."""
    assert len(events) % 2 == 0, events
    for i in range(0, len(events), 2):
        assert events[i][0] == "enter", events
        assert events[i + 1] == ("exit", events[i][1]), events


def _finish(proc: subprocess.Popen, timeout: float = 30) -> int:
    _, err = proc.communicate(timeout=timeout)
    assert proc.returncode == 0, err.decode(errors="replace")
    return proc.returncode


class TestCrossProcessLocking:
    def test_three_processes_never_hold_the_lock_together(self, procs, tmp_path):
        root = tmp_path
        log = root / "events.log"
        started = [procs.spawn(root, log, f"p{i}", hold=0.3) for i in range(3)]
        for proc in started:
            _finish(proc)

        events = _events(log)
        assert len(events) == 6
        _assert_never_overlapping(events)
        assert {tag for _, tag in events} == {"p0", "p1", "p2"}

    def test_contender_waits_and_killed_holder_releases_the_lock(self, procs, tmp_path):
        """A contender that has REACHED the acquisition point must not enter
        while another process holds the lock; the OS must release the lock when
        the holder is killed; and a late-arriving process then acquires it."""
        root = tmp_path
        log = root / "events.log"
        held = root / "holder.held"
        ready = root / "contender.ready"

        holder = procs.spawn(root, log, "holder", hold=60, held_marker=str(held))
        assert _wait_for(held.exists, timeout=30), "holder never locked"

        contender = procs.spawn(
            root, log, "contender", hold=0.1, ready_marker=str(ready)
        )
        assert _wait_for(ready.exists, timeout=30), "contender never started"
        # READY is written immediately BEFORE _file_lock() is entered: it
        # precedes, but does not prove entry into, the native acquisition
        # call. The 1 s observation below covers that gap in practice.
        # Observe well inside msvcrt's ~10 s native retry window.
        time.sleep(1.0)
        assert contender.poll() is None, "contender should be blocked"
        assert _events(log) == [("enter", "holder")]

        holder.kill()
        holder.wait(timeout=30)

        _finish(contender)
        _finish(procs.spawn(root, log, "late", hold=0.1))

        assert _events(log) == [
            ("enter", "holder"),  # never exits: killed while holding
            ("enter", "contender"),
            ("exit", "contender"),
            ("enter", "late"),
            ("exit", "late"),
        ]

    @pytest.mark.skipif(
        os.name != "nt" or os.environ.get("KADER_SLOW_LOCK_TESTS") != "1",
        reason="Windows-only, ~15 s: set KADER_SLOW_LOCK_TESTS=1 to run",
    )
    def test_windows_acquisition_beyond_native_retry_window_fails_explicitly(
        self, procs, tmp_path
    ):
        """msvcrt.LK_LOCK gives up after ~10 one-second retries. A contender
        blocked longer than that must fail EXPLICITLY (an OSError), never
        deadlock and never fall back to running unlocked."""
        root = tmp_path
        log = root / "events.log"
        held = root / "holder.held"
        ready = root / "contender.ready"

        procs.spawn(root, log, "holder", hold=60, held_marker=str(held))
        assert _wait_for(held.exists, timeout=30), "holder never locked"
        contender = procs.spawn(
            root, log, "contender", hold=0.1, ready_marker=str(ready)
        )
        assert _wait_for(ready.exists, timeout=30)

        _, err = contender.communicate(timeout=40)
        assert contender.returncode == 3, err.decode(errors="replace")
        assert "ACQUIRE_FAILED" in err.decode(errors="replace")
        assert _events(log) == [
            ("enter", "holder"),
            ("acquire-failed", "contender"),
        ]


class TestFailedAcquisitionLeavesNoStaleLock:
    def test_failed_cross_process_acquisition_releases_everything(self):
        """If the cross-process lock cannot be acquired: no unlock is attempted,
        the lock-file descriptor is closed, the in-process stripe lock is
        released, and the same logical lock can be acquired afterwards."""
        with tempfile.TemporaryDirectory() as temp_dir:
            backend = FilesystemBackend(root_dir=temp_dir)
            (Path(temp_dir) / "f.txt").write_text("hello")
            calls: list = []

            if fs_module.fcntl is not None:
                adapter, attr = fs_module.fcntl, "flock"
            elif fs_module.msvcrt is not None:
                adapter, attr = fs_module.msvcrt, "locking"
            else:
                pytest.skip("no cross-process lock adapter on this platform")

            def refuse(*args):
                calls.append(args)
                # Real msvcrt raises after its retries; a real unlock of a
                # never-locked region raises too. Both must be harmless.
                raise OSError(36, "lock refused")

            real_open, real_close = os.open, os.close
            lock_fds: list[int] = []
            closed: list[int] = []

            def tracking_open(path, flags, *args, **kwargs):
                fd = real_open(path, flags, *args, **kwargs)
                if Path(path).name.startswith(".kader-lock."):
                    lock_fds.append(fd)
                return fd

            def tracking_close(fd):
                closed.append(fd)
                real_close(fd)

            with (
                patch.object(adapter, attr, side_effect=refuse),
                patch("kader.tools.filesystem.os.open", side_effect=tracking_open),
                patch("kader.tools.filesystem.os.close", side_effect=tracking_close),
            ):
                result = backend.edit("f.txt", "hello", "bye")

            assert result.error is not None
            assert "lock refused" in result.error
            assert len(calls) == 1, f"unlock must not be attempted: {calls}"
            assert len(lock_fds) == 1
            assert lock_fds[0] in closed, "lock-file descriptor left open"

            outcome: dict = {}

            def other_thread():
                outcome["result"] = backend.edit("f.txt", "hello", "world")

            worker = threading.Thread(target=other_thread)
            worker.start()
            worker.join(timeout=10)
            assert not worker.is_alive(), "stripe lock left held: deadlock"
            assert outcome["result"].error is None
            assert (Path(temp_dir) / "f.txt").read_text() == "world"

    def test_unlock_failure_does_not_replace_the_operation_error(self):
        """Both fail: the operation's error keeps precedence and carries the
        unlock failure as a note. Only the unlock fails: it is raised.

        The simulated unlock really releases and THEN reports failure, so this
        tests exception precedence and cleanup only. A native unlock that fails
        WITHOUT releasing is left to the OS: closing the descriptor drops it."""
        with tempfile.TemporaryDirectory() as temp_dir:
            backend = FilesystemBackend(root_dir=temp_dir)
            target = (Path(temp_dir) / "g.txt").resolve()

            if fs_module.fcntl is not None:
                adapter, attr = fs_module.fcntl, "flock"
                is_unlock = lambda args: args[1] == fs_module.fcntl.LOCK_UN  # noqa: E731
            elif fs_module.msvcrt is not None:
                adapter, attr = fs_module.msvcrt, "locking"
                is_unlock = lambda args: args[1] == fs_module.msvcrt.LK_UNLCK  # noqa: E731
            else:
                pytest.skip("no cross-process lock adapter on this platform")

            real = getattr(adapter, attr)

            def failing_unlock(*args):
                if is_unlock(args):
                    real(*args)  # really release, then report a failure
                    raise OSError(5, "unlock failed")
                return real(*args)

            with patch.object(adapter, attr, side_effect=failing_unlock):
                with pytest.raises(ValueError, match="operation failed") as both:
                    with backend._file_lock(target):
                        raise ValueError("operation failed")
                notes = getattr(both.value, "__notes__", [])
                assert any("lock release also failed" in n for n in notes), notes

                with pytest.raises(OSError, match="unlock failed"):
                    with backend._file_lock(target):
                        pass

            # The stripe lock is free again.
            with backend._file_lock(target):
                pass
