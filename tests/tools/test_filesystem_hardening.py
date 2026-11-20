"""
Hardening tests for FilesystemBackend race-safety features.

Covers Stage 1: atomic writes, advisory file locks, and freshness tokens.
"""

import errno
import os
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import patch

import pytest

try:
    import fcntl
except ImportError:
    fcntl = None  # type: ignore

from kader.tools.filesystem import DurabilityUnconfirmedError, FilesystemBackend


class TestAtomicWrites:
    """Tests for atomic write behavior."""

    def test_write_creates_file_atomically(self):
        """A successful write must leave the target and no temp file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            backend = FilesystemBackend(root_dir=temp_dir)
            result = backend.write("hello.txt", "world")

            assert result.error is None
            assert result.path == "hello.txt"
            target = Path(temp_dir) / "hello.txt"
            assert target.read_text() == "world"
            assert not list(Path(temp_dir).glob("*.kader-tmp"))

    def test_write_rollback_on_replace_failure(self):
        """If os.replace fails, the target must not exist and temp cleaned."""
        with tempfile.TemporaryDirectory() as temp_dir:
            backend = FilesystemBackend(root_dir=temp_dir)
            target = Path(temp_dir) / "fail.txt"

            def boom(*args, **kwargs):
                raise OSError("replace refused")

            with patch("os.replace", side_effect=boom) as replace_mock:
                result = backend.write("fail.txt", "new content")

            assert result.error is not None
            assert "replace refused" in result.error
            assert not target.exists()
            assert not list(Path(temp_dir).glob("*.kader-tmp"))
            assert replace_mock.call_count == 1

    def test_windows_transient_replace_retries_then_succeeds(self):
        """A sharing violation should retry once and then complete atomically."""
        with tempfile.TemporaryDirectory() as temp_dir:
            backend = FilesystemBackend(root_dir=temp_dir)
            original_replace = os.replace
            attempts = {"count": 0}

            def transient_then_replace(src, dst):
                attempts["count"] += 1
                if attempts["count"] == 1:
                    exc = OSError("sharing violation")
                    exc.winerror = 32
                    raise exc
                original_replace(src, dst)

            with (
                patch("kader.tools.filesystem.os.name", "nt"),
                patch("os.replace", side_effect=transient_then_replace),
            ):
                result = backend.write("retry.txt", "content")

            assert result.error is None
            assert attempts["count"] == 2
            assert (Path(temp_dir) / "retry.txt").read_text() == "content"

    def test_windows_replace_retry_exhaustion_is_bounded(self):
        """Transient replacement failures should stop after five attempts."""
        with tempfile.TemporaryDirectory() as temp_dir:
            backend = FilesystemBackend(root_dir=temp_dir)

            def always_transient(*args, **kwargs):
                exc = OSError("sharing violation")
                exc.winerror = 32
                raise exc

            with (
                patch("kader.tools.filesystem.os.name", "nt"),
                patch("os.replace", side_effect=always_transient) as replace_mock,
            ):
                result = backend.write("retry.txt", "content")

            assert result.error is not None
            assert replace_mock.call_count == 5
            assert not (Path(temp_dir) / "retry.txt").exists()
            assert not list(Path(temp_dir).glob("*.kader-tmp"))

    def test_edit_atomic_no_partial_file(self):
        """edit() must not leave partial content if the write is interrupted."""
        with tempfile.TemporaryDirectory() as temp_dir:
            backend = FilesystemBackend(root_dir=temp_dir)
            target = Path(temp_dir) / "edit.txt"
            target.write_text("hello world")

            def boom(*args, **kwargs):
                raise OSError("replace refused")

            with patch("os.replace", side_effect=boom):
                result = backend.edit("edit.txt", "world", "universe")

            assert result.error is not None
            assert target.read_text() == "hello world"
            assert not list(Path(temp_dir).glob("*.kader-tmp"))

    def test_upload_files_atomic(self):
        """upload_files must write complete files and leave no temp files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            backend = FilesystemBackend(root_dir=temp_dir)
            files = [
                ("a.txt", b"alpha"),
                ("sub/b.txt", b"beta"),
            ]
            responses = backend.upload_files(files)

            assert len(responses) == 2
            assert all(r.error is None for r in responses)
            assert (Path(temp_dir) / "a.txt").read_bytes() == b"alpha"
            assert (Path(temp_dir) / "sub" / "b.txt").read_bytes() == b"beta"
            assert not list(Path(temp_dir).rglob("*.kader-tmp"))

    def test_durable_writes_can_skip_fsync(self):
        """Non-durable writes should avoid file and directory fsync calls."""
        with tempfile.TemporaryDirectory() as temp_dir:
            backend = FilesystemBackend(
                root_dir=temp_dir,
                durable_writes=False,
            )
            with patch("os.fsync") as fsync_mock:
                result = backend.write("fast.txt", "content")

            assert result.error is None
            fsync_mock.assert_not_called()

    def test_directory_is_synced_after_replace(self):
        """The directory fsync must FOLLOW os.replace(); before it, the rename
        is not yet in the directory and nothing about it is persisted."""
        with tempfile.TemporaryDirectory() as temp_dir:
            backend = FilesystemBackend(root_dir=temp_dir)
            events: list[str] = []
            original_replace = os.replace

            def recording_replace(src, dst):
                events.append("replace")
                original_replace(src, dst)

            def recording_dir_sync(directory):
                events.append("dir-sync")
                assert Path(directory) == Path(temp_dir).resolve()

            with (
                patch("os.replace", side_effect=recording_replace),
                patch.object(
                    FilesystemBackend,
                    "_fsync_directory",
                    side_effect=recording_dir_sync,
                ),
            ):
                result = backend.write("ordered.txt", "content")

            assert result.error is None
            assert events == ["replace", "dir-sync"]

    def test_file_fsync_precedes_replace(self):
        """durable_writes=True: the temp file is fsynced before replacement."""
        with tempfile.TemporaryDirectory() as temp_dir:
            backend = FilesystemBackend(root_dir=temp_dir)
            events: list[str] = []
            original_replace = os.replace

            def recording_replace(src, dst):
                events.append("replace")
                original_replace(src, dst)

            with (
                patch("os.fsync", side_effect=lambda fd: events.append("fsync")),
                patch("os.replace", side_effect=recording_replace),
                patch.object(FilesystemBackend, "_fsync_directory"),
            ):
                result = backend.write("synced.txt", "content")

            assert result.error is None
            assert events == ["fsync", "replace"]

    def test_failed_file_fsync_prevents_replacement(self):
        """If the temp file cannot be fsynced, the target must not change."""
        with tempfile.TemporaryDirectory() as temp_dir:
            backend = FilesystemBackend(root_dir=temp_dir)
            target = Path(temp_dir) / "keep.txt"
            target.write_text("original")

            with (
                patch("os.fsync", side_effect=OSError(errno.EIO, "disk error")),
                patch("os.replace") as replace_mock,
            ):
                result = backend.write("keep.txt", "new content")

            assert result.error is not None
            replace_mock.assert_not_called()
            assert target.read_text() == "original"
            assert not list(Path(temp_dir).glob("*.kader-tmp"))

    @pytest.mark.parametrize("failing", ["write", "flush"])
    def test_failed_write_or_flush_leaves_target_unchanged(self, failing):
        """A write/flush failure must leave the target untouched."""
        with tempfile.TemporaryDirectory() as temp_dir:
            backend = FilesystemBackend(root_dir=temp_dir)
            target = Path(temp_dir) / "keep.bin"
            target.write_bytes(b"original")
            real_fdopen = os.fdopen

            class FailingFile:
                def __init__(self, inner):
                    self._inner = inner

                def __enter__(self):
                    return self

                def __exit__(self, *exc):
                    self._inner.close()
                    return False

                def fileno(self):
                    return self._inner.fileno()

                def write(self, data):
                    if failing == "write":
                        raise OSError(errno.ENOSPC, "no space")
                    return self._inner.write(data)

                def flush(self):
                    if failing == "flush":
                        raise OSError(errno.ENOSPC, "no space")
                    return self._inner.flush()

            with (
                patch(
                    "kader.tools.filesystem.os.fdopen",
                    side_effect=lambda *a, **k: FailingFile(real_fdopen(*a, **k)),
                ),
                patch("os.replace") as replace_mock,
            ):
                responses = backend.upload_files([("keep.bin", b"new")])

            assert responses[0].error is not None
            replace_mock.assert_not_called()
            assert target.read_bytes() == b"original"
            assert not list(Path(temp_dir).glob("*.kader-tmp"))

    def test_fdopen_failure_closes_descriptor_and_keeps_original_error(self):
        """fdopen() failing: the raw fd is closed, and a cleanup failure must
        not mask the original error."""
        with tempfile.TemporaryDirectory() as temp_dir:
            backend = FilesystemBackend(root_dir=temp_dir)
            real_close = os.close
            closed: list[int] = []

            def failing_close(fd):
                closed.append(fd)
                real_close(fd)
                raise OSError(errno.EBADF, "close failed")

            with (
                patch(
                    "kader.tools.filesystem.os.fdopen",
                    side_effect=ValueError("fdopen refused"),
                ),
                patch("kader.tools.filesystem.os.close", side_effect=failing_close),
            ):
                with pytest.raises(ValueError, match="fdopen refused"):
                    backend._atomic_write(Path(temp_dir) / "x.txt", "content")

            assert len(closed) == 1
            assert not list(Path(temp_dir).glob("*.kader-tmp"))

    def test_successful_write_closes_descriptor_exactly_once(self):
        """After fdopen() the file object owns the fd; no second raw close."""
        with tempfile.TemporaryDirectory() as temp_dir:
            backend = FilesystemBackend(root_dir=temp_dir)
            with patch("kader.tools.filesystem.os.close", wraps=os.close) as close_mock:
                result = backend.write("once.txt", "content")

            assert result.error is None
            # Only the directory descriptor (POSIX) may be raw-closed; never
            # the temp-file descriptor, which its file object closed.
            expected = 0 if os.name == "nt" else 1
            assert close_mock.call_count == expected

    def test_temp_file_creation_is_exclusive(self):
        """An existing temp path must be refused, never overwritten."""
        with tempfile.TemporaryDirectory() as temp_dir:
            backend = FilesystemBackend(root_dir=temp_dir)
            fixed = "0" * 32
            squatter = Path(temp_dir) / f"x.txt.{fixed}.kader-tmp"
            squatter.write_text("someone else's file")

            class FixedUuid:
                hex = fixed

            with patch("kader.tools.filesystem.uuid.uuid4", return_value=FixedUuid):
                with pytest.raises(FileExistsError):
                    backend._atomic_write(Path(temp_dir) / "x.txt", "content")

            assert not (Path(temp_dir) / "x.txt").exists()

    def test_directory_sync_failure_after_replace_is_classified(self):
        """A failed dir sync AFTER a successful replace is not an ordinary
        failed write: the target holds the new content and the error says so."""
        with tempfile.TemporaryDirectory() as temp_dir:
            backend = FilesystemBackend(root_dir=temp_dir)
            target = Path(temp_dir) / "replaced.txt"
            target.write_text("old")

            with patch.object(
                FilesystemBackend,
                "_fsync_directory",
                side_effect=OSError(errno.EIO, "disk error"),
            ):
                with pytest.raises(DurabilityUnconfirmedError) as excinfo:
                    backend._atomic_write(target, "new")
                write_result = backend.write("other.txt", "fresh")
                upload = backend.upload_files([("up.bin", b"data")])

            assert excinfo.value.errno == errno.EIO
            assert target.read_text() == "new"
            assert write_result.error is not None
            assert "REPLACED_DURABILITY_UNCONFIRMED" in write_result.error
            assert upload[0].error == "durability_unconfirmed"
            assert not list(Path(temp_dir).glob("*.kader-tmp"))

    def test_non_durable_writes_skip_directory_sync(self):
        """durable_writes=False must not sync the containing directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            backend = FilesystemBackend(root_dir=temp_dir, durable_writes=False)
            with patch.object(FilesystemBackend, "_fsync_directory") as dir_sync:
                result = backend.write("fast.txt", "content")

            assert result.error is None
            dir_sync.assert_not_called()

    @pytest.mark.parametrize(
        "code", [errno.EINVAL, errno.ENOTSUP, getattr(errno, "ENOSYS", 38)]
    )
    def test_directory_sync_tolerates_unsupported_filesystems(self, code):
        """Unsupported-operation errnos from a directory fsync are tolerated."""
        with (
            patch("kader.tools.filesystem.os.name", "posix"),
            patch("kader.tools.filesystem.os.open", return_value=99),
            patch(
                "kader.tools.filesystem.os.fsync",
                side_effect=OSError(code, "unsupported"),
            ),
            patch("kader.tools.filesystem.os.close") as close_mock,
        ):
            FilesystemBackend._fsync_directory(Path("dir"))

        close_mock.assert_called_once_with(99)

    def test_directory_sync_raises_real_io_errors(self):
        """An I/O error from a directory fsync must not be swallowed."""
        with (
            patch("kader.tools.filesystem.os.name", "posix"),
            patch("kader.tools.filesystem.os.open", return_value=99),
            patch(
                "kader.tools.filesystem.os.fsync",
                side_effect=OSError(errno.EIO, "disk error"),
            ),
            patch("kader.tools.filesystem.os.close") as close_mock,
        ):
            with pytest.raises(OSError) as excinfo:
                FilesystemBackend._fsync_directory(Path("dir"))

        assert excinfo.value.errno == errno.EIO
        close_mock.assert_called_once_with(99)

    def test_temp_file_exists_during_write_and_is_removed_after(self):
        """The .kader-tmp file must exist during the atomic rename and be gone after."""
        with tempfile.TemporaryDirectory() as temp_dir:
            backend = FilesystemBackend(root_dir=temp_dir)
            target = Path(temp_dir) / "atomic.txt"
            captured: dict[str, bool] = {"saw_tmp": False}

            original_replace = os.replace

            def capturing_replace(src: str, dst: str) -> None:
                src_path = Path(src)
                if src_path.suffix.endswith(".kader-tmp"):
                    captured["saw_tmp"] = True
                    assert src_path.exists()
                    assert src_path.read_text() == "new content"
                original_replace(src, dst)

            with patch("os.replace", side_effect=capturing_replace):
                result = backend.write("atomic.txt", "new content")

            assert result.error is None
            assert captured["saw_tmp"] is True
            assert target.read_text() == "new content"
            assert not list(Path(temp_dir).glob("*.kader-tmp"))

    def test_temp_file_exists_during_edit_and_is_removed_after(self):
        """edit() must also create and clean up the .kader-tmp mid-flight."""
        with tempfile.TemporaryDirectory() as temp_dir:
            backend = FilesystemBackend(root_dir=temp_dir)
            target = Path(temp_dir) / "atomic_edit.txt"
            target.write_text("old content")
            captured: dict[str, bool] = {"saw_tmp": False}

            original_replace = os.replace

            def capturing_replace(src: str, dst: str) -> None:
                src_path = Path(src)
                if src_path.suffix.endswith(".kader-tmp"):
                    captured["saw_tmp"] = True
                    assert src_path.exists()
                    assert src_path.read_text() == "new content"
                original_replace(src, dst)

            with patch("os.replace", side_effect=capturing_replace):
                result = backend.edit("atomic_edit.txt", "old", "new")

            assert result.error is None
            assert result.occurrences == 1
            assert captured["saw_tmp"] is True
            assert target.read_text() == "new content"
            assert not list(Path(temp_dir).glob("*.kader-tmp"))


class TestFreshnessTokens:
    """Tests for expected_mtime freshness checks."""

    def test_edit_with_matching_mtime_succeeds(self):
        """edit() succeeds when the supplied mtime token matches."""
        with tempfile.TemporaryDirectory() as temp_dir:
            backend = FilesystemBackend(root_dir=temp_dir)
            target = Path(temp_dir) / "fresh.txt"
            target.write_text("hello world")

            mtime_ns = target.stat().st_mtime_ns
            result = backend.edit(
                "fresh.txt", "world", "universe", expected_mtime=mtime_ns
            )

            assert result.error is None
            assert result.occurrences == 1
            assert target.read_text() == "hello universe"

    def test_edit_with_stale_mtime_fails(self):
        """edit() rejects the change when the file has been modified externally."""
        with tempfile.TemporaryDirectory() as temp_dir:
            backend = FilesystemBackend(root_dir=temp_dir)
            target = Path(temp_dir) / "stale.txt"
            target.write_text("hello world")

            original_mtime = target.stat().st_mtime_ns
            # Simulate external modification after the read
            time.sleep(0.01)
            target.write_text("hello planet")

            result = backend.edit(
                "stale.txt",
                "world",
                "universe",
                expected_mtime=original_mtime,
            )

            assert result.error is not None
            assert "stale_file" in result.error
            assert target.read_text() == "hello planet"

    def test_read_includes_mtime_footer_when_enabled(self):
        """read() appends an mtime footer when include_mtime=True."""
        with tempfile.TemporaryDirectory() as temp_dir:
            backend = FilesystemBackend(root_dir=temp_dir, include_mtime=True)
            target = Path(temp_dir) / "mtime.txt"
            target.write_text("line one\nline two")

            result = backend.read("mtime.txt")

            assert "line one" in result
            expected_mtime = target.stat().st_mtime_ns
            assert f"# kader-mtime: {expected_mtime}" in result

    def test_read_omits_mtime_footer_by_default(self):
        """read() does not append an mtime footer by default."""
        with tempfile.TemporaryDirectory() as temp_dir:
            backend = FilesystemBackend(root_dir=temp_dir)
            target = Path(temp_dir) / "nomtime.txt"
            target.write_text("content")

            result = backend.read("nomtime.txt")

            assert "kader-mtime" not in result


class TestFileLocking:
    """Tests for advisory file locking around edits."""

    def test_lock_file_created_and_released(self):
        """edit() must acquire and release the advisory lock file."""
        if fcntl is None:
            pytest.skip("fcntl not available on this platform")

        with tempfile.TemporaryDirectory() as temp_dir:
            backend = FilesystemBackend(root_dir=temp_dir, use_file_lock=True)
            target = Path(temp_dir) / "locked.txt"
            target.write_text("hello world")

            recorded = []
            original_flock = os.flock if hasattr(os, "flock") else None

            def fake_flock(fd, op):
                if op == fcntl.LOCK_EX:
                    recorded.append("lock")
                elif op == fcntl.LOCK_UN:
                    recorded.append("unlock")
                if original_flock is not None:
                    original_flock(fd, op)

            with patch("fcntl.flock", side_effect=fake_flock):
                result = backend.edit("locked.txt", "world", "universe")

            assert result.error is None
            assert "lock" in recorded
            assert "unlock" in recorded

    def test_concurrent_edits_do_not_corrupt_file(self):
        """Concurrent edits to different parts of a file must all be preserved."""
        with tempfile.TemporaryDirectory() as temp_dir:
            backend = FilesystemBackend(root_dir=temp_dir, use_file_lock=True)
            target = Path(temp_dir) / "concurrent.txt"
            target.write_text("A\nB\nC\n")

            errors = []

            def replace_line(old_line: str, new_line: str) -> None:
                result = backend.edit(
                    "concurrent.txt", old_line + "\n", new_line + "\n"
                )
                if result.error:
                    errors.append(result.error)

            threads = [
                threading.Thread(target=replace_line, args=("A", "alpha")),
                threading.Thread(target=replace_line, args=("B", "beta")),
                threading.Thread(target=replace_line, args=("C", "gamma")),
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert not errors
            content = target.read_text()
            assert "alpha" in content
            assert "beta" in content
            assert "gamma" in content
            assert "A\n" not in content
            assert "B\n" not in content
            assert "C\n" not in content

    def test_lock_file_is_stable_after_edit(self):
        """The sibling lock file must remain as a stable inode after edit."""
        with tempfile.TemporaryDirectory() as temp_dir:
            backend = FilesystemBackend(root_dir=temp_dir, use_file_lock=True)
            target = Path(temp_dir) / "clean.txt"
            target.write_text("hello world")

            result = backend.edit("clean.txt", "world", "universe")

            assert result.error is None
            assert (Path(temp_dir) / ".kader-lock.clean.txt").exists()

    def test_lock_file_is_stable_on_edit_failure(self):
        """The stable sibling lock file must remain when edit() fails."""
        with tempfile.TemporaryDirectory() as temp_dir:
            backend = FilesystemBackend(root_dir=temp_dir, use_file_lock=True)
            target = Path(temp_dir) / "fail_clean.txt"
            target.write_text("hello world")

            result = backend.edit("fail_clean.txt", "nonexistent", "replacement")

            assert result.error is not None
            assert (Path(temp_dir) / ".kader-lock.fail_clean.txt").exists()
