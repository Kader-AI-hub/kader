"""
Hardening tests for FilesystemBackend race-safety features.

Covers Stage 1: atomic writes, advisory file locks, and freshness tokens.
"""

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

from kader.tools.filesystem import FilesystemBackend


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

            with patch("os.replace", side_effect=boom):
                result = backend.write("fail.txt", "new content")

            assert result.error is not None
            assert "replace refused" in result.error
            assert not target.exists()
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

    def test_lock_file_is_removed_after_edit(self):
        """No stale .kader-lock file must remain after edit() finishes."""
        with tempfile.TemporaryDirectory() as temp_dir:
            backend = FilesystemBackend(root_dir=temp_dir, use_file_lock=True)
            target = Path(temp_dir) / "clean.txt"
            target.write_text("hello world")

            result = backend.edit("clean.txt", "world", "universe")

            assert result.error is None
            assert not (Path(temp_dir) / ".kader-lock.clean.txt").exists()

    def test_lock_file_is_removed_even_on_edit_failure(self):
        """No stale .kader-lock file must remain when edit() fails."""
        with tempfile.TemporaryDirectory() as temp_dir:
            backend = FilesystemBackend(root_dir=temp_dir, use_file_lock=True)
            target = Path(temp_dir) / "fail_clean.txt"
            target.write_text("hello world")

            result = backend.edit("fail_clean.txt", "nonexistent", "replacement")

            assert result.error is not None
            assert not (Path(temp_dir) / ".kader-lock.fail_clean.txt").exists()
