#!/usr/bin/env python3
"""Tests for scan-phase timeout protection.

Verifies that filesystem operations during the scanning phase (directory
traversal, stat, MD5 hashing, file existence checks) cannot hang indefinitely
when a network mount becomes unresponsive.

Uses threading.Event + time.sleep to simulate blocking filesystem calls
without needing a real stalled network mount.
"""
import hashlib
import os
import threading
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from chunksilo.index import (
    _run_with_timeout,
    _SCAN_TIMEOUT_SENTINEL,
    DirectoryConfig,
    FileInfo,
    IndexConfig,
    LocalFileSystemSource,
    MultiDirectoryDataSource,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _block_forever(*_args, **_kwargs):
    """Simulate a blocking syscall that never returns."""
    threading.Event().wait(timeout=60)


def _make_source(tmp_path, **overrides):
    """Create a LocalFileSystemSource pointing at tmp_path."""
    cfg = DirectoryConfig(path=tmp_path, **overrides)
    return LocalFileSystemSource(cfg)


# ===========================================================================
# TestRunWithTimeout — tests for the _run_with_timeout() helper itself
# ===========================================================================


class TestRunWithTimeout:
    def test_fast_call_returns_result(self):
        """Callable that returns immediately yields its value."""
        assert _run_with_timeout(lambda: 42, timeout_seconds=5) == 42

    def test_slow_call_times_out(self):
        """Callable that sleeps longer than timeout returns the default."""
        result = _run_with_timeout(
            lambda: threading.Event().wait(60) or "never",
            timeout_seconds=1,
            default="timed_out",
        )
        assert result == "timed_out"

    def test_slow_call_returns_sentinel_by_default(self):
        """When no default is given, the sentinel object is returned."""
        result = _run_with_timeout(
            lambda: threading.Event().wait(60),
            timeout_seconds=1,
        )
        assert result is _SCAN_TIMEOUT_SENTINEL

    def test_exception_propagates(self):
        """Callable that raises an exception propagates it."""
        def _raise():
            raise ValueError("boom")

        with pytest.raises(ValueError, match="boom"):
            _run_with_timeout(_raise, timeout_seconds=5)


# ===========================================================================
# TestIsAvailableTimeout — is_available() doesn't hang on unresponsive mounts
# ===========================================================================


class TestIsAvailableTimeout:
    @patch("chunksilo.index.cfgload")
    def test_is_available_returns_false_on_timeout(self, mock_cfg, tmp_path):
        """Patch Path.exists() to block; is_available() returns False within
        a reasonable time rather than hanging for 60s."""
        mock_cfg.get.return_value = 2  # 2-second timeout for fast test

        source = _make_source(tmp_path)

        # Patch the specific base_dir's exists to block
        original_exists = Path.exists
        def _blocking_exists(self_path):
            if self_path == tmp_path:
                threading.Event().wait(60)
            return original_exists(self_path)

        with patch.object(Path, "exists", _blocking_exists):
            start = time.monotonic()
            result = source.is_available()
            elapsed = time.monotonic() - start

        assert result is False
        assert elapsed < 10, f"is_available() took {elapsed:.1f}s, expected < 10s"

    def test_is_available_returns_true_when_responsive(self, tmp_path):
        """Normal directory on tmp_path still returns True (no false positives)."""
        (tmp_path / "file.txt").write_text("hello")
        source = _make_source(tmp_path)
        assert source.is_available() is True


# ===========================================================================
# TestCreateFileInfoTimeout — _create_file_info() doesn't hang on stat/read
# ===========================================================================


class TestCreateFileInfoTimeout:
    @patch("chunksilo.index.cfgload")
    def test_stat_hang_skips_file(self, mock_cfg, tmp_path):
        """When Path.stat blocks, iter_files skips the file with a warning."""
        mock_cfg.get.return_value = 2  # 2s timeout

        f = tmp_path / "test.txt"
        f.write_text("data")
        source = _make_source(tmp_path)

        original_stat = Path.stat
        def _blocking_stat(self_path, *a, **kw):
            if self_path == f:
                threading.Event().wait(60)
            return original_stat(self_path, *a, **kw)

        with patch.object(Path, "stat", _blocking_stat):
            start = time.monotonic()
            results = list(source.iter_files())
            elapsed = time.monotonic() - start

        assert len(results) == 0
        assert elapsed < 15, f"iter_files() took {elapsed:.1f}s, expected < 15s"

    @patch("chunksilo.index.cfgload")
    def test_hash_read_hang_skips_file(self, mock_cfg, tmp_path):
        """When open().read() blocks during MD5 hashing, the file is skipped."""
        mock_cfg.get.return_value = 2  # 2s timeout

        f = tmp_path / "test.txt"
        f.write_text("data")
        source = _make_source(tmp_path)

        original_open = open
        def _blocking_open(path, *args, **kwargs):
            fh = original_open(path, *args, **kwargs)
            if str(path) == str(f) and "b" in (args[0] if args else ""):
                original_read = fh.read
                def _blocking_read(*a, **kw):
                    threading.Event().wait(60)
                    return original_read(*a, **kw)
                fh.read = _blocking_read
            return fh

        with patch("builtins.open", _blocking_open):
            start = time.monotonic()
            results = list(source.iter_files())
            elapsed = time.monotonic() - start

        assert len(results) == 0
        assert elapsed < 15, f"iter_files() took {elapsed:.1f}s, expected < 15s"

    def test_normal_files_unaffected(self, tmp_path):
        """Normal local files are returned with correct hashes."""
        files = {}
        for name in ("a.txt", "b.txt", "c.txt"):
            p = tmp_path / name
            p.write_text(f"content of {name}")
            h = hashlib.md5(p.read_bytes()).hexdigest()
            files[str(p.absolute())] = h

        source = _make_source(tmp_path)
        results = list(source.iter_files())

        assert len(results) == 3
        for fi in results:
            assert fi.hash == files[fi.path], f"Hash mismatch for {fi.path}"


# ===========================================================================
# TestIterFilesWithTimeout — os.walk() hangs are recovered from
# ===========================================================================


class TestIterFilesWithTimeout:
    @patch("chunksilo.index.cfgload")
    def test_walk_hang_on_subdirectory_recovers(self, mock_cfg, tmp_path):
        """When os.walk blocks on a subdirectory, iter_files eventually stops
        rather than hanging forever."""
        mock_cfg.get.return_value = 2  # 2s timeout

        # Create dir structure: subA/file_a.txt, subB/ (will hang)
        sub_a = tmp_path / "subA"
        sub_a.mkdir()
        (sub_a / "file_a.txt").write_text("aaa")
        sub_b = tmp_path / "subB"
        sub_b.mkdir()
        (sub_b / "file_b.txt").write_text("bbb")

        source = _make_source(tmp_path)

        # Replace os.walk to yield first entry normally, then block
        original_walk = os.walk

        def _stalling_walk(top, **kw):
            call_count = 0
            for entry in original_walk(top, **kw):
                call_count += 1
                yield entry
                if call_count >= 2:  # After yielding top + subA, block
                    threading.Event().wait(60)

        with patch("os.walk", _stalling_walk):
            start = time.monotonic()
            results = list(source.iter_files())
            elapsed = time.monotonic() - start

        # Should have at least the file from subA (and maybe top-level)
        # but should NOT hang for 60s
        assert elapsed < 15, f"iter_files() took {elapsed:.1f}s, expected < 15s"

    def test_all_walk_results_returned_when_fast(self, tmp_path):
        """Normal tmp_path with nested dirs returns all files."""
        sub = tmp_path / "nested"
        sub.mkdir()
        (tmp_path / "top.txt").write_text("top")
        (sub / "deep.txt").write_text("deep")

        source = _make_source(tmp_path)
        results = list(source.iter_files())

        paths = {fi.path for fi in results}
        assert str((tmp_path / "top.txt").absolute()) in paths
        assert str((sub / "deep.txt").absolute()) in paths


# ===========================================================================
# TestMultiDirectorySourceTimeout — stalled source doesn't block others
# ===========================================================================


class TestMultiDirectorySourceTimeout:
    def test_stalled_source_does_not_block_others(self, tmp_path):
        """One directory is responsive, the other's is_available blocks.
        Only the working directory's files are indexed."""
        good_dir = tmp_path / "good"
        good_dir.mkdir()
        (good_dir / "file.txt").write_text("hello")

        bad_dir = tmp_path / "bad"
        bad_dir.mkdir()

        config = IndexConfig(
            directories=[
                DirectoryConfig(path=good_dir),
                DirectoryConfig(path=bad_dir),
            ]
        )

        original_is_available = LocalFileSystemSource.is_available

        def _patched_is_available(self):
            if self.base_dir == bad_dir:
                # Simulate stalled mount — return False (as timeout would)
                return False
            return original_is_available(self)

        with patch.object(LocalFileSystemSource, "is_available", _patched_is_available):
            mds = MultiDirectoryDataSource(config)

        assert len(mds.sources) == 1
        assert len(mds.unavailable_dirs) == 1

        results = list(mds.iter_files())
        assert len(results) == 1
        assert results[0].path == str((good_dir / "file.txt").absolute())

    @patch("chunksilo.index.cfgload")
    def test_stalled_source_during_scan_skips_to_next(self, mock_cfg, tmp_path):
        """Both directories pass is_available(), but one's os.walk hangs.
        Files from the other source are still returned."""
        mock_cfg.get.return_value = 2  # 2s timeout

        good_dir = tmp_path / "good"
        good_dir.mkdir()
        (good_dir / "file.txt").write_text("hello")

        bad_dir = tmp_path / "bad"
        bad_dir.mkdir()
        (bad_dir / "file.txt").write_text("stale")

        config = IndexConfig(
            directories=[
                DirectoryConfig(path=good_dir),
                DirectoryConfig(path=bad_dir),
            ]
        )

        mds = MultiDirectoryDataSource(config)
        assert len(mds.sources) == 2

        # Patch os.walk so walks into bad_dir hang, triggering queue timeout
        original_walk = os.walk

        def _stalling_walk(top, **kw):
            if str(top) == str(bad_dir):
                # Block immediately — queue timeout will abort iteration
                threading.Event().wait(60)
                return
                yield  # make it a generator
            yield from original_walk(top, **kw)

        with patch("os.walk", _stalling_walk):
            start = time.monotonic()
            results = list(mds.iter_files())
            elapsed = time.monotonic() - start

        good_paths = {fi.path for fi in results}
        assert str((good_dir / "file.txt").absolute()) in good_paths
        assert elapsed < 15, f"iter_files() took {elapsed:.1f}s, expected < 15s"


# ===========================================================================
# TestLoadFileExistsTimeout — load_file existence check doesn't hang
# ===========================================================================


class TestLoadFileExistsTimeout:
    @patch("chunksilo.index.cfgload")
    def test_exists_hang_skips_file(self, mock_cfg, tmp_path):
        """When Path.exists blocks on a file, load_file returns [] rather
        than hanging."""
        mock_cfg.get.return_value = 2  # 2s timeout

        f = tmp_path / "test.txt"
        f.write_text("data")
        source = _make_source(tmp_path)
        fi = FileInfo(
            path=str(f.absolute()),
            hash="abc",
            last_modified=0,
            source_dir=str(tmp_path.absolute()),
        )

        original_exists = Path.exists
        def _blocking_exists(self_path):
            if self_path == f:
                threading.Event().wait(60)
            return original_exists(self_path)

        with patch.object(Path, "exists", _blocking_exists):
            start = time.monotonic()
            result = source.load_file(fi)
            elapsed = time.monotonic() - start

        assert result == []
        assert elapsed < 10, f"load_file() took {elapsed:.1f}s, expected < 10s"
