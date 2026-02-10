#!/usr/bin/env python3
"""Tests for IndexingUI — the unified terminal output for build_index().

All tests use io.StringIO as the output stream to capture terminal output
without needing a real terminal. No ML models or filesystem indexing required.
"""
import io
import logging
import threading
import time

import pytest


def _make_ui(stream=None):
    """Create an IndexingUI with a local import (avoids module-level config init)."""
    from chunksilo.index import IndexingUI
    return IndexingUI(stream=stream or io.StringIO())


@pytest.fixture
def ui():
    """Create an IndexingUI that writes to a StringIO buffer."""
    return _make_ui()


def _output(ui) -> str:
    """Return all output written so far."""
    return ui._stream.getvalue()


# =============================================================================
# Step mode tests
# =============================================================================


class TestStepMode:
    def test_step_start_done_output(self, ui):
        """step_start + step_done produces 'message... done'."""
        ui.step_start("Loading")
        # Give spinner thread a moment to start
        time.sleep(0.15)
        ui.step_done()
        output = _output(ui)
        assert "Loading... done" in output

    def test_step_done_custom_suffix(self, ui):
        """step_done with custom suffix replaces 'done'."""
        ui.step_start("Checking cache")
        time.sleep(0.05)
        ui.step_done("skipped")
        output = _output(ui)
        assert "Checking cache... skipped" in output

    def test_step_done_clears_state(self, ui):
        """After step_done, internal step state is reset."""
        ui.step_start("Init")
        ui.step_done()
        assert ui._step_message is None
        assert ui._step_thread is None

    def test_multiple_steps_sequential(self, ui):
        """Multiple steps produce separate lines."""
        ui.step_start("Step one")
        ui.step_done()
        ui.step_start("Step two")
        ui.step_done("complete")
        output = _output(ui)
        assert "Step one... done" in output
        assert "Step two... complete" in output


# =============================================================================
# Progress mode tests
# =============================================================================


class TestProgressMode:
    def test_progress_start_renders_bar(self, ui):
        """progress_start renders the initial bar at 0%."""
        ui.progress_start(10, "Processing")
        output = _output(ui)
        assert "Processing" in output
        assert "0.0%" in output
        assert "(0/10)" in output

    def test_progress_update_advances(self, ui):
        """progress_update advances the count and percentage."""
        ui.progress_start(5)
        ui.progress_update(3)
        output = _output(ui)
        assert "60.0%" in output
        assert "(3/5)" in output

    def test_progress_update_clamps_to_total(self, ui):
        """progress_update does not exceed total."""
        ui.progress_start(3)
        ui.progress_update(10)
        assert ui._progress_current == 3
        output = _output(ui)
        assert "100.0%" in output

    def test_progress_set_file_shows_filename(self, ui):
        """progress_set_file displays the filename and phase."""
        ui.progress_start(10)
        ui.progress_set_file("/path/to/report.pdf", "Extracting text")
        output = _output(ui)
        assert "report.pdf" in output
        assert "Extracting text" in output

    def test_progress_set_file_truncates_long_names(self, ui):
        """Long filenames are truncated with ... prefix."""
        ui.progress_start(10)
        long_name = "a" * 60 + ".pdf"
        ui.progress_set_file(f"/path/to/{long_name}")
        output = _output(ui)
        assert "..." in output

    def test_progress_set_heartbeat(self, ui):
        """progress_set_heartbeat updates the animation character."""
        ui.progress_start(10)
        ui.progress_set_file("/path/to/file.txt", "Loading")
        ui.progress_set_heartbeat("⠹")
        output = _output(ui)
        assert "⠹" in output

    def test_progress_done_completes(self, ui):
        """progress_done exits progress mode cleanly."""
        ui.progress_start(2)
        ui.progress_update(2)
        ui.progress_done()
        assert not ui._progress_active
        assert not ui._progress_paused

    def test_progress_zero_total(self, ui):
        """progress_start with total=0 does not crash."""
        ui.progress_start(0)
        ui.progress_update()  # should be a no-op
        assert ui._progress_current == 0


# =============================================================================
# Pause/resume tests
# =============================================================================


class TestProgressPauseResume:
    def test_pause_resume_cycle(self, ui):
        """Pause hides bar, resume redraws it."""
        ui.progress_start(10, "Files")
        ui.progress_update(5)
        ui.progress_pause()
        assert ui._progress_paused
        assert not ui._progress_active

        # Step output while paused
        ui.step_start("Embedding")
        ui.step_done()

        ui.progress_resume()
        assert ui._progress_active
        assert not ui._progress_paused

        output = _output(ui)
        assert "Embedding... done" in output
        assert "50.0%" in output

    def test_pause_without_active_is_noop(self, ui):
        """Calling pause when not in progress mode does nothing."""
        ui.progress_pause()
        assert not ui._progress_paused

    def test_resume_without_pause_is_noop(self, ui):
        """Calling resume without prior pause does nothing."""
        ui.progress_start(5)
        ui.progress_resume()  # should not crash
        assert ui._progress_active


# =============================================================================
# Logging suppression tests
# =============================================================================


class TestLoggingSuppression:
    def test_suppresses_info_logs(self):
        """Inside context manager, INFO logs from chunksilo.index are suppressed."""
        index_logger = logging.getLogger("chunksilo.index")
        stream = io.StringIO()
        ui = _make_ui(stream)

        # Capture log output
        log_stream = io.StringIO()
        handler = logging.StreamHandler(log_stream)
        handler.setLevel(logging.DEBUG)
        index_logger.addHandler(handler)

        try:
            with ui:
                index_logger.info("this should be hidden")
                index_logger.warning("this should be visible")

            log_output = log_stream.getvalue()
            assert "this should be hidden" not in log_output
            assert "this should be visible" in log_output
        finally:
            index_logger.removeHandler(handler)

    def test_restores_logging_after_exit(self):
        """Logging level is restored after context manager exits."""
        index_logger = logging.getLogger("chunksilo.index")
        original_level = index_logger.level

        stream = io.StringIO()
        ui = _make_ui(stream)

        with ui:
            pass

        assert index_logger.level == original_level

    def test_restores_logging_on_exception(self):
        """Logging level is restored even if an exception occurs."""
        index_logger = logging.getLogger("chunksilo.index")
        original_level = index_logger.level

        stream = io.StringIO()
        ui = _make_ui(stream)

        with pytest.raises(ValueError):
            with ui:
                raise ValueError("test error")

        assert index_logger.level == original_level


# =============================================================================
# Thread safety tests
# =============================================================================


class TestThreadSafety:
    def test_concurrent_progress_updates(self, ui):
        """Multiple threads calling progress_update concurrently reach correct total."""
        total = 100
        ui.progress_start(total)

        errors = []

        def update_n(n):
            try:
                for _ in range(n):
                    ui.progress_update(1)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=update_n, args=(25,)) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Errors in threads: {errors}"
        assert ui._progress_current == total


# =============================================================================
# Context manager cleanup tests
# =============================================================================


class TestContextManagerCleanup:
    def test_cleanup_active_step(self):
        """Context manager cleans up an active step on exit."""
        stream = io.StringIO()
        ui = _make_ui(stream)

        with ui:
            ui.step_start("Running")
            # Don't call step_done — context manager should clean up

        output = stream.getvalue()
        assert "interrupted" in output
        assert ui._step_message is None

    def test_cleanup_active_progress(self):
        """Context manager cleans up active progress bar on exit."""
        stream = io.StringIO()
        ui = _make_ui(stream)

        with ui:
            ui.progress_start(10)
            ui.progress_update(5)
            # Don't call progress_done — context manager should clean up

        assert not ui._progress_active
        assert not ui._progress_paused

    def test_cleanup_paused_progress(self):
        """Context manager cleans up paused progress bar on exit."""
        stream = io.StringIO()
        ui = _make_ui(stream)

        with ui:
            ui.progress_start(10)
            ui.progress_pause()
            # Don't call progress_done — context manager should clean up

        assert not ui._progress_active
        assert not ui._progress_paused


# =============================================================================
# General output tests
# =============================================================================


class TestPrint:
    def test_print_outputs_message(self, ui):
        """ui.print() writes message with newline."""
        ui.print("Hello world")
        output = _output(ui)
        assert "Hello world\n" in output

    def test_print_multiple_messages(self, ui):
        """Multiple prints produce separate lines."""
        ui.print("Line 1")
        ui.print("Line 2")
        output = _output(ui)
        assert "Line 1\n" in output
        assert "Line 2\n" in output
