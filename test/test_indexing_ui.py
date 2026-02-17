#!/usr/bin/env python3
"""Tests for IndexingUI — the unified terminal output for build_index().

All tests use io.StringIO as the output stream to capture terminal output
without needing a real terminal. No ML models or filesystem indexing required.
"""
import io
import logging
import re
import sys
import threading
import time
from pathlib import Path

import pytest

_ANSI_RE = re.compile(r"\033\[[0-9;]*m")


class _FakeTTY(io.StringIO):
    """StringIO that pretends to be a TTY for interactive output testing."""
    def isatty(self):
        return True


def _make_ui(stream=None):
    """Create an IndexingUI with a local import (avoids module-level config init)."""
    from chunksilo.index import IndexingUI
    return IndexingUI(stream=stream or _FakeTTY())


@pytest.fixture
def ui():
    """Create an IndexingUI that writes to a fake TTY buffer."""
    return _make_ui()


def _output(ui) -> str:
    """Return all output written so far, with ANSI escape codes stripped."""
    return _ANSI_RE.sub("", ui._stream.getvalue())


def _raw_output(ui) -> str:
    """Return all raw output written so far, including ANSI codes."""
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

    def test_progress_update_preserves_file(self, ui):
        """progress_update does not clear the current file sub-line."""
        ui.progress_start(10)
        ui.progress_set_file("/path/to/report.pdf", "Loading")
        ui.progress_update(1)
        output = _output(ui)
        assert "report.pdf" in output

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


class TestOutputSuppression:
    def test_redirects_stdout_during_context(self):
        """sys.stdout is redirected during context, restored after."""
        stream = io.StringIO()
        ui = _make_ui(stream)
        orig_stdout = sys.stdout

        with ui:
            assert sys.stdout is not orig_stdout
            # UI still writes to its captured stream
            ui.print("visible")

        assert sys.stdout is orig_stdout
        assert "visible" in stream.getvalue()

    def test_redirects_stderr_during_context(self):
        """sys.stderr is redirected during context, restored after."""
        stream = io.StringIO()
        ui = _make_ui(stream)
        orig_stderr = sys.stderr

        with ui:
            assert sys.stderr is not orig_stderr

        assert sys.stderr is orig_stderr

    def test_silences_root_stream_handlers(self):
        """Root logger StreamHandlers are silenced during context."""
        stream = io.StringIO()
        ui = _make_ui(stream)

        log_capture = io.StringIO()
        handler = logging.StreamHandler(log_capture)
        handler.setLevel(logging.DEBUG)
        logging.root.addHandler(handler)

        try:
            with ui:
                third_party = logging.getLogger("some.third.party")
                third_party.info("should be suppressed")

            assert "should be suppressed" not in log_capture.getvalue()
        finally:
            logging.root.removeHandler(handler)

    def test_restores_handler_levels_after_exit(self):
        """Root handler levels are restored after context manager exits."""
        log_capture = io.StringIO()
        handler = logging.StreamHandler(log_capture)
        handler.setLevel(logging.DEBUG)
        logging.root.addHandler(handler)

        try:
            stream = io.StringIO()
            ui = _make_ui(stream)

            with ui:
                assert handler.level > logging.CRITICAL

            assert handler.level == logging.DEBUG
        finally:
            logging.root.removeHandler(handler)

    def test_restores_streams_on_exception(self):
        """stdout/stderr are restored even if an exception occurs."""
        stream = io.StringIO()
        ui = _make_ui(stream)
        orig_stdout = sys.stdout
        orig_stderr = sys.stderr

        with pytest.raises(ValueError):
            with ui:
                raise ValueError("test error")

        assert sys.stdout is orig_stdout
        assert sys.stderr is orig_stderr

    def test_verbose_skips_suppression(self):
        """verbose=True leaves stdout/stderr untouched."""
        from chunksilo.index import IndexingUI
        stream = io.StringIO()
        ui = IndexingUI(stream=stream, verbose=True)
        orig_stdout = sys.stdout
        orig_stderr = sys.stderr

        with ui:
            assert sys.stdout is orig_stdout
            assert sys.stderr is orig_stderr


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

    def test_success_outputs_message(self, ui):
        """ui.success() writes message with newline (plain in non-TTY)."""
        ui.success("All done")
        output = _output(ui)
        assert "All done\n" in output

    def test_error_outputs_message(self, ui):
        """ui.error() writes message with newline (plain in non-TTY)."""
        ui.error("Something failed")
        output = _output(ui)
        assert "Something failed\n" in output


# =============================================================================
# Color support tests
# =============================================================================


def _make_non_tty_ui():
    """Create an IndexingUI with a plain StringIO (non-TTY) stream."""
    from chunksilo.index import IndexingUI
    return IndexingUI(stream=io.StringIO())


class TestColorSupport:
    def test_non_tty_has_no_color_codes(self):
        """Non-TTY stream (StringIO) produces no ANSI color codes."""
        ui = _make_non_tty_ui()
        assert ui.GREEN == ""
        assert ui.RESET == ""
        assert ui.BOLD == ""
        assert ui.DIM == ""

    def test_tty_has_color_codes(self):
        """TTY stream produces ANSI color codes."""
        ui = _make_ui()
        assert ui.GREEN == "\033[32m"
        assert ui.RESET == "\033[0m"
        assert ui.BOLD == "\033[1m"
        assert ui.CYAN == "\033[36m"

    def test_tty_step_done_has_green_suffix(self):
        """On TTY, step_done('done') includes green ANSI code."""
        ui = _make_ui()
        ui.step_start("Loading")
        time.sleep(0.05)
        ui.step_done()
        output = _raw_output(ui)
        assert "\033[32mdone\033[0m" in output

    def test_tty_step_done_skipped_has_yellow(self):
        """On TTY, step_done('skipped') includes yellow ANSI code."""
        ui = _make_ui()
        ui.step_start("Check")
        time.sleep(0.05)
        ui.step_done("skipped")
        output = _raw_output(ui)
        assert "\033[33mskipped\033[0m" in output

    def test_tty_step_done_interrupted_has_red(self):
        """On TTY, step_done('interrupted') includes red ANSI code."""
        ui = _make_ui()
        ui.step_start("Running")
        time.sleep(0.05)
        ui.step_done("interrupted")
        output = _raw_output(ui)
        assert "\033[31minterrupted\033[0m" in output

    def test_tty_progress_bar_uses_block_chars(self):
        """On TTY, progress bar uses █ and ░ instead of # and -."""
        ui = _make_ui()
        ui.progress_start(10, "Files")
        ui.progress_update(5)
        output = _raw_output(ui)
        assert "█" in output
        assert "░" in output
        assert "#" not in output
        assert "-" not in output.split("]")[0]  # no dash in bar area

    def test_tty_success_has_bold_green(self):
        """On TTY, success() wraps message in bold green."""
        ui = _make_ui()
        ui.success("Done!")
        output = _raw_output(ui)
        assert "\033[1;32mDone!\033[0m" in output

    def test_tty_error_has_red(self):
        """On TTY, error() wraps message in red."""
        ui = _make_ui()
        ui.error("Failed!")
        output = _raw_output(ui)
        assert "\033[31mFailed!\033[0m" in output


# =============================================================================
# Spinner clear-to-end-of-line tests
# =============================================================================


class TestSpinnerClearLine:
    def test_spinner_writes_include_clear_eol(self):
        """Spinner animation includes \\033[K to prevent traces."""
        ui = _make_ui()
        ui.step_start("Working")
        # Let spinner write at least one frame
        time.sleep(0.15)
        ui.step_done()
        output = _raw_output(ui)
        # Spinner frames should include \033[K (clear to end of line)
        # Find a spinner character followed by the clear code
        spinner_chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        has_clear = any(
            f"{ch}\033[0m\033[K" in output
            for ch in spinner_chars
        )
        assert has_clear, "Spinner frames should include \\033[K to clear line remnants"


# =============================================================================
# Non-TTY (CI) mode tests
# =============================================================================


class TestNonTTYMode:
    """In non-TTY mode (CI), output should be simple static lines with no animation."""

    def test_step_no_spinner(self):
        """Non-TTY step_start + step_done produces a single line, no spinner."""
        ui = _make_non_tty_ui()
        ui.step_start("Embedding 375 nodes")
        time.sleep(0.15)
        ui.step_done()
        output = ui._stream.getvalue()
        assert output == "Embedding 375 nodes... done\n"

    def test_step_no_ansi_escapes(self):
        """Non-TTY step output contains no ANSI escape codes."""
        ui = _make_non_tty_ui()
        ui.step_start("Loading")
        ui.step_done("skipped")
        output = ui._stream.getvalue()
        assert "\033[" not in output
        assert "\r" not in output

    def test_progress_prints_milestones(self):
        """Non-TTY progress prints at 10% intervals, not every update."""
        ui = _make_non_tty_ui()
        ui.progress_start(100, "Processing")
        for _ in range(100):
            ui.progress_update(1)
        ui.progress_done()
        output = ui._stream.getvalue()
        lines = [l for l in output.strip().split("\n") if l]
        # 0%, 10%, 20%, ... 100% = 11 lines
        assert len(lines) == 11
        assert "Processing [0/100] 0%" in lines[0]
        assert "Processing [100/100] 100%" in lines[-1]

    def test_progress_no_ansi_escapes(self):
        """Non-TTY progress output contains no ANSI escape codes."""
        ui = _make_non_tty_ui()
        ui.progress_start(10, "Files")
        ui.progress_update(10)
        ui.progress_done()
        output = ui._stream.getvalue()
        assert "\033[" not in output
        assert "\r" not in output


# =============================================================================
# step_update() tests (Issue #38)
# =============================================================================


class TestStepUpdate:
    def test_step_update_changes_spinner_message(self, ui):
        """step_update changes the message shown by the spinner."""
        ui.step_start("Scanning for changes")
        time.sleep(0.15)
        ui.step_update("Scanning for changes (100 files)")
        time.sleep(0.15)
        output = _output(ui)
        assert "100 files" in output
        ui.step_done()

    def test_step_update_noop_when_no_active_step(self, ui):
        """step_update is a no-op when no step is active."""
        ui.step_update("Should not crash")
        assert ui._step_message is None

    def test_step_done_uses_original_message(self, ui):
        """step_done uses the original step_start message, not the updated one."""
        ui.step_start("Scanning for changes")
        time.sleep(0.15)
        ui.step_update("Scanning for changes (500 files)")
        time.sleep(0.15)
        ui.step_done("3 new")
        output = _output(ui)
        # The final line should use the original message
        assert "Scanning for changes... 3 new" in output
        # The final \r-written line (last line) should not contain "500 files"
        # On TTY, \r overwrites the line so we check the last \r-segment
        raw = ui._stream.getvalue()
        last_segment = raw.rsplit("\r", 1)[-1]
        last_segment_clean = _ANSI_RE.sub("", last_segment)
        assert "500 files" not in last_segment_clean
        assert "Scanning for changes... 3 new" in last_segment_clean

    def test_step_done_clears_original_message(self, ui):
        """step_done clears both _step_message and _step_original_message."""
        ui.step_start("Loading")
        ui.step_done()
        assert ui._step_message is None
        assert ui._step_original_message is None

    def test_step_update_non_tty(self):
        """step_update on non-TTY is a no-op (message not re-printed)."""
        ui = _make_non_tty_ui()
        ui.step_start("Scanning")
        ui.step_update("Scanning (200 files)")
        ui.step_done("done")
        output = ui._stream.getvalue()
        # Non-TTY prints "Scanning... done\n" — no dynamic update
        assert output == "Scanning... done\n"


# =============================================================================
# PDF heading extraction timeout tests (Issue #39)
# =============================================================================


class TestPdfHeadingExtraction:
    def test_timeout_event_stops_extraction(self):
        """_extract_pdf_headings_from_outline respects timeout_event."""
        from unittest.mock import MagicMock, patch

        from chunksilo.index import _extract_pdf_headings_from_outline

        # Create a mock PDF reader with an outline
        mock_reader = MagicMock()
        mock_item_1 = MagicMock()
        mock_item_1.title = "Chapter 1"
        mock_item_2 = MagicMock()
        mock_item_2.title = "Chapter 2"
        mock_reader.outline = [mock_item_1, mock_item_2]
        mock_reader.get_destination_page_number.return_value = 0
        mock_reader.pages = []

        # Set timeout_event before calling — should return empty
        timeout_event = threading.Event()
        timeout_event.set()

        with patch("pypdf.PdfReader", return_value=mock_reader):
            headings = _extract_pdf_headings_from_outline(
                Path("/fake/doc.pdf"), timeout_event=timeout_event
            )

        # Should return 0 headings because timeout was already set
        assert len(headings) == 0

    def test_page_text_caching_avoids_redundant_calls(self):
        """Page text is extracted once per page, not once per heading."""
        from unittest.mock import MagicMock, patch

        from chunksilo.index import _extract_pdf_headings_from_outline

        mock_reader = MagicMock()
        # 3 headings, all on page 2 (so pages 0 and 1 must be read)
        items = []
        for i in range(3):
            item = MagicMock()
            item.title = f"Heading {i}"
            items.append(item)
        mock_reader.outline = items
        mock_reader.get_destination_page_number.return_value = 2

        mock_page_0 = MagicMock()
        mock_page_0.extract_text.return_value = "Page zero text"
        mock_page_1 = MagicMock()
        mock_page_1.extract_text.return_value = "Page one text"
        mock_page_2 = MagicMock()
        mock_page_2.extract_text.return_value = "Page two text"
        mock_reader.pages = [mock_page_0, mock_page_1, mock_page_2]

        with patch("pypdf.PdfReader", return_value=mock_reader):
            headings = _extract_pdf_headings_from_outline(Path("/fake/doc.pdf"))

        assert len(headings) == 3
        # Each page should only be extracted once despite 3 headings
        assert mock_page_0.extract_text.call_count == 1
        assert mock_page_1.extract_text.call_count == 1

    def test_wall_clock_safety_net(self):
        """Extraction stops when wall-clock max_seconds is exceeded."""
        from unittest.mock import MagicMock, patch

        from chunksilo.index import _extract_pdf_headings_from_outline

        mock_reader = MagicMock()
        items = []
        for i in range(100):
            item = MagicMock()
            item.title = f"Heading {i}"
            items.append(item)
        mock_reader.outline = items
        mock_reader.get_destination_page_number.return_value = 0
        mock_reader.pages = []

        with patch("pypdf.PdfReader", return_value=mock_reader):
            # max_seconds=0 should return immediately
            headings = _extract_pdf_headings_from_outline(
                Path("/fake/doc.pdf"), max_seconds=0
            )

        assert len(headings) == 0


# =============================================================================
# FileProcessingContext timeout tests (Issue #39)
# =============================================================================


class TestFileProcessingContextTimeout:
    def test_check_timeout_raises_on_expired(self):
        """check_timeout raises FileProcessingTimeoutError when time is up."""
        from chunksilo.index import FileProcessingContext, FileProcessingTimeoutError

        ui = _make_ui()
        ui.progress_start(1)
        ctx = FileProcessingContext("test.pdf", ui, timeout_seconds=0.01)
        ctx._start_time = time.time() - 1  # pretend 1 second has passed
        ctx._timeout_event = threading.Event()

        with pytest.raises(FileProcessingTimeoutError):
            ctx.check_timeout()

    def test_check_timeout_raises_when_event_set(self):
        """check_timeout raises when _timeout_event is already set."""
        from chunksilo.index import FileProcessingContext, FileProcessingTimeoutError

        ui = _make_ui()
        ui.progress_start(1)
        ctx = FileProcessingContext("test.pdf", ui, timeout_seconds=300)
        ctx._start_time = time.time()
        ctx._timeout_event = threading.Event()
        ctx._timeout_event.set()

        with pytest.raises(FileProcessingTimeoutError):
            ctx.check_timeout()

    def test_remaining_seconds_returns_positive(self):
        """remaining_seconds returns positive value when time remains."""
        from chunksilo.index import FileProcessingContext

        ui = _make_ui()
        ui.progress_start(1)
        ctx = FileProcessingContext("test.pdf", ui, timeout_seconds=300)
        ctx._start_time = time.time()

        remaining = ctx.remaining_seconds()
        assert remaining is not None
        assert remaining > 290

    def test_remaining_seconds_returns_none_without_timeout(self):
        """remaining_seconds returns None when no timeout is configured."""
        from chunksilo.index import FileProcessingContext

        ui = _make_ui()
        ui.progress_start(1)
        ctx = FileProcessingContext("test.pdf", ui, timeout_seconds=None)
        ctx._start_time = time.time()

        assert ctx.remaining_seconds() is None

    def test_remaining_seconds_clamps_to_zero(self):
        """remaining_seconds returns 0 when timeout has passed."""
        from chunksilo.index import FileProcessingContext

        ui = _make_ui()
        ui.progress_start(1)
        ctx = FileProcessingContext("test.pdf", ui, timeout_seconds=1)
        ctx._start_time = time.time() - 10  # 10 seconds ago

        assert ctx.remaining_seconds() == 0.0


# =============================================================================
# _load_data_with_timeout tests (Issue #39)
# =============================================================================


class TestLoadDataWithTimeout:
    def test_returns_docs_on_success(self):
        """_load_data_with_timeout returns docs when load_data succeeds."""
        from unittest.mock import MagicMock
        from chunksilo.index import _load_data_with_timeout

        mock_reader = MagicMock()
        mock_reader.load_data.return_value = ["doc1", "doc2"]

        result = _load_data_with_timeout(mock_reader, timeout_seconds=5.0)
        assert result == ["doc1", "doc2"]

    def test_returns_empty_on_timeout(self):
        """_load_data_with_timeout returns [] when load_data hangs."""
        from unittest.mock import MagicMock
        from chunksilo.index import _load_data_with_timeout

        mock_reader = MagicMock()
        mock_reader.load_data.side_effect = lambda: time.sleep(10)

        result = _load_data_with_timeout(mock_reader, timeout_seconds=0.1)
        assert result == []


# =============================================================================
# _split_docx_with_timeout tests (Issue #39)
# =============================================================================


class TestSplitDocxWithTimeout:
    def test_returns_docs_on_success(self):
        """_split_docx_with_timeout returns docs when processing succeeds."""
        from unittest.mock import patch, MagicMock
        from chunksilo.index import _split_docx_with_timeout

        mock_doc = MagicMock()
        with patch(
            "chunksilo.index.split_docx_into_heading_documents",
            return_value=[mock_doc],
        ):
            result = _split_docx_with_timeout(Path("/fake/doc.docx"), None, 5.0)

        assert result == [mock_doc]

    def test_returns_empty_on_timeout(self):
        """_split_docx_with_timeout returns [] when processing hangs."""
        from unittest.mock import patch
        from chunksilo.index import _split_docx_with_timeout

        def hang(*args, **kwargs):
            time.sleep(10)

        with patch(
            "chunksilo.index.split_docx_into_heading_documents",
            side_effect=hang,
        ):
            result = _split_docx_with_timeout(Path("/fake/doc.docx"), None, 0.1)

        assert result == []
