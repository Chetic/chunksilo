#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Terminal UI classes for the indexing pipeline.

Contains IndexingUI (progress bars, spinners, output suppression),
FileProcessingContext (per-file timeout and heartbeat), and GracefulAbort
(two-stage Ctrl-C handling).
"""
import logging
import signal
import sys
import threading
import time
from pathlib import Path

from . import cfgload

logger = logging.getLogger(__name__)


class _DevNull:
    """Minimal file-like sink that discards all writes."""

    def write(self, _data: str) -> int:
        return 0

    def flush(self) -> None:
        pass

    def isatty(self) -> bool:
        return False


class IndexingUI:
    """Unified terminal output for the indexing pipeline.

    Owns all stdout writes during build_index(). Provides two display modes:
    - Step mode: "message... done" with animated spinner
    - Progress mode: progress bar with optional sub-line for current file

    All methods are thread-safe via a single lock.
    """

    _SPINNER_CHARS = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

    def __init__(self, stream=None, verbose=False):
        self._stream = stream or sys.stdout
        self._verbose = verbose
        self._lock = threading.Lock()

        # ANSI color codes (disabled when stream is not a TTY)
        _s = self._stream
        self._tty = hasattr(_s, "isatty") and _s.isatty()
        self.RESET = "\033[0m" if self._tty else ""
        self.BOLD = "\033[1m" if self._tty else ""
        self.DIM = "\033[2m" if self._tty else ""
        self.GREEN = "\033[32m" if self._tty else ""
        self.YELLOW = "\033[33m" if self._tty else ""
        self.CYAN = "\033[36m" if self._tty else ""
        self.RED = "\033[31m" if self._tty else ""
        self.BOLD_GREEN = "\033[1;32m" if self._tty else ""
        self.BOLD_CYAN = "\033[1;36m" if self._tty else ""

        # Step/spinner state
        self._step_message: str | None = None
        self._step_original_message: str | None = None
        self._step_stop = threading.Event()
        self._step_thread: threading.Thread | None = None

        # Progress bar state
        self._progress_active = False
        self._progress_paused = False
        self._progress_total = 0
        self._progress_current = 0
        self._progress_desc = ""
        self._progress_unit = "file"
        self._progress_width = 30
        self._progress_file = ""
        self._progress_phase = ""
        self._progress_heartbeat = ""
        self._progress_has_subline = False
        self._progress_last_pct = -1  # last printed percentage (non-TTY)
        self._progress_substep = ""
        self._progress_substep_stop = threading.Event()
        self._progress_substep_thread: threading.Thread | None = None
        self._progress_substep_idx = 0

        # Output suppression state (populated by _suppress_output)
        self._orig_stdout = None
        self._orig_stderr = None
        self._orig_handler_levels: list[tuple[logging.Handler, int]] = []

    # -- context manager --

    def __enter__(self):
        self._suppress_output()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._step_message is not None:
            self.step_done("interrupted")
        self._step_original_message = None
        if self._progress_substep:
            self.progress_substep_done()
        if self._progress_active or self._progress_paused:
            self._progress_active = True
            self._progress_paused = False
            self.progress_done()
        self._restore_output()
        return False

    # -- step mode --

    def step_start(self, message: str) -> None:
        """Begin an animated step: prints 'message... ⠋' with spinner."""
        with self._lock:
            self._step_message = message
            self._step_original_message = message
            self._step_stop.clear()
            if not self._tty:
                self._write(f"{message}... ")
                return
            self._write(f"\r{self.BOLD}{message}{self.RESET}... ")
        self._step_thread = threading.Thread(target=self._step_spin, daemon=True)
        self._step_thread.start()

    def step_update(self, message: str) -> None:
        """Update the step message while the spinner continues."""
        with self._lock:
            if self._step_message is not None:
                self._step_message = message

    def step_done(self, suffix: str = "done") -> None:
        """Complete the current step: replaces spinner with suffix.

        Uses the original step_start message for the final line so that
        dynamic updates (e.g. file counts) don't appear in the completed line.
        """
        self._step_stop.set()
        if self._step_thread:
            self._step_thread.join()
            self._step_thread = None
        with self._lock:
            msg = self._step_original_message or self._step_message or ""
            self._step_message = None
            self._step_original_message = None
            if not self._tty:
                self._write(f"{suffix}\n")
                return
            colored_suffix = self._color_suffix(suffix)
            self._write(f"\r{self.BOLD}{msg}{self.RESET}... {colored_suffix}\033[K\n")

    def _step_spin(self) -> None:
        idx = 0
        while not self._step_stop.is_set():
            with self._lock:
                if self._step_message is not None:
                    self._write(
                        f"\r{self.BOLD}{self._step_message}{self.RESET}... "
                        f"{self.CYAN}{self._SPINNER_CHARS[idx]}{self.RESET}\033[K"
                    )
            idx = (idx + 1) % len(self._SPINNER_CHARS)
            self._step_stop.wait(0.1)

    # -- progress mode --

    def progress_start(self, total: int, desc: str = "Processing files", unit: str = "file") -> None:
        """Enter progress bar mode."""
        with self._lock:
            self._progress_active = True
            self._progress_paused = False
            self._progress_total = max(total, 0)
            self._progress_current = 0
            self._progress_desc = desc
            self._progress_unit = unit
            self._progress_file = ""
            self._progress_phase = ""
            self._progress_heartbeat = ""
            self._progress_has_subline = False
            self._progress_last_pct = -1
            if self._progress_total > 0:
                self._render_progress()

    def progress_update(self, step: int = 1) -> None:
        """Advance the progress bar."""
        with self._lock:
            if not self._progress_active or self._progress_total <= 0:
                return
            self._progress_current = min(self._progress_total, self._progress_current + step)
            self._render_progress()

    def progress_set_file(self, file_path: str, phase: str = "") -> None:
        """Set current file shown on sub-line under the bar."""
        with self._lock:
            if not self._progress_active:
                return
            self._progress_file = file_path
            self._progress_phase = phase
            self._render_progress()

    def progress_set_heartbeat(self, char: str) -> None:
        """Update heartbeat animation character."""
        with self._lock:
            if not self._progress_active:
                return
            self._progress_heartbeat = char
            self._render_progress()

    def progress_pause(self) -> None:
        """Temporarily hide the progress bar for step output."""
        with self._lock:
            if not self._progress_active:
                return
            self._clear_progress_area()
            self._progress_active = False
            self._progress_paused = True

    def progress_resume(self) -> None:
        """Re-show the progress bar after a pause."""
        with self._lock:
            if not self._progress_paused:
                return
            self._progress_active = True
            self._progress_paused = False
            self._progress_has_subline = False
            self._render_progress()

    def progress_substep_start(self, message: str) -> None:
        """Show a substep message on the progress bar sub-line with spinner."""
        with self._lock:
            self._progress_substep = message
            self._progress_file = ""
            self._progress_phase = ""
            self._progress_heartbeat = ""
            self._progress_substep_stop.clear()
            self._progress_substep_idx = 0
            if not self._tty:
                # Print current progress context + substep on one line
                pct_int = 0
                if self._progress_total > 0:
                    pct_int = int(self._progress_current / self._progress_total * 100)
                self._stream.write(
                    f"{self._progress_desc} "
                    f"[{self._progress_current}/{self._progress_total}] "
                    f"{pct_int}% \u2014 {message}... "
                )
                self._stream.flush()
                return
            self._render_progress()
        self._progress_substep_thread = threading.Thread(
            target=self._substep_spin, daemon=True
        )
        self._progress_substep_thread.start()

    def progress_substep_done(self) -> None:
        """Clear the substep message from the progress bar sub-line."""
        self._progress_substep_stop.set()
        if self._progress_substep_thread:
            self._progress_substep_thread.join()
            self._progress_substep_thread = None
        with self._lock:
            self._progress_substep = ""
            self._progress_heartbeat = ""
            if not self._tty:
                self._stream.write("done\n")
                self._stream.flush()
                return
            self._render_progress()

    def _substep_spin(self) -> None:
        """Animate spinner on the progress sub-line for substep."""
        while not self._progress_substep_stop.is_set():
            with self._lock:
                if self._progress_substep:
                    self._progress_heartbeat = self._SPINNER_CHARS[self._progress_substep_idx]
                    self._render_progress()
            self._progress_substep_idx = (self._progress_substep_idx + 1) % len(self._SPINNER_CHARS)
            self._progress_substep_stop.wait(0.1)

    def progress_done(self) -> None:
        """Exit progress bar mode."""
        with self._lock:
            if not self._progress_active:
                return
            if not self._tty:
                # Print final 100% line if not already printed
                if self._progress_last_pct < 100 and self._progress_total > 0:
                    self._write(
                        f"{self._progress_desc} "
                        f"[{self._progress_total}/{self._progress_total}] 100%\n"
                    )
            else:
                # Render final state
                self._render_progress()
                # Move past the progress area
                if self._progress_has_subline:
                    self._write("\n\n")
                else:
                    self._write("\n")
            self._progress_active = False
            self._progress_paused = False
            self._progress_has_subline = False

    def _render_progress(self) -> None:
        """Render progress bar + optional file sub-line. Must hold lock."""
        if self._progress_total <= 0:
            return
        progress = self._progress_current / self._progress_total

        if not self._tty:
            # Non-TTY: print a simple line every 10% to avoid log spam
            pct_int = int(progress * 100)
            threshold = (pct_int // 10) * 10
            if threshold <= self._progress_last_pct:
                return
            self._progress_last_pct = threshold
            self._stream.write(
                f"{self._progress_desc} "
                f"[{self._progress_current}/{self._progress_total}] "
                f"{pct_int}%\n"
            )
            self._stream.flush()
            return

        filled = int(self._progress_width * progress)
        bar_filled = f"{self.GREEN}{'█' * filled}{self.RESET}"
        bar_empty = f"{self.DIM}{'░' * (self._progress_width - filled)}{self.RESET}"
        bar = f"{bar_filled}{bar_empty}"

        # Move cursor to start of progress area
        if self._progress_has_subline:
            self._stream.write("\033[1A\r")

        # Line 1: progress bar
        pct = f"{progress * 100:5.1f}%"
        if progress >= 1.0:
            pct = f"{self.BOLD_GREEN}{pct}{self.RESET}"
        line1 = (
            f"{self.BOLD}{self._progress_desc}{self.RESET} [{bar}] "
            f"{pct}  {self.DIM}({self._progress_current}/{self._progress_total}){self.RESET}"
        )
        self._stream.write(f"\r\033[K{line1}")

        # Line 2: substep message (priority) or current file
        if self._progress_substep:
            subline = f"  {self.DIM}{self._progress_substep}{self.RESET}"
            if self._progress_heartbeat:
                subline += f" {self.CYAN}{self._progress_heartbeat}{self.RESET}"
            self._stream.write(f"\n\033[K{subline}")
            self._progress_has_subline = True
        elif self._progress_file:
            file_display = Path(self._progress_file).name
            if len(file_display) > 50:
                file_display = "..." + file_display[-47:]
            subline = f"  {self.DIM}{file_display}{self.RESET}"
            if self._progress_phase:
                subline += f" {self.DIM}({self._progress_phase}"
                if self._progress_heartbeat:
                    subline += f" {self.CYAN}{self._progress_heartbeat}{self.RESET}{self.DIM}"
                subline += f"){self.RESET}"
            elif self._progress_heartbeat:
                subline += f" {self.CYAN}{self._progress_heartbeat}{self.RESET}"
            self._stream.write(f"\n\033[K{subline}")
            self._progress_has_subline = True
        elif self._progress_has_subline:
            # Clear stale sub-line
            self._stream.write("\n\033[K")

        self._stream.flush()

    def _clear_progress_area(self) -> None:
        """Clear progress bar lines from terminal. Must hold lock."""
        if not self._tty:
            return
        if self._progress_has_subline:
            self._stream.write("\033[1A\r\033[K\n\033[K\033[1A\r")
        else:
            self._stream.write("\r\033[K")
        self._stream.flush()
        self._progress_has_subline = False

    # -- general output --

    def print(self, message: str) -> None:
        """Print a plain text line."""
        with self._lock:
            self._write(f"{message}\n")

    def success(self, message: str) -> None:
        """Print a success message in bold green."""
        with self._lock:
            self._write(f"{self.BOLD_GREEN}{message}{self.RESET}\n")

    def error(self, message: str) -> None:
        """Print an error message in red."""
        with self._lock:
            self._write(f"{self.RED}{message}{self.RESET}\n")

    # -- internal helpers --

    def _color_suffix(self, suffix: str) -> str:
        """Return a color-coded suffix string for step_done output."""
        s = suffix.lower()
        if s in ("done", "no changes"):
            return f"{self.GREEN}{suffix}{self.RESET}"
        if s in ("skipped",):
            return f"{self.YELLOW}{suffix}{self.RESET}"
        if s in ("interrupted",):
            return f"{self.RED}{suffix}{self.RESET}"
        return suffix

    def _write(self, text: str) -> None:
        """Write to stream and flush. Caller must hold lock."""
        self._stream.write(text)
        self._stream.flush()

    def _suppress_output(self) -> None:
        """Redirect stdout/stderr and silence root logger stream handlers.

        IndexingUI captures self._stream at __init__ time, so it keeps
        writing to the real terminal. All 3rd-party code that calls
        print() or writes to sys.stdout/stderr hits _DevNull instead.

        Skipped when verbose=True to allow full debugging output.
        """
        if self._verbose:
            return

        devnull = _DevNull()

        self._orig_stdout = sys.stdout
        self._orig_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull

        for handler in logging.root.handlers:
            if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
                self._orig_handler_levels.append((handler, handler.level))
                handler.setLevel(logging.CRITICAL + 1)

    def _restore_output(self) -> None:
        """Restore stdout/stderr and root logger handler levels."""
        if self._orig_stdout is not None:
            sys.stdout = self._orig_stdout
            self._orig_stdout = None
        if self._orig_stderr is not None:
            sys.stderr = self._orig_stderr
            self._orig_stderr = None

        for handler, level in self._orig_handler_levels:
            handler.setLevel(level)
        self._orig_handler_levels.clear()


class FileProcessingTimeoutError(Exception):
    """Raised when file processing exceeds timeout."""
    pass


class FileProcessingContext:
    """Context manager for file processing with timeout and heartbeat.

    Usage:
        with FileProcessingContext(file_path, ui, timeout=300) as ctx:
            ctx.set_phase("Converting .doc")
            result = process_file()
    """

    def __init__(
        self,
        file_path: str,
        ui: IndexingUI,
        timeout_seconds: float | None = None,
        heartbeat_interval: float = 2.0
    ):
        self.file_path = file_path
        self.ui = ui
        self.timeout_seconds = timeout_seconds
        self.heartbeat_interval = heartbeat_interval

        self._start_time: float | None = None
        self._stop_event = threading.Event()
        self._timeout_event = threading.Event()
        self._heartbeat_thread: threading.Thread | None = None
        self._current_phase = ""

    def __enter__(self):
        """Start timing and heartbeat thread."""
        self._start_time = time.time()

        # Start heartbeat thread
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            daemon=True
        )
        self._heartbeat_thread.start()

        # Update UI with current file
        self.ui.progress_set_file(self.file_path, "")

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop heartbeat and check for slow files."""
        # Stop heartbeat thread
        self._stop_event.set()
        if self._heartbeat_thread:
            self._heartbeat_thread.join(timeout=1.0)

        # Warn about slow files (warnings still pass through)
        if self._start_time:
            duration = time.time() - self._start_time
            slow_threshold = cfgload.get(
                "indexing.logging.slow_file_threshold_seconds", 30
            )
            if duration > slow_threshold:
                logger.warning(
                    f"Slow file processing: {self.file_path} took {duration:.1f}s"
                )

        # Don't suppress exceptions
        return False

    def set_phase(self, phase: str) -> None:
        """Update current operation phase."""
        self._current_phase = phase
        self.ui.progress_set_file(self.file_path, phase)

        # Check for timeout
        self.check_timeout()

    def check_timeout(self) -> None:
        """Check if processing has exceeded timeout (main-thread safe).

        Checks both the _timeout_event (set by heartbeat thread) and
        elapsed wall-clock time. Raises FileProcessingTimeoutError if
        either indicates a timeout.
        """
        if self.timeout_seconds is None or self._start_time is None:
            return

        if self._timeout_event.is_set():
            elapsed = time.time() - self._start_time
            raise FileProcessingTimeoutError(
                f"File processing timed out after {elapsed:.1f}s: {self.file_path}"
            )

        elapsed = time.time() - self._start_time
        if elapsed > self.timeout_seconds:
            self._timeout_event.set()
            raise FileProcessingTimeoutError(
                f"File processing timed out after {elapsed:.1f}s: {self.file_path}"
            )

    def remaining_seconds(self) -> float | None:
        """Return seconds remaining before timeout, or None if no timeout set."""
        if self.timeout_seconds is None or self._start_time is None:
            return None
        remaining = self.timeout_seconds - (time.time() - self._start_time)
        return max(0.0, remaining)

    def _check_timeout(self) -> None:
        """Check if processing has exceeded timeout (used by heartbeat thread)."""
        if self.timeout_seconds is None or self._start_time is None:
            return

        elapsed = time.time() - self._start_time
        if elapsed > self.timeout_seconds:
            self._timeout_event.set()
            raise FileProcessingTimeoutError(
                f"File processing timed out after {elapsed:.1f}s: {self.file_path}"
            )

    def _heartbeat_loop(self) -> None:
        """Background thread that updates heartbeat indicator."""
        if not self.ui._tty:
            # Non-TTY: only monitor for timeouts, skip animation
            while not self._stop_event.is_set():
                try:
                    self._check_timeout()
                except FileProcessingTimeoutError:
                    break
                time.sleep(self.heartbeat_interval)
            return

        spinner_chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        idx = 0

        while not self._stop_event.is_set():
            self.ui.progress_set_heartbeat(spinner_chars[idx])
            idx = (idx + 1) % len(spinner_chars)

            # Check for timeout
            try:
                self._check_timeout()
            except FileProcessingTimeoutError:
                break

            time.sleep(self.heartbeat_interval)


class GracefulAbort:
    """Two-stage Ctrl-C handler for the indexing pipeline.

    First Ctrl-C:  sets abort flag and restores default SIGINT so a second
                   Ctrl-C raises KeyboardInterrupt immediately.
    """

    def __init__(self, ui: "IndexingUI"):
        self._abort = False
        self._ui = ui
        self._original_handler = None

    @property
    def abort_requested(self) -> bool:
        return self._abort

    def install(self) -> None:
        """Register the custom SIGINT handler. Must be called from the main thread."""
        self._original_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self._handle_sigint)

    def uninstall(self) -> None:
        """Restore the original SIGINT handler."""
        if self._original_handler is not None:
            signal.signal(signal.SIGINT, self._original_handler)
            self._original_handler = None

    def _handle_sigint(self, signum, frame):
        self._abort = True
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        self._ui.print(
            f"\n{self._ui.YELLOW}Ctrl-C received. "
            f"Finishing current batch and saving state...{self._ui.RESET}"
        )
        self._ui.print(
            f"{self._ui.DIM}(Press Ctrl-C again to force quit immediately)"
            f"{self._ui.RESET}"
        )
