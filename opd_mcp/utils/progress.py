"""Progress display utilities."""

import itertools
import sys
import threading
import time


class SimpleProgressBar:
    """Lightweight progress bar using only the standard library."""

    def __init__(self, total: int, desc: str, unit: str = "item", width: int = 30):
        self.total = max(total, 0)
        self.desc = desc
        self.unit = unit
        self.width = width
        self.current = 0
        if self.total > 0:
            self._render()

    def update(self, step: int = 1) -> None:
        if self.total <= 0:
            return
        self.current = min(self.total, self.current + step)
        self._render()
        if self.current >= self.total:
            sys.stdout.write("\n")
            sys.stdout.flush()

    def _render(self) -> None:
        progress = self.current / self.total if self.total else 0
        filled = int(self.width * progress)
        bar = "#" * filled + "-" * (self.width - filled)
        sys.stdout.write(
            f"\r{self.desc} [{bar}] {progress * 100:5.1f}% ({self.current}/{self.total} {self.unit}s)"
        )
        sys.stdout.flush()


class Spinner:
    """Simple console spinner to indicate long-running steps."""

    def __init__(self, desc: str, interval: float = 0.1):
        self.desc = desc
        self.interval = interval
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._line = desc

    def __enter__(self):
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._stop_event.set()
        if self._thread:
            self._thread.join()
        # Clear spinner line
        sys.stdout.write("\r" + " " * len(self._line) + "\r")
        sys.stdout.flush()

    def _spin(self) -> None:
        for char in itertools.cycle("|/-\\"):
            if self._stop_event.is_set():
                break
            self._line = f"{self.desc} {char}"
            sys.stdout.write("\r" + self._line)
            sys.stdout.flush()
            time.sleep(self.interval)
