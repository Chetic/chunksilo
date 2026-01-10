"""Logging utilities."""

import os
from datetime import datetime
from pathlib import Path


def rotate_log_if_needed(
    log_path: Path | str,
    max_size_mb: float = 10,
) -> None:
    """Rotate log file if it exists and is over the size limit.

    Args:
        log_path: Path to the log file
        max_size_mb: Maximum log file size in megabytes before rotation
    """
    log_path = Path(log_path)
    max_size_bytes = max_size_mb * 1024 * 1024

    if log_path.exists() and log_path.stat().st_size > max_size_bytes:
        # Generate timestamp suffix with microsecond precision and process ID
        # to prevent collisions when multiple processes rotate simultaneously
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        process_id = os.getpid()
        rotated_name = f"{log_path.stem}_{timestamp}_{process_id}.log"
        rotated_path = log_path.parent / rotated_name
        log_path.rename(rotated_path)
        # Create new empty log file
        log_path.touch()
