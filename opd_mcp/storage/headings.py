"""Heading storage for document structure persistence."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

from opd_mcp.config import HEADING_STORE_PATH

logger = logging.getLogger(__name__)


class HeadingStore:
    """Stores document headings separately from chunk metadata.

    This avoids the LlamaIndex SentenceSplitter metadata size validation issue,
    which checks metadata length before applying exclusions. By storing headings
    in a separate file, we keep chunk metadata small while preserving heading
    data for retrieval.
    """

    def __init__(self, store_path: Path):
        self.store_path = store_path
        self._data: Dict[str, List[dict]] = {}
        self._load()

    def _load(self):
        """Load heading data from disk."""
        if self.store_path.exists():
            try:
                with open(self.store_path, "r", encoding="utf-8") as f:
                    self._data = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load heading store: {e}")
                self._data = {}

    def _save(self):
        """Save heading data to disk."""
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.store_path, "w", encoding="utf-8") as f:
            json.dump(self._data, f)

    def set_headings(self, file_path: str, headings: List[dict]):
        """Store headings for a file."""
        self._data[file_path] = headings
        self._save()

    def get_headings(self, file_path: str) -> List[dict]:
        """Get headings for a file."""
        return self._data.get(file_path, [])

    def remove_headings(self, file_path: str):
        """Remove headings for a file."""
        if file_path in self._data:
            del self._data[file_path]
            self._save()


# Module-level heading store instance (lazy initialized)
_heading_store: Optional[HeadingStore] = None


def get_heading_store(store_path: Optional[Path] = None) -> HeadingStore:
    """Get the singleton HeadingStore instance.

    Args:
        store_path: Optional path override for testing. If provided with a different
                    path than the current store, resets the singleton.
    """
    global _heading_store

    # Allow override for testing
    if store_path is not None:
        if _heading_store is None or _heading_store.store_path != store_path:
            _heading_store = HeadingStore(store_path)
        return _heading_store

    if _heading_store is None:
        _heading_store = HeadingStore(HEADING_STORE_PATH)
    return _heading_store


def reset_heading_store():
    """Reset the singleton HeadingStore instance (for testing)."""
    global _heading_store
    _heading_store = None
