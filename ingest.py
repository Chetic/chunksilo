#!/usr/bin/env python3
"""
Ingestion pipeline for building a RAG index from PDF, DOCX, Markdown, and TXT documents.
"""
import argparse
import itertools
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import List

# Check for --offline flag early and set environment variables BEFORE any imports
# This is critical because HuggingFace libraries read these variables at import time
if "--offline" in sys.argv:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"

from dotenv import load_dotenv
from docx import Document

from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    Settings,
    SimpleDirectoryReader,
    Document as LlamaIndexDocument,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.fastembed import FastEmbedEmbedding

# Load environment variables
load_dotenv()

# Configuration
DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
STORAGE_DIR = Path(os.getenv("STORAGE_DIR", "./storage"))

# Stage 1 (embedding/vector search) configuration
RETRIEVAL_EMBED_MODEL_NAME = os.getenv(
    "RETRIEVAL_EMBED_MODEL_NAME", "BAAI/bge-small-en-v1.5"
)

# Stage 2 (FlashRank reranking, CPU-only, ONNX-based) configuration
RETRIEVAL_RERANK_MODEL_NAME = os.getenv(
    "RETRIEVAL_RERANK_MODEL_NAME", "ms-marco-MiniLM-L-12-v2"
)

# Shared cache directory for embedding and reranking models
RETRIEVAL_MODEL_CACHE_DIR = Path(os.getenv("RETRIEVAL_MODEL_CACHE_DIR", "./models"))

# Text chunking configuration for optimal retrieval
# Chunk size in tokens: 512-1024 tokens is optimal for RAG systems
# Smaller chunks improve precision but may lose context
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
# Overlap: 10-20% of chunk size preserves context across boundaries
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


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


def _embedding_cache_path(model_name: str, cache_dir: Path) -> Path:
    """Return the expected cache directory for a FastEmbed model."""

    return cache_dir / f"models--{model_name.replace('/', '--')}"


def _verify_model_cache_exists(cache_dir: Path) -> bool:
    """
    Verify that the cached model directory exists and contains the expected model files.
    
    FastEmbed caches models in HuggingFace Hub format. This checks if the cache directory
    structure exists and contains the model files.
    
    Args:
        cache_dir: Directory where models are cached
        
    Returns:
        True if the cached model appears to be present, False otherwise
    """
    # FastEmbed uses HuggingFace Hub format for caching
    # The model name maps to a HuggingFace source which determines the cache directory name
    from fastembed import TextEmbedding
    
    try:
        models = TextEmbedding.list_supported_models()
        model_info = [m for m in models if m.get("model") == RETRIEVAL_EMBED_MODEL_NAME]
        if not model_info:
            return False
        
        model_info = model_info[0]
        hf_source = model_info.get("sources", {}).get("hf")
        if not hf_source:
            return False
        
        # Expected cache directory format: models--org--repo
        expected_dir = cache_dir / f"models--{hf_source.replace('/', '--')}"
        if not expected_dir.exists():
            return False
        
        # Check for snapshot directory with model files
        snapshots_dir = expected_dir / "snapshots"
        if not snapshots_dir.exists():
            return False
        
        # Check if any snapshot contains the model file
        model_file = model_info.get("model_file", "model_optimized.onnx")
        for snapshot in snapshots_dir.iterdir():
            if snapshot.is_dir():
                model_path = snapshot / model_file
                if model_path.exists() or model_path.is_symlink():
                    return True
        
        return False
    except Exception:
        # If we can't verify, assume it's not cached
        return False


def _get_cached_model_path(cache_dir: Path, model_name: str) -> Path | None:
    """
    Get the cached model directory path using huggingface_hub's snapshot_download.
    This works completely offline and bypasses fastembed's download_model API calls.
    
    Args:
        cache_dir: Directory where models are cached
        model_name: FastEmbed model name (e.g., 'BAAI/bge-small-en-v1.5')
        
    Returns:
        Path to the cached model directory, or None if not found or huggingface_hub unavailable
    """
    try:
        from huggingface_hub import snapshot_download
        from fastembed import TextEmbedding
        models = TextEmbedding.list_supported_models()
        model_info = [m for m in models if m.get("model") == model_name]
        if model_info:
            hf_source = model_info[0].get("sources", {}).get("hf")
            if hf_source:
                cache_dir_abs = cache_dir.resolve()
                # Use snapshot_download with local_files_only=True to get cached model path
                # This works completely offline and bypasses fastembed's API call
                model_dir = snapshot_download(
                    repo_id=hf_source,
                    local_files_only=True,
                    cache_dir=str(cache_dir_abs)
                )
                return Path(model_dir).resolve()
    except (ImportError, Exception):
        # huggingface_hub might not be available, or model not in cache
        pass
    return None


def _create_fastembed_embedding(cache_dir: Path, offline: bool = False):
    """
    Create a FastEmbedEmbedding instance, using cached model path in offline mode
    to bypass fastembed's download_model API calls.
    
    Args:
        cache_dir: Directory where models are cached
        offline: If True, try to use cached model path to bypass download step
        
    Returns:
        FastEmbedEmbedding instance
    """
    if offline:
        # Try to get cached model path to bypass fastembed's download step
        cached_model_path = _get_cached_model_path(cache_dir, RETRIEVAL_EMBED_MODEL_NAME)
        if cached_model_path:
            logger.info(
                f"Using cached model path to bypass download: {cached_model_path}"
            )
            return FastEmbedEmbedding(
                model_name=RETRIEVAL_EMBED_MODEL_NAME,
                cache_dir=str(cache_dir),
                specific_model_path=str(cached_model_path)
            )
        else:
            logger.warning(
                "Could not find cached model path, falling back to normal initialization"
            )
    
    # Normal initialization (non-offline or fallback)
    return FastEmbedEmbedding(
        model_name=RETRIEVAL_EMBED_MODEL_NAME, cache_dir=str(cache_dir)
    )


def ensure_embedding_model_cached(cache_dir: Path, offline: bool = False) -> None:
    """
    Ensure the embedding model is available in the local cache.
    
    Args:
        cache_dir: Directory where the model should be cached
        offline: If True, fail if model is not available locally instead of downloading
        
    Raises:
        FileNotFoundError: If offline=True and model is not available locally
        RuntimeError: If offline=False and model download/initialization fails
    """
    # First, verify the cache exists by checking the file system
    if offline:
        logger.info("Verifying embedding model cache...")
        if _verify_model_cache_exists(cache_dir):
            logger.info("Embedding model found in cache")
        else:
            logger.error(
                "Offline mode enabled, but embedding model cache not found in %s",
                cache_dir,
            )
            raise FileNotFoundError(
                f"Embedding model '{RETRIEVAL_EMBED_MODEL_NAME}' not found in cache directory '{cache_dir}'. "
                "Run without --offline flag to download the model, or ensure the model is already cached."
            )
    
    # Try to initialize the model using our helper function
    # In offline mode, this uses snapshot_download to get the cached model path
    # and passes it as specific_model_path to bypass fastembed's download_model API calls.
    try:
        logger.info("Initializing embedding model from cache...")
        cache_dir_abs = cache_dir.resolve()
        if offline:
            os.environ["HF_HUB_CACHE"] = str(cache_dir_abs)
        
        _create_fastembed_embedding(cache_dir, offline=offline)
        logger.info("Embedding model initialized successfully")
        return
    except (ValueError, Exception) as e:
        error_str = str(e).lower()
        # Check if the error is about network/download failure but model files exist
        is_network_error = (
            "offline" in error_str
            or "cannot reach" in error_str
            or "could not load model" in error_str
            or "could not download" in error_str
        )
        
        if offline and is_network_error:
            # Double-check that model files actually exist
            if _verify_model_cache_exists(cache_dir):
                # Model files exist, but fastembed tried to make an API call anyway.
                # This is a known limitation of fastembed in offline mode.
                logger.error(
                    "FastEmbed attempted a network request to verify the model, "
                    "even though the model files exist in the cache. "
                    "This is a FastEmbed limitation in offline mode."
                )
                raise FileNotFoundError(
                    f"Embedding model '{RETRIEVAL_EMBED_MODEL_NAME}' files exist in cache directory '{cache_dir}', "
                    "but FastEmbed attempted a network request to verify the model. "
                    "This is a known FastEmbed limitation. "
                    "To work around this, temporarily run without --offline flag to allow "
                    "FastEmbed to verify the model once, then subsequent runs with --offline should work."
                ) from e
            else:
                # Model files don't actually exist
                logger.error(
                    "Offline mode enabled, but embedding model cache not found in %s",
                    cache_dir,
                )
                raise FileNotFoundError(
                    f"Embedding model '{RETRIEVAL_EMBED_MODEL_NAME}' not found in cache directory '{cache_dir}'. "
                    "Run without --offline flag to download the model, or ensure the model is already cached."
                ) from e
        elif offline:
            # Other error in offline mode
            logger.error(
                "Offline mode enabled, but embedding model initialization failed: %s",
                e,
            )
            raise FileNotFoundError(
                f"Embedding model '{RETRIEVAL_EMBED_MODEL_NAME}' not available in cache directory '{cache_dir}'. "
                "Run without --offline flag to download the model, or ensure the model is already cached."
            ) from e
        else:
            # Not offline, but initialization failed (e.g., network issues)
            # This should not happen silently - raise to abort packaging
            logger.error(
                "Failed to initialize/download embedding model: %s",
                e,
            )
            raise RuntimeError(
                f"Failed to download/initialize embedding model '{RETRIEVAL_EMBED_MODEL_NAME}' to cache directory '{cache_dir}'. "
                "This may be due to network connectivity issues. Please retry the download."
            ) from e


def ensure_rerank_model_cached(cache_dir: Path, offline: bool = False) -> Path:
    """Ensure the reranking model is cached locally for CPU inference using FlashRank."""

    try:
        from flashrank import Ranker
    except ImportError as exc:  # pragma: no cover - import guard
        raise ImportError(
            "flashrank is required for reranking. Install dependencies from requirements.txt."
        ) from exc

    cache_dir_abs = cache_dir.resolve()
    logger.info("Ensuring rerank model is available in cache...")

    # Map cross-encoder model names to FlashRank equivalents if needed
    model_name = RETRIEVAL_RERANK_MODEL_NAME
    # Note: FlashRank doesn't have L-6 models, so we map to L-12 equivalents
    model_mapping = {
        "cross-encoder/ms-marco-MiniLM-L-6-v2": "ms-marco-MiniLM-L-12-v2",  # L-6 not available, use L-12
        "ms-marco-MiniLM-L-6-v2": "ms-marco-MiniLM-L-12-v2",  # Direct mapping for L-6
    }
    if model_name in model_mapping:
        model_name = model_mapping[model_name]
    elif model_name.startswith("cross-encoder/"):
        # Extract model name after cross-encoder/ prefix and try to map
        base_name = model_name.replace("cross-encoder/", "")
        # If it's an L-6 model, map to L-12
        if "L-6" in base_name:
            model_name = base_name.replace("L-6", "L-12")
        else:
            model_name = base_name

    try:
        # FlashRank handles model downloading and caching internally
        # Initialize the model to trigger download if needed
        reranker = Ranker(model_name=model_name, cache_dir=str(cache_dir_abs))
        logger.info(f"FlashRank model '{model_name}' initialized successfully")
        # FlashRank caches models in its own format, return the cache directory
        return cache_dir_abs
    except Exception as exc:
        if offline:
            raise FileNotFoundError(
                f"Rerank model '{model_name}' not found in cache directory '{cache_dir_abs}'. "
                "Run without --offline to download it before packaging releases."
            ) from exc
        raise


def _parse_heading_level(style_name: str | None) -> int:
    """Best-effort extraction of a numeric heading level from a DOCX style name."""
    if not style_name:
        return 1
    try:
        if "Heading" in style_name:
            level_str = style_name.replace("Heading", "").strip()
            if level_str:
                return int(level_str)
    except (ValueError, AttributeError):
        pass
    return 1


def split_docx_into_heading_documents(docx_path: Path) -> List[LlamaIndexDocument]:
    """
    Load a DOCX file and split it into one LlamaIndex document **per heading section**.

    Each returned document:
    - text: all paragraphs that belong to a given heading (up to, but not including,
      the next heading of the same or higher level)
    - metadata: includes file + heading information so the MCP server can surface
      section-level citations.
    """
    docs: List[LlamaIndexDocument] = []

    try:
        doc = Document(docx_path)
    except Exception as e:
        logger.warning(f"Failed to open DOCX {docx_path}: {e}")
        return docs

    current_heading: str | None = None
    current_level: int | None = None
    current_body: list[str] = []

    def flush_current():
        """Flush the current heading section into a LlamaIndexDocument."""
        if not current_heading:
            return
        # Join body paragraphs; skip empty sections
        text = "\n".join(line for line in current_body if line is not None).strip()
        if not text:
            return

        metadata = {
            "file_path": str(docx_path),
            "file_name": docx_path.name,
            "source": str(docx_path),
            "heading": current_heading,
            "heading_level": current_level,
        }
        docs.append(LlamaIndexDocument(text=text, metadata=metadata))

    for para in doc.paragraphs:
        style_name = getattr(para.style, "name", "") or ""
        is_heading = (
            style_name.startswith("Heading")
            or style_name.startswith("heading")
            or "Heading" in style_name
        )

        if is_heading and para.text.strip():
            # Starting a new heading section: flush the previous one
            flush_current()

            current_heading = para.text.strip()
            current_level = _parse_heading_level(style_name)
            current_body = []
        else:
            # Regular content: attach to current heading (if any)
            if current_heading is not None:
                current_body.append(para.text)

    # Flush the final section
    flush_current()

    # If no headings were detected, fall back to a single document for the whole file
    if not docs:
        try:
            full_text = "\n".join(p.text for p in doc.paragraphs).strip()
        except Exception:
            full_text = ""

        if full_text:
            metadata = {
                "file_path": str(docx_path),
                "file_name": docx_path.name,
                "source": str(docx_path),
                "heading": None,
                "heading_level": None,
            }
            docs.append(LlamaIndexDocument(text=full_text, metadata=metadata))

    logger.info(
        f"Split DOCX {docx_path} into {len(docs)} heading-based document(s) for indexing"
    )
    return docs


def load_docx_heading_documents(root_dir: Path) -> List[LlamaIndexDocument]:
    """
    Walk the DATA_DIR tree and load all .docx files as heading-based documents.
    """
    docx_paths = list(root_dir.rglob("*.docx"))
    if not docx_paths:
        logger.info("No DOCX files found in data directory; skipping DOCX ingestion.")
        return []

    logger.info(
        f"Found {len(docx_paths)} DOCX file(s); splitting by heading with progress bar..."
    )
    all_docs: List[LlamaIndexDocument] = []
    progress = SimpleProgressBar(len(docx_paths), desc="Processing DOCX files", unit="file")
    for docx_path in docx_paths:
        all_docs.extend(split_docx_into_heading_documents(docx_path))
        progress.update()
    return all_docs


def configure_offline_mode(offline: bool, cache_dir: Path) -> None:
    """
    Configure environment variables to enforce offline mode for HuggingFace libraries.
    
    This prevents HuggingFace Hub, Transformers, and related libraries from making
    network requests even when models are cached locally.
    
    Note: Environment variables should be set BEFORE importing HuggingFace libraries.
    This function ensures they're set and logs the configuration.
    
    Args:
        offline: If True, set environment variables to force offline mode
        cache_dir: Directory where models are cached (used for HF_HOME/HF_HUB_CACHE)
    """
    if offline:
        # Ensure offline variables are set (they should already be set at import time)
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        # Point HuggingFace Hub to our cache directory so it can find cached models
        cache_dir_abs = cache_dir.resolve()
        os.environ["HF_HOME"] = str(cache_dir_abs)
        os.environ["HF_HUB_CACHE"] = str(cache_dir_abs)
        # Also set HF_DATASETS_CACHE to prevent datasets library from making network requests
        os.environ["HF_DATASETS_CACHE"] = str(cache_dir_abs)
        logger.info(
            "Offline mode enabled: HuggingFace libraries configured to avoid network requests. "
            f"Cache directory: {cache_dir_abs}"
        )


def build_index(download_only: bool = False, offline: bool = False) -> None:
    """Build and persist the vector index from documents."""
    # Configure offline mode before any HuggingFace/FastEmbed operations
    configure_offline_mode(offline, RETRIEVAL_MODEL_CACHE_DIR)
    
    logger.info(f"Starting ingestion from {DATA_DIR}")

    # Ensure the embedding model is available offline
    ensure_embedding_model_cached(RETRIEVAL_MODEL_CACHE_DIR, offline=offline)
    try:
        ensure_rerank_model_cached(RETRIEVAL_MODEL_CACHE_DIR, offline=offline)
    except FileNotFoundError as exc:
        if download_only or offline:
            raise
        logger.warning(
            "Rerank model could not be cached yet; continuing without it. Download it before packaging releases. Error: %s",
            exc,
        )
    if download_only:
        logger.info(
            "Embedding and rerank model caches downloaded; skipping index build because --download-models was provided."
        )
        return

    # Ensure data directory exists
    if not DATA_DIR.exists():
        logger.error(f"Data directory {DATA_DIR} does not exist!")
        raise FileNotFoundError(f"Data directory {DATA_DIR} does not exist")

    # Discover whether there are any PDF/Markdown/TXT files before constructing the reader
    pdf_md_txt_files = list(DATA_DIR.rglob("*.pdf")) + list(DATA_DIR.rglob("*.md")) + list(DATA_DIR.rglob("*.txt"))
    docx_files = list(DATA_DIR.rglob("*.docx"))
    total_files = len(pdf_md_txt_files) + len(docx_files)
    logger.info(
        f"Discovered {len(pdf_md_txt_files)} PDF/Markdown/TXT and {len(docx_files)} DOCX file(s) (total: {total_files})."
    )

    docs: List[LlamaIndexDocument] = []

    if pdf_md_txt_files:
        # Initialize reader for non-DOCX formats
        reader = SimpleDirectoryReader(
            input_dir=str(DATA_DIR),
            required_exts=[".pdf", ".md", ".txt"],
            recursive=True,
        )

        # Load documents (PDF/Markdown/TXT)
        logger.info(
            f"Loading {len(pdf_md_txt_files)} PDF/Markdown/TXT file(s) into LlamaIndex documents..."
        )
        docs.extend(reader.load_data())
    else:
        logger.info("No PDF/Markdown/TXT files found in data directory; skipping PDF/MD/TXT ingestion.")

    # Load DOCX documents as one document per heading section
    logger.info("Loading DOCX documents (one chunk per heading)...")
    docx_docs = load_docx_heading_documents(DATA_DIR)
    docs.extend(docx_docs)
    
    if not docs:
        logger.warning(f"No documents found in {DATA_DIR}")
        return
    logger.info(f"Loaded {len(docs)} documents (including DOCX heading chunks)")

    # Initialize embedding model
    logger.info(f"Initializing embedding model: {RETRIEVAL_EMBED_MODEL_NAME}")
    with Spinner("Initializing embedding model from offline cache"):
        embed_model = _create_fastembed_embedding(RETRIEVAL_MODEL_CACHE_DIR, offline=offline)
    Settings.embed_model = embed_model
    logger.info("Embedding model initialized")

    # Configure text splitter for optimal chunking
    # Using sentence-aware splitting with optimal chunk size and overlap
    text_splitter = SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separator=" ",
    )
    Settings.text_splitter = text_splitter
    logger.info(f"Configured text splitter: chunk_size={CHUNK_SIZE}, chunk_overlap={CHUNK_OVERLAP}")

    logger.info(
        "Indexing documents with overall progress bar (progress reflects full document processing)..."
    )
    storage_context = StorageContext.from_defaults()
    index = VectorStoreIndex([], storage_context=storage_context, embed_model=embed_model)

    progress = SimpleProgressBar(len(docs), desc="Indexing documents", unit="doc")
    for doc in docs:
        index.insert(doc)
        progress.update()
    
    # Persist index
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Persisting index to {STORAGE_DIR}")
    index.storage_context.persist(persist_dir=str(STORAGE_DIR))
    
    logger.info(f"Successfully indexed {len(docs)} documents into {STORAGE_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the document index")
    parser.add_argument(
        "--download-models",
        action="store_true",
        help="Download the retrieval models (embedding + reranker) into the offline cache and exit",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Run entirely offline; fail if embedding model is not available locally",
    )
    args = parser.parse_args()

    try:
        build_index(download_only=args.download_models, offline=args.offline)
    except Exception as e:
        logger.error(f"Ingestion failed: {e}", exc_info=True)
        raise

