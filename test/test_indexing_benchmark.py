#!/usr/bin/env python3
"""
Indexing performance benchmark for ChunkSilo.

Measures wall-clock time and throughput for building a vector index from
a diverse document corpus. Used in CI to detect performance regressions.

Usage:
    cd test
    OFFLINE=0 python test_indexing_benchmark.py

Environment variables:
    TEST_DATA_DIR               Where test documents are stored (default: ./test_data)
    TEST_STORAGE_DIR            Where benchmark index is stored (default: ./bench_storage)
    TEST_RESULTS_DIR            Where JSON results are saved (default: ./test_results)
    BENCHMARK_THRESHOLD         Max allowed indexing seconds (default: 600)
    ABORT_ON_DOWNLOAD_FAILURE   Abort if corpus downloads fail (default: 1)
    OFFLINE                     Must be 0 to allow downloads (default: 1)
"""
import json
import logging
import os
import platform
import resource
import shutil
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Reuse corpus download logic from the RAG metrics suite
from test_rag_metrics import download_test_corpus

from chunksilo.index import build_index

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TEST_DATA_DIR = Path(os.getenv("TEST_DATA_DIR", "./test_data"))
TEST_STORAGE_DIR = Path(os.getenv("TEST_STORAGE_DIR", "./bench_storage"))
TEST_RESULTS_DIR = Path(os.getenv("TEST_RESULTS_DIR", "./test_results"))
BENCHMARK_THRESHOLD_SECONDS = int(os.getenv("BENCHMARK_THRESHOLD", "300"))


def _corpus_stats(data_dir: Path) -> dict:
    """Collect file counts and sizes grouped by extension."""
    by_type: dict = {}
    total_files = 0
    total_bytes = 0

    for f in data_dir.rglob("*"):
        if not f.is_file():
            continue
        ext = f.suffix.lstrip(".").lower()
        if ext in ("pdf", "md", "rst", "txt", "docx", "doc"):
            entry = by_type.setdefault(ext, {"count": 0, "size_bytes": 0})
            entry["count"] += 1
            entry["size_bytes"] += f.stat().st_size
            total_files += 1
            total_bytes += f.stat().st_size

    return {
        "total_files": total_files,
        "total_size_bytes": total_bytes,
        "by_type": by_type,
    }


def run_benchmark() -> dict:
    """Download corpus, build a fresh index, and return timing results."""

    # 1. Download / verify corpus
    logger.info("=" * 70)
    logger.info("Step 1: Preparing test corpus")
    logger.info("=" * 70)
    downloaded = download_test_corpus()
    if not any(downloaded.values()):
        raise RuntimeError("No documents available — cannot benchmark.")

    corpus = _corpus_stats(TEST_DATA_DIR)
    logger.info(
        f"Corpus ready: {corpus['total_files']} files, "
        f"{corpus['total_size_bytes'] / 1024 / 1024:.1f} MB"
    )

    # 2. Clean previous benchmark storage for a fresh run
    if TEST_STORAGE_DIR.exists():
        shutil.rmtree(TEST_STORAGE_DIR)
    TEST_STORAGE_DIR.mkdir(parents=True, exist_ok=True)

    # 3. Patch module globals for isolation (same pattern as test_rag_metrics)
    from chunksilo import index as index_mod

    orig = {
        "STORAGE_DIR": index_mod.STORAGE_DIR,
        "STATE_DB_PATH": index_mod.STATE_DB_PATH,
        "BM25_INDEX_DIR": index_mod.BM25_INDEX_DIR,
        "HEADING_STORE_PATH": index_mod.HEADING_STORE_PATH,
        "_config": index_mod._config,
    }

    try:
        index_mod.STORAGE_DIR = TEST_STORAGE_DIR
        index_mod.STATE_DB_PATH = TEST_STORAGE_DIR / "ingestion_state.db"
        index_mod.BM25_INDEX_DIR = TEST_STORAGE_DIR / "bm25_index"
        index_mod.HEADING_STORE_PATH = TEST_STORAGE_DIR / "heading_store.json"
        index_mod._config = {
            **index_mod._config,
            "indexing": {
                "directories": [str(TEST_DATA_DIR)],
                "chunk_size": 512,
                "chunk_overlap": 50,
            },
        }

        # 4. Run the indexing benchmark
        logger.info("=" * 70)
        logger.info("Step 2: Building index (timed)")
        logger.info("=" * 70)

        # Use resource.getrusage for zero-overhead peak memory measurement.
        # tracemalloc tracks every allocation and adds 4-5x overhead to
        # ONNX-heavy workloads (measured: 362s with tracemalloc vs 68s without).
        t_start = time.monotonic()

        build_index()

        elapsed = time.monotonic() - t_start
        # ru_maxrss is peak RSS for the process lifetime.
        # On macOS it's in bytes, on Linux it's in KB.
        max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak_mem = max_rss if sys.platform == "darwin" else max_rss * 1024

        logger.info(f"Indexing completed in {elapsed:.2f}s")

        # 5. Collect post-indexing stats
        chunk_count = 0
        files_indexed = 0
        try:
            from llama_index.core import Settings, StorageContext, load_index_from_storage
            from chunksilo.index import _create_fastembed_embedding

            embed_model = _create_fastembed_embedding(
                index_mod.RETRIEVAL_MODEL_CACHE_DIR, offline=True
            )
            Settings.embed_model = embed_model
            sc = StorageContext.from_defaults(persist_dir=str(TEST_STORAGE_DIR))
            idx = load_index_from_storage(sc, embed_model=embed_model)
            chunk_count = len(idx.docstore.docs)
            seen_files: set = set()
            for doc in idx.docstore.docs.values():
                fp = doc.metadata.get("file_path", "")
                if fp:
                    seen_files.add(fp)
            files_indexed = len(seen_files)
        except Exception as e:
            logger.warning(f"Could not collect post-index stats: {e}")

        passed = elapsed <= BENCHMARK_THRESHOLD_SECONDS

        docs_per_sec = corpus["total_files"] / elapsed if elapsed > 0 else 0
        chunks_per_sec = chunk_count / elapsed if elapsed > 0 else 0

        return {
            "timestamp": int(time.time()),
            "corpus": corpus,
            "indexing": {
                "total_seconds": round(elapsed, 2),
                "chunk_count": chunk_count,
                "files_indexed": files_indexed,
                "documents_per_second": round(docs_per_sec, 3),
                "chunks_per_second": round(chunks_per_sec, 3),
            },
            "memory": {
                "peak_mb": round(peak_mem / 1024 / 1024, 1),
            },
            "threshold": {
                "max_seconds": BENCHMARK_THRESHOLD_SECONDS,
                "passed": passed,
            },
            "environment": {
                "python_version": platform.python_version(),
                "platform": sys.platform,
            },
        }
    finally:
        # Restore originals
        index_mod.STORAGE_DIR = orig["STORAGE_DIR"]
        index_mod.STATE_DB_PATH = orig["STATE_DB_PATH"]
        index_mod.BM25_INDEX_DIR = orig["BM25_INDEX_DIR"]
        index_mod.HEADING_STORE_PATH = orig["HEADING_STORE_PATH"]
        index_mod._config = orig["_config"]


def main() -> None:
    results = run_benchmark()

    # Save results
    TEST_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_file = TEST_RESULTS_DIR / f"benchmark_results_{int(time.time())}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {out_file}")

    # Summary
    ix = results["indexing"]
    mem = results["memory"]
    th = results["threshold"]
    print()
    print("=" * 60)
    print("Indexing Benchmark Results")
    print("=" * 60)
    print(f"  Time:          {ix['total_seconds']:.2f}s")
    print(f"  Files indexed: {ix['files_indexed']}")
    print(f"  Chunks:        {ix['chunk_count']}")
    print(f"  Throughput:    {ix['chunks_per_second']:.1f} chunks/s")
    print(f"  Peak memory:   {mem['peak_mb']:.1f} MB")
    print(f"  Threshold:     {th['max_seconds']}s")
    print(f"  Status:        {'PASSED' if th['passed'] else 'FAILED'}")
    print("=" * 60)

    if not th["passed"]:
        print(
            f"\nFAILED: Indexing took {ix['total_seconds']:.2f}s, "
            f"exceeding the {th['max_seconds']}s threshold."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
