#!/usr/bin/env python3
"""
Automated optimization script for RAG retrieval parameters.

This script:
1. Tests different combinations of RETRIEVAL_EMBED_TOP_K and RETRIEVAL_RERANK_TOP_K
2. Tests different embedding models (CPU-based)
3. Tests different reranking models (CPU-based)
4. Re-optimizes parameters for each model combination
5. Uses test_large_scale.py for benchmarking with a MASSIVE corpus (100+ documents)

The script finds optimal parameters by maximizing a composite score based on:
- Precision@5
- Recall@5
- MRR (Mean Reciprocal Rank)
- NDCG@5 (Normalized Discounted Cumulative Gain)

NOTE: With the expanded corpus (100+ documents including academic papers, technical docs,
literature, and generated content), optimization may take significantly longer. The parameter
search space has been expanded to handle larger document collections effectively.
"""

import asyncio
import itertools
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Try to load .env file if dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv not available, continue without it
    pass

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Force online mode for tests (to download models if needed)
os.environ["OFFLINE"] = "0"

# Prevent Mac from sleeping during optimization
_caffeinate_process = None

def _prevent_sleep():
    """Prevent Mac from sleeping during optimization."""
    global _caffeinate_process
    try:
        # Use caffeinate to prevent sleep
        _caffeinate_process = subprocess.Popen(
            ["caffeinate", "-d"],  # -d prevents display sleep
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        logger.info("Mac sleep prevention enabled (caffeinate)")
    except Exception as e:
        logger.warning(f"Could not prevent sleep: {e}")

def _allow_sleep():
    """Allow Mac to sleep again."""
    global _caffeinate_process
    if _caffeinate_process:
        try:
            _caffeinate_process.terminate()
            _caffeinate_process.wait(timeout=5)
            logger.info("Mac sleep prevention disabled")
        except Exception:
            _caffeinate_process.kill()
        _caffeinate_process = None

# Import test utilities
from test_large_scale import (
    TEST_QUERIES,
    download_test_corpus,
    evaluate_query_with_retriever,
    TEST_DATA_DIR,
    TEST_STORAGE_DIR,
    TEST_RESULTS_DIR,
)

# CPU-based embedding models to test (FastEmbed compatible)
EMBEDDING_MODELS = [
    "sentence-transformers/all-MiniLM-L6-v2",  # Alternative small model
]

# CPU-based reranking models to test (FlashRank compatible)
# Note: L-6 models are automatically mapped to L-12 by the codebase
RERANKING_MODELS = [
    "ms-marco-MiniLM-L-12-v2",  # Default - good balance
]

# Parameter search space - expanded for MASSIVE corpus
# RETRIEVAL_EMBED_TOP_K: Number of candidates from embedding search
# With 100+ documents, we need larger ranges to ensure good retrieval
EMBED_TOP_K_OPTIONS = [20, 100, 300]

# RETRIEVAL_RERANK_TOP_K: Number of final results after reranking
# Expanded range to test different final result sizes
RERANK_TOP_K_OPTIONS = [3, 10]


@dataclass
class OptimizationConfig:
    """Configuration for a single optimization run."""
    embed_model: str
    rerank_model: str
    embed_top_k: int
    rerank_top_k: int


@dataclass
class OptimizationResult:
    """Results from a single optimization run."""
    config: OptimizationConfig
    metrics: Dict[str, float]
    evaluation_time: float
    composite_score: float


def calculate_composite_score(metrics: Dict[str, float]) -> float:
    """
    Calculate a composite score from multiple metrics.
    
    Weights:
    - Precision@5: 0.25 (how many retrieved are relevant)
    - Recall@5: 0.25 (how many relevant are retrieved)
    - MRR: 0.25 (position of first relevant result)
    - NDCG@5: 0.25 (ranking quality)
    
    Returns a score between 0 and 1.
    """
    precision = metrics.get("precision@5", 0.0)
    recall = metrics.get("recall@5", 0.0)
    mrr = metrics.get("mrr", 0.0)
    ndcg = metrics.get("ndcg@5", 0.0)
    
    # Weighted average
    composite = (
        0.25 * precision +
        0.25 * recall +
        0.25 * mrr +
        0.25 * ndcg
    )
    
    return composite


async def run_benchmark_with_config(
    config: OptimizationConfig,
    test_storage_dir: Path,
    test_data_dir: Path,
) -> OptimizationResult:
    """
    Run benchmark with a specific configuration.
    
    This function:
    1. Sets environment variables for the configuration
    2. Rebuilds the index if needed (if embedding model changed)
    3. Runs the test suite
    4. Returns results
    """
    logger.info(f"\n{'=' * 80}")
    logger.info(f"Testing Configuration:")
    logger.info(f"  Embedding Model: {config.embed_model}")
    logger.info(f"  Rerank Model: {config.rerank_model}")
    logger.info(f"  Embed Top K: {config.embed_top_k}")
    logger.info(f"  Rerank Top K: {config.rerank_top_k}")
    logger.info(f"{'=' * 80}")
    
    start_time = time.time()
    
    # Store original environment variables
    original_embed_model = os.environ.get("RETRIEVAL_EMBED_MODEL_NAME")
    original_rerank_model = os.environ.get("RETRIEVAL_RERANK_MODEL_NAME")
    original_embed_top_k = os.environ.get("RETRIEVAL_EMBED_TOP_K")
    original_rerank_top_k = os.environ.get("RETRIEVAL_RERANK_TOP_K")
    original_data_dir = os.environ.get("DATA_DIR")
    original_storage_dir = os.environ.get("STORAGE_DIR")
    
    try:
        # Set configuration environment variables
        os.environ["RETRIEVAL_EMBED_MODEL_NAME"] = config.embed_model
        os.environ["RETRIEVAL_RERANK_MODEL_NAME"] = config.rerank_model
        os.environ["RETRIEVAL_EMBED_TOP_K"] = str(config.embed_top_k)
        os.environ["RETRIEVAL_RERANK_TOP_K"] = str(config.rerank_top_k)
        os.environ["DATA_DIR"] = str(test_data_dir)
        os.environ["STORAGE_DIR"] = str(test_storage_dir)
        
        # Re-import modules to pick up new environment variables
        import importlib
        import ingest
        import mcp_server
        importlib.reload(ingest)
        importlib.reload(mcp_server)
        
        # Re-import after reload
        from ingest import build_index as build_test_index
        from mcp_server import retrieve_docs as retrieve_docs_reloaded
        
        # Check if we need to rebuild the index (if embedding model changed)
        # We'll use a simple heuristic: rebuild if storage doesn't exist or if model changed
        storage_exists = (test_storage_dir / "docstore.json").exists()
        model_cache_file = test_storage_dir / ".embed_model"
        
        needs_rebuild = False
        if not storage_exists:
            needs_rebuild = True
        elif model_cache_file.exists():
            with open(model_cache_file, "r") as f:
                cached_model = f.read().strip()
            if cached_model != config.embed_model:
                needs_rebuild = True
        else:
            needs_rebuild = True
        
        if needs_rebuild:
            logger.info("Rebuilding index with new embedding model...")
            build_test_index()
            # Cache the embedding model name
            model_cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(model_cache_file, "w") as f:
                f.write(config.embed_model)
        else:
            logger.info("Using existing index (embedding model unchanged)")
        
        # Run evaluation queries
        evaluations = []
        for query, keywords, patterns, difficulty in TEST_QUERIES:
            try:
                eval_result = await evaluate_query_with_retriever(
                    query, keywords, patterns, difficulty, retrieve_docs_reloaded
                )
                evaluations.append(eval_result)
            except Exception as e:
                logger.error(f"Error evaluating query '{query}': {e}")
                evaluations.append({
                    "query": query,
                    "error": str(e),
                })
        
        # Calculate aggregate metrics
        successful_evals = [e for e in evaluations if "error" not in e]
        
        if not successful_evals:
            logger.warning("No successful evaluations!")
            return OptimizationResult(
                config=config,
                metrics={},
                evaluation_time=time.time() - start_time,
                composite_score=0.0,
            )
        
        aggregate_metrics = {
            "precision@1": sum(e["metrics"]["precision@1"] for e in successful_evals) / len(successful_evals),
            "precision@5": sum(e["metrics"]["precision@5"] for e in successful_evals) / len(successful_evals),
            "recall@5": sum(e["metrics"]["recall@5"] for e in successful_evals) / len(successful_evals),
            "mrr": sum(e["metrics"]["mrr"] for e in successful_evals) / len(successful_evals),
            "ndcg@5": sum(e["metrics"].get("ndcg@5", 0) for e in successful_evals) / len(successful_evals),
        }
        
        composite_score = calculate_composite_score(aggregate_metrics)
        
        logger.info(f"\nResults:")
        logger.info(f"  Precision@1: {aggregate_metrics['precision@1']:.3f}")
        logger.info(f"  Precision@5: {aggregate_metrics['precision@5']:.3f}")
        logger.info(f"  Recall@5: {aggregate_metrics['recall@5']:.3f}")
        logger.info(f"  MRR: {aggregate_metrics['mrr']:.3f}")
        logger.info(f"  NDCG@5: {aggregate_metrics['ndcg@5']:.3f}")
        logger.info(f"  Composite Score: {composite_score:.3f}")
        
        return OptimizationResult(
            config=config,
            metrics=aggregate_metrics,
            evaluation_time=time.time() - start_time,
            composite_score=composite_score,
        )
        
    finally:
        # Restore original environment variables
        if original_embed_model:
            os.environ["RETRIEVAL_EMBED_MODEL_NAME"] = original_embed_model
        elif "RETRIEVAL_EMBED_MODEL_NAME" in os.environ:
            del os.environ["RETRIEVAL_EMBED_MODEL_NAME"]
        
        if original_rerank_model:
            os.environ["RETRIEVAL_RERANK_MODEL_NAME"] = original_rerank_model
        elif "RETRIEVAL_RERANK_MODEL_NAME" in os.environ:
            del os.environ["RETRIEVAL_RERANK_MODEL_NAME"]
        
        if original_embed_top_k:
            os.environ["RETRIEVAL_EMBED_TOP_K"] = original_embed_top_k
        elif "RETRIEVAL_EMBED_TOP_K" in os.environ:
            del os.environ["RETRIEVAL_EMBED_TOP_K"]
        
        if original_rerank_top_k:
            os.environ["RETRIEVAL_RERANK_TOP_K"] = original_rerank_top_k
        elif "RETRIEVAL_RERANK_TOP_K" in os.environ:
            del os.environ["RETRIEVAL_RERANK_TOP_K"]
        
        if original_data_dir:
            os.environ["DATA_DIR"] = original_data_dir
        elif "DATA_DIR" in os.environ:
            del os.environ["DATA_DIR"]
        
        if original_storage_dir:
            os.environ["STORAGE_DIR"] = original_storage_dir
        elif "STORAGE_DIR" in os.environ:
            del os.environ["STORAGE_DIR"]


async def optimize_parameters_for_model_pair(
    embed_model: str,
    rerank_model: str,
    test_storage_dir: Path,
    test_data_dir: Path,
) -> Tuple[OptimizationResult, List[OptimizationResult]]:
    """
    Find optimal parameters for a specific embedding/reranking model pair.
    
    Returns:
        Tuple of (best_result, all_results)
    """
    logger.info(f"\n{'#' * 80}")
    logger.info(f"Optimizing parameters for:")
    logger.info(f"  Embedding: {embed_model}")
    logger.info(f"  Reranking: {rerank_model}")
    logger.info(f"{'#' * 80}")
    
    all_results = []
    
    # Test all parameter combinations
    total_combinations = len(EMBED_TOP_K_OPTIONS) * len(RERANK_TOP_K_OPTIONS)
    current = 0
    
    for embed_top_k, rerank_top_k in itertools.product(EMBED_TOP_K_OPTIONS, RERANK_TOP_K_OPTIONS):
        current += 1
        logger.info(f"\n[{current}/{total_combinations}] Testing embed_top_k={embed_top_k}, rerank_top_k={rerank_top_k}")
        
        config = OptimizationConfig(
            embed_model=embed_model,
            rerank_model=rerank_model,
            embed_top_k=embed_top_k,
            rerank_top_k=rerank_top_k,
        )
        
        result = await run_benchmark_with_config(config, test_storage_dir, test_data_dir)
        all_results.append(result)
    
    # Find best result
    best_result = max(all_results, key=lambda r: r.composite_score)
    
    logger.info(f"\n{'#' * 80}")
    logger.info(f"Best parameters for {embed_model} + {rerank_model}:")
    logger.info(f"  Embed Top K: {best_result.config.embed_top_k}")
    logger.info(f"  Rerank Top K: {best_result.config.rerank_top_k}")
    logger.info(f"  Composite Score: {best_result.composite_score:.3f}")
    logger.info(f"  Precision@5: {best_result.metrics.get('precision@5', 0):.3f}")
    logger.info(f"  Recall@5: {best_result.metrics.get('recall@5', 0):.3f}")
    logger.info(f"  MRR: {best_result.metrics.get('mrr', 0):.3f}")
    logger.info(f"  NDCG@5: {best_result.metrics.get('ndcg@5', 0):.3f}")
    logger.info(f"{'#' * 80}")
    
    return best_result, all_results


def _get_checkpoint_file() -> Path:
    """Get the checkpoint file path."""
    TEST_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return TEST_RESULTS_DIR / "optimization_checkpoint.json"

def _load_checkpoint() -> Optional[Dict[str, Any]]:
    """Load checkpoint if it exists."""
    checkpoint_file = _get_checkpoint_file()
    if checkpoint_file.exists():
        try:
            with open(checkpoint_file, "r") as f:
                checkpoint = json.load(f)
            logger.info(f"Found checkpoint file: {checkpoint_file}")
            logger.info(f"Resuming from checkpoint with {len(checkpoint.get('model_results', []))} completed model combinations")
            return checkpoint
        except Exception as e:
            logger.warning(f"Could not load checkpoint: {e}")
    return None

def _save_checkpoint(
    all_model_results: List[Dict[str, Any]],
    overall_best: Optional[Dict[str, Any]],
    overall_best_score: float,
    completed_combinations: set,
) -> None:
    """Save checkpoint with current progress."""
    checkpoint_file = _get_checkpoint_file()
    checkpoint = {
        "timestamp": time.time(),
        "overall_best": overall_best,
        "overall_best_score": overall_best_score,
        "model_results": all_model_results,
        "completed_combinations": list(completed_combinations),
        "parameter_search_space": {
            "embed_top_k_options": EMBED_TOP_K_OPTIONS,
            "rerank_top_k_options": RERANK_TOP_K_OPTIONS,
        },
        "models_tested": {
            "embedding_models": EMBEDDING_MODELS,
            "reranking_models": RERANKING_MODELS,
        },
    }
    try:
        with open(checkpoint_file, "w") as f:
            json.dump(checkpoint, f, indent=2)
        logger.debug(f"Checkpoint saved: {checkpoint_file}")
    except Exception as e:
        logger.warning(f"Could not save checkpoint: {e}")

async def run_full_optimization() -> Dict[str, Any]:
    """
    Run full optimization across all model combinations.
    
    Process:
    1. Check for existing checkpoint and resume if present
    2. Download test corpus (once)
    3. For each embedding model:
       a. For each reranking model:
          - Skip if already completed (from checkpoint)
          - Optimize parameters
          - Record results
          - Save checkpoint
    4. Find overall best configuration
    """
    # Prevent Mac from sleeping
    _prevent_sleep()
    
    try:
        logger.info("=" * 80)
        logger.info("RAG Retrieval Parameter Optimization")
        logger.info("=" * 80)
        
        # Check for checkpoint
        checkpoint = _load_checkpoint()
        all_model_results = checkpoint.get("model_results", []) if checkpoint else []
        overall_best = checkpoint.get("overall_best") if checkpoint else None
        overall_best_score = checkpoint.get("overall_best_score", -1.0) if checkpoint else -1.0
        completed_combinations = set(
            tuple(c) for c in checkpoint.get("completed_combinations", [])
        ) if checkpoint else set()
        
        if checkpoint:
            logger.info(f"Resuming optimization - {len(all_model_results)} combinations already completed")
        
        # Step 1: Download test corpus
        logger.info("\n" + "=" * 80)
        logger.info("Step 1: Downloading Test Corpus")
        logger.info("=" * 80)
        
        downloaded_files = download_test_corpus()
        
        if not any(downloaded_files.values()):
            logger.error("No documents downloaded. Cannot proceed with optimization.")
            return {"error": "No documents downloaded"}
        
        # Step 2: Create unique storage directories for each model combination
        # This prevents index conflicts when switching models
        base_test_storage = TEST_STORAGE_DIR
        base_test_data = TEST_DATA_DIR
        
        # Step 3: Test each model combination
        total_combinations = len(EMBEDDING_MODELS) * len(RERANKING_MODELS)
        current_combination = 0
    
        for embed_model in EMBEDDING_MODELS:
            for rerank_model in RERANKING_MODELS:
                current_combination += 1
                combination_key = (embed_model, rerank_model)
                
                # Skip if already completed
                if combination_key in completed_combinations:
                    logger.info(f"\n{'=' * 80}")
                    logger.info(f"Model Combination {current_combination}/{total_combinations} (SKIPPED - already completed)")
                    logger.info(f"  Embedding: {embed_model}")
                    logger.info(f"  Reranking: {rerank_model}")
                    logger.info(f"{'=' * 80}")
                    continue
                
                logger.info(f"\n{'=' * 80}")
                logger.info(f"Model Combination {current_combination}/{total_combinations}")
                logger.info(f"{'=' * 80}")
                
                # Create unique storage directory for this model combination
                model_storage_dir = base_test_storage / f"{embed_model.replace('/', '_')}_{rerank_model.replace('/', '_')}"
                
                try:
                    best_result, all_results = await optimize_parameters_for_model_pair(
                        embed_model,
                        rerank_model,
                        model_storage_dir,
                        base_test_data,
                    )
                    
                    result_entry = {
                        "embed_model": embed_model,
                        "rerank_model": rerank_model,
                        "best_config": {
                            "embed_top_k": best_result.config.embed_top_k,
                            "rerank_top_k": best_result.config.rerank_top_k,
                        },
                        "best_metrics": best_result.metrics,
                        "best_composite_score": best_result.composite_score,
                        "all_results": [
                            {
                                "embed_top_k": r.config.embed_top_k,
                                "rerank_top_k": r.config.rerank_top_k,
                                "metrics": r.metrics,
                                "composite_score": r.composite_score,
                            }
                            for r in all_results
                        ],
                    }
                    
                    all_model_results.append(result_entry)
                    completed_combinations.add(combination_key)
                    
                    if best_result.composite_score > overall_best_score:
                        overall_best_score = best_result.composite_score
                        overall_best = {
                            "embed_model": embed_model,
                            "rerank_model": rerank_model,
                            "embed_top_k": best_result.config.embed_top_k,
                            "rerank_top_k": best_result.config.rerank_top_k,
                            "metrics": best_result.metrics,
                            "composite_score": best_result.composite_score,
                        }
                    
                    # Save checkpoint after each model combination
                    _save_checkpoint(all_model_results, overall_best, overall_best_score, completed_combinations)
                    logger.info(f"Checkpoint saved - {len(completed_combinations)}/{total_combinations} combinations completed")
                        
                except Exception as e:
                    logger.error(f"Error optimizing {embed_model} + {rerank_model}: {e}")
                    import traceback
                    traceback.print_exc()
                    # Still save checkpoint even on error
                    _save_checkpoint(all_model_results, overall_best, overall_best_score, completed_combinations)
                    continue
    
        # Step 4: Save final results
        TEST_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        results_file = TEST_RESULTS_DIR / f"optimization_results_{int(time.time())}.json"
        
        results = {
            "timestamp": time.time(),
            "overall_best": overall_best,
            "model_results": all_model_results,
            "parameter_search_space": {
                "embed_top_k_options": EMBED_TOP_K_OPTIONS,
                "rerank_top_k_options": RERANK_TOP_K_OPTIONS,
            },
            "models_tested": {
                "embedding_models": EMBEDDING_MODELS,
                "reranking_models": RERANKING_MODELS,
            },
        }
        
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        
        # Also save as latest checkpoint
        _save_checkpoint(all_model_results, overall_best, overall_best_score, completed_combinations)
        
        logger.info(f"\n{'=' * 80}")
        logger.info("Optimization Complete")
        logger.info(f"{'=' * 80}")
        
        if overall_best:
            logger.info("\nOverall Best Configuration:")
            logger.info(f"  Embedding Model: {overall_best['embed_model']}")
            logger.info(f"  Reranking Model: {overall_best['rerank_model']}")
            logger.info(f"  Embed Top K: {overall_best['embed_top_k']}")
            logger.info(f"  Rerank Top K: {overall_best['rerank_top_k']}")
            logger.info(f"  Composite Score: {overall_best['composite_score']:.3f}")
            logger.info(f"  Precision@5: {overall_best['metrics'].get('precision@5', 0):.3f}")
            logger.info(f"  Recall@5: {overall_best['metrics'].get('recall@5', 0):.3f}")
            logger.info(f"  MRR: {overall_best['metrics'].get('mrr', 0):.3f}")
            logger.info(f"  NDCG@5: {overall_best['metrics'].get('ndcg@5', 0):.3f}")
            
            # Show top 5 configurations
            logger.info("\n" + "=" * 80)
            logger.info("Top 5 Model Combinations (by Composite Score):")
            logger.info("=" * 80)
            
            sorted_results = sorted(
                all_model_results,
                key=lambda x: x.get("best_composite_score", 0),
                reverse=True
            )
            
            for i, result in enumerate(sorted_results[:5], 1):
                logger.info(f"\n{i}. {result['embed_model']} + {result['rerank_model']}")
                logger.info(f"   Embed Top K: {result['best_config']['embed_top_k']}, "
                           f"Rerank Top K: {result['best_config']['rerank_top_k']}")
                logger.info(f"   Composite Score: {result.get('best_composite_score', 0):.3f}")
                metrics = result.get('best_metrics', {})
                logger.info(f"   Precision@5: {metrics.get('precision@5', 0):.3f}, "
                           f"Recall@5: {metrics.get('recall@5', 0):.3f}, "
                           f"MRR: {metrics.get('mrr', 0):.3f}, "
                           f"NDCG@5: {metrics.get('ndcg@5', 0):.3f}")
        
        logger.info(f"\nResults saved to: {results_file}")
        
        # Remove checkpoint file on successful completion
        checkpoint_file = _get_checkpoint_file()
        if checkpoint_file.exists():
            try:
                checkpoint_file.unlink()
                logger.info("Checkpoint file removed (optimization complete)")
            except Exception as e:
                logger.warning(f"Could not remove checkpoint file: {e}")
        
        return results
    
    finally:
        # Always allow sleep when done
        _allow_sleep()


def main():
    """Main entry point."""
    try:
        results = asyncio.run(run_full_optimization())
        
        if "error" in results:
            logger.error(f"Optimization failed: {results['error']}")
            sys.exit(1)
        
        logger.info("\n" + "=" * 80)
        logger.info("Optimization Completed Successfully")
        logger.info("=" * 80)
    except KeyboardInterrupt:
        logger.info("\n" + "=" * 80)
        logger.info("Optimization interrupted by user")
        logger.info("Progress has been saved to checkpoint file")
        logger.info("Run the script again to resume from where it left off")
        logger.info("=" * 80)
        _allow_sleep()
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        _allow_sleep()
        sys.exit(1)


if __name__ == "__main__":
    main()
