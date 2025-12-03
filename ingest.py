#!/usr/bin/env python3
"""
Ingestion pipeline for building a RAG index from PDF, DOCX, and Markdown documents.
"""
import os
import logging
from pathlib import Path
from dotenv import load_dotenv

from llama_index.core import VectorStoreIndex, StorageContext, Settings, SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Load environment variables
load_dotenv()

# Configuration
DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
STORAGE_DIR = Path(os.getenv("STORAGE_DIR", "./storage"))
EMB_MODEL_NAME = os.getenv("EMB_MODEL_NAME", "BAAI/bge-small-en-v1.5")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def build_index():
    """Build and persist the vector index from documents."""
    logger.info(f"Starting ingestion from {DATA_DIR}")
    
    # Ensure data directory exists
    if not DATA_DIR.exists():
        logger.error(f"Data directory {DATA_DIR} does not exist!")
        raise FileNotFoundError(f"Data directory {DATA_DIR} does not exist")
    
    # Initialize reader
    reader = SimpleDirectoryReader(
        input_dir=str(DATA_DIR),
        required_exts=[".pdf", ".docx", ".md"],
        recursive=True,
    )
    
    # Load documents
    logger.info("Loading documents...")
    docs = reader.load_data()
    
    if not docs:
        logger.warning(f"No documents found in {DATA_DIR}")
        return
    
    logger.info(f"Loaded {len(docs)} documents")
    
    # Initialize embedding model
    logger.info(f"Initializing embedding model: {EMB_MODEL_NAME}")
    embed_model = HuggingFaceEmbedding(model_name=EMB_MODEL_NAME)
    Settings.embed_model = embed_model
    
    # Build index
    logger.info("Building vector index...")
    index = VectorStoreIndex.from_documents(
        docs,
        embed_model=embed_model,
        show_progress=True,
    )
    
    # Persist index
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Persisting index to {STORAGE_DIR}")
    index.storage_context.persist(persist_dir=str(STORAGE_DIR))
    
    logger.info(f"Successfully indexed {len(docs)} documents into {STORAGE_DIR}")


if __name__ == "__main__":
    try:
        build_index()
    except Exception as e:
        logger.error(f"Ingestion failed: {e}", exc_info=True)
        raise

