"""
HealthResearchRAG: A Retrieval-Augmented Generation System for Health Research Documents.

This package provides a flexible and extensible RAG system designed specifically
for scientific documents in health research domains.
"""

__version__ = "0.1.0"

from .config import Configuration, config
from .document_processor import DocumentProcessor
from .indexing import DocumentIndexer
from .rag_engine import RAGEngine
from .utils import (
    setup_logger,
    check_dependencies,
    check_ollama_model,
    pull_ollama_model,
    create_example_config
)

# Set up default logger
import logging
logger = setup_logger('healthrag', logging.INFO)
