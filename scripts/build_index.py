#!/usr/bin/env python3
"""
Script to build a search index for the HealthResearchRAG system.
Creates a FAISS index from processed documents.
"""

import os
import sys
import argparse
from pathlib import Path
import yaml

# Add parent directory to path to import healthrag package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from healthrag import (
    Configuration, 
    DocumentIndexer, 
    setup_logger,
    check_dependencies
)

def main():
    """Main function to build the search index."""
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Build search index for HealthResearchRAG.')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to configuration file')
    parser.add_argument('--text-dir', type=str, help='Directory containing text documents')
    parser.add_argument('--markdown-dir', type=str, help='Directory containing markdown documents')
    parser.add_argument('--pdf-dir', type=str, help='Directory containing PDF documents (for direct processing)')
    parser.add_argument('--index-dir', type=str, help='Directory to save the index')
    parser.add_argument('--embedding-model', type=str, help='Embedding model to use')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Set up logger
    log_level = 'DEBUG' if args.verbose else 'INFO'
    logger = setup_logger('healthrag.build_index', getattr(logging, log_level))
    
    # Check dependencies
    logger.info("Checking dependencies...")
    deps = check_dependencies()
    missing = [dep for dep, available in deps.items() if not available and dep in [
        'langchain_text_splitters', 'langchain_huggingface', 'faiss'
    ]]
    
    if missing:
        logger.error(f"Missing required dependencies: {', '.join(missing)}")
        logger.error("Please install them with: pip install " + " ".join(missing))
        return 1
    
    # Load configuration
    config_path = args.config
    if os.path.exists(config_path):
        logger.info(f"Loading configuration from {config_path}")
        config = Configuration(config_path)
    else:
        logger.warning(f"Configuration file {config_path} not found, using defaults")
        config = Configuration()
    
    # Override config with command line arguments if provided
    if args.text_dir:
        config.set_path('processed_dir', args.text_dir)
    
    if args.markdown_dir:
        config.set_path('markdown_dir', args.markdown_dir)
    
    if args.pdf_dir:
        config.set_path('documents_dir', args.pdf_dir)
    
    if args.index_dir:
        config.set_path('index_dir', args.index_dir)
    
    if args.embedding_model:
        config.set_embedding_model(args.embedding_model)
    
    # Create indexer
    indexer = DocumentIndexer(config)
    
    # Get paths
    text_dir = config.get_path('processed_dir')
    markdown_dir = config.get_path('markdown_dir')
    pdf_dir = config.get_path('documents_dir')
    index_dir = config.get_path('index_dir')
    
    # Create document directory mapping
    document_dirs = {
        'text_dir': text_dir,
        'markdown_dir': markdown_dir,
        'pdf_dir': pdf_dir
    }
    
    # Check if at least one source directory has documents
    has_sources = False
    for dir_name, dir_path in document_dirs.items():
        if os.path.exists(dir_path) and any(os.listdir(dir_path)):
            has_sources = True
            break
    
    if not has_sources:
        logger.error("No source documents found. Please run process_documents.py first.")
        return 1
    
    # Build the index
    logger.info(f"Building index in {index_dir}")
    try:
        stats = indexer.create_index(document_dirs, index_dir)
        
        # Print stats
        logger.info(f"Index created with {stats['total_chunks']} chunks from {stats['indexed_docs']} documents")
        logger.info(f"Using {stats['method_used']} as source")
        
        logger.info("Index building completed successfully")
        return 0
    except Exception as e:
        logger.error(f"Error building index: {str(e)}")
        logger.debug("", exc_info=True)
        return 1

if __name__ == '__main__':
    import logging
    sys.exit(main())
