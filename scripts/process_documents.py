#!/usr/bin/env python3
"""
Script to process PDF documents for the HealthResearchRAG system.
Converts PDFs to text and markdown formats.
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
    DocumentProcessor, 
    setup_logger,
    check_dependencies
)

def main():
    """Main function to process documents."""
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Process PDF documents for HealthResearchRAG.')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to configuration file')
    parser.add_argument('--pdf-dir', type=str, help='Directory containing PDF documents')
    parser.add_argument('--output-dir', type=str, help='Directory to save processed documents')
    parser.add_argument('--markdown-dir', type=str, help='Directory to save markdown documents')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Set up logger
    log_level = 'DEBUG' if args.verbose else 'INFO'
    logger = setup_logger('healthrag.process_documents', getattr(logging, log_level))
    
    # Check dependencies
    logger.info("Checking dependencies...")
    deps = check_dependencies()
    missing = [dep for dep, available in deps.items() if not available and dep in [
        'pymupdf', 'langchain_community'
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
    if args.pdf_dir:
        config.set_path('documents_dir', args.pdf_dir)
    
    if args.output_dir:
        config.set_path('processed_dir', args.output_dir)
    
    if args.markdown_dir:
        config.set_path('markdown_dir', args.markdown_dir)
    
    # Create processor
    processor = DocumentProcessor(config)
    
    # Get paths
    pdf_dir = config.get_path('documents_dir')
    text_dir = config.get_path('processed_dir')
    markdown_dir = config.get_path('markdown_dir')
    
    # Check if directories exist
    if not os.path.exists(pdf_dir):
        logger.error(f"PDF directory {pdf_dir} does not exist")
        return 1
    
    # Create output directories if they don't exist
    os.makedirs(text_dir, exist_ok=True)
    if markdown_dir:
        os.makedirs(markdown_dir, exist_ok=True)
    
    # Process documents
    logger.info(f"Processing documents from {pdf_dir}")
    stats = processor.process_documents(pdf_dir, text_dir, markdown_dir)
    
    # Print stats
    logger.info(f"Processed {stats['processed_text']} documents to text")
    if 'processed_markdown' in stats and stats['processed_markdown'] > 0:
        logger.info(f"Processed {stats['processed_markdown']} documents to markdown")
    
    if stats['failed'] > 0:
        logger.warning(f"Failed to process {stats['failed']} documents")
        
    if stats['processed_text'] == 0 and (not 'processed_markdown' in stats or stats['processed_markdown'] == 0):
        logger.error("No documents were processed successfully")
        return 1
    
    logger.info("Document processing completed successfully")
    return 0

if __name__ == '__main__':
    import logging
    sys.exit(main())
