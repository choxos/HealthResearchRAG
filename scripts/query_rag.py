#!/usr/bin/env python3
"""
Script to query the HealthResearchRAG system.
Loads the index and starts an interactive question-answering session.
"""

import os
import sys
import argparse
from pathlib import Path
import yaml
import time

# Add parent directory to path to import healthrag package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from healthrag import (
    Configuration, 
    DocumentIndexer,
    RAGEngine,
    setup_logger,
    check_dependencies,
    check_ollama_model,
    pull_ollama_model
)

def main():
    """Main function to query the RAG system."""
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Query the HealthResearchRAG system.')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to configuration file')
    parser.add_argument('--index-dir', type=str, help='Directory containing the index')
    parser.add_argument('--model', type=str, help='LLM model to use (e.g., llama3, deepseek-r1:8b)')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    parser.add_argument('--question', type=str, help='Single question to answer (non-interactive mode)')
    
    args = parser.parse_args()
    
    # Set up logger
    log_level = 'DEBUG' if args.verbose else 'INFO'
    logger = setup_logger('healthrag.query', getattr(logging, log_level))
    
    # Check dependencies
    logger.info("Checking dependencies...")
    deps = check_dependencies()
    missing = [dep for dep, available in deps.items() if not available and dep in [
        'langchain', 'langchain_ollama', 'langchain_huggingface', 'faiss'
    ]]
    
    if missing:
        logger.error(f"Missing required dependencies: {', '.join(missing)}")
        logger.error("Please install them with: pip install " + " ".join(missing))
        return 1
    
    # Check if Ollama is available
    if not deps.get('ollama', False):
        logger.error("Ollama is not installed or not in PATH. Please install Ollama.")
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
    if args.index_dir:
        config.set_path('index_dir', args.index_dir)
    
    if args.model:
        llm_config = config.get_llm_config()
        config.set_llm_model(llm_config['provider'], args.model)
    
    # Get model info
    llm_config = config.get_llm_config()
    model_name = llm_config['name']
    
    # Check if model is available in Ollama
    logger.info(f"Checking if model {model_name} is available in Ollama...")
    if not check_ollama_model(model_name):
        logger.warning(f"Model {model_name} not found in Ollama")
        pull = input(f"Do you want to pull the model {model_name}? (y/n): ")
        if pull.lower() == 'y':
            if not pull_ollama_model(model_name):
                logger.error(f"Failed to pull model {model_name}")
                return 1
        else:
            logger.error(f"Model {model_name} is required but not available")
            return 1
    
    # Get index path
    index_dir = config.get_path('index_dir')
    
    # Check if index exists
    if not os.path.exists(index_dir):
        logger.error(f"Index directory {index_dir} does not exist.")
        logger.error("Please run build_index.py first to create the index.")
        return 1
    
    # Load the index
    logger.info(f"Loading index from {index_dir}...")
    indexer = DocumentIndexer(config)
    
    try:
        vector_store = indexer.load_index(index_dir)
        logger.info("Index loaded successfully")
    except Exception as e:
        logger.error(f"Error loading index: {str(e)}")
        logger.debug("", exc_info=True)
        return 1
    
    # Initialize RAG engine
    logger.info("Initializing RAG engine...")
    engine = RAGEngine(config, vector_store, logger)
    
    # Display system info
    project_title = config.get_project_title()
    embedding_model = config.get_embedding_model()
    
    print("\n" + "="*50)
    print(f"{project_title}")
    print("="*50)
    print(f"Model: {model_name}")
    print(f"Embedding: {embedding_model}")
    
    # Check if single question mode
    if args.question:
        print("\nQuestion:", args.question)
        start_time = time.time()
        result = engine.generate_answer(args.question)
        elapsed_time = time.time() - start_time
        
        print("\nAnswer:")
        print(result['answer'])
        
        if result.get('sources'):
            print("\nSources:")
            for source in result['sources']:
                print(f"- {source}")
        
        print(f"\nGenerated in {elapsed_time:.2f} seconds")
        return 0
    
    # Interactive mode
    print("\nInteractive query mode. Type 'exit', 'quit', or press Ctrl+C to exit.")
    print("-"*50)
    
    try:
        while True:
            print()
            question = input("Question: ")
            
            if question.lower() in ['exit', 'quit', 'q']:
                print("Exiting...")
                break
            
            if not question.strip():
                continue
            
            start_time = time.time()
            result = engine.generate_answer(question)
            elapsed_time = time.time() - start_time
            
            print("\nAnswer:")
            print(result['answer'])
            
            if result.get('sources'):
                print("\nSources:")
                for source in result['sources']:
                    print(f"- {source}")
            
            print(f"\nGenerated in {elapsed_time:.2f} seconds")
            print("-"*50)
            
    except KeyboardInterrupt:
        print("\nInterrupted. Exiting...")
    
    return 0

if __name__ == '__main__':
    import logging
    sys.exit(main())
