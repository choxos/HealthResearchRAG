"""
Utility functions for HealthResearchRAG.
"""

import os
import sys
import subprocess
import platform
import logging
from typing import Dict, List, Any, Optional, Tuple
import yaml

def setup_logger(name: str, log_level: int = logging.INFO) -> logging.Logger:
    """
    Set up a logger with the specified name and level.
    
    Args:
        name: Logger name
        log_level: Logging level
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # Create console handler
    handler = logging.StreamHandler()
    handler.setLevel(log_level)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(handler)
    
    return logger

def check_dependencies() -> Dict[str, bool]:
    """
    Check if all required dependencies are installed.
    
    Returns:
        Dictionary of dependencies and their availability
    """
    dependencies = {
        "langchain": False,
        "langchain_community": False,
        "langchain_ollama": False,
        "langchain_huggingface": False,
        "langchain_text_splitters": False,
        "pymupdf": False,
        "pymupdf4llm": False,
        "faiss": False,
        "ollama": False,
        "transformers": False,
        "yaml": False
    }
    
    # Check each dependency
    try:
        import langchain
        dependencies["langchain"] = True
    except ImportError:
        pass
    
    try:
        import langchain_community
        dependencies["langchain_community"] = True
    except ImportError:
        pass
    
    try:
        import langchain_ollama
        dependencies["langchain_ollama"] = True
    except ImportError:
        pass
    
    try:
        import langchain_huggingface
        dependencies["langchain_huggingface"] = True
    except ImportError:
        pass
    
    try:
        import langchain_text_splitters
        dependencies["langchain_text_splitters"] = True
    except ImportError:
        pass
    
    try:
        import fitz
        dependencies["pymupdf"] = True
    except ImportError:
        pass
    
    try:
        import pymupdf4llm
        dependencies["pymupdf4llm"] = True
    except ImportError:
        pass
    
    try:
        import faiss
        dependencies["faiss"] = True
    except ImportError:
        try:
            import faiss_cpu
            dependencies["faiss"] = True
        except ImportError:
            pass
    
    try:
        import yaml
        dependencies["yaml"] = True
    except ImportError:
        pass
    
    try:
        import transformers
        dependencies["transformers"] = True
    except ImportError:
        pass
    
    # Check if Ollama is installed (using subprocess)
    try:
        if platform.system() == "Windows":
            from shutil import which
            dependencies["ollama"] = which("ollama.exe") is not None
        else:
            subprocess.run(["which", "ollama"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            dependencies["ollama"] = True
    except (subprocess.SubprocessError, FileNotFoundError):
        pass
    
    return dependencies

def check_ollama_model(model_name: str) -> bool:
    """
    Check if a specific model is available in Ollama.
    
    Args:
        model_name: Name of the model to check
        
    Returns:
        Whether the model is available
    """
    try:
        result = subprocess.run(
            ["ollama", "list"], 
            check=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True
        )
        return model_name in result.stdout
    except (subprocess.SubprocessError, FileNotFoundError):
        return False

def pull_ollama_model(model_name: str) -> bool:
    """
    Pull a model using Ollama.
    
    Args:
        model_name: Name of the model to pull
        
    Returns:
        Whether the pull was successful
    """
    try:
        print(f"Pulling the {model_name} model... This may take some time.")
        subprocess.run(
            ["ollama", "pull", model_name], 
            check=True, 
            stdout=sys.stdout, 
            stderr=sys.stderr
        )
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        return False

def create_example_config(output_path: str) -> None:
    """
    Create an example configuration file.
    
    Args:
        output_path: Path to save the example config
    """
    example_config = {
        "project": {
            "title": "COVID-19 Research RAG System",
            "description": "A RAG system for COVID-19 research papers"
        },
        "paths": {
            "documents_dir": "Documents",
            "processed_dir": "Processed",
            "markdown_dir": "Markdown",
            "index_dir": "Index"
        },
        "models": {
            "llm": {
                "provider": "ollama",
                "name": "llama3",
                "parameters": {
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "max_tokens": 2000
                }
            },
            "embedding": {
                "name": "BAAI/bge-small-en-v1.5",
                "dimensions": 384,
                "normalize": True
            }
        },
        "retrieval": {
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "results_count": 5
        },
        "prompts": {
            "system_template": """You are an expert in COVID-19 research, analyzing scientific documents to provide accurate information.
Use the following context extracted from scientific literature to answer the question accurately.

Context:
{context}

Based on this information, provide a comprehensive and relevant answer to the following question:
Question: {question}

If the context doesn't contain sufficient information, clearly state that the information is not available in the provided documents.
When appropriate, provide step-by-step explanation and highlight key considerations.
Always cite the sources of your information from the context by mentioning the document names."""
        }
    }
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Write config to file
    with open(output_path, 'w') as f:
        yaml.dump(example_config, f, default_flow_style=False)
    
    print(f"Example configuration created at {output_path}")
