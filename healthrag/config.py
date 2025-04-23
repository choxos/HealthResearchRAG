"""
Configuration module for HealthResearchRAG.
Contains settings for file paths, model parameters and application defaults.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List

# Default base configuration
DEFAULT_CONFIG = {
    "project": {
        "title": "Health Research RAG System",
        "description": "A RAG system for health research documents"
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
        "system_template": """You are an expert in health research, analyzing scientific documents to provide accurate information.
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


class Configuration:
    """Configuration manager for HealthResearchRAG application."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration with default values and optionally load from file.
        
        Args:
            config_path: Path to YAML configuration file (optional)
        """
        self.config = DEFAULT_CONFIG.copy()
        
        if config_path and os.path.exists(config_path):
            self.load_from_file(config_path)
            
    def load_from_file(self, config_path: str) -> None:
        """
        Load configuration from a YAML file.
        
        Args:
            config_path: Path to YAML configuration file
        """
        try:
            with open(config_path, 'r') as file:
                user_config = yaml.safe_load(file)
                self._update_nested_dict(self.config, user_config)
        except Exception as e:
            print(f"Error loading configuration file: {e}")
    
    def save_to_file(self, config_path: str) -> None:
        """
        Save current configuration to a YAML file.
        
        Args:
            config_path: Path to save the configuration file
        """
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w') as file:
            yaml.dump(self.config, file, default_flow_style=False)
    
    def _update_nested_dict(self, d: Dict, u: Dict) -> Dict:
        """
        Update a nested dictionary with another nested dictionary.
        
        Args:
            d: Dictionary to update
            u: Dictionary with updates
            
        Returns:
            Updated dictionary
        """
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                self._update_nested_dict(d[k], v)
            else:
                d[k] = v
        return d
    
    def get_project_title(self) -> str:
        """Get the project title."""
        return self.config["project"]["title"]
    
    def set_project_title(self, title: str) -> None:
        """Set the project title."""
        self.config["project"]["title"] = title
    
    def get_paths(self) -> Dict[str, str]:
        """Get all configured paths."""
        return self.config["paths"]
    
    def get_path(self, path_name: str) -> str:
        """Get a specific path by name."""
        return self.config["paths"].get(path_name, "")
    
    def set_path(self, path_name: str, path_value: str) -> None:
        """Set a specific path."""
        self.config["paths"][path_name] = path_value
    
    def get_llm_config(self) -> Dict[str, Any]:
        """Get the LLM configuration."""
        return self.config["models"]["llm"]
    
    def set_llm_model(self, provider: str, model_name: str) -> None:
        """Set the LLM provider and model name."""
        self.config["models"]["llm"]["provider"] = provider
        self.config["models"]["llm"]["name"] = model_name
    
    def get_embedding_model(self) -> str:
        """Get the embedding model name."""
        return self.config["models"]["embedding"]["name"]
    
    def set_embedding_model(self, model_name: str) -> None:
        """Set the embedding model name."""
        self.config["models"]["embedding"]["name"] = model_name
    
    def get_chunk_settings(self) -> Dict[str, int]:
        """Get chunk size and overlap settings."""
        return {
            "chunk_size": self.config["retrieval"]["chunk_size"],
            "chunk_overlap": self.config["retrieval"]["chunk_overlap"]
        }
    
    def get_system_prompt_template(self) -> str:
        """Get the system prompt template."""
        return self.config["prompts"]["system_template"]
    
    def set_system_prompt_template(self, template: str) -> None:
        """Set the system prompt template."""
        self.config["prompts"]["system_template"] = template
    
    def create_project_directories(self, base_dir: str) -> Dict[str, str]:
        """
        Create project directories based on configuration.
        
        Args:
            base_dir: Base directory for the project
            
        Returns:
            Dictionary of created directory paths
        """
        paths = {}
        for path_name, relative_path in self.get_paths().items():
            full_path = os.path.join(base_dir, relative_path)
            os.makedirs(full_path, exist_ok=True)
            paths[path_name] = full_path
        
        return paths


# Create a default configuration instance
config = Configuration()
