"""
RAG engine module for HealthResearchRAG.
Provides the core retrieval and answer generation functionality.
"""

import os
import time
from typing import Dict, List, Any, Optional, Tuple, Union
import logging

from .config import Configuration

# Import optional dependencies with fallbacks
try:
    from langchain.prompts import PromptTemplate
    from langchain.chains import RetrievalQA
    LANGCHAIN_CORE_AVAILABLE = True
except ImportError:
    LANGCHAIN_CORE_AVAILABLE = False

try:
    from langchain_ollama import OllamaLLM
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False


class RAGEngine:
    """
    Core RAG engine that combines retrieval with language model generation.
    """
    
    def __init__(self, config: Configuration, vector_store: Any = None, logger: Optional[logging.Logger] = None):
        """
        Initialize the RAG engine.
        
        Args:
            config: Configuration object
            vector_store: Pre-loaded vector store (optional)
            logger: Logger instance (optional)
        """
        self.config = config
        self.vector_store = vector_store
        self.logger = logger or logging.getLogger(__name__)
        self.qa_chain = None
        
        # Check dependencies
        self.dependencies_available = self._check_dependencies()
        
        # Initialize the LLM if vector store is provided
        if vector_store is not None:
            self.initialize_llm()
    
    def _check_dependencies(self) -> bool:
        """
        Check if all required dependencies are available.
        
        Returns:
            Whether all dependencies are available
        """
        core_available = LANGCHAIN_CORE_AVAILABLE
        ollama_available = OLLAMA_AVAILABLE
        
        if not core_available:
            self.logger.warning("LangChain core dependencies not available. Install with: pip install langchain")
        
        if not ollama_available:
            self.logger.warning("LangChain Ollama not available. Install with: pip install langchain-ollama")
            
        return core_available and ollama_available
    
    def initialize_llm(self) -> None:
        """
        Initialize the language model and QA chain.
        """
        if not self.dependencies_available:
            raise ImportError("Required dependencies not available")
            
        if self.vector_store is None:
            raise ValueError("Vector store must be provided before initializing LLM")
            
        # Get LLM configuration
        llm_config = self.config.get_llm_config()
        provider = llm_config.get("provider", "ollama")
        model_name = llm_config.get("name", "llama3")
        parameters = llm_config.get("parameters", {})
        
        # Initialize the LLM based on provider
        if provider == "ollama":
            llm = OllamaLLM(model=model_name, **parameters)
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
            
        # Create prompt template
        system_template = self.config.get_system_prompt_template()
        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=system_template
        )
        
        # Get retrieval settings
        results_count = self.config.config["retrieval"].get("results_count", 5)
        
        # Create the QA chain
        self.logger.info(f"Creating QA chain with {model_name}")
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(search_kwargs={"k": results_count}),
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )
        
        self.logger.info("RAG engine initialized successfully")
    
    def set_vector_store(self, vector_store: Any) -> None:
        """
        Set the vector store for retrieval.
        
        Args:
            vector_store: Vector store object
        """
        self.vector_store = vector_store
        self.initialize_llm()
    
    def generate_answer(self, question: str) -> Dict[str, Any]:
        """
        Generate an answer for a given question.
        
        Args:
            question: User question
            
        Returns:
            Dictionary containing answer and source information
        """
        if self.qa_chain is None:
            raise ValueError("QA chain not initialized. Call initialize_llm() first.")
            
        start_time = time.time()
        self.logger.info(f"Processing question: {question}")
        
        # Generate answer
        try:
            result = self.qa_chain.invoke({"query": question})
            
            # Extract the answer and source documents
            answer = result.get("result", "No answer generated")
            source_docs = result.get("source_documents", [])
            
            # Extract unique sources for citation
            sources = set()
            for doc in source_docs:
                if hasattr(doc, "metadata") and "source" in doc.metadata:
                    sources.add(doc.metadata["source"])
            
            elapsed_time = time.time() - start_time
            self.logger.info(f"Answer generated in {elapsed_time:.2f} seconds")
            
            return {
                "answer": answer,
                "sources": list(sources),
                "processing_time": elapsed_time,
                "raw_result": result
            }
            
        except Exception as e:
            self.logger.error(f"Error generating answer: {str(e)}")
            return {
                "error": str(e),
                "answer": "An error occurred while generating the answer.",
                "sources": []
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the RAG engine.
        
        Returns:
            Dictionary with stats
        """
        stats = {
            "model": "Not initialized",
            "retriever_k": 0,
            "vector_store_ready": self.vector_store is not None,
            "qa_chain_ready": self.qa_chain is not None
        }
        
        if self.qa_chain is not None:
            llm_config = self.config.get_llm_config()
            stats["model"] = f"{llm_config.get('provider')}/{llm_config.get('name')}"
            stats["retriever_k"] = self.config.config["retrieval"].get("results_count", 5)
            
        return stats
