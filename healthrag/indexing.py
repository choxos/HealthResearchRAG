"""
Indexing module for HealthResearchRAG.
Handles document splitting, embedding and vector storage.
"""

import os
import shutil
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import json
import pickle

# Import optional dependencies with fallbacks
try:
    from langchain_text_splitters import (
        RecursiveCharacterTextSplitter,
        MarkdownTextSplitter
    )
    SPLITTERS_AVAILABLE = True
except ImportError:
    SPLITTERS_AVAILABLE = False

try:
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    VECTORSTORE_AVAILABLE = True
except ImportError:
    VECTORSTORE_AVAILABLE = False

try:
    from langchain_community.document_loaders import PyMuPDFLoader
    DIRECT_PDF_AVAILABLE = True
except ImportError:
    DIRECT_PDF_AVAILABLE = False


class DocumentIndexer:
    """Handles document indexing for RAG system."""
    
    def __init__(self, config: Any):
        """
        Initialize document indexer with configuration.
        
        Args:
            config: Configuration object containing indexing settings
        """
        self.config = config
        
        # Check available modules
        self.can_use_splitters = SPLITTERS_AVAILABLE
        self.can_use_vectorstore = VECTORSTORE_AVAILABLE
        self.can_use_direct_pdf = DIRECT_PDF_AVAILABLE
        
        # Load defaults from config
        chunk_settings = config.get_chunk_settings()
        self.chunk_size = chunk_settings.get("chunk_size", 1000)
        self.chunk_overlap = chunk_settings.get("chunk_overlap", 200)
        self.embedding_model = config.get_embedding_model()
    
    def check_dependencies(self) -> Dict[str, bool]:
        """
        Check which indexing capabilities are available.
        
        Returns:
            Dictionary of available capabilities
        """
        return {
            "text_splitters": self.can_use_splitters,
            "vector_store": self.can_use_vectorstore,
            "direct_pdf": self.can_use_direct_pdf
        }
    
    def create_index(self, 
                    document_dirs: Dict[str, str], 
                    index_dir: str,
                    embedding_model: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a search index from processed documents.
        
        Args:
            document_dirs: Dictionary of document directories 
                           (text_dir, markdown_dir, pdf_dir)
            index_dir: Directory to save the index
            embedding_model: Embedding model to use (optional)
            
        Returns:
            Dictionary with indexing statistics
        """
        if not self.can_use_splitters or not self.can_use_vectorstore:
            raise ImportError("Required dependencies not available: langchain-text-splitters or vector store")
        
        stats = {
            "total_chunks": 0,
            "indexed_docs": 0,
            "method_used": "none"
        }
        
        # Create index directory
        os.makedirs(index_dir, exist_ok=True)
        
        # Determine which content to use (markdown > text > pdf)
        if "markdown_dir" in document_dirs and os.path.exists(document_dirs["markdown_dir"]):
            md_files = [f for f in os.listdir(document_dirs["markdown_dir"]) if f.endswith('.md')]
            if md_files:
                chunks = self._process_markdown_files(document_dirs["markdown_dir"])
                stats["method_used"] = "markdown"
                stats["indexed_docs"] = len(md_files)
        
        elif "text_dir" in document_dirs and os.path.exists(document_dirs["text_dir"]):
            txt_files = [f for f in os.listdir(document_dirs["text_dir"]) if f.endswith('.txt')]
            if txt_files:
                chunks = self._process_text_files(document_dirs["text_dir"])
                stats["method_used"] = "text"
                stats["indexed_docs"] = len(txt_files)
        
        elif "pdf_dir" in document_dirs and os.path.exists(document_dirs["pdf_dir"]) and self.can_use_direct_pdf:
            pdf_files = [f for f in os.listdir(document_dirs["pdf_dir"]) if f.lower().endswith('.pdf')]
            if pdf_files:
                chunks = self._process_pdf_files(document_dirs["pdf_dir"])
                stats["method_used"] = "pdf"
                stats["indexed_docs"] = len(pdf_files)
        
        else:
            raise ValueError("No valid document source found")
        
        # Create embeddings
        if embedding_model is None:
            embedding_model = self.embedding_model
            
        embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        
        # Extract texts and metadata for indexing
        texts = [doc["content"] for doc in chunks]
        metadatas = [{"source": doc["source"]} for doc in chunks]
        
        # Create and save the vector store
        vector_store = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
        vector_store.save_local(index_dir)
        
        # Save index metadata
        index_metadata = {
            "embedding_model": embedding_model,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "document_count": stats["indexed_docs"],
            "chunk_count": len(texts),
            "method": stats["method_used"]
        }
        
        with open(os.path.join(index_dir, "metadata.json"), "w") as f:
            json.dump(index_metadata, f, indent=2)
        
        stats["total_chunks"] = len(texts)
        return stats
    
    def _process_markdown_files(self, markdown_dir: str) -> List[Dict[str, str]]:
        """
        Process markdown files using a markdown-aware splitter.
        
        Args:
            markdown_dir: Directory containing markdown files
            
        Returns:
            List of document chunks with content and source
        """
        chunks = []
        splitter = MarkdownTextSplitter(
            chunk_size=self.chunk_size, 
            chunk_overlap=self.chunk_overlap
        )
        
        for file_name in os.listdir(markdown_dir):
            if file_name.endswith('.md'):
                file_path = os.path.join(markdown_dir, file_name)
                
                # Read markdown file
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                
                # Split the content
                file_chunks = splitter.split_text(content)
                
                # Add chunks with source metadata
                for chunk in file_chunks:
                    chunks.append({
                        "content": chunk,
                        "source": file_name
                    })
        
        return chunks
    
    def _process_text_files(self, text_dir: str) -> List[Dict[str, str]]:
        """
        Process text files using recursive character splitter.
        
        Args:
            text_dir: Directory containing text files
            
        Returns:
            List of document chunks with content and source
        """
        chunks = []
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, 
            chunk_overlap=self.chunk_overlap,
            length_function=len
        )
        
        for file_name in os.listdir(text_dir):
            if file_name.endswith('.txt'):
                file_path = os.path.join(text_dir, file_name)
                
                # Read text file
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                
                # Split the content
                file_chunks = splitter.split_text(content)
                
                # Add chunks with source metadata
                for chunk in file_chunks:
                    chunks.append({
                        "content": chunk,
                        "source": file_name
                    })
        
        return chunks
    
    def _process_pdf_files(self, pdf_dir: str) -> List[Dict[str, str]]:
        """
        Process PDF files directly using PyMuPDFLoader.
        
        Args:
            pdf_dir: Directory containing PDF files
            
        Returns:
            List of document chunks with content and source
        """
        if not self.can_use_direct_pdf:
            raise ImportError("PyMuPDFLoader is not available")
            
        all_docs = []
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, 
            chunk_overlap=self.chunk_overlap,
            length_function=len
        )
        
        for file_name in os.listdir(pdf_dir):
            if file_name.lower().endswith('.pdf'):
                file_path = os.path.join(pdf_dir, file_name)
                
                # Load the PDF
                loader = PyMuPDFLoader(file_path)
                documents = loader.load()
                
                # Add source to metadata and add to collection
                for doc in documents:
                    doc.metadata["source"] = file_name
                    
                all_docs.extend(documents)
        
        # Split all documents
        split_docs = splitter.split_documents(all_docs)
        
        # Convert to our standard format
        chunks = []
        for doc in split_docs:
            chunks.append({
                "content": doc.page_content,
                "source": doc.metadata.get("source", "unknown")
            })
        
        return chunks
        
    def load_index(self, 
                  index_dir: str, 
                  embedding_model: Optional[str] = None) -> Any:
        """
        Load an existing index from disk.
        
        Args:
            index_dir: Directory containing the index
            embedding_model: Embedding model to use (optional)
            
        Returns:
            Vector store object
        """
        if not self.can_use_vectorstore:
            raise ImportError("Vector store dependencies not available")
            
        # Try to load metadata
        metadata_path = os.path.join(index_dir, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
                if embedding_model is None and "embedding_model" in metadata:
                    embedding_model = metadata["embedding_model"]
        
        # Use default if not specified
        if embedding_model is None:
            embedding_model = self.embedding_model
            
        # Load embeddings
        embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        
        # Load the vector store
        vector_store = FAISS.load_local(
            index_dir, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        
        return vector_store
