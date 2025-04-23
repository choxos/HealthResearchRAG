"""
Document processing module for HealthResearchRAG.
Handles PDF conversion, text extraction, and document preparation.
"""

import os
import shutil
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
import traceback

# Import optional dependencies with fallbacks
try:
    from langchain_community.document_loaders import PyMuPDFLoader
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

try:
    import pymupdf4llm
    PYMUPDF4LLM_AVAILABLE = True
except ImportError:
    PYMUPDF4LLM_AVAILABLE = False

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False


class DocumentProcessor:
    """Handles processing of PDF documents into text and markdown formats."""
    
    def __init__(self, config: Any):
        """
        Initialize document processor with configuration.
        
        Args:
            config: Configuration object containing paths and settings
        """
        self.config = config
        self.supported_extensions = ['.pdf', '.PDF']
        
        # Check available modules and set capabilities
        self.can_use_langchain = LANGCHAIN_AVAILABLE
        self.can_use_markdown = PYMUPDF4LLM_AVAILABLE
        self.can_use_basic_pdf = PYMUPDF_AVAILABLE
    
    def check_dependencies(self) -> Dict[str, bool]:
        """
        Check which document processing capabilities are available.
        
        Returns:
            Dictionary of available capabilities
        """
        return {
            "langchain_pdf": self.can_use_langchain,
            "markdown_conversion": self.can_use_markdown,
            "basic_pdf": self.can_use_basic_pdf
        }
    
    def list_documents(self, 
                       doc_dir: str, 
                       extensions: Optional[List[str]] = None) -> List[str]:
        """
        List document files in the specified directory.
        
        Args:
            doc_dir: Directory to scan for documents
            extensions: List of file extensions to filter by
            
        Returns:
            List of document filenames
        """
        if extensions is None:
            extensions = self.supported_extensions
            
        return [
            f for f in os.listdir(doc_dir) 
            if any(f.endswith(ext) for ext in extensions)
        ]
    
    def process_documents(self, 
                         doc_dir: str, 
                         output_dir: str, 
                         markdown_dir: Optional[str] = None) -> Dict[str, int]:
        """
        Process all documents in a directory.
        
        Args:
            doc_dir: Directory containing documents
            output_dir: Directory to save processed text
            markdown_dir: Directory to save markdown (if available)
            
        Returns:
            Dictionary with processing statistics
        """
        stats = {
            "total": 0,
            "processed_text": 0,
            "processed_markdown": 0,
            "failed": 0
        }
        
        # Ensure output directories exist
        os.makedirs(output_dir, exist_ok=True)
        if markdown_dir:
            os.makedirs(markdown_dir, exist_ok=True)
        
        # Get list of documents
        documents = self.list_documents(doc_dir)
        stats["total"] = len(documents)
        
        # Process each document
        for doc_name in documents:
            doc_path = os.path.join(doc_dir, doc_name)
            base_name = os.path.splitext(doc_name)[0]
            
            try:
                # Process to text
                if self.can_use_langchain:
                    text_file = os.path.join(output_dir, f"{base_name}.txt")
                    self._convert_pdf_to_text_langchain(doc_path, text_file)
                    stats["processed_text"] += 1
                elif self.can_use_basic_pdf:
                    text_file = os.path.join(output_dir, f"{base_name}.txt")
                    self._convert_pdf_to_text_basic(doc_path, text_file)
                    stats["processed_text"] += 1
                
                # Process to markdown if available and requested
                if markdown_dir and self.can_use_markdown:
                    md_file = os.path.join(markdown_dir, f"{base_name}.md")
                    self._convert_pdf_to_markdown(doc_path, md_file)
                    stats["processed_markdown"] += 1
                    
            except Exception as e:
                print(f"Error processing {doc_name}: {e}")
                traceback.print_exc()
                stats["failed"] += 1
        
        return stats
    
    def _convert_pdf_to_text_langchain(self, pdf_path: str, output_path: str) -> None:
        """
        Convert PDF to text using LangChain's PyMuPDFLoader.
        
        Args:
            pdf_path: Path to the PDF file
            output_path: Path to save the text output
        """
        if not self.can_use_langchain:
            raise ImportError("LangChain PyMuPDFLoader is not available")
            
        loader = PyMuPDFLoader(pdf_path)
        documents = loader.load()
        
        combined_text = ""
        for doc in documents:
            combined_text += doc.page_content + "\n\n"
            # Could also preserve metadata if needed
        
        with open(output_path, "w", encoding="utf-8") as text_file:
            text_file.write(combined_text)
    
    def _convert_pdf_to_text_basic(self, pdf_path: str, output_path: str) -> None:
        """
        Convert PDF to text using basic PyMuPDF.
        
        Args:
            pdf_path: Path to the PDF file
            output_path: Path to save the text output
        """
        if not self.can_use_basic_pdf:
            raise ImportError("PyMuPDF is not available")
            
        with fitz.open(pdf_path) as doc:
            text = ""
            for page in doc:
                text += page.get_text() + "\n\n"
        
        with open(output_path, "w", encoding="utf-8") as text_file:
            text_file.write(text)
    
    def _convert_pdf_to_markdown(self, pdf_path: str, output_path: str) -> None:
        """
        Convert PDF to markdown using pymupdf4llm.
        
        Args:
            pdf_path: Path to the PDF file
            output_path: Path to save the markdown output
        """
        if not self.can_use_markdown:
            raise ImportError("pymupdf4llm is not available")
            
        md_text = pymupdf4llm.to_markdown(pdf_path)
        
        with open(output_path, "wb") as md_file:
            md_file.write(md_text.encode("utf-8"))
