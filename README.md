# HealthResearchRAG

A flexible Retrieval-Augmented Generation (RAG) system designed specifically for health research documents. This package provides a complete solution for processing scientific papers, creating searchable indices, and generating accurate answers to domain-specific questions using local LLMs via Ollama.

## Features

- **Document Processing**: Convert PDF research papers to text and markdown formats with structure preservation
- **Flexible Indexing**: Create vector embeddings of document chunks using various embedding models
- **Multiple LLM Support**: Use any model available in Ollama with customizable prompts
- **Command-line Interface**: Easy-to-use scripts for processing, indexing, and querying
- **Health Research Focus**: Specialized for scientific and medical domain knowledge
- **Configurable**: Easily adapt to different health research domains with custom configuration

## Installation

```bash
# Basic installation
pip install healthrag

# Full installation with all features
pip install healthrag[full]
```

### Prerequisites

- Python 3.8 or later
- [Ollama](https://ollama.com/) installed and accessible in your PATH

## Quick Start

1. **Create a project structure**:

   ```bash
   mkdir myproject
   cd myproject
   ```

2. **Create a configuration file**:

   Create a file named `config.yaml` with your project configuration:

   ```yaml
   project:
     title: "Diabetes Research RAG System"
     description: "A RAG system for diabetes research papers"
   
   paths:
     documents_dir: "Documents"
     processed_dir: "Processed"
     markdown_dir: "Markdown"
     index_dir: "Index"
   
   models:
     llm:
       provider: "ollama"
       name: "deepseek-r1:8b"
       parameters:
         temperature: 0.1
     embedding:
       name: "BAAI/bge-small-en-v1.5"
   
   prompts:
     system_template: "You are an expert in diabetes research, analyzing scientific documents to provide accurate information.\n\nContext:\n{context}\n\nBased on this information, answer the following question:\nQuestion: {question}\n\nIf the context doesn't contain sufficient information, say so. Cite your sources."
   ```

3. **Process your documents**:

   Place your PDF files in the `Documents` folder and run:

   ```bash
   healthrag-process --config config.yaml
   ```

4. **Build the search index**:

   ```bash
   healthrag-index --config config.yaml
   ```

5. **Query the system**:

   ```bash
   healthrag-query --config config.yaml
   ```

## Python API

You can also use HealthResearchRAG as a Python library in your own code:

```python
from healthrag import Configuration, DocumentProcessor, DocumentIndexer, RAGEngine

# Load configuration
config = Configuration("config.yaml")

# Process documents
processor = DocumentProcessor(config)
processor.process_documents("Documents", "Processed", "Markdown")

# Build index
indexer = DocumentIndexer(config)
indexer.create_index(
    {"text_dir": "Processed", "markdown_dir": "Markdown"},
    "Index"
)

# Load index and initialize engine
vector_store = indexer.load_index("Index")
engine = RAGEngine(config, vector_store)

# Query
result = engine.generate_answer("What are the latest treatments for type 2 diabetes?")
print(result["answer"])
print("Sources:", result["sources"])
```

## Customization

### Using Different Models

You can use any model available in Ollama by changing the configuration:

```yaml
models:
  llm:
    provider: "ollama"
    name: "llama3"  # or "mistral", "deepseek-r1:8b", etc.
```

### Custom Prompts

Customize the system prompt to your specific health domain:

```yaml
prompts:
  system_template: "You are an expert in oncology research, analyzing scientific documents...\n\nContext:\n{context}\n\nQuestion: {question}"
```

### Embedding Models

Change the embedding model to alter retrieval characteristics:

```yaml
models:
  embedding:
    name: "sentence-transformers/all-mpnet-base-v2"
```

## Command-line Tools

### Document Processing

```bash
healthrag-process --config config.yaml --pdf-dir Documents --output-dir Processed --markdown-dir Markdown
```

### Indexing

```bash
healthrag-index --config config.yaml --text-dir Processed --markdown-dir Markdown --index-dir Index
```

### Querying

```bash
# Interactive mode
healthrag-query --config config.yaml --model deepseek-r1:8b

# Single question mode
healthrag-query --config config.yaml --question "What are the risk factors for heart disease?"
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
