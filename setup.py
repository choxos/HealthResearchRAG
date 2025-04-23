from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="healthrag",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A RAG system for health research documents",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/healthrag",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "langchain>=0.1.0",
        "langchain-community>=0.0.1",
        "langchain-huggingface>=0.0.1",
        "langchain-ollama>=0.0.1", 
        "langchain-text-splitters>=0.0.1",
        "pymupdf>=1.22.0",
        "faiss-cpu>=1.7.0",
        "transformers>=4.0.0",
        "sentence-transformers>=2.0.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "full": [
            "pymupdf4llm>=0.0.1",
            "sentencepiece>=0.1.99",
        ],
        "web": [
            "flask>=2.0.0",
            "flask-cors>=3.0.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "healthrag-process=scripts.process_documents:main",
            "healthrag-index=scripts.build_index:main",
            "healthrag-query=scripts.query_rag:main",
        ],
    },
)
