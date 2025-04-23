"""
Flask web application for HealthResearchRAG.
Provides a web interface for document processing, indexing, and querying.
"""

import os
import sys
import json
import time
import tempfile
import logging
from pathlib import Path
from flask import Flask, request, jsonify, render_template, send_from_directory, redirect, url_for
from werkzeug.utils import secure_filename

# Add parent directory to path to import healthrag package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from healthrag import (
    Configuration, 
    DocumentProcessor, 
    DocumentIndexer,
    RAGEngine,
    setup_logger,
    check_dependencies,
    check_ollama_model,
    pull_ollama_model,
    create_example_config
)

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB limit

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Set up logger
logger = setup_logger('healthrag.webapp', logging.INFO)

# Global variables
config = None
engine = None
project_info = {
    'status': 'not_initialized',
    'project_path': None,
    'config_path': None,
    'last_activity': None
}

# Allowed file extensions
ALLOWED_EXTENSIONS = {'pdf'}

def allowed_file(filename):
    """Check if file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def initialize_project(project_path, config_path=None):
    """Initialize a project with the given path and config."""
    global config, project_info
    
    project_info['project_path'] = project_path
    project_info['config_path'] = config_path
    
    # Set up project directories
    try:
        os.makedirs(project_path, exist_ok=True)
        
        # Load or create configuration
        if config_path and os.path.exists(config_path):
            config = Configuration(config_path)
        else:
            config = Configuration()
            
            # If no config exists, create a default one in the project folder
            if not config_path:
                config_path = os.path.join(project_path, 'config.yaml')
                
            config.save_to_file(config_path)
            project_info['config_path'] = config_path
        
        # Create project directories
        config.create_project_directories(project_path)
        
        # Update status
        project_info['status'] = 'initialized'
        project_info['last_activity'] = time.strftime('%Y-%m-%d %H:%M:%S')
        
        return True
    except Exception as e:
        logger.error(f"Error initializing project: {str(e)}")
        project_info['status'] = 'error'
        return False

def initialize_engine():
    """Initialize the RAG engine if possible."""
    global config, engine, project_info
    
    if not config or project_info['status'] != 'initialized':
        return False
    
    # Check if index exists
    index_path = os.path.join(project_info['project_path'], config.get_path('index_dir'))
    if not os.path.exists(index_path) or not os.listdir(index_path):
        logger.warning(f"Index not found at {index_path}")
        return False
    
    # Check Ollama and model
    llm_config = config.get_llm_config()
    model_name = llm_config['name']
    if not check_ollama_model(model_name):
        logger.warning(f"Model {model_name} not available in Ollama")
        return False
    
    try:
        # Load the index
        indexer = DocumentIndexer(config)
        vector_store = indexer.load_index(index_path)
        
        # Initialize the engine
        engine = RAGEngine(config, vector_store, logger)
        
        project_info['status'] = 'engine_ready'
        project_info['last_activity'] = time.strftime('%Y-%m-%d %H:%M:%S')
        return True
    except Exception as e:
        logger.error(f"Error initializing engine: {str(e)}")
        return False

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html', project_info=project_info)

@app.route('/setup', methods=['GET', 'POST'])
def setup():
    """Handle project setup."""
    global project_info
    
    if request.method == 'POST':
        project_title = request.form.get('project_title', 'Health Research RAG')
        project_path = request.form.get('project_path')
        config_path = request.form.get('config_path')
        
        # Use default paths if not provided
        if not project_path:
            project_path = os.path.join(os.getcwd(), 'projects', secure_filename(project_title))
        
        # Initialize the project
        if initialize_project(project_path, config_path):
            # Update project title
            config.set_project_title(project_title)
            config.save_to_file(project_info['config_path'])
            
            return redirect(url_for('index'))
        else:
            return render_template('setup.html', error="Failed to initialize project")
    
    return render_template('setup.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    """Handle document uploads."""
    if request.method == 'POST':
        if 'files' not in request.files:
            return jsonify({'error': 'No files part in the request'}), 400
        
        files = request.files.getlist('files')
        if not files or all(not file.filename for file in files):
            return jsonify({'error': 'No files selected'}), 400
        
        # Save uploaded files
        saved_files = []
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join(project_info['project_path'], 
                                         config.get_path('documents_dir'), 
                                         filename)
                file.save(file_path)
                saved_files.append(filename)
        
        return jsonify({
            'success': True,
            'message': f'Successfully uploaded {len(saved_files)} files',
            'files': saved_files
        })
    
    return render_template('upload.html', project_info=project_info)

@app.route('/process', methods=['GET', 'POST'])
def process():
    """Process the uploaded documents."""
    if request.method == 'POST':
        if project_info['status'] != 'initialized':
            return jsonify({'error': 'Project not initialized'}), 400
        
        try:
            # Set up paths
            pdf_dir = os.path.join(project_info['project_path'], config.get_path('documents_dir'))
            text_dir = os.path.join(project_info['project_path'], config.get_path('processed_dir'))
            markdown_dir = os.path.join(project_info['project_path'], config.get_path('markdown_dir'))
            
            # Process documents
            processor = DocumentProcessor(config)
            stats = processor.process_documents(pdf_dir, text_dir, markdown_dir)
            
            return jsonify({
                'success': True,
                'message': 'Documents processed successfully',
                'stats': stats
            })
        except Exception as e:
            logger.error(f"Error processing documents: {str(e)}")
            return jsonify({'error': str(e)}), 500
    
    return render_template('process.html', project_info=project_info)

@app.route('/index-documents', methods=['GET', 'POST'])
def index_documents():
    """Build the search index."""
    if request.method == 'POST':
        if project_info['status'] != 'initialized':
            return jsonify({'error': 'Project not initialized'}), 400
        
        try:
            # Set up paths
            text_dir = os.path.join(project_info['project_path'], config.get_path('processed_dir'))
            markdown_dir = os.path.join(project_info['project_path'], config.get_path('markdown_dir'))
            pdf_dir = os.path.join(project_info['project_path'], config.get_path('documents_dir'))
            index_dir = os.path.join(project_info['project_path'], config.get_path('index_dir'))
            
            # Create directory mapping
            document_dirs = {
                'text_dir': text_dir,
                'markdown_dir': markdown_dir,
                'pdf_dir': pdf_dir
            }
            
            # Build the index
            indexer = DocumentIndexer(config)
            stats = indexer.create_index(document_dirs, index_dir)
            
            # Try to initialize the engine
            initialize_engine()
            
            return jsonify({
                'success': True,
                'message': 'Index built successfully',
                'stats': stats
            })
        except Exception as e:
            logger.error(f"Error building index: {str(e)}")
            return jsonify({'error': str(e)}), 500
    
    return render_template('index_documents.html', project_info=project_info)

@app.route('/query', methods=['GET', 'POST'])
def query():
    """Query the RAG system."""
    if request.method == 'POST':
        if project_info['status'] != 'engine_ready':
            # Try to initialize engine if not ready
            if not initialize_engine():
                return jsonify({'error': 'Engine not ready. Please build the index first.'}), 400
        
        question = request.form.get('question')
        if not question:
            return jsonify({'error': 'No question provided'}), 400
        
        try:
            result = engine.generate_answer(question)
            return jsonify({
                'success': True,
                'answer': result['answer'],
                'sources': result.get('sources', []),
                'processing_time': result.get('processing_time', 0)
            })
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            return jsonify({'error': str(e)}), 500
    
    return render_template('query.html', project_info=project_info)

@app.route('/config', methods=['GET', 'POST'])
def config_page():
    """View and edit the configuration."""
    global config
    
    if request.method == 'POST':
        if project_info['status'] not in ['initialized', 'engine_ready']:
            return jsonify({'error': 'Project not initialized'}), 400
        
        try:
            # Get form data
            project_title = request.form.get('project_title')
            model_name = request.form.get('model_name')
            embedding_model = request.form.get('embedding_model')
            system_prompt = request.form.get('system_prompt')
            
            # Update configuration
            if project_title:
                config.set_project_title(project_title)
            
            if model_name:
                llm_config = config.get_llm_config()
                config.set_llm_model(llm_config['provider'], model_name)
            
            if embedding_model:
                config.set_embedding_model(embedding_model)
            
            if system_prompt:
                config.set_system_prompt_template(system_prompt)
            
            # Save configuration
            config.save_to_file(project_info['config_path'])
            
            # Reset engine if it was initialized
            if project_info['status'] == 'engine_ready':
                project_info['status'] = 'initialized'
                engine = None
            
            return jsonify({
                'success': True,
                'message': 'Configuration updated successfully'
            })
        except Exception as e:
            logger.error(f"Error updating configuration: {str(e)}")
            return jsonify({'error': str(e)}), 500
    
    return render_template('config.html', 
                          project_info=project_info,
                          config=config.config if config else None)

@app.route('/static/<path:path>')
def send_static(path):
    """Serve static files."""
    return send_from_directory('static', path)

if __name__ == '__main__':
    # Check dependencies
    deps = check_dependencies()
    missing = [dep for dep, available in deps.items() if not available and dep in [
        'langchain', 'langchain_ollama', 'langchain_huggingface', 'faiss'
    ]]
    
    if missing:
        logger.error(f"Missing required dependencies: {', '.join(missing)}")
        logger.error("Please install them with: pip install " + " ".join(missing))
        sys.exit(1)
    
    # Check if Ollama is available
    if not deps.get('ollama', False):
        logger.error("Ollama is not installed or not in PATH. Please install Ollama.")
        sys.exit(1)
    
    # Start the app
    app.run(debug=True, host='0.0.0.0', port=5000)
