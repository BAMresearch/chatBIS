#!/usr/bin/env python3
"""
Flask web application for the openBIS Chatbot.
"""

import os
import logging
from pathlib import Path
from flask import Flask, render_template, request, jsonify, session

from openbis_chatbot.query.query import RAGQueryEngine
from openbis_chatbot.query.conversation_engine import ConversationEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Default paths
DEFAULT_DATA_DIR = os.path.join(os.getcwd(), "data", "processed")

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'openbis-chatbot-secret-key-change-in-production')

# Initialize the engines
query_engine = None
conversation_engine = None


def initialize_engines(data_dir=DEFAULT_DATA_DIR, model="qwen3"):
    """Initialize both the query engine and conversation engine."""
    global query_engine, conversation_engine
    try:
        logger.info(f"Initializing engines with data from {data_dir}...")

        # Initialize RAG query engine (for backward compatibility)
        query_engine = RAGQueryEngine(
            data_dir=data_dir,
            api_key=None,
            model=model
        )

        # Initialize conversation engine with memory
        conversation_engine = ConversationEngine(
            data_dir=data_dir,
            model=model,
            memory_db_path=os.path.join(data_dir, "conversation_memory.db")
        )

        logger.info("Engines initialized successfully.")
        return True
    except Exception as e:
        logger.error(f"Error initializing engines: {e}")
        return False


@app.route('/')
def index():
    """Render the chat interface."""
    return render_template('index.html')


@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat requests with conversation memory."""
    global conversation_engine

    # Check if the conversation engine is initialized
    if conversation_engine is None:
        success = initialize_engines()
        if not success:
            return jsonify({
                'success': False,
                'error': 'Failed to initialize conversation engine. Please check the logs.'
            })

    # Get the query and session info from the request
    data = request.json
    query = data.get('query', '')
    session_id = data.get('session_id', None)

    if not query.strip():
        return jsonify({
            'success': False,
            'error': 'Query cannot be empty.'
        })

    try:
        # Process the conversation with memory
        response, session_id, metadata = conversation_engine.chat(query, session_id)

        # Return the response with session info and metadata
        return jsonify({
            'success': True,
            'answer': response,
            'session_id': session_id,
            'metadata': metadata
        })

    except Exception as e:
        logger.error(f"Error in conversation: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        })


@app.route('/api/chat/history/<session_id>', methods=['GET'])
def get_chat_history(session_id):
    """Get conversation history for a session."""
    global conversation_engine

    if conversation_engine is None:
        return jsonify({
            'success': False,
            'error': 'Conversation engine not initialized.'
        })

    try:
        history = conversation_engine.get_conversation_history(session_id)
        return jsonify({
            'success': True,
            'history': history
        })
    except Exception as e:
        logger.error(f"Error getting chat history: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        })


@app.route('/api/chat/clear/<session_id>', methods=['POST'])
def clear_chat_history(session_id):
    """Clear conversation history for a session."""
    global conversation_engine

    if conversation_engine is None:
        return jsonify({
            'success': False,
            'error': 'Conversation engine not initialized.'
        })

    try:
        success = conversation_engine.clear_session(session_id)
        return jsonify({
            'success': success,
            'message': 'Chat history cleared.' if success else 'Failed to clear chat history.'
        })
    except Exception as e:
        logger.error(f"Error clearing chat history: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        })


def run_app(host='0.0.0.0', port=5000, debug=False, data_dir=DEFAULT_DATA_DIR, model="qwen3"):
    """Run the Flask application."""
    # Initialize the engines
    initialize_engines(data_dir, model)

    # Run the Flask application
    app.run(host=host, port=port, debug=debug)


if __name__ == '__main__':
    run_app(debug=True)
