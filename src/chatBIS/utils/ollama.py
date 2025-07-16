"""
Ollama utilities for the openBIS Chatbot.
"""

import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


def check_ollama_availability() -> Tuple[bool, Optional[str]]:
    """
    Check if Ollama is available and running.
    
    Returns:
        A tuple containing a boolean indicating if Ollama is available and a message
    """
    try:
        from langchain_ollama import OllamaEmbeddings
        
        # Check if Ollama server is running
        try:
            embeddings = OllamaEmbeddings(model="nomic-embed-text")
            # Test with a simple embedding
            test_embedding = embeddings.embed_query("test")
            if test_embedding:
                return True, "Ollama server is running and embeddings are working."
            else:
                return False, "Ollama embeddings returned empty result."
        except Exception as e:
            return False, f"Error connecting to Ollama server: {e}"
    except ImportError:
        return False, "Langchain Ollama package not available."
