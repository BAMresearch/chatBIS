"""
Unit tests for the EmbeddingGenerator class.
"""

import numpy as np
from unittest.mock import patch, MagicMock

from src.chatBIS.processor.processor import EmbeddingGenerator


class TestEmbeddingGenerator:
    """Tests for the EmbeddingGenerator class."""

    @patch("src.openbis_chatbot.processor.processor.OLLAMA_AVAILABLE", True)
    @patch("src.openbis_chatbot.processor.processor.OllamaEmbeddings")
    def test_init_with_ollama(self, mock_ollama_embeddings):
        """Test initialization with Ollama available."""
        # Mock the OllamaEmbeddings class
        mock_embeddings = MagicMock()
        mock_ollama_embeddings.return_value = mock_embeddings
        
        # Create an EmbeddingGenerator
        generator = EmbeddingGenerator(api_key="test_key")
        
        # Check that OllamaEmbeddings was initialized correctly
        mock_ollama_embeddings.assert_called_once_with(model="nomic-embed-text")
        
        # Check that the embeddings_model attribute was set correctly
        assert generator.embeddings_model == mock_embeddings
        
        # Check that the api_key attribute was set correctly
        assert generator.api_key == "test_key"

    @patch("src.openbis_chatbot.processor.processor.OLLAMA_AVAILABLE", False)
    def test_init_without_ollama(self):
        """Test initialization without Ollama available."""
        # Create an EmbeddingGenerator
        generator = EmbeddingGenerator(api_key="test_key")
        
        # Check that the embeddings_model attribute is None
        assert generator.embeddings_model is None
        
        # Check that the api_key attribute was set correctly
        assert generator.api_key == "test_key"

    @patch("src.openbis_chatbot.processor.processor.OLLAMA_AVAILABLE", True)
    def test_generate_embeddings_with_ollama(self):
        """Test generating embeddings with Ollama."""
        # Create an EmbeddingGenerator
        generator = EmbeddingGenerator()
        
        # Mock the embeddings_model
        mock_embeddings_model = MagicMock()
        mock_embeddings_model.embed_query.side_effect = lambda text: [0.1] * 768
        generator.embeddings_model = mock_embeddings_model
        
        # Generate embeddings
        texts = ["Text 1", "Text 2", "Text 3"]
        embeddings = generator.generate_embeddings(texts)
        
        # Check that the embeddings_model was called for each text
        assert mock_embeddings_model.embed_query.call_count == 3
        
        # Check that the correct number of embeddings was returned
        assert len(embeddings) == 3
        
        # Check that each embedding has the correct shape
        for embedding in embeddings:
            assert len(embedding) == 768

    @patch("src.openbis_chatbot.processor.processor.OLLAMA_AVAILABLE", True)
    def test_generate_embeddings_with_ollama_error(self):
        """Test generating embeddings with Ollama when an error occurs."""
        # Create an EmbeddingGenerator
        generator = EmbeddingGenerator()
        
        # Mock the embeddings_model to raise an exception
        mock_embeddings_model = MagicMock()
        mock_embeddings_model.embed_query.side_effect = Exception("Test error")
        generator.embeddings_model = mock_embeddings_model
        
        # Mock the _generate_dummy_embedding method
        generator._generate_dummy_embedding = MagicMock(return_value=[0.2] * 1536)
        
        # Generate embeddings
        texts = ["Text 1", "Text 2", "Text 3"]
        embeddings = generator.generate_embeddings(texts)
        
        # Check that the embeddings_model was called
        assert mock_embeddings_model.embed_query.call_count == 1
        
        # Check that _generate_dummy_embedding was called for each text
        assert generator._generate_dummy_embedding.call_count == 3
        
        # Check that the correct number of embeddings was returned
        assert len(embeddings) == 3
        
        # Check that each embedding has the correct shape
        for embedding in embeddings:
            assert len(embedding) == 1536

    @patch("src.openbis_chatbot.processor.processor.OLLAMA_AVAILABLE", False)
    def test_generate_embeddings_without_ollama(self):
        """Test generating embeddings without Ollama."""
        # Create an EmbeddingGenerator
        generator = EmbeddingGenerator()
        
        # Mock the _generate_dummy_embedding method
        generator._generate_dummy_embedding = MagicMock(return_value=[0.2] * 1536)
        
        # Generate embeddings
        texts = ["Text 1", "Text 2", "Text 3"]
        embeddings = generator.generate_embeddings(texts)
        
        # Check that _generate_dummy_embedding was called for each text
        assert generator._generate_dummy_embedding.call_count == 3
        
        # Check that the correct number of embeddings was returned
        assert len(embeddings) == 3
        
        # Check that each embedding has the correct shape
        for embedding in embeddings:
            assert len(embedding) == 1536

    def test_generate_embeddings_empty_texts(self):
        """Test generating embeddings with an empty list of texts."""
        # Create an EmbeddingGenerator
        generator = EmbeddingGenerator()
        
        # Generate embeddings
        embeddings = generator.generate_embeddings([])
        
        # Check that an empty list was returned
        assert embeddings == []

    def test_generate_dummy_embedding(self):
        """Test generating a dummy embedding."""
        # Create an EmbeddingGenerator
        generator = EmbeddingGenerator()
        
        # Generate a dummy embedding
        embedding = generator._generate_dummy_embedding(dim=100)
        
        # Check that the embedding has the correct shape
        assert len(embedding) == 100
        
        # Check that the embedding is normalized (unit length)
        np_embedding = np.array(embedding)
        assert np.isclose(np.linalg.norm(np_embedding), 1.0)
