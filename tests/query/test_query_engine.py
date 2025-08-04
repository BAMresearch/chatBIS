"""
Unit tests for the RAGQueryEngine class.
"""

import pytest
import tempfile
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.chatBIS.query.query import RAGQueryEngine


class TestRAGQueryEngine:
    """Tests for the RAGQueryEngine class."""

    @patch("src.chatBIS.query.query.OLLAMA_AVAILABLE", True)
    @patch("src.chatBIS.query.query.OllamaEmbeddings")
    @patch("src.chatBIS.query.query.ChatOllama")
    def test_init_with_ollama(self, mock_chat_ollama, mock_ollama_embeddings):
        """Test initialization with Ollama available."""
        # Mock the OllamaEmbeddings and ChatOllama classes
        mock_embeddings = MagicMock()
        mock_ollama_embeddings.return_value = mock_embeddings

        mock_chat = MagicMock()
        mock_chat_ollama.return_value = mock_chat

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a chunks.json file
            chunks_file = Path(temp_dir) / "chunks.json"

            # Write test data to the chunks file
            with open(chunks_file, "w", encoding="utf-8") as f:
                f.write('[{"title": "Test", "content": "Test content", "embedding": [0.1, 0.2], "url": "https://example.com", "chunk_id": "test_0"}]')

            # Create a query engine
            engine = RAGQueryEngine(
                data_dir=temp_dir,
                api_key="test_key",
                model="test_model"
            )

            # We're not using mock_file anymore since we're writing to a real file
            # mock_file.assert_called_once_with(chunks_file, "r", encoding="utf-8")

            # Check that OllamaEmbeddings was initialized correctly
            mock_ollama_embeddings.assert_called_once_with(model="nomic-embed-text")

            # Check that ChatOllama was initialized correctly
            mock_chat_ollama.assert_called_once_with(model="test_model")

            # Check that the attributes were initialized correctly
            assert engine.data_dir == Path(temp_dir)
            assert engine.api_key == "test_key"
            assert engine.model == "test_model"
            assert engine.embeddings_model == mock_embeddings
            assert engine.llm == mock_chat
            assert len(engine.chunks) == 1
            assert engine.chunks[0]["title"] == "Test"
            assert engine.chunks[0]["content"] == "Test content"
            assert engine.chunks[0]["embedding"] == [0.1, 0.2]
            assert engine.chunks[0]["url"] == "https://example.com"
            assert engine.chunks[0]["chunk_id"] == "test_0"

    @patch("src.chatBIS.query.query.OLLAMA_AVAILABLE", False)
    def test_init_without_ollama(self):
        """Test initialization without Ollama available."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a chunks.json file
            chunks_file = Path(temp_dir) / "chunks.json"

            # Write test data to the chunks file
            with open(chunks_file, "w", encoding="utf-8") as f:
                f.write('[{"title": "Test", "content": "Test content", "embedding": [0.1, 0.2], "url": "https://example.com", "chunk_id": "test_0"}]')

            # Create a query engine
            engine = RAGQueryEngine(
                data_dir=temp_dir,
                api_key="test_key",
                model="test_model"
            )

            # We're not using mock_file anymore since we're writing to a real file
            # mock_file.assert_called_once_with(chunks_file, "r", encoding="utf-8")

            # Check that the attributes were initialized correctly
            assert engine.data_dir == Path(temp_dir)
            assert engine.api_key == "test_key"
            assert engine.model == "test_model"
            assert engine.embeddings_model is None
            assert engine.llm is None
            assert len(engine.chunks) == 1
            assert engine.chunks[0]["title"] == "Test"
            assert engine.chunks[0]["content"] == "Test content"
            assert engine.chunks[0]["embedding"] == [0.1, 0.2]
            assert engine.chunks[0]["url"] == "https://example.com"
            assert engine.chunks[0]["chunk_id"] == "test_0"

    def test_load_chunks(self):
        """Test loading chunks from a file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a chunks.json file
            chunks_file = Path(temp_dir) / "chunks.json"

            # Write test data to the chunks file
            with open(chunks_file, "w", encoding="utf-8") as f:
                f.write('[{"title": "Test", "content": "Test content", "embedding": [0.1, 0.2], "url": "https://example.com", "chunk_id": "test_0"}]')

            # Create a query engine
            engine = RAGQueryEngine(
                data_dir=temp_dir,
                api_key="test_key"
            )

            # We're not using mock_file anymore since we're writing to a real file
            # mock_file.assert_called_once_with(chunks_file, "r", encoding="utf-8")

            # Check that the chunks were loaded correctly
            assert len(engine.chunks) == 1
            assert engine.chunks[0]["title"] == "Test"
            assert engine.chunks[0]["content"] == "Test content"
            assert engine.chunks[0]["embedding"] == [0.1, 0.2]
            assert engine.chunks[0]["url"] == "https://example.com"
            assert engine.chunks[0]["chunk_id"] == "test_0"

    def test_load_chunks_file_not_found(self):
        """Test loading chunks when the file doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Try to create a query engine
            with pytest.raises(FileNotFoundError):
                RAGQueryEngine(
                    data_dir=temp_dir,
                    api_key="test_key"
                )

    @patch("src.chatBIS.query.query.OLLAMA_AVAILABLE", True)
    def test_generate_embedding_with_ollama(self):
        """Test generating an embedding with Ollama."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a chunks.json file
            chunks_file = Path(temp_dir) / "chunks.json"

            # Write test data to the chunks file
            with open(chunks_file, "w", encoding="utf-8") as f:
                f.write('[{"title": "Test", "content": "Test content", "embedding": [0.1, 0.2], "url": "https://example.com", "chunk_id": "test_0"}]')

            # Create a query engine
            engine = RAGQueryEngine(
                data_dir=temp_dir,
                api_key="test_key"
            )

        # Mock the embeddings_model
        mock_embeddings_model = MagicMock()
        mock_embeddings_model.embed_query.return_value = [0.1] * 768
        engine.embeddings_model = mock_embeddings_model

        # Mock the _load_chunks method to avoid loading chunks
        engine._load_chunks = MagicMock(return_value=[])
        engine.chunks = []

        # Generate an embedding
        embedding = engine.generate_embedding("Test query")

        # Check that the embeddings_model was called
        mock_embeddings_model.embed_query.assert_called_once_with("Test query")

        # Check that the embedding has the correct shape
        assert len(embedding) == 768
        assert embedding == [0.1] * 768

    @patch("src.chatBIS.query.query.OLLAMA_AVAILABLE", True)
    def test_generate_embedding_with_ollama_error(self):
        """Test generating an embedding with Ollama when an error occurs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a chunks.json file
            chunks_file = Path(temp_dir) / "chunks.json"

            # Write test data to the chunks file
            with open(chunks_file, "w", encoding="utf-8") as f:
                f.write('[{"title": "Test", "content": "Test content", "embedding": [0.1, 0.2], "url": "https://example.com", "chunk_id": "test_0"}]')

            # Create a query engine
            engine = RAGQueryEngine(
                data_dir=temp_dir,
                api_key="test_key"
            )

        # Mock the embeddings_model to raise an exception
        mock_embeddings_model = MagicMock()
        mock_embeddings_model.embed_query.side_effect = Exception("Test error")
        engine.embeddings_model = mock_embeddings_model

        # Mock the _generate_dummy_embedding method
        engine._generate_dummy_embedding = MagicMock(return_value=[0.2] * 1536)

        # Mock the _load_chunks method to avoid loading chunks
        engine._load_chunks = MagicMock(return_value=[])
        engine.chunks = []

        # Generate an embedding
        embedding = engine.generate_embedding("Test query")

        # Check that the embeddings_model was called
        mock_embeddings_model.embed_query.assert_called_once_with("Test query")

        # Check that _generate_dummy_embedding was called
        engine._generate_dummy_embedding.assert_called_once()

        # Check that the embedding has the correct shape
        assert len(embedding) == 1536
        assert embedding == [0.2] * 1536

    @patch("src.chatBIS.query.query.OLLAMA_AVAILABLE", False)
    def test_generate_embedding_without_ollama(self):
        """Test generating an embedding without Ollama."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a chunks.json file
            chunks_file = Path(temp_dir) / "chunks.json"

            # Write test data to the chunks file
            with open(chunks_file, "w", encoding="utf-8") as f:
                f.write('[{"title": "Test", "content": "Test content", "embedding": [0.1, 0.2], "url": "https://example.com", "chunk_id": "test_0"}]')

            # Create a query engine
            engine = RAGQueryEngine(
                data_dir=temp_dir,
                api_key="test_key"
            )

        # Mock the _generate_dummy_embedding method
        engine._generate_dummy_embedding = MagicMock(return_value=[0.2] * 1536)

        # Mock the _load_chunks method to avoid loading chunks
        engine._load_chunks = MagicMock(return_value=[])
        engine.chunks = []

        # Generate an embedding
        embedding = engine.generate_embedding("Test query")

        # Check that _generate_dummy_embedding was called
        engine._generate_dummy_embedding.assert_called_once()

        # Check that the embedding has the correct shape
        assert len(embedding) == 1536
        assert embedding == [0.2] * 1536

    def test_generate_dummy_embedding(self):
        """Test generating a dummy embedding."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a chunks.json file
            chunks_file = Path(temp_dir) / "chunks.json"

            # Write test data to the chunks file
            with open(chunks_file, "w", encoding="utf-8") as f:
                f.write('[{"title": "Test", "content": "Test content", "embedding": [0.1, 0.2], "url": "https://example.com", "chunk_id": "test_0"}]')

            # Create a query engine
            engine = RAGQueryEngine(
                data_dir=temp_dir,
                api_key="test_key"
            )

        # Mock the _load_chunks method to avoid loading chunks
        engine._load_chunks = MagicMock(return_value=[])
        engine.chunks = []

        # Generate a dummy embedding
        embedding = engine._generate_dummy_embedding(dim=100)

        # Check that the embedding has the correct shape
        assert len(embedding) == 100

        # Check that the embedding is normalized (unit length)
        np_embedding = np.array(embedding)
        assert np.isclose(np.linalg.norm(np_embedding), 1.0)

    def test_retrieve_relevant_chunks(self):
        """Test retrieving relevant chunks."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a chunks.json file
            chunks_file = Path(temp_dir) / "chunks.json"

            # Write test data to the chunks file
            with open(chunks_file, "w", encoding="utf-8") as f:
                f.write('[{"title": "Test", "content": "Test content", "embedding": [0.1, 0.2], "url": "https://example.com", "chunk_id": "test_0"}]')

            # Create a query engine
            engine = RAGQueryEngine(
                data_dir=temp_dir,
                api_key="test_key"
            )

        # Mock the _load_chunks method to avoid loading chunks
        engine._load_chunks = MagicMock(return_value=[])

        # Create test chunks with very distinct embeddings to ensure consistent test results
        # Make the second chunk have exactly the same embedding as our query [0.2] * 768
        # Make the other chunks have very different embeddings
        engine.chunks = [
            {"title": "Test 1", "content": "Test content 1", "embedding": [-0.5] * 768, "url": "https://example.com/1", "chunk_id": "test_1"},
            {"title": "Test 2", "content": "Test content 2", "embedding": [0.2] * 768, "url": "https://example.com/2", "chunk_id": "test_2"},
            {"title": "Test 3", "content": "Test content 3", "embedding": [-0.5] * 768, "url": "https://example.com/3", "chunk_id": "test_3"}
        ]

        # Mock the generate_embedding method
        engine.generate_embedding = MagicMock(return_value=[0.2] * 768)

        # Retrieve relevant chunks
        chunks = engine.retrieve_relevant_chunks("Test query", top_k=2)

        # Check that generate_embedding was called
        engine.generate_embedding.assert_called_once_with("Test query")

        # Check that the correct number of chunks was returned
        assert len(chunks) == 2

        # Check that the chunks are sorted by similarity (most similar first)
        # The chunk with embedding [0.2] * 768 should be most similar to the query embedding [0.2] * 768
        assert chunks[0]["chunk_id"] == "test_2"

    @patch("src.chatBIS.query.query.OLLAMA_AVAILABLE", True)
    def test_generate_answer_with_ollama(self):
        """Test generating an answer with Ollama."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a chunks.json file
            chunks_file = Path(temp_dir) / "chunks.json"

            # Write test data to the chunks file
            with open(chunks_file, "w", encoding="utf-8") as f:
                f.write('[{"title": "Test", "content": "Test content", "embedding": [0.1, 0.2], "url": "https://example.com", "chunk_id": "test_0"}]')

            # Create a query engine
            engine = RAGQueryEngine(
                data_dir=temp_dir,
                api_key="test_key"
            )

        # Mock the llm
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Test answer"
        mock_llm.invoke.return_value = mock_response
        engine.llm = mock_llm

        # Mock the _create_prompt method
        engine._create_prompt = MagicMock(return_value="Test prompt")

        # Mock the _load_chunks method to avoid loading chunks
        engine._load_chunks = MagicMock(return_value=[])
        engine.chunks = []

        # Generate an answer
        relevant_chunks = [
            {"title": "Test 1", "content": "Test content 1", "embedding": [0.1] * 768, "url": "https://example.com/1", "chunk_id": "test_1"},
            {"title": "Test 2", "content": "Test content 2", "embedding": [0.2] * 768, "url": "https://example.com/2", "chunk_id": "test_2"}
        ]
        answer = engine.generate_answer("Test query", relevant_chunks)

        # Check that _create_prompt was called
        engine._create_prompt.assert_called_once_with("Test query", relevant_chunks)

        # Check that the llm was called
        mock_llm.invoke.assert_called_once()
        args = mock_llm.invoke.call_args[0][0]
        assert "Test prompt" in args

        # Check that the answer is correct
        assert answer == "Test answer"

    @patch("src.chatBIS.query.query.OLLAMA_AVAILABLE", True)
    def test_generate_answer_with_ollama_error(self):
        """Test generating an answer with Ollama when an error occurs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a chunks.json file
            chunks_file = Path(temp_dir) / "chunks.json"

            # Write test data to the chunks file
            with open(chunks_file, "w", encoding="utf-8") as f:
                f.write('[{"title": "Test", "content": "Test content", "embedding": [0.1, 0.2], "url": "https://example.com", "chunk_id": "test_0"}]')

            # Create a query engine
            engine = RAGQueryEngine(
                data_dir=temp_dir,
                api_key="test_key"
            )

        # Mock the llm to raise an exception
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = Exception("Test error")
        engine.llm = mock_llm

        # Mock the _create_prompt method
        engine._create_prompt = MagicMock(return_value="Test prompt")

        # Mock the _load_chunks method to avoid loading chunks
        engine._load_chunks = MagicMock(return_value=[])
        engine.chunks = []

        # Generate an answer
        relevant_chunks = [
            {"title": "Test 1", "content": "Test content 1", "embedding": [0.1] * 768, "url": "https://example.com/1", "chunk_id": "test_1"},
            {"title": "Test 2", "content": "Test content 2", "embedding": [0.2] * 768, "url": "https://example.com/2", "chunk_id": "test_2"}
        ]
        answer = engine.generate_answer("Test query", relevant_chunks)

        # Check that _create_prompt was called
        engine._create_prompt.assert_called_once_with("Test query", relevant_chunks)

        # Check that the llm was called
        mock_llm.invoke.assert_called_once()

        # Check that the answer indicates an error
        assert "Error generating answer" in answer
        assert "Test error" in answer

    @patch("src.chatBIS.query.query.OLLAMA_AVAILABLE", False)
    def test_generate_answer_without_ollama(self):
        """Test generating an answer without Ollama."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a chunks.json file
            chunks_file = Path(temp_dir) / "chunks.json"

            # Write test data to the chunks file
            with open(chunks_file, "w", encoding="utf-8") as f:
                f.write('[{"title": "Test", "content": "Test content", "embedding": [0.1, 0.2], "url": "https://example.com", "chunk_id": "test_0"}]')

            # Create a query engine
            engine = RAGQueryEngine(
                data_dir=temp_dir,
                api_key="test_key"
            )

        # Mock the _load_chunks method to avoid loading chunks
        engine._load_chunks = MagicMock(return_value=[])
        engine.chunks = []
        engine.llm = None

        # Generate an answer
        relevant_chunks = [
            {"title": "Test 1", "content": "Test content 1", "embedding": [0.1] * 768, "url": "https://example.com/1", "chunk_id": "test_1"},
            {"title": "Test 2", "content": "Test content 2", "embedding": [0.2] * 768, "url": "https://example.com/2", "chunk_id": "test_2"}
        ]
        answer = engine.generate_answer("Test query", relevant_chunks)

        # Check that the answer indicates Ollama is not available
        assert "Ollama not available" in answer

    def test_create_prompt(self):
        """Test creating a prompt."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a chunks.json file
            chunks_file = Path(temp_dir) / "chunks.json"

            # Write test data to the chunks file
            with open(chunks_file, "w", encoding="utf-8") as f:
                f.write('[{"title": "Test", "content": "Test content", "embedding": [0.1, 0.2], "url": "https://example.com", "chunk_id": "test_0"}]')

            # Create a query engine
            engine = RAGQueryEngine(
                data_dir=temp_dir,
                api_key="test_key"
            )

        # Mock the _load_chunks method to avoid loading chunks
        engine._load_chunks = MagicMock(return_value=[])
        engine.chunks = []

        # Create a prompt
        relevant_chunks = [
            {"title": "Test 1", "content": "Test content 1", "embedding": [0.1] * 768, "url": "https://example.com/1", "chunk_id": "test_1"},
            {"title": "Test 2", "content": "Test content 2", "embedding": [0.2] * 768, "url": "https://example.com/2", "chunk_id": "test_2"}
        ]
        prompt = engine._create_prompt("Test query", relevant_chunks)

        # Check that the prompt contains the query
        assert "Question: Test query" in prompt

        # Check that the prompt contains the relevant chunks
        assert "Document 1:" in prompt
        assert "Title: Test 1" in prompt
        assert "Content: Test content 1" in prompt
        assert "Document 2:" in prompt
        assert "Title: Test 2" in prompt
        assert "Content: Test content 2" in prompt

        # Check that the prompt contains the instructions
        assert "answer the question in a conversational and helpful way" in prompt
        assert "Do not mention the sources or documents in your answer" in prompt

    def test_query(self):
        """Test the query method."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a chunks.json file
            chunks_file = Path(temp_dir) / "chunks.json"

            # Write test data to the chunks file
            with open(chunks_file, "w", encoding="utf-8") as f:
                f.write('[{"title": "Test", "content": "Test content", "embedding": [0.1, 0.2], "url": "https://example.com", "chunk_id": "test_0"}]')

            # Create a query engine
            engine = RAGQueryEngine(
                data_dir=temp_dir,
                api_key="test_key"
            )

        # Mock the retrieve_relevant_chunks method
        relevant_chunks = [
            {"title": "Test 1", "content": "Test content 1", "embedding": [0.1] * 768, "url": "https://example.com/1", "chunk_id": "test_1"},
            {"title": "Test 2", "content": "Test content 2", "embedding": [0.2] * 768, "url": "https://example.com/2", "chunk_id": "test_2"}
        ]
        engine.retrieve_relevant_chunks = MagicMock(return_value=relevant_chunks)

        # Mock the generate_answer method
        engine.generate_answer = MagicMock(return_value="Test answer")

        # Mock the _load_chunks method to avoid loading chunks
        engine._load_chunks = MagicMock(return_value=[])
        engine.chunks = []

        # Query the engine
        answer, chunks = engine.query("Test query", top_k=2)

        # Check that retrieve_relevant_chunks was called
        engine.retrieve_relevant_chunks.assert_called_once_with("Test query", top_k=2)

        # Check that generate_answer was called
        engine.generate_answer.assert_called_once_with("Test query", relevant_chunks)

        # Check that the answer and chunks are correct
        assert answer == "Test answer"
        assert chunks == relevant_chunks
