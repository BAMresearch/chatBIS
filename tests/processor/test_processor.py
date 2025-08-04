"""
Unit tests for the RAGProcessor class.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open

from src.chatBIS.processor.processor import RAGProcessor, ContentChunker, EmbeddingGenerator


class TestRAGProcessor:
    """Tests for the RAGProcessor class."""

    def test_init(self):
        """Test initialization of the processor."""
        with tempfile.TemporaryDirectory() as temp_dir:
            processor = RAGProcessor(
                input_dir="./input",
                output_dir=temp_dir,
                api_key="test_key",
                min_chunk_size=100,
                max_chunk_size=1000,
                chunk_overlap=50
            )
            
            # Check that the attributes were initialized correctly
            assert processor.input_dir == Path("./input")
            assert processor.output_dir == Path(temp_dir)
            
            # Check that the chunker was initialized correctly
            assert isinstance(processor.chunker, ContentChunker)
            assert processor.chunker.min_chunk_size == 100
            assert processor.chunker.max_chunk_size == 1000
            assert processor.chunker.chunk_overlap == 50
            
            # Check that the embedding generator was initialized correctly
            assert isinstance(processor.embedding_generator, EmbeddingGenerator)
            assert processor.embedding_generator.api_key == "test_key"
            
            # Check that the output directory was created
            assert os.path.exists(temp_dir)

    @patch("builtins.open", new_callable=mock_open, read_data="Title: Test Page\nURL: https://example.com/test\n---\n\nTest content.")
    def test_process_file(self, mock_file):
        """Test processing a single file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            processor = RAGProcessor(
                input_dir="./input",
                output_dir=temp_dir
            )
            
            # Mock the chunker
            mock_chunker = MagicMock()
            mock_chunker.chunk_content.return_value = ["Chunk 1", "Chunk 2"]
            processor.chunker = mock_chunker
            
            # Mock the embedding generator
            mock_embedding_generator = MagicMock()
            mock_embedding_generator.generate_embeddings.return_value = [[0.1] * 768, [0.2] * 768]
            processor.embedding_generator = mock_embedding_generator
            
            # Process a file
            file_path = Path("./input/test.txt")
            chunks = processor.process_file(file_path)
            
            # Check that the file was opened
            mock_file.assert_called_once_with(file_path, "r", encoding="utf-8")
            
            # Check that the chunker was called
            mock_chunker.chunk_content.assert_called_once_with("Test content.")
            
            # Check that the embedding generator was called
            mock_embedding_generator.generate_embeddings.assert_called_once_with(["Chunk 1", "Chunk 2"])
            
            # Check that the correct number of chunks was returned
            assert len(chunks) == 2
            
            # Check that each chunk has the correct attributes
            for i, chunk in enumerate(chunks):
                assert chunk["title"] == "Test Page"
                assert chunk["url"] == "https://example.com/test"
                assert chunk["content"] == f"Chunk {i+1}"
                assert chunk["embedding"] == [0.1] * 768 if i == 0 else [0.2] * 768
                assert chunk["chunk_id"] == f"test_{i}"

    @patch("pathlib.Path.glob")
    def test_process_all_files(self, mock_glob):
        """Test processing all files in a directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            processor = RAGProcessor(
                input_dir="./input",
                output_dir=temp_dir
            )
            
            # Mock the glob method to return a list of files
            mock_glob.return_value = [Path("./input/file1.txt"), Path("./input/file2.txt")]
            
            # Mock the process_file method
            processor.process_file = MagicMock(side_effect=[
                [{"title": "File 1", "content": "Chunk 1", "embedding": [0.1] * 768, "url": "https://example.com/file1", "chunk_id": "file1_0"}],
                [{"title": "File 2", "content": "Chunk 2", "embedding": [0.2] * 768, "url": "https://example.com/file2", "chunk_id": "file2_0"}]
            ])
            
            # Process all files
            chunks = processor.process_all_files()
            
            # Check that the glob method was called
            mock_glob.assert_called_once_with("*.txt")
            
            # Check that the process_file method was called for each file
            assert processor.process_file.call_count == 2
            
            # Check that the correct number of chunks was returned
            assert len(chunks) == 2
            
            # Check that each chunk has the correct attributes
            assert chunks[0]["title"] == "File 1"
            assert chunks[0]["content"] == "Chunk 1"
            assert chunks[0]["embedding"] == [0.1] * 768
            assert chunks[0]["url"] == "https://example.com/file1"
            assert chunks[0]["chunk_id"] == "file1_0"
            
            assert chunks[1]["title"] == "File 2"
            assert chunks[1]["content"] == "Chunk 2"
            assert chunks[1]["embedding"] == [0.2] * 768
            assert chunks[1]["url"] == "https://example.com/file2"
            assert chunks[1]["chunk_id"] == "file2_0"

    @patch("pathlib.Path.glob")
    def test_process_all_files_with_error(self, mock_glob):
        """Test processing all files when an error occurs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            processor = RAGProcessor(
                input_dir="./input",
                output_dir=temp_dir
            )
            
            # Mock the glob method to return a list of files
            mock_glob.return_value = [Path("./input/file1.txt"), Path("./input/file2.txt")]
            
            # Mock the process_file method to raise an exception for the second file
            processor.process_file = MagicMock(side_effect=[
                [{"title": "File 1", "content": "Chunk 1", "embedding": [0.1] * 768, "url": "https://example.com/file1", "chunk_id": "file1_0"}],
                Exception("Test error")
            ])
            
            # Process all files
            chunks = processor.process_all_files()
            
            # Check that the glob method was called
            mock_glob.assert_called_once_with("*.txt")
            
            # Check that the process_file method was called for each file
            assert processor.process_file.call_count == 2
            
            # Check that only the chunks from the first file were returned
            assert len(chunks) == 1
            assert chunks[0]["title"] == "File 1"

    @patch("json.dump")
    @patch("pandas.DataFrame.to_csv")
    def test_save_processed_data(self, mock_to_csv, mock_json_dump):
        """Test saving processed data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            processor = RAGProcessor(
                input_dir="./input",
                output_dir=temp_dir
            )
            
            # Create some test chunks
            chunks = [
                {"title": "File 1", "content": "Chunk 1", "embedding": [0.1] * 768, "url": "https://example.com/file1", "chunk_id": "file1_0"},
                {"title": "File 2", "content": "Chunk 2", "embedding": [0.2] * 768, "url": "https://example.com/file2", "chunk_id": "file2_0"}
            ]
            
            # Mock the open function
            with patch("builtins.open", mock_open()) as mock_file:
                # Save the processed data
                processor.save_processed_data(chunks)
                
                # Check that the JSON file was opened
                mock_file.assert_called_with(Path(temp_dir) / "chunks.json", "w", encoding="utf-8")
                
                # Check that json.dump was called with the chunks
                mock_json_dump.assert_called_once()
                args, kwargs = mock_json_dump.call_args
                assert args[0] == chunks
                
                # Check that pandas.DataFrame.to_csv was called
                mock_to_csv.assert_called_once()
                args, kwargs = mock_to_csv.call_args
                assert kwargs["index"] is False

    def test_process(self):
        """Test the full processing pipeline."""
        with tempfile.TemporaryDirectory() as temp_dir:
            processor = RAGProcessor(
                input_dir="./input",
                output_dir=temp_dir
            )
            
            # Mock the process_all_files method
            processor.process_all_files = MagicMock(return_value=[
                {"title": "File 1", "content": "Chunk 1", "embedding": [0.1] * 768, "url": "https://example.com/file1", "chunk_id": "file1_0"},
                {"title": "File 2", "content": "Chunk 2", "embedding": [0.2] * 768, "url": "https://example.com/file2", "chunk_id": "file2_0"}
            ])
            
            # Mock the save_processed_data method
            processor.save_processed_data = MagicMock()
            
            # Process the data
            processor.process()
            
            # Check that the process_all_files method was called
            processor.process_all_files.assert_called_once()
            
            # Check that the save_processed_data method was called with the chunks
            processor.save_processed_data.assert_called_once_with([
                {"title": "File 1", "content": "Chunk 1", "embedding": [0.1] * 768, "url": "https://example.com/file1", "chunk_id": "file1_0"},
                {"title": "File 2", "content": "Chunk 2", "embedding": [0.2] * 768, "url": "https://example.com/file2", "chunk_id": "file2_0"}
            ])
