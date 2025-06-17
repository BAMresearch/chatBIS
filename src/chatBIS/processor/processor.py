#!/usr/bin/env python3
"""
RAG Processor for ReadtheDocs Content

This module processes the content scraped from ReadtheDocs sites and prepares it
for use in a RAG (Retrieval Augmented Generation) pipeline. It chunks the content,
creates embeddings, and stores them in a vector database.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Configure logging
logger = logging.getLogger(__name__)

# Try to import Ollama embeddings, but don't fail if it's not available
try:
    from langchain_ollama import OllamaEmbeddings
    OLLAMA_AVAILABLE = True
    # Check if Ollama server is running
    try:
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        # Test with a simple embedding
        test_embedding = embeddings.embed_query("test")
        if test_embedding:
            logger.info("Ollama server is running and embeddings are working.")
        else:
            logger.warning("Ollama embeddings returned empty result.")
            OLLAMA_AVAILABLE = False
    except Exception as e:
        logger.warning(f"Error connecting to Ollama server: {e}")
        OLLAMA_AVAILABLE = False
except ImportError:
    logger.warning("Langchain Ollama package not available. Using dummy embeddings.")
    OLLAMA_AVAILABLE = False


class ContentChunker:
    """Class for chunking content into smaller pieces."""

    def __init__(
        self,
        min_chunk_size: int = 100,
        max_chunk_size: int = 1000,
        chunk_overlap: int = 50
    ):
        """
        Initialize the chunker.

        Args:
            min_chunk_size: The minimum size of a chunk in characters
            max_chunk_size: The maximum size of a chunk in characters
            chunk_overlap: The overlap between chunks in characters
        """
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_content(self, content: str) -> List[str]:
        """
        Chunk the content into smaller pieces.

        Args:
            content: The content to chunk

        Returns:
            A list of content chunks
        """
        # Split the content into paragraphs
        paragraphs = [p for p in content.split("\n\n") if p.strip()]

        chunks = []
        current_chunk = ""
        section_heading = None
        subsection_heading = None

        for paragraph in paragraphs:
            # Check if this is a heading
            is_main_heading = paragraph.strip().startswith('# ')
            is_section_heading = paragraph.strip().startswith('## ')
            is_subsection_heading = paragraph.strip().startswith('### ')

            # If we have a new main heading or section heading, start a new chunk
            if is_main_heading or is_section_heading:
                # Save the current chunk if it's not empty and meets the minimum size
                if current_chunk and len(current_chunk) >= self.min_chunk_size:
                    chunks.append(current_chunk.strip())

                # Start a new chunk with this heading
                current_chunk = paragraph
                section_heading = paragraph
                subsection_heading = None
                continue

            # If we have a new subsection heading, keep it with the current section if possible
            if is_subsection_heading:
                subsection_heading = paragraph

                # If the current chunk is already large, start a new one
                if len(current_chunk) >= self.max_chunk_size * 0.7:
                    if current_chunk and len(current_chunk) >= self.min_chunk_size:
                        chunks.append(current_chunk.strip())

                    # Start a new chunk with the section heading and this subsection
                    if section_heading and section_heading != subsection_heading:
                        current_chunk = section_heading + "\n\n" + subsection_heading
                    else:
                        current_chunk = subsection_heading
                else:
                    # Add the subsection to the current chunk
                    if current_chunk:
                        current_chunk += "\n\n" + subsection_heading
                    else:
                        current_chunk = subsection_heading
                continue

            # If adding this paragraph would make the chunk too large, save the current chunk
            if len(current_chunk) + len(paragraph) > self.max_chunk_size and len(current_chunk) >= self.min_chunk_size:
                chunks.append(current_chunk.strip())

                # Start a new chunk with context from previous headings
                if subsection_heading:
                    # If we have a subsection, include both section and subsection headings
                    if section_heading and section_heading != subsection_heading:
                        current_chunk = section_heading + "\n\n" + subsection_heading + "\n\n"
                    else:
                        current_chunk = subsection_heading + "\n\n"
                elif section_heading:
                    # Otherwise just include the section heading
                    current_chunk = section_heading + "\n\n"
                else:
                    current_chunk = ""

            # Add the paragraph to the current chunk
            if current_chunk:
                current_chunk += "\n\n" + paragraph
            else:
                current_chunk = paragraph

        # Add the last chunk if it's not empty and meets the minimum size
        if current_chunk and len(current_chunk) >= self.min_chunk_size:
            chunks.append(current_chunk.strip())

        return chunks


class EmbeddingGenerator:
    """Class for generating embeddings for content chunks."""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the embedding generator.

        Args:
            api_key: Not used for Ollama, kept for compatibility
        """
        # api_key is not used for Ollama, but kept for compatibility
        self.api_key = api_key

        if OLLAMA_AVAILABLE:
            logger.info("Using Ollama for embeddings")
            self.embeddings_model = OllamaEmbeddings(model="nomic-embed-text")
        else:
            logger.warning("Using dummy embeddings (random vectors)")
            self.embeddings_model = None

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: The texts to generate embeddings for

        Returns:
            A list of embeddings (one for each text)
        """
        if not texts:
            return []

        if OLLAMA_AVAILABLE and self.embeddings_model:
            try:
                # Use Ollama's embedding API
                embeddings = []
                for text in texts:
                    # Process one text at a time
                    embedding = self.embeddings_model.embed_query(text)
                    embeddings.append(embedding)
                return embeddings

            except Exception as e:
                logger.error(f"Error generating embeddings with Ollama: {e}")
                logger.warning("Falling back to dummy embeddings")

        # If Ollama is not available or there was an error, use dummy embeddings
        return [self._generate_dummy_embedding() for _ in texts]

    def _generate_dummy_embedding(self, dim: int = 1536) -> List[float]:
        """
        Generate a dummy embedding (random vector).

        Args:
            dim: The dimension of the embedding

        Returns:
            A random vector of the specified dimension
        """
        # Generate a random vector
        embedding = np.random.normal(0, 1, dim)

        # Normalize it to unit length
        embedding = embedding / np.linalg.norm(embedding)

        return embedding.tolist()


class RAGProcessor:
    """Class for processing content for RAG."""

    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        api_key: Optional[str] = None,
        min_chunk_size: int = 100,
        max_chunk_size: int = 1000,
        chunk_overlap: int = 50
    ):
        """
        Initialize the RAG processor.

        Args:
            input_dir: The directory containing the scraped content
            output_dir: The directory to save the processed content to
            api_key: Not used for Ollama, kept for compatibility
            min_chunk_size: The minimum size of a chunk in characters
            max_chunk_size: The maximum size of a chunk in characters
            chunk_overlap: The overlap between chunks in characters
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)

        # Create the output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.chunker = ContentChunker(
            min_chunk_size=min_chunk_size,
            max_chunk_size=max_chunk_size,
            chunk_overlap=chunk_overlap
        )

        # api_key is not used for Ollama, but kept for compatibility
        self.embedding_generator = EmbeddingGenerator(api_key=api_key)

    def process_file(self, file_path: Path) -> List[Dict]:
        """
        Process a single file.

        Args:
            file_path: The path to the file to process

        Returns:
            A list of dictionaries containing the processed chunks
        """
        logger.info(f"Processing {file_path}")

        # Read the file
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Extract metadata
        title = ""
        url = ""

        for line in content.split("\n")[:10]:  # Look at the first 10 lines
            if line.startswith("Title: "):
                title = line[len("Title: "):]
            elif line.startswith("URL: "):
                url = line[len("URL: "):]
            elif line.startswith("---"):
                # End of metadata
                break

        # Remove metadata from content
        content = content.split("---\n\n", 1)[-1]

        # Chunk the content
        chunks = self.chunker.chunk_content(content)

        # Generate embeddings
        embeddings = self.embedding_generator.generate_embeddings(chunks)

        # Create a list of dictionaries for each chunk
        processed_chunks = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            processed_chunks.append({
                "title": title,
                "url": url,
                "content": chunk,
                "embedding": embedding,
                "chunk_id": f"{file_path.stem}_{i}"
            })

        return processed_chunks

    def process_all_files(self) -> List[Dict]:
        """
        Process all files in the input directory.

        Returns:
            A list of dictionaries containing all processed chunks
        """
        all_chunks = []

        # Find all text files in the input directory
        for file_path in self.input_dir.glob("*.txt"):
            try:
                chunks = self.process_file(file_path)
                all_chunks.extend(chunks)
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")

        return all_chunks

    def save_processed_data(self, chunks: List[Dict]) -> None:
        """
        Save the processed data.

        Args:
            chunks: The processed chunks to save
        """
        # Save the chunks as a JSON file
        chunks_file = self.output_dir / "chunks.json"
        with open(chunks_file, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved {len(chunks)} chunks to {chunks_file}")

        # Create a DataFrame without the embeddings (for easier viewing)
        df = pd.DataFrame([{k: v for k, v in chunk.items() if k != "embedding"} for chunk in chunks])

        # Save the DataFrame as a CSV file
        csv_file = self.output_dir / "chunks.csv"
        df.to_csv(csv_file, index=False)

        logger.info(f"Saved chunk metadata to {csv_file}")

    def process(self) -> None:
        """
        Process all files and save the results.
        """
        logger.info(f"Processing files in {self.input_dir}")

        # Process all files
        chunks = self.process_all_files()

        # Save the processed data
        self.save_processed_data(chunks)

        logger.info(f"Finished processing {len(chunks)} chunks")
