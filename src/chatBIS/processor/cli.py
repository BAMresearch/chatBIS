#!/usr/bin/env python3
"""
Command-line interface for the RAG processor.
"""

import argparse
import logging
import sys
from dotenv import load_dotenv

from openbis_chatbot.processor.processor import RAGProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


def parse_args(args=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Process content for RAG.")
    parser.add_argument("--input", required=True, help="The directory containing the scraped content")
    parser.add_argument("--output", required=True, help="The directory to save the processed content to")
    parser.add_argument("--api-key", help="Not used for Ollama, kept for compatibility")
    parser.add_argument("--min-chunk-size", type=int, default=100, help="The minimum size of a chunk in characters")
    parser.add_argument("--max-chunk-size", type=int, default=1000, help="The maximum size of a chunk in characters")
    parser.add_argument("--chunk-overlap", type=int, default=50, help="The overlap between chunks in characters")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    return parser.parse_args(args)


def run_with_args(args):
    """Run the processor with the given arguments."""
    # Set logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # API key is not used for Ollama, but kept for compatibility
    api_key = args.api_key

    # Create and run the processor
    processor = RAGProcessor(
        input_dir=args.input,
        output_dir=args.output,
        api_key=api_key,
        min_chunk_size=args.min_chunk_size,
        max_chunk_size=args.max_chunk_size,
        chunk_overlap=args.chunk_overlap
    )

    try:
        processor.process()
        return 0
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        return 1


def main():
    """Main entry point for the script."""
    args = parse_args()
    return run_with_args(args)


if __name__ == "__main__":
    sys.exit(main())
