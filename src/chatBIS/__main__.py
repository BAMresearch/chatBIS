"""
Main entry point for the openBIS Chatbot package.
"""

import argparse
import os
import sys
import logging

from openbis_chatbot.scraper.cli import main as scraper_main
from openbis_chatbot.processor.cli import main as processor_main
from openbis_chatbot.query.cli import main as query_main
from openbis_chatbot.web.cli import main as web_main
from openbis_chatbot.utils.logging import setup_logging

# Configure logging
logger = logging.getLogger(__name__)

# Default paths
DEFAULT_DATA_DIR = os.path.join(os.getcwd(), "data")
DEFAULT_RAW_DIR = os.path.join(DEFAULT_DATA_DIR, "raw")
DEFAULT_PROCESSED_DIR = os.path.join(DEFAULT_DATA_DIR, "processed")
DEFAULT_OPENBIS_URL = "https://openbis.readthedocs.io/en/latest/"
DEFAULT_MAX_PAGES = 100

def run_full_pipeline():
    """Run the full pipeline: scrape, process, and start the chatbot."""
    logger.info("Starting the full pipeline...")

    # Create directories if they don't exist
    os.makedirs(DEFAULT_RAW_DIR, exist_ok=True)
    os.makedirs(DEFAULT_PROCESSED_DIR, exist_ok=True)

    # Save original argv
    original_argv = sys.argv.copy()

    try:
        # Run scraper
        logger.info(f"Scraping data from {DEFAULT_OPENBIS_URL}...")
        from openbis_chatbot.scraper.cli import parse_args as scraper_parse_args
        from openbis_chatbot.scraper.cli import run_with_args as scraper_run

        scraper_args = scraper_parse_args([
            "--url", DEFAULT_OPENBIS_URL,
            "--output", DEFAULT_RAW_DIR,
            "--max-pages", str(DEFAULT_MAX_PAGES)
        ])

        scraper_result = scraper_run(scraper_args)
        if scraper_result != 0:
            logger.error("Scraping failed.")
            return scraper_result

        # Run processor
        logger.info(f"Processing data from {DEFAULT_RAW_DIR}...")
        from openbis_chatbot.processor.cli import parse_args as processor_parse_args
        from openbis_chatbot.processor.cli import run_with_args as processor_run

        processor_args = processor_parse_args([
            "--input", DEFAULT_RAW_DIR,
            "--output", DEFAULT_PROCESSED_DIR
        ])

        processor_result = processor_run(processor_args)
        if processor_result != 0:
            logger.error("Processing failed.")
            return processor_result

        # Run query
        logger.info(f"Starting chatbot with data from {DEFAULT_PROCESSED_DIR}...")
        from openbis_chatbot.query.cli import parse_args as query_parse_args
        from openbis_chatbot.query.cli import run_with_args as query_run

        query_args = query_parse_args([
            "--data", DEFAULT_PROCESSED_DIR
        ])

        return query_run(query_args)

    finally:
        # Restore original argv
        sys.argv = original_argv

def check_data_exists():
    """Check if processed data exists."""
    chunks_path = os.path.join(DEFAULT_PROCESSED_DIR, "chunks.json")
    return os.path.exists(chunks_path)

def run_web_interface():
    """Run the web interface with default settings."""
    setup_logging(logging.INFO)

    if check_data_exists():
        logger.info("Found existing processed data. Starting web interface...")
        # Save original argv
        original_argv = sys.argv.copy()

        # Temporarily modify sys.argv for the web module
        sys.argv = ["openbis-chatbot"]

        # Import and use the parse_args function from web.cli
        from openbis_chatbot.web.cli import parse_args
        args = parse_args([
            "--data", DEFAULT_PROCESSED_DIR,
            "--host", "127.0.0.1",
            "--port", "5000"
        ])

        # Restore original argv
        sys.argv = original_argv

        # Call web_main with the parsed args
        from openbis_chatbot.web.cli import run_with_args
        return run_with_args(args)
    else:
        logger.info("No processed data found. Running full pipeline first...")
        result = run_full_pipeline()
        if result != 0:
            return result

        # Now run the web interface
        return run_web_interface()

def auto_mode():
    """Automatically determine what to do based on data availability."""
    setup_logging(logging.INFO)

    if check_data_exists():
        logger.info("Found existing processed data. Starting chatbot...")
        # Instead of modifying sys.argv, directly call the query_main with the right arguments
        # Save original argv
        original_argv = sys.argv.copy()

        # Temporarily modify sys.argv for the query module
        sys.argv = ["openbis-chatbot"]

        # Import and use the parse_args function from query.cli
        from openbis_chatbot.query.cli import parse_args
        args = parse_args([
            "--data", DEFAULT_PROCESSED_DIR
        ])

        # Restore original argv
        sys.argv = original_argv

        # Call query_main with the parsed args
        from openbis_chatbot.query.cli import run_with_args
        return run_with_args(args)
    else:
        logger.info("No processed data found. Running full pipeline...")
        return run_full_pipeline()

def main():
    """Main entry point for the package."""
    # Check if no arguments were provided
    if len(sys.argv) == 1:
        return auto_mode()

    parser = argparse.ArgumentParser(
        description="openBIS Chatbot - A RAG-based chatbot for the openBIS documentation.",
        prog="openbis-chatbot"
    )

    # Add --web flag to run the web interface
    parser.add_argument("--web", action="store_true", help="Run the web interface instead of the CLI")

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Scraper command
    scraper_parser = subparsers.add_parser("scrape", help="Scrape content from a ReadtheDocs site")
    scraper_parser.add_argument("--url", required=True, help="The base URL of the ReadtheDocs site")
    scraper_parser.add_argument("--output", required=True, help="The directory to save the scraped content to")
    scraper_parser.add_argument("--version", help="The specific version to scrape (e.g., 'en/latest')")
    scraper_parser.add_argument("--delay", type=float, default=0.5, help="The delay between requests in seconds")
    scraper_parser.add_argument("--max-pages", type=int, help="The maximum number of pages to scrape")
    scraper_parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    # Processor command
    processor_parser = subparsers.add_parser("process", help="Process content for RAG")
    processor_parser.add_argument("--input", required=True, help="The directory containing the scraped content")
    processor_parser.add_argument("--output", required=True, help="The directory to save the processed content to")
    processor_parser.add_argument("--api-key", help="Not used for Ollama, kept for compatibility")
    processor_parser.add_argument("--min-chunk-size", type=int, default=100, help="The minimum size of a chunk in characters")
    processor_parser.add_argument("--max-chunk-size", type=int, default=1000, help="The maximum size of a chunk in characters")
    processor_parser.add_argument("--chunk-overlap", type=int, default=50, help="The overlap between chunks in characters")
    processor_parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    # Query command
    query_parser = subparsers.add_parser("query", help="Query processed content using RAG")
    query_parser.add_argument("--data", required=True, help="The directory containing the processed content")
    query_parser.add_argument("--api-key", help="Not used for Ollama, kept for compatibility")
    query_parser.add_argument("--model", default="qwen3", help="The Ollama model to use for chat")
    query_parser.add_argument("--top-k", type=int, default=5, help="The number of chunks to retrieve")
    query_parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    # Web command
    web_parser = subparsers.add_parser("web", help="Run the web interface")
    web_parser.add_argument("--data", default=DEFAULT_PROCESSED_DIR, help="The directory containing the processed content")
    web_parser.add_argument("--host", default="0.0.0.0", help="The host to run the web interface on")
    web_parser.add_argument("--port", type=int, default=5000, help="The port to run the web interface on")
    web_parser.add_argument("--model", default="qwen3", help="The Ollama model to use for chat")
    web_parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    # Auto command (hidden, for internal use)
    subparsers.add_parser("auto", help=argparse.SUPPRESS)

    args = parser.parse_args()

    # Check if --web flag is set
    if hasattr(args, 'web') and args.web:
        return run_web_interface()

    if args.command == "scrape":
        return scraper_main()
    elif args.command == "process":
        return processor_main()
    elif args.command == "query":
        return query_main()
    elif args.command == "web":
        return web_main()
    elif args.command == "auto":
        return auto_mode()
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
