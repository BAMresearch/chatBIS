#!/usr/bin/env python3
"""
Command-line interface for the ReadtheDocs scraper.
"""

import argparse
import logging
import sys

from openbis_chatbot.scraper.scraper import ReadTheDocsScraper

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args(args=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Scrape content from a ReadtheDocs site.")
    parser.add_argument("--url", required=True, help="The base URL of the ReadtheDocs site")
    parser.add_argument("--output", required=True, help="The directory to save the scraped content to")
    parser.add_argument("--version", help="The specific version to scrape (e.g., 'en/latest')")
    parser.add_argument("--delay", type=float, default=0.5, help="The delay between requests in seconds")
    parser.add_argument("--max-pages", type=int, help="The maximum number of pages to scrape")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    return parser.parse_args(args)


def run_with_args(args):
    """Run the scraper with the given arguments."""
    # Set logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # Create and run the scraper
    scraper = ReadTheDocsScraper(
        base_url=args.url,
        output_dir=args.output,
        target_version=args.version,
        delay=args.delay,
        max_pages=args.max_pages
    )

    try:
        scraper.scrape()
        return 0
    except Exception as e:
        logger.error(f"Error during scraping: {e}")
        return 1


def main():
    """Main entry point for the script."""
    args = parse_args()
    return run_with_args(args)


if __name__ == "__main__":
    sys.exit(main())
