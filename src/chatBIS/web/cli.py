#!/usr/bin/env python3
"""
Command-line interface for the openBIS Chatbot web interface.
"""

import argparse
import logging
import sys
import os

from openbis_chatbot.web.app import run_app

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Default paths
DEFAULT_DATA_DIR = os.path.join(os.getcwd(), "data", "processed")


def parse_args(args=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run the openBIS Chatbot web interface.")
    parser.add_argument("--data", default=DEFAULT_DATA_DIR, help="The directory containing the processed content")
    parser.add_argument("--host", default="0.0.0.0", help="The host to run the web interface on")
    parser.add_argument("--port", type=int, default=5000, help="The port to run the web interface on")
    parser.add_argument("--model", default="qwen3", help="The Ollama model to use for chat")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    return parser.parse_args(args)


def run_with_args(args):
    """Run the web interface with the given arguments."""
    try:
        logger.info(f"Starting web interface on {args.host}:{args.port}...")
        logger.info(f"Using data from {args.data}")
        logger.info(f"Using model {args.model}")

        run_app(
            host=args.host,
            port=args.port,
            debug=args.debug,
            data_dir=args.data,
            model=args.model
        )

        return 0

    except Exception as e:
        logger.error(f"Error running web interface: {e}")
        return 1


def main():
    """Main entry point for the script."""
    args = parse_args()
    return run_with_args(args)


if __name__ == "__main__":
    sys.exit(main())
