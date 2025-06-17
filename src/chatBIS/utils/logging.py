"""
Logging utilities for the openBIS Chatbot.
"""

import logging
import sys


def setup_logging(level=logging.INFO):
    """
    Set up logging for the application.
    
    Args:
        level: The logging level to use
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout
    )
    
    # Reduce logging from other libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    
    return logging.getLogger(__name__)
