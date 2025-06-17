#!/usr/bin/env python3
"""
Command-line interface for the conversation engine with memory.
"""

import argparse
import logging
import sys
import os
import re
from dotenv import load_dotenv

from openbis_chatbot.query.conversation_engine import ConversationEngine

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
    parser = argparse.ArgumentParser(description="Chat with openBIS Assistant using conversation memory.")
    parser.add_argument("--data", required=True, help="The directory containing the processed content")
    parser.add_argument("--model", default="qwen3", help="The Ollama model to use for chat")
    parser.add_argument("--memory-db", help="Path to SQLite database for conversation memory (default: data/conversation_memory.db)")
    parser.add_argument("--session-id", help="Session ID to continue a previous conversation")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    return parser.parse_args(args)


def clean_response(response):
    """Remove <think></think> tags from the response."""
    # Remove everything between <think> and </think> tags (including the tags)
    cleaned = re.sub(r'<think>.*?</think>\s*', '', response, flags=re.DOTALL)
    return cleaned.strip()


def run_with_args(args):
    """Run the conversation engine with the given arguments."""
    # Set logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # Set up memory database path
    if args.memory_db:
        memory_db_path = args.memory_db
    else:
        memory_db_path = os.path.join(args.data, "conversation_memory.db")

    try:
        # Create the conversation engine
        conversation_engine = ConversationEngine(
            data_dir=args.data,
            model=args.model,
            memory_db_path=memory_db_path
        )

        # Get or create session ID
        session_id = args.session_id
        if session_id:
            print(f"📝 Continuing conversation with session: {session_id}")
        else:
            session_id = conversation_engine.create_session()
            print(f"📝 Started new conversation with session: {session_id}")

        # Interactive conversation loop
        print("🤖 openBIS Assistant with Memory")
        print("I'm here to help you with questions about openBIS. I'll remember our conversation!")
        print("Type 'exit' or 'quit' to end our conversation.")
        print("Type 'clear' to start a new conversation.")
        print()

        while True:
            # Get the query from the user
            query = input("You: ")

            # Exit if the user types 'exit' or 'quit'
            if query.lower() in ["exit", "quit"]:
                print("Goodbye! Have a great day!")
                break

            # Clear conversation if user types 'clear'
            if query.lower() == "clear":
                conversation_engine.clear_session(session_id)
                session_id = conversation_engine.create_session()
                print(f"🔄 Started new conversation with session: {session_id}")
                continue

            # Skip empty queries
            if not query.strip():
                continue

            try:
                # Process the conversation with memory
                response, session_id, metadata = conversation_engine.chat(query, session_id)

                # Clean the response to remove <think></think> tags
                clean_answer = clean_response(response)

                # Print the cleaned answer
                print(f"\nAssistant: {clean_answer}\n")

                # Log metadata if verbose
                if args.verbose:
                    logger.info(f"Conversation metadata: {metadata}")

            except Exception as e:
                logger.error(f"Error in conversation: {e}")
                print("I'm sorry, I encountered an error. Please try again or ask a different question.")

        return 0

    except Exception as e:
        logger.error(f"Error initializing query engine: {e}")
        return 1


def main():
    """Main entry point for the script."""
    args = parse_args()
    return run_with_args(args)


if __name__ == "__main__":
    sys.exit(main())
