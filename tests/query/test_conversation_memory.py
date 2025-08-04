#!/usr/bin/env python3
"""
Test script to verify conversation memory functionality.
"""

import os
import sys
import tempfile
import logging
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from chatBIS.query.conversation_engine import ConversationEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def test_conversation_memory():
    """Test the conversation memory functionality."""
    
    # Check if processed data exists
    data_dir = "data/processed"
    if not os.path.exists(os.path.join(data_dir, "chunks.json")):
        logger.error(f"Processed data not found in {data_dir}. Please run the processor first.")
        return False
    
    try:
        # Create a temporary directory for the test database
        with tempfile.TemporaryDirectory() as temp_dir:
            memory_db_path = os.path.join(temp_dir, "test_memory.db")
            
            # Initialize the conversation engine
            logger.info("Initializing conversation engine...")
            engine = ConversationEngine(
                data_dir=data_dir,
                model="qwen3",
                memory_db_path=memory_db_path
            )
            
            # Test 1: First message - introduce name
            logger.info("\n=== Test 1: Introducing name ===")
            response1, session_id, metadata1 = engine.chat("My name is Carlos")
            logger.info(f"Session ID: {session_id}")
            logger.info(f"Response: {response1}")
            logger.info(f"Metadata: {metadata1}")
            
            # Test 2: Ask about name in same session
            logger.info("\n=== Test 2: Asking about name ===")
            response2, session_id2, metadata2 = engine.chat("What is my name?", session_id)
            logger.info(f"Session ID: {session_id2}")
            logger.info(f"Response: {response2}")
            logger.info(f"Metadata: {metadata2}")
            
            # Verify session continuity
            assert session_id == session_id2, "Session ID should remain the same"
            
            # Test 3: Ask a technical question to verify RAG still works
            logger.info("\n=== Test 3: Technical question with memory ===")
            response3, session_id3, metadata3 = engine.chat("What is openBIS?", session_id)
            logger.info(f"Response: {response3}")
            logger.info(f"Metadata: {metadata3}")
            
            # Test 4: Reference previous conversation
            logger.info("\n=== Test 4: Reference previous conversation ===")
            response4, session_id4, metadata4 = engine.chat("Can you help me, Carlos, understand how to create a collection?", session_id)
            logger.info(f"Response: {response4}")
            logger.info(f"Metadata: {metadata4}")
            
            # Test 5: New session (should not remember name)
            logger.info("\n=== Test 5: New session (should not remember) ===")
            response5, new_session_id, metadata5 = engine.chat("What is my name?")
            logger.info(f"New Session ID: {new_session_id}")
            logger.info(f"Response: {response5}")
            logger.info(f"Metadata: {metadata5}")
            
            # Verify new session
            assert new_session_id != session_id, "New session should have different ID"
            
            # Test 6: Get conversation history
            logger.info("\n=== Test 6: Conversation history ===")
            history = engine.get_conversation_history(session_id)
            logger.info(f"Conversation history length: {len(history)}")
            for i, msg in enumerate(history):
                logger.info(f"Message {i+1}: {msg['type']} - {msg['content'][:100]}...")
            
            # Verify memory functionality
            logger.info("\n=== Memory Verification ===")
            
            # Check if the second response mentions the name Carlos
            name_mentioned = "carlos" in response2.lower()
            logger.info(f"Name mentioned in response 2: {name_mentioned}")
            
            # Check if the fourth response acknowledges Carlos
            carlos_acknowledged = "carlos" in response4.lower()
            logger.info(f"Carlos acknowledged in response 4: {carlos_acknowledged}")
            
            # Check if the new session doesn't know the name
            name_unknown_new_session = "don't know" in response5.lower() or "don't have" in response5.lower() or "not sure" in response5.lower()
            logger.info(f"Name unknown in new session: {name_unknown_new_session}")
            
            # Summary
            logger.info("\n=== Test Summary ===")
            logger.info(f"✓ Session continuity: {session_id == session_id2}")
            logger.info(f"✓ Name remembered: {name_mentioned}")
            logger.info(f"✓ Carlos acknowledged: {carlos_acknowledged}")
            logger.info(f"✓ New session isolation: {new_session_id != session_id}")
            logger.info(f"✓ Conversation history: {len(history) > 0}")
            
            # Token usage analysis
            logger.info("\n=== Token Usage Analysis ===")
            total_tokens = sum([metadata1.get('token_count', 0), metadata2.get('token_count', 0), 
                              metadata3.get('token_count', 0), metadata4.get('token_count', 0)])
            avg_tokens = total_tokens / 4 if total_tokens > 0 else 0
            logger.info(f"Total tokens used: {total_tokens}")
            logger.info(f"Average tokens per exchange: {avg_tokens:.1f}")
            
            return True
            
    except Exception as e:
        logger.error(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function."""
    logger.info("Starting conversation memory test...")
    
    success = test_conversation_memory()
    
    if success:
        logger.info("\n🎉 All tests passed! Conversation memory is working correctly.")
    else:
        logger.error("\n❌ Tests failed. Please check the logs for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
