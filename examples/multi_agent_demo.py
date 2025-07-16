#!/usr/bin/env python3
"""
Multi-Agent openBIS Chatbot Demo

This script demonstrates the multi-agent capabilities of the openBIS chatbot,
showing both RAG (documentation) and function calling (pybis) functionality.
"""

import os
import sys
import logging

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from openbis_chatbot.query.conversation_engine import ConversationEngine

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def demo_conversation():
    """Run a demo conversation showing both RAG and function calling."""
    
    print("🤖 Multi-Agent openBIS Chatbot Demo")
    print("=" * 50)
    print("This demo shows both documentation queries (RAG) and function calls (pybis)")
    print()
    
    # Check if data exists
    data_dir = "data/processed"
    if not os.path.exists(data_dir):
        print("❌ Error: Processed data not found.")
        print("Please run the following commands first:")
        print("1. python -m openbis_chatbot scrape")
        print("2. python -m openbis_chatbot process")
        return
    
    try:
        # Initialize the conversation engine
        print("🔧 Initializing multi-agent conversation engine...")
        engine = ConversationEngine(data_dir=data_dir)
        session_id = engine.create_session()
        print(f"✅ Session created: {session_id}")
        print()
        
        # Demo queries that showcase different agent capabilities
        demo_queries = [
            {
                "query": "What is openBIS?",
                "expected_agent": "RAG Agent",
                "description": "Basic documentation query"
            },
            {
                "query": "Check if I'm connected to openBIS",
                "expected_agent": "Function Calling Agent", 
                "description": "Connection status check"
            },
            {
                "query": "How do I create a sample in openBIS?",
                "expected_agent": "RAG Agent",
                "description": "Documentation about sample creation"
            },
            {
                "query": "List all available tools",
                "expected_agent": "Function Calling Agent",
                "description": "Show available pybis functions"
            },
            {
                "query": "What is pybis and how do I use it?",
                "expected_agent": "RAG Agent", 
                "description": "Documentation about pybis"
            },
            {
                "query": "Connect to openBIS server at https://demo.openbis.ch",
                "expected_agent": "Function Calling Agent",
                "description": "Attempt to connect to openBIS (will fail without credentials)"
            }
        ]
        
        for i, demo in enumerate(demo_queries, 1):
            print(f"📝 Demo {i}: {demo['description']}")
            print(f"Expected Agent: {demo['expected_agent']}")
            print(f"Query: \"{demo['query']}\"")
            print()
            
            # Send query to the chatbot
            response, _, metadata = engine.chat(demo['query'], session_id)
            
            # Show the routing decision
            decision = metadata.get('decision', 'unknown')
            agent_map = {
                'rag': 'RAG Agent',
                'function_call': 'Function Calling Agent',
                'conversation': 'Conversation Agent'
            }
            actual_agent = agent_map.get(decision, f'Unknown ({decision})')
            
            print(f"🎯 Actual Agent: {actual_agent}")
            print(f"📊 Metadata: {metadata}")
            print("💬 Response:")
            print(f"   {response}")
            print()
            print("-" * 50)
            print()
        
        print("✅ Demo completed successfully!")
        print()
        print("🔍 Key Observations:")
        print("• Documentation queries are handled by the RAG Agent")
        print("• Action requests are handled by the Function Calling Agent") 
        print("• The router automatically decides which agent to use")
        print("• Conversation history is maintained across all interactions")
        print()
        
    except Exception as e:
        print(f"❌ Error during demo: {e}")
        logger.exception("Demo failed")

def interactive_mode():
    """Run interactive mode where users can test queries."""
    
    print("🤖 Interactive Multi-Agent Mode")
    print("=" * 40)
    print("Type your queries to test the multi-agent system.")
    print("Type 'quit' to exit.")
    print()
    
    data_dir = "data/processed"
    if not os.path.exists(data_dir):
        print("❌ Error: Processed data not found.")
        return
    
    try:
        engine = ConversationEngine(data_dir=data_dir)
        session_id = engine.create_session()
        print(f"✅ Session created: {session_id}")
        print()
        
        while True:
            try:
                query = input("You: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if not query:
                    continue
                
                print("🤔 Processing...")
                response, _, metadata = engine.chat(query, session_id)
                
                decision = metadata.get('decision', 'unknown')
                agent_map = {
                    'rag': 'RAG',
                    'function_call': 'Function',
                    'conversation': 'Conversation'
                }
                agent = agent_map.get(decision, decision)
                
                print(f"🤖 Assistant ({agent}): {response}")
                print()
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                print()
                
    except Exception as e:
        print(f"❌ Error initializing interactive mode: {e}")

def main():
    """Main function."""
    print("Multi-Agent openBIS Chatbot")
    print("Choose a mode:")
    print("1. Demo mode (automated demonstration)")
    print("2. Interactive mode (manual testing)")
    print()
    
    try:
        choice = input("Enter choice (1 or 2): ").strip()
        
        if choice == "1":
            demo_conversation()
        elif choice == "2":
            interactive_mode()
        else:
            print("Invalid choice. Please enter 1 or 2.")
            
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
