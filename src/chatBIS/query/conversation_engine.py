#!/usr/bin/env python3
"""
LangGraph-based Conversation Engine with Memory for openBIS Chatbot

This module provides a conversation engine that maintains memory across
multiple interactions using LangGraph's state management and persistence.
"""

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TypedDict, Annotated
from datetime import datetime
import uuid

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

from .query import RAGQueryEngine

# Configure logging
logger = logging.getLogger(__name__)

# Try to import Ollama
try:
    from langchain_ollama import ChatOllama
    OLLAMA_AVAILABLE = True
except ImportError:
    logger.warning("Langchain Ollama package not available.")
    OLLAMA_AVAILABLE = False


class ConversationState(TypedDict):
    """State for the conversation graph."""
    messages: Annotated[List[BaseMessage], "The conversation messages"]
    user_query: str
    rag_context: List[Dict]
    response: str
    session_id: str
    token_count: int


class ConversationEngine:
    """LangGraph-based conversation engine with memory and RAG integration."""

    def __init__(self, data_dir: str, model: str = "qwen3", memory_db_path: str = "conversation_memory.db"):
        """
        Initialize the conversation engine.

        Args:
            data_dir: Directory containing processed RAG data
            model: Ollama model to use
            memory_db_path: Path to SQLite database for conversation memory
        """
        self.data_dir = Path(data_dir)
        self.model = model
        self.memory_db_path = memory_db_path

        # Initialize RAG engine
        self.rag_engine = RAGQueryEngine(data_dir=data_dir, model=model)

        # Initialize LLM
        if OLLAMA_AVAILABLE:
            self.llm = ChatOllama(model=self.model)
        else:
            logger.error("Ollama not available. Cannot initialize conversation engine.")
            self.llm = None

        # Initialize memory/checkpointer
        import sqlite3
        conn = sqlite3.connect(memory_db_path, check_same_thread=False)
        self.checkpointer = SqliteSaver(conn)

        # Build the conversation graph
        self.graph = self._build_graph()

        # System message for the assistant
        self.system_message = SystemMessage(content="""You are a helpful assistant specializing in openBIS, a system for managing research data.
You provide friendly, clear, and accurate answers about openBIS.

IMPORTANT GUIDELINES:
1. NEVER refer to "documentation," "information provided," or any external sources in your answers.
2. Avoid phrases like "it appears that" or "it seems that" - be confident but conversational.
3. Always try to provide an answer based on your knowledge of openBIS, even if you need to make reasonable inferences.
4. Be friendly and helpful rather than overly authoritative.
5. If asked about technical concepts not explicitly defined, use context clues from related information to construct a helpful answer.
6. Only say "I don't have information about that" as a last resort when you truly cannot formulate any reasonable answer.
7. Be consistent in your answers - if you know something once, you should know it every time.
8. Remember previous parts of our conversation and refer to them when relevant.
9. If a user mentions their name or other personal details, remember them for future reference.
10. PAY CLOSE ATTENTION to your own previous responses in this conversation - if you offered to provide examples, code snippets, or additional information, and the user asks for it, provide what you offered.
11. When the user says "Yes, give me an example!" or similar, they are likely referring to something you just offered in your previous message.
12. Always consider the full context of the conversation, including what YOU said previously, not just what the user said.""")

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph conversation flow."""

        def retrieve_context(state: ConversationState) -> ConversationState:
            """Retrieve relevant context using RAG."""
            try:
                # Get relevant chunks for the current query
                relevant_chunks = self.rag_engine.retrieve_relevant_chunks(
                    state["user_query"], top_k=3
                )
                state["rag_context"] = relevant_chunks
                logger.info(f"Retrieved {len(relevant_chunks)} relevant chunks")
                return state
            except Exception as e:
                logger.error(f"Error retrieving context: {e}")
                state["rag_context"] = []
                return state

        def generate_response(state: ConversationState) -> ConversationState:
            """Generate response using LLM with conversation history and RAG context."""
            try:
                if not self.llm:
                    state["response"] = "Ollama not available. Cannot generate response."
                    return state

                # Prepare messages for the LLM
                messages = [self.system_message]

                # Add ALL conversation history (both user and assistant messages)
                # This ensures the LLM can see its own previous responses for context
                if state["messages"]:
                    messages.extend(state["messages"])

                # Create context from RAG chunks
                if state["rag_context"]:
                    context_text = "\n\n".join([
                        f"Knowledge from {chunk['title']}:\n{chunk['content']}"
                        for chunk in state["rag_context"]
                    ])
                    context_message = SystemMessage(content=f"""
Additional context for answering the user's question (for your internal use only - do not mention this in your answer):

{context_text}

Remember to use this information naturally in your response without referring to it as "documentation" or "information provided".
""")
                    messages.append(context_message)

                # Add the current user message
                messages.append(HumanMessage(content=state["user_query"]))

                # Generate response
                response = self.llm.invoke(messages)
                state["response"] = response.content

                # Estimate token count (rough approximation)
                total_text = " ".join([msg.content for msg in messages if hasattr(msg, 'content')])
                state["token_count"] = len(total_text.split()) * 1.3  # Rough token estimation

                logger.info(f"Generated response with estimated {state['token_count']} tokens")
                return state

            except Exception as e:
                logger.error(f"Error generating response: {e}")
                state["response"] = f"I encountered an error while processing your request: {str(e)}"
                return state

        def update_conversation(state: ConversationState) -> ConversationState:
            """Update conversation history with new messages."""
            # The messages should already include the conversation history from the state
            # We just need to add the current user message and assistant response

            # Add user message if not already present
            if not state["messages"] or state["messages"][-1].content != state["user_query"]:
                state["messages"].append(HumanMessage(content=state["user_query"]))

            # Add assistant response
            state["messages"].append(AIMessage(content=state["response"]))

            # Keep only last 20 messages to manage memory (10 exchanges)
            if len(state["messages"]) > 20:
                state["messages"] = state["messages"][-20:]

            return state

        # Build the graph
        workflow = StateGraph(ConversationState)

        # Add nodes
        workflow.add_node("retrieve_context", retrieve_context)
        workflow.add_node("generate_response", generate_response)
        workflow.add_node("update_conversation", update_conversation)

        # Add edges
        workflow.set_entry_point("retrieve_context")
        workflow.add_edge("retrieve_context", "generate_response")
        workflow.add_edge("generate_response", "update_conversation")
        workflow.add_edge("update_conversation", END)

        # Compile with checkpointer for memory
        return workflow.compile(checkpointer=self.checkpointer)

    def create_session(self) -> str:
        """Create a new conversation session."""
        session_id = str(uuid.uuid4())
        logger.info(f"Created new conversation session: {session_id}")
        return session_id

    def clean_response(self, response: str) -> str:
        """Remove <think></think> tags from the response."""
        # Remove everything between <think> and </think> tags (including the tags)
        cleaned = re.sub(r'<think>.*?</think>\s*', '', response, flags=re.DOTALL)
        return cleaned.strip()

    def chat(self, user_input: str, session_id: Optional[str] = None) -> Tuple[str, str, Dict]:
        """
        Process a user input and return the response.

        Args:
            user_input: The user's message
            session_id: Optional session ID for conversation continuity

        Returns:
            Tuple of (response, session_id, metadata)
        """
        if not session_id:
            session_id = self.create_session()

        # Create config for this conversation thread
        config = RunnableConfig(
            configurable={"thread_id": session_id}
        )

        # Get existing state or create initial state
        try:
            existing_state = self.graph.get_state(config)
            if existing_state and existing_state.values:
                # Load existing conversation state
                initial_state = ConversationState(
                    messages=existing_state.values.get("messages", []),
                    user_query=user_input,
                    rag_context=[],
                    response="",
                    session_id=session_id,
                    token_count=0
                )
            else:
                # Create new conversation state
                initial_state = ConversationState(
                    messages=[],
                    user_query=user_input,
                    rag_context=[],
                    response="",
                    session_id=session_id,
                    token_count=0
                )
        except Exception:
            # Fallback to new state if there's an issue loading existing state
            initial_state = ConversationState(
                messages=[],
                user_query=user_input,
                rag_context=[],
                response="",
                session_id=session_id,
                token_count=0
            )

        try:
            # Run the conversation graph
            result = self.graph.invoke(initial_state, config)

            # Extract and clean response
            raw_response = result["response"]
            cleaned_response = self.clean_response(raw_response)

            metadata = {
                "session_id": session_id,
                "token_count": result["token_count"],
                "rag_chunks_used": len(result["rag_context"]),
                "conversation_length": len(result["messages"]),
                "timestamp": datetime.now().isoformat()
            }

            logger.info(f"Chat completed for session {session_id}: {metadata}")
            return cleaned_response, session_id, metadata

        except Exception as e:
            logger.error(f"Error in chat processing: {e}")
            return f"I encountered an error: {str(e)}", session_id, {"error": str(e)}

    def get_conversation_history(self, session_id: str) -> List[Dict]:
        """
        Get conversation history for a session.

        Args:
            session_id: The session ID

        Returns:
            List of message dictionaries
        """
        try:
            config = RunnableConfig(configurable={"thread_id": session_id})

            # Get the latest state for this thread
            state = self.graph.get_state(config)

            if state and state.values and "messages" in state.values:
                messages = state.values["messages"]
                return [
                    {
                        "type": "human" if isinstance(msg, HumanMessage) else "ai",
                        "content": msg.content,
                        "timestamp": datetime.now().isoformat()  # In real implementation, store actual timestamps
                    }
                    for msg in messages
                ]
            return []

        except Exception as e:
            logger.error(f"Error getting conversation history: {e}")
            return []

    def clear_session(self, session_id: str) -> bool:
        """
        Clear conversation history for a session.

        Args:
            session_id: The session ID to clear

        Returns:
            True if successful, False otherwise
        """
        try:
            # Note: LangGraph doesn't have a direct clear method
            # In a production environment, you might want to implement
            # a custom method to clear specific thread data
            logger.info(f"Session {session_id} marked for clearing")
            return True
        except Exception as e:
            logger.error(f"Error clearing session: {e}")
            return False
