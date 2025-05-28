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
from ..tools import PyBISToolManager

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
    # Multi-agent routing fields
    decision: str  # "rag", "function_call", "conversation"
    tool_action: Optional[Dict]  # Tool to execute and parameters
    tool_output: Optional[str]  # Result from tool execution
    final_response: str  # Final formatted response


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

        # Initialize tool manager
        self.tool_manager = PyBISToolManager()

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
        """Build the LangGraph conversation flow with multi-agent architecture."""

        def router_agent(state: ConversationState) -> ConversationState:
            """Enhanced router agent that decides which path to take based on comprehensive keyword analysis."""
            try:
                user_query = state["user_query"].lower()

                # Enhanced keyword-based routing for comprehensive pybis functionality

                # Connection-related keywords (always function call)
                connection_keywords = ['connect', 'login', 'disconnect', 'logout', 'connection', 'session']

                # Action keywords that indicate function calling
                action_keywords = [
                    # CRUD operations
                    'create', 'new', 'make', 'add', 'insert',
                    'update', 'modify', 'change', 'edit', 'set',
                    'delete', 'remove', 'drop',
                    'list', 'show', 'display', 'get', 'find', 'search', 'retrieve',

                    # openBIS entities
                    'space', 'spaces', 'project', 'projects',
                    'experiment', 'experiments', 'collection', 'collections',
                    'sample', 'samples', 'object', 'objects',
                    'dataset', 'datasets', 'data',

                    # Masterdata
                    'type', 'types', 'property', 'properties', 'vocabulary', 'vocabularies',
                    'sample_type', 'experiment_type', 'dataset_type',

                    # File operations
                    'upload', 'download', 'file', 'files'
                ]

                # Documentation/explanation keywords (RAG)
                rag_keywords = [
                    'how', 'what', 'why', 'when', 'where', 'which', 'who',
                    'explain', 'describe', 'tell', 'about',
                    'documentation', 'docs', 'help', 'guide', 'tutorial', 'manual',
                    'definition', 'meaning', 'purpose', 'concept',
                    'difference', 'compare', 'versus', 'vs',
                    'example', 'examples', 'sample', 'demo'
                ]

                # Specific patterns that indicate function calling
                function_patterns = [
                    'in openbis',  # "list samples in openbis"
                    'from openbis',  # "get data from openbis"
                    'to openbis',  # "connect to openbis"
                    'on openbis',  # "create sample on openbis"
                ]

                # Specific patterns that indicate RAG (highest priority for documentation)
                rag_patterns = [
                    'how to',  # "how to create a sample"
                    'how can i',  # "how can i register"
                    'how do i',  # "how do i create"
                    'what is',  # "what is a sample"
                    'what are',  # "what are collections"
                    'can you explain',  # "can you explain"
                    'tell me about',  # "tell me about"
                    'explain how',  # "explain how to"
                    'show me how',  # "show me how to"
                    'help me',  # "help me understand"
                    'i want to know',  # "i want to know how"
                    'i need to know',  # "i need to know how"
                ]

                # Check for connection keywords (highest priority for function calls)
                if any(keyword in user_query for keyword in connection_keywords):
                    state["decision"] = "function_call"
                    logger.info(f"Router decision: function_call (connection keyword) for query: {state['user_query']}")
                    return state

                # Check for RAG patterns (second highest priority - documentation questions)
                has_rag_patterns = any(pattern in user_query for pattern in rag_patterns)
                if has_rag_patterns:
                    state["decision"] = "rag"
                    logger.info(f"Router decision: rag (documentation pattern) for query: {state['user_query']}")
                    return state

                # Check for function patterns (third priority)
                has_function_patterns = any(pattern in user_query for pattern in function_patterns)
                if has_function_patterns:
                    state["decision"] = "function_call"
                else:
                    # Check for keyword presence
                    has_action_keywords = any(keyword in user_query for keyword in action_keywords)
                    has_rag_keywords = any(keyword in user_query for keyword in rag_keywords)

                    # Decision logic
                    if has_action_keywords and not has_rag_keywords:
                        state["decision"] = "function_call"
                    elif has_rag_keywords and not has_action_keywords:
                        state["decision"] = "rag"
                    elif has_action_keywords and has_rag_keywords:
                        # Both present - check which is more prominent or use context
                        if any(word in user_query for word in ['how to', 'what is', 'explain']):
                            state["decision"] = "rag"
                        else:
                            state["decision"] = "function_call"
                    else:
                        # No clear indicators - default to RAG for safety
                        state["decision"] = "rag"

                logger.info(f"Router decision: {state['decision']} for query: {state['user_query']}")
                return state

            except Exception as e:
                logger.error(f"Error in router agent: {e}")
                state["decision"] = "rag"  # Default fallback
                return state

        def rag_agent(state: ConversationState) -> ConversationState:
            """RAG agent for documentation queries."""
            try:
                # Get relevant chunks for the current query
                relevant_chunks = self.rag_engine.retrieve_relevant_chunks(
                    state["user_query"], top_k=3
                )
                state["rag_context"] = relevant_chunks
                logger.info(f"Retrieved {len(relevant_chunks)} relevant chunks")

                # Generate RAG response
                if not OLLAMA_AVAILABLE or not self.llm:
                    state["response"] = "Ollama not available. Cannot generate response."
                    return state

                # Build conversation context
                messages = []

                # Add conversation history
                if state["messages"]:
                    messages.extend(state["messages"])

                # Add system message with RAG context
                if relevant_chunks:
                    context_text = "\n\n".join([
                        f"Source: {chunk.get('title', 'Unknown')} ({chunk.get('url', 'No URL')})\nContent: {chunk['content']}"
                        for chunk in relevant_chunks
                    ])

                    context_message = SystemMessage(content=f"""You are a helpful assistant specializing in openBIS, a system for managing research data.
You provide friendly, clear, and accurate answers about openBIS based on the provided context.

Context from openBIS documentation:
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

                logger.info(f"Generated RAG response with estimated {state['token_count']} tokens")
                return state

            except Exception as e:
                logger.error(f"Error in RAG agent: {e}")
                state["response"] = f"I encountered an error: {str(e)}"
                state["rag_context"] = []
                return state

        def function_calling_agent(state: ConversationState) -> ConversationState:
            """Function calling agent for pybis tool execution."""
            try:
                if not OLLAMA_AVAILABLE or not self.llm:
                    state["tool_output"] = "Ollama not available. Cannot execute functions."
                    return state

                # Get available tools
                available_tools = self.tool_manager.get_tools()

                if not available_tools:
                    state["tool_output"] = "No tools available. pybis may not be installed."
                    return state

                # Create tool descriptions for the LLM
                tool_descriptions = []
                for tool in available_tools:
                    tool_descriptions.append(f"- {tool.name}: {tool.description}")

                tools_text = "\n".join(tool_descriptions)

                # Create prompt for tool selection
                prompt = f"""You are a function calling assistant for openBIS. The user wants to perform an action.

Available tools:
{tools_text}

User query: {state["user_query"]}

Based on the user's query, determine:
1. Which tool to use (if any)
2. What parameters to extract from the query

Respond in this exact format:
TOOL: tool_name
PARAMETERS: param1=value1, param2=value2

If no tool is appropriate, respond with:
TOOL: none
REASON: explanation why no tool is suitable"""

                # Get tool selection from LLM
                response = self.llm.invoke(prompt)
                response_text = response.content.strip()

                # Parse the response
                lines = response_text.split('\n')
                tool_name = None
                parameters = ""

                for line in lines:
                    if line.startswith('TOOL:'):
                        tool_name = line.split(':', 1)[1].strip()
                    elif line.startswith('PARAMETERS:'):
                        parameters = line.split(':', 1)[1].strip()

                if tool_name == "none" or not tool_name:
                    state["tool_output"] = "I couldn't identify an appropriate action for your request. Could you be more specific?"
                    return state

                # Find the tool
                selected_tool = None
                for tool in available_tools:
                    if tool.name == tool_name:
                        selected_tool = tool
                        break

                if not selected_tool:
                    state["tool_output"] = f"Tool '{tool_name}' not found."
                    return state

                # Execute the tool
                logger.info(f"Executing tool: {tool_name} with parameters: {parameters}")
                result = selected_tool.func(parameters)
                state["tool_output"] = result

                logger.info(f"Tool execution completed: {tool_name}")
                return state

            except Exception as e:
                logger.error(f"Error in function calling agent: {e}")
                state["tool_output"] = f"Error executing function: {str(e)}"
                return state

        def format_response(state: ConversationState) -> ConversationState:
            """Format the final response based on the agent that processed the query."""
            try:
                if state["decision"] == "rag":
                    # Use RAG response
                    state["final_response"] = state.get("response", "No response generated.")
                elif state["decision"] == "function_call":
                    # Use tool output
                    tool_output = state.get("tool_output", "No tool output available.")
                    state["final_response"] = tool_output
                else:
                    # Fallback
                    state["final_response"] = state.get("response", "I'm not sure how to help with that.")

                # Set the response field for compatibility
                state["response"] = state["final_response"]

                logger.info(f"Formatted final response for decision: {state['decision']}")
                return state

            except Exception as e:
                logger.error(f"Error formatting response: {e}")
                state["final_response"] = f"Error formatting response: {str(e)}"
                state["response"] = state["final_response"]
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

        # Add nodes for multi-agent architecture
        workflow.add_node("router", router_agent)
        workflow.add_node("rag_agent", rag_agent)
        workflow.add_node("function_calling_agent", function_calling_agent)
        workflow.add_node("format_response", format_response)
        workflow.add_node("update_conversation", update_conversation)

        # Add conditional routing function
        def route_decision(state: ConversationState) -> str:
            """Route based on the router's decision."""
            decision = state.get("decision", "rag")
            if decision == "function_call":
                return "function_calling_agent"
            else:
                return "rag_agent"

        # Add edges
        workflow.set_entry_point("router")
        workflow.add_conditional_edges(
            "router",
            route_decision,
            {
                "rag_agent": "rag_agent",
                "function_calling_agent": "function_calling_agent"
            }
        )
        workflow.add_edge("rag_agent", "format_response")
        workflow.add_edge("function_calling_agent", "format_response")
        workflow.add_edge("format_response", "update_conversation")
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
                    token_count=0,
                    decision="",
                    tool_action=None,
                    tool_output=None,
                    final_response=""
                )
            else:
                # Create new conversation state
                initial_state = ConversationState(
                    messages=[],
                    user_query=user_input,
                    rag_context=[],
                    response="",
                    session_id=session_id,
                    token_count=0,
                    decision="",
                    tool_action=None,
                    tool_output=None,
                    final_response=""
                )
        except Exception:
            # Fallback to new state if there's an issue loading existing state
            initial_state = ConversationState(
                messages=[],
                user_query=user_input,
                rag_context=[],
                response="",
                session_id=session_id,
                token_count=0,
                decision="",
                tool_action=None,
                tool_output=None,
                final_response=""
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
                "decision": result.get("decision", "unknown"),
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
