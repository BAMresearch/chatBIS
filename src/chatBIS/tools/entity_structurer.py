#!/usr/bin/env python3
"""
EntityStructurer - LLM-based Natural Language to JSON Translator

This module provides LangChain tools for converting natural language queries
into structured ActionRequest JSON and formatting confirmation summaries for
user validation in the structured interaction workflow.
"""

import logging
import json
from typing import Dict, List, Any, Optional
from langchain_core.tools import Tool
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate

from ..models.entity import ActionRequest, Action, ActionType, EntityType

# Configure logging
logger = logging.getLogger(__name__)

# Try to import Ollama
try:
    from langchain_ollama import ChatOllama
    OLLAMA_AVAILABLE = True
except ImportError:
    logger.warning("Langchain Ollama package not available.")
    OLLAMA_AVAILABLE = False


class EntityStructurer:
    """
    LLM-based translator between natural language and ActionRequest JSON.
    
    This class provides tools for converting user queries into structured
    openBIS operations and formatting confirmation summaries.
    """
    
    def __init__(self, model: str = "qwen3"):
        """
        Initialize the EntityStructurer.
        
        Args:
            model: Ollama model to use for natural language processing
        """
        self.model = model
        self.llm = None
        
        if OLLAMA_AVAILABLE:
            try:
                self.llm = ChatOllama(model=model, temperature=0.1)
                logger.info(f"EntityStructurer initialized with model: {model}")
            except Exception as e:
                logger.error(f"Failed to initialize Ollama model {model}: {e}")
                self.llm = None
        else:
            logger.warning("Ollama not available, EntityStructurer will use fallback methods")
        
        # Set up the Pydantic output parser
        self.parser = PydanticOutputParser(pydantic_object=ActionRequest)
        
        # Create the prompt template
        self.prompt_template = PromptTemplate(
            template=self._get_structuring_prompt_template(),
            input_variables=["user_query", "conversation_history"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()}
        )
    
    def _get_structuring_prompt_template(self) -> str:
        """Get the prompt template for structuring user requests."""
        return """You are an expert at converting natural language requests into structured openBIS operations.

Your task is to analyze the user's query and convert it into a valid ActionRequest JSON that follows the openBIS entity hierarchy and operations.

## openBIS Entity Hierarchy:
- SPACE: Top-level container (e.g., /MY_LAB)
- PROJECT: Within a space (e.g., /MY_LAB/PROJECT_ALPHA)  
- EXPERIMENT: Within a project (e.g., /MY_LAB/PROJECT_ALPHA/EXP_001)
- OBJECT: Within an experiment (e.g., /MY_LAB/PROJECT_ALPHA/EXP_001/SAMPLE_001)
- DATASET: Attached to objects (files/data)

## Action Types:
- CREATE: Make new entities
- GET: Retrieve specific entities by identifier
- LIST: Get all entities of a type
- UPDATE: Modify existing entities
- DELETE: Remove entities

## Key Rules:
1. For CREATE actions:
   - Always specify the correct location/container
   - Include required fields like 'code' and 'type' in payload
   - Use proper openBIS type names (e.g., 'CHEMICAL', 'RAW_IMAGE', 'CELL_LINE')

2. For GET/UPDATE/DELETE actions:
   - Use identifier field with either permId or full identifier path

3. For relationships:
   - Use full identifier paths in parents/children arrays
   - Parents flow upward in hierarchy, children flow downward

4. Common openBIS types:
   - Objects: CHEMICAL, CELL_LINE, RAW_IMAGE, PROCESSED_IMAGE, ANTIBODY
   - Experiments: COLLECTION_EXPERIMENT, SCREENING_EXPERIMENT
   - Datasets: RAW_DATA, ANALYZED_DATA, IMAGE_ANALYSIS_DATA

## Conversation History:
{conversation_history}

## User Query:
{user_query}

## Instructions:
Analyze the user's request and create a valid ActionRequest JSON. Consider:
- What entities need to be created, retrieved, or modified?
- What is the proper hierarchy and containment?
- What properties and relationships are specified?
- Are there any implicit requirements (like creating parent containers)?

If the request is ambiguous, make reasonable assumptions based on common openBIS patterns.

{format_instructions}

ActionRequest JSON:"""
    
    def structure_user_request(self, user_query: str, conversation_history: List[Dict] = None) -> Dict[str, Any]:
        """
        Convert a natural language query into an ActionRequest JSON.
        
        Args:
            user_query: The user's natural language request
            conversation_history: Previous conversation messages for context
            
        Returns:
            Dictionary representing the ActionRequest
        """
        if not self.llm:
            # Fallback to rule-based structuring
            return self._fallback_structure_request(user_query)
        
        try:
            # Format conversation history
            history_text = ""
            if conversation_history:
                recent_messages = conversation_history[-6:]  # Last 3 exchanges
                for msg in recent_messages:
                    role = msg.get('role', 'unknown')
                    content = msg.get('content', '')
                    history_text += f"{role}: {content}\n"
            
            # Create the prompt
            prompt = self.prompt_template.format(
                user_query=user_query,
                conversation_history=history_text or "No previous conversation."
            )
            
            # Get LLM response
            response = self.llm.invoke(prompt)
            response_text = response.content.strip()
            
            # Parse the JSON response
            try:
                # Try to parse with Pydantic parser first
                action_request = self.parser.parse(response_text)
                return action_request.dict()
            except Exception as parse_error:
                logger.warning(f"Pydantic parsing failed: {parse_error}")
                # Try direct JSON parsing
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    json_text = response_text[json_start:json_end]
                    parsed_json = json.loads(json_text)
                    # Validate with Pydantic
                    action_request = ActionRequest(**parsed_json)
                    return action_request.dict()
                else:
                    raise ValueError("No valid JSON found in response")
            
        except Exception as e:
            logger.error(f"Error structuring request with LLM: {e}")
            # Fall back to rule-based approach
            return self._fallback_structure_request(user_query)
    
    def _fallback_structure_request(self, user_query: str) -> Dict[str, Any]:
        """
        Fallback rule-based structuring when LLM is not available.
        
        Args:
            user_query: The user's natural language request
            
        Returns:
            Dictionary representing a basic ActionRequest
        """
        query_lower = user_query.lower()
        
        # Simple pattern matching for common requests
        if any(word in query_lower for word in ['create', 'make', 'new']):
            # CREATE action
            if 'project' in query_lower:
                return {
                    "actions": [{
                        "action": "CREATE",
                        "entity": "PROJECT",
                        "location": {"space": "/DEFAULT_SPACE"},
                        "payload": {
                            "code": "NEW_PROJECT",
                            "type": "DEFAULT",
                            "properties": {"DESCRIPTION": "Created via chatBIS"}
                        }
                    }]
                }
            elif any(word in query_lower for word in ['sample', 'object']):
                return {
                    "actions": [{
                        "action": "CREATE", 
                        "entity": "OBJECT",
                        "location": {"experiment": "/DEFAULT_SPACE/DEFAULT_PROJECT/DEFAULT_EXPERIMENT"},
                        "payload": {
                            "code": "NEW_SAMPLE",
                            "type": "CHEMICAL",
                            "properties": {"NOTES": "Created via chatBIS"}
                        }
                    }]
                }
        
        elif any(word in query_lower for word in ['get', 'find', 'show', 'retrieve']):
            # GET action
            if 'project' in query_lower:
                return {
                    "actions": [{
                        "action": "LIST",
                        "entity": "PROJECT"
                    }]
                }
            elif any(word in query_lower for word in ['sample', 'object']):
                return {
                    "actions": [{
                        "action": "LIST",
                        "entity": "OBJECT"
                    }]
                }
        
        # Default fallback
        return {
            "actions": [{
                "action": "LIST",
                "entity": "PROJECT"
            }]
        }
    
    def format_confirmation_summary(self, action_request: Dict[str, Any]) -> str:
        """
        Create a human-readable summary of an ActionRequest for user confirmation.
        
        Args:
            action_request: Dictionary representing the ActionRequest
            
        Returns:
            Formatted markdown string for user confirmation
        """
        try:
            # Validate and parse the request
            request = ActionRequest(**action_request)
            
            summary_lines = ["## Planned Actions", ""]
            summary_lines.append("I'm ready to perform the following operations:")
            summary_lines.append("")
            
            for i, action in enumerate(request.actions, 1):
                action_desc = self._format_single_action(action, i)
                summary_lines.append(action_desc)
                summary_lines.append("")
            
            # Add warning for destructive actions
            if request.has_destructive_actions():
                summary_lines.extend([
                    "⚠️  **Warning**: This request includes operations that will modify your openBIS instance.",
                    ""
                ])
            
            # Add transaction info
            if request.transaction and len(request.actions) > 1:
                summary_lines.extend([
                    "📦 All actions will be executed as a single transaction (all succeed or all fail).",
                    ""
                ])
            
            summary_lines.extend([
                "**Please confirm:**",
                "- Type 'yes' to proceed with these actions",
                "- Type 'no' to cancel", 
                "- Describe any changes you'd like me to make"
            ])
            
            return "\n".join(summary_lines)
            
        except Exception as e:
            logger.error(f"Error formatting confirmation summary: {e}")
            return f"Error formatting confirmation: {str(e)}"
    
    def _format_single_action(self, action: Action, index: int) -> str:
        """Format a single action for the confirmation summary."""
        action_verb = {
            "CREATE": "Create",
            "GET": "Retrieve", 
            "LIST": "List",
            "UPDATE": "Update",
            "DELETE": "Delete"
        }.get(action.action, action.action)
        
        entity_name = {
            "SPACE": "Space",
            "PROJECT": "Project",
            "EXPERIMENT": "Experiment", 
            "OBJECT": "Sample/Object",
            "DATASET": "Dataset"
        }.get(action.entity, action.entity)
        
        if action.action == "CREATE":
            code = action.payload.code if action.payload else "AUTO_GENERATED"
            entity_type = action.payload.type if action.payload else "DEFAULT"
            
            location_desc = ""
            if action.location:
                if action.location.space:
                    location_desc += f" in Space `{action.location.space}`"
                if action.location.project:
                    location_desc += f" in Project `{action.location.project}`"
                if action.location.experiment:
                    location_desc += f" in Experiment `{action.location.experiment}`"
                if action.location.object:
                    location_desc += f" attached to Object `{action.location.object}`"
            
            return f"{index}. **{action_verb} {entity_name}**: `{code}` (type: `{entity_type}`){location_desc}"
        
        elif action.action in ["GET", "UPDATE", "DELETE"]:
            identifier = ""
            if action.identifier:
                if action.identifier.identifier:
                    identifier = action.identifier.identifier
                elif action.identifier.permId:
                    identifier = action.identifier.permId
            
            return f"{index}. **{action_verb} {entity_name}**: `{identifier}`"
        
        elif action.action == "LIST":
            return f"{index}. **{action_verb} all {entity_name}s**"
        
        else:
            return f"{index}. **{action_verb} {entity_name}**"
    
    def get_langchain_tools(self) -> List[Tool]:
        """
        Get LangChain Tool objects for use in the conversation engine.
        
        Returns:
            List of LangChain Tool objects
        """
        tools = []
        
        # Structure user request tool
        tools.append(Tool(
            name="structure_user_request",
            description="Convert a natural language query into a structured ActionRequest JSON for openBIS operations. Use this when the user wants to create, retrieve, update, or delete openBIS entities.",
            func=self._structure_request_tool
        ))
        
        # Format confirmation summary tool
        tools.append(Tool(
            name="format_confirmation_summary", 
            description="Create a human-readable summary of an ActionRequest for user confirmation. Takes an ActionRequest JSON and returns formatted markdown.",
            func=self._format_summary_tool
        ))
        
        return tools
    
    def _structure_request_tool(self, input_str: str) -> str:
        """Tool function for structuring user requests."""
        try:
            # Parse input - could be just the query or JSON with query and history
            try:
                input_data = json.loads(input_str)
                user_query = input_data.get("query", input_str)
                conversation_history = input_data.get("history", [])
            except json.JSONDecodeError:
                user_query = input_str
                conversation_history = []
            
            result = self.structure_user_request(user_query, conversation_history)
            return json.dumps(result, indent=2)
            
        except Exception as e:
            logger.error(f"Error in structure_request_tool: {e}")
            return f"Error structuring request: {str(e)}"
    
    def _format_summary_tool(self, input_str: str) -> str:
        """Tool function for formatting confirmation summaries."""
        try:
            # Parse the ActionRequest JSON
            action_request = json.loads(input_str)
            return self.format_confirmation_summary(action_request)
            
        except Exception as e:
            logger.error(f"Error in format_summary_tool: {e}")
            return f"Error formatting summary: {str(e)}"


# Factory function for creating EntityStructurer tools
def get_entity_structurer_tools(model: str = "qwen3") -> List[Tool]:
    """
    Factory function to create EntityStructurer tools.
    
    Args:
        model: Ollama model to use
        
    Returns:
        List of LangChain Tool objects
    """
    structurer = EntityStructurer(model=model)
    return structurer.get_langchain_tools()
