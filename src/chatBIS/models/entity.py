#!/usr/bin/env python3
"""
Entity Models for chatBIS Structured Interactions

This module defines the standardized JSON schema and Pydantic models for
handling openBIS entity operations in a structured, type-safe manner.
The models provide a canonical intermediate representation that separates
concerns between natural language understanding and deterministic API calls.
"""

from typing import List, Dict, Optional, Literal, Any, Union
from pydantic import BaseModel, Field, field_validator, model_validator

# Define the literal types for controlled vocabularies
ActionType = Literal["CREATE", "GET", "UPDATE", "LIST", "DELETE"]
EntityType = Literal["SPACE", "PROJECT", "EXPERIMENT", "OBJECT", "DATASET"]
DatasetKind = Literal["PHYSICAL", "LINK", "CONTAINER"]


class Identifier(BaseModel):
    """
    Uniquely identifies an EXISTING openBIS entity for GET, UPDATE, or DELETE actions.
    
    At least one of permId or identifier must be provided.
    """
    permId: Optional[str] = Field(None, description="The permanent, unique ID of an entity.")
    identifier: Optional[str] = Field(None, description="The full openBIS identifier string, e.g., /SPACE/PROJECT/CODE.")
    
    @model_validator(mode='after')
    def validate_identifier_fields(self):
        """Ensure at least one identifier field is provided."""
        if not self.permId and not self.identifier:
            raise ValueError("At least one of 'permId' or 'identifier' must be provided")

        return self


class Location(BaseModel):
    """
    Defines the hierarchical container for a CREATE action. The key should match
    the pybis argument for containment.
    
    The hierarchy follows openBIS structure: Space -> Project -> Experiment -> Object -> Dataset
    """
    space: Optional[str] = Field(None, description="Identifier of the parent space.")
    project: Optional[str] = Field(None, description="Identifier of the parent project.")
    experiment: Optional[str] = Field(None, description="Identifier of the parent experiment/collection.")
    object: Optional[str] = Field(None, alias="sample", description="Identifier of the parent object/sample for a dataset.")
    
    @model_validator(mode='after')
    def validate_hierarchy(self):
        """Validate that the location hierarchy makes sense."""
        # If project is specified, space should be too (or derivable)
        if self.project and not self.space:
            # Try to extract space from project identifier
            if self.project.startswith('/') and self.project.count('/') >= 2:
                space_part = self.project.split('/')[1]
                self.space = f"/{space_part}"

        # If experiment is specified, project should be too (or derivable)
        if self.experiment and not self.project:
            if self.experiment.startswith('/') and self.experiment.count('/') >= 3:
                parts = self.experiment.split('/')
                self.project = f"/{parts[1]}/{parts[2]}"
                if not self.space:
                    self.space = f"/{parts[1]}"

        return self


class Payload(BaseModel):
    """
    Contains the data to be written to or read from openBIS.
    
    This model accommodates all entity types and their specific requirements.
    """
    code: Optional[str] = Field(None, description="The code for the new entity. Can be omitted if auto-generated.")
    type: Optional[str] = Field(None, description="The openBIS type of the entity (e.g., 'CHEMICAL', 'RAW_IMAGE').")
    properties: Optional[Dict[str, Any]] = Field(default_factory=dict, description="A dictionary of property key-value pairs.")
    
    # Lineage relationships
    parents: Optional[List[str]] = Field(default_factory=list, description="A list of parent identifiers for lineage.")
    children: Optional[List[str]] = Field(default_factory=list, description="A list of child identifiers for lineage.")
    
    # Dataset-specific fields
    kind: Optional[DatasetKind] = Field(None, description="The kind of dataset (PHYSICAL, LINK, CONTAINER).")
    files: Optional[List[str]] = Field(None, description="A list of local file paths to upload for a dataset.")
    folder: Optional[str] = Field(None, description="A local folder path to upload for a dataset.")
    
    # Additional metadata fields
    description: Optional[str] = Field(None, description="Description of the entity.")
    tags: Optional[List[str]] = Field(default_factory=list, description="List of tags associated with the entity.")


class Metadata(BaseModel):
    """
    Read-only metadata retrieved from openBIS during a GET action.
    
    This represents the system-generated information that cannot be directly modified.
    """
    permId: str = Field(..., description="The permanent ID of the entity.")
    identifier: str = Field(..., description="The full identifier of the entity.")
    registrator: Optional[str] = Field(None, description="Username of the person who registered the entity.")
    registrationDate: Optional[str] = Field(None, description="ISO format date when the entity was registered.")
    modifier: Optional[str] = Field(None, description="Username of the person who last modified the entity.")
    modificationDate: Optional[str] = Field(None, description="ISO format date when the entity was last modified.")
    
    # Additional system metadata
    frozen: Optional[bool] = Field(None, description="Whether the entity is frozen (read-only).")
    frozenForChildren: Optional[bool] = Field(None, description="Whether children of this entity are frozen.")
    frozenForParents: Optional[bool] = Field(None, description="Whether parents of this entity are frozen.")
    frozenForDataSets: Optional[bool] = Field(None, description="Whether datasets of this entity are frozen.")


class Action(BaseModel):
    """
    A single, self-contained action to be performed on an openBIS entity.
    
    Each action is atomic and represents one operation (CREATE, GET, UPDATE, LIST, DELETE)
    on one entity type (SPACE, PROJECT, EXPERIMENT, OBJECT, DATASET).
    """
    action: ActionType = Field(..., description="The type of action to perform.")
    entity: EntityType = Field(..., description="The type of openBIS entity to operate on.")
    
    # For existing entities (GET, UPDATE, DELETE)
    identifier: Optional[Identifier] = Field(None, description="Identifier for an existing entity (for GET/UPDATE/DELETE).")
    
    # For new entities (CREATE)
    location: Optional[Location] = Field(None, description="The location/container for a new entity (for CREATE).")
    
    # Data payload
    payload: Optional[Payload] = Field(None, description="The data associated with the action.")
    
    # Response metadata (populated after execution)
    metadata: Optional[Metadata] = Field(None, description="Read-only metadata returned from openBIS (for GET).")
    
    # Execution result
    result: Optional[Dict[str, Any]] = Field(None, description="Result of the action execution.")
    error: Optional[str] = Field(None, description="Error message if the action failed.")
    
    @model_validator(mode='after')
    def validate_action_requirements(self):
        """Validate that required fields are present for each action type."""
        if self.action in ['GET', 'UPDATE', 'DELETE']:
            if not self.identifier:
                raise ValueError(f"{self.action} actions require an 'identifier' field")

        if self.action == 'CREATE':
            if not self.location and self.entity != 'SPACE':  # Spaces don't need a location
                raise ValueError(f"CREATE actions for {self.entity} require a 'location' field")
            if not self.payload:
                raise ValueError("CREATE actions require a 'payload' field")

        if self.action == 'UPDATE':
            if not self.payload:
                raise ValueError("UPDATE actions require a 'payload' field")

        return self


class ActionRequest(BaseModel):
    """
    The root object for a request, containing one or more actions.
    
    This represents a complete interaction request that can contain multiple
    related actions to be executed as a transaction.
    """
    actions: List[Action] = Field(..., min_items=1, description="List of actions to be performed.")
    
    # Request metadata
    request_id: Optional[str] = Field(None, description="Unique identifier for this request.")
    description: Optional[str] = Field(None, description="Human-readable description of what this request does.")
    
    # Execution options
    dry_run: Optional[bool] = Field(False, description="If true, validate but don't execute the actions.")
    transaction: Optional[bool] = Field(True, description="If true, execute all actions in a single transaction.")
    
    @field_validator('actions')
    @classmethod
    def validate_actions_consistency(cls, actions):
        """Validate that actions are consistent and can be executed together."""
        if not actions:
            raise ValueError("At least one action must be provided")

        # Check for conflicting actions on the same entity
        entity_actions = {}
        for action in actions:
            key = f"{action.entity}:{action.identifier or action.location}"
            if key in entity_actions:
                existing_action = entity_actions[key]
                if existing_action.action == 'DELETE' or action.action == 'DELETE':
                    raise ValueError("Cannot combine DELETE with other actions on the same entity")
            entity_actions[key] = action

        return actions
    
    def get_actions_by_type(self, action_type: ActionType) -> List[Action]:
        """Get all actions of a specific type."""
        return [action for action in self.actions if action.action == action_type]
    
    def get_actions_by_entity(self, entity_type: EntityType) -> List[Action]:
        """Get all actions for a specific entity type."""
        return [action for action in self.actions if action.entity == entity_type]
    
    def has_destructive_actions(self) -> bool:
        """Check if the request contains any destructive actions (CREATE, UPDATE, DELETE)."""
        destructive_actions = {'CREATE', 'UPDATE', 'DELETE'}
        return any(action.action in destructive_actions for action in self.actions)
    
    def summary(self) -> str:
        """Generate a human-readable summary of the request."""
        if self.description:
            return self.description
        
        action_counts = {}
        for action in self.actions:
            key = f"{action.action} {action.entity}"
            action_counts[key] = action_counts.get(key, 0) + 1
        
        summary_parts = []
        for action_desc, count in action_counts.items():
            if count == 1:
                summary_parts.append(action_desc)
            else:
                summary_parts.append(f"{count}x {action_desc}")
        
        return f"Request with {len(summary_parts)} action types: {', '.join(summary_parts)}"
