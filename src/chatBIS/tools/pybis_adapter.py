#!/usr/bin/env python3
"""
PybisAdapter - Deterministic Translator for openBIS Operations

This module provides a deterministic, code-based translator between the
ActionRequest JSON schema and pybis function calls. It handles the execution
of structured openBIS operations without using LLMs, ensuring reliable and
predictable interactions with the openBIS API.
"""

import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

from ..models.entity import ActionRequest, Action, ActionType, EntityType, Metadata

# Configure logging
logger = logging.getLogger(__name__)

# Try to import pybis
try:
    import pybis
    PYBIS_AVAILABLE = True
except ImportError:
    logger.warning("pybis package not available. PybisAdapter will be disabled.")
    PYBIS_AVAILABLE = False


class PybisAdapterError(Exception):
    """Custom exception for PybisAdapter errors."""
    pass


class PybisAdapter:
    """
    Deterministic translator between ActionRequest JSON and pybis function calls.
    
    This class provides methods to execute structured openBIS operations and
    convert pybis objects back to our standardized JSON format.
    """
    
    def __init__(self):
        """Initialize the PybisAdapter."""
        if not PYBIS_AVAILABLE:
            raise ImportError("pybis package not available")
        
        # Entity type to pybis method mapping
        self.create_methods = {
            'SPACE': 'new_space',
            'PROJECT': 'new_project', 
            'EXPERIMENT': 'new_experiment',
            'OBJECT': 'new_sample',  # Note: objects are samples in pybis
            'DATASET': 'new_dataset'
        }
        
        self.get_methods = {
            'SPACE': 'get_space',
            'PROJECT': 'get_project',
            'EXPERIMENT': 'get_experiment', 
            'OBJECT': 'get_sample',
            'DATASET': 'get_dataset'
        }
        
        self.list_methods = {
            'SPACE': 'get_spaces',
            'PROJECT': 'get_projects',
            'EXPERIMENT': 'get_experiments',
            'OBJECT': 'get_samples',
            'DATASET': 'get_datasets'
        }
    
    def execute_actions(self, request: Union[Dict, ActionRequest], pybis_instance: 'pybis.Openbis') -> List[Dict[str, Any]]:
        """
        Execute a list of actions against openBIS using transactions.
        
        Args:
            request: ActionRequest object or dictionary matching the schema
            pybis_instance: Active pybis.Openbis instance
            
        Returns:
            List of result dictionaries, one for each action
            
        Raises:
            PybisAdapterError: If execution fails
        """
        if not PYBIS_AVAILABLE:
            raise PybisAdapterError("pybis package not available")
        
        # Convert dict to ActionRequest if needed
        if isinstance(request, dict):
            try:
                request = ActionRequest(**request)
            except Exception as e:
                raise PybisAdapterError(f"Invalid ActionRequest format: {e}")
        
        results = []
        created_entities = []  # Track entities for transaction
        
        try:
            # Check if we need a transaction (any CREATE/UPDATE/DELETE operations)
            needs_transaction = request.has_destructive_actions() and request.transaction
            
            if needs_transaction:
                logger.info(f"Starting transaction for {len(request.actions)} actions")
            
            # Process each action
            for i, action in enumerate(request.actions):
                try:
                    logger.info(f"Executing action {i+1}/{len(request.actions)}: {action.action} {action.entity}")
                    
                    if action.action == "CREATE":
                        result = self._execute_create_action(action, pybis_instance, created_entities)
                    elif action.action == "GET":
                        result = self._execute_get_action(action, pybis_instance)
                    elif action.action == "LIST":
                        result = self._execute_list_action(action, pybis_instance)
                    elif action.action == "UPDATE":
                        result = self._execute_update_action(action, pybis_instance, created_entities)
                    elif action.action == "DELETE":
                        result = self._execute_delete_action(action, pybis_instance)
                    else:
                        raise PybisAdapterError(f"Unsupported action type: {action.action}")
                    
                    results.append(result)
                    
                except Exception as e:
                    error_result = {
                        "action": action.action,
                        "entity": action.entity,
                        "success": False,
                        "error": str(e),
                        "action_index": i
                    }
                    results.append(error_result)
                    
                    # If we're in a transaction and an error occurs, we should fail fast
                    if needs_transaction:
                        logger.error(f"Transaction failed at action {i+1}: {e}")
                        raise PybisAdapterError(f"Transaction failed at action {i+1}: {e}")
            
            # Commit transaction if needed
            if needs_transaction and created_entities:
                logger.info(f"Committing transaction with {len(created_entities)} entities")
                transaction = pybis_instance.new_transaction(*created_entities)
                transaction.commit()
                logger.info("Transaction committed successfully")
                
                # Update results with committed entity information
                for result in results:
                    if result.get("success") and "entity_object" in result:
                        entity_obj = result["entity_object"]
                        if hasattr(entity_obj, 'permId'):
                            result["permId"] = entity_obj.permId
                        if hasattr(entity_obj, 'identifier'):
                            result["identifier"] = entity_obj.identifier
                        # Remove the entity object from the result
                        del result["entity_object"]
            
            logger.info(f"Successfully executed {len(request.actions)} actions")
            return results
            
        except Exception as e:
            logger.error(f"Failed to execute actions: {e}")
            # Clean up any created entities that weren't committed
            for entity in created_entities:
                try:
                    if hasattr(entity, 'delete') and not getattr(entity, '_committed', False):
                        entity.delete()
                except:
                    pass  # Ignore cleanup errors
            raise PybisAdapterError(f"Action execution failed: {e}")
    
    def _execute_create_action(self, action: Action, pybis_instance: 'pybis.Openbis', created_entities: List) -> Dict[str, Any]:
        """Execute a CREATE action."""
        if not action.payload:
            raise PybisAdapterError("CREATE action requires payload")
        
        entity_type = action.entity
        method_name = self.create_methods.get(entity_type)
        
        if not method_name:
            raise PybisAdapterError(f"Unsupported entity type for CREATE: {entity_type}")
        
        create_method = getattr(pybis_instance, method_name)
        
        # Build arguments from location and payload
        kwargs = {}
        
        # Add location/containment arguments
        if action.location:
            location_dict = action.location.dict(exclude_none=True)
            kwargs.update(location_dict)
        
        # Add payload arguments
        payload_dict = action.payload.dict(exclude_none=True)
        
        # Handle special mappings
        if 'properties' in payload_dict:
            kwargs['props'] = payload_dict.pop('properties')
        
        # Add other payload fields
        for key, value in payload_dict.items():
            if key not in ['parents', 'children', 'files', 'folder']:  # Handle these separately
                kwargs[key] = value
        
        # Create the entity
        logger.debug(f"Creating {entity_type} with args: {kwargs}")
        entity = create_method(**kwargs)
        
        # Handle relationships
        if action.payload.parents:
            for parent_id in action.payload.parents:
                parent = self._resolve_entity_reference(parent_id, pybis_instance)
                entity.add_parent(parent)
        
        if action.payload.children:
            for child_id in action.payload.children:
                child = self._resolve_entity_reference(child_id, pybis_instance)
                entity.add_child(child)
        
        # For datasets, handle file uploads
        if entity_type == "DATASET":
            if action.payload.files:
                for file_path in action.payload.files:
                    entity.add_file(file_path)
            if action.payload.folder:
                entity.add_folder(action.payload.folder)
        
        # Add to transaction list
        created_entities.append(entity)
        
        return {
            "action": "CREATE",
            "entity": entity_type,
            "success": True,
            "message": f"Created {entity_type} (will be committed with transaction)",
            "entity_object": entity,  # Temporary, will be removed after commit
            "code": getattr(entity, 'code', None)
        }
    
    def _execute_get_action(self, action: Action, pybis_instance: 'pybis.Openbis') -> Dict[str, Any]:
        """Execute a GET action."""
        if not action.identifier:
            raise PybisAdapterError("GET action requires identifier")
        
        entity_type = action.entity
        method_name = self.get_methods.get(entity_type)
        
        if not method_name:
            raise PybisAdapterError(f"Unsupported entity type for GET: {entity_type}")
        
        get_method = getattr(pybis_instance, method_name)
        
        # Determine which identifier to use
        if action.identifier.permId:
            entity = get_method(action.identifier.permId)
        elif action.identifier.identifier:
            entity = get_method(action.identifier.identifier)
        else:
            raise PybisAdapterError("No valid identifier provided")
        
        if not entity:
            raise PybisAdapterError(f"{entity_type} not found")
        
        # Convert to our JSON format
        json_response = self.pybis_to_json_response(entity)
        
        return {
            "action": "GET",
            "entity": entity_type,
            "success": True,
            "data": json_response
        }
    
    def _execute_list_action(self, action: Action, pybis_instance: 'pybis.Openbis') -> Dict[str, Any]:
        """Execute a LIST action."""
        entity_type = action.entity
        method_name = self.list_methods.get(entity_type)
        
        if not method_name:
            raise PybisAdapterError(f"Unsupported entity type for LIST: {entity_type}")
        
        list_method = getattr(pybis_instance, method_name)
        
        # Execute the list operation
        entities = list_method()
        
        # Convert to our JSON format
        json_responses = [self.pybis_to_json_response(entity) for entity in entities]
        
        return {
            "action": "LIST",
            "entity": entity_type,
            "success": True,
            "count": len(json_responses),
            "data": json_responses
        }
    
    def _execute_update_action(self, action: Action, pybis_instance: 'pybis.Openbis', created_entities: List) -> Dict[str, Any]:
        """Execute an UPDATE action."""
        if not action.identifier or not action.payload:
            raise PybisAdapterError("UPDATE action requires both identifier and payload")
        
        # First get the entity
        get_action = Action(action="GET", entity=action.entity, identifier=action.identifier)
        get_result = self._execute_get_action(get_action, pybis_instance)
        
        if not get_result["success"]:
            raise PybisAdapterError(f"Could not find entity to update: {get_result.get('error', 'Unknown error')}")
        
        # Get the actual entity object for updating
        entity_type = action.entity
        method_name = self.get_methods.get(entity_type)
        get_method = getattr(pybis_instance, method_name)
        
        if action.identifier.permId:
            entity = get_method(action.identifier.permId)
        else:
            entity = get_method(action.identifier.identifier)
        
        # Apply updates from payload
        payload_dict = action.payload.dict(exclude_none=True)
        
        if 'properties' in payload_dict:
            for key, value in payload_dict['properties'].items():
                entity.set_property(key, value)
        
        if 'description' in payload_dict:
            entity.description = payload_dict['description']
        
        # Add to transaction list
        created_entities.append(entity)
        
        return {
            "action": "UPDATE",
            "entity": entity_type,
            "success": True,
            "message": f"Updated {entity_type} (will be committed with transaction)",
            "entity_object": entity
        }
    
    def _execute_delete_action(self, action: Action, pybis_instance: 'pybis.Openbis') -> Dict[str, Any]:
        """Execute a DELETE action."""
        if not action.identifier:
            raise PybisAdapterError("DELETE action requires identifier")
        
        # First get the entity to ensure it exists
        get_action = Action(action="GET", entity=action.entity, identifier=action.identifier)
        get_result = self._execute_get_action(get_action, pybis_instance)
        
        if not get_result["success"]:
            raise PybisAdapterError(f"Could not find entity to delete: {get_result.get('error', 'Unknown error')}")
        
        # Get the actual entity object for deletion
        entity_type = action.entity
        method_name = self.get_methods.get(entity_type)
        get_method = getattr(pybis_instance, method_name)
        
        if action.identifier.permId:
            entity = get_method(action.identifier.permId)
        else:
            entity = get_method(action.identifier.identifier)
        
        # Delete the entity
        entity.delete()
        
        return {
            "action": "DELETE",
            "entity": entity_type,
            "success": True,
            "message": f"Deleted {entity_type}",
            "identifier": action.identifier.dict()
        }
    
    def _resolve_entity_reference(self, reference: str, pybis_instance: 'pybis.Openbis') -> Any:
        """Resolve an entity reference (identifier or permId) to a pybis object."""
        # Try to determine entity type from identifier format
        if reference.startswith('/') and reference.count('/') >= 2:
            # This looks like a full identifier
            parts = reference.split('/')
            if len(parts) == 3:  # /SPACE/PROJECT
                return pybis_instance.get_project(reference)
            elif len(parts) == 4:  # /SPACE/PROJECT/EXPERIMENT
                return pybis_instance.get_experiment(reference)
            else:
                # Could be an object/sample
                return pybis_instance.get_sample(reference)
        else:
            # Assume it's a permId, try different entity types
            for get_method in [pybis_instance.get_sample, pybis_instance.get_experiment, 
                             pybis_instance.get_project, pybis_instance.get_dataset]:
                try:
                    entity = get_method(reference)
                    if entity:
                        return entity
                except:
                    continue
            
            raise PybisAdapterError(f"Could not resolve entity reference: {reference}")
    
    def pybis_to_json_response(self, pybis_object: Any) -> Dict[str, Any]:
        """
        Convert a pybis object to our standardized JSON Action format.
        
        Args:
            pybis_object: A pybis entity object (Space, Project, Experiment, Sample, Dataset)
            
        Returns:
            Dictionary conforming to the Action schema with action: "GET"
        """
        if not pybis_object:
            raise PybisAdapterError("Cannot convert None object to JSON")
        
        # Determine entity type
        entity_type = self._get_entity_type_from_pybis_object(pybis_object)
        
        # Build metadata
        metadata = Metadata(
            permId=getattr(pybis_object, 'permId', ''),
            identifier=getattr(pybis_object, 'identifier', ''),
            registrator=getattr(pybis_object, 'registrator', None),
            registrationDate=getattr(pybis_object, 'registrationDate', None),
            modifier=getattr(pybis_object, 'modifier', None),
            modificationDate=getattr(pybis_object, 'modificationDate', None)
        )
        
        # Build payload
        payload_data = {
            "code": getattr(pybis_object, 'code', None),
            "type": getattr(pybis_object, 'type', None),
            "properties": getattr(pybis_object, 'props', {}) or {},
            "description": getattr(pybis_object, 'description', None)
        }
        
        # Add relationships if available
        try:
            parents = getattr(pybis_object, 'parents', None)
            if parents:
                payload_data["parents"] = [p.identifier for p in parents]
        except:
            payload_data["parents"] = []
        
        try:
            children = getattr(pybis_object, 'children', None)
            if children:
                payload_data["children"] = [c.identifier for c in children]
        except:
            payload_data["children"] = []
        
        return {
            "action": "GET",
            "entity": entity_type,
            "metadata": metadata.dict(),
            "payload": payload_data
        }
    
    def _get_entity_type_from_pybis_object(self, pybis_object: Any) -> str:
        """Determine the entity type from a pybis object."""
        class_name = pybis_object.__class__.__name__.lower()
        
        if 'space' in class_name:
            return "SPACE"
        elif 'project' in class_name:
            return "PROJECT"
        elif 'experiment' in class_name:
            return "EXPERIMENT"
        elif 'sample' in class_name:
            return "OBJECT"  # Samples are objects in our schema
        elif 'dataset' in class_name:
            return "DATASET"
        else:
            # Fallback: try to determine from identifier structure
            identifier = getattr(pybis_object, 'identifier', '')
            if identifier:
                parts = identifier.split('/')
                if len(parts) == 2:
                    return "SPACE"
                elif len(parts) == 3:
                    return "PROJECT"
                elif len(parts) == 4:
                    return "EXPERIMENT"
                else:
                    return "OBJECT"
            
            raise PybisAdapterError(f"Cannot determine entity type for object: {class_name}")
