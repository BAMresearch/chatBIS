#!/usr/bin/env python3
"""
Comprehensive PyBIS Tools for chatBIS

This module provides LangChain Tool wrappers for all major pybis functions,
enabling the chatbot to execute a wide range of actions on openBIS instances.

Based on pybis v1.37.3 documentation from:
- https://pypi.org/project/pybis/
- https://openbis.readthedocs.io/en/latest/software-developer-documentation/apis/python-v3-api.html
"""

import logging
import os
import re
from typing import Dict, List, Any
from datetime import datetime
from langchain_core.tools import Tool

# Check if pandas is available for advanced date filtering
try:
    import importlib.util
    PANDAS_AVAILABLE = importlib.util.find_spec("pandas") is not None
except ImportError:
    PANDAS_AVAILABLE = False

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not available, continue without it

logger = logging.getLogger(__name__)

# Try to import pybis
try:
    import pybis
    PYBIS_AVAILABLE = True
except ImportError:
    logger.warning("pybis package not available. Function calling will be disabled.")
    PYBIS_AVAILABLE = False


class PyBISConnection:
    """Manages pybis connection state."""

    def __init__(self):
        self.openbis = None
        self.is_connected = False
        self.server_url = None
        self.username = None

    def connect(self, server_url: str, username: str, password: str, verify_certificates: bool = True) -> bool:
        """Connect to openBIS server."""
        if not PYBIS_AVAILABLE:
            raise ImportError("pybis package not available")

        try:
            self.openbis = pybis.Openbis(server_url, verify_certificates=verify_certificates)
            self.openbis.login(username, password)
            self.is_connected = True
            self.server_url = server_url
            self.username = username
            logger.info(f"Successfully connected to openBIS at {server_url} as {username}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to openBIS: {e}")
            self.is_connected = False
            return False

    def disconnect(self):
        """Disconnect from openBIS server."""
        if self.openbis and self.is_connected:
            try:
                self.openbis.logout()
                self.is_connected = False
                logger.info("Disconnected from openBIS")
            except Exception as e:
                logger.error(f"Error during disconnect: {e}")


# Global connection instance
_connection = PyBISConnection()


class PyBISToolManager:
    """Manages pybis tools and connection state."""

    def __init__(self):
        self.connection = _connection
        self.tools = self._create_tools()
        self._auto_connect_attempted = False

    def connect(self, server_url: str, username: str, password: str, verify_certificates: bool = True) -> bool:
        """Connect to openBIS server."""
        return self.connection.connect(server_url, username, password, verify_certificates)

    def disconnect(self):
        """Disconnect from openBIS server."""
        self.connection.disconnect()

    def is_connected(self) -> bool:
        """Check if connected to openBIS."""
        return self.connection.is_connected

    def get_tools(self) -> List[Tool]:
        """Get list of available tools."""
        return self.tools

    def _create_tools(self) -> List[Tool]:
        """Create comprehensive LangChain Tool objects for all major pybis functions."""
        tools = []

        # === CONNECTION MANAGEMENT ===
        tools.append(Tool(
            name="connect_to_openbis",
            description="Connect to an openBIS server. Required before using any other openBIS functions. Parameters: server_url (string, e.g. 'https://demo.openbis.ch'), username (string), password (string), verify_certificates (boolean, default True)",
            func=self._connect_tool
        ))

        tools.append(Tool(
            name="disconnect_from_openbis",
            description="Disconnect from the openBIS server and clean up the session.",
            func=self._disconnect_tool
        ))

        tools.append(Tool(
            name="check_openbis_connection",
            description="Check if currently connected to openBIS server and show connection details.",
            func=self._check_connection_tool
        ))

        # === SPACE MANAGEMENT ===
        tools.append(Tool(
            name="list_spaces",
            description="List all spaces in openBIS. Spaces are used for authorization and to separate working groups.",
            func=self._list_spaces_tool
        ))

        tools.append(Tool(
            name="get_space",
            description="Get details of a specific space by code. Parameters: space_code (string)",
            func=self._get_space_tool
        ))

        tools.append(Tool(
            name="create_space",
            description="Create a new space in openBIS. Parameters: space_code (string), description (string, optional)",
            func=self._create_space_tool
        ))

        # === PROJECT MANAGEMENT ===
        tools.append(Tool(
            name="list_projects",
            description="List projects in the user's space in openBIS. Projects live within spaces and contain experiments. Automatically filters by the user's own space unless space parameter is explicitly provided. Supports date filtering: 'projects in February 2024', 'projects from 2023', 'projects last month', etc.",
            func=self._list_projects_tool
        ))

        tools.append(Tool(
            name="get_project",
            description="Get details of a specific project by identifier. Parameters: project_identifier (string, format: '/SPACE/PROJECT')",
            func=self._get_project_tool
        ))

        tools.append(Tool(
            name="create_project",
            description="Create a new project in the user's space in openBIS. Parameters: code (string), description (string, optional). Space defaults to user's own space.",
            func=self._create_project_tool
        ))

        # === EXPERIMENT/COLLECTION MANAGEMENT ===
        tools.append(Tool(
            name="list_experiments",
            description="List ALL experiments (collections) in the user's space in openBIS. Shows total count and all experiments by default. Automatically filters by the user's own space unless space parameter is explicitly provided. Optional parameters: project (string), experiment_type (string), limit (integer, only use when user asks for 'last N' or 'recent N' experiments). Supports date filtering: 'experiments in February 2024', 'experiments from 2023', 'experiments last month', etc.",
            func=self._list_experiments_tool
        ))

        tools.append(Tool(
            name="get_experiment",
            description="Get details of a specific experiment by identifier. Parameters: experiment_identifier (string, format: '/SPACE/PROJECT/EXPERIMENT')",
            func=self._get_experiment_tool
        ))

        tools.append(Tool(
            name="create_experiment",
            description="Create a new experiment in openBIS. Parameters: experiment_type (string), project (string, format: '/SPACE/PROJECT'), code (string), properties (dict, optional)",
            func=self._create_experiment_tool
        ))

        # === SAMPLE/OBJECT MANAGEMENT ===
        tools.append(Tool(
            name="list_samples",
            description="List ALL samples (objects) in the user's space in openBIS (basic info only - identifier, type, registration date). Shows total count and all samples by default. Automatically filters by the user's own space unless space parameter is explicitly provided. Optional parameters: sample_type (string), project (string), experiment (string), limit (integer, only use when user asks for 'last N' or 'recent N' samples). Supports date filtering: 'samples in February 2024', 'samples from 2023', 'samples last month', etc.",
            func=self._list_samples_tool
        ))

        tools.append(Tool(
            name="get_sample",
            description="Get details of a specific sample by identifier. Parameters: sample_identifier (string, format: '/SPACE/SAMPLE_CODE' or permId)",
            func=self._get_sample_tool
        ))

        tools.append(Tool(
            name="create_sample",
            description="Create a new sample in the user's space in openBIS. Parameters: sample_type (string), code (string), experiment (string, optional), properties (dict, optional). Space defaults to user's own space.",
            func=self._create_sample_tool
        ))

        tools.append(Tool(
            name="update_sample",
            description="Update an existing sample's properties. Parameters: sample_identifier (string), properties (dict)",
            func=self._update_sample_tool
        ))

        tools.append(Tool(
            name="list_samples_detailed",
            description="List ALL samples with detailed information including properties and registration dates. Shows ALL samples by default unless user specifies a limit (e.g., 'show me 20 samples'). Use this when user specifically asks for 'properties', 'detailed info', or 'all information'. Supports date filtering: 'samples in February 2024', 'samples from 2023', etc. Optional parameters: sample_type (string), project (string), experiment (string), limit (integer, only when user specifies), show_properties (boolean, default True).",
            func=self._list_samples_detailed_tool
        ))

        # === DATASET MANAGEMENT ===
        tools.append(Tool(
            name="list_datasets",
            description="List datasets in openBIS. Datasets contain the actual data files. Optional parameters: dataset_type (string), sample (string), experiment (string), limit (integer, default 20)",
            func=self._list_datasets_tool
        ))

        tools.append(Tool(
            name="get_dataset",
            description="Get details of a specific dataset by identifier. Parameters: dataset_identifier (string, permId or code)",
            func=self._get_dataset_tool
        ))

        tools.append(Tool(
            name="create_dataset",
            description="Create a new dataset in openBIS. Parameters: dataset_type (string), sample (string, optional), experiment (string, optional), files (list, optional), properties (dict, optional)",
            func=self._create_dataset_tool
        ))

        # === MASTERDATA MANAGEMENT ===
        tools.append(Tool(
            name="list_sample_types",
            description="List all sample types (object types) in openBIS. Sample types define the structure and properties of samples.",
            func=self._list_sample_types_tool
        ))

        tools.append(Tool(
            name="get_sample_type",
            description="Get details of a specific sample type. Parameters: sample_type_code (string)",
            func=self._get_sample_type_tool
        ))

        tools.append(Tool(
            name="list_experiment_types",
            description="List all experiment types (collection types) in openBIS. Experiment types define the structure of experiments.",
            func=self._list_experiment_types_tool
        ))

        tools.append(Tool(
            name="list_dataset_types",
            description="List all dataset types in openBIS. Dataset types define the structure and properties of datasets.",
            func=self._list_dataset_types_tool
        ))

        tools.append(Tool(
            name="list_property_types",
            description="List all property types in openBIS. Property types define the data types and constraints for entity properties.",
            func=self._list_property_types_tool
        ))

        tools.append(Tool(
            name="list_vocabularies",
            description="List all controlled vocabularies in openBIS. Vocabularies define allowed values for certain properties.",
            func=self._list_vocabularies_tool
        ))

        tools.append(Tool(
            name="get_vocabulary",
            description="Get details of a specific vocabulary and its terms. Parameters: vocabulary_code (string)",
            func=self._get_vocabulary_tool
        ))

        # === ADDITIONAL PYBIS TOOLS FROM API ===
        # Add comprehensive tools for all pybis functions from pybis_api_slim.json

        # Session-level Openbis methods
        tools.append(Tool(
            name="openbis_get_server_information",
            description="Get server information from openBIS. No parameters required.",
            func=self._openbis_get_server_information_tool
        ))

        tools.append(Tool(
            name="openbis_get_session_info",
            description="Get current session information. No parameters required.",
            func=self._openbis_get_session_info_tool
        ))

        tools.append(Tool(
            name="openbis_is_session_active",
            description="Check if current session is active. No parameters required.",
            func=self._openbis_is_session_active_tool
        ))

        tools.append(Tool(
            name="openbis_create_permid",
            description="Generate a new permanent ID from the server. No parameters required.",
            func=self._openbis_create_permid_tool
        ))

        tools.append(Tool(
            name="openbis_get_datastores",
            description="Get all available data stores. No parameters required.",
            func=self._openbis_get_datastores_tool
        ))

        tools.append(Tool(
            name="openbis_get_plugins",
            description="Get all available plugins. Optional parameters: plugin_type (string).",
            func=self._openbis_get_plugins_tool
        ))

        tools.append(Tool(
            name="openbis_get_plugin",
            description="Get details of a specific plugin. Parameters: plugin_name (string).",
            func=self._openbis_get_plugin_tool
        ))

        tools.append(Tool(
            name="openbis_get_external_data_management_systems",
            description="Get all external data management systems. No parameters required.",
            func=self._openbis_get_external_data_management_systems_tool
        ))

        tools.append(Tool(
            name="openbis_get_external_data_management_system",
            description="Get details of a specific external DMS. Parameters: dms_id (string).",
            func=self._openbis_get_external_data_management_system_tool
        ))

        tools.append(Tool(
            name="openbis_get_persons",
            description="Get all persons (users) in openBIS. No parameters required.",
            func=self._openbis_get_persons_tool
        ))

        tools.append(Tool(
            name="openbis_get_person",
            description="Get details of a specific person. Parameters: person_id (string).",
            func=self._openbis_get_person_tool
        ))

        tools.append(Tool(
            name="openbis_get_groups",
            description="Get all groups in openBIS. No parameters required.",
            func=self._openbis_get_groups_tool
        ))

        tools.append(Tool(
            name="openbis_get_group",
            description="Get details of a specific group. Parameters: group_code (string).",
            func=self._openbis_get_group_tool
        ))

        tools.append(Tool(
            name="openbis_get_role_assignments",
            description="Get all role assignments. Optional parameters: person (string), group (string), space (string), project (string).",
            func=self._openbis_get_role_assignments_tool
        ))

        tools.append(Tool(
            name="openbis_get_tags",
            description="Get all tags in openBIS. No parameters required.",
            func=self._openbis_get_tags_tool
        ))

        tools.append(Tool(
            name="openbis_get_tag",
            description="Get details of a specific tag. Parameters: tag_code (string).",
            func=self._openbis_get_tag_tool
        ))

        # Entity-specific methods
        tools.append(Tool(
            name="sample_delete",
            description="Delete a sample. Parameters: identifier (string, sample identifier), reason (string), permanently (boolean, default False).",
            func=self._sample_delete_tool
        ))

        tools.append(Tool(
            name="sample_get_datasets",
            description="Get datasets associated with a sample. Parameters: identifier (string, sample identifier).",
            func=self._sample_get_datasets_tool
        ))

        tools.append(Tool(
            name="sample_get_projects",
            description="Get projects associated with a sample. Parameters: identifier (string, sample identifier).",
            func=self._sample_get_projects_tool
        ))

        tools.append(Tool(
            name="sample_is_marked_to_be_deleted",
            description="Check if a sample is marked for deletion. Parameters: identifier (string, sample identifier).",
            func=self._sample_is_marked_to_be_deleted_tool
        ))

        tools.append(Tool(
            name="sample_mark_to_be_deleted",
            description="Mark a sample for deletion. Parameters: identifier (string, sample identifier).",
            func=self._sample_mark_to_be_deleted_tool
        ))

        tools.append(Tool(
            name="sample_unmark_to_be_deleted",
            description="Unmark a sample for deletion. Parameters: identifier (string, sample identifier).",
            func=self._sample_unmark_to_be_deleted_tool
        ))

        tools.append(Tool(
            name="sample_save",
            description="Save changes to a sample. Parameters: identifier (string, sample identifier).",
            func=self._sample_save_tool
        ))

        tools.append(Tool(
            name="sample_set_properties",
            description="Set properties on a sample. Parameters: identifier (string, sample identifier), properties (dict of property names and values).",
            func=self._sample_set_properties_tool
        ))

        # Dataset methods
        tools.append(Tool(
            name="dataset_delete",
            description="Delete a dataset. Parameters: identifier (string, dataset permId), reason (string), permanently (boolean, default False).",
            func=self._dataset_delete_tool
        ))

        tools.append(Tool(
            name="dataset_download",
            description="Download dataset files. Parameters: identifier (string, dataset permId), files (list, optional), destination (string, optional), create_default_folders (boolean, default True), wait_until_finished (boolean, default True).",
            func=self._dataset_download_tool
        ))

        tools.append(Tool(
            name="dataset_get_file_list",
            description="Get list of files in a dataset. Parameters: identifier (string, dataset permId), recursive (boolean, default True), start_folder (string, default '/').",
            func=self._dataset_get_file_list_tool
        ))

        tools.append(Tool(
            name="dataset_get_files",
            description="Get DataFrame of all files in a dataset. Parameters: identifier (string, dataset permId), start_folder (string, default '/').",
            func=self._dataset_get_files_tool
        ))

        tools.append(Tool(
            name="dataset_is_marked_to_be_deleted",
            description="Check if a dataset is marked for deletion. Parameters: identifier (string, dataset permId).",
            func=self._dataset_is_marked_to_be_deleted_tool
        ))

        tools.append(Tool(
            name="dataset_mark_to_be_deleted",
            description="Mark a dataset for deletion. Parameters: identifier (string, dataset permId).",
            func=self._dataset_mark_to_be_deleted_tool
        ))

        tools.append(Tool(
            name="dataset_unmark_to_be_deleted",
            description="Unmark a dataset for deletion. Parameters: identifier (string, dataset permId).",
            func=self._dataset_unmark_to_be_deleted_tool
        ))

        tools.append(Tool(
            name="dataset_save",
            description="Save changes to a dataset. Parameters: identifier (string, dataset permId).",
            func=self._dataset_save_tool
        ))

        tools.append(Tool(
            name="dataset_set_properties",
            description="Set properties on a dataset. Parameters: identifier (string, dataset permId), properties (dict of property names and values).",
            func=self._dataset_set_properties_tool
        ))

        tools.append(Tool(
            name="dataset_archive",
            description="Archive a dataset. Parameters: identifier (string, dataset permId), remove_from_data_store (boolean, default True).",
            func=self._dataset_archive_tool
        ))

        tools.append(Tool(
            name="dataset_unarchive",
            description="Unarchive a dataset. Parameters: identifier (string, dataset permId).",
            func=self._dataset_unarchive_tool
        ))

        # Experiment methods
        tools.append(Tool(
            name="experiment_delete",
            description="Delete an experiment. Parameters: identifier (string, experiment identifier), reason (string), permanently (boolean, default False).",
            func=self._experiment_delete_tool
        ))

        tools.append(Tool(
            name="experiment_get_datasets",
            description="Get datasets associated with an experiment. Parameters: identifier (string, experiment identifier).",
            func=self._experiment_get_datasets_tool
        ))

        tools.append(Tool(
            name="experiment_get_samples",
            description="Get samples associated with an experiment. Parameters: identifier (string, experiment identifier).",
            func=self._experiment_get_samples_tool
        ))

        tools.append(Tool(
            name="experiment_get_projects",
            description="Get projects associated with an experiment. Parameters: identifier (string, experiment identifier).",
            func=self._experiment_get_projects_tool
        ))

        tools.append(Tool(
            name="experiment_add_samples",
            description="Add samples to an experiment. Parameters: identifier (string, experiment identifier), samples (list of sample identifiers).",
            func=self._experiment_add_samples_tool
        ))

        tools.append(Tool(
            name="experiment_del_samples",
            description="Remove samples from an experiment. Parameters: identifier (string, experiment identifier), samples (list of sample identifiers).",
            func=self._experiment_del_samples_tool
        ))

        tools.append(Tool(
            name="experiment_is_marked_to_be_deleted",
            description="Check if an experiment is marked for deletion. Parameters: identifier (string, experiment identifier).",
            func=self._experiment_is_marked_to_be_deleted_tool
        ))

        tools.append(Tool(
            name="experiment_mark_to_be_deleted",
            description="Mark an experiment for deletion. Parameters: identifier (string, experiment identifier).",
            func=self._experiment_mark_to_be_deleted_tool
        ))

        tools.append(Tool(
            name="experiment_unmark_to_be_deleted",
            description="Unmark an experiment for deletion. Parameters: identifier (string, experiment identifier).",
            func=self._experiment_unmark_to_be_deleted_tool
        ))

        tools.append(Tool(
            name="experiment_save",
            description="Save changes to an experiment. Parameters: identifier (string, experiment identifier).",
            func=self._experiment_save_tool
        ))

        tools.append(Tool(
            name="experiment_set_properties",
            description="Set properties on an experiment. Parameters: identifier (string, experiment identifier), properties (dict of property names and values).",
            func=self._experiment_set_properties_tool
        ))

        # Project methods
        tools.append(Tool(
            name="project_delete",
            description="Delete a project. Parameters: identifier (string, project identifier), reason (string), permanently (boolean, default False).",
            func=self._project_delete_tool
        ))

        tools.append(Tool(
            name="project_get_experiments",
            description="Get experiments in a project. Parameters: identifier (string, project identifier).",
            func=self._project_get_experiments_tool
        ))

        tools.append(Tool(
            name="project_get_datasets",
            description="Get datasets in a project. Parameters: identifier (string, project identifier).",
            func=self._project_get_datasets_tool
        ))

        tools.append(Tool(
            name="project_get_samples",
            description="Get samples in a project. Parameters: identifier (string, project identifier).",
            func=self._project_get_samples_tool
        ))

        tools.append(Tool(
            name="project_get_sample",
            description="Get a specific sample in a project. Parameters: identifier (string, project identifier), sample_code (string).",
            func=self._project_get_sample_tool
        ))

        tools.append(Tool(
            name="project_is_marked_to_be_deleted",
            description="Check if a project is marked for deletion. Parameters: identifier (string, project identifier).",
            func=self._project_is_marked_to_be_deleted_tool
        ))

        tools.append(Tool(
            name="project_mark_to_be_deleted",
            description="Mark a project for deletion. Parameters: identifier (string, project identifier).",
            func=self._project_mark_to_be_deleted_tool
        ))

        tools.append(Tool(
            name="project_unmark_to_be_deleted",
            description="Unmark a project for deletion. Parameters: identifier (string, project identifier).",
            func=self._project_unmark_to_be_deleted_tool
        ))

        tools.append(Tool(
            name="project_save",
            description="Save changes to a project. Parameters: identifier (string, project identifier).",
            func=self._project_save_tool
        ))

        # Space methods
        tools.append(Tool(
            name="space_delete",
            description="Delete a space. Parameters: space_code (string), reason (string), permanently (boolean, default False).",
            func=self._space_delete_tool
        ))

        tools.append(Tool(
            name="space_get_experiments",
            description="Get experiments in a space. Parameters: space_code (string).",
            func=self._space_get_experiments_tool
        ))

        tools.append(Tool(
            name="space_get_samples",
            description="Get samples in a space. Parameters: space_code (string).",
            func=self._space_get_samples_tool
        ))

        tools.append(Tool(
            name="space_get_projects",
            description="Get projects in a space. Parameters: space_code (string).",
            func=self._space_get_projects_tool
        ))

        tools.append(Tool(
            name="space_is_marked_to_be_deleted",
            description="Check if a space is marked for deletion. Parameters: space_code (string).",
            func=self._space_is_marked_to_be_deleted_tool
        ))

        tools.append(Tool(
            name="space_mark_to_be_deleted",
            description="Mark a space for deletion. Parameters: space_code (string).",
            func=self._space_mark_to_be_deleted_tool
        ))

        tools.append(Tool(
            name="space_unmark_to_be_deleted",
            description="Unmark a space for deletion. Parameters: space_code (string).",
            func=self._space_unmark_to_be_deleted_tool
        ))

        tools.append(Tool(
            name="space_save",
            description="Save changes to a space. Parameters: space_code (string).",
            func=self._space_save_tool
        ))

        return tools

    def _auto_connect_from_env(self):
        """Attempt to auto-connect using environment variables."""
        if self._auto_connect_attempted:
            return False

        self._auto_connect_attempted = True

        # Get credentials from environment variables
        server_url = os.getenv('OPENBIS_URL')
        username = os.getenv('OPENBIS_USERNAME')
        password = os.getenv('OPENBIS_PASSWORD')

        if not all([server_url, username, password]):
            logger.warning("openBIS credentials not found in environment variables. "
                         "Set OPENBIS_URL, OPENBIS_USERNAME, and OPENBIS_PASSWORD to enable auto-connection.")
            return False

        logger.info(f"Attempting auto-connection to openBIS at {server_url} as {username}")
        success = self.connection.connect(server_url, username, password, verify_certificates=True)

        if success:
            logger.info("Auto-connection to openBIS successful")
        else:
            logger.error("Auto-connection to openBIS failed")

        return success

    def _get_user_space(self):
        """Get the user's space name from the username (uppercase)."""
        username = os.getenv('OPENBIS_USERNAME')
        if username:
            return username.upper()
        return None

    def _ensure_connected(self):
        """Ensure we're connected to openBIS, attempting auto-connection if needed."""
        if not self.connection.is_connected:
            # Try auto-connection first
            if self._auto_connect_from_env():
                return

            raise ConnectionError("Not connected to openBIS. Please connect first using connect_to_openbis.")

    # === CONNECTION MANAGEMENT TOOLS ===

    def _connect_tool(self, input_str: str) -> str:
        """Tool function for connecting to openBIS."""
        try:
            # Parse input - expecting format like "server_url=..., username=..., password=..."
            params = self._parse_tool_input(input_str)

            server_url = params.get('server_url')
            username = params.get('username')
            password = params.get('password')
            verify_certificates = params.get('verify_certificates', True)

            if not all([server_url, username, password]):
                return "Error: Missing required parameters. Need server_url, username, and password."

            success = self.connection.connect(server_url, username, password, verify_certificates)
            if success:
                return f"Successfully connected to openBIS at {server_url} as {username}"
            else:
                return "Failed to connect to openBIS. Please check your credentials and server URL."

        except Exception as e:
            return f"Error connecting to openBIS: {str(e)}"

    def _disconnect_tool(self, input_str: str = "") -> str:
        """Tool function for disconnecting from openBIS."""
        try:
            self.connection.disconnect()
            return "Disconnected from openBIS"
        except Exception as e:
            return f"Error disconnecting: {str(e)}"

    def _check_connection_tool(self, input_str: str = "") -> str:
        """Tool function for checking connection status."""
        if self.connection.is_connected:
            return f"Connected to openBIS at {self.connection.server_url} as {self.connection.username}"
        else:
            return "Not connected to openBIS"

    # === SPACE MANAGEMENT TOOLS ===

    def _list_spaces_tool(self, input_str: str = "") -> str:
        """Tool function for listing spaces."""
        try:
            self._ensure_connected()

            spaces = self.connection.openbis.get_spaces()

            if len(spaces) == 0:
                return "No spaces found."

            result = f"Found {len(spaces)} spaces:\n"
            for idx, space in enumerate(spaces):
                result += f"{idx+1}. {space.code}"
                if hasattr(space, 'description') and space.description:
                    result += f" - {space.description}"
                result += "\n"

            return result

        except Exception as e:
            return f"Error listing spaces: {str(e)}"

    def _get_space_tool(self, input_str: str) -> str:
        """Tool function for getting space details."""
        try:
            self._ensure_connected()
            params = self._parse_tool_input(input_str)

            space_code = params.get('space_code')
            if not space_code:
                return "Error: space_code parameter is required."

            space = self.connection.openbis.get_space(space_code)

            if space is None:
                return f"Space '{space_code}' not found."

            # Format space information
            result = f"Space: {space.code}\n"
            if hasattr(space, 'description') and space.description:
                result += f"Description: {space.description}\n"
            if hasattr(space, 'registrator'):
                result += f"Registrator: {space.registrator}\n"
            if hasattr(space, 'registrationDate'):
                result += f"Registration Date: {space.registrationDate}\n"

            return result

        except Exception as e:
            return f"Error getting space: {str(e)}"

    def _create_space_tool(self, input_str: str) -> str:
        """Tool function for creating a space."""
        try:
            self._ensure_connected()
            params = self._parse_tool_input(input_str)

            space_code = params.get('space_code')
            description = params.get('description', '')

            if not space_code:
                return "Error: space_code parameter is required."

            # Create space
            space = self.connection.openbis.new_space(
                code=space_code,
                description=description
            )

            space.save()

            return f"Successfully created space: {space.code}"

        except Exception as e:
            return f"Error creating space: {str(e)}"

    # === PROJECT MANAGEMENT TOOLS ===

    def _list_projects_tool(self, input_str: str) -> str:
        """Tool function for listing projects."""
        try:
            self._ensure_connected()
            params = self._parse_tool_input(input_str)

            # Always use the user's own space unless explicitly specified
            space = params.get('space')
            if not space:
                space = self._get_user_space()

            projects = self.connection.openbis.get_projects(space=space)

            # Apply date filtering if specified
            date_filters = {k: v for k, v in params.items() if k in ['year', 'month']}
            if date_filters:
                projects = self._filter_by_date(projects, date_filters)

            if len(projects) == 0:
                filter_desc = ""
                if date_filters:
                    if 'month' in date_filters and 'year' in date_filters:
                        month_names = ['', 'January', 'February', 'March', 'April', 'May', 'June',
                                     'July', 'August', 'September', 'October', 'November', 'December']
                        filter_desc = f" from {month_names[date_filters['month']]} {date_filters['year']}"
                    elif 'year' in date_filters:
                        filter_desc = f" from {date_filters['year']}"
                return f"No projects found{filter_desc}{' in space ' + space if space else ''}."

            total_count = len(projects)
            filter_desc = ""
            if date_filters:
                if 'month' in date_filters and 'year' in date_filters:
                    month_names = ['', 'January', 'February', 'March', 'April', 'May', 'June',
                                 'July', 'August', 'September', 'October', 'November', 'December']
                    filter_desc = f" from {month_names[date_filters['month']]} {date_filters['year']}"
                elif 'year' in date_filters:
                    filter_desc = f" from {date_filters['year']}"

            result = f"Found {total_count} projects{filter_desc}{' in space ' + space if space else ''}:\n"
            for idx, project in enumerate(projects):
                result += f"{idx+1}. {project.identifier}"
                if hasattr(project, 'description') and project.description:
                    result += f" - {project.description}"
                # Add registration date if filtering by date
                if date_filters and hasattr(project, 'registrationDate'):
                    result += f" (registered: {project.registrationDate})"
                result += "\n"

            return result

        except Exception as e:
            return f"Error listing projects: {str(e)}"

    def _get_project_tool(self, input_str: str) -> str:
        """Tool function for getting project details."""
        try:
            self._ensure_connected()
            params = self._parse_tool_input(input_str)

            project_identifier = params.get('project_identifier')
            if not project_identifier:
                return "Error: project_identifier parameter is required."

            project = self.connection.openbis.get_project(project_identifier)

            if project is None:
                return f"Project '{project_identifier}' not found."

            # Format project information
            result = f"Project: {project.identifier}\n"
            result += f"Code: {project.code}\n"
            result += f"Space: {project.space}\n"
            if hasattr(project, 'description') and project.description:
                result += f"Description: {project.description}\n"
            if hasattr(project, 'registrator'):
                result += f"Registrator: {project.registrator}\n"
            if hasattr(project, 'registrationDate'):
                result += f"Registration Date: {project.registrationDate}\n"

            return result

        except Exception as e:
            return f"Error getting project: {str(e)}"

    def _create_project_tool(self, input_str: str) -> str:
        """Tool function for creating a project."""
        try:
            self._ensure_connected()
            params = self._parse_tool_input(input_str)

            # Always use the user's own space unless explicitly specified
            space = params.get('space')
            if not space:
                space = self._get_user_space()

            code = params.get('code')
            description = params.get('description', '')

            if not all([space, code]):
                return "Error: space and code parameters are required."

            # Create project
            project = self.connection.openbis.new_project(
                space=space,
                code=code,
                description=description
            )

            project.save()

            return f"Successfully created project: {project.identifier}"

        except Exception as e:
            return f"Error creating project: {str(e)}"

    # === EXPERIMENT MANAGEMENT TOOLS ===

    def _list_experiments_tool(self, input_str: str) -> str:
        """Tool function for listing experiments."""
        try:
            self._ensure_connected()
            params = self._parse_tool_input(input_str)

            # Always use the user's own space unless explicitly specified
            space = params.get('space')
            if not space:
                space = self._get_user_space()

            project = params.get('project')
            experiment_type = params.get('experiment_type')
            limit = params.get('limit')  # No default limit

            # Get experiments
            experiments = self.connection.openbis.get_experiments(
                space=space,
                project=project,
                type=experiment_type
            )

            # Apply date filtering if specified
            date_filters = {k: v for k, v in params.items() if k in ['year', 'month']}
            if date_filters:
                experiments = self._filter_by_date(experiments, date_filters)

            if len(experiments) == 0:
                filter_desc = ""
                if date_filters:
                    if 'month' in date_filters and 'year' in date_filters:
                        month_names = ['', 'January', 'February', 'March', 'April', 'May', 'June',
                                     'July', 'August', 'September', 'October', 'November', 'December']
                        filter_desc = f" from {month_names[date_filters['month']]} {date_filters['year']}"
                    elif 'year' in date_filters:
                        filter_desc = f" from {date_filters['year']}"
                return f"No experiments found{filter_desc} matching the criteria."

            total_count = len(experiments)
            experiments_to_show = experiments

            # Apply limit only if explicitly requested
            if limit is not None:
                limit = int(limit)
                if limit > 0:
                    # Sort by registration date (most recent first) when limiting
                    if hasattr(experiments, 'df'):
                        experiments_df = experiments.df.sort_values('registrationDate', ascending=False)
                        experiments_to_show = experiments_df.head(limit)
                    else:
                        experiments_to_show = sorted(experiments,
                                                   key=lambda x: getattr(x, 'registrationDate', ''),
                                                   reverse=True)[:limit]

            # Format response
            if limit is not None and limit > 0:
                # When limiting, show the count of displayed items and include dates
                displayed_count = len(experiments_to_show) if hasattr(experiments_to_show, '__len__') else limit
                result = f"Showing {displayed_count} most recent experiments (out of {total_count} total):\n"

                if hasattr(experiments_to_show, 'iterrows'):
                    for idx, (_, experiment_data) in enumerate(experiments_to_show.iterrows()):
                        reg_date = experiment_data.get('registrationDate', 'N/A')
                        result += f"{idx+1}. {experiment_data.get('identifier', 'N/A')} ({experiment_data.get('type', 'N/A')}) - {reg_date}\n"
                else:
                    for idx, experiment in enumerate(experiments_to_show):
                        reg_date = getattr(experiment, 'registrationDate', 'N/A')
                        result += f"{idx+1}. {experiment.identifier} ({experiment.type}) - {reg_date}\n"
            else:
                # When showing all, just show the total count
                result = f"Found {total_count} experiments:\n"

                if hasattr(experiments_to_show, 'iterrows'):
                    for idx, (_, experiment_data) in enumerate(experiments_to_show.iterrows()):
                        result += f"{idx+1}. {experiment_data.get('identifier', 'N/A')} ({experiment_data.get('type', 'N/A')})\n"
                else:
                    for idx, experiment in enumerate(experiments_to_show):
                        result += f"{idx+1}. {experiment.identifier} ({experiment.type})\n"

            return result

        except Exception as e:
            return f"Error listing experiments: {str(e)}"

    def _get_experiment_tool(self, input_str: str) -> str:
        """Tool function for getting experiment details."""
        try:
            self._ensure_connected()
            params = self._parse_tool_input(input_str)

            experiment_identifier = params.get('experiment_identifier')
            if not experiment_identifier:
                return "Error: experiment_identifier parameter is required."

            experiment = self.connection.openbis.get_experiment(experiment_identifier)

            if experiment is None:
                return f"Experiment '{experiment_identifier}' not found."

            # Format experiment information
            result = f"Experiment: {experiment.identifier}\n"
            result += f"Type: {experiment.type}\n"
            result += f"Project: {experiment.project}\n"

            if hasattr(experiment, 'properties') and experiment.properties:
                result += "Properties:\n"
                for key, value in experiment.properties.items():
                    result += f"  {key}: {value}\n"

            return result

        except Exception as e:
            return f"Error getting experiment: {str(e)}"

    def _create_experiment_tool(self, input_str: str) -> str:
        """Tool function for creating an experiment."""
        try:
            self._ensure_connected()
            params = self._parse_tool_input(input_str)

            experiment_type = params.get('experiment_type')
            project = params.get('project')
            code = params.get('code')
            properties = params.get('properties', {})

            if not all([experiment_type, project, code]):
                return "Error: experiment_type, project, and code parameters are required."

            # Create experiment
            experiment = self.connection.openbis.new_experiment(
                type=experiment_type,
                project=project,
                code=code,
                props=properties
            )

            experiment.save()

            return f"Successfully created experiment: {experiment.identifier}"

        except Exception as e:
            return f"Error creating experiment: {str(e)}"

    # === SAMPLE MANAGEMENT TOOLS ===

    def _list_samples_tool(self, input_str: str) -> str:
        """Tool function for listing samples."""
        try:
            self._ensure_connected()
            params = self._parse_tool_input(input_str)

            sample_type = params.get('sample_type')
            # Always use the user's own space unless explicitly specified
            space = params.get('space')
            if not space:
                space = self._get_user_space()

            project = params.get('project')
            experiment = params.get('experiment')
            limit = params.get('limit')  # No default limit

            # Get samples
            samples = self.connection.openbis.get_samples(
                type=sample_type,
                space=space,
                project=project,
                experiment=experiment
            )

            # Convert to list if it's not already
            if hasattr(samples, '__iter__') and not isinstance(samples, list):
                samples_list = list(samples)
            else:
                samples_list = samples

            # Apply date filtering if specified
            date_filters = {k: v for k, v in params.items() if k in ['year', 'month']}
            if date_filters:
                samples_list = self._filter_by_date(samples_list, date_filters)

            if len(samples_list) == 0:
                filter_desc = ""
                if date_filters:
                    if 'month' in date_filters and 'year' in date_filters:
                        month_names = ['', 'January', 'February', 'March', 'April', 'May', 'June',
                                     'July', 'August', 'September', 'October', 'November', 'December']
                        filter_desc = f" from {month_names[date_filters['month']]} {date_filters['year']}"
                    elif 'year' in date_filters:
                        filter_desc = f" from {date_filters['year']}"
                return f"No samples found{filter_desc} matching the criteria."

            total_count = len(samples_list)
            samples_to_show = samples_list

            # Apply limit only if explicitly requested
            if limit is not None:
                limit = int(limit)
                if limit > 0:
                    # Sort by registration date (most recent first) when limiting
                    samples_to_show = sorted(samples_list,
                                           key=lambda x: getattr(x, 'registrationDate', ''),
                                           reverse=True)[:limit]

            # Check if dates should be shown
            show_dates = params.get('show_dates', False) or date_filters or (limit is not None and limit > 0)

            # Format response
            if limit is not None and limit > 0:
                # When limiting, show the count of displayed items and include dates
                displayed_count = len(samples_to_show)
                result = f"Showing {displayed_count} most recent samples (out of {total_count} total):\n"

                for idx, sample in enumerate(samples_to_show):
                    reg_date = getattr(sample, 'registrationDate', 'N/A')
                    result += f"{idx+1}. {sample.identifier} ({sample.type}) - {reg_date}\n"
            else:
                # When showing all, show dates if requested or if date filtering was applied
                if show_dates:
                    result = f"Found {total_count} samples with registration dates:\n"
                    for idx, sample in enumerate(samples_to_show):
                        reg_date = getattr(sample, 'registrationDate', 'N/A')
                        result += f"{idx+1}. {sample.identifier} ({sample.type}) - {reg_date}\n"
                else:
                    result = f"Found {total_count} samples:\n"
                    for idx, sample in enumerate(samples_to_show):
                        result += f"{idx+1}. {sample.identifier} ({sample.type})\n"

            return result

        except Exception as e:
            return f"Error listing samples: {str(e)}"

    def _get_sample_tool(self, input_str: str) -> str:
        """Tool function for getting sample details."""
        try:
            self._ensure_connected()
            params = self._parse_tool_input(input_str)

            sample_identifier = params.get('sample_identifier')
            if not sample_identifier:
                return "Error: sample_identifier parameter is required."

            sample = self.connection.openbis.get_sample(sample_identifier)

            if sample is None:
                return f"Sample '{sample_identifier}' not found."

            # Format sample information
            result = f"Sample: {sample.identifier}\n"
            result += f"Type: {sample.type}\n"
            result += f"Space: {sample.space}\n"

            if hasattr(sample, 'properties') and sample.properties:
                result += "Properties:\n"
                for key, value in sample.properties.items():
                    result += f"  {key}: {value}\n"

            return result

        except Exception as e:
            return f"Error getting sample: {str(e)}"

    def _create_sample_tool(self, input_str: str) -> str:
        """Tool function for creating a sample."""
        try:
            self._ensure_connected()
            params = self._parse_tool_input(input_str)

            sample_type = params.get('sample_type')
            # Always use the user's own space unless explicitly specified
            space = params.get('space')
            if not space:
                space = self._get_user_space()

            code = params.get('code')
            properties = params.get('properties', {})

            if not all([sample_type, space, code]):
                return "Error: sample_type, space, and code parameters are required."

            # Create sample
            sample = self.connection.openbis.new_sample(
                type=sample_type,
                space=space,
                code=code,
                props=properties
            )

            sample.save()

            return f"Successfully created sample: {sample.identifier}"

        except Exception as e:
            return f"Error creating sample: {str(e)}"

    def _update_sample_tool(self, input_str: str) -> str:
        """Tool function for updating a sample."""
        try:
            self._ensure_connected()
            params = self._parse_tool_input(input_str)

            sample_identifier = params.get('sample_identifier')
            properties = params.get('properties', {})

            if not sample_identifier:
                return "Error: sample_identifier parameter is required."

            # Get existing sample
            sample = self.connection.openbis.get_sample(sample_identifier)

            if sample is None:
                return f"Sample '{sample_identifier}' not found."

            # Update properties
            for key, value in properties.items():
                sample.props[key] = value

            sample.save()

            return f"Successfully updated sample: {sample.identifier}"

        except Exception as e:
            return f"Error updating sample: {str(e)}"

    def _list_samples_detailed_tool(self, input_str: str) -> str:
        """Tool function for listing samples with detailed information."""
        try:
            self._ensure_connected()
            params = self._parse_tool_input(input_str)

            sample_type = params.get('sample_type')
            # Always use the user's own space unless explicitly specified
            space = params.get('space')
            if not space:
                space = self._get_user_space()

            project = params.get('project')
            experiment = params.get('experiment')
            limit = params.get('limit')  # No default limit - show all unless user specifies
            show_properties = params.get('show_properties', True)

            # Get samples - include properties in the request
            samples = self.connection.openbis.get_samples(
                type=sample_type,
                space=space,
                project=project,
                experiment=experiment,
                props="*"  # Get all properties
            )

            # Convert to list if it's not already
            if hasattr(samples, '__iter__') and not isinstance(samples, list):
                samples_list = list(samples)
            else:
                samples_list = samples

            # Apply date filtering if specified
            date_filters = {k: v for k, v in params.items() if k in ['year', 'month']}
            if date_filters:
                samples_list = self._filter_by_date(samples_list, date_filters)

            if len(samples_list) == 0:
                filter_desc = ""
                if date_filters:
                    if 'month' in date_filters and 'year' in date_filters:
                        month_names = ['', 'January', 'February', 'March', 'April', 'May', 'June',
                                     'July', 'August', 'September', 'October', 'November', 'December']
                        filter_desc = f" from {month_names[date_filters['month']]} {date_filters['year']}"
                    elif 'year' in date_filters:
                        filter_desc = f" from {date_filters['year']}"
                return f"No samples found{filter_desc} matching the criteria."

            total_count = len(samples_list)

            # Apply limit only if explicitly specified by user
            if limit and limit > 0:
                samples_to_show = samples_list[:int(limit)]
                result = f"Found {total_count} samples (showing {len(samples_to_show)} with details):\n\n"
            else:
                samples_to_show = samples_list
                result = f"Found {total_count} samples with details:\n\n"

            for idx, sample in enumerate(samples_to_show):
                result += f"{idx+1}. Sample: {sample.identifier}\n"
                result += f"   Type: {sample.type}\n"
                result += f"   Space: {sample.space}\n"

                # Add registration date if available
                if hasattr(sample, 'registrationDate') and sample.registrationDate:
                    result += f"   Registration Date: {sample.registrationDate}\n"

                # Add registrator if available
                if hasattr(sample, 'registrator') and sample.registrator:
                    result += f"   Registrator: {sample.registrator}\n"

                # Add properties if requested and available
                if show_properties:
                    # Try different ways to access properties based on pybis documentation
                    properties = None
                    if hasattr(sample, 'props') and sample.props:
                        # Use .props attribute (recommended by pybis docs)
                        try:
                            properties = sample.props.all() if hasattr(sample.props, 'all') else sample.props
                        except Exception:
                            properties = None
                    elif hasattr(sample, 'p') and sample.p:
                        # Use .p attribute (alternative in pybis docs)
                        try:
                            properties = sample.p() if callable(sample.p) else sample.p
                        except Exception:
                            properties = None
                    elif hasattr(sample, 'properties') and sample.properties:
                        # Fallback to .properties
                        properties = sample.properties

                    if properties and isinstance(properties, dict) and len(properties) > 0:
                        result += "   Properties: "
                        # Only show property names, not values (cleaner output)
                        prop_names = [key for key in properties.keys() if properties[key] is not None and str(properties[key]).strip()]
                        if prop_names:
                            result += ", ".join(prop_names) + "\n"
                        else:
                            result += "None\n"
                    else:
                        result += "   Properties: None\n"

                result += "\n"  # Add spacing between samples

            return result

        except Exception as e:
            return f"Error listing samples with details: {str(e)}"

    def _list_datasets_tool(self, input_str: str) -> str:
        """Tool function for listing datasets."""
        try:
            self._ensure_connected()
            params = self._parse_tool_input(input_str)

            dataset_type = params.get('dataset_type')
            limit = params.get('limit', 50)  # Don't convert to int yet

            # Get datasets
            datasets = self.connection.openbis.get_datasets(type=dataset_type)

            # Convert to list if it's not already
            if hasattr(datasets, '__iter__') and not isinstance(datasets, list):
                datasets_list = list(datasets)
            else:
                datasets_list = datasets

            if len(datasets_list) == 0:
                return "No datasets found matching the criteria."

            # Apply limit
            if limit and limit > 0:
                limit_int = int(limit)
                datasets_to_show = datasets_list[:limit_int]
            else:
                datasets_to_show = datasets_list

            # Format response
            result = f"Found {len(datasets_to_show)} datasets:\n"
            for idx, dataset in enumerate(datasets_to_show):
                # Try to get the best identifier available
                identifier = "N/A"
                if hasattr(dataset, 'permId') and dataset.permId:
                    identifier = dataset.permId
                elif hasattr(dataset, 'code') and dataset.code:
                    identifier = dataset.code
                elif hasattr(dataset, 'identifier') and dataset.identifier:
                    identifier = dataset.identifier

                # Get dataset type
                dataset_type_str = "UNKNOWN"
                if hasattr(dataset, 'type') and dataset.type:
                    dataset_type_str = dataset.type

                # Get what the dataset is attached to (clean format)
                attached_to = ""
                if hasattr(dataset, 'sample') and dataset.sample:
                    # Extract clean sample name/ID
                    sample_ref = str(dataset.sample)
                    if hasattr(dataset.sample, 'identifier'):
                        sample_name = dataset.sample.identifier.split('/')[-1]  # Get just the name part
                        attached_to = f" → Sample: {sample_name}"
                    elif hasattr(dataset.sample, 'code'):
                        attached_to = f" → Sample: {dataset.sample.code}"
                    else:
                        # Fallback: extract name from string representation
                        if '/' in sample_ref:
                            sample_name = sample_ref.split('/')[-1]
                            attached_to = f" → Sample: {sample_name}"
                        else:
                            attached_to = f" → Sample: {sample_ref}"
                elif hasattr(dataset, 'experiment') and dataset.experiment:
                    # Extract clean experiment name/ID
                    exp_ref = str(dataset.experiment)
                    if hasattr(dataset.experiment, 'identifier'):
                        exp_name = dataset.experiment.identifier.split('/')[-1]  # Get just the name part
                        attached_to = f" → Experiment: {exp_name}"
                    elif hasattr(dataset.experiment, 'code'):
                        attached_to = f" → Experiment: {dataset.experiment.code}"
                    else:
                        # Fallback: extract name from string representation
                        if '/' in exp_ref:
                            exp_name = exp_ref.split('/')[-1]
                            attached_to = f" → Experiment: {exp_name}"
                        else:
                            attached_to = f" → Experiment: {exp_ref}"
                elif hasattr(dataset, 'project') and dataset.project:
                    # Extract clean project name/ID
                    proj_ref = str(dataset.project)
                    if hasattr(dataset.project, 'identifier'):
                        proj_name = dataset.project.identifier.split('/')[-1]  # Get just the name part
                        attached_to = f" → Project: {proj_name}"
                    elif hasattr(dataset.project, 'code'):
                        attached_to = f" → Project: {dataset.project.code}"
                    else:
                        # Fallback: extract name from string representation
                        if '/' in proj_ref:
                            proj_name = proj_ref.split('/')[-1]
                            attached_to = f" → Project: {proj_name}"
                        else:
                            attached_to = f" → Project: {proj_ref}"

                result += f"{idx+1}. {identifier} ({dataset_type_str}){attached_to}\n"

            return result

        except Exception as e:
            return f"Error listing datasets: {str(e)}"

    def _get_dataset_tool(self, input_str: str) -> str:
        """Tool function for getting dataset details."""
        try:
            self._ensure_connected()
            params = self._parse_tool_input(input_str)

            dataset_identifier = params.get('dataset_identifier')
            if not dataset_identifier:
                return "Error: dataset_identifier parameter is required."

            dataset = self.connection.openbis.get_dataset(dataset_identifier)

            if dataset is None:
                return f"Dataset '{dataset_identifier}' not found."

            # Format dataset information
            result = f"Dataset: {dataset.code}\n"
            result += f"Type: {dataset.type}\n"

            if hasattr(dataset, 'properties') and dataset.properties:
                result += "Properties:\n"
                for key, value in dataset.properties.items():
                    result += f"  {key}: {value}\n"

            return result

        except Exception as e:
            return f"Error getting dataset: {str(e)}"

    def _create_dataset_tool(self, input_str: str) -> str:
        """Tool function for creating a dataset."""
        try:
            self._ensure_connected()
            params = self._parse_tool_input(input_str)

            dataset_type = params.get('dataset_type')
            sample = params.get('sample')
            experiment = params.get('experiment')
            files = params.get('files', [])
            properties = params.get('properties', {})

            if not dataset_type:
                return "Error: dataset_type parameter is required."

            # Create dataset
            dataset = self.connection.openbis.new_dataset(
                type=dataset_type,
                sample=sample,
                experiment=experiment,
                files=files,
                props=properties
            )

            dataset.save()

            return f"Successfully created dataset: {dataset.code}"

        except Exception as e:
            return f"Error creating dataset: {str(e)}"

    # === MASTERDATA MANAGEMENT TOOLS ===

    def _list_sample_types_tool(self, input_str: str = "") -> str:
        """Tool function for listing sample types."""
        try:
            self._ensure_connected()

            sample_types = self.connection.openbis.get_sample_types()

            if len(sample_types) == 0:
                return "No sample types found."

            result = f"Found {len(sample_types)} sample types:\n"
            for idx, sample_type in enumerate(sample_types):
                result += f"{idx+1}. {sample_type.code}"
                if hasattr(sample_type, 'description') and sample_type.description:
                    result += f" - {sample_type.description}"
                result += "\n"

            return result

        except Exception as e:
            return f"Error listing sample types: {str(e)}"

    def _get_sample_type_tool(self, input_str: str) -> str:
        """Tool function for getting sample type details."""
        try:
            self._ensure_connected()
            params = self._parse_tool_input(input_str)

            sample_type_code = params.get('sample_type_code')
            if not sample_type_code:
                return "Error: sample_type_code parameter is required."

            sample_type = self.connection.openbis.get_sample_type(sample_type_code)

            if sample_type is None:
                return f"Sample type '{sample_type_code}' not found."

            # Format sample type information
            result = f"Sample Type: {sample_type.code}\n"
            if hasattr(sample_type, 'description') and sample_type.description:
                result += f"Description: {sample_type.description}\n"
            if hasattr(sample_type, 'generatedCodePrefix'):
                result += f"Generated Code Prefix: {sample_type.generatedCodePrefix}\n"
            if hasattr(sample_type, 'autoGeneratedCode'):
                result += f"Auto Generated Code: {sample_type.autoGeneratedCode}\n"

            # Get property assignments
            try:
                property_assignments = sample_type.get_property_assignments()
                if property_assignments:
                    result += "Properties:\n"
                    for prop in property_assignments:
                        result += f"  - {prop.propertyType}\n"
            except Exception:
                pass

            return result

        except Exception as e:
            return f"Error getting sample type: {str(e)}"

    def _list_experiment_types_tool(self, input_str: str = "") -> str:
        """Tool function for listing experiment types."""
        try:
            self._ensure_connected()

            experiment_types = self.connection.openbis.get_experiment_types()

            if len(experiment_types) == 0:
                return "No experiment types found."

            result = f"Found {len(experiment_types)} experiment types:\n"
            for idx, experiment_type in enumerate(experiment_types):
                result += f"{idx+1}. {experiment_type.code}"
                if hasattr(experiment_type, 'description') and experiment_type.description:
                    result += f" - {experiment_type.description}"
                result += "\n"

            return result

        except Exception as e:
            return f"Error listing experiment types: {str(e)}"

    def _list_dataset_types_tool(self, input_str: str = "") -> str:
        """Tool function for listing dataset types."""
        try:
            self._ensure_connected()

            dataset_types = self.connection.openbis.get_dataset_types()

            if len(dataset_types) == 0:
                return "No dataset types found."

            result = f"Found {len(dataset_types)} dataset types:\n"
            for idx, dataset_type in enumerate(dataset_types):
                result += f"{idx+1}. {dataset_type.code}"
                if hasattr(dataset_type, 'description') and dataset_type.description:
                    result += f" - {dataset_type.description}"
                result += "\n"

            return result

        except Exception as e:
            return f"Error listing dataset types: {str(e)}"

    def _list_property_types_tool(self, input_str: str = "") -> str:
        """Tool function for listing property types."""
        try:
            self._ensure_connected()

            property_types = self.connection.openbis.get_property_types()

            if len(property_types) == 0:
                return "No property types found."

            result = f"Found {len(property_types)} property types:\n"
            for idx, property_type in enumerate(property_types):
                result += f"{idx+1}. {property_type.code} ({property_type.dataType})"
                if hasattr(property_type, 'description') and property_type.description:
                    result += f" - {property_type.description}"
                result += "\n"
                if idx >= 19:  # Limit display to first 20
                    result += "... (showing first 20 results)\n"
                    break

            return result

        except Exception as e:
            return f"Error listing property types: {str(e)}"

    def _list_vocabularies_tool(self, input_str: str = "") -> str:
        """Tool function for listing vocabularies."""
        try:
            self._ensure_connected()

            vocabularies = self.connection.openbis.get_vocabularies()

            if len(vocabularies) == 0:
                return "No vocabularies found."

            result = f"Found {len(vocabularies)} vocabularies:\n"
            for idx, vocabulary in enumerate(vocabularies):
                result += f"{idx+1}. {vocabulary.code}"
                if hasattr(vocabulary, 'description') and vocabulary.description:
                    result += f" - {vocabulary.description}"
                result += "\n"

            return result

        except Exception as e:
            return f"Error listing vocabularies: {str(e)}"

    def _get_vocabulary_tool(self, input_str: str) -> str:
        """Tool function for getting vocabulary details."""
        try:
            self._ensure_connected()
            params = self._parse_tool_input(input_str)

            vocabulary_code = params.get('vocabulary_code')
            if not vocabulary_code:
                return "Error: vocabulary_code parameter is required."

            vocabulary = self.connection.openbis.get_vocabulary(vocabulary_code)

            if vocabulary is None:
                return f"Vocabulary '{vocabulary_code}' not found."

            # Format vocabulary information
            result = f"Vocabulary: {vocabulary.code}\n"
            if hasattr(vocabulary, 'description') and vocabulary.description:
                result += f"Description: {vocabulary.description}\n"

            # Get terms
            try:
                terms = vocabulary.get_terms()
                if terms:
                    result += f"Terms ({len(terms)}):\n"
                    for idx, (term_code, term) in enumerate(terms.items()):
                        result += f"  {idx+1}. {term_code}"
                        if hasattr(term, 'label') and term.label:
                            result += f" - {term.label}"
                        result += "\n"
                        if idx >= 9:  # Limit display to first 10 terms
                            result += "  ... (showing first 10 terms)\n"
                            break
            except Exception:
                pass

            return result

        except Exception as e:
            return f"Error getting vocabulary: {str(e)}"

    # === ADDITIONAL PYBIS TOOL IMPLEMENTATIONS ===

    # Session-level Openbis methods
    def _openbis_get_server_information_tool(self, input_str: str = "") -> str:
        """Tool function for getting server information."""
        try:
            self._ensure_connected()
            info = self.connection.openbis.get_server_information()
            logger.info("Retrieved server information")

            result = "Server Information:\n"
            if hasattr(info, 'major_version'):
                result += f"Version: {info.major_version}.{info.minor_version}\n"
            if hasattr(info, 'api_version'):
                result += f"API Version: {info.api_version}\n"
            if hasattr(info, 'project_samples_enabled'):
                result += f"Project Samples Enabled: {info.project_samples_enabled}\n"

            return result
        except Exception as e:
            return f"Error getting server information: {str(e)}"

    def _openbis_get_session_info_tool(self, input_str: str = "") -> str:
        """Tool function for getting session information."""
        try:
            self._ensure_connected()
            info = self.connection.openbis.get_session_info()
            logger.info("Retrieved session information")

            result = "Session Information:\n"
            if hasattr(info, 'userName'):
                result += f"User: {info.userName}\n"
            if hasattr(info, 'sessionToken'):
                result += f"Session Token: {info.sessionToken[:20]}...\n"
            if hasattr(info, 'homeGroupCode'):
                result += f"Home Group: {info.homeGroupCode}\n"

            return result
        except Exception as e:
            return f"Error getting session information: {str(e)}"

    def _openbis_is_session_active_tool(self, input_str: str = "") -> str:
        """Tool function for checking if session is active."""
        try:
            self._ensure_connected()
            is_active = self.connection.openbis.is_session_active()
            logger.info(f"Session active status: {is_active}")
            return f"Session is {'active' if is_active else 'not active'}"
        except Exception as e:
            return f"Error checking session status: {str(e)}"

    def _openbis_create_permid_tool(self, input_str: str = "") -> str:
        """Tool function for creating a new permanent ID."""
        try:
            self._ensure_connected()
            perm_id = self.connection.openbis.create_permId()
            logger.info(f"Created new permId: {perm_id}")
            return f"Generated new permanent ID: {perm_id}"
        except Exception as e:
            return f"Error creating permanent ID: {str(e)}"

    def _openbis_get_datastores_tool(self, input_str: str = "") -> str:
        """Tool function for getting all data stores."""
        try:
            self._ensure_connected()
            datastores = self.connection.openbis.get_datastores()
            logger.info(f"Retrieved {len(datastores)} data stores")

            if len(datastores) == 0:
                return "No data stores found."

            result = f"Found {len(datastores)} data stores:\n"
            for idx, ds in enumerate(datastores):
                result += f"{idx+1}. {ds.code}"
                if hasattr(ds, 'label') and ds.label:
                    result += f" - {ds.label}"
                if hasattr(ds, 'downloadUrl') and ds.downloadUrl:
                    result += f" ({ds.downloadUrl})"
                result += "\n"

            return result
        except Exception as e:
            return f"Error getting data stores: {str(e)}"

    def _openbis_get_plugins_tool(self, input_str: str) -> str:
        """Tool function for getting plugins."""
        try:
            self._ensure_connected()
            params = self._parse_tool_input(input_str)

            plugin_type = params.get('plugin_type')
            plugins = self.connection.openbis.get_plugins(plugin_type=plugin_type)
            logger.info(f"Retrieved {len(plugins)} plugins")

            if len(plugins) == 0:
                return "No plugins found."

            result = f"Found {len(plugins)} plugins:\n"
            for idx, plugin in enumerate(plugins):
                result += f"{idx+1}. {plugin.name}"
                if hasattr(plugin, 'pluginType'):
                    result += f" ({plugin.pluginType})"
                if hasattr(plugin, 'description') and plugin.description:
                    result += f" - {plugin.description}"
                result += "\n"

            return result
        except Exception as e:
            return f"Error getting plugins: {str(e)}"

    def _openbis_get_plugin_tool(self, input_str: str) -> str:
        """Tool function for getting a specific plugin."""
        try:
            self._ensure_connected()
            params = self._parse_tool_input(input_str)

            plugin_name = params.get('plugin_name')
            if not plugin_name:
                return "Error: plugin_name parameter is required."

            plugin = self.connection.openbis.get_plugin(plugin_name)
            logger.info(f"Retrieved plugin: {plugin_name}")

            if plugin is None:
                return f"Plugin '{plugin_name}' not found."

            result = f"Plugin: {plugin.name}\n"
            if hasattr(plugin, 'pluginType'):
                result += f"Type: {plugin.pluginType}\n"
            if hasattr(plugin, 'description') and plugin.description:
                result += f"Description: {plugin.description}\n"
            if hasattr(plugin, 'script') and plugin.script:
                result += f"Script: {plugin.script[:100]}...\n"

            return result
        except Exception as e:
            return f"Error getting plugin: {str(e)}"

    def _openbis_get_external_data_management_systems_tool(self, input_str: str = "") -> str:
        """Tool function for getting external data management systems."""
        try:
            self._ensure_connected()
            systems = self.connection.openbis.get_external_data_management_systems()
            logger.info(f"Retrieved {len(systems)} external DMS")

            if len(systems) == 0:
                return "No external data management systems found."

            result = f"Found {len(systems)} external data management systems:\n"
            for idx, system in enumerate(systems):
                result += f"{idx+1}. {system.code}"
                if hasattr(system, 'label') and system.label:
                    result += f" - {system.label}"
                if hasattr(system, 'address') and system.address:
                    result += f" ({system.address})"
                result += "\n"

            return result
        except Exception as e:
            return f"Error getting external data management systems: {str(e)}"

    def _openbis_get_external_data_management_system_tool(self, input_str: str) -> str:
        """Tool function for getting a specific external DMS."""
        try:
            self._ensure_connected()
            params = self._parse_tool_input(input_str)

            dms_id = params.get('dms_id')
            if not dms_id:
                return "Error: dms_id parameter is required."

            system = self.connection.openbis.get_external_data_management_system(dms_id)
            logger.info(f"Retrieved external DMS: {dms_id}")

            if system is None:
                return f"External DMS '{dms_id}' not found."

            result = f"External DMS: {system.code}\n"
            if hasattr(system, 'label') and system.label:
                result += f"Label: {system.label}\n"
            if hasattr(system, 'address') and system.address:
                result += f"Address: {system.address}\n"
            if hasattr(system, 'addressType') and system.addressType:
                result += f"Address Type: {system.addressType}\n"

            return result
        except Exception as e:
            return f"Error getting external DMS: {str(e)}"

    def _openbis_get_persons_tool(self, input_str: str = "") -> str:
        """Tool function for getting all persons."""
        try:
            self._ensure_connected()
            persons = self.connection.openbis.get_persons()
            logger.info(f"Retrieved {len(persons)} persons")

            if len(persons) == 0:
                return "No persons found."

            result = f"Found {len(persons)} persons:\n"
            for idx, person in enumerate(persons):
                result += f"{idx+1}. {person.userId}"
                if hasattr(person, 'firstName') and person.firstName:
                    result += f" ({person.firstName}"
                    if hasattr(person, 'lastName') and person.lastName:
                        result += f" {person.lastName}"
                    result += ")"
                if hasattr(person, 'email') and person.email:
                    result += f" - {person.email}"
                result += "\n"

            return result
        except Exception as e:
            return f"Error getting persons: {str(e)}"

    def _openbis_get_person_tool(self, input_str: str) -> str:
        """Tool function for getting a specific person."""
        try:
            self._ensure_connected()
            params = self._parse_tool_input(input_str)

            person_id = params.get('person_id')
            if not person_id:
                return "Error: person_id parameter is required."

            person = self.connection.openbis.get_person(person_id)
            logger.info(f"Retrieved person: {person_id}")

            if person is None:
                return f"Person '{person_id}' not found."

            result = f"Person: {person.userId}\n"
            if hasattr(person, 'firstName') and person.firstName:
                result += f"First Name: {person.firstName}\n"
            if hasattr(person, 'lastName') and person.lastName:
                result += f"Last Name: {person.lastName}\n"
            if hasattr(person, 'email') and person.email:
                result += f"Email: {person.email}\n"
            if hasattr(person, 'space') and person.space:
                result += f"Space: {person.space}\n"

            return result
        except Exception as e:
            return f"Error getting person: {str(e)}"

    def _openbis_get_groups_tool(self, input_str: str = "") -> str:
        """Tool function for getting all groups."""
        try:
            self._ensure_connected()
            groups = self.connection.openbis.get_groups()
            logger.info(f"Retrieved {len(groups)} groups")

            if len(groups) == 0:
                return "No groups found."

            result = f"Found {len(groups)} groups:\n"
            for idx, group in enumerate(groups):
                result += f"{idx+1}. {group.code}"
                if hasattr(group, 'description') and group.description:
                    result += f" - {group.description}"
                result += "\n"

            return result
        except Exception as e:
            return f"Error getting groups: {str(e)}"

    def _openbis_get_group_tool(self, input_str: str) -> str:
        """Tool function for getting a specific group."""
        try:
            self._ensure_connected()
            params = self._parse_tool_input(input_str)

            group_code = params.get('group_code')
            if not group_code:
                return "Error: group_code parameter is required."

            group = self.connection.openbis.get_group(group_code)
            logger.info(f"Retrieved group: {group_code}")

            if group is None:
                return f"Group '{group_code}' not found."

            result = f"Group: {group.code}\n"
            if hasattr(group, 'description') and group.description:
                result += f"Description: {group.description}\n"
            if hasattr(group, 'registrator') and group.registrator:
                result += f"Registrator: {group.registrator}\n"
            if hasattr(group, 'registrationDate') and group.registrationDate:
                result += f"Registration Date: {group.registrationDate}\n"

            return result
        except Exception as e:
            return f"Error getting group: {str(e)}"

    def _openbis_get_role_assignments_tool(self, input_str: str) -> str:
        """Tool function for getting role assignments."""
        try:
            self._ensure_connected()
            params = self._parse_tool_input(input_str)

            person = params.get('person')
            group = params.get('group')
            space = params.get('space')
            project = params.get('project')

            role_assignments = self.connection.openbis.get_role_assignments(
                person=person, group=group, space=space, project=project
            )
            logger.info(f"Retrieved {len(role_assignments)} role assignments")

            if len(role_assignments) == 0:
                return "No role assignments found."

            result = f"Found {len(role_assignments)} role assignments:\n"
            for idx, assignment in enumerate(role_assignments):
                result += f"{idx+1}. Role: {assignment.role}"
                if hasattr(assignment, 'user') and assignment.user:
                    result += f", User: {assignment.user}"
                if hasattr(assignment, 'authorizationGroup') and assignment.authorizationGroup:
                    result += f", Group: {assignment.authorizationGroup}"
                if hasattr(assignment, 'space') and assignment.space:
                    result += f", Space: {assignment.space}"
                if hasattr(assignment, 'project') and assignment.project:
                    result += f", Project: {assignment.project}"
                result += "\n"

            return result
        except Exception as e:
            return f"Error getting role assignments: {str(e)}"

    def _openbis_get_tags_tool(self, input_str: str = "") -> str:
        """Tool function for getting all tags."""
        try:
            self._ensure_connected()
            tags = self.connection.openbis.get_tags()
            logger.info(f"Retrieved {len(tags)} tags")

            if len(tags) == 0:
                return "No tags found."

            result = f"Found {len(tags)} tags:\n"
            for idx, tag in enumerate(tags):
                result += f"{idx+1}. {tag.code}"
                if hasattr(tag, 'description') and tag.description:
                    result += f" - {tag.description}"
                result += "\n"

            return result
        except Exception as e:
            return f"Error getting tags: {str(e)}"

    def _openbis_get_tag_tool(self, input_str: str) -> str:
        """Tool function for getting a specific tag."""
        try:
            self._ensure_connected()
            params = self._parse_tool_input(input_str)

            tag_code = params.get('tag_code')
            if not tag_code:
                return "Error: tag_code parameter is required."

            tag = self.connection.openbis.get_tag(tag_code)
            logger.info(f"Retrieved tag: {tag_code}")

            if tag is None:
                return f"Tag '{tag_code}' not found."

            result = f"Tag: {tag.code}\n"
            if hasattr(tag, 'description') and tag.description:
                result += f"Description: {tag.description}\n"
            if hasattr(tag, 'owner') and tag.owner:
                result += f"Owner: {tag.owner}\n"

            return result
        except Exception as e:
            return f"Error getting tag: {str(e)}"

    # === ENTITY-SPECIFIC METHODS ===

    # Sample methods
    def _sample_delete_tool(self, input_str: str) -> str:
        """Tool function for deleting a sample."""
        try:
            self._ensure_connected()
            params = self._parse_tool_input(input_str)

            identifier = params.get('identifier')
            reason = params.get('reason', 'Deleted via chatBIS')
            permanently = params.get('permanently', False)

            if not identifier:
                return "Error: identifier parameter is required."

            sample = self.connection.openbis.get_sample(identifier)
            if sample is None:
                return f"Sample '{identifier}' not found."

            sample.delete(reason=reason, permanently=permanently)
            logger.info(f"Deleted sample: {identifier}")
            return f"Successfully deleted sample: {identifier}"
        except Exception as e:
            return f"Error deleting sample: {str(e)}"

    def _sample_get_datasets_tool(self, input_str: str) -> str:
        """Tool function for getting datasets associated with a sample."""
        try:
            self._ensure_connected()
            params = self._parse_tool_input(input_str)

            identifier = params.get('identifier')
            if not identifier:
                return "Error: identifier parameter is required."

            sample = self.connection.openbis.get_sample(identifier)
            if sample is None:
                return f"Sample '{identifier}' not found."

            datasets = sample.get_datasets()
            logger.info(f"Retrieved {len(datasets)} datasets for sample: {identifier}")

            if len(datasets) == 0:
                return f"No datasets found for sample '{identifier}'."

            result = f"Found {len(datasets)} datasets for sample '{identifier}':\n"
            for idx, dataset in enumerate(datasets):
                if hasattr(dataset, 'code'):
                    result += f"{idx+1}. {dataset.code}"
                elif hasattr(dataset, 'permId'):
                    result += f"{idx+1}. {dataset.permId}"
                else:
                    result += f"{idx+1}. {str(dataset)}"
                result += "\n"

            return result
        except Exception as e:
            return f"Error getting datasets for sample: {str(e)}"

    def _sample_get_projects_tool(self, input_str: str) -> str:
        """Tool function for getting projects associated with a sample."""
        try:
            self._ensure_connected()
            params = self._parse_tool_input(input_str)

            identifier = params.get('identifier')
            if not identifier:
                return "Error: identifier parameter is required."

            sample = self.connection.openbis.get_sample(identifier)
            if sample is None:
                return f"Sample '{identifier}' not found."

            projects = sample.get_projects()
            logger.info(f"Retrieved {len(projects)} projects for sample: {identifier}")

            if len(projects) == 0:
                return f"No projects found for sample '{identifier}'."

            result = f"Found {len(projects)} projects for sample '{identifier}':\n"
            for idx, project in enumerate(projects):
                if hasattr(project, 'identifier'):
                    result += f"{idx+1}. {project.identifier}"
                else:
                    result += f"{idx+1}. {str(project)}"
                result += "\n"

            return result
        except Exception as e:
            return f"Error getting projects for sample: {str(e)}"

    def _sample_is_marked_to_be_deleted_tool(self, input_str: str) -> str:
        """Tool function for checking if a sample is marked for deletion."""
        try:
            self._ensure_connected()
            params = self._parse_tool_input(input_str)

            identifier = params.get('identifier')
            if not identifier:
                return "Error: identifier parameter is required."

            sample = self.connection.openbis.get_sample(identifier)
            if sample is None:
                return f"Sample '{identifier}' not found."

            is_marked = sample.is_marked_to_be_deleted()
            logger.info(f"Sample {identifier} deletion status: {is_marked}")
            return f"Sample '{identifier}' is {'marked' if is_marked else 'not marked'} for deletion."
        except Exception as e:
            return f"Error checking sample deletion status: {str(e)}"

    def _sample_mark_to_be_deleted_tool(self, input_str: str) -> str:
        """Tool function for marking a sample for deletion."""
        try:
            self._ensure_connected()
            params = self._parse_tool_input(input_str)

            identifier = params.get('identifier')
            if not identifier:
                return "Error: identifier parameter is required."

            sample = self.connection.openbis.get_sample(identifier)
            if sample is None:
                return f"Sample '{identifier}' not found."

            sample.mark_to_be_deleted()
            logger.info(f"Marked sample for deletion: {identifier}")
            return f"Successfully marked sample '{identifier}' for deletion."
        except Exception as e:
            return f"Error marking sample for deletion: {str(e)}"

    def _sample_unmark_to_be_deleted_tool(self, input_str: str) -> str:
        """Tool function for unmarking a sample for deletion."""
        try:
            self._ensure_connected()
            params = self._parse_tool_input(input_str)

            identifier = params.get('identifier')
            if not identifier:
                return "Error: identifier parameter is required."

            sample = self.connection.openbis.get_sample(identifier)
            if sample is None:
                return f"Sample '{identifier}' not found."

            sample.unmark_to_be_deleted()
            logger.info(f"Unmarked sample for deletion: {identifier}")
            return f"Successfully unmarked sample '{identifier}' for deletion."
        except Exception as e:
            return f"Error unmarking sample for deletion: {str(e)}"

    def _sample_save_tool(self, input_str: str) -> str:
        """Tool function for saving changes to a sample."""
        try:
            self._ensure_connected()
            params = self._parse_tool_input(input_str)

            identifier = params.get('identifier')
            if not identifier:
                return "Error: identifier parameter is required."

            sample = self.connection.openbis.get_sample(identifier)
            if sample is None:
                return f"Sample '{identifier}' not found."

            sample.save()
            logger.info(f"Saved sample: {identifier}")
            return f"Successfully saved sample: {identifier}"
        except Exception as e:
            return f"Error saving sample: {str(e)}"

    def _sample_set_properties_tool(self, input_str: str) -> str:
        """Tool function for setting properties on a sample."""
        try:
            self._ensure_connected()
            params = self._parse_tool_input(input_str)

            identifier = params.get('identifier')
            properties = params.get('properties', {})

            if not identifier:
                return "Error: identifier parameter is required."

            sample = self.connection.openbis.get_sample(identifier)
            if sample is None:
                return f"Sample '{identifier}' not found."

            sample.set_properties(properties)
            logger.info(f"Set properties on sample: {identifier}")
            return f"Successfully set properties on sample: {identifier}"
        except Exception as e:
            return f"Error setting sample properties: {str(e)}"

    # Dataset methods
    def _dataset_delete_tool(self, input_str: str) -> str:
        """Tool function for deleting a dataset."""
        try:
            self._ensure_connected()
            params = self._parse_tool_input(input_str)

            identifier = params.get('identifier')
            reason = params.get('reason', 'Deleted via chatBIS')
            permanently = params.get('permanently', False)

            if not identifier:
                return "Error: identifier parameter is required."

            dataset = self.connection.openbis.get_dataset(identifier)
            if dataset is None:
                return f"Dataset '{identifier}' not found."

            dataset.delete(reason=reason, permanently=permanently)
            logger.info(f"Deleted dataset: {identifier}")
            return f"Successfully deleted dataset: {identifier}"
        except Exception as e:
            return f"Error deleting dataset: {str(e)}"

    def _dataset_download_tool(self, input_str: str) -> str:
        """Tool function for downloading dataset files."""
        try:
            self._ensure_connected()
            params = self._parse_tool_input(input_str)

            identifier = params.get('identifier')
            files = params.get('files')
            destination = params.get('destination')
            create_default_folders = params.get('create_default_folders', True)
            wait_until_finished = params.get('wait_until_finished', True)

            if not identifier:
                return "Error: identifier parameter is required."

            dataset = self.connection.openbis.get_dataset(identifier)
            if dataset is None:
                return f"Dataset '{identifier}' not found."

            dataset.download(
                files=files,
                destination=destination,
                create_default_folders=create_default_folders,
                wait_until_finished=wait_until_finished
            )
            logger.info(f"Downloaded dataset: {identifier}")
            return f"Successfully downloaded dataset: {identifier}"
        except Exception as e:
            return f"Error downloading dataset: {str(e)}"

    def _dataset_get_file_list_tool(self, input_str: str) -> str:
        """Tool function for getting file list from a dataset."""
        try:
            self._ensure_connected()
            params = self._parse_tool_input(input_str)

            identifier = params.get('identifier')
            recursive = params.get('recursive', True)
            start_folder = params.get('start_folder', '/')

            if not identifier:
                return "Error: identifier parameter is required."

            dataset = self.connection.openbis.get_dataset(identifier)
            if dataset is None:
                return f"Dataset '{identifier}' not found."

            file_list = dataset.get_file_list(recursive=recursive, start_folder=start_folder)
            logger.info(f"Retrieved file list for dataset: {identifier}")

            if len(file_list) == 0:
                return f"No files found in dataset '{identifier}'."

            result = f"Found {len(file_list)} files in dataset '{identifier}':\n"
            for idx, file_path in enumerate(file_list):
                result += f"{idx+1}. {file_path}\n"
                if idx >= 19:  # Limit display to first 20 files
                    result += "... (showing first 20 files)\n"
                    break

            return result
        except Exception as e:
            return f"Error getting file list: {str(e)}"

    def _dataset_get_files_tool(self, input_str: str) -> str:
        """Tool function for getting files DataFrame from a dataset."""
        try:
            self._ensure_connected()
            params = self._parse_tool_input(input_str)

            identifier = params.get('identifier')
            start_folder = params.get('start_folder', '/')

            if not identifier:
                return "Error: identifier parameter is required."

            dataset = self.connection.openbis.get_dataset(identifier)
            if dataset is None:
                return f"Dataset '{identifier}' not found."

            files_df = dataset.get_files(start_folder=start_folder)
            logger.info(f"Retrieved files DataFrame for dataset: {identifier}")

            if len(files_df) == 0:
                return f"No files found in dataset '{identifier}'."

            result = f"Found {len(files_df)} files in dataset '{identifier}':\n"
            for idx, (_, file_info) in enumerate(files_df.iterrows()):
                result += f"{idx+1}. {file_info.get('path', 'N/A')}"
                if 'size' in file_info:
                    result += f" ({file_info['size']} bytes)"
                result += "\n"
                if idx >= 19:  # Limit display to first 20 files
                    result += "... (showing first 20 files)\n"
                    break

            return result
        except Exception as e:
            return f"Error getting files: {str(e)}"

    def _dataset_is_marked_to_be_deleted_tool(self, input_str: str) -> str:
        """Tool function for checking if a dataset is marked for deletion."""
        try:
            self._ensure_connected()
            params = self._parse_tool_input(input_str)

            identifier = params.get('identifier')
            if not identifier:
                return "Error: identifier parameter is required."

            dataset = self.connection.openbis.get_dataset(identifier)
            if dataset is None:
                return f"Dataset '{identifier}' not found."

            is_marked = dataset.is_marked_to_be_deleted()
            logger.info(f"Dataset {identifier} deletion status: {is_marked}")
            return f"Dataset '{identifier}' is {'marked' if is_marked else 'not marked'} for deletion."
        except Exception as e:
            return f"Error checking dataset deletion status: {str(e)}"

    def _dataset_mark_to_be_deleted_tool(self, input_str: str) -> str:
        """Tool function for marking a dataset for deletion."""
        try:
            self._ensure_connected()
            params = self._parse_tool_input(input_str)

            identifier = params.get('identifier')
            if not identifier:
                return "Error: identifier parameter is required."

            dataset = self.connection.openbis.get_dataset(identifier)
            if dataset is None:
                return f"Dataset '{identifier}' not found."

            dataset.mark_to_be_deleted()
            logger.info(f"Marked dataset for deletion: {identifier}")
            return f"Successfully marked dataset '{identifier}' for deletion."
        except Exception as e:
            return f"Error marking dataset for deletion: {str(e)}"

    def _dataset_unmark_to_be_deleted_tool(self, input_str: str) -> str:
        """Tool function for unmarking a dataset for deletion."""
        try:
            self._ensure_connected()
            params = self._parse_tool_input(input_str)

            identifier = params.get('identifier')
            if not identifier:
                return "Error: identifier parameter is required."

            dataset = self.connection.openbis.get_dataset(identifier)
            if dataset is None:
                return f"Dataset '{identifier}' not found."

            dataset.unmark_to_be_deleted()
            logger.info(f"Unmarked dataset for deletion: {identifier}")
            return f"Successfully unmarked dataset '{identifier}' for deletion."
        except Exception as e:
            return f"Error unmarking dataset for deletion: {str(e)}"

    def _dataset_save_tool(self, input_str: str) -> str:
        """Tool function for saving changes to a dataset."""
        try:
            self._ensure_connected()
            params = self._parse_tool_input(input_str)

            identifier = params.get('identifier')
            if not identifier:
                return "Error: identifier parameter is required."

            dataset = self.connection.openbis.get_dataset(identifier)
            if dataset is None:
                return f"Dataset '{identifier}' not found."

            dataset.save()
            logger.info(f"Saved dataset: {identifier}")
            return f"Successfully saved dataset: {identifier}"
        except Exception as e:
            return f"Error saving dataset: {str(e)}"

    def _dataset_set_properties_tool(self, input_str: str) -> str:
        """Tool function for setting properties on a dataset."""
        try:
            self._ensure_connected()
            params = self._parse_tool_input(input_str)

            identifier = params.get('identifier')
            properties = params.get('properties', {})

            if not identifier:
                return "Error: identifier parameter is required."

            dataset = self.connection.openbis.get_dataset(identifier)
            if dataset is None:
                return f"Dataset '{identifier}' not found."

            dataset.set_properties(properties)
            logger.info(f"Set properties on dataset: {identifier}")
            return f"Successfully set properties on dataset: {identifier}"
        except Exception as e:
            return f"Error setting dataset properties: {str(e)}"

    def _dataset_archive_tool(self, input_str: str) -> str:
        """Tool function for archiving a dataset."""
        try:
            self._ensure_connected()
            params = self._parse_tool_input(input_str)

            identifier = params.get('identifier')
            remove_from_data_store = params.get('remove_from_data_store', True)

            if not identifier:
                return "Error: identifier parameter is required."

            dataset = self.connection.openbis.get_dataset(identifier)
            if dataset is None:
                return f"Dataset '{identifier}' not found."

            dataset.archive(remove_from_data_store=remove_from_data_store)
            logger.info(f"Archived dataset: {identifier}")
            return f"Successfully archived dataset: {identifier}"
        except Exception as e:
            return f"Error archiving dataset: {str(e)}"

    def _dataset_unarchive_tool(self, input_str: str) -> str:
        """Tool function for unarchiving a dataset."""
        try:
            self._ensure_connected()
            params = self._parse_tool_input(input_str)

            identifier = params.get('identifier')
            if not identifier:
                return "Error: identifier parameter is required."

            dataset = self.connection.openbis.get_dataset(identifier)
            if dataset is None:
                return f"Dataset '{identifier}' not found."

            dataset.unarchive()
            logger.info(f"Unarchived dataset: {identifier}")
            return f"Successfully unarchived dataset: {identifier}"
        except Exception as e:
            return f"Error unarchiving dataset: {str(e)}"

    # Experiment methods
    def _experiment_delete_tool(self, input_str: str) -> str:
        """Tool function for deleting an experiment."""
        try:
            self._ensure_connected()
            params = self._parse_tool_input(input_str)

            identifier = params.get('identifier')
            reason = params.get('reason', 'Deleted via chatBIS')
            permanently = params.get('permanently', False)

            if not identifier:
                return "Error: identifier parameter is required."

            experiment = self.connection.openbis.get_experiment(identifier)
            if experiment is None:
                return f"Experiment '{identifier}' not found."

            experiment.delete(reason=reason, permanently=permanently)
            logger.info(f"Deleted experiment: {identifier}")
            return f"Successfully deleted experiment: {identifier}"
        except Exception as e:
            return f"Error deleting experiment: {str(e)}"

    def _experiment_get_datasets_tool(self, input_str: str) -> str:
        """Tool function for getting datasets associated with an experiment."""
        try:
            self._ensure_connected()
            params = self._parse_tool_input(input_str)

            identifier = params.get('identifier')
            if not identifier:
                return "Error: identifier parameter is required."

            experiment = self.connection.openbis.get_experiment(identifier)
            if experiment is None:
                return f"Experiment '{identifier}' not found."

            datasets = experiment.get_datasets()
            logger.info(f"Retrieved {len(datasets)} datasets for experiment: {identifier}")

            if len(datasets) == 0:
                return f"No datasets found for experiment '{identifier}'."

            result = f"Found {len(datasets)} datasets for experiment '{identifier}':\n"
            for idx, dataset in enumerate(datasets):
                if hasattr(dataset, 'code'):
                    result += f"{idx+1}. {dataset.code}"
                elif hasattr(dataset, 'permId'):
                    result += f"{idx+1}. {dataset.permId}"
                else:
                    result += f"{idx+1}. {str(dataset)}"
                result += "\n"

            return result
        except Exception as e:
            return f"Error getting datasets for experiment: {str(e)}"

    def _experiment_get_samples_tool(self, input_str: str) -> str:
        """Tool function for getting samples associated with an experiment."""
        try:
            self._ensure_connected()
            params = self._parse_tool_input(input_str)

            identifier = params.get('identifier')
            if not identifier:
                return "Error: identifier parameter is required."

            experiment = self.connection.openbis.get_experiment(identifier)
            if experiment is None:
                return f"Experiment '{identifier}' not found."

            samples = experiment.get_samples()
            logger.info(f"Retrieved {len(samples)} samples for experiment: {identifier}")

            if len(samples) == 0:
                return f"No samples found for experiment '{identifier}'."

            result = f"Found {len(samples)} samples for experiment '{identifier}':\n"
            for idx, sample in enumerate(samples):
                if hasattr(sample, 'identifier'):
                    result += f"{idx+1}. {sample.identifier}"
                else:
                    result += f"{idx+1}. {str(sample)}"
                result += "\n"

            return result
        except Exception as e:
            return f"Error getting samples for experiment: {str(e)}"

    def _experiment_get_projects_tool(self, input_str: str) -> str:
        """Tool function for getting projects associated with an experiment."""
        try:
            self._ensure_connected()
            params = self._parse_tool_input(input_str)

            identifier = params.get('identifier')
            if not identifier:
                return "Error: identifier parameter is required."

            experiment = self.connection.openbis.get_experiment(identifier)
            if experiment is None:
                return f"Experiment '{identifier}' not found."

            projects = experiment.get_projects()
            logger.info(f"Retrieved {len(projects)} projects for experiment: {identifier}")

            if len(projects) == 0:
                return f"No projects found for experiment '{identifier}'."

            result = f"Found {len(projects)} projects for experiment '{identifier}':\n"
            for idx, project in enumerate(projects):
                if hasattr(project, 'identifier'):
                    result += f"{idx+1}. {project.identifier}"
                else:
                    result += f"{idx+1}. {str(project)}"
                result += "\n"

            return result
        except Exception as e:
            return f"Error getting projects for experiment: {str(e)}"

    def _experiment_add_samples_tool(self, input_str: str) -> str:
        """Tool function for adding samples to an experiment."""
        try:
            self._ensure_connected()
            params = self._parse_tool_input(input_str)

            identifier = params.get('identifier')
            samples = params.get('samples', [])

            if not identifier:
                return "Error: identifier parameter is required."
            if not samples:
                return "Error: samples parameter is required."

            experiment = self.connection.openbis.get_experiment(identifier)
            if experiment is None:
                return f"Experiment '{identifier}' not found."

            # Convert sample identifiers to sample objects
            sample_objects = []
            for sample_id in samples:
                sample = self.connection.openbis.get_sample(sample_id)
                if sample:
                    sample_objects.append(sample)

            experiment.add_samples(*sample_objects)
            logger.info(f"Added {len(sample_objects)} samples to experiment: {identifier}")
            return f"Successfully added {len(sample_objects)} samples to experiment: {identifier}"
        except Exception as e:
            return f"Error adding samples to experiment: {str(e)}"

    def _experiment_del_samples_tool(self, input_str: str) -> str:
        """Tool function for removing samples from an experiment."""
        try:
            self._ensure_connected()
            params = self._parse_tool_input(input_str)

            identifier = params.get('identifier')
            samples = params.get('samples', [])

            if not identifier:
                return "Error: identifier parameter is required."
            if not samples:
                return "Error: samples parameter is required."

            experiment = self.connection.openbis.get_experiment(identifier)
            if experiment is None:
                return f"Experiment '{identifier}' not found."

            # Convert sample identifiers to sample objects
            sample_objects = []
            for sample_id in samples:
                sample = self.connection.openbis.get_sample(sample_id)
                if sample:
                    sample_objects.append(sample)

            experiment.del_samples(sample_objects)
            logger.info(f"Removed {len(sample_objects)} samples from experiment: {identifier}")
            return f"Successfully removed {len(sample_objects)} samples from experiment: {identifier}"
        except Exception as e:
            return f"Error removing samples from experiment: {str(e)}"

    def _experiment_is_marked_to_be_deleted_tool(self, input_str: str) -> str:
        """Tool function for checking if an experiment is marked for deletion."""
        try:
            self._ensure_connected()
            params = self._parse_tool_input(input_str)

            identifier = params.get('identifier')
            if not identifier:
                return "Error: identifier parameter is required."

            experiment = self.connection.openbis.get_experiment(identifier)
            if experiment is None:
                return f"Experiment '{identifier}' not found."

            is_marked = experiment.is_marked_to_be_deleted()
            logger.info(f"Experiment {identifier} deletion status: {is_marked}")
            return f"Experiment '{identifier}' is {'marked' if is_marked else 'not marked'} for deletion."
        except Exception as e:
            return f"Error checking experiment deletion status: {str(e)}"

    def _experiment_mark_to_be_deleted_tool(self, input_str: str) -> str:
        """Tool function for marking an experiment for deletion."""
        try:
            self._ensure_connected()
            params = self._parse_tool_input(input_str)

            identifier = params.get('identifier')
            if not identifier:
                return "Error: identifier parameter is required."

            experiment = self.connection.openbis.get_experiment(identifier)
            if experiment is None:
                return f"Experiment '{identifier}' not found."

            experiment.mark_to_be_deleted()
            logger.info(f"Marked experiment for deletion: {identifier}")
            return f"Successfully marked experiment '{identifier}' for deletion."
        except Exception as e:
            return f"Error marking experiment for deletion: {str(e)}"

    def _experiment_unmark_to_be_deleted_tool(self, input_str: str) -> str:
        """Tool function for unmarking an experiment for deletion."""
        try:
            self._ensure_connected()
            params = self._parse_tool_input(input_str)

            identifier = params.get('identifier')
            if not identifier:
                return "Error: identifier parameter is required."

            experiment = self.connection.openbis.get_experiment(identifier)
            if experiment is None:
                return f"Experiment '{identifier}' not found."

            experiment.unmark_to_be_deleted()
            logger.info(f"Unmarked experiment for deletion: {identifier}")
            return f"Successfully unmarked experiment '{identifier}' for deletion."
        except Exception as e:
            return f"Error unmarking experiment for deletion: {str(e)}"

    def _experiment_save_tool(self, input_str: str) -> str:
        """Tool function for saving changes to an experiment."""
        try:
            self._ensure_connected()
            params = self._parse_tool_input(input_str)

            identifier = params.get('identifier')
            if not identifier:
                return "Error: identifier parameter is required."

            experiment = self.connection.openbis.get_experiment(identifier)
            if experiment is None:
                return f"Experiment '{identifier}' not found."

            experiment.save()
            logger.info(f"Saved experiment: {identifier}")
            return f"Successfully saved experiment: {identifier}"
        except Exception as e:
            return f"Error saving experiment: {str(e)}"

    def _experiment_set_properties_tool(self, input_str: str) -> str:
        """Tool function for setting properties on an experiment."""
        try:
            self._ensure_connected()
            params = self._parse_tool_input(input_str)

            identifier = params.get('identifier')
            properties = params.get('properties', {})

            if not identifier:
                return "Error: identifier parameter is required."

            experiment = self.connection.openbis.get_experiment(identifier)
            if experiment is None:
                return f"Experiment '{identifier}' not found."

            experiment.set_properties(properties)
            logger.info(f"Set properties on experiment: {identifier}")
            return f"Successfully set properties on experiment: {identifier}"
        except Exception as e:
            return f"Error setting experiment properties: {str(e)}"

    # Project methods
    def _project_delete_tool(self, input_str: str) -> str:
        """Tool function for deleting a project."""
        try:
            self._ensure_connected()
            params = self._parse_tool_input(input_str)

            identifier = params.get('identifier')
            reason = params.get('reason', 'Deleted via chatBIS')
            permanently = params.get('permanently', False)

            if not identifier:
                return "Error: identifier parameter is required."

            project = self.connection.openbis.get_project(identifier)
            if project is None:
                return f"Project '{identifier}' not found."

            project.delete(reason=reason, permanently=permanently)
            logger.info(f"Deleted project: {identifier}")
            return f"Successfully deleted project: {identifier}"
        except Exception as e:
            return f"Error deleting project: {str(e)}"

    def _project_get_experiments_tool(self, input_str: str) -> str:
        """Tool function for getting experiments in a project."""
        try:
            self._ensure_connected()
            params = self._parse_tool_input(input_str)

            identifier = params.get('identifier')
            if not identifier:
                return "Error: identifier parameter is required."

            project = self.connection.openbis.get_project(identifier)
            if project is None:
                return f"Project '{identifier}' not found."

            experiments = project.get_experiments()
            logger.info(f"Retrieved {len(experiments)} experiments for project: {identifier}")

            if len(experiments) == 0:
                return f"No experiments found in project '{identifier}'."

            result = f"Found {len(experiments)} experiments in project '{identifier}':\n"
            for idx, experiment in enumerate(experiments):
                if hasattr(experiment, 'identifier'):
                    result += f"{idx+1}. {experiment.identifier}"
                else:
                    result += f"{idx+1}. {str(experiment)}"
                result += "\n"

            return result
        except Exception as e:
            return f"Error getting experiments for project: {str(e)}"

    def _project_get_datasets_tool(self, input_str: str) -> str:
        """Tool function for getting datasets in a project."""
        try:
            self._ensure_connected()
            params = self._parse_tool_input(input_str)

            identifier = params.get('identifier')
            if not identifier:
                return "Error: identifier parameter is required."

            project = self.connection.openbis.get_project(identifier)
            if project is None:
                return f"Project '{identifier}' not found."

            datasets = project.get_datasets()
            logger.info(f"Retrieved {len(datasets)} datasets for project: {identifier}")

            if len(datasets) == 0:
                return f"No datasets found in project '{identifier}'."

            result = f"Found {len(datasets)} datasets in project '{identifier}':\n"
            for idx, dataset in enumerate(datasets):
                if hasattr(dataset, 'code'):
                    result += f"{idx+1}. {dataset.code}"
                elif hasattr(dataset, 'permId'):
                    result += f"{idx+1}. {dataset.permId}"
                else:
                    result += f"{idx+1}. {str(dataset)}"
                result += "\n"

            return result
        except Exception as e:
            return f"Error getting datasets for project: {str(e)}"

    def _project_get_samples_tool(self, input_str: str) -> str:
        """Tool function for getting samples in a project."""
        try:
            self._ensure_connected()
            params = self._parse_tool_input(input_str)

            identifier = params.get('identifier')
            if not identifier:
                return "Error: identifier parameter is required."

            project = self.connection.openbis.get_project(identifier)
            if project is None:
                return f"Project '{identifier}' not found."

            samples = project.get_samples()
            logger.info(f"Retrieved {len(samples)} samples for project: {identifier}")

            if len(samples) == 0:
                return f"No samples found in project '{identifier}'."

            result = f"Found {len(samples)} samples in project '{identifier}':\n"
            for idx, sample in enumerate(samples):
                if hasattr(sample, 'identifier'):
                    result += f"{idx+1}. {sample.identifier}"
                else:
                    result += f"{idx+1}. {str(sample)}"
                result += "\n"

            return result
        except Exception as e:
            return f"Error getting samples for project: {str(e)}"

    def _project_get_sample_tool(self, input_str: str) -> str:
        """Tool function for getting a specific sample in a project."""
        try:
            self._ensure_connected()
            params = self._parse_tool_input(input_str)

            identifier = params.get('identifier')
            sample_code = params.get('sample_code')

            if not identifier:
                return "Error: identifier parameter is required."
            if not sample_code:
                return "Error: sample_code parameter is required."

            project = self.connection.openbis.get_project(identifier)
            if project is None:
                return f"Project '{identifier}' not found."

            sample = project.get_sample(sample_code)
            logger.info(f"Retrieved sample {sample_code} for project: {identifier}")

            if sample is None:
                return f"Sample '{sample_code}' not found in project '{identifier}'."

            result = f"Sample: {sample.identifier}\n"
            result += f"Type: {sample.type}\n"
            result += f"Space: {sample.space}\n"

            if hasattr(sample, 'properties') and sample.properties:
                result += "Properties:\n"
                for key, value in sample.properties.items():
                    result += f"  {key}: {value}\n"

            return result
        except Exception as e:
            return f"Error getting sample for project: {str(e)}"

    def _project_is_marked_to_be_deleted_tool(self, input_str: str) -> str:
        """Tool function for checking if a project is marked for deletion."""
        try:
            self._ensure_connected()
            params = self._parse_tool_input(input_str)

            identifier = params.get('identifier')
            if not identifier:
                return "Error: identifier parameter is required."

            project = self.connection.openbis.get_project(identifier)
            if project is None:
                return f"Project '{identifier}' not found."

            is_marked = project.is_marked_to_be_deleted()
            logger.info(f"Project {identifier} deletion status: {is_marked}")
            return f"Project '{identifier}' is {'marked' if is_marked else 'not marked'} for deletion."
        except Exception as e:
            return f"Error checking project deletion status: {str(e)}"

    def _project_mark_to_be_deleted_tool(self, input_str: str) -> str:
        """Tool function for marking a project for deletion."""
        try:
            self._ensure_connected()
            params = self._parse_tool_input(input_str)

            identifier = params.get('identifier')
            if not identifier:
                return "Error: identifier parameter is required."

            project = self.connection.openbis.get_project(identifier)
            if project is None:
                return f"Project '{identifier}' not found."

            project.mark_to_be_deleted()
            logger.info(f"Marked project for deletion: {identifier}")
            return f"Successfully marked project '{identifier}' for deletion."
        except Exception as e:
            return f"Error marking project for deletion: {str(e)}"

    def _project_unmark_to_be_deleted_tool(self, input_str: str) -> str:
        """Tool function for unmarking a project for deletion."""
        try:
            self._ensure_connected()
            params = self._parse_tool_input(input_str)

            identifier = params.get('identifier')
            if not identifier:
                return "Error: identifier parameter is required."

            project = self.connection.openbis.get_project(identifier)
            if project is None:
                return f"Project '{identifier}' not found."

            project.unmark_to_be_deleted()
            logger.info(f"Unmarked project for deletion: {identifier}")
            return f"Successfully unmarked project '{identifier}' for deletion."
        except Exception as e:
            return f"Error unmarking project for deletion: {str(e)}"

    def _project_save_tool(self, input_str: str) -> str:
        """Tool function for saving changes to a project."""
        try:
            self._ensure_connected()
            params = self._parse_tool_input(input_str)

            identifier = params.get('identifier')
            if not identifier:
                return "Error: identifier parameter is required."

            project = self.connection.openbis.get_project(identifier)
            if project is None:
                return f"Project '{identifier}' not found."

            project.save()
            logger.info(f"Saved project: {identifier}")
            return f"Successfully saved project: {identifier}"
        except Exception as e:
            return f"Error saving project: {str(e)}"

    # Space methods
    def _space_delete_tool(self, input_str: str) -> str:
        """Tool function for deleting a space."""
        try:
            self._ensure_connected()
            params = self._parse_tool_input(input_str)

            space_code = params.get('space_code')
            reason = params.get('reason', 'Deleted via chatBIS')
            permanently = params.get('permanently', False)

            if not space_code:
                return "Error: space_code parameter is required."

            space = self.connection.openbis.get_space(space_code)
            if space is None:
                return f"Space '{space_code}' not found."

            space.delete(reason=reason, permanently=permanently)
            logger.info(f"Deleted space: {space_code}")
            return f"Successfully deleted space: {space_code}"
        except Exception as e:
            return f"Error deleting space: {str(e)}"

    def _space_get_experiments_tool(self, input_str: str) -> str:
        """Tool function for getting experiments in a space."""
        try:
            self._ensure_connected()
            params = self._parse_tool_input(input_str)

            space_code = params.get('space_code')
            if not space_code:
                return "Error: space_code parameter is required."

            space = self.connection.openbis.get_space(space_code)
            if space is None:
                return f"Space '{space_code}' not found."

            experiments = space.get_experiments()
            logger.info(f"Retrieved {len(experiments)} experiments for space: {space_code}")

            if len(experiments) == 0:
                return f"No experiments found in space '{space_code}'."

            result = f"Found {len(experiments)} experiments in space '{space_code}':\n"
            for idx, experiment in enumerate(experiments):
                if hasattr(experiment, 'identifier'):
                    result += f"{idx+1}. {experiment.identifier}"
                else:
                    result += f"{idx+1}. {str(experiment)}"
                result += "\n"

            return result
        except Exception as e:
            return f"Error getting experiments for space: {str(e)}"

    def _space_get_samples_tool(self, input_str: str) -> str:
        """Tool function for getting samples in a space."""
        try:
            self._ensure_connected()
            params = self._parse_tool_input(input_str)

            space_code = params.get('space_code')
            if not space_code:
                return "Error: space_code parameter is required."

            space = self.connection.openbis.get_space(space_code)
            if space is None:
                return f"Space '{space_code}' not found."

            samples = space.get_samples()
            logger.info(f"Retrieved {len(samples)} samples for space: {space_code}")

            if len(samples) == 0:
                return f"No samples found in space '{space_code}'."

            result = f"Found {len(samples)} samples in space '{space_code}':\n"
            for idx, sample in enumerate(samples):
                if hasattr(sample, 'identifier'):
                    result += f"{idx+1}. {sample.identifier}"
                else:
                    result += f"{idx+1}. {str(sample)}"
                result += "\n"

            return result
        except Exception as e:
            return f"Error getting samples for space: {str(e)}"

    def _space_get_projects_tool(self, input_str: str) -> str:
        """Tool function for getting projects in a space."""
        try:
            self._ensure_connected()
            params = self._parse_tool_input(input_str)

            space_code = params.get('space_code')
            if not space_code:
                return "Error: space_code parameter is required."

            space = self.connection.openbis.get_space(space_code)
            if space is None:
                return f"Space '{space_code}' not found."

            projects = space.get_projects()
            logger.info(f"Retrieved {len(projects)} projects for space: {space_code}")

            if len(projects) == 0:
                return f"No projects found in space '{space_code}'."

            result = f"Found {len(projects)} projects in space '{space_code}':\n"
            for idx, project in enumerate(projects):
                if hasattr(project, 'identifier'):
                    result += f"{idx+1}. {project.identifier}"
                else:
                    result += f"{idx+1}. {str(project)}"
                result += "\n"

            return result
        except Exception as e:
            return f"Error getting projects for space: {str(e)}"

    def _space_is_marked_to_be_deleted_tool(self, input_str: str) -> str:
        """Tool function for checking if a space is marked for deletion."""
        try:
            self._ensure_connected()
            params = self._parse_tool_input(input_str)

            space_code = params.get('space_code')
            if not space_code:
                return "Error: space_code parameter is required."

            space = self.connection.openbis.get_space(space_code)
            if space is None:
                return f"Space '{space_code}' not found."

            is_marked = space.is_marked_to_be_deleted()
            logger.info(f"Space {space_code} deletion status: {is_marked}")
            return f"Space '{space_code}' is {'marked' if is_marked else 'not marked'} for deletion."
        except Exception as e:
            return f"Error checking space deletion status: {str(e)}"

    def _space_mark_to_be_deleted_tool(self, input_str: str) -> str:
        """Tool function for marking a space for deletion."""
        try:
            self._ensure_connected()
            params = self._parse_tool_input(input_str)

            space_code = params.get('space_code')
            if not space_code:
                return "Error: space_code parameter is required."

            space = self.connection.openbis.get_space(space_code)
            if space is None:
                return f"Space '{space_code}' not found."

            space.mark_to_be_deleted()
            logger.info(f"Marked space for deletion: {space_code}")
            return f"Successfully marked space '{space_code}' for deletion."
        except Exception as e:
            return f"Error marking space for deletion: {str(e)}"

    def _space_unmark_to_be_deleted_tool(self, input_str: str) -> str:
        """Tool function for unmarking a space for deletion."""
        try:
            self._ensure_connected()
            params = self._parse_tool_input(input_str)

            space_code = params.get('space_code')
            if not space_code:
                return "Error: space_code parameter is required."

            space = self.connection.openbis.get_space(space_code)
            if space is None:
                return f"Space '{space_code}' not found."

            space.unmark_to_be_deleted()
            logger.info(f"Unmarked space for deletion: {space_code}")
            return f"Successfully unmarked space '{space_code}' for deletion."
        except Exception as e:
            return f"Error unmarking space for deletion: {str(e)}"

    def _space_save_tool(self, input_str: str) -> str:
        """Tool function for saving changes to a space."""
        try:
            self._ensure_connected()
            params = self._parse_tool_input(input_str)

            space_code = params.get('space_code')
            if not space_code:
                return "Error: space_code parameter is required."

            space = self.connection.openbis.get_space(space_code)
            if space is None:
                return f"Space '{space_code}' not found."

            space.save()
            logger.info(f"Saved space: {space_code}")
            return f"Successfully saved space: {space_code}"
        except Exception as e:
            return f"Error saving space: {str(e)}"

    def _parse_tool_input(self, input_str: str) -> Dict[str, Any]:
        """Parse tool input string into parameters dictionary."""
        params = {}
        if not input_str.strip():
            return params

        try:
            # Handle both key=value pairs and natural language descriptions
            input_lower = input_str.lower()

            # Check for special requests
            if 'properties' in input_lower or 'property' in input_lower:
                params['show_properties'] = True

            if 'creation date' in input_lower or 'registration date' in input_lower:
                params['show_dates'] = True

            # Parse key=value pairs if present
            for pair in input_str.split(','):
                if '=' in pair:
                    key, value = pair.split('=', 1)
                    key = key.strip()
                    value = value.strip()

                    # Try to convert to appropriate type
                    if value.lower() in ['true', 'false']:
                        params[key] = value.lower() == 'true'
                    elif value.isdigit():
                        params[key] = int(value)
                    else:
                        # Remove quotes if present
                        if value.startswith('"') and value.endswith('"'):
                            value = value[1:-1]
                        elif value.startswith("'") and value.endswith("'"):
                            value = value[1:-1]
                        params[key] = value

        except Exception as e:
            logger.error(f"Error parsing tool input '{input_str}': {e}")

        # Parse date filters from the input string
        params.update(self._parse_date_filters(input_str))

        return params

    def _parse_date_filters(self, input_str: str) -> Dict[str, Any]:
        """Parse date-related filters from input string."""
        filters = {}
        input_lower = input_str.lower()

        # Parse month/year patterns like "February 2024", "Feb 2024", "2024-02"
        month_year_patterns = [
            r'(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{4})',
            r'(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s+(\d{4})',
            r'(\d{4})-(\d{1,2})',
            r'(\d{1,2})/(\d{4})'
        ]

        for pattern in month_year_patterns:
            match = re.search(pattern, input_lower)
            if match:
                if pattern.startswith(r'(\d{4})'):  # Year-month format
                    year, month = match.groups()
                    filters['year'] = int(year)
                    filters['month'] = int(month)
                elif pattern.endswith(r'(\d{4})'):  # Month-year format
                    month_str, year = match.groups()
                    filters['year'] = int(year)
                    if month_str.isdigit():
                        filters['month'] = int(month_str)
                    else:
                        # Convert month name to number
                        month_names = {
                            'january': 1, 'jan': 1, 'february': 2, 'feb': 2,
                            'march': 3, 'mar': 3, 'april': 4, 'apr': 4,
                            'may': 5, 'june': 6, 'jun': 6, 'july': 7, 'jul': 7,
                            'august': 8, 'aug': 8, 'september': 9, 'sep': 9,
                            'october': 10, 'oct': 10, 'november': 11, 'nov': 11,
                            'december': 12, 'dec': 12
                        }
                        filters['month'] = month_names.get(month_str, None)
                break

        # Parse year-only patterns like "2024", "in 2023"
        if 'year' not in filters:
            year_match = re.search(r'\b(20\d{2})\b', input_lower)
            if year_match:
                filters['year'] = int(year_match.group(1))

        # Parse relative date patterns like "last month", "this year"
        if 'last month' in input_lower:
            now = datetime.now()
            if now.month == 1:
                filters['year'] = now.year - 1
                filters['month'] = 12
            else:
                filters['year'] = now.year
                filters['month'] = now.month - 1

        if 'this month' in input_lower:
            now = datetime.now()
            filters['year'] = now.year
            filters['month'] = now.month

        if 'this year' in input_lower:
            filters['year'] = datetime.now().year

        if 'last year' in input_lower:
            filters['year'] = datetime.now().year - 1

        return filters

    def _filter_by_date(self, items, date_filters: Dict[str, Any]):
        """Filter items by registration date based on parsed date filters."""
        if not date_filters or not items:
            return items

        try:
            # Convert items to list if it's not already
            if hasattr(items, '__iter__') and not isinstance(items, (str, list)):
                items_list = list(items)
            elif isinstance(items, list):
                items_list = items
            else:
                return items

            if not items_list:
                return items

            # Filter items by date
            filtered_items = []
            for item in items_list:
                try:
                    reg_date_str = getattr(item, 'registrationDate', None)
                    if not reg_date_str:
                        continue

                    # Parse the registration date
                    if isinstance(reg_date_str, str):
                        # Handle different date formats
                        if 'T' in reg_date_str:
                            # ISO format: 2024-07-08T13:45:35 or 2024-07-08 13:45:35
                            date_part = reg_date_str.split('T')[0] if 'T' in reg_date_str else reg_date_str.split(' ')[0]
                        else:
                            date_part = reg_date_str.split(' ')[0]

                        # Parse YYYY-MM-DD format
                        year, month, _ = map(int, date_part.split('-'))

                        # Check year filter
                        if 'year' in date_filters and year != date_filters['year']:
                            continue

                        # Check month filter
                        if 'month' in date_filters and month != date_filters['month']:
                            continue

                        # Item passed all filters
                        filtered_items.append(item)

                except Exception as e:
                    # If we can't parse the date, skip this item
                    logger.debug(f"Could not parse date for item {getattr(item, 'identifier', 'unknown')}: {e}")
                    continue

            return filtered_items

        except Exception as e:
            logger.warning(f"Date filtering failed: {e}")
            return items


def get_available_tools() -> List[Tool]:
    """Get list of available pybis tools."""
    if not PYBIS_AVAILABLE:
        return []

    manager = PyBISToolManager()
    return manager.get_tools()
