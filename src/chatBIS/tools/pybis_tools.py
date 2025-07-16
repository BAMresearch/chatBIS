#!/usr/bin/env python3
"""
Comprehensive PyBIS Tools for openBIS Chatbot

This module provides LangChain Tool wrappers for all major pybis functions,
enabling the chatbot to execute a wide range of actions on openBIS instances.

Based on pybis v1.37.3 documentation from:
- https://pypi.org/project/pybis/
- https://openbis.readthedocs.io/en/latest/software-developer-documentation/apis/python-v3-api.html
"""

import logging
import os
from typing import Dict, List, Any
from langchain_core.tools import Tool

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
            description="List projects in the user's space in openBIS. Projects live within spaces and contain experiments. Automatically filters by the user's own space unless space parameter is explicitly provided.",
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
            description="List ALL experiments (collections) in the user's space in openBIS. Shows total count and all experiments by default. Automatically filters by the user's own space unless space parameter is explicitly provided. Optional parameters: project (string), experiment_type (string), limit (integer, only use when user asks for 'last N' or 'recent N' experiments)",
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
            description="List ALL samples (objects) in the user's space in openBIS. Shows total count and all samples by default. Automatically filters by the user's own space unless space parameter is explicitly provided. Optional parameters: sample_type (string), project (string), experiment (string), limit (integer, only use when user asks for 'last N' or 'recent N' samples)",
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

            if len(projects) == 0:
                return f"No projects found{' in space ' + space if space else ''}."

            result = f"Found {len(projects)} projects{' in space ' + space if space else ''}:\n"
            for idx, project in enumerate(projects):
                result += f"{idx+1}. {project.identifier}"
                if hasattr(project, 'description') and project.description:
                    result += f" - {project.description}"
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

            if len(experiments) == 0:
                return "No experiments found matching the criteria."

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

            if len(samples) == 0:
                return "No samples found matching the criteria."

            total_count = len(samples)
            samples_to_show = samples

            # Apply limit only if explicitly requested
            if limit is not None:
                limit = int(limit)
                if limit > 0:
                    # Sort by registration date (most recent first) when limiting
                    if hasattr(samples, 'df'):
                        samples_df = samples.df.sort_values('registrationDate', ascending=False)
                        samples_to_show = samples_df.head(limit)
                    else:
                        samples_to_show = sorted(samples,
                                               key=lambda x: getattr(x, 'registrationDate', ''),
                                               reverse=True)[:limit]

            # Format response
            if limit is not None and limit > 0:
                # When limiting, show the count of displayed items and include dates
                displayed_count = len(samples_to_show) if hasattr(samples_to_show, '__len__') else limit
                result = f"Showing {displayed_count} most recent samples (out of {total_count} total):\n"

                if hasattr(samples_to_show, 'iterrows'):
                    for idx, (_, sample_data) in enumerate(samples_to_show.iterrows()):
                        reg_date = sample_data.get('registrationDate', 'N/A')
                        result += f"{idx+1}. {sample_data.get('identifier', 'N/A')} ({sample_data.get('type', 'N/A')}) - {reg_date}\n"
                else:
                    for idx, sample in enumerate(samples_to_show):
                        reg_date = getattr(sample, 'registrationDate', 'N/A')
                        result += f"{idx+1}. {sample.identifier} ({sample.type}) - {reg_date}\n"
            else:
                # When showing all, just show the total count
                result = f"Found {total_count} samples:\n"

                if hasattr(samples_to_show, 'iterrows'):
                    for idx, (_, sample_data) in enumerate(samples_to_show.iterrows()):
                        result += f"{idx+1}. {sample_data.get('identifier', 'N/A')} ({sample_data.get('type', 'N/A')})\n"
                else:
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

    def _list_datasets_tool(self, input_str: str) -> str:
        """Tool function for listing datasets."""
        try:
            self._ensure_connected()
            params = self._parse_tool_input(input_str)

            dataset_type = params.get('dataset_type')
            limit = int(params.get('limit', 20))

            # Get datasets
            datasets = self.connection.openbis.get_datasets(type=dataset_type)

            # Limit results
            datasets_list = datasets.df.head(limit) if hasattr(datasets, 'df') else datasets[:limit]

            if len(datasets_list) == 0:
                return "No datasets found matching the criteria."

            # Format response
            result = f"Found {len(datasets_list)} datasets:\n"
            for idx, dataset in enumerate(datasets_list.iterrows() if hasattr(datasets_list, 'iterrows') else enumerate(datasets_list)):
                if hasattr(datasets_list, 'iterrows'):
                    _, dataset_data = dataset
                    result += f"{idx+1}. {dataset_data.get('code', 'N/A')} ({dataset_data.get('type', 'N/A')})\n"
                else:
                    result += f"{idx+1}. {dataset}\n"
                if idx >= 9:  # Limit display
                    break

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



    def _parse_tool_input(self, input_str: str) -> Dict[str, Any]:
        """Parse tool input string into parameters dictionary."""
        params = {}
        if not input_str.strip():
            return params

        try:
            # Simple parsing for key=value pairs separated by commas
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

        return params


def get_available_tools() -> List[Tool]:
    """Get list of available pybis tools."""
    if not PYBIS_AVAILABLE:
        return []

    manager = PyBISToolManager()
    return manager.get_tools()
