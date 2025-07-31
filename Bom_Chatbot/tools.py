# tools.py
"""Enhanced LangGraph tools that delegate logic to services."""
import json
from typing import List, Dict, Any

from langchain.tools import tool
from dataclasses import asdict

from container import Container
from exceptions import AgentError

# Global container instance
_container: Container = None


def initialize_tools(container: Container):
    """Initializes the tools with the dependency container."""
    global _container
    _container = container


def get_tools(container: Container) -> List[Any]:
    """Get all available tools, ensuring they are initialized."""
    initialize_tools(container)
    return [
        analyze_schematic,
        search_component_data,
        create_empty_bom,
        get_boms,
        add_parts_to_bom,
    ]


@tool
def analyze_schematic(image_url: str) -> Dict[str, Any]:
    """🔍 ANALYZE SCHEMATIC: Extracts and enhances component data from a schematic URL.

    This tool runs a complete workflow:
    1. Analyzes the schematic image to identify components.
    2. Searches for detailed data for each component.
    3. Returns a structured result with all findings.

    Args:
        image_url: The public URL of the schematic image to analyze.

    Returns:
        A dictionary containing the search results, including a list of components.
    """
    if not _container:
        raise AgentError("Tools not initialized.")
    # Delegate the entire complex workflow to the service
    search_result = _container.workflow_service.run_schematic_analysis_workflow(image_url)
    return asdict(search_result)


@tool
def search_component_data(components_json: str) -> Dict[str, Any]:
    """🔍 SEARCH COMPONENT DATA: Searches for data on a given list of components.

    Args:
        components_json: A JSON string representing a list of components to search.

    Returns:
        A dictionary containing the search results.
    """
    if not _container:
        raise AgentError("Tools not initialized.")

    # Delegate parsing and searching to the respective services
    components = _container.parsing_service.parse_and_convert_to_components(components_json)
    search_result = _container.analysis_service.search_component_data(components)
    return asdict(search_result)


@tool
def create_empty_bom(name: str, columns: str, description: str = "", parent_path: str = "") -> Dict[str, Any]:
    """📋 CREATE EMPTY BOM: Creates a new, empty Bill of Materials."""
    if not _container:
        raise AgentError("Tools not initialized.")
    try:
        columns_list = json.loads(columns) if isinstance(columns, str) else columns
    except json.JSONDecodeError:
        return {"error": "Invalid JSON format for columns"}

    return _container.bom_service.create_empty_bom(
        name=name, columns=columns_list, description=description, parent_path=parent_path
    )


@tool
def get_boms(project_name: str = "", **kwargs) -> Dict[str, Any]:
    """📋 GET BOMS: Lists existing Bills of Materials with optional filters."""
    if not _container:
        raise AgentError("Tools not initialized.")
    return _container.bom_service.get_boms(project_name=project_name, **kwargs)


@tool
def add_parts_to_bom(name: str, parent_path: str, parts_json: str) -> Dict[str, Any]:
    """➕ ADD PARTS TO BOM: Adds components to an existing BOM."""
    if not _container:
        raise AgentError("Tools not initialized.")
    return _container.bom_service.add_parts_to_bom(
        name=name, parent_path=parent_path, parts_data=parts_json
    )