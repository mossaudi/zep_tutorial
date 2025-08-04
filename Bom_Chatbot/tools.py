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


class ToolResponse:
    """Smart tool response that can bypass LLM processing for large datasets."""

    @staticmethod
    def direct_display(content: str, bypass_llm: bool = False):
        """Return content that should be displayed directly."""
        if bypass_llm:
            # For very large datasets, return minimal info to LLM
            return f"DIRECT_DISPLAY_MARKER: Content displayed directly to user. Size: {len(content)} chars"
        return content

    @staticmethod
    def structured_data(data: dict):
        """Return structured data for LLM processing."""
        return data


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
    """📋 CREATE EMPTY BOM: Create new Bill of Materials structure for custom BOM projects.

        🎯 BEST FOR:
        - Starting a new BOM project
        - Custom BOM structure with specific columns
        - Preparing BOM before adding components

        🔧 WHEN TO USE:
        - Want to create BOM manually
        - Need custom column structure
        - Preparing for component addition workflow

        💡 WHAT HAPPENS NEXT:
        - Use 'add_parts_to_bom' to populate with components
        - Use 'get_boms' to verify creation

        Args:
            name: New BOM's name (mandatory)
            columns: JSON array of column names (optional)
                if not given use all the default columns [
                    "cpn", "mpn", "manufacturer", "description",
                    "quantity", "uploadedcomments", "uploadedlifecycle"
                ]
            description: New BOM's description (optional)
            parent_path: Saving path in pattern "project>subproject1>subproject2" (optional)

        Returns:
            JSON response with operation status
        """
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
def get_boms(project_name: str = "", **kwargs) -> str:
    """📋 GET BOMS: Lists existing Bills of Materials in a hierarchical tree view.

    Shows projects with their BOMs and root-level BOMs in an organized tree structure.

    Args:
        project_name: Optional filter for specific project name
        **kwargs: Additional date filters (creation_date_from, creation_date_to, etc.)

    Returns:
        A formatted string showing the BOM tree structure.
    """
    if not _container:
        raise AgentError("Tools not initialized.")

    try:
        bom_result = _container.bom_service.get_boms(project_name=project_name, **kwargs)

        if not bom_result.get("success", False):
            error_msg = f"❌ Failed to retrieve BOMs"
            return ToolResponse.direct_display(error_msg)

        from Bom_Chatbot.models import BOMTreeResult
        bom_tree = BOMTreeResult(**bom_result["bom_tree"])
        formatted_output = _container.formatter.format_bom_tree(bom_tree)

        # For large datasets, bypass LLM processing
        bypass_llm = bom_tree.total_boms > 1000
        return ToolResponse.direct_display(formatted_output, bypass_llm)

    except Exception as e:
        error_msg = f"❌ Error retrieving BOMs: {str(e)}"
        return ToolResponse.direct_display(error_msg)

@tool
def add_parts_to_bom(name: str, parent_path: str, parts_json: str) -> Dict[str, Any]:
    """➕ ADD PARTS TO BOM: Adds components to an existing BOM."""
    if not _container:
        raise AgentError("Tools not initialized.")
    return _container.bom_service.add_parts_to_bom(
        name=name, parent_path=parent_path, parts_data=parts_json
    )