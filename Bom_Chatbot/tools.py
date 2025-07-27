# tools.py
"""Refactored LangGraph tools using the new service architecture."""

import json

from langchain.tools import tool

from clients.silicon_expert import SiliconExpertClient
from config import AppConfig
from exceptions import AgentError, DataValidationError
from models import Component
from services.analysis import ComponentAnalysisService
from services.formatter import ComponentTableFormatter, ComponentDataConverter
from services.workflow import BOMWorkflowService, BOMManagementService
from services.intelligent_selection import ConversationContext


class ToolDependencies:
    """Container for tool dependencies."""

    def __init__(self, config: AppConfig, llm):
        self.config = config
        self.llm = llm

        # Initialize clients and services
        self.silicon_expert_client = SiliconExpertClient(config.silicon_expert)
        self.analysis_service = ComponentAnalysisService(llm, self.silicon_expert_client)
        self.workflow_service = BOMWorkflowService(self.analysis_service, self.silicon_expert_client)
        self.bom_service = BOMManagementService(self.silicon_expert_client)
        self.formatter = ComponentTableFormatter()
        self.converter = ComponentDataConverter()
        self.conversation_context = ConversationContext(
            recent_messages=[],
            previous_tool_results=[],
            available_data={}
        )


# Global dependencies instance (will be initialized in main)
_dependencies: ToolDependencies = None


def initialize_tools(config: AppConfig, llm) -> None:
    """Initialize tool dependencies."""
    global _dependencies
    _dependencies = ToolDependencies(config, llm)


def get_dependencies() -> ToolDependencies:
    """Get tool dependencies."""
    if _dependencies is None:
        raise AgentError("Tools not initialized. Call initialize_tools() first.")
    return _dependencies


@tool
def analyze_schematic(image_url: str) -> str:
    """Analyze a schematic design image from a URL and return enhanced component details.

    Args:
        image_url: URL of the schematic image to analyze

    Returns:
        Formatted table showing component analysis results with Silicon Expert data
    """
    try:
        deps = get_dependencies()
        search_result = deps.analysis_service.analyze_schematic(image_url)

        if not search_result.success:
            return f"Analysis failed: {search_result.error_message}"

        # NEW: Update conversation context
        if search_result.components:
            component_data = [
                {
                    'name': comp.name,
                    'part_number': comp.part_number,
                    'manufacturer': comp.manufacturer,
                    'description': comp.description
                }
                for comp in search_result.components
            ]
            deps.conversation_context.available_data['active_components'] = component_data
            deps.conversation_context.available_data['component_count'] = len(component_data)

        table_output = deps.formatter.format_search_result(search_result)

        # NEW: Add intelligent suggestions
        if search_result.components:
            suggestions = (
                "\n💡 INTELLIGENT SUGGESTIONS:\n"
                "Based on your analysis, the system recommends:\n"
                f"1. 'create_bom_from_schematic' - Complete workflow with {len(search_result.components)} components\n"
                "2. 'search_component_data' - Enhance component information\n"
                "3. 'create_empty_bom' - Custom BOM structure\n"
                "\nJust tell me what you'd like to do next!"
            )
            table_output += suggestions

        return table_output

    except Exception as e:
        return f"Error analyzing schematic: {str(e)}"


@tool
def search_component_data(components_json: str) -> str:
    """Search for component data using Silicon Expert API.

    Args:
        components_json: JSON string containing component data

    Returns:
        Formatted table showing enhanced component information
    """
    try:
        deps = get_dependencies()

        # Parse components from JSON
        try:
            components_data = json.loads(components_json)
            if not isinstance(components_data, list):
                raise DataValidationError(
                    "Component data must be a JSON array",
                    "component_list",
                    ["Expected list, got " + type(components_data).__name__]
                )

            # Convert to Component objects
            components = []
            for i, item in enumerate(components_data):
                if not isinstance(item, dict):
                    continue

                component = Component(
                    name=item.get('name', item.get('component_name', f'Component_{i + 1}')),
                    part_number=item.get('part_number'),
                    manufacturer=item.get('manufacturer'),
                    description=item.get('description'),
                    value=item.get('value') or item.get('rating'),
                    features=item.get('features', ''),
                    quantity=item.get('quantity', '1')
                )
                components.append(component)

        except json.JSONDecodeError as e:
            return f"Error: Invalid JSON format - {str(e)}"
        except DataValidationError as e:
            return f"Error: {e.message}"

        if not components:
            return "Error: No valid components found in the provided data"

        # Search for component data
        enhanced_components = deps.silicon_expert_client.search_components(components)

        # Calculate statistics
        successful_searches = len([c for c in enhanced_components if c.silicon_expert_data])
        failed_searches = len(enhanced_components) - successful_searches

        from .models import SearchResult
        search_result = SearchResult(
            success=True,
            components=enhanced_components,
            successful_searches=successful_searches,
            failed_searches=failed_searches
        )

        # Format and return results
        table_output = deps.formatter.format_search_result(search_result)
        return f"COMPONENT_SEARCH_COMPLETE:{table_output}"

    except Exception as e:
        return f"Error searching component data: {str(e)}"


@tool
def create_empty_bom(name: str, columns: str, description: str = "", parent_path: str = "") -> str:
    """Create an empty BOM in SiliconExpert P5 BOM manager.

    Args:
        name: New BOM's name (mandatory)
        columns: JSON array of column names (mandatory)
        description: New BOM's description (optional)
        parent_path: Saving path in pattern "project>subproject1>subproject2" (optional)

    Returns:
        JSON response with operation status
    """
    try:
        deps = get_dependencies()

        # Parse columns
        try:
            columns_list = json.loads(columns) if isinstance(columns, str) else columns
            if not isinstance(columns_list, list):
                return "Error: Columns must be a JSON array"
        except json.JSONDecodeError:
            return "Error: Invalid JSON format for columns"

        # Create BOM
        result = deps.bom_service.create_empty_bom(
            name=name,
            columns=columns_list,
            description=description,
            parent_path=parent_path
        )

        return json.dumps(result, indent=2)

    except Exception as e:
        return f"Error creating empty BOM: {str(e)}"


@tool
def get_boms(project_name: str = "", bom_creation_date_from: str = "",
             bom_creation_date_to: str = "", bom_modification_date_from: str = "",
             bom_modification_date_to: str = "") -> str:
    """Get BOM meta information from SiliconExpert P5 BOM manager.

    Args:
        project_name: Filter by project name (optional)
        bom_creation_date_from: Filter on BOM creation date from (optional)
        bom_creation_date_to: Filter on BOM creation date to (optional)
        bom_modification_date_from: Filter on BOM modification date from (optional)
        bom_modification_date_to: Filter on BOM modification date to (optional)

    Returns:
        JSON response with BOM information
    """
    try:
        deps = get_dependencies()

        result = deps.bom_service.get_boms(
            project_name=project_name,
            bom_creation_date_from=bom_creation_date_from,
            bom_creation_date_to=bom_creation_date_to,
            bom_modification_date_from=bom_modification_date_from,
            bom_modification_date_to=bom_modification_date_to
        )

        return json.dumps(result, indent=2)

    except Exception as e:
        return f"Error getting BOMs: {str(e)}"


@tool
def add_parts_to_bom(name: str, parent_path: str, parts_json: str) -> str:
    """Add/upload parts into a BOM.

    Args:
        name: BOM name (mandatory)
        parent_path: BOM saving path (can be empty string)
        parts_json: JSON array of part data with 'mpn' and 'manufacturer' required

    Returns:
        JSON response with operation status
    """
    try:
        deps = get_dependencies()

        result = deps.bom_service.add_parts_to_bom(
            name=name,
            parent_path=parent_path,
            parts_data=parts_json
        )

        return json.dumps(result, indent=2)

    except Exception as e:
        return f"Error adding parts to BOM: {str(e)}"


@tool
def create_bom_from_schematic(image_url: str, bom_name: str, parent_path: str = "",
                              description: str = "BOM created from schematic analysis") -> str:
    """Complete workflow: Analyze schematic and create BOM with the identified components.

    Args:
        image_url: URL of the schematic image
        bom_name: Name for the new BOM
        parent_path: Optional parent path for BOM organization
        description: Optional description for the BOM

    Returns:
        JSON response with complete workflow results
    """
    try:
        deps = get_dependencies()

        # Execute complete workflow
        workflow_result = deps.workflow_service.create_bom_from_schematic(
            image_url=image_url,
            bom_name=bom_name,
            parent_path=parent_path,
            description=description
        )

        # Convert workflow result to dict for JSON serialization
        result_dict = {
            "success": workflow_result.success,
            "workflow_steps": workflow_result.workflow_steps,
            "summary": workflow_result.summary,
            "error_message": workflow_result.error_message
        }

        # Add detailed results if available
        if workflow_result.search_result:
            result_dict["components_found"] = len(workflow_result.search_result.components)
            result_dict["successful_searches"] = workflow_result.search_result.successful_searches
            result_dict["failed_searches"] = workflow_result.search_result.failed_searches

        if workflow_result.bom_creation:
            result_dict["bom_creation_status"] = workflow_result.bom_creation.get("Status", {})

        if workflow_result.parts_addition:
            result_dict["parts_addition_status"] = workflow_result.parts_addition.get("Status", {})

        return json.dumps(result_dict, indent=2)

    except Exception as e:
        return f"Error in create BOM from schematic workflow: {str(e)}"


@tool
def parametric_search(product_line: str,
                      selected_filters: str = "[]",
                      level: int = 3,
                      keyword: str = "",
                      page_number: int = 1,
                      page_size: int = 50) -> str:
    """Search parts by technical criteria using parametric search.

    Args:
        product_line: Product line name (e.g., "Laser Diodes", "MOSFETs")
        selected_filters: JSON array of filter objects with fetName and values
        level: Taxonomy tree level (1=main category, 2=sub category, 3=product line)
        keyword: Filter on part number, description, or manufacturer name
        page_number: Page number for pagination (default: 1)
        page_size: Number of parts per page (default: 50, max: 500)

    Returns:
        JSON response with parametric search results

    Examples:
        # Search laser diodes with current range filter
        parametric_search("Laser Diodes", '[{"fetName": "Maximum Output Current", "values":[{"value": "40000 TO 50000"}]}]')

        # Search MOSFETs with multiple values for same feature
        parametric_search("MOSFETs", '[{"fetName": "Typical Gate Charge @ Vgs", "values":[{"value": "0.49"}, {"value": "170"}]}]')

        # Search with multiple features
        parametric_search("MOSFETs", '[{"fetName": "Power Supply Type", "values":[{"value": "Single"}]}, {"fetName": "Maximum Single Supply Voltage", "values":[{"value": "3 to 4"}]}]')
    """
    try:
        deps = get_dependencies()

        # Parse filters from JSON
        try:
            filters_list = json.loads(selected_filters) if selected_filters else []
            if not isinstance(filters_list, list):
                return "Error: selected_filters must be a JSON array"
        except json.JSONDecodeError as e:
            return f"Error: Invalid JSON format for selected_filters - {str(e)}"

        # Perform parametric search
        result = deps.silicon_expert_client.parametric_search(
            product_line=product_line,
            selected_filters=filters_list if filters_list else None,
            level=level,
            keyword=keyword,
            page_number=page_number,
            page_size=page_size
        )

        return json.dumps(result, indent=2)

    except Exception as e:
        return f"Error in parametric search: {str(e)}"


# Export tools list
def get_tools():
    """Get all available tools."""
    return [
        analyze_schematic,
        search_component_data,
        create_empty_bom,
        get_boms,
        add_parts_to_bom,
        create_bom_from_schematic,
        parametric_search
    ]