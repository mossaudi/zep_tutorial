# tools.py
"""Refactored LangGraph tools with enhanced descriptions and optimized parametric search."""

import json
from typing import Optional

from langchain.tools import tool

from clients.silicon_expert import SiliconExpertClient
from config import AppConfig
from exceptions import AgentError, DataValidationError
from models import Component
from services.analysis import ComponentAnalysisService
from services.formatter import ComponentTableFormatter, ComponentDataConverter
from services.workflow import BOMWorkflowService, BOMManagementService


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
    """🔍 ANALYZE SCHEMATIC: Extracts structured component data from a schematic image.

    This tool analyzes a schematic image and returns a raw, structured JSON string.
    The agent should then inspect this JSON to decide the next best action, such as
    using 'parametric_search' or 'search_component_data'.

    Args:
        image_url: URL of the schematic image to analyze

    Returns:
        A string containing ONLY the extracted JSON data from the analysis.
    """
    try:
        deps = get_dependencies()
        # This service call already returns a clean JSON string or raises an exception.
        analysis_json = deps.analysis_service.analyze_schematic(image_url)

        # --- FIX: Return ONLY the JSON string. ---
        # Do not wrap it in conversational text. The agent's system prompt
        # is already designed to handle and reason over the raw JSON output.
        return analysis_json

    except Exception as e:
        # Returning a structured error is also a good practice.
        error_payload = {"error": "An exception occurred during schematic analysis.", "details": str(e)}
        return json.dumps(error_payload)


@tool
def parametric_search(product_line: str,
                      selected_filters: str = "[]",
                      level: int = 3,
                      keyword: str = "",
                      page_number: int = 1,
                      page_size: int = 50) -> str:
    """🎯 PARAMETRIC SEARCH: Search components by technical specifications (PREFERRED for technical specs).

    🚀 PRIORITY TOOL: Use this when you have:
    - Product line names (MOSFETs, Microcontrollers, Operational Amplifiers, etc.)
    - Technical specifications (voltage, current, package type)
    - Structured component data with plName/selectedFilters
    - Components from analyze_schematic output

    🔧 WHEN TO USE:
    - Previous tool output contains "plName" and "selectedFilters"
    - User mentions specific component categories
    - Need precise filtering by technical parameters
    - Want the most accurate component matching for technical specs

    💡 WHAT HAPPENS NEXT:
    - Create new BOM with results
    - Add components to existing BOM
    - Refine search with additional filters

    Args:
        product_line: Product line name (e.g., "Laser Diodes", "MOSFETs")
        selected_filters: JSON array of filter objects with fetName and values
        level: Taxonomy tree level (1=main category, 2=sub category, 3=product line)
        keyword: Filter on part number, description, or manufacturer name
        page_number: Page number for pagination (default: 1)
        page_size: Number of parts per page (default: 50, max: 500)

    Returns:
        JSON response with parametric search results plus intelligent suggestions

    Examples:
        # Search laser diodes with current range filter
        parametric_search("Laser Diodes", '[{"fetName": "Maximum Output Current", "values":[{"value": "40000 TO 50000"}]}]')

        # Search MOSFETs with multiple values for same feature
        parametric_search("MOSFETs", '[{"fetName": "Typical Gate Charge @ Vgs", "values":[{"value": "0.49"}, {"value": "170"}]}]')
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

        # Enhanced result processing
        if result.get('Status', {}).get('Success') == 'true':
            search_results = result.get('Result', {})
            total_items = search_results.get('TotalItems', 0)
            parts = search_results.get('PartsList', [])

            # Add intelligent suggestions to the response
            enhanced_result = result.copy()
            enhanced_result['IntelligentSuggestions'] = {
                'success': True,
                'total_found': total_items,
                'showing': len(parts),
                'next_steps': [
                    f"Found {total_items} {product_line} matching your criteria",
                    "Use 'create_empty_bom' to create a new BOM with selected parts",
                    "Or use 'add_parts_to_bom' to add to existing BOM",
                    "Refine search with additional filters if needed"
                ],
                'parametric_optimization': f"✅ Parametric search optimal for {product_line}"
            }

            return json.dumps(enhanced_result, indent=2)
        else:
            return json.dumps(result, indent=2)

    except Exception as e:
        return f"Error in parametric search: {str(e)}"


@tool
def search_component_data(components_json: str) -> str:
    """🔍 SEARCH COMPONENT DATA: General component search using keywords (use when parametric search isn't suitable).

    🎯 BEST FOR:
    - Known part numbers or basic component information
    - When you have component JSON data without technical specifications
    - General component database searches
    - Alternative when parametric search isn't suitable

    🔧 WHEN TO USE:
    - Have component names/part numbers but no technical specs
    - Need to enhance basic component information
    - Parametric data is not available
    - Searching by manufacturer or generic descriptions

    💡 WHAT HAPPENS NEXT:
    - Create BOM with found components
    - Add to existing BOM
    - Use results for further analysis

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

        from models import SearchResult
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
    """📋 GET BOMS: List existing Bill of Materials for BOM management and organization.

    🎯 BEST FOR:
    - Viewing existing BOM projects
    - Finding BOM names for updates
    - Managing BOM inventory

    🔧 WHEN TO USE:
    - Need to see available BOMs
    - Looking for specific BOM to update
    - BOM management and organization

    💡 WHAT HAPPENS NEXT:
    - Use 'add_parts_to_bom' to update specific BOM
    - Reference BOM names for other operations

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
    """➕ ADD PARTS TO BOM: Add components to existing BOM for BOM updates and population.

    🎯 BEST FOR:
    - Adding components to existing BOM
    - Updating BOM with new parts
    - Populating empty BOM structure

    🔧 WHEN TO USE:
    - Have existing BOM name
    - Want to add component data to BOM
    - Follow-up after component search/analysis

    💡 WHAT HAPPENS NEXT:
    - BOM is updated with new components
    - Use 'get_boms' to verify additions

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
    """🚀 COMPLETE WORKFLOW: Full schematic-to-BOM automation (end-to-end solution).

    🎯 BEST FOR:
    - End-to-end automation from schematic to finished BOM
    - When you want complete workflow in one step
    - Time-saving automated BOM creation

    🔧 WHEN TO USE:
    - Have schematic image and want finished BOM
    - Don't need intermediate steps or customization
    - Want fastest path from schematic to BOM

    💡 WHAT HAPPENS NEXT:
    - Complete BOM ready for use
    - Use 'get_boms' to access created BOM

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


def _generate_parametric_suggestions(components) -> str:
    """Generate parametric search suggestions from components."""
    parametric_components = []

    for comp in components:
        # Check if component has technical specifications that would benefit from parametric search
        if (comp.silicon_expert_data and
                comp.silicon_expert_data.product_line and
                _has_technical_specs(comp)):
            parametric_components.append({
                'name': comp.name,
                'product_line': comp.silicon_expert_data.product_line,
                'part_number': comp.effective_part_number,
                'manufacturer': comp.effective_manufacturer
            })

    if not parametric_components:
        return ""

    suggestions = ""
    for comp in parametric_components[:5]:  # Limit to top 5
        suggestions += f"📊 {comp['name']}: {comp['product_line']} ({comp['manufacturer']})\n"
        suggestions += f"   Part: {comp['part_number']}\n\n"

    return suggestions


def _has_technical_specs(component) -> bool:
    """Check if component has technical specifications suitable for parametric search."""
    if not component.silicon_expert_data:
        return False

    # Check for product lines that typically have rich parametric data
    parametric_product_lines = [
        'MOSFETs', 'Microcontrollers', 'Operational Amplifiers', 'Voltage Regulators',
        'Laser Diodes', 'Power Management', 'Analog Switches', 'Logic Gates',
        'Memory', 'Processors', 'Transceivers', 'Comparators', 'ADCs', 'DACs'
    ]

    product_line = component.silicon_expert_data.product_line or ""
    return any(pl.lower() in product_line.lower() for pl in parametric_product_lines)


def _has_parametric_data(components) -> bool:
    """Check if any components have data suitable for parametric search."""
    return any(_has_technical_specs(comp) for comp in components)


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