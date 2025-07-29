# tools.py - Complete fix with robust JSON parsing
"""Enhanced LangGraph tools with robust JSON parsing and improved fallback logic."""

import json
from typing import Optional, List, Dict, Any

from langchain.tools import tool

from clients.silicon_expert import SiliconExpertClient
from config import AppConfig
from exceptions import AgentError, DataValidationError
from models import Component, EnhancedComponent, ParametricSearchResult
from services.analysis import ComponentAnalysisService
from services.formatter import ComponentTableFormatter, ComponentDataConverter
from services.workflow import EnhancedBOMWorkflowService, BOMManagementService


class ToolDependencies:
    """Container for tool dependencies."""

    def __init__(self, config: AppConfig, llm):
        self.config = config
        self.llm = llm

        # Initialize clients and services
        self.silicon_expert_client = SiliconExpertClient(config.silicon_expert)
        self.analysis_service = ComponentAnalysisService(llm, self.silicon_expert_client)
        self.workflow_service = EnhancedBOMWorkflowService(self.analysis_service, self.silicon_expert_client)
        self.bom_service = BOMManagementService(self.silicon_expert_client)
        self.formatter = ComponentTableFormatter()
        self.converter = ComponentDataConverter()


# Global dependencies instance
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


def _parse_analysis_json(analysis_json: str) -> List[Dict[str, Any]]:
    """
    Robust parsing of analysis JSON that handles multiple possible formats with comprehensive error handling.

    Expected formats:
    1. {"components": [...]} - Full analysis result
    2. [...] - Direct component array
    3. {"boardInfo": {...}, "components": [...], ...} - Complete schematic analysis
    4. Single component object - Will be wrapped in array
    5. Empty or malformed JSON - Will provide helpful error messages
    """
    if not analysis_json or not analysis_json.strip():
        raise AgentError("Empty or null analysis JSON received")

    try:
        analysis_data = json.loads(analysis_json)
    except json.JSONDecodeError as e:
        # Try to provide helpful error context
        error_location = getattr(e, 'pos', 0)
        context_start = max(0, error_location - 50)
        context_end = min(len(analysis_json), error_location + 50)
        context = analysis_json[context_start:context_end]

        raise AgentError(
            f"Failed to parse analysis JSON at position {error_location}. "
            f"Context: '...{context}...'. Error: {str(e)}"
        )

    # Handle None or empty data
    if analysis_data is None:
        raise AgentError("Analysis JSON parsed to None - no data available")

    # Case 1: Direct array of components
    if isinstance(analysis_data, list):
        if not analysis_data:
            raise AgentError("Analysis JSON contains empty component array")

        # Validate that it's actually an array of component-like objects
        return _validate_component_data(analysis_data)

    # Case 2: Dictionary structures
    elif isinstance(analysis_data, dict):

        # Case 2a: Standard format with 'components' key
        if 'components' in analysis_data:
            components_data = analysis_data['components']

            if components_data is None:
                raise AgentError("'components' field is null in analysis JSON")

            if not isinstance(components_data, list):
                # Try to convert single component to list
                if isinstance(components_data, dict):
                    return [components_data]
                else:
                    raise AgentError(
                        f"'components' field must be array or object, got {type(components_data).__name__}")

            if not components_data:
                raise AgentError("'components' array is empty in analysis JSON")

            return _validate_component_data(components_data)

        # Case 2b: Object that might be a single component
        elif any(key in analysis_data for key in ['plName', 'designator', 'name', 'component_name', 'part_number']):
            return [analysis_data]

        # Case 2c: Try common alternative keys
        elif 'result' in analysis_data:
            result_data = analysis_data['result']
            if isinstance(result_data, list):
                return _validate_component_data(result_data)
            elif isinstance(result_data, dict) and 'components' in result_data:
                return _validate_component_data(result_data['components'])

        elif 'data' in analysis_data:
            data_field = analysis_data['data']
            if isinstance(data_field, list):
                return _validate_component_data(data_field)
            elif isinstance(data_field, dict) and 'components' in data_field:
                return _validate_component_data(data_field['components'])

        # Case 2d: Check for nested structures
        elif 'Result' in analysis_data:  # Silicon Expert style
            result_data = analysis_data['Result']
            if isinstance(result_data, dict):
                if 'components' in result_data:
                    return _validate_component_data(result_data['components'])
                elif 'PartsList' in result_data:
                    return _validate_component_data(result_data['PartsList'])

        # Case 2e: Unknown object structure - provide detailed debugging info
        else:
            available_keys = list(analysis_data.keys())
            sample_data = str(analysis_data)[:200] + "..." if len(str(analysis_data)) > 200 else str(analysis_data)

            raise AgentError(
                f"Unexpected JSON structure. Expected 'components' key or component-like fields. "
                f"Available keys: {available_keys}. "
                f"Sample data: {sample_data}"
            )

    # Case 3: Unexpected types
    else:
        raise AgentError(
            f"Unexpected JSON type: {type(analysis_data).__name__}. Expected object or array. "
            f"Value: {str(analysis_data)[:100]}..."
        )


def _validate_component_data(components_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Additional validation to ensure component data is usable."""
    if not components_data:
        raise AgentError("No component data provided for validation")

    valid_components = []
    warnings = []

    for i, component in enumerate(components_data):
        if not isinstance(component, dict):
            warnings.append(f"Component {i + 1}: Not a dictionary, skipping")
            continue

        # Check for minimum required fields or recognizable patterns
        has_identifier = any(
            component.get(field) for field in
            ['plName', 'designator', 'name', 'component_name', 'part_number', 'mpn', 'selectedFilters']
        )

        if not has_identifier:
            warnings.append(f"Component {i + 1}: No recognizable identifier fields, but keeping anyway")

        valid_components.append(component)

    if warnings:
        print("Component validation warnings:")
        for warning in warnings:
            print(f"  - {warning}")

    if not valid_components:
        raise AgentError("No valid components found after validation")

    return valid_components


def _convert_to_component_objects(components_data: List[Dict[str, Any]]) -> List[Component]:
    """Convert parsed JSON data to Component objects with enhanced error handling."""
    components = []
    errors = []

    for i, item in enumerate(components_data):
        try:
            if not isinstance(item, dict):
                errors.append(f"Component {i + 1}: Expected dictionary, got {type(item).__name__}")
                continue

            # Handle different naming conventions for designator/name
            designator = (item.get('designator') or
                          item.get('name') or
                          item.get('component_name') or
                          f'Component_{i + 1}')

            # Extract quantity - handle both string and numeric
            quantity = item.get('quantity', '1')
            if isinstance(quantity, (int, float)):
                quantity = str(quantity)
            elif not isinstance(quantity, str):
                quantity = '1'

            # Handle selectedFilters properly
            selected_filters = item.get('selectedFilters', [])
            if selected_filters is None:
                selected_filters = []
            elif not isinstance(selected_filters, list):
                selected_filters = []

            component = Component(
                name=designator,  # Use designator as the primary name
                part_number=item.get('part_number'),
                manufacturer=item.get('manufacturer'),
                description=item.get('description'),
                value=item.get('value'),
                features=item.get('features'),
                quantity=quantity,
                designator=designator,
                functional_block=item.get('functionalBlock') or item.get('functional_block'),
                notes=item.get('notes'),
                pl_name=item.get('plName') or item.get('pl_name'),
                selected_filters=selected_filters
            )
            components.append(component)

        except Exception as e:
            errors.append(f"Component {i + 1}: Error creating component object - {str(e)}")
            continue

    if errors:
        print("Component conversion errors:")
        for error in errors:
            print(f"  - {error}")

    if not components:
        raise AgentError(f"No valid components could be created. Errors: {'; '.join(errors)}")

    return components


# HIDDEN HELPER FUNCTIONS (not exposed as tools)
def _get_taxonomy_data() -> Dict[str, Any]:
    """Hidden helper: Get taxonomy data for internal use."""
    try:
        deps = get_dependencies()
        taxonomy_result = deps.silicon_expert_client.get_taxonomy()

        if taxonomy_result.get('Status', {}).get('Success') == 'true':
            product_lines = deps.silicon_expert_client.get_all_product_lines()
            return {
                'success': True,
                'taxonomy': taxonomy_result,
                'product_lines': product_lines
            }
        return {'success': False, 'error': 'Failed to retrieve taxonomy'}
    except Exception as e:
        return {'success': False, 'error': str(e)}


def _keyword_search_fallback(components: List[Component]) -> List[EnhancedComponent]:
    """Hidden helper: Fallback keyword search using combined component data."""
    try:
        deps = get_dependencies()

        # Create enhanced keyword search components
        keyword_components = []
        for component in components:
            # Combine all available data into description for comprehensive search
            search_parts = []

            if component.part_number:
                search_parts.append(component.part_number)
            if component.manufacturer:
                search_parts.append(component.manufacturer)
            if component.name:
                search_parts.append(component.name)
            if component.value:
                search_parts.append(component.value)
            if component.features:
                search_parts.append(component.features)
            if component.description:
                search_parts.append(component.description)

            # Create a keyword-optimized component
            keyword_component = Component(
                name=component.name,
                part_number=component.part_number,
                manufacturer=component.manufacturer,
                description=" ".join(search_parts).strip(),  # Combined search string
                value=component.value,
                features=component.features,
                quantity=component.quantity,
                designator=component.designator,
                functional_block=component.functional_block,
                notes=component.notes,
                pl_name=component.pl_name,
                selected_filters=component.selected_filters
            )
            keyword_components.append(keyword_component)

        # Perform search with enhanced descriptions
        return deps.silicon_expert_client.search_components(keyword_components)

    except Exception as e:
        print(f"Keyword search fallback failed: {str(e)}")
        return []


def _smart_parametric_search_with_fallback(component: Component) -> Optional[EnhancedComponent]:
    """Hidden helper: Smart parametric search with automatic fallback to keyword search."""
    deps = get_dependencies()

    # First attempt: Try parametric search if plName and filters are available
    if component.pl_name and component.selected_filters:
        try:
            # Ensure taxonomy mapping
            taxonomy_data = _get_taxonomy_data()
            if taxonomy_data['success']:
                mapped_pl_name = deps.silicon_expert_client.find_matching_product_line(component.pl_name)
                if mapped_pl_name:
                    # Perform parametric search
                    result = deps.silicon_expert_client.parametric_search(
                        product_line=mapped_pl_name,
                        selected_filters=component.selected_filters,
                        level=3,
                        keyword="",
                        page_number=1,
                        page_size=10
                    )

                    if result.get('Status', {}).get('Success') == 'true':
                        parts_list = result.get('Result', {}).get('PartsList', [])
                        if parts_list:
                            # Success! Create enhanced component from first result
                            enhanced = _create_enhanced_component_from_parametric_result(
                                parts_list[0], mapped_pl_name
                            )
                            enhanced.name = component.name  # Preserve original name
                            enhanced.designator = component.designator
                            enhanced.parametric_search_used = True
                            return enhanced
        except Exception as e:
            print(f"Parametric search failed for {component.pl_name}: {str(e)}")

    # Second attempt: Fallback to keyword search
    try:
        fallback_results = _keyword_search_fallback([component])
        if fallback_results and fallback_results[0].silicon_expert_data:
            enhanced = fallback_results[0]
            enhanced.name = component.name  # Preserve original name
            enhanced.designator = component.designator
            enhanced.parametric_search_used = False
            return enhanced
    except Exception as e:
        print(f"Keyword search fallback failed for {component.name}: {str(e)}")

    # Final fallback: Return component with no data
    enhanced = EnhancedComponent(
        name=component.name,
        part_number=component.part_number,
        manufacturer=component.manufacturer,
        description=component.description,
        value=component.value,
        features=component.features,
        quantity=component.quantity,
        designator=component.designator,
        functional_block=component.functional_block,
        notes=component.notes,
        pl_name=component.pl_name,
        selected_filters=component.selected_filters,
        search_result="No data found in parametric or keyword search"
    )
    return enhanced


# PUBLIC TOOLS (exposed to user)
@tool
def analyze_schematic(image_url: str) -> str:
    """🔍 ANALYZE SCHEMATIC: Extracts and enhances component data from schematic with automatic search.

    This enhanced tool now automatically performs intelligent component searching:
    1. Extracts components from schematic
    2. For each component with plName/selectedFilters: tries parametric search
    3. Falls back to keyword search if parametric search fails
    4. Returns comprehensive component data with enhanced information

    Args:
        image_url: URL of the schematic image to analyze

    Returns:
        Formatted table with enhanced component data from automatic searches
    """
    try:
        deps = get_dependencies()

        # Step 1: Analyze schematic to get raw component data
        print("🔍 Starting schematic analysis...")
        analysis_json = deps.analysis_service.analyze_schematic(image_url)

        # Step 2: Robust parsing using helper function
        try:
            print("📝 Parsing analysis JSON...")
            components_data = _parse_analysis_json(analysis_json)

            if not components_data:
                return "Error: No components found in schematic analysis"

            print(f"✅ Found {len(components_data)} component entries")

        except AgentError as e:
            return f"Error parsing schematic analysis: {str(e)}"
        except Exception as e:
            return f"Error: Unexpected parsing error - {str(e)}"

        # Step 3: Convert to Component objects using helper function
        try:
            print("🔧 Converting to component objects...")
            components = _convert_to_component_objects(components_data)

            if not components:
                return "Error: No valid components found after conversion"

            print(f"✅ Successfully converted {len(components)} components")

        except Exception as e:
            return f"Error converting component data: {str(e)}"

        # Step 4: Smart search with automatic fallback
        print("🔍 Starting intelligent component search...")
        enhanced_components = []
        successful_parametric = 0
        successful_keyword = 0
        failed_searches = 0

        for i, component in enumerate(components, 1):
            print(f"Processing component {i}/{len(components)}: {component.name}")
            enhanced = _smart_parametric_search_with_fallback(component)
            if enhanced:
                if enhanced.parametric_search_used and enhanced.silicon_expert_data:
                    successful_parametric += 1
                elif not enhanced.parametric_search_used and enhanced.silicon_expert_data:
                    successful_keyword += 1
                else:
                    failed_searches += 1
                enhanced_components.append(enhanced)

        # Step 5: Format results
        from models import SearchResult
        search_result = SearchResult(
            success=True,
            components=enhanced_components,
            successful_searches=successful_parametric + successful_keyword,
            failed_searches=failed_searches
        )

        # Enhanced table output with search method indicators
        table_output = deps.formatter.format_search_result(search_result)

        # Add search method summary
        summary = f"\n🔍 SEARCH SUMMARY:\n"
        summary += f"• Parametric Search Success: {successful_parametric} components\n"
        summary += f"• Keyword Search Success: {successful_keyword} components\n"
        summary += f"• No Data Found: {failed_searches} components\n"
        summary += f"• Total Processed: {len(components)} components\n"

        return f"SCHEMATIC_ANALYSIS_COMPLETE:{table_output}{summary}"

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Full error details: {error_details}")
        return f"Error in enhanced schematic analysis: {str(e)}"


def _create_enhanced_component_from_parametric_result(part_data: Dict[str, Any],
                                                      product_line: str) -> EnhancedComponent:
    """Create enhanced component from parametric search result."""
    from models import SiliconExpertData, ParametricFeature

    component = EnhancedComponent(
        name=part_data.get('PartNumber', 'Unknown'),
        part_number=part_data.get('PartNumber'),
        manufacturer=part_data.get('Manufacturer'),
        description=part_data.get('Description'),
        quantity="1",
        parametric_search_used=True
    )

    se_data = SiliconExpertData(
        com_id=part_data.get('ComID'),
        part_number=part_data.get('PartNumber'),
        manufacturer=part_data.get('Manufacturer'),
        description=part_data.get('Description'),
        lifecycle=part_data.get('Lifecycle'),
        rohs=part_data.get('RoHS'),
        rohs_version=part_data.get('RoHSVersion'),
        datasheet=part_data.get('Datasheet'),
        product_line=product_line,
        taxonomy_path=part_data.get('TaxonomyPath'),
        yeol=part_data.get('YEOL'),
        resilience_rating=part_data.get('ResilienceRating'),
        military_status=part_data.get('MilitaryStatus'),
        aml_status=part_data.get('AMLStatus')
    )

    # Extract parametric features
    features_data = part_data.get('ParametricFeatures', {})
    if isinstance(features_data, dict):
        for feature_name, feature_info in features_data.items():
            if isinstance(feature_info, dict):
                value = feature_info.get('value', str(feature_info))
                unit = feature_info.get('unit')
            else:
                value = str(feature_info)
                unit = None

            se_data.add_parametric_feature(feature_name, value, unit)

    component.silicon_expert_data = se_data
    return component


@tool
def parametric_search(product_line: str,
                      selected_filters: str = "[]",
                      level: int = 3,
                      keyword: str = "",
                      page_number: int = 1,
                      page_size: int = 50) -> str:
    """🎯 PARAMETRIC SEARCH: Enhanced parametric search with automatic taxonomy mapping.

    Args:
        product_line: Product line name (automatically mapped to taxonomy)
        selected_filters: JSON array of filter objects
        level: Taxonomy tree level (default: 3)
        keyword: Additional keyword filter
        page_number: Page number (default: 1)
        page_size: Results per page (default: 50, max: 500)

    Returns:
        Formatted table with enhanced component data
    """
    try:
        deps = get_dependencies()

        # Parse filters
        try:
            filters_list = json.loads(selected_filters) if selected_filters else []
            if not isinstance(filters_list, list):
                return "Error: selected_filters must be a JSON array"
        except json.JSONDecodeError as e:
            return f"Error: Invalid JSON format for selected_filters - {str(e)}"

        # Auto-load taxonomy for mapping (hidden from user)
        taxonomy_data = _get_taxonomy_data()
        if not taxonomy_data['success']:
            return f"Error: Could not load taxonomy for product line mapping"

        # Perform parametric search with enhanced features
        result = deps.silicon_expert_client.parametric_search(
            product_line=product_line,
            selected_filters=filters_list if filters_list else None,
            level=level,
            keyword=keyword,
            page_number=page_number,
            page_size=page_size
        )

        if result.get('Status', {}).get('Success') == 'true':
            search_results = result.get('Result', {})
            total_items = search_results.get('TotalItems', 0)
            parts_list = search_results.get('PartsList', [])

            if not parts_list:
                return json.dumps({
                    'message': f'No parts found for product line: {product_line}',
                    'suggestions': [
                        'Try different filter values',
                        'Use broader search criteria',
                        'Consider alternative product lines'
                    ]
                }, indent=2)

            # Process results
            enhanced_components = []
            for i, part in enumerate(parts_list[:20]):  # Limit for performance
                component = _create_enhanced_component_from_parametric_result(part, product_line)
                enhanced_components.append(component)

            # Create result object
            parametric_result = ParametricSearchResult(
                success=True,
                product_line=product_line,
                total_items=total_items,
                components=enhanced_components,
                search_filters=filters_list
            )

            # Format results
            table_output = deps.formatter.format_parametric_search_result(parametric_result)
            return f"PARAMETRIC_SEARCH_COMPLETE:{table_output}"

        else:
            error_msg = result.get('Status', {}).get('Message', 'Unknown error')
            return json.dumps({
                'error': f'Parametric search failed: {error_msg}',
                'product_line': product_line
            }, indent=2)

    except Exception as e:
        return f"Error in parametric search: {str(e)}"


@tool
def search_component_data(components_json: str) -> str:
    """🔍 SEARCH COMPONENT DATA: General component search using keywords.

    Args:
        components_json: JSON string containing component data

    Returns:
        Formatted table with enhanced component information
    """
    try:
        deps = get_dependencies()

        # Parse and validate components
        try:
            components_data = json.loads(components_json)
            if not isinstance(components_data, list):
                raise DataValidationError(
                    "Component data must be a JSON array",
                    "component_list",
                    ["Expected list, got " + type(components_data).__name__]
                )

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
                    quantity=item.get('quantity', '1'),
                    designator=item.get('designator'),
                    functional_block=item.get('functional_block'),
                    notes=item.get('notes'),
                    pl_name=item.get('pl_name'),
                    selected_filters=item.get('selected_filters', [])
                )
                components.append(component)

        except json.JSONDecodeError as e:
            return f"Error: Invalid JSON format - {str(e)}"
        except DataValidationError as e:
            return f"Error: {e.message}"

        if not components:
            return "Error: No valid components found in the provided data"

        # Enhanced search with fallback
        enhanced_components = []
        for component in components:
            # Try smart search with fallback
            enhanced = _smart_parametric_search_with_fallback(component)
            if enhanced:
                enhanced_components.append(enhanced)

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

        # Format results
        table_output = deps.formatter.format_search_result(search_result)
        return f"COMPONENT_SEARCH_COMPLETE:{table_output}"

    except Exception as e:
        return f"Error searching component data: {str(e)}"


# BOM Management Tools (unchanged but optimized)
@tool
def create_empty_bom(name: str, columns: str, description: str = "", parent_path: str = "") -> str:
    """📋 CREATE EMPTY BOM: Create new Bill of Materials structure."""
    try:
        deps = get_dependencies()

        try:
            columns_list = json.loads(columns) if isinstance(columns, str) else columns
            if not isinstance(columns_list, list):
                return "Error: Columns must be a JSON array"
        except json.JSONDecodeError:
            return "Error: Invalid JSON format for columns"

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
    """📋 GET BOMS: List existing Bill of Materials."""
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
    """➕ ADD PARTS TO BOM: Add components to existing BOM."""
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
    """🚀 COMPLETE WORKFLOW: Full schematic-to-BOM automation."""
    try:
        deps = get_dependencies()

        workflow_result = deps.workflow_service.create_bom_from_schematic(
            image_url=image_url,
            bom_name=bom_name,
            parent_path=parent_path,
            description=description
        )

        result_dict = {
            "success": workflow_result.success,
            "workflow_steps": workflow_result.workflow_steps,
            "summary": workflow_result.summary,
            "error_message": workflow_result.error_message
        }

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


# Export only public tools
def get_tools():
    """Get all available public tools."""
    return [
        analyze_schematic,  # Enhanced with automatic search
        parametric_search,  # Enhanced with taxonomy mapping
        search_component_data,  # Enhanced with smart fallback
        create_empty_bom,
        get_boms,
        add_parts_to_bom,
        create_bom_from_schematic
    ]