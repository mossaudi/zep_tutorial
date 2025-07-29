# services/workflow.py
"""Enhanced workflow orchestration service with intelligent search integration."""

import json
from typing import List, Dict, Any, Optional

from Bom_Chatbot.models import BOMInfo, WorkflowResult, EnhancedComponent, Component
from Bom_Chatbot.constants import DEFAULT_BOM_COLUMNS
from Bom_Chatbot.services.analysis import ComponentAnalysisService
from Bom_Chatbot.services.formatter import ComponentDataConverter
from Bom_Chatbot.clients.silicon_expert import SiliconExpertClient
from Bom_Chatbot.services.progress import get_progress_tracker
from Bom_Chatbot.exceptions import BOMError, AgentError


class EnhancedBOMWorkflowService:
    """Enhanced orchestration with intelligent search capabilities."""

    def __init__(self,
                 analysis_service: ComponentAnalysisService,
                 silicon_expert_client: SiliconExpertClient):
        self.analysis_service = analysis_service
        self.silicon_expert_client = silicon_expert_client
        self.progress = get_progress_tracker()
        self.converter = ComponentDataConverter()

    def _parse_analysis_json(self, analysis_json: str) -> List[Dict[str, Any]]:
        """
        Robust parsing of analysis JSON that handles multiple possible formats.

        Expected formats:
        1. {"components": [...]} - Full analysis result
        2. [...] - Direct component array
        3. {"boardInfo": {...}, "components": [...], ...} - Complete schematic analysis
        """
        try:
            analysis_data = json.loads(analysis_json)
        except json.JSONDecodeError as e:
            raise AgentError(f"Failed to parse analysis JSON: {str(e)}")

        # Case 1: Direct array of components
        if isinstance(analysis_data, list):
            self.progress.info("JSON Parsing", "Found direct component array format")
            return analysis_data

        # Case 2: Object with 'components' key
        elif isinstance(analysis_data, dict):
            if 'components' in analysis_data:
                self.progress.info("JSON Parsing", "Found object with 'components' key")
                components_data = analysis_data['components']
                if isinstance(components_data, list):
                    return components_data
                else:
                    raise AgentError("'components' field must be an array")

            # Case 3: Object that might be a single component (fallback)
            elif any(key in analysis_data for key in ['plName', 'designator', 'name']):
                self.progress.info("JSON Parsing", "Found single component object, wrapping in array")
                return [analysis_data]

            # Case 4: Empty object or unexpected structure
            else:
                raise AgentError(
                    f"Unexpected JSON structure. Expected 'components' key or direct array. Found keys: {list(analysis_data.keys())}")

        else:
            raise AgentError(f"Unexpected JSON type: {type(analysis_data)}. Expected object or array.")

    def _convert_to_component_objects(self, components_data: List[Dict[str, Any]]) -> List[Component]:
        """Convert parsed JSON data to Component objects."""
        components = []

        for i, item in enumerate(components_data):
            if not isinstance(item, dict):
                self.progress.warning("Component Conversion", f"Skipping non-object item at index {i}")
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

            component = Component(
                name=designator,  # Use designator as the primary name
                part_number=item.get('part_number'),  # Will be filled by search
                manufacturer=item.get('manufacturer'),  # Will be filled by search
                description=item.get('description'),  # Will be filled by search
                value=item.get('value'),
                features=item.get('features'),
                quantity=quantity,
                designator=designator,
                functional_block=item.get('functionalBlock') or item.get('functional_block'),
                notes=item.get('notes'),
                pl_name=item.get('plName') or item.get('pl_name'),
                selected_filters=item.get('selectedFilters', []) or item.get('selected_filters', [])
            )
            components.append(component)

        return components

    def _get_taxonomy_data(self) -> Dict[str, Any]:
        """Get taxonomy data for internal use."""
        try:
            taxonomy_result = self.silicon_expert_client.get_taxonomy()

            if taxonomy_result.get('Status', {}).get('Success') == 'true':
                product_lines = self.silicon_expert_client.get_all_product_lines()
                return {
                    'success': True,
                    'taxonomy': taxonomy_result,
                    'product_lines': product_lines
                }
            return {'success': False, 'error': 'Failed to retrieve taxonomy'}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _keyword_search_fallback(self, components: List[Component]) -> List[EnhancedComponent]:
        """Fallback keyword search using combined component data."""
        try:
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
            return self.silicon_expert_client.search_components(keyword_components)

        except Exception as e:
            self.progress.error("Keyword Search Fallback", str(e))
            return []

    def _search_with_fallback(self, component: Component) -> Optional[EnhancedComponent]:
        """Smart parametric search with automatic fallback to keyword search."""

        # keyword search
        self.progress.info("Smart Search", f"Falling back to keyword search for {component.name}")
        try:
            fallback_results = self._keyword_search_fallback([component])
            if fallback_results and fallback_results[0].silicon_expert_data:
                enhanced = fallback_results[0]
                self.progress.success("Keyword Search", f"Found data for {component.name}")
                return enhanced
        except Exception as e:
            self.progress.warning("Keyword Search", f"Failed for {component.name}: {str(e)}")

        # Final fallback: Return component with no data
        self.progress.warning("Component Search", f"No data found for {component.name}")
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


    # Keep original method for backwards compatibility
    def create_bom_from_schematic(self, *args, **kwargs) -> WorkflowResult:
        """Backwards compatible method that uses enhanced search."""
        return self.create_bom_from_schematic_with_smart_search(*args, **kwargs)

    def add_components_to_existing_bom(self,
                                       bom_name: str,
                                       parent_path: str,
                                       components: List[EnhancedComponent]) -> Dict[str, Any]:
        """Add components to an existing BOM."""
        self.progress.info("BOM Update", f"Adding {len(components)} components to BOM: {bom_name}")

        try:
            parts_for_bom = self.converter.components_to_bom_parts(components)

            result = self.silicon_expert_client.add_parts_to_bom(
                bom_name, parent_path, parts_for_bom
            )

            if result.get("Status", {}).get("Success") == "true":
                self.progress.success("BOM Update", f"Successfully added {len(parts_for_bom)} parts")
            else:
                error_msg = "Failed to add parts to BOM"
                if "Status" in result and "Message" in result["Status"]:
                    error_msg += f": {result['Status']['Message']}"
                self.progress.error("BOM Update", error_msg)

            return result

        except Exception as e:
            self.progress.error("BOM Update", str(e))
            raise BOMError(f"Failed to add components to BOM: {str(e)}", bom_name=bom_name)


class BOMManagementService:
    """Service for BOM management operations (unchanged)."""

    def __init__(self, silicon_expert_client: SiliconExpertClient):
        self.silicon_expert_client = silicon_expert_client
        self.progress = get_progress_tracker()

    def create_empty_bom(self,
                         name: str,
                         columns: List[str],
                         description: str = "",
                         parent_path: str = "") -> Dict[str, Any]:
        """Create an empty BOM with specified structure."""
        self.progress.info("BOM Creation", f"Creating empty BOM: {name}")

        try:
            bom_info = BOMInfo(
                name=name,
                columns=columns,
                description=description,
                parent_path=parent_path
            )

            result = self.silicon_expert_client.create_empty_bom(bom_info)

            if result.get("Status", {}).get("Success") == "true":
                self.progress.success("BOM Creation", f"BOM '{name}' created successfully")
            else:
                error_msg = "Failed to create BOM"
                if "Status" in result and "Message" in result["Status"]:
                    error_msg += f": {result['Status']['Message']}"
                self.progress.error("BOM Creation", error_msg)

            return result

        except Exception as e:
            self.progress.error("BOM Creation", str(e))
            raise BOMError(f"Failed to create BOM: {str(e)}", bom_name=name)

    def get_boms(self,
                 project_name: str = "",
                 bom_creation_date_from: str = "",
                 bom_creation_date_to: str = "",
                 bom_modification_date_from: str = "",
                 bom_modification_date_to: str = "") -> Dict[str, Any]:
        """Get BOM information with optional filters."""
        self.progress.info("BOM Retrieval", "Fetching BOM information...")

        try:
            result = self.silicon_expert_client.get_boms(
                project_name=project_name,
                bom_creation_date_from=bom_creation_date_from,
                bom_creation_date_to=bom_creation_date_to,
                bom_modification_date_from=bom_modification_date_from,
                bom_modification_date_to=bom_modification_date_to
            )

            # Count BOMs in result
            bom_count = 0
            if result.get("Status", {}).get("Success") == "true" and "Result" in result:
                bom_count = len(result["Result"]) if isinstance(result["Result"], list) else 0

            self.progress.success("BOM Retrieval", f"Retrieved information for {bom_count} BOMs")
            return result

        except Exception as e:
            self.progress.error("BOM Retrieval", str(e))
            raise BOMError(f"Failed to retrieve BOMs: {str(e)}")

    def add_parts_to_bom(self,
                         name: str,
                         parent_path: str,
                         parts_data: str) -> Dict[str, Any]:
        """Add parts to an existing BOM from JSON data."""
        self.progress.info("Parts Addition", f"Adding parts to BOM: {name}")

        try:
            # Parse parts JSON
            if isinstance(parts_data, str):
                parts_list = json.loads(parts_data)
            else:
                parts_list = parts_data

            # Validate parts data
            if not isinstance(parts_list, list):
                raise AgentError("Parts data must be a JSON array")

            for i, part in enumerate(parts_list):
                if not isinstance(part, dict):
                    raise AgentError(f"Part {i + 1} must be a JSON object")
                if 'mpn' not in part or 'manufacturer' not in part:
                    raise AgentError(f"Part {i + 1} must have 'mpn' and 'manufacturer' fields")

            result = self.silicon_expert_client.add_parts_to_bom(
                name, parent_path, parts_list
            )

            if result.get("Status", {}).get("Success") == "true":
                self.progress.success("Parts Addition", f"Successfully added {len(parts_list)} parts")
            else:
                error_msg = "Failed to add parts to BOM"
                if "Status" in result and "Message" in result["Status"]:
                    error_msg += f": {result['Status']['Message']}"
                self.progress.error("Parts Addition", error_msg)

            return result

        except json.JSONDecodeError as e:
            error_msg = f"Invalid JSON in parts data: {str(e)}"
            self.progress.error("Parts Addition", error_msg)
            raise AgentError(error_msg)

        except Exception as e:
            self.progress.error("Parts Addition", str(e))
            raise BOMError(f"Failed to add parts to BOM: {str(e)}", bom_name=name)