# services/formatter.py
"""Component table formatting service."""
import json
from typing import List, Dict, Any, Optional
from collections import Counter

from Bom_Chatbot.models import EnhancedComponent, SearchResult
from Bom_Chatbot.constants import DEFAULT_BOM_COLUMNS
from Bom_Chatbot.services.progress import get_progress_tracker


class ComponentTableFormatter:
    """Formats component data into readable tables with suggestions."""

    def __init__(self):
        self.progress = get_progress_tracker()

    def format_search_result(self, search_result: SearchResult) -> str:
        """Format a SearchResult into a comprehensive table display."""
        if not search_result.components:
            return "No component data available to display."

        try:
            # Import tabulate with fallback
            try:
                from tabulate import tabulate
                tabulate_available = True
            except ImportError:
                tabulate_available = False
                self.progress.warning(
                    "Import Warning",
                    "tabulate not available, using fallback formatting"
                )

            # Build table data
            table_data = self._build_table_data(search_result.components)

            # Create formatted output
            output = self._create_header()

            # Add table content
            if tabulate_available:
                output += self._format_with_tabulate(table_data)
            else:
                output += self._format_fallback(table_data)

            # Add summary and suggestions
            output += self._create_summary(search_result)
            output += self._create_suggestions()
            output += self._create_example_commands()

            return output

        except Exception as e:
            self.progress.error("Table Formatting", str(e))
            return (
                f"Error formatting component table: {str(e)}\n"
                f"Component count: {len(search_result.components)}"
            )

    def _build_table_data(self, components: List[EnhancedComponent]) -> List[Dict[str, str]]:
        """Build table data from components."""
        table_data = []

        for i, component in enumerate(components, 1):
            # Safe extraction with fallbacks
            component_name = component.name or f'Component {i}'
            part_number = component.effective_part_number
            manufacturer = component.effective_manufacturer
            description = component.effective_description

            # Truncate long descriptions
            if description != 'N/A' and len(description) > 50:
                description = description[:47] + '...'

            # Get Silicon Expert specific data
            lifecycle = 'N/A'
            rohs = 'N/A'
            match_rating = 'N/A'

            if component.silicon_expert_data:
                lifecycle = component.silicon_expert_data.lifecycle or 'N/A'
                rohs = component.silicon_expert_data.rohs or 'N/A'
                match_rating = component.silicon_expert_data.match_rating or 'N/A'

            row = {
                '#': str(i),
                'Component': component_name,
                'Part Number': part_number,
                'Manufacturer': manufacturer,
                'Description': description,
                'Lifecycle': lifecycle,
                'RoHS': rohs,
                'Match Rating': match_rating
            }
            table_data.append(row)

        return table_data

    def _create_header(self) -> str:
        """Create table header."""
        return (
                "\n" + "=" * 120 + "\n"
                                   "COMPONENT ANALYSIS RESULTS\n" +
                "=" * 120 + "\n"
        )

    def _format_with_tabulate(self, table_data: List[Dict[str, str]]) -> str:
        """Format table using tabulate library."""
        try:
            from tabulate import tabulate

            # Ensure all values are strings
            clean_table_data = []
            for row in table_data:
                clean_row = {
                    key: str(value) if value is not None else 'N/A'
                    for key, value in row.items()
                }
                clean_table_data.append(clean_row)

            return tabulate(
                clean_table_data,
                headers='keys',
                tablefmt='grid',
                maxcolwidths=[3, 20, 20, 15, 40, 12, 8, 12]
            )

        except Exception as e:
            self.progress.warning("Tabulate Formatting", f"Falling back due to: {str(e)}")
            return self._format_fallback(table_data)

    def _format_fallback(self, table_data: List[Dict[str, str]]) -> str:
        """Fallback formatting when tabulate is not available."""
        output = "Note: Using fallback formatting\n\n"

        for row in table_data:
            try:
                output += f"COMPONENT #{row.get('#', '?')}\n"
                output += f"├─ Name: {row.get('Component', 'Unknown')}\n"
                output += f"├─ Part Number: {row.get('Part Number', 'N/A')}\n"
                output += f"├─ Manufacturer: {row.get('Manufacturer', 'N/A')}\n"
                output += f"├─ Description: {row.get('Description', 'N/A')}\n"
                output += f"├─ Lifecycle: {row.get('Lifecycle', 'N/A')}\n"
                output += f"├─ RoHS: {row.get('RoHS', 'N/A')}\n"
                output += f"└─ Match Rating: {row.get('Match Rating', 'N/A')}\n"
                output += "\n" + "-" * 60 + "\n"
            except Exception as row_error:
                output += f"Error formatting component: {str(row_error)}\n"
                output += f"Raw data: {str(row)}\n\n"

        return output

    def _create_summary(self, search_result: SearchResult) -> str:
        """Create summary statistics."""
        components = search_result.components
        total_components = len(components)
        components_with_data = search_result.successful_searches

        # Count lifecycle and RoHS status
        lifecycle_counts = Counter()
        rohs_counts = Counter()

        for component in components:
            if component.silicon_expert_data:
                lifecycle = component.silicon_expert_data.lifecycle or 'Unknown'
                rohs = component.silicon_expert_data.rohs or 'Unknown'
            else:
                lifecycle = 'Unknown'
                rohs = 'Unknown'

            lifecycle_counts[lifecycle] += 1
            rohs_counts[rohs] += 1

        output = "\n" + "=" * 120 + "\n"
        output += f"\nSUMMARY:\n"
        output += f"- Total Components: {total_components}\n"
        output += f"- Components with Silicon Expert Data: {components_with_data}\n"
        output += f"- Data Match Rate: {search_result.success_rate:.1f}%\n"

        # Add lifecycle distribution if meaningful
        if lifecycle_counts and lifecycle_counts != Counter({'Unknown': total_components}):
            output += f"\nLifecycle Distribution:\n"
            for lifecycle, count in lifecycle_counts.most_common():
                output += f"  • {lifecycle}: {count} components\n"

        # Add RoHS compliance if meaningful
        if rohs_counts and rohs_counts != Counter({'Unknown': total_components}):
            output += f"\nRoHS Compliance:\n"
            for rohs, count in rohs_counts.most_common():
                output += f"  • {rohs}: {count} components\n"

        return output

    def _create_suggestions(self) -> str:
        """Create next steps suggestions."""
        return (
                "\n" + "=" * 120 + "\n"
                                   "SUGGESTED NEXT STEPS:\n" +
                "=" * 120 + "\n"
                            "1. CREATE NEW BOM:\n"
                            "   → Use 'create_bom_from_schematic' to automatically create a BOM with these components\n"
                            "   → Or use 'create_empty_bom' to create a custom BOM structure\n\n"

                            "2. ADD TO EXISTING BOM:\n"
                            "   → Use 'get_boms' to view existing BOMs\n"
                            "   → Then use 'add_parts_to_bom' to add these components to an existing BOM\n\n"

                            "3. COMPONENT ANALYSIS:\n"
                            "   → Review lifecycle status for obsolescence planning\n"
                            "   → Check RoHS compliance for regulatory requirements\n"
                            "   → Examine match ratings for data accuracy\n\n"

                            "4. DETAILED COMPONENT INFO:\n"
                            "   → Use 'search_component_data' with specific part numbers for detailed analysis\n"
                            "   → Access datasheets and technical specifications\n\n"
        )

    def _create_example_commands(self) -> str:
        """Create example commands section."""
        columns_json = str(DEFAULT_BOM_COLUMNS).replace("'", '"')

        return (
                "EXAMPLE COMMANDS:\n"
                f"• Create new BOM: create_empty_bom(name='MyProject_BOM', columns='{columns_json}')\n"
                "• View existing BOMs: get_boms()\n"
                "• Add parts to existing BOM: add_parts_to_bom(name='BOM_NAME', parent_path='PROJECT_PATH', parts_json='COMPONENT_DATA')\n" +
                "=" * 120 + "\n"
        )

    def format_intelligent_suggestions(self, available_data: Dict[str, Any]) -> str:
        """Generate intelligent suggestions based on available data."""
        suggestions = []

        if available_data.get('active_components'):
            component_count = available_data.get('component_count', 0)
            suggestions.append(f"🔧 You have {component_count} components ready for BOM creation")

        if available_data.get('last_bom_name'):
            bom_name = available_data['last_bom_name']
            suggestions.append(f"📋 Recent BOM: '{bom_name}' available for updates")

        if not suggestions:
            suggestions.append("🚀 Start by analyzing a schematic: 'analyze schematic at [URL]'")

        return (
                "\n" + "=" * 80 + "\n"
                                  "🧠 INTELLIGENT SUGGESTIONS:\n" +
                "=" * 80 + "\n" +
                "\n".join(f"  {suggestion}" for suggestion in suggestions) + "\n"
        )


class ComponentDataConverter:
    """Converts component data between different formats."""

    @staticmethod
    def components_to_json(components: List[EnhancedComponent]) -> str:
        """Convert enhanced components to JSON format."""
        component_dicts = []

        for component in components:
            # Create base component dict
            component_dict = {
                'name': component.name,
                'part_number': component.part_number,
                'manufacturer': component.manufacturer,
                'description': component.description,
                'value': component.value,
                'features': component.features,
                'quantity': component.quantity
            }

            # Add Silicon Expert data if available
            if component.silicon_expert_data:
                se_data = component.silicon_expert_data
                component_dict.update({
                    'se_com_id': se_data.com_id,
                    'se_part_number': se_data.part_number,
                    'se_manufacturer': se_data.manufacturer,
                    'se_description': se_data.description,
                    'se_lifecycle': se_data.lifecycle,
                    'se_rohs': se_data.rohs,
                    'se_rohs_version': se_data.rohs_version,
                    'se_datasheet': se_data.datasheet,
                    'se_product_line': se_data.product_line,
                    'se_taxonomy_path': se_data.taxonomy_path,
                    'se_match_rating': se_data.match_rating,
                    'se_match_comment': se_data.match_comment,
                    'se_yeol': se_data.yeol,
                    'se_resilience_rating': se_data.resilience_rating,
                    'se_military_status': se_data.military_status,
                    'se_aml_status': se_data.aml_status,
                    'se_search_query': se_data.search_query,
                    'se_total_items': se_data.total_items,
                    'se_all_matches': se_data.all_matches
                })

            # Add search result if no Silicon Expert data
            if component.search_result:
                component_dict['se_search_result'] = component.search_result

            # Remove None values
            component_dict = {k: v for k, v in component_dict.items() if v is not None}
            component_dicts.append(component_dict)

        return json.dumps(component_dicts, indent=2)

    @staticmethod
    def components_to_bom_parts(components: List[EnhancedComponent]) -> List[Dict[str, str]]:
        """Convert enhanced components to BOM parts format."""
        return [component.to_bom_part() for component in components]