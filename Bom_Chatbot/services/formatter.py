# services/formatter.py
"""Enhanced component table formatting service with parametric search support."""
import json
from typing import List, Dict, Any, Optional
from collections import Counter

from Bom_Chatbot.models import EnhancedComponent, SearchResult, ParametricSearchResult
from Bom_Chatbot.constants import DEFAULT_BOM_COLUMNS
from Bom_Chatbot.services.progress import get_progress_tracker


class ComponentTableFormatter:
    """Enhanced formatter for component data with parametric features support."""

    def __init__(self):
        self.progress = get_progress_tracker()

    def format_parametric_search_result(self, parametric_result: ParametricSearchResult) -> str:
        """Format a ParametricSearchResult into a comprehensive table display."""
        if not parametric_result.components:
            return self._create_empty_parametric_result(parametric_result)

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

            # Build enhanced table data
            table_data = self._build_parametric_table_data(parametric_result.components)

            # Create formatted output
            output = self._create_parametric_header(parametric_result)

            # Add table content
            if tabulate_available:
                output += self._format_parametric_with_tabulate(table_data)
            else:
                output += self._format_parametric_fallback(table_data)

            # Add parametric features summary
            output += self._create_parametric_features_summary(parametric_result.components)

            # Add summary and suggestions
            output += self._create_parametric_summary(parametric_result)
            output += self._create_parametric_suggestions()
            output += self._create_parametric_example_commands()

            return output

        except Exception as e:
            self.progress.error("Parametric Table Formatting", str(e))
            return (
                f"Error formatting parametric search table: {str(e)}\n"
                f"Component count: {len(parametric_result.components)}"
            )

    def _create_empty_parametric_result(self, parametric_result: ParametricSearchResult) -> str:
        """Create output for empty parametric search results."""
        return (
            f"\n{'=' * 120}\n"
            f"PARAMETRIC SEARCH RESULTS: {parametric_result.product_line}\n"
            f"{'=' * 120}\n"
            f"No components found matching the specified criteria.\n\n"
            f"SUGGESTIONS:\n"
            f"• Broaden search filters or remove some constraints\n"
            f"• Check product line name with get_taxonomy tool\n"
            f"• Try alternative product line categories\n"
            f"• Use keyword search for more general results\n"
            f"{'=' * 120}\n"
        )

    def _build_parametric_table_data(self, components: List[EnhancedComponent]) -> List[Dict[str, str]]:
        """Build enhanced table data from parametric search components."""
        table_data = []

        for i, component in enumerate(components, 1):
            # Basic component information
            component_name = component.name or f'Component {i}'
            part_number = component.effective_part_number
            manufacturer = component.effective_manufacturer
            description = component.effective_description

            # Truncate long descriptions
            if description != 'N/A' and len(description) > 40:
                description = description[:37] + '...'

            # Get Silicon Expert specific data
            lifecycle = 'N/A'
            rohs = 'N/A'
            datasheet = 'N/A'

            if component.silicon_expert_data:
                lifecycle = component.silicon_expert_data.lifecycle or 'N/A'
                rohs = component.silicon_expert_data.rohs or 'N/A'
                datasheet = 'Available' if component.silicon_expert_data.datasheet else 'N/A'

            # Extract key parametric features
            key_features = self._extract_key_parametric_features(component)

            row = {
                '#': str(i),
                'Part Number': part_number,
                'Manufacturer': manufacturer,
                'Description': description,
                'Lifecycle': lifecycle,
                'RoHS': rohs,
                'Datasheet': datasheet,
                **key_features  # Add parametric features as columns
            }
            table_data.append(row)

        return table_data

    def _extract_key_parametric_features(self, component: EnhancedComponent) -> Dict[str, str]:
        """Extract key parametric features for table display."""
        features = {}

        if not component.has_parametric_data:
            return features

        parametric_features = component.silicon_expert_data.parametric_features

        # Priority features to show in main table
        priority_features = [
            'Package Type', 'Package', 'Supply Voltage', 'Operating Temperature',
            'Power Dissipation', 'Current Rating', 'Voltage Rating',
            'Frequency', 'Capacitance', 'Resistance', 'Tolerance'
        ]

        # Add up to 4 most relevant features to keep table readable
        feature_count = 0
        for feature_name in priority_features:
            if feature_count >= 4:  # Limit to keep table width manageable
                break

            if feature_name in parametric_features:
                feature_value = str(parametric_features[feature_name])
                # Truncate long values
                if len(feature_value) > 15:
                    feature_value = feature_value[:12] + '...'
                features[feature_name] = feature_value
                feature_count += 1

        # If no priority features found, add any available features
        if not features and parametric_features:
            for name, feature_obj in list(parametric_features.items())[:4]:
                feature_value = str(feature_obj)
                if len(feature_value) > 15:
                    feature_value = feature_value[:12] + '...'
                features[name] = feature_value

        return features

    def _create_parametric_header(self, parametric_result: ParametricSearchResult) -> str:
        """Create header for parametric search results."""
        return (
            f"\n{'=' * 140}\n"
            f"PARAMETRIC SEARCH RESULTS: {parametric_result.product_line}\n"
            f"{'=' * 140}\n"
            f"Total Items Found: {parametric_result.total_items} | "
            f"Showing: {len(parametric_result.components)} | "
            f"With Parametric Data: {parametric_result.components_with_features}\n"
            f"{'=' * 140}\n"
        )

    def _format_parametric_with_tabulate(self, table_data: List[Dict[str, str]]) -> str:
        """Format parametric table using tabulate library."""
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

            # Dynamic column widths based on content
            headers = list(clean_table_data[0].keys()) if clean_table_data else []
            max_widths = []

            for header in headers:
                if header == '#':
                    max_widths.append(3)
                elif header in ['Part Number', 'Manufacturer']:
                    max_widths.append(20)
                elif header == 'Description':
                    max_widths.append(30)
                elif header in ['Lifecycle', 'RoHS', 'Datasheet']:
                    max_widths.append(10)
                else:  # Parametric features
                    max_widths.append(15)

            return tabulate(
                clean_table_data,
                headers='keys',
                tablefmt='grid',
                maxcolwidths=max_widths
            )

        except Exception as e:
            self.progress.warning("Tabulate Formatting", f"Falling back due to: {str(e)}")
            return self._format_parametric_fallback(table_data)

    def _format_parametric_fallback(self, table_data: List[Dict[str, str]]) -> str:
        """Fallback formatting for parametric data when tabulate is not available."""
        output = "Note: Using fallback formatting\n\n"

        for row in table_data:
            try:
                output += f"COMPONENT #{row.get('#', '?')}\n"
                output += f"├─ Part Number: {row.get('Part Number', 'Unknown')}\n"
                output += f"├─ Manufacturer: {row.get('Manufacturer', 'N/A')}\n"
                output += f"├─ Description: {row.get('Description', 'N/A')}\n"
                output += f"├─ Lifecycle: {row.get('Lifecycle', 'N/A')}\n"
                output += f"├─ RoHS: {row.get('RoHS', 'N/A')}\n"
                output += f"├─ Datasheet: {row.get('Datasheet', 'N/A')}\n"

                # Add parametric features
                for key, value in row.items():
                    if key not in ['#', 'Part Number', 'Manufacturer', 'Description', 'Lifecycle', 'RoHS', 'Datasheet']:
                        output += f"├─ {key}: {value}\n"

                output += f"└─ [End of Component #{row.get('#', '?')}]\n"
                output += "\n" + "-" * 80 + "\n"
            except Exception as row_error:
                output += f"Error formatting component: {str(row_error)}\n"
                output += f"Raw data: {str(row)}\n\n"

        return output

    def _create_parametric_features_summary(self, components: List[EnhancedComponent]) -> str:
        """Create a summary of all available parametric features."""
        all_features = set()
        feature_coverage = {}

        for component in components:
            if component.has_parametric_data:
                for feature_name in component.silicon_expert_data.parametric_features.keys():
                    all_features.add(feature_name)
                    feature_coverage[feature_name] = feature_coverage.get(feature_name, 0) + 1

        if not all_features:
            return "\nNo detailed parametric features available for these components.\n"

        output = f"\n{'=' * 140}\n"
        output += "AVAILABLE PARAMETRIC FEATURES:\n"
        output += f"{'=' * 140}\n"

        # Sort features by coverage (most common first)
        sorted_features = sorted(feature_coverage.items(), key=lambda x: x[1], reverse=True)

        for feature_name, count in sorted_features[:15]:  # Show top 15 features
            coverage_pct = (count / len(components)) * 100
            output += f"• {feature_name}: {count}/{len(components)} components ({coverage_pct:.1f}%)\n"

        if len(sorted_features) > 15:
            output += f"• ... and {len(sorted_features) - 15} more features\n"

        return output

    def _create_parametric_summary(self, parametric_result: ParametricSearchResult) -> str:
        """Create summary statistics for parametric search."""
        components = parametric_result.components
        total_components = len(components)
        components_with_features = parametric_result.components_with_features

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

        output = f"\n{'=' * 140}\n"
        output += "SEARCH SUMMARY:\n"
        output += f"{'=' * 140}\n"
        output += f"• Product Line: {parametric_result.product_line}\n"
        output += f"• Total Available: {parametric_result.total_items} components\n"
        output += f"• Components Shown: {total_components}\n"
        output += f"• With Parametric Data: {components_with_features} ({(components_with_features / total_components * 100):.1f}%)\n"

        # Add lifecycle distribution if meaningful
        if lifecycle_counts and lifecycle_counts != Counter({'Unknown': total_components}):
            output += f"\nLifecycle Distribution:\n"
            for lifecycle, count in lifecycle_counts.most_common():
                percentage = (count / total_components) * 100
                output += f"  • {lifecycle}: {count} components ({percentage:.1f}%)\n"

        # Add RoHS compliance if meaningful
        if rohs_counts and rohs_counts != Counter({'Unknown': total_components}):
            output += f"\nRoHS Compliance:\n"
            for rohs, count in rohs_counts.most_common():
                percentage = (count / total_components) * 100
                output += f"  • {rohs}: {count} components ({percentage:.1f}%)\n"

        return output

    def _create_parametric_suggestions(self) -> str:
        """Create next steps suggestions for parametric search results."""
        return (
            f"\n{'=' * 140}\n"
            "RECOMMENDED NEXT STEPS:\n"
            f"{'=' * 140}\n"
            "1. 📋 CREATE BOM FROM RESULTS:\n"
            "   → Use 'create_empty_bom' to create a new BOM structure\n"
            "   → Then use 'add_parts_to_bom' to add selected components\n"
            "   → Or use 'create_bom_from_schematic' for complete automation\n\n"

            "2. 🔍 REFINE SEARCH:\n"
            "   → Add more specific filters to narrow down results\n"
            "   → Use different parameter ranges or values\n"
            "   → Try related product lines for alternatives\n\n"

            "3. 📊 ANALYZE COMPONENTS:\n"
            "   → Review parametric features for design requirements\n"
            "   → Check lifecycle status for long-term availability\n"
            "   → Verify RoHS compliance for regulatory requirements\n"
            "   → Access datasheets for detailed specifications\n\n"

            "4. 🔄 ITERATE DESIGN:\n"
            "   → Compare different component options\n"
            "   → Optimize for cost, performance, or availability\n"
            "   → Consider alternative components with similar specs\n\n"
        )

    def _create_parametric_example_commands(self) -> str:
        """Create example commands section for parametric search results."""
        columns_json = str(DEFAULT_BOM_COLUMNS).replace("'", '"')

        return (
            "EXAMPLE COMMANDS:\n"
            f"• Create BOM: create_empty_bom(name='Parametric_BOM', columns='{columns_json}')\n"
            "• View existing BOMs: get_boms()\n"
            "• Refine search: parametric_search('ProductLine', '[{\"fetName\": \"Feature\", \"values\":[{\"value\": \"NewValue\"}]}]')\n"
            "• Get all product lines: get_taxonomy()\n"
            f"{'=' * 140}\n"
        )

    def format_search_result(self, search_result: SearchResult) -> str:
        """Format a SearchResult into a comprehensive table display (original method)."""
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
        """Build table data from components (original method)."""
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
        """Create table header (original method)."""
        return (
                "\n" + "=" * 120 + "\n"
                                   "COMPONENT ANALYSIS RESULTS\n" +
                "=" * 120 + "\n"
        )

    def _format_with_tabulate(self, table_data: List[Dict[str, str]]) -> str:
        """Format table using tabulate library (original method)."""
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
        """Fallback formatting when tabulate is not available (original method)."""
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
        """Create summary statistics (original method)."""
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
        """Create next steps suggestions (original method)."""
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
        """Create example commands section (original method)."""
        columns_json = str(DEFAULT_BOM_COLUMNS).replace("'", '"')

        return (
                "EXAMPLE COMMANDS:\n"
                f"• Create new BOM: create_empty_bom(name='MyProject_BOM', columns='{columns_json}')\n"
                "• View existing BOMs: get_boms()\n"
                "• Add parts to existing BOM: add_parts_to_bom(name='BOM_NAME', parent_path='PROJECT_PATH', parts_json='COMPONENT_DATA')\n" +
                "=" * 120 + "\n"
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
                'quantity': component.quantity,
                'designator': component.designator,
                'functional_block': component.functional_block,
                'notes': component.notes,
                'pl_name': component.pl_name,
                'selected_filters': component.selected_filters
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

                # Add parametric features
                if se_data.parametric_features:
                    parametric_dict = {}
                    for name, feature in se_data.parametric_features.items():
                        parametric_dict[name] = {
                            'value': feature.value,
                            'unit': feature.unit
                        }
                    component_dict['se_parametric_features'] = parametric_dict

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