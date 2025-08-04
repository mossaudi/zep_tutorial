# services/formatter.py
"""Enhanced component table formatting service with parametric search support."""
from collections import Counter
from typing import List, Dict

from Bom_Chatbot.constants import DEFAULT_BOM_COLUMNS
from Bom_Chatbot.models import EnhancedComponent, SearchResult, BOMTreeResult
from Bom_Chatbot.services.progress import get_progress_tracker


class ComponentTableFormatter:
    """Enhanced formatter for component data with parametric features support."""

    def __init__(self):
        self.progress = get_progress_tracker()

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

    def format_bom_tree(self, bom_tree: 'BOMTreeResult') -> str:
        """Format BOM tree structure with projects and root BOMs - optimized for large datasets."""
        if not bom_tree.success:
            return f"❌ Failed to retrieve BOM information: {bom_tree.error_message or 'Unknown error'}"

        if bom_tree.total_boms == 0:
            return "📋 No BOMs found in your account."

        try:
            # For large datasets, use a more efficient approach
            is_large_dataset = bom_tree.total_boms > 100

            output = self._create_bom_tree_header(bom_tree)

            if is_large_dataset:
                # For large datasets, show summary tree without detailed table
                output += self._format_large_bom_tree(bom_tree)
            else:
                # For smaller datasets, show full tree with table
                output += self._format_standard_bom_tree(bom_tree)

            # Add summary
            output += self._create_bom_summary(bom_tree)
            output += self._create_bom_next_steps()

            return output

        except Exception as e:
            self.progress.error("BOM Tree Formatting", str(e))
            return f"❌ Error formatting BOM tree: {str(e)}\nTotal BOMs: {bom_tree.total_boms}"

    def _format_large_bom_tree(self, bom_tree: 'BOMTreeResult') -> str:
        """Optimized formatting for large BOM datasets."""
        output = "\n📁 BOM HIERARCHY (SUMMARY VIEW):\n"
        output += "=" * 80 + "\n"

        # Show projects with BOM counts only
        if bom_tree.projects:
            output += "\n📁 PROJECTS:\n"
            for project in bom_tree.projects[:20]:  # Limit to first 20 projects
                bom_count = len(project["boms"])
                total_parts = sum(int(bom.get("parts_count", '0')) for bom in project["boms"])
                output += f"├── 📁 {project["name"]} ({bom_count} BOMs, {total_parts:,} total parts)\n"

            if len(bom_tree.projects) > 20:
                remaining = len(bom_tree.projects) - 20
                output += f"└── ... and {remaining} more projects\n"

        # Show root BOMs summary
        if bom_tree.root_boms:
            output += f"\n📁 ROOT LEVEL: {len(bom_tree.root_boms)} BOMs\n"
            for bom in bom_tree.root_boms[:10]:  # Limit to first 10
                output += f"├── 📋 {bom["name"]} ({bom["parts_count"]} parts)\n"

            if len(bom_tree.root_boms) > 10:
                remaining = len(bom_tree.root_boms) - 10
                output += f"└── ... and {remaining} more BOMs\n"

        output += f"\n💡 TIP: Use project_name filter to view specific project details\n"
        output += f"💡 Example: get_boms(project_name='YourProjectName')\n"

        return output

    def _format_standard_bom_tree(self, bom_tree: 'BOMTreeResult') -> str:
        """Standard formatting for smaller BOM datasets."""
        output = "\n📁 BOM HIERARCHY:\n"
        output += "=" * 80 + "\n"

        # Display projects with their BOMs
        if bom_tree.projects:
            for project in bom_tree.projects:
                output += f"\n📁 PROJECT: {project['name']}\n"
                if project["boms"]:
                    for i, bom in enumerate(project["boms"], 1):
                        is_last = i == len(project["boms"])
                        prefix = "└── " if is_last else "├── "
                        output += f"   {prefix}📋 {bom["name"]} ({bom["parts_count"]} parts)\n"
                        output += f"       Created: {bom["creation_date"]} | Modified: {bom["modification_date"]}\n"
                else:
                    output += "   └── (No BOMs in this project)\n"

        # Display root-level BOMs
        if bom_tree.root_boms:
            output += f"\n📁 ROOT LEVEL BOMs:\n"
            for i, bom in enumerate(bom_tree.root_boms, 1):
                is_last = i == len(bom_tree.root_boms)
                prefix = "└── " if is_last else "├── "
                output += f"{prefix}📋 {bom['name']} ({bom['parts_count']} parts)\n"
                output += f"    Created: {bom['creation_date']} | Modified: {bom['modification_date']}\n"

        # Add detailed table only for smaller datasets
        if bom_tree.total_boms <= 50:
            try:
                from tabulate import tabulate
                output += "\n" + "=" * 80 + "\n"
                output += "📊 DETAILED BOM TABLE:\n"
                output += "=" * 80 + "\n"
                output += self._format_bom_table_with_tabulate(bom_tree)
            except ImportError:
                pass  # Skip table if tabulate not available

        return output

    def _create_bom_tree_header(self, bom_tree: 'BOMTreeResult') -> str:
        """Create header for BOM tree display."""
        return (
                "\n" + "=" * 80 + "\n"
                                  "BOM MANAGEMENT OVERVIEW\n" +
                "=" * 80 + "\n"
                           f"📊 Total Projects: {bom_tree.total_projects}\n"
                           f"📋 Total BOMs: {bom_tree.total_boms}\n"
                           f"🏠 Root-level BOMs: {len(bom_tree.root_boms)}\n"
        )

    def _format_bom_table_with_tabulate(self, bom_tree: 'BOMTreeResult') -> str:
        """Format detailed BOM table using tabulate."""
        try:
            from tabulate import tabulate

            table_data = []

            # Add project BOMs
            for project in bom_tree.projects:
                for bom in project["boms"]:
                    table_data.append({
                        'Project': project["name"],
                        'BOM Name': bom["name"],
                        'Parts': bom["parts_count"],
                        'Created': bom["creation_date"],
                        'Modified': bom["modification_date"],
                        'Created By': bom["created_user"].split('@')[0] if '@' in bom["created_user"] else bom["created_user"]
                    })

            # Add root BOMs
            for bom in bom_tree.root_boms:
                table_data.append({
                    'Project': '(Root Level)',
                    'BOM Name': bom.name,
                    'Parts': bom.parts_count,
                    'Created': bom.creation_date,
                    'Modified': bom.modification_date,
                    'Created By': bom.created_user.split('@')[0] if '@' in bom.created_user else bom.created_user
                })

            return tabulate(
                table_data,
                headers='keys',
                tablefmt='grid',
                maxcolwidths=[20, 25, 8, 12, 12, 15]
            )

        except Exception as e:
            self.progress.warning("BOM Table Formatting", f"Falling back due to: {str(e)}")
            return self._format_bom_table_fallback(bom_tree)

    def _format_bom_table_fallback(self, bom_tree: 'BOMTreeResult') -> str:
        """Fallback formatting for BOM table."""
        output = "\n📊 DETAILED BOM INFORMATION:\n"
        output += "=" * 80 + "\n"

        bom_count = 1

        # Project BOMs
        for project in bom_tree.projects:
            for bom in project["boms"]:
                output += f"\n#{bom_count} - {bom["name"]}\n"
                output += f"├─ Project: {project["name"]}\n"
                output += f"├─ Parts Count: {bom["parts_count"]}\n"
                output += f"├─ Created: {bom["creation_date"]}\n"
                output += f"├─ Modified: {bom["modification_date"]}\n"
                output += f"└─ Created By: {bom['created_user']}\n"
                output += "-" * 60 + "\n"
                bom_count += 1

        # Root BOMs
        for bom in bom_tree.root_boms:
            output += f"\n#{bom_count} - {bom['name']}\n"
            output += f"├─ Project: (Root Level)\n"
            output += f"├─ Parts Count: {bom['parts_count']}\n"
            output += f"├─ Created: {bom['creation_date']}\n"
            output += f"├─ Modified: {bom['modification_date']}\n"
            output += f"└─ Created By: {bom['created_user']}\n"
            output += "-" * 60 + "\n"
            bom_count += 1

        return output

    def _create_bom_summary(self, bom_tree: 'BOMTreeResult') -> str:
        """Create BOM summary statistics."""
        total_parts = 0
        for project in bom_tree.projects:
            for bom in project['boms']:
                parts_count_str = bom.get('parts_count', '0')
                if parts_count_str:
                    total_parts += int(parts_count_str)
        for bom in bom_tree.root_boms:
            parts_count_str = bom.get('parts_count', '0')
            if parts_count_str:
                total_parts += int(parts_count_str)

        output = "\n" + "=" * 80 + "\n"
        output += "📈 SUMMARY STATISTICS:\n"
        output += "=" * 80 + "\n"
        output += f"• Total Projects: {bom_tree.total_projects}\n"
        output += f"• Total BOMs: {bom_tree.total_boms}\n"
        output += f"• Total Parts Across All BOMs: {total_parts:,}\n"

        if bom_tree.projects:
            output += f"• BOMs in Projects: {sum(len(p['boms']) for p in bom_tree.projects)}\n"
        if bom_tree.root_boms:
            output += f"• Root-level BOMs: {len(bom_tree.root_boms)}\n"

        return output

    def _create_bom_next_steps(self) -> str:
        """Create next steps for BOM management."""
        return (
                "\n" + "=" * 80 + "\n"
                                  "🚀 NEXT STEPS:\n" +
                "=" * 80 + "\n"
                           "1. CREATE NEW BOM:\n"
                           "   → Use 'create_empty_bom' to create a new BOM structure\n\n"

                           "2. ADD COMPONENTS TO EXISTING BOM:\n"
                           "   → Use 'add_parts_to_bom' with the BOM name from above\n\n"

                           "3. ANALYZE SCHEMATIC:\n"
                           "   → Use 'analyze_schematic' to extract components and add them to a BOM\n\n"

                           "4. EXAMPLE COMMANDS:\n"
                           "   → add_parts_to_bom(name='my BOM', parent_path='my Project', parts_json='[...]')\n"
                           "   → create_empty_bom(name='New_BOM', columns='[\"mpn\", \"manufacturer\", ...]')\n" +
                "=" * 80 + "\n"
        )