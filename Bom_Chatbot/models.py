# models.py
"""Enhanced data models for the LangGraph agent with parametric search support."""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum
import json


class LifecycleStatus(Enum):
    """Component lifecycle status."""
    ACTIVE = "Active"
    DISCONTINUED = "Discontinued"
    EOL = "End of Life"
    NRND = "Not Recommended for New Designs"
    UNKNOWN = "Unknown"


class RoHSStatus(Enum):
    """RoHS compliance status."""
    COMPLIANT = "Compliant"
    NON_COMPLIANT = "Non-Compliant"
    UNKNOWN = "Unknown"

@dataclass
class SiliconExpertData:
    """Enhanced data from Silicon Expert API with parametric features."""
    com_id: Optional[str] = None
    part_number: Optional[str] = None
    manufacturer: Optional[str] = None
    description: Optional[str] = None
    lifecycle: Optional[str] = None
    rohs: Optional[str] = None
    rohs_version: Optional[str] = None
    datasheet: Optional[str] = None
    product_line: Optional[str] = None
    taxonomy_path: Optional[str] = None
    match_rating: Optional[str] = None
    match_comment: Optional[str] = None
    yeol: Optional[str] = None
    resilience_rating: Optional[str] = None
    military_status: Optional[str] = None
    aml_status: Optional[str] = None
    search_query: Optional[str] = None
    total_items: Optional[str] = None
    all_matches: List[Dict[str, Any]] = field(default_factory=list)

    # New fields for parametric data
    package_type: Optional[str] = None
    supply_voltage: Optional[str] = None
    operating_temperature: Optional[str] = None
    power_dissipation: Optional[str] = None
    frequency: Optional[str] = None


@dataclass
class Component:
    """Basic component information."""
    name: str
    part_number: Optional[str] = None
    manufacturer: Optional[str] = None
    description: Optional[str] = None
    value: Optional[str] = None
    features: Optional[str] = None
    quantity: Optional[str] = "1"

    # New fields from schematic analysis
    designator: Optional[str] = None
    functional_block: Optional[str] = None
    notes: Optional[str] = None
    pl_name: Optional[str] = None  # Product line from schematic analysis
    selected_filters: List[Dict[str, Any]] = field(default_factory=list)

    def get_search_parts(self) -> List[str]:
        """Get parts of component data suitable for search."""
        parts = []

        if self.part_number:
            parts.append(self.part_number.strip())
        if self.manufacturer:
            parts.append(self.manufacturer.strip())
        # if self.name and self.name not in parts:
        #     parts.append(self.name.strip())
        if self.description and len(self.description) < 100:
            parts.append(self.description.strip())
        if self.features and len(self.features) < 50:
            parts.append(self.features.strip())
        if self.value:
            parts.append(self.value.strip())

        return [part for part in parts if part]


@dataclass
class EnhancedComponent(Component):
    """Component with Silicon Expert data and parametric features."""
    silicon_expert_data: Optional[SiliconExpertData] = None
    search_result: Optional[str] = None

    @property
    def effective_part_number(self) -> str:
        """Get the most reliable part number."""
        if self.silicon_expert_data and self.silicon_expert_data.part_number:
            return self.silicon_expert_data.part_number
        return self.part_number or "N/A"

    @property
    def effective_manufacturer(self) -> str:
        """Get the most reliable manufacturer."""
        if self.silicon_expert_data and self.silicon_expert_data.manufacturer:
            return self.silicon_expert_data.manufacturer
        return self.manufacturer or "N/A"

    @property
    def effective_description(self) -> str:
        """Get the most reliable description."""
        if self.silicon_expert_data and self.silicon_expert_data.description:
            return self.silicon_expert_data.description
        return self.description or "N/A"

    def to_bom_part(self) -> Dict[str, str]:
        """Convert to BOM part format with enhanced data."""
        bom_part = {
            'mpn': self.effective_part_number,
            'manufacturer': self.effective_manufacturer,
            'description': self.effective_description,
            'quantity': self.quantity or '1',
            'uploadedcomments': 'Added from schematic analysis'
        }

        # Add designator if available
        if self.designator:
            bom_part['designator'] = self.designator

        return bom_part


@dataclass
class BOMInfo:
    """BOM information."""
    name: str
    parent_path: str = ""
    description: str = ""
    columns: List[str] = field(default_factory=list)
    creation_date: Optional[str] = None
    modification_date: Optional[str] = None


@dataclass
class SearchResult:
    """Search operation result."""
    success: bool
    components: List[EnhancedComponent] = field(default_factory=list)
    successful_searches: int = 0
    failed_searches: int = 0
    error_message: Optional[str] = None

    @property
    def total_components(self) -> int:
        return len(self.components)

    @property
    def success_rate(self) -> float:
        if self.total_components == 0:
            return 0.0
        return (self.successful_searches / self.total_components) * 100


@dataclass
class AnalysisResult:
    """Schematic analysis result."""
    image_url: str
    components: List[Component] = field(default_factory=list)
    raw_response: Optional[str] = None
    success: bool = True
    error_message: Optional[str] = None

    # New fields for enhanced analysis
    board_info: Optional[Dict[str, str]] = None
    power_requirements: Optional[Dict[str, Any]] = None
    analysis_quality: Optional[Dict[str, Any]] = None

    def to_json(self) -> str:
        """Convert components to JSON format."""
        component_dicts = []
        for component in self.components:
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
            # Remove None values
            component_dict = {k: v for k, v in component_dict.items() if v is not None}
            component_dicts.append(component_dict)

        return json.dumps(component_dicts, indent=2)


@dataclass
class WorkflowResult:
    """Complete workflow operation result."""
    success: bool
    workflow_steps: List[str] = field(default_factory=list)
    schematic_analysis: Optional[AnalysisResult] = None
    search_result: Optional[SearchResult] = None
    bom_creation: Optional[Dict[str, Any]] = None
    parts_addition: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    summary: Optional[str] = None


@dataclass
class BOMDetails:
    """Detailed BOM information from API response."""
    name: str
    parts_count: str
    creation_date: str
    modification_date: str
    created_user: str
    modified_user: str
    project_name: Optional[str] = None  # None for root-level BOMs

    @classmethod
    def from_api_response(cls, bom_data: Dict[str, Any], project_name: Optional[str] = None) -> 'BOMDetails':
        """Create BOMDetails from API response data."""
        return cls(
            name=bom_data.get('BOMName', ''),
            parts_count=bom_data.get('PartsCount', '0'),
            creation_date=bom_data.get('CreationDate', ''),
            modification_date=bom_data.get('ModificationDate', ''),
            created_user=bom_data.get('createdUser', ''),
            modified_user=bom_data.get('modifiedUser', ''),
            project_name=project_name
        )


@dataclass
class ProjectDetails:
    """Project information with associated BOMs."""
    name: str
    boms: List[BOMDetails] = field(default_factory=list)

    @classmethod
    def from_api_response(cls, project_data: Dict[str, Any]) -> 'ProjectDetails':
        """Create ProjectDetails from API response data."""
        project_name = project_data.get('ProjectName', '')
        boms = []

        bom_list = project_data.get('BOMs', [])
        if isinstance(bom_list, list):
            for bom_data in bom_list:
                if len(bom_data.get('BOMName', '')) > 0:
                    boms.append(BOMDetails.from_api_response(bom_data, project_name))

        return cls(name=project_name, boms=boms)


@dataclass
class BOMTreeResult:
    """Complete BOM tree structure result."""
    success: bool
    projects: List[ProjectDetails] = field(default_factory=list)
    root_boms: List[BOMDetails] = field(default_factory=list)
    total_projects: int = 0
    total_boms: int = 0
    error_message: Optional[str] = None

    @classmethod
    def from_api_response(cls, api_response: Dict[str, Any]) -> 'BOMTreeResult':
        """Create BOMTreeResult from Silicon Expert API response."""
        if not api_response.get('Status', {}).get('success') == 'true':
            return cls(
                success=False,
                error_message=api_response.get('Status', {}).get('message', 'Unknown error')
            )

        projects = []
        root_boms = []

        # Parse projects
        projects_data = api_response.get('Projects', {})
        if projects_data and 'Project' in projects_data:
            project_list = projects_data['Project']
            if isinstance(project_list, list):
                for project_data in project_list:
                    if len(project_data.get('ProjectName', '')) > 0:
                        projects.append(ProjectDetails.from_api_response(project_data))
            elif isinstance(project_list, dict):
                # Single project case
                projects.append(ProjectDetails.from_api_response(project_list))

        # Parse root-level BOMs
        boms_data = api_response.get('BOMs', [])
        if isinstance(boms_data, list):
            for bom_data in boms_data:
                if len(bom_data.get('BOMName', '')) > 0:
                    root_boms.append(BOMDetails.from_api_response(bom_data))

        # Calculate totals
        total_boms = len(root_boms) + sum(len(project.boms) for project in projects)

        return cls(
            success=True,
            projects=projects,
            root_boms=root_boms,
            total_projects=len(projects),
            total_boms=total_boms
        )