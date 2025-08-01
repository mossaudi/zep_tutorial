# models.py
"""Data models for the LangGraph agent."""

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
    """Data from Silicon Expert API."""
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

    def get_search_parts(self) -> List[str]:
        """Get parts of component data suitable for search."""
        parts = []

        if self.part_number:
            parts.append(self.part_number.strip())
        if self.manufacturer:
            parts.append(self.manufacturer.strip())
        if self.name and self.name not in parts:
            parts.append(self.name.strip())
        if self.description and len(self.description) < 100:
            parts.append(self.description.strip())
        if self.features and len(self.features) < 50:
            parts.append(self.features.strip())
        if self.value:
            parts.append(self.value.strip())

        return [part for part in parts if part]


@dataclass
class EnhancedComponent(Component):
    """Component with Silicon Expert data."""
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
        """Convert to BOM part format."""
        return {
            'mpn': self.effective_part_number,
            'manufacturer': self.effective_manufacturer,
            'description': self.effective_description,
            'quantity': self.quantity or '1',
            'uploadedcomments': 'Added from schematic analysis'
        }


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
                'quantity': component.quantity
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