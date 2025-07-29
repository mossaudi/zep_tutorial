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
class ParametricFeature:
    """Represents a parametric feature from Silicon Expert."""
    name: str
    value: str
    unit: Optional[str] = None
    min_value: Optional[str] = None
    max_value: Optional[str] = None
    typical_value: Optional[str] = None

    def __str__(self) -> str:
        if self.unit:
            return f"{self.value} {self.unit}"
        return self.value


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
    parametric_features: Dict[str, ParametricFeature] = field(default_factory=dict)
    package_type: Optional[str] = None
    supply_voltage: Optional[str] = None
    operating_temperature: Optional[str] = None
    power_dissipation: Optional[str] = None
    frequency: Optional[str] = None

    def add_parametric_feature(self, name: str, value: str, unit: Optional[str] = None) -> None:
        """Add a parametric feature to the component."""
        self.parametric_features[name] = ParametricFeature(
            name=name,
            value=value,
            unit=unit
        )

    def get_feature_value(self, feature_name: str) -> Optional[str]:
        """Get the value of a specific feature."""
        feature = self.parametric_features.get(feature_name)
        return str(feature) if feature else None


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
    """Component with Silicon Expert data and parametric features."""
    silicon_expert_data: Optional[SiliconExpertData] = None
    search_result: Optional[str] = None
    parametric_search_used: bool = False

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

    @property
    def has_parametric_data(self) -> bool:
        """Check if component has parametric features."""
        return (self.silicon_expert_data and
                bool(self.silicon_expert_data.parametric_features))

    def get_parametric_summary(self) -> Dict[str, str]:
        """Get a summary of key parametric features."""
        if not self.has_parametric_data:
            return {}

        summary = {}
        features = self.silicon_expert_data.parametric_features

        # Key features to highlight
        key_features = [
            'Package Type', 'Supply Voltage', 'Operating Temperature',
            'Power Dissipation', 'Frequency', 'Current Rating',
            'Voltage Rating', 'Capacitance', 'Resistance', 'Tolerance'
        ]

        for key in key_features:
            if key in features:
                summary[key] = str(features[key])

        return summary

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

        # Add key parametric features as comments
        if self.has_parametric_data:
            parametric_summary = self.get_parametric_summary()
            if parametric_summary:
                features_str = ', '.join([f"{k}: {v}" for k, v in parametric_summary.items()])
                bom_part['uploadedcomments'] += f" | Features: {features_str}"

        return bom_part


@dataclass
class ParametricSearchResult:
    """Result from parametric search with enhanced component data."""
    success: bool
    product_line: str
    total_items: int
    components: List[EnhancedComponent] = field(default_factory=list)
    search_filters: List[Dict[str, Any]] = field(default_factory=list)
    error_message: Optional[str] = None

    @property
    def components_with_features(self) -> int:
        """Count components that have parametric features."""
        return len([c for c in self.components if c.has_parametric_data])


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
    parametric_results: List[ParametricSearchResult] = field(default_factory=list)
    bom_creation: Optional[Dict[str, Any]] = None
    parts_addition: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    summary: Optional[str] = None