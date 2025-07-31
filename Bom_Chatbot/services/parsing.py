# services/parsing.py
"""Service for parsing and validating component data from LLM responses."""

import json
from typing import List, Dict, Any

from Bom_Chatbot.models import Component
from Bom_Chatbot.exceptions import DataValidationError, JSONProcessingError


class ParsingService:
    """Handles robust parsing and conversion of component JSON."""

    def parse_and_convert_to_components(self, json_string: str) -> List[Component]:
        """
        Parses a JSON string and converts it into a list of Component objects.

        Args:
            json_string: The raw JSON string, assumed to be valid.

        Returns:
            A list of Component objects.
        """
        try:
            data = json.loads(json_string)
        except json.JSONDecodeError as e:
            raise JSONProcessingError(f"Invalid JSON format: {e}", json_string, "json_loading")

        if isinstance(data, dict) and 'components' in data:
            components_data = data['components']
        elif isinstance(data, list):
            components_data = data
        else:
            raise DataValidationError("JSON must be a list of components or an object with a 'components' key.")

        if not isinstance(components_data, list):
            raise DataValidationError(f"'components' field must be a list, got {type(components_data).__name__}.")

        return self._convert_to_component_objects(components_data)

    def _convert_to_component_objects(self, components_data: List[Dict[str, Any]]) -> List[Component]:
        """Converts a list of dictionaries to Component objects."""
        components: List[Component] = []
        for i, item in enumerate(components_data):
            if not isinstance(item, dict):
                # Optionally log a warning here
                continue

            designator = item.get('designator') or item.get('name') or f'Component_{i + 1}'
            quantity = item.get('quantity', '1')

            component = Component(
                name=designator,
                part_number=item.get('part_number'),
                manufacturer=item.get('manufacturer'),
                description=item.get('description'),
                value=item.get('value'),
                features=item.get('features'),
                quantity=str(quantity),
                designator=designator,
                functional_block=item.get('functionalBlock'),
                notes=item.get('notes'),
                pl_name=item.get('plName'),
                selected_filters=item.get('selectedFilters', [])
            )
            components.append(component)

        if not components:
            raise DataValidationError("No valid components could be processed from the provided data.")

        return components