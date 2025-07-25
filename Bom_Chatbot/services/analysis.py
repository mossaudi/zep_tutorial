# services/analysis.py
"""Component analysis service."""

import json
import re
from typing import List, Optional
from langchain_core.messages import HumanMessage
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from Bom_Chatbot.models import Component, EnhancedComponent, AnalysisResult, SearchResult
from Bom_Chatbot.exceptions import SchematicAnalysisError, JSONProcessingError, DataValidationError
from Bom_Chatbot.services.progress import get_progress_tracker
from Bom_Chatbot.clients.silicon_expert import SiliconExpertClient


class JSONProcessor:
    """Handles JSON cleaning and validation."""

    @staticmethod
    def clean_llm_response(response_text: str) -> str:
        """Clean up LLM response to extract valid JSON."""
        if not response_text:
            raise JSONProcessingError("Empty response text", "", "initial_validation")

        # Remove markdown code blocks
        response_text = re.sub(r'```(?:json)?\s*', '', response_text, flags=re.IGNORECASE)
        response_text = response_text.strip()

        def find_json_boundaries(text: str, start_char: str, end_char: str):
            """Find matching start and end positions for JSON structure."""
            start_pos = text.find(start_char)
            if start_pos == -1:
                return None, None

            count = 0
            in_string = False
            escape_next = False

            for i in range(start_pos, len(text)):
                char = text[i]

                if escape_next:
                    escape_next = False
                    continue

                if char == '\\' and in_string:
                    escape_next = True
                    continue

                if char == '"' and not escape_next:
                    in_string = not in_string
                    continue

                if not in_string:
                    if char == start_char:
                        count += 1
                    elif char == end_char:
                        count -= 1
                        if count == 0:
                            return start_pos, i + 1

            return start_pos, len(text)

        # Try to find JSON array first
        start_pos, end_pos = find_json_boundaries(response_text, '[', ']')

        # If no array found, try to find JSON object
        if start_pos is None:
            start_pos, end_pos = find_json_boundaries(response_text, '{', '}')

        if start_pos is None:
            raise JSONProcessingError(
                "No JSON structure found in response",
                response_text,
                "boundary_detection"
            )

        json_text = response_text[start_pos:end_pos]

        # Validate by parsing
        try:
            parsed = json.loads(json_text)
            return json.dumps(parsed, indent=2)
        except json.JSONDecodeError as e:
            # Try minimal cleanup
            try:
                cleaned = json_text.strip()
                # Remove trailing commas before closing brackets/braces
                cleaned = re.sub(r',(\s*[\]\}])', r'\1', cleaned)

                parsed = json.loads(cleaned)
                return json.dumps(parsed, indent=2)
            except json.JSONDecodeError:
                raise JSONProcessingError(
                    f"JSON parsing failed: {str(e)}",
                    json_text,
                    "json_parsing"
                )

    @staticmethod
    def parse_components(json_text: str) -> List[Component]:
        """Parse JSON text into Component objects."""
        try:
            data = json.loads(json_text)
        except json.JSONDecodeError as e:
            raise JSONProcessingError(f"Invalid JSON: {str(e)}", json_text, "component_parsing")

        if not isinstance(data, list):
            raise DataValidationError(
                "Component data must be a JSON array",
                "component_list",
                ["Expected list, got " + type(data).__name__]
            )

        components = []
        validation_errors = []

        for i, item in enumerate(data):
            if not isinstance(item, dict):
                validation_errors.append(f"Component {i + 1}: Expected dict, got {type(item).__name__}")
                continue

            # Extract component data with flexible key mapping
            name = (item.get('name') or
                    item.get('component_name') or
                    f'Component_{i + 1}')

            # Handle features (can be string or list)
            features = item.get('features', '')
            if isinstance(features, list):
                features = ' '.join(str(f).strip() for f in features if f)
            elif not isinstance(features, str):
                features = str(features) if features else ''

            component = Component(
                name=name,
                part_number=item.get('part_number'),
                manufacturer=item.get('manufacturer'),
                description=item.get('description'),
                value=item.get('value') or item.get('rating'),
                features=features,
                quantity=item.get('quantity', '1')
            )

            components.append(component)

        if validation_errors:
            raise DataValidationError(
                "Component validation errors found",
                "component_validation",
                validation_errors
            )

        return components


class SchematicAnalyzer:
    """Handles schematic image analysis using LLM."""

    def __init__(self, llm: ChatGoogleGenerativeAI):
        self.llm = llm
        self.progress = get_progress_tracker()
        self.json_processor = JSONProcessor()

        self.prompt_template = PromptTemplate(
            input_variables=["image_url"],
            template=(
                "As an expert electrical engineer, please analyze the schematic design "
                "at the following URL: {image_url}. "
                "List all components in JSON format with component name, part number, "
                "manufacturer, and features. Return ONLY valid JSON array format."
            )
        )

    def analyze(self, image_url: str) -> AnalysisResult:
        """Analyze schematic image and extract components."""
        self.progress.info("Schematic Analysis", f"Processing image from {image_url}")

        try:
            # Step 1: LLM Analysis
            self.progress.info("AI Analysis", "Gemini LLM analyzing schematic components...")

            message = self.prompt_template.format(image_url=image_url)
            response = self.llm.invoke([HumanMessage(content=message)])
            raw_response = response.content

            if not raw_response:
                raise SchematicAnalysisError(
                    "Empty response from LLM",
                    image_url
                )

            self.progress.success("AI Analysis", "Received response from LLM")

            # Step 2: JSON Processing
            self.progress.info("Data Processing", "Cleaning and validating JSON response...")

            try:
                cleaned_json = self.json_processor.clean_llm_response(raw_response)
                components = self.json_processor.parse_components(cleaned_json)

                self.progress.success("Data Processing", f"Parsed {len(components)} components")

                return AnalysisResult(
                    image_url=image_url,
                    components=components,
                    raw_response=raw_response,
                    success=True
                )

            except (JSONProcessingError, DataValidationError) as e:
                self.progress.error("Data Processing", str(e))
                return AnalysisResult(
                    image_url=image_url,
                    components=[],
                    raw_response=raw_response,
                    success=False,
                    error_message=str(e)
                )

        except Exception as e:
            self.progress.error("Schematic Analysis", str(e))
            return AnalysisResult(
                image_url=image_url,
                components=[],
                success=False,
                error_message=str(e)
            )


class ComponentAnalysisService:
    """Main service for component analysis operations."""

    def __init__(self, llm: ChatGoogleGenerativeAI,
                 silicon_expert_client: SiliconExpertClient):
        self.llm = llm
        self.silicon_expert_client = silicon_expert_client
        self.schematic_analyzer = SchematicAnalyzer(llm)
        self.progress = get_progress_tracker()

    def analyze_schematic(self, image_url: str) -> SearchResult:
        """Complete schematic analysis workflow."""
        self.progress.info("Component Analysis", "Starting complete analysis workflow")

        # Step 1: Analyze schematic
        analysis_result = self.schematic_analyzer.analyze(image_url)

        if not analysis_result.success or not analysis_result.components:
            return SearchResult(
                success=False,
                components=[],
                error_message=analysis_result.error_message or "No components found"
            )

        # Step 2: Enhance with Silicon Expert data
        self.progress.info(
            "Data Enhancement",
            f"Searching Silicon Expert database for {len(analysis_result.components)} components"
        )

        try:
            enhanced_components = self.silicon_expert_client.search_components(
                analysis_result.components
            )

            # Calculate statistics
            successful_searches = len([c for c in enhanced_components if c.silicon_expert_data])
            failed_searches = len(enhanced_components) - successful_searches

            self.progress.success(
                "Component Analysis",
                "Component data enhanced with Silicon Expert information"
            )

            return SearchResult(
                success=True,
                components=enhanced_components,
                successful_searches=successful_searches,
                failed_searches=failed_searches
            )

        except Exception as e:
            self.progress.error("Data Enhancement", str(e))
            # Return basic components without enhancement
            basic_enhanced = [
                EnhancedComponent(**component.__dict__)
                for component in analysis_result.components
            ]

            return SearchResult(
                success=False,
                components=basic_enhanced,
                successful_searches=0,
                failed_searches=len(basic_enhanced),
                error_message=f"Silicon Expert enhancement failed: {str(e)}"
            )

    def search_component_data(self, components: List[Component]) -> SearchResult:
        """Search for component data using Silicon Expert."""
        self.progress.info("Component Search", f"Searching data for {len(components)} components")

        try:
            enhanced_components = self.silicon_expert_client.search_components(components)

            successful_searches = len([c for c in enhanced_components if c.silicon_expert_data])
            failed_searches = len(enhanced_components) - successful_searches

            return SearchResult(
                success=True,
                components=enhanced_components,
                successful_searches=successful_searches,
                failed_searches=failed_searches
            )

        except Exception as e:
            self.progress.error("Component Search", str(e))
            return SearchResult(
                success=False,
                components=[],
                error_message=str(e)
            )