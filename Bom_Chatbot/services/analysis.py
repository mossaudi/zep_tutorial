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
                "As an expert electrical engineer with 15+ years of PCB design experience, "
                "perform a comprehensive schematic analysis of the design at: {image_url}\n\n"
                
                "ANALYSIS REQUIREMENTS:\n"
                "1. SYSTEMATIC SCANNING: Analyze top-left to bottom-right, then by functional blocks\n"
                "2. COMPLETE EXTRACTION: Identify ALL visible components including reference designators\n"
                "3. TECHNICAL PRECISION: Extract exact specifications from visible part numbers and values\n"
                "4. FUNCTIONAL GROUPING: Organize components by circuit function (Power, Interface, Control, etc.)\n\n"
                
                "COMPONENT SPECIFICATION EXTRACTION:\n"
                "For each component type, extract these parameters:\n"
                "- Semiconductors: Part number, package type, supply voltage range, current ratings, key electrical parameters\n"
                "- Passives: Precise values with units, tolerance, power/voltage ratings, package size, material properties\n"
                "- Connectors: Type, pin count, current/voltage ratings, mechanical specifications\n\n"
                
                "REQUIRED JSON OUTPUT FORMAT:\n"
                "{{\n"
                "  \"boardInfo\": {{\n"
                "    \"title\": \"Board name from schematic\",\n"
                "    \"revision\": \"Hardware revision if visible\",\n"
                "    \"date\": \"Design date if visible\",\n"
                "    \"totalComponents\": \"Total component count\"\n"
                "  }},\n"
                "  \"components\": [\n"
                "    {{\n"
                "      \"plName\": \"Component Category (e.g., MOSFETs, Microcontrollers)\",\n"
                "      \"selectedFilters\": [\n"
                "        {{\n"
                "          \"fetName\": \"Part Number\",\n"
                "          \"values\": [{{\"value\": \"Exact manufacturer part number\"}}]\n"
                "        }},\n"
                "        {{\n"
                "          \"fetName\": \"Package Type\",\n"
                "          \"values\": [{{\"value\": \"Package with pin count (e.g., SOIC-8, QFN-32)\"}}]\n"
                "        }},\n"
                "        {{\n"
                "          \"fetName\": \"Supply Voltage Range\",\n"
                "          \"values\": [{{\"value\": \"Voltage range with units (e.g., 3.0V to 5.5V)\"}}]\n"
                "        }},\n"
                "        {{\n"
                "          \"fetName\": \"Key Electrical Parameter\",\n"
                "          \"values\": [{{\"value\": \"Specification with units and conditions\"}}]\n"
                "        }}\n"
                "      ],\n"
                "      \"designator\": \"Reference designator from schematic (e.g., U1, R5, C12)\",\n"
                "      \"quantity\": \"Number of this component type\",\n"
                "      \"functionalBlock\": \"Circuit function (Power Management, Interface, Control, etc.)\",\n"
                "      \"notes\": \"Critical specifications or alternatives\"\n"
                "    }}\n"
                "  ],\n"
                "  \"powerRequirements\": {{\n"
                "    \"inputVoltage\": \"Input voltage specifications\",\n"
                "    \"outputVoltages\": [\"List of regulated output voltages\"],\n"
                "    \"estimatedCurrent\": \"Typical current consumption\"\n"
                "  }},\n"
                "  \"analysisQuality\": {{\n"
                "    \"componentsIdentified\": \"Number of components successfully identified\",\n"
                "    \"confidenceLevel\": \"High/Medium/Low based on schematic clarity\",\n"
                "    \"missingData\": [\"List any unclear or missing component information\"]\n"
                "  }}\n"
                "}}\n\n"
                
                "CRITICAL REQUIREMENTS:\n"
                "- Return ONLY valid JSON - no explanatory text before or after\n"
                "- Include ALL visible components, even if specifications are partial\n"
                "- Use exact part numbers when visible, generic types when not\n"
                "- Verify electrical specifications are realistic and manufacturable\n"
                "- Group similar components but list each unique specification\n"
                "- Include quantity counts and reference designators\n"
                "- Provide confidence assessment for component identification\n\n"
                
                "Begin comprehensive analysis now:"
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