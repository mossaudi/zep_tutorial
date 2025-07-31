# services/analysis.py
"""Component analysis and search services."""
import re

from aiohttp.web_response import json_response
from langchain_core.messages import HumanMessage
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from Bom_Chatbot.exceptions import SchematicAnalysisError
from Bom_Chatbot.services.progress import get_progress_tracker


class SchematicAnalyzer:
    """Handles schematic image analysis using an LLM."""

    def __init__(self, llm: ChatGoogleGenerativeAI):
        self.llm = llm
        self.progress = get_progress_tracker()
        # The prompt is updated to explicitly ask for JSON output.
        self.prompt_template = PromptTemplate(
            input_variables=["image_url"],
            template=(
                "Analyze the schematic at {image_url}. Identify all components, including reference designators, "
                "specifications, and functional groupings. Your response MUST be a single, valid JSON object. "
                "The JSON object should contain a key 'components', which is a list of all identified components. "
                "Each component object in the list should have keys like 'designator', 'part_number', "
                "'description', 'functionalBlock', etc. Do not include any text before or after the JSON object."
            )
        )

    def analyze(self, image_url: str) -> str:
        """
        Analyzes a schematic image and returns the component data as a raw JSON string.
        Assumes the LLM is configured to return valid JSON.
        """
        self.progress.info("Schematic Analysis", f"Processing image from {image_url}")
        try:
            self.progress.info("AI Analysis", "Gemini LLM analyzing schematic...")
            message = self.prompt_template.format(image_url=image_url)

            # For this to be fully effective, the LLM should be in JSON mode
            response = self.llm.invoke([HumanMessage(content=message)])
            raw_response = response.content

            json_pattern = re.compile(r'(\{.*\}|\[.*\])', re.DOTALL)  # Regex to match JSON objects or arrays
            match = json_pattern.search(raw_response)
            if match:
                raw_response = match.group(0)
            else:
                raise SchematicAnalysisError("LLM did not return a valid JSON object.", image_url)

            self.progress.success("AI Analysis", "Received JSON response from LLM.")
            return raw_response

        except Exception as e:
            self.progress.error("Schematic Analysis", str(e))
            raise SchematicAnalysisError(f"Analysis failed: {e}", image_url=image_url)


# The ComponentAnalysisService is now much simpler
class ComponentAnalysisService:
    """Main service for component analysis and search operations."""

    def __init__(self, llm: ChatGoogleGenerativeAI, silicon_expert_client):
        self.schematic_analyzer = SchematicAnalyzer(llm)
        self.silicon_expert_client = silicon_expert_client

    def analyze_schematic(self, image_url: str) -> str:
        """Analyzes the schematic and returns the raw JSON string."""
        return self.schematic_analyzer.analyze(image_url)

    def search_component_data(self, components):
        """Searches component data using Silicon Expert."""
        # This method can now be simplified or moved to the workflow
        # if all searches go through the intelligent fallback logic.
        return self.silicon_expert_client.search_components(components)