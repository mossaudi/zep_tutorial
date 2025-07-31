# services/workflow.py
"""Enhanced workflow orchestration service with intelligent search integration."""
from dataclasses import asdict
from typing import List, Optional

from Bom_Chatbot.clients.silicon_expert import SiliconExpertClient
from Bom_Chatbot.models import EnhancedComponent, Component, SearchResult
from Bom_Chatbot.services.analysis import ComponentAnalysisService
from Bom_Chatbot.services.parsing import ParsingService
from Bom_Chatbot.services.progress import get_progress_tracker


class BOMWorkflowService:
    """Orchestrates complex workflows like schematic analysis and component enhancement."""

    def __init__(self,
                 analysis_service: ComponentAnalysisService,
                 parsing_service: ParsingService,
                 silicon_expert_client: SiliconExpertClient):
        self.analysis_service = analysis_service
        self.parsing_service = parsing_service
        self.silicon_expert_client = silicon_expert_client
        self.progress = get_progress_tracker()

    def run_schematic_analysis_workflow(self, image_url: str) -> SearchResult:
        """
        Orchestrates the full schematic analysis and intelligent component search.

        Returns:
            A SearchResult object containing the enhanced components and statistics.
        """
        self.progress.info("Workflow", f"Starting schematic analysis for {image_url}")

        # 1. Get raw component data from the analysis service
        analysis_json = self.analysis_service.analyze_schematic(image_url)

        # 2. Use the parsing service to get structured Component objects
        components = self.parsing_service.parse_and_convert_to_components(analysis_json)
        self.progress.info("Workflow", f"Parsed {len(components)} components from schematic.")

        # 3. Perform intelligent search with fallback for each component
        enhanced_components: List[EnhancedComponent] = []
        for component in components:
            enhanced = self._search_with_fallback(component)
            if enhanced:
                enhanced_components.append(enhanced)

        # 4. Compile and return the final SearchResult
        successful_searches = len([c for c in enhanced_components if c.silicon_expert_data])
        failed_searches = len(enhanced_components) - successful_searches

        self.progress.success("Workflow", "Schematic analysis workflow completed.")
        return SearchResult(
            success=True,
            components=[asdict(c) for c in enhanced_components],  # Convert to dict for serialization
            successful_searches=successful_searches,
            failed_searches=failed_searches
        )

    def _search_with_fallback(self, component: Component) -> Optional[EnhancedComponent]:
        """Performs a keyword search for a component."""
        self.progress.info("Smart Search", f"Performing keyword search for {component.name}")
        try:
            # The client's search_components method expects a list
            search_results = self.silicon_expert_client.search_components([component])
            if search_results and search_results[0].silicon_expert_data:
                self.progress.success("Keyword Search", f"Found data for {component.name}")
                return search_results[0]
        except Exception as e:
            self.progress.warning("Keyword Search", f"Failed for {component.name}: {e}")

        self.progress.warning("Component Search", f"No data found for {component.name}")
        # Return an EnhancedComponent with no SiliconExpert data as a fallback
        return EnhancedComponent(
            **asdict(component),
            search_result="No data found via keyword search"
        )