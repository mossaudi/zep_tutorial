# container.py
"""Manages service dependencies for the application."""

from langchain_google_genai import ChatGoogleGenerativeAI

from config import AppConfig
from clients.silicon_expert import SiliconExpertClient
from services.analysis import ComponentAnalysisService
from services.workflow import BOMWorkflowService
from services.bom_management import BOMManagementService
from services.parsing import ParsingService

class Container:
    """A container for managing and injecting service dependencies."""

    def __init__(self, config: AppConfig, llm: ChatGoogleGenerativeAI):
        self.config = config
        self.llm = llm

        # --- Clients ---
        self.silicon_expert_client = SiliconExpertClient(config.silicon_expert)

        # --- Services ---
        self.parsing_service = ParsingService()
        self.analysis_service = ComponentAnalysisService(llm, self.silicon_expert_client)
        self.bom_service = BOMManagementService(self.silicon_expert_client)
        self.workflow_service = BOMWorkflowService(
            analysis_service=self.analysis_service,
            parsing_service=self.parsing_service,
            silicon_expert_client=self.silicon_expert_client
        )