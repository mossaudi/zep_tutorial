# services/workflow.py
"""Workflow orchestration service."""

import json
from typing import List, Dict, Any, Optional

from Bom_Chatbot.models import BOMInfo, WorkflowResult, EnhancedComponent
from Bom_Chatbot.constants import DEFAULT_BOM_COLUMNS
from Bom_Chatbot.services.analysis import ComponentAnalysisService
from Bom_Chatbot.services.formatter import ComponentDataConverter
from Bom_Chatbot.clients.silicon_expert import SiliconExpertClient
from Bom_Chatbot.services.progress import get_progress_tracker
from Bom_Chatbot.exceptions import BOMError, AgentError


class BOMWorkflowService:
    """Orchestrates complete BOM creation workflows."""

    def __init__(self,
                 analysis_service: ComponentAnalysisService,
                 silicon_expert_client: SiliconExpertClient):
        self.analysis_service = analysis_service
        self.silicon_expert_client = silicon_expert_client
        self.progress = get_progress_tracker()
        self.converter = ComponentDataConverter()

    def create_bom_from_schematic(self,
                                  image_url: str,
                                  bom_name: str,
                                  parent_path: str = "",
                                  description: str = "BOM created from schematic analysis",
                                  columns: Optional[List[str]] = None) -> WorkflowResult:
        """Complete workflow: Analyze schematic and create BOM."""

        if columns is None:
            columns = DEFAULT_BOM_COLUMNS.copy()

        self.progress.info("BOM Workflow", f"Starting complete workflow for BOM: {bom_name}")

        workflow_result = WorkflowResult(success=False)
        workflow_result.workflow_steps.append("Step 1: Analyzing schematic image...")

        try:
            # Step 1: Analyze schematic
            self.progress.info("Workflow Step 1", "Schematic analysis starting...")

            search_result = self.analysis_service.analyze_schematic(image_url)
            workflow_result.search_result = search_result

            if not search_result.success or not search_result.components:
                error_msg = search_result.error_message or "No components found in schematic"
                self.progress.error("Schematic Analysis", error_msg)
                workflow_result.error_message = error_msg
                return workflow_result

            self.progress.success(
                "Component Extraction",
                f"Extracted {len(search_result.components)} components"
            )

            # Step 2: Create empty BOM
            workflow_result.workflow_steps.append("Step 2: Creating empty BOM structure...")
            self.progress.info("Workflow Step 2", f"Creating BOM: {bom_name}")

            bom_info = BOMInfo(
                name=bom_name,
                parent_path=parent_path,
                description=description,
                columns=columns
            )

            bom_result = self.silicon_expert_client.create_empty_bom(bom_info)
            workflow_result.bom_creation = bom_result

            if bom_result.get("Status", {}).get("Success") == "true":
                self.progress.success("BOM Creation", f"BOM '{bom_name}' created successfully")
            else:
                error_msg = "Failed to create BOM"
                if "Status" in bom_result and "Message" in bom_result["Status"]:
                    error_msg += f": {bom_result['Status']['Message']}"

                self.progress.error("BOM Creation", error_msg)
                workflow_result.error_message = error_msg
                return workflow_result

            # Step 3: Prepare and add parts
            workflow_result.workflow_steps.append("Step 3: Adding components to BOM...")
            self.progress.info("Workflow Step 3", f"Preparing {len(search_result.components)} parts for BOM")

            # Convert components to BOM format
            parts_for_bom = self.converter.components_to_bom_parts(search_result.components)

            self.progress.info("Parts Addition", f"Adding {len(parts_for_bom)} parts to BOM...")

            parts_result = self.silicon_expert_client.add_parts_to_bom(
                bom_name, parent_path, parts_for_bom
            )
            workflow_result.parts_addition = parts_result

            if parts_result.get("Status", {}).get("Success") == "true":
                self.progress.success("Parts Addition", f"Successfully added {len(parts_for_bom)} parts")
            else:
                error_msg = "Failed to add parts to BOM"
                if "Status" in parts_result and "Message" in parts_result["Status"]:
                    error_msg += f": {parts_result['Status']['Message']}"

                self.progress.error("Parts Addition", error_msg)
                workflow_result.error_message = error_msg
                return workflow_result

            # Step 4: Success
            workflow_result.workflow_steps.append("Step 4: Workflow completed!")
            workflow_result.summary = (
                f"Successfully created BOM '{bom_name}' with {len(parts_for_bom)} "
                f"components from schematic"
            )
            workflow_result.success = True

            self.progress.success(
                "BOM Workflow",
                f"Complete! BOM '{bom_name}' created with {len(parts_for_bom)} components"
            )

            return workflow_result

        except BOMError as e:
            self.progress.error("BOM Workflow", f"BOM operation failed: {str(e)}")
            workflow_result.error_message = f"BOM operation failed: {str(e)}"
            return workflow_result

        except Exception as e:
            self.progress.error("BOM Workflow", f"Unexpected error: {str(e)}")
            workflow_result.error_message = f"Workflow failed: {str(e)}"
            return workflow_result

    def add_components_to_existing_bom(self,
                                       bom_name: str,
                                       parent_path: str,
                                       components: List[EnhancedComponent]) -> Dict[str, Any]:
        """Add components to an existing BOM."""
        self.progress.info("BOM Update", f"Adding {len(components)} components to BOM: {bom_name}")

        try:
            parts_for_bom = self.converter.components_to_bom_parts(components)

            result = self.silicon_expert_client.add_parts_to_bom(
                bom_name, parent_path, parts_for_bom
            )

            if result.get("Status", {}).get("Success") == "true":
                self.progress.success("BOM Update", f"Successfully added {len(parts_for_bom)} parts")
            else:
                error_msg = "Failed to add parts to BOM"
                if "Status" in result and "Message" in result["Status"]:
                    error_msg += f": {result['Status']['Message']}"
                self.progress.error("BOM Update", error_msg)

            return result

        except Exception as e:
            self.progress.error("BOM Update", str(e))
            raise BOMError(f"Failed to add components to BOM: {str(e)}", bom_name=bom_name)


class BOMManagementService:
    """Service for BOM management operations."""

    def __init__(self, silicon_expert_client: SiliconExpertClient):
        self.silicon_expert_client = silicon_expert_client
        self.progress = get_progress_tracker()

    def create_empty_bom(self,
                         name: str,
                         columns: List[str],
                         description: str = "",
                         parent_path: str = "") -> Dict[str, Any]:
        """Create an empty BOM with specified structure."""
        self.progress.info("BOM Creation", f"Creating empty BOM: {name}")

        try:
            bom_info = BOMInfo(
                name=name,
                columns=columns,
                description=description,
                parent_path=parent_path
            )

            result = self.silicon_expert_client.create_empty_bom(bom_info)

            if result.get("Status", {}).get("Success") == "true":
                self.progress.success("BOM Creation", f"BOM '{name}' created successfully")
            else:
                error_msg = "Failed to create BOM"
                if "Status" in result and "Message" in result["Status"]:
                    error_msg += f": {result['Status']['Message']}"
                self.progress.error("BOM Creation", error_msg)

            return result

        except Exception as e:
            self.progress.error("BOM Creation", str(e))
            raise BOMError(f"Failed to create BOM: {str(e)}", bom_name=name)

    def get_boms(self,
                 project_name: str = "",
                 bom_creation_date_from: str = "",
                 bom_creation_date_to: str = "",
                 bom_modification_date_from: str = "",
                 bom_modification_date_to: str = "") -> Dict[str, Any]:
        """Get BOM information with optional filters."""
        self.progress.info("BOM Retrieval", "Fetching BOM information...")

        try:
            result = self.silicon_expert_client.get_boms(
                project_name=project_name,
                bom_creation_date_from=bom_creation_date_from,
                bom_creation_date_to=bom_creation_date_to,
                bom_modification_date_from=bom_modification_date_from,
                bom_modification_date_to=bom_modification_date_to
            )

            # Count BOMs in result
            bom_count = 0
            if result.get("Status", {}).get("Success") == "true" and "Result" in result:
                bom_count = len(result["Result"]) if isinstance(result["Result"], list) else 0

            self.progress.success("BOM Retrieval", f"Retrieved information for {bom_count} BOMs")
            return result

        except Exception as e:
            self.progress.error("BOM Retrieval", str(e))
            raise BOMError(f"Failed to retrieve BOMs: {str(e)}")

    def add_parts_to_bom(self,
                         name: str,
                         parent_path: str,
                         parts_data: str) -> Dict[str, Any]:
        """Add parts to an existing BOM from JSON data."""
        self.progress.info("Parts Addition", f"Adding parts to BOM: {name}")

        try:
            # Parse parts JSON
            if isinstance(parts_data, str):
                parts_list = json.loads(parts_data)
            else:
                parts_list = parts_data

            # Validate parts data
            if not isinstance(parts_list, list):
                raise AgentError("Parts data must be a JSON array")

            for i, part in enumerate(parts_list):
                if not isinstance(part, dict):
                    raise AgentError(f"Part {i + 1} must be a JSON object")
                if 'mpn' not in part or 'manufacturer' not in part:
                    raise AgentError(f"Part {i + 1} must have 'mpn' and 'manufacturer' fields")

            result = self.silicon_expert_client.add_parts_to_bom(
                name, parent_path, parts_list
            )

            if result.get("Status", {}).get("Success") == "true":
                self.progress.success("Parts Addition", f"Successfully added {len(parts_list)} parts")
            else:
                error_msg = "Failed to add parts to BOM"
                if "Status" in result and "Message" in result["Status"]:
                    error_msg += f": {result['Status']['Message']}"
                self.progress.error("Parts Addition", error_msg)

            return result

        except json.JSONDecodeError as e:
            error_msg = f"Invalid JSON in parts data: {str(e)}"
            self.progress.error("Parts Addition", error_msg)
            raise AgentError(error_msg)

        except Exception as e:
            self.progress.error("Parts Addition", str(e))
            raise BOMError(f"Failed to add parts to BOM: {str(e)}", bom_name=name)