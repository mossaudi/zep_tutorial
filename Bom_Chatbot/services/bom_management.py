# services/bom_management.py
"""Enhanced workflow orchestration service with intelligent search integration."""

import json
from typing import List, Dict, Any

from Bom_Chatbot.clients.silicon_expert import SiliconExpertClient
from Bom_Chatbot.exceptions import BOMError, AgentError
from Bom_Chatbot.models import BOMInfo
from Bom_Chatbot.services.progress import get_progress_tracker


class BOMManagementService:
    """Service for BOM management operations (unchanged)."""

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