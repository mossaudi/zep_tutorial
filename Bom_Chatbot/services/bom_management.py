# services/bom_management.py
"""Enhanced workflow orchestration service with intelligent search integration."""

import json
from dataclasses import asdict
from typing import List, Dict, Any

from Bom_Chatbot.clients.silicon_expert import SiliconExpertClient
from Bom_Chatbot.exceptions import BOMError, AgentError
from Bom_Chatbot.models import BOMInfo
from Bom_Chatbot.services.progress import get_progress_tracker
from Bom_Chatbot.utils.cache import TTLCache, cached_operation


class BOMManagementService:
    """Service for BOM management operations (unchanged)."""

    def __init__(self, silicon_expert_client: SiliconExpertClient):
        self.silicon_expert_client = silicon_expert_client
        self.progress = get_progress_tracker()
        self.cache = TTLCache(default_ttl=300)  # 5 minute cache

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

            if result.get("status", {}).get("success") == "TRUE":
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

    @cached_operation(lambda self: self.cache,
                      key_func=lambda self, **kwargs: f"boms_{hash(str(sorted(kwargs.items())))}")
    def get_boms(self,
                 project_name: str = "",
                 bom_creation_date_from: str = "",
                 bom_creation_date_to: str = "",
                 bom_modification_date_from: str = "",
                 bom_modification_date_to: str = "") -> Dict[str, Any]:
        """Get BOM information with enhanced parsing and tree structure."""
        self.progress.info("BOM Retrieval", "Fetching BOM information...")

        try:
            # Get raw API response
            api_result = self.silicon_expert_client.get_boms(
                project_name=project_name,
                bom_creation_date_from=bom_creation_date_from,
                bom_creation_date_to=bom_creation_date_to,
                bom_modification_date_from=bom_modification_date_from,
                bom_modification_date_to=bom_modification_date_to
            )

            # Parse into structured format
            from Bom_Chatbot.models import BOMTreeResult
            bom_tree = BOMTreeResult.from_api_response(api_result)

            if bom_tree.success:
                self.progress.success(
                    "BOM Retrieval",
                    f"Retrieved {bom_tree.total_projects} projects with {bom_tree.total_boms} total BOMs"
                )
            else:
                self.progress.error("BOM Retrieval", bom_tree.error_message or "Unknown error")

            # Return both the parsed structure and original API response
            return {
                "bom_tree": asdict(bom_tree),
                "raw_api_response": api_result,
                "success": bom_tree.success,
                "total_projects": bom_tree.total_projects,
                "total_boms": bom_tree.total_boms
            }

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