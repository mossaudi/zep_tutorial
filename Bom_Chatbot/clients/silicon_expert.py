# clients/silicon_expert.py
"""Silicon Expert API client."""

import requests
import json
from typing import List, Dict, Any, Optional
from dataclasses import asdict

from Bom_Chatbot.config import SiliconExpertConfig
from Bom_Chatbot.constants import DEFAULT_PAGE_SIZE, MAX_SEARCH_RESULTS, HTTP_OK
from Bom_Chatbot.exceptions import (
    SiliconExpertError, AuthenticationError, ComponentSearchError, BOMError, ConfigurationError, ParametricSearchError
)
from Bom_Chatbot.models import Component, SiliconExpertData, EnhancedComponent, BOMInfo
from Bom_Chatbot.services.progress import get_progress_tracker


class SiliconExpertClient:
    """Client for Silicon Expert API operations."""

    def __init__(self, config: SiliconExpertConfig):
        self.config = config
        self.session = requests.Session()
        self.is_authenticated = False
        self.progress = get_progress_tracker()

        if not config.is_valid():
            raise ConfigurationError("Invalid Silicon Expert configuration")

    def authenticate(self) -> bool:
        """Authenticate with Silicon Expert API."""
        self.progress.info("Authentication", "Connecting to Silicon Expert API...")

        try:
            response = self.session.post(
                f"{self.config.base_url}/search/authenticateUser",
                headers={'content-type': 'application/x-www-form-urlencoded'},
                data={
                    'login': self.config.username,
                    'apiKey': self.config.api_key
                }
            )

            if response.status_code == HTTP_OK:
                self.is_authenticated = True
                self.progress.success("Authentication", "Successfully authenticated")
                return True
            else:
                self.is_authenticated = False
                error_msg = f"HTTP {response.status_code}"
                self.progress.error("Authentication", error_msg)
                raise AuthenticationError(
                    f"Authentication failed: {error_msg}",
                    provider="Silicon Expert",
                    status_code=response.status_code
                )

        except requests.RequestException as e:
            self.progress.error("Authentication", str(e))
            raise AuthenticationError(
                f"Authentication request failed: {str(e)}",
                provider="Silicon Expert"
            )

    def _ensure_authenticated(self) -> None:
        """Ensure the client is authenticated."""
        if not self.is_authenticated:
            self.authenticate()

    def _build_search_query(self, component: Component) -> str:
        """Build search query from component data."""
        search_parts = component.get_search_parts()
        return ' '.join(search_parts)

    def search_component(self, component: Component) -> EnhancedComponent:
        """Search for a single component."""
        self._ensure_authenticated()

        search_query = self._build_search_query(component)
        if not search_query:
            enhanced = EnhancedComponent(**asdict(component))
            enhanced.search_result = "No component data available for search"
            return enhanced

        try:
            # Show search preview
            query_preview = search_query[:50] + "..." if len(search_query) > 50 else search_query
            self.progress.info("API Query", f"Searching: '{query_preview}'")

            params = {
                'fmt': 'json',
                'pageNumber': '1',
                'pageSize': str(DEFAULT_PAGE_SIZE),
                'description': search_query
            }

            response = self.session.get(
                f"{self.config.base_url}/search/partsearch",
                params=params
            )

            if response.status_code != HTTP_OK:
                raise ComponentSearchError(
                    f"Search request failed with HTTP {response.status_code}",
                    component_name=component.name,
                    search_query=search_query,
                    status_code=response.status_code
                )

            api_data = response.json()

            # Handle authentication errors
            if api_data and api_data.get('Status', {}).get('Code') == '39':
                self.progress.info("Re-authentication", "Session expired, re-authenticating...")
                self.is_authenticated = False
                self._ensure_authenticated()
                # Retry the request
                response = self.session.get(
                    f"{self.config.base_url}/search/partsearch",
                    params=params
                )
                if response.status_code == HTTP_OK:
                    api_data = response.json()
                else:
                    raise ComponentSearchError(
                        f"Retry request failed with HTTP {response.status_code}",
                        component_name=component.name,
                        search_query=search_query,
                        status_code=response.status_code
                    )

            # Process successful results
            enhanced = EnhancedComponent(**asdict(component))

            if (api_data and
                    api_data.get('Status', {}).get('Success') == 'true' and
                    'Result' in api_data and
                    isinstance(api_data['Result'], list) and
                    len(api_data['Result']) > 0):

                first_result = api_data['Result'][0]
                match_rating = first_result.get('MatchRating', 'Unknown')
                part_number_found = first_result.get('PartNumber', 'Unknown')

                self.progress.success(
                    "Component Found",
                    f"Match: {part_number_found} (Rating: {match_rating})"
                )

                # Create Silicon Expert data
                se_data = SiliconExpertData(
                    com_id=first_result.get('ComID'),
                    part_number=first_result.get('PartNumber'),
                    manufacturer=first_result.get('Manufacturer'),
                    description=first_result.get('Description'),
                    lifecycle=first_result.get('Lifecycle'),
                    rohs=first_result.get('RoHS'),
                    rohs_version=first_result.get('RoHSVersion'),
                    datasheet=first_result.get('Datasheet'),
                    product_line=first_result.get('PlName'),
                    taxonomy_path=first_result.get('TaxonomyPath'),
                    match_rating=first_result.get('MatchRating'),
                    match_comment=first_result.get('MatchRatingComment'),
                    yeol=first_result.get('YEOL'),
                    resilience_rating=first_result.get('ResilienceRating'),
                    military_status=first_result.get('MilitaryStatus'),
                    aml_status=first_result.get('AMLStatus'),
                    search_query=search_query,
                    total_items=str(api_data.get('TotalItems', 'Unknown'))
                )

                # Add additional matches if available
                if len(api_data['Result']) > 1:
                    for result in api_data['Result'][:MAX_SEARCH_RESULTS]:
                        match_info = {
                            'com_id': result.get('ComID'),
                            'part_number': result.get('PartNumber'),
                            'manufacturer': result.get('Manufacturer'),
                            'match_rating': result.get('MatchRating'),
                            'lifecycle': result.get('Lifecycle')
                        }
                        se_data.all_matches.append(match_info)

                enhanced.silicon_expert_data = se_data

            else:
                # Handle no results
                error_msg = "No matching parts found"
                if api_data and 'Status' in api_data:
                    status_msg = api_data['Status'].get('Message', '')
                    if status_msg and status_msg != "Successful Operation":
                        error_msg = f"API Message: {status_msg}"

                enhanced.search_result = error_msg
                self.progress.error("Component Search", f"No match found for {component.name}")

            return enhanced

        except ComponentSearchError:
            raise
        except Exception as e:
            self.progress.error("Component Search", str(e))
            enhanced = EnhancedComponent(**asdict(component))
            enhanced.search_result = f"Search error: {str(e)}"
            return enhanced

    def search_components(self, components: List[Component]) -> List[EnhancedComponent]:
        """Search for multiple components."""
        self.progress.info("Component Search", f"Processing {len(components)} components...")

        enhanced_components = []
        successful_searches = 0
        failed_searches = 0

        for i, component in enumerate(components, 1):
            self.progress.info(
                "Component Search",
                f"[{i}/{len(components)}] Searching: {component.name}"
            )

            try:
                enhanced = self.search_component(component)
                if enhanced.silicon_expert_data:
                    successful_searches += 1
                else:
                    failed_searches += 1
                enhanced_components.append(enhanced)

            except Exception as e:
                self.progress.error("Component Search", f"Failed to search {component.name}: {str(e)}")
                failed_searches += 1
                # Create a failed enhancement
                enhanced = EnhancedComponent(**asdict(component))
                enhanced.search_result = f"Search failed: {str(e)}"
                enhanced_components.append(enhanced)

        self.progress.success(
            "Component Enhancement",
            f"Processed {len(components)} components - {successful_searches} successful, {failed_searches} failed"
        )

        return enhanced_components

    def create_empty_bom(self, bom_info: BOMInfo) -> Dict[str, Any]:
        """Create an empty BOM."""
        self._ensure_authenticated()

        payload = {
            "name": bom_info.name,
            "columns": bom_info.columns,
            "description": bom_info.description
        }

        if bom_info.parent_path:
            payload["parentPath"] = bom_info.parent_path

        try:
            response = self.session.post(
                f"{self.config.base_url}/bom/add-empty-bom",
                headers={'Content-Type': 'application/json'},
                json=payload
            )

            if response.status_code == HTTP_OK:
                return response.json()
            else:
                raise BOMError(
                    f"Failed to create BOM: HTTP {response.status_code}",
                    bom_name=bom_info.name,
                    status_code=response.status_code,
                    response_data=response.text
                )

        except requests.RequestException as e:
            raise BOMError(
                f"BOM creation request failed: {str(e)}",
                bom_name=bom_info.name
            )

    def add_parts_to_bom(self, bom_name: str, parent_path: str,
                         parts: List[Dict[str, str]]) -> Dict[str, Any]:
        """Add parts to an existing BOM."""
        self._ensure_authenticated()

        payload = {
            "name": bom_name,
            "parentPath": parent_path,
            "parts": parts
        }

        try:
            response = self.session.post(
                f"{self.config.base_url}/bom/add-parts-to-bom",
                headers={'Content-Type': 'application/json'},
                json=payload
            )

            if response.status_code == HTTP_OK:
                return response.json()
            else:
                raise BOMError(
                    f"Failed to add parts to BOM: HTTP {response.status_code}",
                    bom_name=bom_name,
                    status_code=response.status_code,
                    response_data=response.text
                )

        except requests.RequestException as e:
            raise BOMError(
                f"Add parts request failed: {str(e)}",
                bom_name=bom_name
            )

    def get_boms(self, project_name: str = "",
                 bom_creation_date_from: str = "",
                 bom_creation_date_to: str = "",
                 bom_modification_date_from: str = "",
                 bom_modification_date_to: str = "") -> Dict[str, Any]:
        """Get BOM information."""
        self._ensure_authenticated()

        params = {"fmt": "json"}

        if project_name:
            params["projectName"] = project_name
        if bom_creation_date_from:
            params["bomCreationDateFrom"] = bom_creation_date_from
        if bom_creation_date_to:
            params["bomCreationDateTo"] = bom_creation_date_to
        if bom_modification_date_from:
            params["bomModificationDateFrom"] = bom_modification_date_from
        if bom_modification_date_to:
            params["bomModificationDateTo"] = bom_modification_date_to

        try:
            response = self.session.post(
                f"{self.config.base_url}/search/GetBOMs",
                params=params
            )

            if response.status_code == HTTP_OK:
                return response.json()
            else:
                raise BOMError(
                    f"Failed to get BOMs: HTTP {response.status_code}",
                    status_code=response.status_code,
                    response_data=response.text
                )

        except requests.RequestException as e:
            raise BOMError(f"Get BOMs request failed: {str(e)}")


def parametric_search(self,
                      product_line: str,
                      selected_filters: Optional[List[Dict[str, Any]]] = None,
                      level: int = 3,
                      keyword: str = "",
                      page_number: int = 1,
                      page_size: int = 50) -> Dict[str, Any]:
    """Search parts by technical criteria using parametric search."""
    self._ensure_authenticated()

    self.progress.info("Parametric Search", f"Searching product line: {product_line}")

    try:
        params = {
            'plName': product_line,
            'fmt': 'json',
            'level': str(level),
            'pageNumber': str(page_number),
            'pageSize': str(min(page_size, 500))  # Max 500 per API docs
        }

        if keyword:
            params['keyword'] = keyword

        if selected_filters:
            params['selectedFilters'] = json.dumps(selected_filters)

        response = self.session.post(
            f"{self.config.base_url}/search/parametric/getSearchResult",
            params=params
        )

        if response.status_code != HTTP_OK:
            raise ParametricSearchError(
                f"Parametric search failed with HTTP {response.status_code}",
                product_line=product_line,
                filters=json.dumps(selected_filters) if selected_filters else "",
                status_code=response.status_code
            )

        api_data = response.json()

        # Handle authentication errors
        if api_data and api_data.get('Status', {}).get('Code') == '39':
            self.progress.info("Re-authentication", "Session expired, re-authenticating...")
            self.is_authenticated = False
            self._ensure_authenticated()
            # Retry the request
            response = self.session.post(
                f"{self.config.base_url}/search/parametric/getSearchResult",
                params=params
            )
            if response.status_code == HTTP_OK:
                api_data = response.json()

        if api_data.get('Status', {}).get('Success') == 'true':
            total_items = api_data.get('Result', {}).get('TotalItems', '0')
            self.progress.success(
                "Parametric Search",
                f"Found {total_items} parts in {product_line}"
            )
        else:
            error_msg = api_data.get('Status', {}).get('Message', 'Unknown error')
            self.progress.error("Parametric Search", error_msg)

        return api_data

    except requests.RequestException as e:
        self.progress.error("Parametric Search", str(e))
        raise ParametricSearchError(
            f"Parametric search request failed: {str(e)}",
            product_line=product_line,
            filters=json.dumps(selected_filters) if selected_filters else ""
        )