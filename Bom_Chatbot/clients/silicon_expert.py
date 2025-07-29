# clients/silicon_expert.py
"""Enhanced Silicon Expert API client with taxonomy support."""

import requests
import json
from typing import List, Dict, Any, Optional
from dataclasses import asdict
from difflib import get_close_matches

from Bom_Chatbot.config import SiliconExpertConfig
from Bom_Chatbot.constants import DEFAULT_PAGE_SIZE, MAX_SEARCH_RESULTS, HTTP_OK
from Bom_Chatbot.exceptions import (
    SiliconExpertError, AuthenticationError, ComponentSearchError, BOMError, ConfigurationError
)
from Bom_Chatbot.models import Component, SiliconExpertData, EnhancedComponent, BOMInfo
from Bom_Chatbot.services.progress import get_progress_tracker


class TaxonomyMapper:
    """Maps product line names to Silicon Expert taxonomy."""

    def __init__(self):
        self.taxonomy_map: Dict[str, Dict[str, Any]] = {}
        self.product_lines: List[str] = []
        self.is_loaded = False

    def _ensure_list(self, item):
        """Ensure an item is a list, wrapping single items in a list."""
        if item is None:
            return []
        if isinstance(item, list):
            return item
        return [item]

    def _extract_product_lines(self, node, hierarchy_path):
        """
        Recursively extract product lines from any level of the taxonomy tree.

        Args:
            node: Current node in the taxonomy tree
            hierarchy_path: List of dictionaries containing the path from root to current node
        """
        if not isinstance(node, dict):
            return

        # If this node has ProductLines, extract them
        if 'ProductLines' in node:
            product_lines_data = node['ProductLines']

            # Handle both dict with 'ProductLine' key and direct ProductLine data
            if isinstance(product_lines_data, dict):
                if 'ProductLine' in product_lines_data:
                    product_lines = self._ensure_list(product_lines_data['ProductLine'])
                else:
                    # If ProductLines is a dict but doesn't have 'ProductLine' key,
                    # treat the dict itself as a product line
                    product_lines = [product_lines_data]
            else:
                product_lines = self._ensure_list(product_lines_data)

            # Process each product line
            for pl in product_lines:
                if not isinstance(pl, dict):
                    continue

                pl_name = pl.get('plName', '')
                pl_id = pl.get('plID', '')
                product_count = pl.get('productCount', 0)
                package_features_type = pl.get('packageFeaturesType', 'S')

                if pl_name:
                    self.product_lines.append(pl_name)

                    # Build the taxonomy entry with full hierarchy
                    taxonomy_entry = {
                        'original_name': pl_name,
                        'pl_id': pl_id,
                        'product_count': product_count,
                        'package_features_type': package_features_type,
                    }

                    # Add hierarchy information
                    for i, level in enumerate(hierarchy_path):
                        level_name = level.get('name', '')
                        level_id = level.get('id', '')
                        level_type = level.get('type', '')

                        if level_type == 'pl_type':
                            taxonomy_entry['pl_type'] = level_name
                        elif level_type == 'main_category':
                            taxonomy_entry['main_category'] = level_name
                            taxonomy_entry['main_category_id'] = level_id
                        elif level_type == 'sub_category':
                            taxonomy_entry['sub_category'] = level_name
                            taxonomy_entry['sub_category_id'] = level_id
                        else:
                            # For additional levels, use generic naming
                            taxonomy_entry[f'level_{i}'] = level_name
                            taxonomy_entry[f'level_{i}_id'] = level_id
                            taxonomy_entry[f'level_{i}_type'] = level_type

                    self.taxonomy_map[pl_name.lower()] = taxonomy_entry

        # Recursively process child categories
        self._process_categories(node, hierarchy_path)

    def _process_categories(self, node, hierarchy_path):
        """Process different types of category lists in the node."""
        category_keys = [
            'MainCategoryList', 'SubCategoryList', 'CategoryList',
            'MainCategory', 'SubCategory', 'Category'
        ]

        for key in category_keys:
            if key in node:
                categories_data = node[key]

                # Handle different structures
                if isinstance(categories_data, dict):
                    # Check for nested category keys
                    for nested_key in ['MainCategory', 'SubCategory', 'Category']:
                        if nested_key in categories_data:
                            categories = self._ensure_list(categories_data[nested_key])
                            category_type = nested_key.lower().replace('category', '_category')
                            self._process_category_list(categories, hierarchy_path, category_type)
                            break
                    else:
                        # If no nested keys found, treat the dict as a single category
                        categories = [categories_data]
                        category_type = key.lower().replace('list', '').replace('category', '_category')
                        self._process_category_list(categories, hierarchy_path, category_type)
                else:
                    categories = self._ensure_list(categories_data)
                    category_type = key.lower().replace('list', '').replace('category', '_category')
                    self._process_category_list(categories, hierarchy_path, category_type)

    def _process_category_list(self, categories, hierarchy_path, category_type):
        """Process a list of categories."""
        for category in categories:
            if not isinstance(category, dict):
                continue

            cat_name = category.get('CategoryName', '')
            cat_id = category.get('CategoryID', '')

            # Create new hierarchy path for this category
            new_hierarchy = hierarchy_path + [{
                'name': cat_name,
                'id': cat_id,
                'type': category_type
            }]

            # Recursively process this category
            self._extract_product_lines(category, new_hierarchy)

    def load_taxonomy(self, taxonomy_data: Dict[str, Any]) -> None:
        """Load taxonomy data and create mapping structures."""
        if not taxonomy_data or not taxonomy_data.get('Status', {}).get('Success') == 'true':
            return

        self.taxonomy_map.clear()
        self.product_lines.clear()

        result = taxonomy_data.get('Result', {})
        taxonomy_list = result.get('TaxonomyList', {})

        # Handle both list and dict structures for TaxonomyList
        if isinstance(taxonomy_list, dict) and 'Taxonomy' in taxonomy_list:
            taxonomies = self._ensure_list(taxonomy_list['Taxonomy'])
        else:
            taxonomies = self._ensure_list(taxonomy_list)

        for taxonomy in taxonomies:
            if not isinstance(taxonomy, dict):
                continue

            pl_type = taxonomy.get('PlType', '')

            # Start with the pl_type in the hierarchy
            hierarchy_path = [{
                'name': pl_type,
                'id': '',
                'type': 'pl_type'
            }]

            # Extract product lines from this taxonomy branch
            self._extract_product_lines(taxonomy, hierarchy_path)

        self.is_loaded = True

    def find_product_line(self, search_term: str) -> Dict[str, Any]:
        """Find a product line by exact or fuzzy matching."""
        if not self.is_loaded:
            return {}

        search_lower = search_term.lower()

        # Exact match first
        if search_lower in self.taxonomy_map:
            return self.taxonomy_map[search_lower]

        # Fuzzy matching
        from difflib import get_close_matches
        matches = get_close_matches(search_lower, self.taxonomy_map.keys(), n=1, cutoff=0.6)
        if matches:
            return self.taxonomy_map[matches[0]]

        return {}

    def get_all_product_lines(self) -> List[str]:
        """Get all available product line names."""
        return sorted(self.product_lines)

    def get_product_lines_by_category(self, category_name: str) -> List[str]:
        """Get all product lines under a specific category."""
        category_lower = category_name.lower()
        matching_lines = []

        for pl_name, details in self.taxonomy_map.items():
            # Check all category levels
            for key, value in details.items():
                if 'category' in key and isinstance(value, str):
                    if category_lower in value.lower():
                        matching_lines.append(details['original_name'])
                        break

        return sorted(matching_lines)

    def get_hierarchy_info(self, product_line: str) -> Dict[str, Any]:
        """Get the full hierarchy information for a product line."""
        return self.find_product_line(product_line)

    def find_best_match(self, input_pl_name: str) -> Optional[Dict[str, Any]]:
        """Find the best matching product line from taxonomy."""
        if not self.is_loaded or not input_pl_name:
            return None

        input_lower = input_pl_name.lower()

        # Exact match first
        if input_lower in self.taxonomy_map:
            return self.taxonomy_map[input_lower]

        # Fuzzy matching using difflib
        close_matches = get_close_matches(
            input_lower,
            self.taxonomy_map.keys(),
            n=1,
            cutoff=0.6
        )

        if close_matches:
            return self.taxonomy_map[close_matches[0]]

        # Partial matching for common cases
        for key, value in self.taxonomy_map.items():
            if input_lower in key or key in input_lower:
                return value

        return None

    def get_all_product_lines(self) -> List[str]:
        """Get all available product lines."""
        return self.product_lines.copy()

    def get_taxonomy_info(self, pl_name: str) -> Optional[Dict[str, Any]]:
        """Get complete taxonomy information for a product line."""
        match = self.find_best_match(pl_name)
        return match


class SiliconExpertClient:
    """Enhanced client for Silicon Expert API operations with taxonomy support."""

    def __init__(self, config: SiliconExpertConfig):
        self.config = config
        self.session = requests.Session()
        self.is_authenticated = False
        self.progress = get_progress_tracker()
        self.taxonomy_mapper = TaxonomyMapper()

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
                # Load taxonomy after successful authentication
                self._load_taxonomy()
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

