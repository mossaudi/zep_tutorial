from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent, ToolNode, tools_condition
import getpass
from langchain.chains import LLMChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
import os
import requests
from dotenv import load_dotenv
from typing import Annotated
import json
import pandas as pd
import time
from typing import Generator
import sys
from tabulate import tabulate
from typing_extensions import TypedDict
from IPython.display import Image, display
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# Load environment variables from .env file
load_dotenv()

# Get the access token from environment variables
ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")
BASE_URL = os.getenv("BASE_URL")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Silicon Expert API credentials
SILICON_EXPERT_USERNAME = os.getenv("SILICON_EXPERT_USERNAME")
SILICON_EXPERT_API_KEY = os.getenv("SILICON_EXPERT_API_KEY")
SILICON_EXPERT_BASE_URL = "https://api.siliconexpert.com/ProductAPI"

if not GOOGLE_API_KEY:
    GOOGLE_API_KEY = getpass.getpass("Enter API key for Google Gemini: ")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# Global session object for Silicon Expert API
silicon_expert_session = requests.Session()
is_authenticated = False


def authenticate_silicon_expert():
    """Authenticate with Silicon Expert API using session cookies."""
    global is_authenticated

    show_progress_step("Authentication", "Connecting to Silicon Expert API...")

    if not SILICON_EXPERT_USERNAME or not SILICON_EXPERT_API_KEY:
        show_error_step("Authentication", "Missing credentials")
        return False

    try:
        response = silicon_expert_session.post(
            f"{SILICON_EXPERT_BASE_URL}/search/authenticateUser",
            headers={'content-type': 'application/x-www-form-urlencoded'},
            data={
                'login': SILICON_EXPERT_USERNAME,
                'apiKey': SILICON_EXPERT_API_KEY
            }
        )

        if response.status_code == 200:
            is_authenticated = True
            show_completion_step("Authentication", "Successfully authenticated")
            return True
        else:
            is_authenticated = False
            show_error_step("Authentication", f"HTTP {response.status_code}")
            return False
    except Exception as e:
        show_error_step("Authentication", str(e))
        is_authenticated = False
        return False


def _ensure_authenticated():
    """Ensure Silicon Expert API is authenticated, retry if needed."""
    global is_authenticated

    if not is_authenticated:
        if not authenticate_silicon_expert():
            raise Exception("Could not authenticate with Silicon Expert API. Please check credentials.")

    return True


def _search_component_data_internal(components_json: str) -> str:
    """Internal function to search for additional component data using Silicon Expert Keyword Search API."""
    global is_authenticated

    try:
        components = json.loads(components_json)
        if not isinstance(components, list):
            return "Error: Input should be a JSON array of components"

        total_components = len(components)
        show_progress_step("Component Search", f"Processing {total_components} components...")

        # Authenticate if needed
        if not is_authenticated:
            if not authenticate_silicon_expert():
                return "Error: Could not authenticate with Silicon Expert API"

        enhanced_components = []
        successful_searches = 0
        failed_searches = 0

        for i, component in enumerate(components, 1):
            component_name = component.get('name', component.get('component_name', f'Component {i}'))
            show_progress_step("Component Search", f"[{i}/{total_components}] Searching: {component_name}")

            enhanced_component = component.copy()

            # Combine all available component data into search string
            search_parts = []

            part_number = component.get('part_number', '').strip()
            if part_number:
                search_parts.append(part_number)

            manufacturer = component.get('manufacturer', '').strip()
            if manufacturer:
                search_parts.append(manufacturer)

            component_name_clean = component.get('name', component.get('component_name', '')).strip()
            if component_name_clean and component_name_clean not in search_parts:
                search_parts.append(component_name_clean)

            description = component.get('description', '').strip()
            if description and len(description) < 100:
                search_parts.append(description)

            # Handle features (both string and list)
            features = component.get('features', '')
            if features:
                if isinstance(features, list):
                    features_str = ' '.join(str(item).strip() for item in features if item)
                    if features_str and len(features_str) < 100:
                        search_parts.append(features_str)
                elif isinstance(features, str):
                    features_str = features.strip()
                    if features_str and len(features_str) < 50:
                        search_parts.append(features_str)

            value = component.get('value', component.get('rating', '')).strip()
            if value:
                search_parts.append(value)

            combined_search_string = ' '.join(search_parts)

            if combined_search_string:
                try:
                    # Show what we're searching for
                    search_preview = combined_search_string[:50] + "..." if len(
                        combined_search_string) > 50 else combined_search_string
                    show_progress_step("API Query", f"Searching: '{search_preview}'")

                    params = {
                        'fmt': 'json',
                        'pageNumber': '1',
                        'pageSize': '5',
                        'description': combined_search_string
                    }

                    response = silicon_expert_session.get(
                        f"{SILICON_EXPERT_BASE_URL}/search/partsearch",
                        params=params
                    )

                    if response.status_code == 200:
                        api_data = response.json()

                        # Handle authentication errors
                        if (api_data and api_data.get('Status', {}).get('Code') == '39'):
                            show_progress_step("Re-authentication", "Session expired, re-authenticating...")
                            is_authenticated = False
                            if authenticate_silicon_expert():
                                retry_response = silicon_expert_session.get(
                                    f"{SILICON_EXPERT_BASE_URL}/search/partsearch",
                                    params=params
                                )
                                if retry_response.status_code == 200:
                                    api_data = retry_response.json()
                                else:
                                    enhanced_component[
                                        'se_search_result'] = f"Retry failed: HTTP {retry_response.status_code}"
                                    failed_searches += 1
                                    continue
                            else:
                                enhanced_component['se_search_result'] = "Re-authentication failed"
                                failed_searches += 1
                                continue

                        # Process successful results
                        if (api_data and
                                api_data.get('Status', {}).get('Success') == 'true' and
                                'Result' in api_data and
                                isinstance(api_data['Result'], list) and
                                len(api_data['Result']) > 0):

                            first_result = api_data['Result'][0]
                            match_rating = first_result.get('MatchRating', 'Unknown')
                            part_number_found = first_result.get('PartNumber', 'Unknown')

                            show_completion_step("Component Found",
                                                 f"Match: {part_number_found} (Rating: {match_rating})")

                            # Extract and store enhanced data
                            enhanced_component.update({
                                'se_com_id': first_result.get('ComID'),
                                'se_part_number': first_result.get('PartNumber'),
                                'se_manufacturer': first_result.get('Manufacturer'),
                                'se_description': first_result.get('Description'),
                                'se_lifecycle': first_result.get('Lifecycle'),
                                'se_rohs': first_result.get('RoHS'),
                                'se_rohs_version': first_result.get('RoHSVersion'),
                                'se_datasheet': first_result.get('Datasheet'),
                                'se_product_line': first_result.get('PlName'),
                                'se_taxonomy_path': first_result.get('TaxonomyPath'),
                                'se_match_rating': first_result.get('MatchRating'),
                                'se_match_comment': first_result.get('MatchRatingComment'),
                                'se_yeol': first_result.get('YEOL'),
                                'se_resilience_rating': first_result.get('ResilienceRating'),
                                'se_military_status': first_result.get('MilitaryStatus'),
                                'se_aml_status': first_result.get('AMLStatus'),
                                'se_search_query': combined_search_string
                            })

                            enhanced_component['se_total_items'] = api_data.get('TotalItems', 'Unknown')

                            if len(api_data['Result']) > 1:
                                enhanced_component['se_all_matches'] = []
                                for result in api_data['Result'][:5]:
                                    match_info = {
                                        'com_id': result.get('ComID'),
                                        'part_number': result.get('PartNumber'),
                                        'manufacturer': result.get('Manufacturer'),
                                        'match_rating': result.get('MatchRating'),
                                        'lifecycle': result.get('Lifecycle')
                                    }
                                    enhanced_component['se_all_matches'].append(match_info)

                            successful_searches += 1

                        else:
                            # Handle no results
                            error_msg = "No matching parts found"
                            if api_data and 'Status' in api_data:
                                status_code = api_data['Status'].get('Code', '')
                                status_msg = api_data['Status'].get('Message', '')
                                if status_msg and status_msg != "Successful Operation":
                                    error_msg = f"API Message: {status_msg}"

                            enhanced_component['se_search_result'] = error_msg
                            enhanced_component['se_search_query'] = combined_search_string
                            failed_searches += 1
                            show_error_step("Component Search", f"No match found for {component_name}")

                    else:
                        enhanced_component['se_search_result'] = f"HTTP Error: {response.status_code}"
                        enhanced_component['se_search_query'] = combined_search_string
                        failed_searches += 1
                        show_error_step("API Request", f"HTTP {response.status_code}")

                except Exception as api_error:
                    enhanced_component['se_search_result'] = f"Search error: {str(api_error)}"
                    enhanced_component['se_search_query'] = combined_search_string
                    failed_searches += 1
                    show_error_step("Component Search", str(api_error))
            else:
                enhanced_component['se_search_result'] = "No component data available for search"
                failed_searches += 1

            enhanced_components.append(enhanced_component)

        # Final summary
        show_completion_step("Component Enhancement",
                             f"Processed {total_components} components - {successful_searches} successful, {failed_searches} failed")

        return json.dumps(enhanced_components, indent=2)

    except json.JSONDecodeError:
        show_error_step("JSON Parsing", "Invalid JSON format in input")
        return "Error: Invalid JSON format in input"
    except Exception as e:
        show_error_step("Component Search", str(e))
        return f"Error searching component data: {str(e)}"


def _clean_json_response(response_text: str) -> str:
    """Clean up LLM response to extract valid JSON by finding JSON boundaries."""
    import json
    import re

    # Remove markdown code blocks first
    response_text = re.sub(r'```(?:json)?\s*', '', response_text, flags=re.IGNORECASE)
    response_text = response_text.strip()

    # Try to find JSON array or object
    def find_json_boundaries(text, start_char, end_char):
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

        return start_pos, len(text)  # If no proper ending found, take rest

    # Try to find JSON array first
    start_pos, end_pos = find_json_boundaries(response_text, '[', ']')

    # If no array found, try to find JSON object
    if start_pos is None:
        start_pos, end_pos = find_json_boundaries(response_text, '{', '}')

    # If still no JSON found, return original
    if start_pos is None:
        return response_text

    # Extract the JSON part
    json_text = response_text[start_pos:end_pos]

    # Validate by parsing
    try:
        parsed = json.loads(json_text)
        return json.dumps(parsed, indent=2)
    except json.JSONDecodeError:
        # If parsing fails, try some minimal cleanup
        try:
            # Only fix obvious issues without risking content corruption
            cleaned = json_text.strip()
            # Remove trailing commas before closing brackets/braces
            cleaned = re.sub(r',(\s*[\]\}])', r'\1', cleaned)

            parsed = json.loads(cleaned)
            return json.dumps(parsed, indent=2)
        except:
            # Return the extracted content even if not valid JSON
            # Let the calling function handle the error
            return json_text


def format_components_table(components_json: str) -> str:
    """Format component data into a readable table and provide next step suggestions."""
    try:
        # Check if tabulate is available
        try:
            from tabulate import tabulate
        except ImportError:
            tabulate = None
            show_progress_step("Import Warning", "tabulate not available, using fallback formatting")

        # Handle None or empty input
        if not components_json:
            return "No component data available to display."

        # Handle case where components_json is not a string
        if not isinstance(components_json, str):
            return f"Error: Expected string input, got {type(components_json)}"

        # Try to parse JSON
        try:
            components = json.loads(components_json)
        except json.JSONDecodeError as e:
            return f"Error: Invalid JSON format - {str(e)}\nReceived data: {components_json[:200]}..."

        if not isinstance(components, list):
            return "Error: Component data should be a JSON array/list"

        if not components:
            return "No components found to display."

        # Prepare data for table display
        table_data = []
        for i, component in enumerate(components, 1):
            # Safe get with fallbacks
            component_name = (component.get('name') or
                              component.get('component_name') or
                              f'Component {i}')

            part_number = (component.get('se_part_number') or
                           component.get('part_number') or
                           'N/A')

            manufacturer = (component.get('se_manufacturer') or
                            component.get('manufacturer') or
                            'N/A')

            description = (component.get('se_description') or
                           component.get('description') or
                           'N/A')

            # Truncate long descriptions
            if description != 'N/A' and len(description) > 50:
                description = description[:50] + '...'

            lifecycle = component.get('se_lifecycle', 'N/A')
            rohs = component.get('se_rohs', 'N/A')
            match_rating = component.get('se_match_rating', 'N/A')

            row = {
                '#': i,
                'Component': component_name,
                'Part Number': part_number,
                'Manufacturer': manufacturer,
                'Description': description,
                'Lifecycle': lifecycle,
                'RoHS': rohs,
                'Match Rating': match_rating
            }
            table_data.append(row)

        # Create table using tabulate (with error handling)
        table_output = "\n" + "=" * 120 + "\n"
        table_output += "COMPONENT ANALYSIS RESULTS\n"
        table_output += "=" * 120 + "\n"

        # Create table using tabulate (with comprehensive error handling)
        table_output = "\n" + "=" * 120 + "\n"
        table_output += "COMPONENT ANALYSIS RESULTS\n"
        table_output += "=" * 120 + "\n"

        try:
            # Validate table_data structure first
            if not table_data or not isinstance(table_data, list):
                raise ValueError("Invalid table data structure")

            # Check if all required keys exist in each row
            required_keys = ['#', 'Component', 'Part Number', 'Manufacturer', 'Description', 'Lifecycle', 'RoHS',
                             'Match Rating']
            for row in table_data:
                if not isinstance(row, dict) or not all(key in row for key in required_keys):
                    raise ValueError("Missing required keys in table data")

            # Try tabulate if available
            if tabulate is not None:
                # Ensure all values are strings to avoid NoneType issues
                clean_table_data = []
                for row in table_data:
                    clean_row = {}
                    for key, value in row.items():
                        # Convert None to 'N/A' and ensure string type
                        if value is None:
                            clean_row[key] = 'N/A'
                        else:
                            clean_row[key] = str(value)
                    clean_table_data.append(clean_row)

                table_output += tabulate(clean_table_data, headers='keys', tablefmt='grid',
                                         maxcolwidths=[3, 20, 20, 15, 40, 12, 8, 12])
            else:
                raise ImportError("tabulate not available")

        except Exception as table_error:
            # Comprehensive fallback - plain text format
            table_output += f"Note: Using fallback formatting due to: {str(table_error)}\n\n"

            # Create a simple but readable format
            for i, row in enumerate(table_data):
                try:
                    table_output += f"COMPONENT #{row.get('#', i + 1)}\n"
                    table_output += f"├─ Name: {row.get('Component', 'Unknown')}\n"
                    table_output += f"├─ Part Number: {row.get('Part Number', 'N/A')}\n"
                    table_output += f"├─ Manufacturer: {row.get('Manufacturer', 'N/A')}\n"
                    table_output += f"├─ Description: {row.get('Description', 'N/A')}\n"
                    table_output += f"├─ Lifecycle: {row.get('Lifecycle', 'N/A')}\n"
                    table_output += f"├─ RoHS: {row.get('RoHS', 'N/A')}\n"
                    table_output += f"└─ Match Rating: {row.get('Match Rating', 'N/A')}\n"
                    table_output += "\n" + "-" * 60 + "\n"
                except Exception as row_error:
                    table_output += f"Error formatting component {i + 1}: {str(row_error)}\n"
                    table_output += f"Raw data: {str(row)}\n\n"

        table_output += "\n" + "=" * 120 + "\n"

        # Add summary statistics
        total_components = len(components)
        components_with_data = len([c for c in components if c.get('se_part_number')])
        lifecycle_counts = {}
        rohs_counts = {}

        for component in components:
            lifecycle = component.get('se_lifecycle', 'Unknown')
            rohs = component.get('se_rohs', 'Unknown')
            lifecycle_counts[lifecycle] = lifecycle_counts.get(lifecycle, 0) + 1
            rohs_counts[rohs] = rohs_counts.get(rohs, 0) + 1

        table_output += f"\nSUMMARY:\n"
        table_output += f"- Total Components: {total_components}\n"
        table_output += f"- Components with Silicon Expert Data: {components_with_data}\n"
        table_output += f"- Data Match Rate: {(components_with_data / total_components) * 100:.1f}%\n"

        if lifecycle_counts and lifecycle_counts != {'Unknown': total_components}:
            table_output += f"\nLifecycle Distribution:\n"
            for lifecycle, count in lifecycle_counts.items():
                table_output += f"  • {lifecycle}: {count} components\n"

        if rohs_counts and rohs_counts != {'Unknown': total_components}:
            table_output += f"\nRoHS Compliance:\n"
            for rohs, count in rohs_counts.items():
                table_output += f"  • {rohs}: {count} components\n"

        # Add next steps suggestions
        table_output += "\n" + "=" * 120 + "\n"
        table_output += "SUGGESTED NEXT STEPS:\n"
        table_output += "=" * 120 + "\n"
        table_output += "1. CREATE NEW BOM:\n"
        table_output += "   → Use 'create_bom_from_schematic' to automatically create a BOM with these components\n"
        table_output += "   → Or use 'create_empty_bom' to create a custom BOM structure\n\n"

        table_output += "2. ADD TO EXISTING BOM:\n"
        table_output += "   → Use 'get_boms' to view existing BOMs\n"
        table_output += "   → Then use 'add_parts_to_bom' to add these components to an existing BOM\n\n"

        table_output += "3. COMPONENT ANALYSIS:\n"
        table_output += "   → Review lifecycle status for obsolescence planning\n"
        table_output += "   → Check RoHS compliance for regulatory requirements\n"
        table_output += "   → Examine match ratings for data accuracy\n\n"

        table_output += "4. DETAILED COMPONENT INFO:\n"
        table_output += "   → Use 'search_component_data' with specific part numbers for detailed analysis\n"
        table_output += "   → Access datasheets and technical specifications\n\n"

        # Add commands the user can copy-paste
        table_output += "EXAMPLE COMMANDS:\n"
        table_output += f"• Create new BOM: create_empty_bom(name='MyProject_BOM', columns='[\"cpn\", \"mpn\", \"manufacturer\", \"description\", \"quantity\"]')\n"
        table_output += f"• View existing BOMs: get_boms()\n"
        table_output += f"• Add parts to existing BOM: add_parts_to_bom(name='BOM_NAME', parent_path='PROJECT_PATH', parts_json='COMPONENT_DATA')\n"
        table_output += "=" * 120 + "\n"

        return table_output

    except Exception as e:
        return f"Error formatting component table: {str(e)}\nInput type: {type(components_json)}\nInput preview: {str(components_json)[:200] if components_json else 'None'}"


# Enhanced State to track component analysis results
class State(TypedDict):
    messages: Annotated[list, add_messages]
    component_data: str
    needs_table_display: bool
    progress_messages: list  # Track progress steps
    current_step: str  # Current operation being performed

# Add progress tracking utilities
def show_progress_step(step_name: str, details: str = "") -> str:
    """Display a progress step with timestamp"""
    timestamp = time.strftime("%H:%M:%S")
    progress_msg = f"🔄 [{timestamp}] {step_name}"
    if details:
        progress_msg += f": {details}"
    print(progress_msg)
    return progress_msg

def show_completion_step(step_name: str, result: str = "") -> str:
    """Display a completed step"""
    timestamp = time.strftime("%H:%M:%S")
    completion_msg = f"✅ [{timestamp}] {step_name} completed"
    if result:
        completion_msg += f" - {result}"
    print(completion_msg)
    return completion_msg

def show_error_step(step_name: str, error: str) -> str:
    """Display an error step"""
    timestamp = time.strftime("%H:%M:%S")
    error_msg = f"❌ [{timestamp}] {step_name} failed: {error}"
    print(error_msg)
    return error_msg

# Initialize the Gemini language model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.7
)


@tool
def analyze_schematic(image_url: str) -> str:
    """Analyze a schematic design image from a URL and return enhanced component details with Silicon Expert data."""

    show_progress_step("Schematic Analysis", f"Processing image from {image_url}")

    schematic_prompt = PromptTemplate(
        input_variables=["image_url"],
        template="As an expert electrical engineer, please analyze the schematic design at the following URL: {image_url}. List all components in JSON format with component name, part number, manufacturer, and features. Return ONLY valid JSON array format."
    )

    message = schematic_prompt.format(image_url=image_url)
    try:
        # Step 1: Image Analysis
        show_progress_step("AI Analysis", "Gemini LLM analyzing schematic components...")
        response = llm.invoke([HumanMessage(content=message)])
        raw_response = response.content
        show_completion_step("AI Analysis", f"Identified components in schematic")

        # Step 2: JSON Cleaning
        show_progress_step("Data Processing", "Cleaning and validating JSON response...")
        cleaned_json = _clean_json_response(raw_response)

        # Count components for progress
        try:
            components = json.loads(cleaned_json)
            component_count = len(components) if isinstance(components, list) else 0
            show_completion_step("Data Processing", f"Parsed {component_count} components")
        except:
            component_count = 0

        # Step 3: Silicon Expert Enhancement
        show_progress_step("Data Enhancement", f"Searching Silicon Expert database for {component_count} components...")
        enhanced_data = _search_component_data_internal(cleaned_json)
        show_completion_step("Schematic Analysis", "Component data enhanced with Silicon Expert information")

        return enhanced_data

    except Exception as e:
        show_error_step("Schematic Analysis", str(e))
        return f"Error analyzing schematic: {str(e)}"


@tool
def search_component_data(components_json: str) -> str:
    """Search for additional component data using Silicon Expert Keyword Search API.

    Args:
        components_json: JSON string containing component data from schematic analysis

    Returns:
        Enhanced JSON string with additional component information from Silicon Expert API
    """
    enhanced_data = _search_component_data_internal(components_json)
    return f"COMPONENT_SEARCH_COMPLETE:{enhanced_data}"


# NEW BOM MANAGEMENT TOOLS

@tool
def create_empty_bom(name: str, columns: str, description: str = "", parent_path: str = "") -> str:
    """Create an empty BOM in SiliconExpert P5 BOM manager.

    Args:
        name: New BOM's name (mandatory)
        columns: JSON array of column names to be created in the BOM (mandatory).
                Default columns: cpn, mpn, manufacturer, description, quantity, price, uploadedcomments, uploadedlifecycle
                Additional custom columns can be added to default columns
        description: New BOM's description (optional)
        parent_path: Saving path in pattern "project>subproject1>subproject2" (optional)

    Returns:
        JSON response with operation status
    """
    try:
        _ensure_authenticated()

        # Parse columns if it's a JSON string
        if isinstance(columns, str):
            columns_list = json.loads(columns)
        else:
            columns_list = columns

        payload = {
            "name": name,
            "columns": columns_list,
            "description": description
        }

        if parent_path:
            payload["parentPath"] = parent_path

        response = silicon_expert_session.post(
            f"{SILICON_EXPERT_BASE_URL}/bom/add-empty-bom",
            headers={'Content-Type': 'application/json'},
            json=payload
        )

        if response.status_code == 200:
            return json.dumps(response.json(), indent=2)
        else:
            return f"Error creating BOM: HTTP {response.status_code} - {response.text}"

    except Exception as e:
        return f"Error creating empty BOM: {str(e)}"


@tool
def get_boms(project_name: str = "", bom_creation_date_from: str = "",
             bom_creation_date_to: str = "", bom_modification_date_from: str = "",
             bom_modification_date_to: str = "") -> str:
    """Get BOM meta information from SiliconExpert P5 BOM manager.

    Args:
        project_name: Filter by project name (optional)
        bom_creation_date_from: Filter on BOM creation date from (optional)
        bom_creation_date_to: Filter on BOM creation date to (optional)
        bom_modification_date_from: Filter on BOM modification date from (optional)
        bom_modification_date_to: Filter on BOM modification date to (optional)

    Returns:
        JSON response with BOM information including projects, BOMs, creation dates, etc.
    """
    try:
        _ensure_authenticated()

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

        response = silicon_expert_session.post(
            f"{SILICON_EXPERT_BASE_URL}/search/GetBOMs",
            params=params
        )

        if response.status_code == 200:
            return json.dumps(response.json(), indent=2)
        else:
            return f"Error getting BOMs: HTTP {response.status_code} - {response.text}"

    except Exception as e:
        return f"Error getting BOMs: {str(e)}"


@tool
def add_parts_to_bom(name: str, parent_path: str, parts_json: str) -> str:
    """Add/upload parts into a BOM.

    Args:
        name: BOM name (mandatory)
        parent_path: BOM saving path in pattern "project>subproject1>subproject2" (can be empty string)
        parts_json: JSON array of part data with column name: column value pairs.
                   'mpn' and 'manufacturer' are mandatory for each part.
                   Example: '[{"mpn": "bav99", "manufacturer": "vishay", "cpn": "000824-OSP"}]'

    Returns:
        JSON response with operation status
    """
    try:
        _ensure_authenticated()

        # Parse parts if it's a JSON string
        if isinstance(parts_json, str):
            parts_list = json.loads(parts_json)
        else:
            parts_list = parts_json

        payload = {
            "name": name,
            "parentPath": parent_path,
            "parts": parts_list
        }

        response = silicon_expert_session.post(
            f"{SILICON_EXPERT_BASE_URL}/bom/add-parts-to-bom",
            headers={'Content-Type': 'application/json'},
            json=payload
        )

        if response.status_code == 200:
            return json.dumps(response.json(), indent=2)
        else:
            return f"Error adding parts to BOM: HTTP {response.status_code} - {response.text}"

    except Exception as e:
        return f"Error adding parts to BOM: {str(e)}"


@tool
def create_bom_from_schematic(image_url: str, bom_name: str, parent_path: str = "",
                              description: str = "BOM created from schematic analysis") -> str:
    """Complete workflow: Analyze schematic and create BOM with the identified components."""

    show_progress_step("BOM Workflow", f"Starting complete workflow for BOM: {bom_name}")

    try:
        results = {"workflow_steps": [], "timestamps": []}

        # Step 1: Analyze schematic
        step_msg = "Step 1: Analyzing schematic image..."
        results["workflow_steps"].append(step_msg)
        show_progress_step("Workflow Step 1", "Schematic analysis starting...")

        schematic_result = analyze_schematic(image_url)
        results["schematic_analysis"] = schematic_result

        # Extract components
        try:
            start_idx = schematic_result.find('[')
            end_idx = schematic_result.rfind(']') + 1
            if start_idx != -1 and end_idx != 0:
                components_json = schematic_result[start_idx:end_idx]
                components = json.loads(components_json)
                show_completion_step("Component Extraction", f"Extracted {len(components)} components")
            else:
                show_error_step("Component Extraction", "Could not extract component data")
                return "Error: Could not extract component data from schematic analysis"
        except Exception as e:
            show_error_step("Component Parsing", str(e))
            return "Error: Could not parse component data from schematic analysis"

        # Step 2: Create empty BOM
        step_msg = "Step 2: Creating empty BOM structure..."
        results["workflow_steps"].append(step_msg)
        show_progress_step("Workflow Step 2", f"Creating BOM: {bom_name}")

        columns = ["cpn", "mpn", "manufacturer", "description", "quantity", "uploadedcomments"]
        bom_result = create_empty_bom(bom_name, json.dumps(columns), description, parent_path)
        results["bom_creation"] = json.loads(bom_result)

        if results["bom_creation"].get("Status", {}).get("Success") == "true":
            show_completion_step("BOM Creation", f"BOM '{bom_name}' created successfully")
        else:
            show_error_step("BOM Creation", "Failed to create BOM")
            return json.dumps(results, indent=2)

        # Step 3: Prepare and add parts
        step_msg = "Step 3: Adding components to BOM..."
        results["workflow_steps"].append(step_msg)
        show_progress_step("Workflow Step 3", f"Preparing {len(components)} parts for BOM")

        parts_for_bom = []
        for component in components:
            part_data = {}

            if component.get('part_number') or component.get('se_part_number'):
                part_data['mpn'] = component.get('se_part_number') or component.get('part_number')
            elif component.get('mpn'):
                part_data['mpn'] = component.get('mpn')
            else:
                part_data['mpn'] = component.get('name', 'Unknown')

            if component.get('manufacturer') or component.get('se_manufacturer'):
                part_data['manufacturer'] = component.get('se_manufacturer') or component.get('manufacturer')
            else:
                part_data['manufacturer'] = 'Unknown'

            if component.get('description') or component.get('se_description'):
                part_data['description'] = component.get('se_description') or component.get('description')

            part_data['quantity'] = component.get('quantity', '1')
            part_data['uploadedcomments'] = f"Added from schematic analysis"
            parts_for_bom.append(part_data)

        show_progress_step("Parts Addition", f"Adding {len(parts_for_bom)} parts to BOM...")
        parts_result = add_parts_to_bom(bom_name, parent_path, json.dumps(parts_for_bom))
        results["parts_addition"] = json.loads(parts_result)

        if json.loads(parts_result).get("Status", {}).get("Success") == "true":
            show_completion_step("Parts Addition", f"Successfully added {len(parts_for_bom)} parts")
        else:
            show_error_step("Parts Addition", "Failed to add parts to BOM")

        results["workflow_steps"].append("Step 4: Workflow completed!")
        results[
            "summary"] = f"Successfully created BOM '{bom_name}' with {len(parts_for_bom)} components from schematic"

        show_completion_step("BOM Workflow", f"Complete! BOM '{bom_name}' created with {len(parts_for_bom)} components")

        return json.dumps(results, indent=2)

    except Exception as e:
        show_error_step("BOM Workflow", str(e))
        return f"Error in create BOM from schematic workflow: {str(e)}"


# Create the tools list with all tools
tools = [
    analyze_schematic,
    search_component_data,
    create_empty_bom,
    get_boms,
    add_parts_to_bom,
    create_bom_from_schematic
]

# Bind tools to the LLM
llm_with_tools = llm.bind_tools(tools)

# Create ToolNode with the tools
tool_node = ToolNode(tools)

# Create the graph builder
graph_builder = StateGraph(State)


def chatbot(state: State):
    """Main chatbot node that processes user input and decides next actions."""
    response = llm_with_tools.invoke(state["messages"])

    # Check if any tool was called and if it returned component data
    if hasattr(response, 'tool_calls') and response.tool_calls:
        # If tools were called, don't process the response content yet
        # Let the tools execute first
        return {"messages": [response]}

    # Check if the last message in the conversation contains component analysis results
    # This happens after tool execution
    if state["messages"]:
        last_message = state["messages"][-1]
        if hasattr(last_message, 'content') and last_message.content:
            # Check if this is a tool response containing JSON component data
            try:
                # Try to parse as JSON to see if it's component data
                parsed_data = json.loads(last_message.content)
                if (isinstance(parsed_data, list) and
                        len(parsed_data) > 0 and
                        isinstance(parsed_data[0], dict) and
                        ('part_number' in parsed_data[0] or 'name' in parsed_data[0] or 'component_name' in parsed_data[
                            0])):
                    # This looks like component data, trigger table display
                    return {
                        "messages": [response],
                        "component_data": last_message.content,
                        "needs_table_display": True
                    }
            except (json.JSONDecodeError, TypeError, KeyError, IndexError):
                # Not JSON component data, continue normally
                pass

    return {
        "messages": [response],
        "component_data": "",
        "needs_table_display": False
    }


def table_display_node(state: State):
    """Node that displays component analysis results in a formatted table with next step suggestions."""
    component_data = state.get("component_data", "")

    # Debug logging
    show_progress_step("Table Display", f"Component data type: {type(component_data)}")

    # If no component_data in state, try to get it from the last tool message
    if not component_data and state.get("messages"):
        show_progress_step("Table Display", "Searching for component data in messages...")
        for message in reversed(state["messages"]):
            if hasattr(message, 'content') and message.content:
                # Check if it's a tool message
                if hasattr(message, 'name') and message.name in ['analyze_schematic', 'search_component_data']:
                    component_data = message.content
                    show_progress_step("Table Display", f"Found component data from tool: {message.name}")
                    break
                # Check if it looks like JSON component data
                elif (isinstance(message.content, str) and
                      message.content.strip().startswith('[') and
                      message.content.strip().endswith(']')):
                    try:
                        # Validate it's actually component data
                        test_data = json.loads(message.content)
                        if (isinstance(test_data, list) and
                                len(test_data) > 0 and
                                isinstance(test_data[0], dict) and
                                any(key in test_data[0] for key in ['name', 'component_name', 'part_number'])):
                            component_data = message.content
                            show_progress_step("Table Display", "Found component data from message content")
                            break
                    except:
                        continue

    if not component_data:
        show_error_step("Table Display", "No component data available")
        return {"messages": [AIMessage(content="No component data available to display.")]}

    try:
        show_progress_step("Table Display", "Formatting component table...")
        # Format the component data into a table with suggestions
        table_output = format_components_table(component_data)
        show_completion_step("Table Display", "Component table formatted successfully")

        # Create a message with the formatted table
        table_message = AIMessage(content=table_output)

        return {
            "messages": [table_message],
            "needs_table_display": False  # Reset the flag
        }

    except Exception as e:
        show_error_step("Table Display", str(e))
        error_message = AIMessage(content=f"Error displaying component table: {str(e)}")
        return {
            "messages": [error_message],
            "needs_table_display": False
        }


def should_display_table(state: State) -> str:
    """Conditional edge function to determine if table display is needed."""
    # Check if we need to display table from state flag
    if state.get("needs_table_display", False):
        show_progress_step("Flow Decision", "Table display requested via state flag")
        return "table_display"

    # Check if the last message is from a tool and contains component data
    if state.get("messages"):
        last_message = state["messages"][-1]

        # Check for tool messages
        if (hasattr(last_message, 'content') and
                last_message.content and
                hasattr(last_message, 'name') and
                last_message.name in ['analyze_schematic', 'search_component_data']):

            try:
                # Try to parse the tool response as JSON
                parsed_data = json.loads(last_message.content)
                if (isinstance(parsed_data, list) and
                        len(parsed_data) > 0 and
                        isinstance(parsed_data[0], dict)):
                    show_progress_step("Flow Decision", f"Component data found in tool: {last_message.name}")
                    return "table_display"
            except:
                pass

        # Check for any message that looks like component JSON
        elif (hasattr(last_message, 'content') and
              isinstance(last_message.content, str) and
              last_message.content.strip().startswith('[') and
              last_message.content.strip().endswith(']')):
            try:
                parsed_data = json.loads(last_message.content)
                if (isinstance(parsed_data, list) and
                        len(parsed_data) > 0 and
                        isinstance(parsed_data[0], dict) and
                        any(key in parsed_data[0] for key in ['name', 'component_name', 'part_number'])):
                    show_progress_step("Flow Decision", "Component data found in message content")
                    return "table_display"
            except:
                pass

    show_progress_step("Flow Decision", "No component data found, ending workflow")
    return "end"


# Add nodes to the graph
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", tool_node)
graph_builder.add_node("table_display", table_display_node)

# Add edges
graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)

# Any time a tool is called, we return to the chatbot to decide the next step
graph_builder.add_edge("tools", "chatbot")

# Add conditional edge from chatbot to either table display or end
graph_builder.add_conditional_edges(
    "chatbot",
    should_display_table,
    {
        "table_display": "table_display",
        "end": END
    }
)

# After table display, end the conversation (user can start new interaction)
graph_builder.add_edge("table_display", END)

# Start with chatbot
graph_builder.add_edge(START, "chatbot")

# Create a MemorySaver object to act as the checkpointer
memory = MemorySaver()

# Compile the graph, passing in the 'memory' object as the checkpointer
graph = graph_builder.compile(checkpointer=memory)

try:
    display(Image(graph.get_graph().draw_mermaid_png()))
except Exception:
    # This requires some extra dependencies and is optional
    pass


def stream_graph_updates(user_input: str):
    """Stream graph updates with enhanced progress display"""
    print(f"\n{'=' * 80}")
    print(f"🚀 PROCESSING REQUEST: {user_input}")
    print(f"{'=' * 80}")

    config = {"configurable": {"thread_id": "1"}}

    try:
        for event in graph.stream({"messages": [HumanMessage(content=user_input)]}, config):
            for value in event.values():
                if "messages" in value and value["messages"]:
                    last_message = value["messages"][-1]
                    if hasattr(last_message, 'content') and last_message.content:
                        # Don't print raw JSON data, let the table display handle it
                        if not (last_message.content.startswith('[') and last_message.content.endswith(']')):
                            print(f"\n💬 Assistant: {last_message.content}")

    except Exception as e:
        show_error_step("Graph Execution", str(e))
        print(f"❌ Error: {e}")

    print(f"\n{'=' * 80}")
    print("✨ REQUEST COMPLETED")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    print("Enhanced LangGraph Agent with Component Table Display")
    print("=" * 60)
    print("Available commands:")
    print("- Analyze schematic: 'analyze schematic at [URL]'")
    print("- Search components: 'search component data for [JSON]'")
    print("- Create BOM: 'create empty bom named [NAME]'")
    print("- Get BOMs: 'get existing boms'")
    print("- Add parts to BOM: 'add parts to bom [BOM_NAME]'")
    print("- Full workflow: 'create bom from schematic at [URL] named [NAME]'")
    print("=" * 60)

    while True:
        try:
            user_input = input("\nUser: ")
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            stream_graph_updates(user_input)
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
            # fallback if input() is not available
            user_input = "What can you help me with regarding schematic analysis and BOM management?"
            print("User: " + user_input)
            stream_graph_updates(user_input)
            break