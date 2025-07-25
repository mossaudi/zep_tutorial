from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent, ToolNode, tools_condition
import getpass
from langchain.chains import LLMChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.tools import tool
import os
import requests
from dotenv import load_dotenv
from typing import Annotated
import json

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
SILICON_EXPERT_BASE_URL = "https://api.siliconexpert.com/ProductAPI/search"

if not GOOGLE_API_KEY:
    GOOGLE_API_KEY = getpass.getpass("Enter API key for Google Gemini: ")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# Global session object for Silicon Expert API
silicon_expert_session = requests.Session()
is_authenticated = False


def authenticate_silicon_expert():
    """Authenticate with Silicon Expert API using session cookies."""
    global is_authenticated

    if not SILICON_EXPERT_USERNAME or not SILICON_EXPERT_API_KEY:
        return False

    try:
        response = silicon_expert_session.post(
            f"{SILICON_EXPERT_BASE_URL}/authenticateUser",
            headers={'content-type': 'application/x-www-form-urlencoded'},
            data={
                'login': SILICON_EXPERT_USERNAME,
                'apiKey': SILICON_EXPERT_API_KEY
            }
        )

        if response.status_code == 200:
            # Session automatically stores cookies, including JSESSIONID
            is_authenticated = True
            return True
        else:
            is_authenticated = False
            return False
    except Exception as e:
        print(f"Authentication error: {e}")
        is_authenticated = False
        return False


def _search_component_data_internal(components_json: str) -> str:
    """Internal function to search for additional component data using Silicon Expert Keyword Search API."""
    global is_authenticated

    try:
        # Parse the input JSON
        components = json.loads(components_json)
        if not isinstance(components, list):
            return "Error: Input should be a JSON array of components"

        # Authenticate if needed
        if not is_authenticated:
            if not authenticate_silicon_expert():
                return "Error: Could not authenticate with Silicon Expert API. Please check credentials."

        enhanced_components = []

        for component in components:
            enhanced_component = component.copy()

            # Extract part number and manufacturer for search
            part_number = component.get('part_number', '').strip()
            manufacturer = component.get('manufacturer', '').strip()

            if part_number:
                try:
                    # Prepare search parameters for keyword search API
                    params = {
                        'fmt': 'json',
                        'pageNumber': '1',
                        'pageSize': '5'  # Get top 5 results
                    }

                    # Use part number as primary search term
                    if manufacturer:
                        # Search with both part number and manufacturer
                        params['partNumber'] = f"{part_number} {manufacturer}"
                    else:
                        # Search with part number only
                        params['partNumber'] = part_number

                    # Make the GET request using the session (cookies handled automatically)
                    response = silicon_expert_session.get(
                        f"{SILICON_EXPERT_BASE_URL}/partsearch",
                        params=params
                    )

                    if response.status_code == 200:
                        api_data = response.json()

                        # Check for authentication error first (Code 39)
                        if (api_data and
                                api_data.get('Status', {}).get('Code') == '39' and
                                api_data.get('Status', {}).get('Message') == 'You are not authenticated'):

                            # Try to re-authenticate
                            is_authenticated = False
                            if authenticate_silicon_expert():
                                # Retry the request with new session
                                retry_response = silicon_expert_session.get(
                                    f"{SILICON_EXPERT_BASE_URL}/partsearch",
                                    params=params
                                )
                                if retry_response.status_code == 200:
                                    api_data = retry_response.json()
                                else:
                                    enhanced_component[
                                        'se_search_result'] = f"Retry failed: HTTP {retry_response.status_code}"
                                    continue
                            else:
                                enhanced_component['se_search_result'] = "Re-authentication failed"
                                continue

                        # Check if the response has the expected structure for successful search
                        if (api_data and
                                api_data.get('Status', {}).get('Success') == 'true' and
                                'Result' in api_data and
                                isinstance(api_data['Result'], list) and
                                len(api_data['Result']) > 0):

                            # Get the first (best match) result
                            first_result = api_data['Result'][0]

                            # Extract key data from the first result
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
                                'se_aml_status': first_result.get('AMLStatus')
                            })

                            # Add information about total results found
                            enhanced_component['se_total_items'] = api_data.get('TotalItems', 'Unknown')

                            # Add all results if there are multiple matches
                            if len(api_data['Result']) > 1:
                                enhanced_component['se_all_matches'] = []
                                for result in api_data['Result'][:5]:  # Limit to first 5
                                    match_info = {
                                        'com_id': result.get('ComID'),
                                        'part_number': result.get('PartNumber'),
                                        'manufacturer': result.get('Manufacturer'),
                                        'match_rating': result.get('MatchRating'),
                                        'lifecycle': result.get('Lifecycle')
                                    }
                                    enhanced_component['se_all_matches'].append(match_info)

                        else:
                            # Handle API error response or no results
                            error_msg = "No matching parts found"
                            if api_data and 'Status' in api_data:
                                status_code = api_data['Status'].get('Code', '')
                                status_msg = api_data['Status'].get('Message', '')
                                success = api_data['Status'].get('Success', '')

                                if success == 'false':
                                    error_msg = f"API Error (Code {status_code}): {status_msg}"
                                elif status_msg and status_msg != "Successful Operation":
                                    error_msg = f"API Message: {status_msg}"

                            enhanced_component['se_search_result'] = error_msg

                    else:
                        enhanced_component[
                            'se_search_result'] = f"HTTP Error: {response.status_code} - {response.text[:100]}"

                except Exception as api_error:
                    enhanced_component['se_search_result'] = f"Search error: {str(api_error)}"
            else:
                enhanced_component['se_search_result'] = "No part number provided"

            enhanced_components.append(enhanced_component)

        return json.dumps(enhanced_components, indent=2)

    except json.JSONDecodeError:
        return "Error: Invalid JSON format in input"
    except Exception as e:
        return f"Error searching component data: {str(e)}"


@tool
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

class State(TypedDict):
    messages: Annotated[list, add_messages]


# Initialize the Gemini language model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.7
)


# Define the tool using the @tool decorator
@tool
def analyze_schematic(image_url: str) -> str:
    """Analyze a schematic design image from a URL and return enhanced component details with Silicon Expert data.

    This tool automatically:
    1. Analyzes the schematic to identify components
    2. Searches Silicon Expert database for additional component information
    3. Returns comprehensive component data including lifecycle, availability, and technical specs

    Args:
        image_url: The URL of the schematic image to analyze

    Returns:
        A JSON formatted string with enhanced component details including Silicon Expert data
    """
    schematic_prompt = PromptTemplate(
        input_variables=["image_url"],
        template="As an expert electrical engineer, please analyze the schematic design at the following URL: {image_url}. List all components in JSON format with component name, part number, manufacturer, and features. Return ONLY valid JSON array format."
    )

    message = schematic_prompt.format(image_url=image_url)
    try:
        # Step 1: Analyze the schematic
        response = llm.invoke([{"role": "user", "content": message}])
        raw_response = response.content

        # Step 2: Clean up the JSON response
        cleaned_json = _clean_json_response(raw_response)

        # Step 3: Automatically search for component data using Silicon Expert
        enhanced_data = _search_component_data_internal(cleaned_json)

        # Return the enhanced data with both schematic analysis and Silicon Expert information
        return f"Schematic analysis complete with enhanced component data:\n\n{enhanced_data}"

    except Exception as e:
        return f"Error analyzing schematic: {str(e)}"


@tool
def search_component_data(components_json: str) -> str:
    """Search for additional component data using Silicon Expert Keyword Search API.

    Args:
        components_json: JSON string containing component data from schematic analysis

    Returns:
        Enhanced JSON string with additional component information from Silicon Expert API
    """
    return _search_component_data_internal(components_json)


# Create the tools list
tools = [analyze_schematic, search_component_data]

# Bind tools to the LLM
llm_with_tools = llm.bind_tools(tools)

# Create ToolNode with the tools
tool_node = ToolNode(tools)

# Create the graph builder
graph_builder = StateGraph(State)


def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


# Add nodes to the graph
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", tool_node)

# Add edges
graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
# Any time a tool is called, we return to the chatbot to decide the next step
graph_builder.add_edge("tools", "chatbot")
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
    config = {"configurable": {"thread_id": "1"}}
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}, config):
        for value in event.values():
            if "messages" in value:
                print("Assistant:", value["messages"][-1].content)


if __name__ == "__main__":
    while True:
        try:
            user_input = input("User: ")
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
            user_input = "What do you know about LangGraph?"
            print("User: " + user_input)
            stream_graph_updates(user_input)
            break