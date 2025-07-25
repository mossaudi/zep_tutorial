"""
Enhanced Schematic Analysis Chatbot
A LangGraph-based chatbot for analyzing schematics and searching component data.
"""

import os
import json
import getpass
import logging
from typing import List, Dict, Any, Optional, Annotated
from typing_extensions import TypedDict

import requests
from dotenv import load_dotenv

from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langchain.prompts import PromptTemplate
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

try:
    from IPython.display import Image, display
    IPYTHON_AVAILABLE = True
except ImportError:
    IPYTHON_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ComponentData:
    """Data class for component information."""

    def __init__(self, name: str = "", part_number: str = "",
                 manufacturer: str = "", features: str = ""):
        self.name = name
        self.part_number = part_number
        self.manufacturer = manufacturer
        self.features = features

    def to_dict(self) -> Dict[str, str]:
        return {
            'name': self.name,
            'part_number': self.part_number,
            'manufacturer': self.manufacturer,
            'features': self.features
        }


class SiliconExpertAPI:
    """Handler for Silicon Expert API interactions."""

    def __init__(self, username: str, api_key: str, base_url: str):
        self.username = username
        self.api_key = api_key
        self.base_url = base_url
        self.session = requests.Session()
        self.is_authenticated = False

        # Set reasonable timeouts
        self.session.timeout = 300000

    def authenticate(self) -> bool:
        """Authenticate with Silicon Expert API."""
        if not self.username or not self.api_key:
            logger.error("Missing Silicon Expert credentials")
            return False

        try:
            response = self.session.post(
                f"{self.base_url}/authenticateUser",
                headers={'content-type': 'application/x-www-form-urlencoded'},
                data={'login': self.username, 'apiKey': self.api_key},
                timeout=100000
            )

            if response.status_code == 200:
                self.is_authenticated = True
                logger.info("Successfully authenticated with Silicon Expert API")
                return True
            else:
                logger.error(f"Authentication failed: HTTP {response.status_code}")
                return False

        except requests.exceptions.RequestException as e:
            logger.error(f"Authentication error: {e}")
            return False

    def search_component(self, part_number: str, manufacturer: str = "") -> Dict[str, Any]:
        """Search for component data."""
        if not self.is_authenticated and not self.authenticate():
            return {"error": "Authentication failed"}

        try:
            params = {
                'fmt': 'json',
                'pageNumber': '1',
                'pageSize': '5',
                'partNumber': f"{part_number} {manufacturer}".strip()
            }

            response = self.session.get(
                f"{self.base_url}/partsearch",
                params=params,
                timeout=150000
            )

            if response.status_code != 200:
                return {"error": f"HTTP {response.status_code}: {response.text[:100]}"}

            api_data = response.json()

            # Handle authentication errors
            if (api_data.get('Status', {}).get('Code') == '39' and
                api_data.get('Status', {}).get('Message') == 'You are not authenticated'):

                logger.warning("Session expired, re-authenticating...")
                self.is_authenticated = False
                if self.authenticate():
                    return self.search_component(part_number, manufacturer)
                else:
                    return {"error": "Re-authentication failed"}

            return self._process_search_results(api_data)

        except requests.exceptions.RequestException as e:
            logger.error(f"Search request error: {e}")
            return {"error": f"Request failed: {str(e)}"}
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            return {"error": "Invalid JSON response from API"}
        except Exception as e:
            logger.error(f"Unexpected error in component search: {e}")
            return {"error": f"Unexpected error: {str(e)}"}

    def _process_search_results(self, api_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process and structure API search results."""
        if not (api_data and
                api_data.get('Status', {}).get('Success') == 'true' and
                'Result' in api_data and
                isinstance(api_data['Result'], list) and
                len(api_data['Result']) > 0):

            # Extract error message if available
            error_msg = "No matching parts found"
            if api_data and 'Status' in api_data:
                status = api_data['Status']
                if status.get('Success') == 'false':
                    code = status.get('Code', '')
                    message = status.get('Message', '')
                    error_msg = f"API Error (Code {code}): {message}"

            return {"error": error_msg}

        # Process successful results
        first_result = api_data['Result'][0]

        result = {
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
            'se_total_items': api_data.get('TotalItems', 'Unknown')
        }

        # Add additional matches if available
        if len(api_data['Result']) > 1:
            result['se_all_matches'] = [
                {
                    'com_id': r.get('ComID'),
                    'part_number': r.get('PartNumber'),
                    'manufacturer': r.get('Manufacturer'),
                    'match_rating': r.get('MatchRating'),
                    'lifecycle': r.get('Lifecycle')
                }
                for r in api_data['Result'][:5]
            ]

        return result


class Config:
    """Configuration manager for environment variables and API keys."""

    def __init__(self):
        load_dotenv()
        self._validate_and_load_config()

    def _validate_and_load_config(self):
        """Load and validate configuration from environment variables."""
        self.access_token = os.getenv("ACCESS_TOKEN")
        self.base_url = os.getenv("BASE_URL")
        self.google_api_key = self._get_google_api_key()

        # Silicon Expert configuration
        self.silicon_expert_username = os.getenv("SILICON_EXPERT_USERNAME")
        self.silicon_expert_api_key = os.getenv("SILICON_EXPERT_API_KEY")
        self.silicon_expert_base_url = os.getenv(
            "SILICON_EXPERT_BASE_URL",
            "https://api.siliconexpert.com/ProductAPI/search"
        )

        # Validate required configuration
        if not self.google_api_key:
            raise ValueError("Google API key is required")

        if not self.silicon_expert_username or not self.silicon_expert_api_key:
            logger.warning("Silicon Expert credentials not found. Component search will be limited.")

    def _get_google_api_key(self) -> str:
        """Get Google API key from environment or prompt user."""
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            try:
                api_key = getpass.getpass("Enter API key for Google Gemini: ")
                os.environ["GOOGLE_API_KEY"] = api_key
            except (KeyboardInterrupt, EOFError):
                logger.error("API key input cancelled")
                raise ValueError("Google API key is required")
        return api_key


class State(TypedDict):
    """State definition for the chatbot graph."""
    messages: Annotated[list, add_messages]


# Global variables to hold the instances for tools
_chatbot_instance = None

@tool
def analyze_schematic(image_url: str) -> str:
    """Analyze a schematic design image and return component details in JSON format.

    Args:
        image_url: The URL of the schematic image to analyze

    Returns:
        JSON formatted string with component details
    """
    if not image_url or not isinstance(image_url, str):
        return json.dumps({"error": "Invalid image URL provided"})

    if not _chatbot_instance:
        return json.dumps({"error": "Chatbot not initialized"})

    try:
        schematic_prompt = PromptTemplate(
            input_variables=["image_url"],
            template=(
                "You are an expert electrical engineer, analyze the schematic design at: {image_url}. "
                "List all identifiable components in JSON array format with: "
                "component name, part number (if visible), manufacturer (if visible), and key features. "
                "Return ONLY valid JSON array format. If no components are found, return an empty array."
            )
        )

        message = schematic_prompt.format(image_url=image_url)
        response = _chatbot_instance.llm.invoke([{"role": "user", "content": message}])

        # Validate JSON response
        try:
            json.loads(response.content)
            return response.content
        except json.JSONDecodeError:
            logger.warning("LLM returned invalid JSON, wrapping in error response")
            return json.dumps({"error": "Failed to parse schematic - invalid response format"})

    except Exception as e:
        logger.error(f"Error analyzing schematic: {e}")
        return json.dumps({"error": f"Analysis failed: {str(e)}"})


@tool
def search_component_data(components_json: str) -> str:
    """Search for additional component data using Silicon Expert API.

    Args:
        components_json: JSON string containing component data

    Returns:
        Enhanced JSON string with additional component information
    """
    if not _chatbot_instance:
        return json.dumps({"error": "Chatbot not initialized"})

    if not _chatbot_instance.silicon_expert_api:
        return json.dumps({"error": "Silicon Expert API not configured"})

    try:
        components = json.loads(components_json)
        if not isinstance(components, list):
            return json.dumps({"error": "Input should be a JSON array of components"})

        enhanced_components = []

        for component in components:
            if not isinstance(component, dict):
                logger.warning(f"Skipping invalid component: {component}")
                continue

            enhanced_component = component.copy()
            part_number = component.get('part_number', '').strip()
            manufacturer = component.get('manufacturer', '').strip()

            if part_number:
                search_result = _chatbot_instance.silicon_expert_api.search_component(
                    part_number, manufacturer
                )

                if "error" in search_result:
                    enhanced_component['se_search_result'] = search_result["error"]
                else:
                    enhanced_component.update(search_result)
            else:
                enhanced_component['se_search_result'] = "No part number provided"

            enhanced_components.append(enhanced_component)

        return json.dumps(enhanced_components, indent=2)

    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {e}")
        return json.dumps({"error": "Invalid JSON format in input"})
    except Exception as e:
        logger.error(f"Error searching component data: {e}")
        return json.dumps({"error": f"Search failed: {str(e)}"})


class SchematicAnalysisChatbot:
    """Main chatbot class for schematic analysis."""

    def __init__(self):
        global _chatbot_instance

        self.config = Config()
        self.silicon_expert_api = None

        if (self.config.silicon_expert_username and
            self.config.silicon_expert_api_key):
            self.silicon_expert_api = SiliconExpertAPI(
                self.config.silicon_expert_username,
                self.config.silicon_expert_api_key,
                self.config.silicon_expert_base_url
            )

        self.llm = self._initialize_llm()

        # Set global instance for tools to access
        _chatbot_instance = self

        self.graph = self._build_graph()

    def _initialize_llm(self) -> ChatGoogleGenerativeAI:
        """Initialize the Google Gemini language model."""
        try:
            return ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                google_api_key=self.config.google_api_key,
                temperature=0.7
            )
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            raise

    def _build_graph(self):
        """Build and compile the LangGraph workflow."""
        try:
            # Bind tools to LLM - use the standalone functions
            tools = [analyze_schematic, search_component_data]
            llm_with_tools = self.llm.bind_tools(tools)
            tool_node = ToolNode(tools)

            # Define chatbot node
            def chatbot_node(state: State):
                try:
                    return {"messages": [llm_with_tools.invoke(state["messages"])]}
                except Exception as e:
                    logger.error(f"Chatbot node error: {e}")
                    error_message = f"I encountered an error: {str(e)}. Please try again."
                    return {"messages": [{"role": "assistant", "content": error_message}]}

            # Build graph
            graph_builder = StateGraph(State)
            graph_builder.add_node("chatbot", chatbot_node)
            graph_builder.add_node("tools", tool_node)

            # Add edges
            graph_builder.add_conditional_edges("chatbot", tools_condition)
            graph_builder.add_edge("tools", "chatbot")
            graph_builder.add_edge(START, "chatbot")

            # Compile with memory
            memory = MemorySaver()
            graph = graph_builder.compile(checkpointer=memory)

            # Display graph if possible
            if IPYTHON_AVAILABLE:
                try:
                    display(Image(graph.get_graph().draw_mermaid_png()))
                except Exception as e:
                    logger.info(f"Could not display graph: {e}")

            return graph

        except Exception as e:
            logger.error(f"Failed to build graph: {e}")
            raise

    def stream_response(self, user_input: str, thread_id: str = "1") -> None:
        """Stream chatbot responses for user input."""
        config = {"configurable": {"thread_id": thread_id}}

        try:
            for event in self.graph.stream(
                {"messages": [{"role": "user", "content": user_input}]},
                config
            ):
                for value in event.values():
                    if "messages" in value and value["messages"]:
                        message = value["messages"][-1]
                        if hasattr(message, 'content'):
                            print("Assistant:", message.content)
                        else:
                            print("Assistant:", str(message))
        except Exception as e:
            logger.error(f"Error streaming response: {e}")
            print(f"Assistant: I encountered an error: {str(e)}. Please try again.")

    def run_interactive(self) -> None:
        """Run the chatbot in interactive mode."""
        print("🤖 Schematic Analysis Chatbot")
        print("Type 'quit', 'exit', or 'q' to stop")
        print("-" * 50)

        while True:
            try:
                user_input = input("\nUser: ").strip()

                if not user_input:
                    continue

                if user_input.lower() in ["quit", "exit", "q"]:
                    print("\n👋 Goodbye!")
                    break

                self.stream_response(user_input)

            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except EOFError:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                logger.error(f"Interactive mode error: {e}")
                print(f"Error: {e}")


def main():
    """Main entry point."""
    try:
        chatbot = SchematicAnalysisChatbot()
        chatbot.run_interactive()
    except Exception as e:
        logger.error(f"Failed to start chatbot: {e}")
        print(f"Failed to start chatbot: {e}")


if __name__ == "__main__":
    main()