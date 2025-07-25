from langgraph.prebuilt import create_react_agent
import getpass
from langchain.chains import LLMChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
import os
import requests
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()
# Get the access token from environment variables
ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")
BASE_URL = os.getenv("BASE_URL")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    GOOGLE_API_KEY = getpass.getpass("Enter API key for Google Gemini: ")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
# Initialize the Gemini language model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.7
)
# Define the API client
class APIClient:
    def __init__(self, base_url):
        self.base_url = base_url
    def _get_headers(self):
        return {
            'access_token': ACCESS_TOKEN,
            'content-type': 'application/json'
        }
    def post(self, endpoint, data=None):
        response = requests.post(f"{self.base_url}/{endpoint}", headers=self._get_headers(), json=data)
        response.raise_for_status()  # Raise an error for bad responses
        return response.json()
# Create a BOM API client
class BOMAPI(APIClient):
    def create_project(self, user_id, project_name, comment="", parent_id=0, view_permission=1, private_flag=1):
        endpoint = "project/createProject"
        data = {
            "userId": user_id,
            "projectName": project_name,
            "comment": comment,
            "parentId": parent_id,
            "viewPermission": view_permission,
            "privateFlag": private_flag
        }
        return self.post(endpoint, data=data)
    def create_empty_bom(self, user_id, bom_name, project_id=0, comment="", private_flag=1):
        endpoint = "bom/createEmptyBom"
        data = {
            "userID": user_id,
            "bomName": bom_name,
            "projectId": project_id,
            "comment": comment,
            "privateFlag": private_flag
        }
        return self.post(endpoint, data=data)
    def add_parts_to_bom(self, user_id, bom_id, parts):
        endpoint = "bom/add-parts-to-bom"
        data = {
            "rowDTO": parts,
            "userNavigationDTO": {
                "bomId": bom_id,
                "aclId": 123143,  # Placeholder value; update as needed
                "userId": user_id,
                "userType": 1,  # Assuming userType is 1 for this example
                "pageType": 1000000  # Placeholder value; update as needed
            }
        }
        return self.post(endpoint, data=data)
# Initialize the BOM API client
bom_api = BOMAPI(base_url=BASE_URL)
# Define prompt templates
weather_prompt = PromptTemplate(
    input_variables=["topic"],
    template="Tell me a fun fact about {topic}.",
)
schematic_prompt = PromptTemplate(
    input_variables=["image_url"],
    template="As an expert electrical engineer, please analyze the schematic design at the following URL: {image_url}. List all components in JSON format with component name, part number, manufacturer, and features."
)
# Define tools for the agent
def create_empty_bom_tool(user_id: int, bom_name: str) -> str:
    try:
        response = bom_api.create_empty_bom(user_id=user_id, bom_name=bom_name)
        return f"BOM created successfully: {response}"
    except requests.HTTPError as e:
        return f"HTTP error creating BOM: {str(e)}"
    except Exception as e:
        return f"Error creating BOM: {str(e)}"
def create_project_tool(user_id: int, project_name: str) -> str:
    try:
        response = bom_api.create_project(user_id=user_id, project_name=project_name)
        return f"Project created successfully: {response}"
    except requests.HTTPError as e:
        return f"HTTP error creating project: {str(e)}"
    except Exception as e:
        return f"Error creating project: {str(e)}"
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"
def analyze_schematic(image_url: str) -> str:
    """Analyze a schematic design image from a URL and return component details in JSON format."""
    message = schematic_prompt.format(image_url=image_url)
    try:
        response = llm.invoke(message)
        return response
    except Exception as e:
        return f"Error analyzing schematic: {str(e)}"
def add_parts_to_bom_tool(user_id: int, bom_id: int, parts: list) -> str:
    """Add parts to a BOM."""
    try:
        response = bom_api.add_parts_to_bom(user_id=user_id, bom_id=bom_id, parts=parts)
        return f"Parts added successfully: {response}"
    except requests.HTTPError as e:
        return f"HTTP error adding parts: {str(e)}"
    except Exception as e:
        return f"Error adding parts: {str(e)}"
# Create the agent with both tools
agent = create_react_agent(
    model=llm,
    tools=[get_weather, analyze_schematic, create_empty_bom_tool, create_project_tool, add_parts_to_bom_tool],
    prompt="You are a helpful assistant."
)
# Function to process user input
def process_user_input(user_input: str):
    # Step 1: Ask the LLM for a plan or tasks based on the user input
    plan_prompt = f"Based on the following request, what tasks or plans should be generated? '{user_input}'"
    tasks = llm.invoke(plan_prompt)
    # Step 2: Determine which tool to use based on the tasks
    if "weather" in tasks.lower():
        city = user_input  # Assume user input is a city name
        return get_weather(city)
    elif "analyze" in tasks.lower():
        # Assume user input contains an image URL for analysis
        image_url = user_input.split("analyze ")[-1].strip()
        return analyze_schematic(image_url)
    elif "create project" in tasks.lower():
        # Example: extract user_id and project_name from user_input
        user_id = 1  # Placeholder for actual user ID
        project_name = user_input.split("create project ")[-1].strip()
        return create_project_tool(user_id, project_name)
    elif "create bom" in tasks.lower():
        # Example: extract user_id and bom_name from user_input
        user_id = 1  # Placeholder for actual user ID
        bom_name = user_input.split("create bom ")[-1].strip()
        return create_empty_bom_tool(user_id, bom_name)
    elif "add parts" in tasks.lower():
        # Example: extract user_id, bom_id, and parts from user_input
        user_id = 1  # Placeholder for actual user ID
        bom_id = 760901  # Placeholder for actual BOM ID
        parts = [
            {
                "comId": 42855598,
                "latestMpn": "C0603C474K4RACAUTO",
                "latestSupplier": "KEMET Corporation",
                "manId": "1364",
                "matchedMpn": "C0603C474K4RACAUTO",
                "matchedSupplier": "KEMET Corporation"
            },
            {
                "comId": 409368487,
                "latestMpn": "C0603X103K5RAC3121",
                "latestSupplier": "KEMET Corporation",
                "manId": "1364",
                "matchedMpn": "C0603X103K5RAC3121",
                "matchedSupplier": "KEMET Corporation"
            }
        ]  # This should be dynamically extracted based on user input
        return add_parts_to_bom_tool(user_id, bom_id, parts)
    else:
        return "I'm sorry, I couldn't determine the task from your request."
# Run the agent
while True:
    user_input = input("Enter your request (or 'exit' to quit): ")
    if user_input.lower() == 'exit':
        confirm_exit = input("Are you sure you want to exit? (yes/no): ")
        if confirm_exit.lower() == 'yes':
            break
    result = process_user_input(user_input)
    print(result)