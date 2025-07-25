import getpass

from langchain.chains import LLMChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv # Import load_dotenv

# Load environment variables from .env file
load_dotenv()

if not os.environ.get("GOOGLE_API_KEY"):
  os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter API key for Google Gemini: ")

# Initialize the Gemini language model
# The GOOGLE_API_KEY will now be loaded from your .env file
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"), # Get the key from environment variables
    temperature=0.7
)

# Define a prompt template
prompt = PromptTemplate(
    input_variables=["topic"],
    template="Tell me a fun fact about {topic}.",
)

# Initialize the LLMChain
chain = prompt | llm

# Run the chain with some input
response = chain.invoke({"topic":"Spring boot"})

# Print the response
print(response)