import getpass
from typing import Annotated, Sequence
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from langchain.chains import LLMChain
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv # Import load_dotenv

# Load environment variables from .env file
load_dotenv()

if not os.environ.get("GOOGLE_API_KEY"):
  os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter API key for Google Gemini: ")

# --- 1. Define the State ---
# The state is a dictionary that will be passed between nodes.
# `add_messages` is a helper that automatically appends new messages
# to the `messages` list.
class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# --- 2. Initialize Model and Graph ---
# Set your OpenAI API key
# os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY_HERE"

# It's recommended to use ChatOpenAI for conversational tasks.
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"), # Get the key from environment variables
    temperature=0.7
)

# Create the state graph
graph_builder = StateGraph(State)

# --- 3. Define the Graph Nodes ---
# A node is a function that performs an action.
# It takes the current state as input and returns a dictionary
# to update the state.
def chatbot(state: State):
    """
    This is the primary node that calls the LLM.
    It takes the entire message history and returns the AI's response.
    """
    print("---LLM CALLED---")
    # The `invoke` method automatically uses the message history
    response = llm.invoke(state["messages"])
    # We return the response as an AIMessage to be added to the state
    return {"messages": [response]}

# --- 4. Build the Graph ---
# Add the 'chatbot' node to the graph.
graph_builder.add_node("chatbot", chatbot)

# Set the entry point for the graph. The first node to be called is 'chatbot'.
graph_builder.set_entry_point("chatbot")

# Set the finish point. After 'chatbot' is called, the graph execution will end.
graph_builder.add_edge("chatbot", END)

# Compile the graph into a runnable object.
graph = graph_builder.compile()


# --- 5. Run the Graph ---
# This loop handles the conversation with the user.
while True:
    user_input = input("You: ")
    if user_input.lower() in ["quit", "exit", "q"]:
        print("Goodbye!")
        break

    # The `stream` method executes the graph and returns an iterator of events.
    # We pass the user's message in the correct state format.
    for event in graph.stream(input = {"messages": [HumanMessage(content=user_input)]}):
        # The `event` dictionary contains the output of the node that just ran.
        for value in event.values():
            # We print the last message, which is the AI's response.
            print("Bot:", value["messages"][-1].content)