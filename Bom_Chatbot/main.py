# main.py
"""Refactored LangGraph agent main application."""

import getpass
import time
from typing import Annotated
from typing_extensions import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent, ToolNode, tools_condition
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from config import AppConfig
from exceptions import ConfigurationError, AgentError
from tools import initialize_tools, get_tools
from services.progress import get_progress_tracker, ProgressTracker, ConsoleProgressObserver


class State(TypedDict):
    """Enhanced state for component analysis tracking."""
    messages: Annotated[list, add_messages]
    component_data: str
    needs_table_display: bool
    current_step: str


class LangGraphAgent:
    """Main LangGraph agent application."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.progress = get_progress_tracker()

        # Validate configuration
        config_errors = config.validate()
        if config_errors:
            error_msg = "Configuration errors:\n" + "\n".join(f"- {error}" for error in config_errors)
            raise ConfigurationError(error_msg)

        # Initialize LLM
        self._initialize_llm()

        # Initialize tools
        initialize_tools(config, self.llm)
        self.tools = get_tools()

        # Create graph
        self._create_graph()

        self.progress.success("Agent Initialization", "LangGraph agent ready")

    def _initialize_llm(self) -> None:
        """Initialize the Gemini language model."""
        self.progress.info("LLM Initialization", "Setting up Gemini model...")

        google_api_key = self.config.google_api_key
        if not google_api_key:
            google_api_key = getpass.getpass("Enter API key for Google Gemini: ")

        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=google_api_key,
            temperature=0.7
        )

        self.progress.success("LLM Initialization", "Gemini model configured")

    def _create_graph(self) -> None:
        """Create and compile the LangGraph workflow."""
        self.progress.info("Graph Creation", "Building LangGraph workflow...")

        # Bind tools to LLM
        llm_with_tools = self.llm.bind_tools(self.tools)

        # Create tool node
        tool_node = ToolNode(self.tools)

        # Create graph builder
        graph_builder = StateGraph(State)

        # Add nodes
        graph_builder.add_node("chatbot", self._chatbot_node)
        graph_builder.add_node("tools", tool_node)
        graph_builder.add_node("table_display", self._table_display_node)

        # Add edges
        graph_builder.add_conditional_edges("chatbot", tools_condition)
        graph_builder.add_edge("tools", "chatbot")
        graph_builder.add_conditional_edges(
            "chatbot",
            self._should_display_table,
            {
                "table_display": "table_display",
                "end": END
            }
        )
        graph_builder.add_edge("table_display", END)
        graph_builder.add_edge(START, "chatbot")

        # Create memory and compile
        memory = MemorySaver()
        self.graph = graph_builder.compile(checkpointer=memory)

        self.progress.success("Graph Creation", "LangGraph workflow compiled")

    def _chatbot_node(self, state: State):
        """Main chatbot node that processes user input."""
        llm_with_tools = self.llm.bind_tools(self.tools)
        response = llm_with_tools.invoke(state["messages"])

        # Check if any tool was called
        if hasattr(response, 'tool_calls') and response.tool_calls:
            return {"messages": [response]}

        # Check if response contains component analysis results
        if self._contains_component_data(state):
            last_message = state["messages"][-1]
            return {
                "messages": [response],
                "component_data": last_message.content,
                "needs_table_display": True
            }

        return {
            "messages": [response],
            "component_data": "",
            "needs_table_display": False
        }

    def _table_display_node(self, state: State):
        """Node that handles component analysis table display."""
        # The table display is now handled within the tools themselves
        # This node just resets the display flag
        return {
            "messages": [],
            "needs_table_display": False
        }

    def _should_display_table(self, state: State) -> str:
        """Determine if table display is needed."""
        if state.get("needs_table_display", False):
            self.progress.info("Flow Decision", "Table display requested")
            return "table_display"

        # Check for component data in tool responses
        if self._contains_component_data(state):
            self.progress.info("Flow Decision", "Component data found in messages")
            return "table_display"

        return "end"

    def _contains_component_data(self, state: State) -> bool:
        """Check if state contains component analysis data."""
        if not state.get("messages"):
            return False

        last_message = state["messages"][-1]

        # Check for tool messages with component data
        if (hasattr(last_message, 'content') and
                hasattr(last_message, 'name') and
                last_message.name in ['analyze_schematic', 'search_component_data']):
            return True

        # Check for component data markers
        if (hasattr(last_message, 'content') and
                isinstance(last_message.content, str) and
                "COMPONENT_SEARCH_COMPLETE:" in last_message.content):
            return True

        return False

    def process_request(self, user_input: str) -> None:
        """Process a user request through the graph."""
        self.progress.info("Request Processing", f"Processing: {user_input}")

        config = {"configurable": {"thread_id": "1"}}

        try:
            for event in self.graph.stream(
                    {"messages": [HumanMessage(content=user_input)]},
                    config
            ):
                for value in event.values():
                    if "messages" in value and value["messages"]:
                        last_message = value["messages"][-1]
                        if (hasattr(last_message, 'content') and
                                last_message.content and
                                not self._is_raw_data(last_message.content)):
                            print(f"\n💬 Assistant: {last_message.content}")

        except Exception as e:
            self.progress.error("Request Processing", str(e))
            print(f"❌ Error processing request: {e}")

    def _is_raw_data(self, content: str) -> bool:
        """Check if content is raw data that shouldn't be printed directly."""
        return (
                content.startswith('[') and content.endswith(']') or
                content.startswith('COMPONENT_SEARCH_COMPLETE:')
        )

    def run_interactive(self) -> None:
        """Run the agent in interactive mode."""
        print("Enhanced LangGraph Agent with Component Analysis")
        print("=" * 60)
        print("Available commands:")
        print("- Analyze schematic: 'analyze schematic at [URL]'")
        print("- Search components: 'search component data for [JSON]'")
        print("- Parametric search: 'parametric search [PRODUCT_LINE] with filters [JSON]'")
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

                print(f"\n{'=' * 80}")
                print(f"🚀 PROCESSING REQUEST: {user_input}")
                print(f"{'=' * 80}")

                self.process_request(user_input)

                print(f"\n{'=' * 80}")
                print("✨ REQUEST COMPLETED")
                print(f"{'=' * 80}")

            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                self.progress.error("Interactive Mode", str(e))
                print(f"Error: {e}")


def main():
    """Main entry point."""
    try:
        # Load configuration
        config = AppConfig.from_env()

        # Create and run agent
        agent = LangGraphAgent(config)
        agent.run_interactive()

    except ConfigurationError as e:
        print(f"Configuration Error: {e}")
        print("Please check your environment variables and try again.")
        return 1

    except AgentError as e:
        print(f"Agent Error: {e}")
        return 1

    except Exception as e:
        print(f"Unexpected Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())