# main_simplified.py
"""Enhanced LangGraph agent with a refined, modular architecture."""

import getpass
import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from config import AppConfig
from container import Container
from exceptions import ConfigurationError
from models import SearchResult
from services.formatter import ComponentTableFormatter
from tools import get_tools


class IntelligentBOMAgent:
    """Enhanced LangGraph agent with a modular, service-oriented architecture."""

    def __init__(self, config: AppConfig):
        self.config = config
        self._validate_config()
        self._initialize_components()
        self._create_react_agent()
        print("✅ Intelligent BOM Agent is ready.")

    def _validate_config(self):
        config_errors = self.config.validate()
        if config_errors:
            error_msg = "Configuration errors:\n" + "\n".join(f"- {error}" for error in config_errors)
            raise ConfigurationError(error_msg)

    def _initialize_components(self):
        """Initialize LLM, container, tools, and formatter."""
        google_api_key = self.config.google_api_key or getpass.getpass("Enter API key for Google Gemini: ")

        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-lite",
            google_api_key=google_api_key,
            temperature=0.1,
            max_output_tokens=30000,
            # enable JSON mode to guarantee valid JSON output
            model_kwargs={"response_format": {"type": "json_object"}}
        )
        print("✅ Gemini model configured.")

        # Centralized dependency container
        self.container = Container(self.config, self.llm)
        self.tools = get_tools(self.container)
        self.formatter = ComponentTableFormatter()
        print(f"✅ Initialized {len(self.tools)} tools.")

    def _create_react_agent(self):
        """Create ReAct agent with an enhanced system prompt."""
        system_prompt = """You are an expert Bill of Materials (BOM) management assistant. Your goal is to help users analyze electronic schematics, find component data, and manage BOMs efficiently.

You will use the available tools to answer user requests. When a user provides a schematic, your primary tool is `analyze_schematic`, which automatically extracts all components and enriches them with data from the Silicon Expert database.

Your final answer should be the direct result from the tool. If the tool returns component data, present it clearly.

## Available Tools:
- `analyze_schematic`: Analyzes a schematic from a URL and returns a full component list with lifecycle and compliance data.
- `search_component_data`: Searches for specific components provided as a JSON list.
- `create_empty_bom`: Creates a new, empty BOM.
- `add_parts_to_bom`: Adds a list of components to an existing BOM.
- `get_boms`: Lists all available BOMs."""

        self.agent = create_react_agent(self.llm, self.tools, checkpointer=MemorySaver())
        self.system_prompt = system_prompt
        self.config_dict = {"configurable": {"thread_id": "1"}}
        print("✅ ReAct agent created.")

    def _format_final_response(self, content: Any) -> str:
        """Intelligently format the agent's final output."""
        if isinstance(content, str):
            try:
                # Attempt to parse as JSON for potential structured data
                data = json.loads(content)
                # If it's a dict and looks like our SearchResult, format it
                if isinstance(data, dict) and 'components' in data and 'success_rate' in data:
                    # Re-create the dataclass object for formatting
                    search_result = SearchResult(**data)
                    return self.formatter.format_search_result(search_result)
                # Otherwise, just return the pretty-printed JSON
                return json.dumps(data, indent=2)
            except (json.JSONDecodeError, TypeError):
                # Not JSON, just return the string content
                return content
        elif isinstance(content, dict):
            if 'components' in content and 'success_rate' in content:
                search_result = SearchResult(**content)
                return self.formatter.format_search_result(search_result)
            return json.dumps(content, indent=2)

        return str(content)

    def process_request(self, user_input: str):
        """Process a user request using the ReAct agent."""
        print(f"\n{'=' * 80}\n🤖 PROCESSING REQUEST: {user_input}\n{'=' * 80}")

        try:
            messages = [SystemMessage(content=self.system_prompt), HumanMessage(content=user_input)]
            result = self.agent.invoke({"messages": messages}, self.config_dict)

            if result and "messages" in result:
                final_message = result["messages"][-1]
                formatted_output = self._format_final_response(final_message.content)
                print(f"\n🤖 Assistant:\n{formatted_output}")

            print(f"\n{'=' * 80}\n✨ REQUEST COMPLETED\n{'=' * 80}")

        except Exception as e:
            print(f"❌ Error processing request: {e}")
            import traceback
            traceback.print_exc()

    def run_interactive(self):
        """Run the agent in an interactive command-line mode."""
        print("\n🤖 Welcome to the Intelligent BOM Agent!")
        print("   Example: 'analyze schematic at [URL]'")
        print("   Type 'quit' or 'exit' to end.")
        print("-" * 80)

        while True:
            try:
                user_input = input("\n👤 User: ")
                if user_input.lower() in ["quit", "exit", "q"]:
                    print("🤖 Goodbye!")
                    break
                self.process_request(user_input)
            except KeyboardInterrupt:
                print("\n🤖 Goodbye!")
                break
            except Exception as e:
                print(f"❌ An unexpected error occurred: {e}")


def main():
    """Main entry point."""
    try:
        config = AppConfig.from_env()
        agent = IntelligentBOMAgent(config)
        agent.run_interactive()
    except (ConfigurationError, Exception) as e:
        print(f"❌ Fatal Error: {e}")
        return 1
    return 0


if __name__ == "__main__":
    exit(main())