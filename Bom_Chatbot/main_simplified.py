# main_simplified.py
"""Simplified LangGraph agent using ReAct pattern with enhanced tool descriptions."""

import getpass
from typing import Annotated, List # Import List for the new type hint

from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langgraph.graph import MessagesState
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage # Import BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from config import AppConfig
from exceptions import ConfigurationError, AgentError
from tools import initialize_tools, get_tools


# Use MessagesState for LangGraph compatibility
class State(MessagesState):
    """State class inheriting from MessagesState for LangGraph compatibility."""
    # Add 'remaining_steps' to the state schema
    remaining_steps: List[str] # Or appropriate type depending on how it's used
    messages: List[BaseMessage] # Ensure messages is also explicitly defined as it's part of MessagesState


class SimplifiedLangGraphAgent:
    """Simplified LangGraph agent using ReAct pattern with enhanced tool descriptions."""

    def __init__(self, config: AppConfig):
        self.config = config

        # Validate configuration
        config_errors = config.validate()
        if config_errors:
            error_msg = "Configuration errors:\n" + "\n".join(f"- {error}" for error in config_errors)
            raise ConfigurationError(error_msg)

        # Initialize components
        self._initialize_llm()
        self._initialize_tools()
        self._create_react_agent()

        print("✅ Simplified LangGraph agent ready with parametric search optimization")

    def _initialize_llm(self) -> None:
        """Initialize the Gemini language model."""
        google_api_key = self.config.google_api_key
        if not google_api_key:
            google_api_key = getpass.getpass("Enter API key for Google Gemini: ")

        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=google_api_key,
            temperature=0.7
        )
        print("✅ Gemini model configured")

    def _initialize_tools(self) -> None:
        """Initialize tools."""
        initialize_tools(self.config, self.llm)
        self.tools = get_tools()
        print(f"✅ Initialized {len(self.tools)} tools with enhanced descriptions")

    def _create_react_agent(self) -> None:
        """Create ReAct agent with enhanced system prompt."""
        memory = MemorySaver()

        # Enhanced system prompt for intelligent behavior
        system_prompt = """You are an expert BOM (Bill of Materials) management assistant with advanced parametric search capabilities.

🎯 YOUR EXPERTISE:
- Schematic analysis and component identification
- Parametric component searching with technical specifications
- BOM creation and management workflows
- Silicon Expert database integration

🚀 PARAMETRIC SEARCH OPTIMIZATION:
- ALWAYS prioritize 'parametric_search' when you see structured component data with technical specifications
- Look for JSON with "plName" and "selectedFilters" - this means parametric search is optimal
- Component categories like MOSFETs, Microcontrollers, etc. are perfect for parametric search
- Parametric search gives more precise results than general component search

🔧 INTELLIGENT TOOL SELECTION:
1. For schematic images → use 'analyze_schematic' (often leads to parametric search)
2. For technical specifications → use 'parametric_search'
3. For complete automation → use 'create_bom_from_schematic'
4. For basic searches → use 'search_component_data'
5. For BOM management → use create/get/add BOM tools

💡 CONVERSATION FLOW:
- Analyze tool outputs to determine next best steps
- Suggest follow-up actions based on results
- Explain why certain tools are recommended
- Provide examples of commands when helpful
- Always be helpful and guide users through workflows

🎨 COMMUNICATION STYLE:
- Clear, professional, and helpful
- Explain technical concepts when needed
- Provide specific next-step recommendations
- Use appropriate emojis for clarity (🎯🚀🔧💡)

Remember: You have access to powerful tools - use them intelligently based on the context and user needs!"""

        # Create agent without state_modifier (deprecated parameter)
        # The system prompt will be included in the first message instead
        self.agent = create_react_agent(
            self.llm,
            self.tools,
            state_schema=State,
            checkpointer=memory
        )

        # Store system prompt for use in conversations
        self.system_prompt = system_prompt

        # Configuration for conversations
        self.config_dict = {"configurable": {"thread_id": "1"}}
        print("✅ ReAct agent created with enhanced system prompt")

    def process_request(self, user_input: str) -> None:
        """Process user request using ReAct pattern."""
        print(f"\n{'=' * 80}")
        print(f"🚀 PROCESSING REQUEST: {user_input}")
        print(f"{'=' * 80}")

        try:
            # Include system prompt in the message flow
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=user_input)
            ]

            # Simple ReAct invocation - handles everything automatically
            result = self.agent.invoke(
                {"messages": messages},
                self.config_dict
            )

            # Display the final result
            if result and "messages" in result:
                final_message = result["messages"][-1]
                if hasattr(final_message, 'content') and final_message.content:
                    if not self._is_raw_data(final_message.content):
                        print(f"\n💬 Assistant: {final_message.content}")

            print(f"\n{'=' * 80}")
            print("✨ REQUEST COMPLETED")
            print(f"{'=' * 80}")

        except Exception as e:
            print(f"❌ Error processing request: {e}")

    def _is_raw_data(self, content: str) -> bool:
        """Check if content is raw data that shouldn't be re-displayed."""
        return (
                content.startswith('[') and content.endswith(']') or
                content.startswith('COMPONENT_SEARCH_COMPLETE:') or
                content.startswith('{') and '"Status"' in content or
                'COMPONENT ANALYSIS RESULTS' in content
        )

    def run_interactive(self) -> None:
        """Run interactive mode with enhanced guidance."""
        print("🚀 Simplified LangGraph Agent with Parametric Search Optimization")
        print("=" * 80)
        print("🎯 PARAMETRIC SEARCH OPTIMIZED COMMANDS:")
        print("- Analyze schematic: 'analyze schematic at [URL]'")
        print("  → Automatically optimized for parametric search when technical specs are found")
        print("- Parametric search: 'parametric search [PRODUCT_LINE] with filters [JSON]'")
        print("  → Best for technical specifications and precise component filtering")
        print()
        print("📋 STANDARD BOM COMMANDS:")
        print("- Search components: 'search component data for [JSON]'")
        print("- Create BOM: 'create empty bom named [NAME]'")
        print("- Get BOMs: 'get existing boms'")
        print("- Add parts to BOM: 'add parts to bom [BOM_NAME]'")
        print("- Full workflow: 'create bom from schematic at [URL] named [NAME]'")
        print("=" * 80)
        print("💡 TIP: The agent automatically detects when parametric search is optimal!")
        print("🤖 POWERED BY: Enhanced ReAct pattern for intelligent tool selection")
        print("=" * 80)

        while True:
            try:
                user_input = input("\nUser: ")
                if user_input.lower() in ["quit", "exit", "q"]:
                    print("Goodbye!")
                    break

                self.process_request(user_input)

            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")


def main():
    """Main entry point."""
    try:
        # Load configuration
        config = AppConfig.from_env()

        # Create and run simplified agent
        agent = SimplifiedLangGraphAgent(config)
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