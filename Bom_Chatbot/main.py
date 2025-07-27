# Updated main.py with enhanced tool selection
"""Enhanced LangGraph agent with parametric search optimization."""

import getpass
import time
from typing import Annotated, List
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
from services.intelligent_selection import EnhancedIntelligentToolSelector, ConversationContext
from typing import Optional


class State(TypedDict):
    """Enhanced state for intelligent tool selection with parametric search optimization."""
    messages: Annotated[list, add_messages]
    component_data: str
    needs_table_display: bool
    current_step: str
    conversation_context: ConversationContext
    recommended_tools: List[str]
    last_tool_output: Optional[str]  # NEW: Track last tool output


class EnhancedLangGraphAgent:
    """Enhanced LangGraph agent with parametric search optimization."""

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

        # Use enhanced tool selector
        self.tool_selector = EnhancedIntelligentToolSelector(self.llm, self.tools)
        self.conversation_context = ConversationContext(
            recent_messages=[],
            previous_tool_results=[],
            available_data={},
            last_tool_output=None  # NEW: Initialize last tool output
        )

        # Create graph
        self._create_graph()

        self.progress.success("Agent Initialization",
                              "Enhanced LangGraph agent ready with parametric search optimization")

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
        """Create and compile the enhanced LangGraph workflow."""
        self.progress.info("Graph Creation", "Building enhanced LangGraph workflow...")

        tool_node = ToolNode(self.tools)
        graph_builder = StateGraph(State)

        # Add nodes
        graph_builder.add_node("intelligent_selection", self._enhanced_intelligent_selection_node)
        graph_builder.add_node("chatbot", self._enhanced_chatbot_node)
        graph_builder.add_node("tools", tool_node)
        graph_builder.add_node("table_display", self._table_display_node)

        # Add edges
        graph_builder.add_edge(START, "intelligent_selection")
        graph_builder.add_edge("intelligent_selection", "chatbot")
        graph_builder.add_conditional_edges("chatbot", tools_condition)
        graph_builder.add_edge("tools", "chatbot")
        graph_builder.add_conditional_edges(
            "chatbot", self._should_display_table,
            {"table_display": "table_display", "end": END}
        )
        graph_builder.add_edge("table_display", END)

        memory = MemorySaver()
        self.graph = graph_builder.compile(checkpointer=memory)
        self.progress.success("Graph Creation", "Enhanced LangGraph workflow compiled")

    def _enhanced_intelligent_selection_node(self, state: State):
        """Enhanced intelligent selection with parametric search awareness."""
        if not state.get("messages"):
            return state

        last_message = state["messages"][-1]
        if hasattr(last_message, 'content') and isinstance(last_message.content, str):

            # Update conversation context with last tool output
            context = state.get("conversation_context", self.conversation_context)
            context.recent_messages.append(last_message.content)

            # NEW: Include last tool output in context
            if state.get("last_tool_output"):
                context.last_tool_output = state["last_tool_output"]

            # Get enhanced recommendations
            recommendations = self.tool_selector.select_tools(last_message.content, context)

            recommended_tool_names = []
            for rec in recommendations:
                recommended_tool_names.append(rec.tool_name)

                # Enhanced progress reporting
                confidence_indicator = "🎯" if rec.confidence > 0.8 else "🔍"
                self.progress.info(
                    "Enhanced Tool Selection",
                    f"{confidence_indicator} {rec.tool_name} (confidence: {rec.confidence:.2f}): {rec.reasoning}"
                )

            return {
                "conversation_context": context,
                "recommended_tools": recommended_tool_names
            }
        return state

    def _enhanced_chatbot_node(self, state: State):
        """Enhanced chatbot with parametric search optimization."""
        recommended_tools = state.get("recommended_tools", [])

        if recommended_tools:
            dynamic_tools = []
            for tool_name in recommended_tools[:3]:  # Limit to top 3 recommendations
                tool = next((t for t in self.tools if t.name == tool_name), None)
                if tool:
                    dynamic_tools.append(tool)

            tools_to_bind = dynamic_tools if dynamic_tools else self.tools

            tool_names = [t.name for t in tools_to_bind]

            # Enhanced progress reporting
            if 'parametric_search' in tool_names:
                self.progress.info("Parametric Optimization",
                                   f"🚀 Prioritizing parametric search: {', '.join(tool_names)}")
            else:
                self.progress.info("Dynamic Tools", f"Using: {', '.join(tool_names)}")
        else:
            tools_to_bind = self.tools

        llm_with_tools = self.llm.bind_tools(tools_to_bind)
        response = llm_with_tools.invoke(state["messages"])

        # NEW: Capture tool output for next iteration
        response_updates = {"messages": [response]}

        if hasattr(response, 'tool_calls') and response.tool_calls:
            return response_updates

        # Check if response contains component data and capture it
        if hasattr(response, 'content') and response.content:
            response_updates["last_tool_output"] = response.content

            if self._contains_component_data(state):
                last_message = state["messages"][-1]
                response_updates.update({
                    "component_data": last_message.content,
                    "needs_table_display": True
                })
            else:
                response_updates.update({
                    "component_data": "",
                    "needs_table_display": False
                })

        return response_updates

    def _table_display_node(self, state: State):
        """Enhanced table display node."""
        # The table display is handled within the tools themselves
        # This node just resets the display flag and provides suggestions

        # NEW: Generate intelligent suggestions based on last output
        suggestions = self._generate_next_step_suggestions(state)
        if suggestions:
            print("\n" + "=" * 80)
            print("🧠 INTELLIGENT NEXT STEPS:")
            print("=" * 80)
            print(suggestions)

        return {
            "messages": [],
            "needs_table_display": False,
            "last_tool_output": None  # Reset for next iteration
        }

    def _generate_next_step_suggestions(self, state: State) -> str:
        """Generate intelligent suggestions based on current state."""
        suggestions = []

        last_output = state.get("last_tool_output", "")
        context = state.get("conversation_context", self.conversation_context)

        # Check if we have parametric-ready data
        if context.last_tool_output and context._contains_parametric_format(context.last_tool_output):
            suggestions.append("🎯 Found structured component data - perfect for parametric search!")
            suggestions.append("   Try: 'run parametric search on these components'")

        # Check component count
        if context.available_data.get('component_count'):
            count = context.available_data['component_count']
            suggestions.append(f"📊 You have {count} components ready for BOM creation")
            suggestions.append("   Try: 'create bom from these components named MyProject_BOM'")

        # General suggestions
        if not suggestions:
            suggestions.append("🚀 Continue with: BOM creation, parametric search, or component analysis")

        return "\n".join(suggestions)

    def _should_display_table(self, state: State) -> str:
        """Enhanced table display decision logic."""
        if state.get("needs_table_display", False):
            self.progress.info("Flow Decision", "Table display requested")
            return "table_display"

        # Check for component data in tool responses
        if self._contains_component_data(state):
            self.progress.info("Flow Decision", "Component data found in messages")
            return "table_display"

        return "end"

    def _contains_component_data(self, state: State) -> bool:
        """Enhanced component data detection."""
        if not state.get("messages"):
            return False

        last_message = state["messages"][-1]

        # Check for tool messages with component data
        if (hasattr(last_message, 'content') and
                hasattr(last_message, 'name') and
                last_message.name in ['analyze_schematic', 'search_component_data', 'parametric_search']):
            return True

        # Check for component data markers
        if (hasattr(last_message, 'content') and
                isinstance(last_message.content, str)):
            # Check for various component data indicators
            component_indicators = [
                "COMPONENT_SEARCH_COMPLETE:",
                '"plName"',
                '"selectedFilters"',
                "Component Analysis Results",
                "parametric search results"
            ]

            return any(indicator in last_message.content for indicator in component_indicators)

        return False

    def process_request(self, user_input: str) -> None:
        """Enhanced request processing with parametric search awareness."""
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

                            # Enhanced message display with parametric search detection
                            content = last_message.content
                            if '"plName"' in content and '"selectedFilters"' in content:
                                print(f"\n🎯 Assistant (Parametric-Ready): {content}")
                            else:
                                print(f"\n💬 Assistant: {content}")

        except Exception as e:
            self.progress.error("Request Processing", str(e))
            print(f"❌ Error processing request: {e}")

    def _is_raw_data(self, content: str) -> bool:
        """Enhanced raw data detection."""
        return (
                content.startswith('[') and content.endswith(']') or
                content.startswith('COMPONENT_SEARCH_COMPLETE:') or
                content.startswith('{') and '"Status"' in content  # API responses
        )

    def run_interactive(self) -> None:
        """Enhanced interactive mode with parametric search guidance."""
        print("🚀 Enhanced LangGraph Agent with Parametric Search Optimization")
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
        print("💡 TIP: The system automatically detects when parametric search is optimal!")
        print("=" * 80)

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
    """Enhanced main entry point."""
    try:
        # Load configuration
        config = AppConfig.from_env()

        # Create and run enhanced agent
        agent = EnhancedLangGraphAgent(config)
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