# main_simplified.py
"""Enhanced LangGraph agent with hidden intelligent search capabilities."""

import getpass
from typing import List

from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from config import AppConfig
from exceptions import ConfigurationError, AgentError
from tools import initialize_tools, get_tools


class IntelligentBOMAgent:
    """Enhanced LangGraph agent with hidden intelligent search capabilities."""

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

        print("✅ Intelligent BOM Agent ready with hidden smart search capabilities")

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
        print(f"✅ Initialized {len(self.tools)} tools with intelligent search automation")

    def _create_react_agent(self) -> None:
        """Create ReAct agent with enhanced system prompt."""
        memory = MemorySaver()

        # Enhanced system prompt with intelligent automation
        system_prompt = """You are an expert BOM (Bill of Materials) management assistant with sophisticated automated search intelligence.
            
            🎯 YOUR CORE EXPERTISE:
            - Schematic analysis with AUTOMATIC component enhancement
            - Intelligent search routing (parametric → keyword fallback)
            - Silicon Expert integration with hidden taxonomy mapping
            - BOM creation and management workflows
            - Component lifecycle and compliance analysis
            
            🤖 INTELLIGENT AUTOMATION:
            Your tools now include HIDDEN smart search capabilities:
            - Taxonomy mapping happens automatically (no user interaction needed)
            - Parametric search attempts first when technical specs are available
            - Automatic fallback to keyword search if parametric fails
            - Combined description search for maximum component coverage
            - All search methods are invisible to the user - they see only results
            
            🚀 ENHANCED TOOL BEHAVIOR:
            
            1. **analyze_schematic** - NOW FULLY AUTOMATED:
               - Extracts components from schematic
               - Automatically tries parametric search for components with plName/selectedFilters
               - Falls back to keyword search with combined descriptions if parametric fails
               - Returns comprehensive enhanced component data
               - Shows search method summary (parametric vs keyword success rates)
            
            2. **parametric_search** - TAXONOMY-ENHANCED:
               - Automatically maps product lines to correct taxonomy
               - No need for user to run separate taxonomy tools
               - Enhanced with complete feature extraction
               - Intelligent error handling and suggestions
            
            3. **search_component_data** - SMART FALLBACK:
               - Each component gets smart search with automatic fallback
               - Combines all available data into comprehensive search strings
               - Maximizes component data retrieval success rates
            
            4. **create_bom_from_schematic** - END-TO-END AUTOMATION:
               - Combines schematic analysis with intelligent component search
               - Automatically creates and populates BOM with enhanced data
               - Single command for complete workflow
            
            🔧 HIDDEN CAPABILITIES (User Never Sees These):
            - Automatic taxonomy loading and mapping
            - Intelligent keyword search with combined descriptions
            - Per-component intelligent search routing
            - Smart fallback mechanisms
            - Search method optimization
            
            📊 USER EXPERIENCE IMPROVEMENTS:
            - Single tool calls now provide comprehensive results
            - No need for users to understand search method differences
            - Automatic optimization for best possible component data
            - Clear success/failure reporting with method attribution
            - Actionable next-step suggestions
            
            🎨 COMMUNICATION STYLE:
            - Focus on RESULTS, not process complexity
            - Highlight automated enhancements: "automatically searched", "enhanced with", "intelligent fallback"
            - Show success metrics: "X components via parametric, Y via keyword search"
            - Provide clear next steps for BOM creation
            - Emphasize the power of automated search intelligence
            
            💡 KEY MESSAGING:
            - "Your schematic has been analyzed with automatic component enhancement"
            - "Intelligent search found data for X/Y components using multiple methods"
            - "Components automatically enhanced with parametric specifications"
            - "Smart search optimization achieved Z% success rate"
            
            🔄 WORKFLOW INTELLIGENCE:
            1. User requests schematic analysis → Automatic multi-method search → Enhanced results
            2. User requests parametric search → Automatic taxonomy mapping → Optimized results
            3. User requests component search → Automatic smart routing → Best possible data
            4. All taxonomy and keyword operations happen invisibly in the background
            
            🎯 AVAILABLE TOOLS (User-Visible):
            - analyze_schematic: Enhanced with automatic intelligent search
            - parametric_search: Enhanced with taxonomy mapping
            - search_component_data: Enhanced with smart fallback
            - create_empty_bom: Standard BOM creation
            - get_boms: List existing BOMs
            - add_parts_to_bom: Add parts to existing BOM
            - create_bom_from_schematic: Complete end-to-end automation
            
            Remember: You now have sophisticated automated search intelligence. Users get the benefits of multiple search methods without complexity. Focus on results and the power of your automated enhancements!"""

        # Create agent
        self.agent = create_react_agent(
            self.llm,
            self.tools,
            checkpointer=memory
        )

        # Store system prompt for use in conversations
        self.system_prompt = system_prompt

        # Configuration for conversations
        self.config_dict = {"configurable": {"thread_id": "1"}}
        print("✅ ReAct agent created with intelligent automation")

    def process_request(self, user_input: str) -> None:
        """Process user request using ReAct pattern."""
        print(f"\n{'=' * 80}")
        print(f"🤖 PROCESSING REQUEST: {user_input}")
        print(f"{'=' * 80}")

        try:
            # Include system prompt as the first message
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=user_input)
            ]

            # Simple ReAct invocation
            result = self.agent.invoke(
                {"messages": messages},
                self.config_dict
            )

            # Display the final result
            if result and "messages" in result:
                final_message = result["messages"][-1]
                if hasattr(final_message, 'content') and final_message.content:
                    if not self._is_raw_data(final_message.content):
                        print(f"\n🤖 Assistant: {final_message.content}")

            print(f"\n{'=' * 80}")
            print("✨ REQUEST COMPLETED WITH INTELLIGENT AUTOMATION")
            print(f"{'=' * 80}")

        except Exception as e:
            print(f"❌ Error processing request: {e}")
            print(f"Error type: {type(e).__name__}")
            # Add more debugging info
            import traceback
            traceback.print_exc()

    def _is_raw_data(self, content: str) -> bool:
        """Check if content is raw data that shouldn't be re-displayed."""
        return (
                content.startswith('[') and content.endswith(']') or
                content.startswith('COMPONENT_SEARCH_COMPLETE:') or
                content.startswith('PARAMETRIC_SEARCH_COMPLETE:') or
                content.startswith('SCHEMATIC_ANALYSIS_COMPLETE:') or
                content.startswith('{') and '"Status"' in content or
                'COMPONENT ANALYSIS RESULTS' in content or
                'PARAMETRIC SEARCH RESULTS' in content
        )

    def run_interactive(self) -> None:
        """Run interactive mode with enhanced guidance."""
        print("🤖 Intelligent BOM Agent with Hidden Smart Search")
        print("=" * 80)
        print("🧠 INTELLIGENT AUTOMATION FEATURES:")
        print("- Automatic taxonomy mapping (hidden from user)")
        print("- Smart parametric → keyword search fallback")
        print("- Combined description search for maximum coverage")
        print("- All search optimization happens automatically")
        print()
        print("🎯 ENHANCED CAPABILITIES:")
        print("- analyze_schematic: Now includes automatic component enhancement")
        print("- parametric_search: Auto-maps product lines, extracts full features")
        print("- search_component_data: Smart fallback for comprehensive results")
        print("- create_bom_from_schematic: Complete end-to-end automation")
        print()
        print("📋 USAGE EXAMPLES:")
        print("- 'analyze schematic at [URL]' → Auto-enhanced component data")
        print("- 'parametric search Laser Diodes with current 40-50mA' → Auto-taxonomy mapping")
        print("- 'create bom from schematic at [URL] named MyProject' → Complete automation")
        print("- 'search components [JSON]' → Smart search with automatic fallback")
        print("=" * 80)
        print("💡 TIP: Every search is now automatically optimized for best results!")
        print("🎯 POWERED BY: Hidden intelligent search with automatic fallback")
        print("=" * 80)

        while True:
            try:
                user_input = input("\n🤖 User: ")
                if user_input.lower() in ["quit", "exit", "q"]:
                    print("🤖 Goodbye! Thanks for using Intelligent BOM Agent!")
                    break

                self.process_request(user_input)

            except KeyboardInterrupt:
                print("\n🤖 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")


def main():
    """Main entry point."""
    try:
        # Load configuration
        config = AppConfig.from_env()

        # Create and run intelligent agent
        agent = IntelligentBOMAgent(config)
        agent.run_interactive()

    except ConfigurationError as e:
        print(f"❌ Configuration Error: {e}")
        print("Please check your environment variables and try again.")
        return 1

    except AgentError as e:
        print(f"❌ Agent Error: {e}")
        return 1

    except Exception as e:
        print(f"❌ Unexpected Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())