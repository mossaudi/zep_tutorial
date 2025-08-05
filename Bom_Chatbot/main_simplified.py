# main_simplified.py
"""LangGraph agent with a refined, modular architecture."""

import asyncio
import getpass
import json
from typing import Any, Dict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from Bom_Chatbot.container import Container
from analytics.agent_analytics import OperationType
from config import AppConfig
from exceptions import ConfigurationError
from intelligence.smart_routing import IntentCategory
from security.input_validator import DataValidationError
from services.formatter import ComponentTableFormatter
from services.resilience import retry_with_backoff, RetryConfig
from tools import get_tools


class BOMAgent:
    """LangGraph agent"""

    def __init__(self, config: AppConfig):
        self.config = config
        self._validate_config()
        self._initialize_components()
        self._create_react_agent()
        print("🚀 BOM Agent ready!")

    def _validate_config(self):
        """Enhanced configuration validation."""
        config_errors = self.config.validate()
        if config_errors:
            error_msg = "Configuration errors:\n" + "\n".join(f"- {error}" for error in config_errors)
            raise ConfigurationError(error_msg)

    def _initialize_components(self):
        """Initialize all enhanced components."""
        google_api_key = self.config.llm.google_api_key or getpass.getpass("Enter API key for Google Gemini: ")

        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-lite",
            google_api_key=google_api_key,
            temperature=0.1,
            max_output_tokens=30000
        )
        print("✅ Gemini model configured.")

        # container
        self.container = Container(self.config, self.llm)
        self.tools = get_tools(self.container)
        self.formatter = ComponentTableFormatter()
        print(f"✅ Initialized {len(self.tools)} tools.")

        self._setup_enhanced_handlers()
        print(f"✅ handlers configured.")

    def _create_react_agent(self):
        """Create enhanced ReAct agent."""
        system_prompt = """You are an Bill of Materials (BOM) management assistant with advanced capabilities:

        ## CORE CAPABILITIES:
        - **Security**: All inputs are validated and sanitized
        - **Memory**: I remember context across conversations
        - **Resilience**: Automatic retry with fallback strategies
        - **Performance**: Intelligent LLM routing for cost optimization
        - **Learning**: I adapt based on successful interaction patterns
        - **Analytics**: All operations are monitored and optimized

        ## PROCESSING STRATEGY:
        - Simple operations bypass LLM for efficiency
        - Complex analysis uses optimal LLM selection
        - Concurrent processing for batch operations
        - Intelligent caching prevents redundant work

        ## AVAILABLE TOOLS:
        - `analyze_schematic`: Advanced schematic analysis with component extraction
        - `search_component_data`: Multi-source component data search
        - `create_empty_bom`: Intelligent BOM creation with templates
        - `add_parts_to_bom`: Batch parts addition with validation
        - `get_boms`: Hierarchical BOM management view

        I provide actionable insights and learn from each interaction to improve future responses.
        """

        self.agent = create_react_agent(self.llm, self.tools, checkpointer=MemorySaver())
        self.system_prompt = system_prompt
        self.config_dict = {"configurable": {"thread_id": "enhanced_session"}}
        print("✅ ReAct agent created.")

    async def process_request_enhanced(self, user_input: str):
        """request processing"""
        # Start analytics tracking
        operation_id = self.container.analytics.start_operation(
            OperationType.SCHEMATIC_ANALYSIS,  # Will be determined by intent classification
            {"user_input": user_input}
        )

        print(f"\n{'=' * 80}\n🧠 PROCESSING: {user_input}\n{'=' * 80}")

        try:
            # Input validation and security
            await self._validate_input_security(user_input)

            # Intent classification for smart routing
            intent_result = self.container.intent_classifier.classify_intent(
                user_input,
                self.container.memory.get_context_summary()
            )

            print(f"🎯 Intent: {intent_result.intent} ({intent_result.confidence:.2f} confidence)")

            # Check for learned response patterns
            learned_response = self.container.learning_system.suggest_response(
                user_input,
                self.container.memory.get_context_summary()
            )

            if learned_response and learned_response["confidence"] > 0.8:
                print(f"🧠 Using learned pattern: {learned_response['pattern_id']}")
                result = learned_response["suggested_response"]
                self._record_successful_interaction(user_input, result, operation_id)
                return result

            # Intelligent routing based on complexity
            if intent_result.suggested_route == "direct_handler":
                result = await self._handle_direct_operation(user_input, intent_result)
            elif intent_result.category == IntentCategory.ANALYSIS:
                result = await self._handle_analysis_operation(user_input, intent_result)
            else:
                result = await self._handle_llm_operation(user_input, intent_result)

            # Record successful interaction for learning
            self._record_successful_interaction(user_input, result, operation_id)
            return result

        except Exception as e:
            error_msg = f"❌ Error: {e}"

            # Try fallback strategies
            fallback_result = self.container.fallback_handler.handle_api_quota_exceeded("request_processing")
            if fallback_result.get("success"):
                result = fallback_result
            else:
                result = error_msg

            # Record failed interaction
            self.container.analytics.end_operation(
                operation_id,
                success=False,
                error_message=str(e)
            )

            return result

    async def _validate_input_security(self, user_input: str):
        """Security validation."""
        try:
            # Check for image URLs and validate them
            import re
            urls = re.findall(r'https?://[^\s]+', user_input)
            for url in urls:
                self.container.security_validator.validate_image_url(url)

            # Validate JSON if present
            json_pattern = r'\{.*\}'
            json_matches = re.findall(json_pattern, user_input, re.DOTALL)
            for json_str in json_matches:
                self.container.security_validator.validate_json_input(json_str)

        except DataValidationError as e:
            raise ConfigurationError(f"Input validation failed: {e.message}")

    @retry_with_backoff(RetryConfig(max_attempts=3))
    async def _handle_direct_operation(self, user_input: str, intent_result) -> str:
        """Handle direct operations with retry."""
        if "list" in user_input.lower() and "bom" in user_input.lower():
            bom_result = self.container.bom_service.get_boms()
            if bom_result.get("success"):
                from Bom_Chatbot.models import BOMTreeResult
                bom_tree = BOMTreeResult(**bom_result["bom_tree"])
                return self.container.formatter.format_bom_tree(bom_tree)
            return "❌ Failed to retrieve BOMs"

        return "Direct operation completed"

    async def _handle_analysis_operation(self, user_input: str, intent_result) -> str:
        """Handle analysis operations with concurrent processing."""
        if "image_url" in intent_result.parameters:
            image_url = intent_result.parameters["image_url"]

            # Use workflow service for comprehensive analysis
            search_result = self.container.workflow_service.run_schematic_analysis_workflow(image_url)
            return self.container.formatter.format_search_result(search_result)

        return "Analysis operation requires image URL"

    async def _handle_llm_operation(self, user_input: str, intent_result) -> str:
        """Handle complex operations requiring LLM."""
        # Use intelligent LLM routing
        context = self.container.memory.get_context_summary()

        llm_result = await self.container.llm_router.route_and_execute(
            user_input,
            context,
            [SystemMessage(content=self.system_prompt), HumanMessage(content=user_input)]
        )

        print(f"💰 LLM Cost: ${llm_result['estimated_cost']:.4f} ({llm_result['llm_used']})")

        if hasattr(llm_result["response"], 'content'):
            return llm_result["response"].content
        return str(llm_result["response"])

    def _record_successful_interaction(self, user_input: str, result: str, operation_id: str):
        """Record successful interaction for learning."""
        # Learning system
        self.container.learning_system.record_interaction(
            user_input,
            result,
            self.container.memory.get_context_summary(),
            {"task_completed": True}
        )

        # Memory management
        self.container.memory.update_context(
            "general_interaction",
            {"user_input": user_input, "result": str(result)[:200]},
            success=True
        )

        # Analytics
        self.container.analytics.end_operation(
            operation_id,
            success=True,
            output_data=result,
            llm_used="system"
        )

    def _setup_enhanced_handlers(self):
        """Setup enhanced direct handlers."""
        self.direct_handlers = EnhancedDirectHandlerRegistry(self.container)

    def get_comprehensive_report(self) -> Dict[str, Any]:
        """Get comprehensive system report."""
        return {
            "performance": self.container.analytics.get_performance_report(),
            "learning": self.container.learning_system.get_learning_report(),
            "llm_usage": self.container.llm_router.get_usage_report(),
            "cache_stats": self.container.intelligent_cache.get_stats(),
            "memory_context": self.container.memory.get_context_summary(),
            "intent_classification": self.container.intent_classifier.get_classification_report()
        }

    async def run_interactive_enhanced(self):
        """Enhanced interactive mode."""
        print("\n🚀 Welcome to the BOM Agent!")
        print("   🛡️  Enhanced Security | 🧠 Smart Learning | ⚡ Optimized Performance")
        print("   Type 'report' for comprehensive analytics")
        print("   Type 'quit' or 'exit' to end.")
        print("-" * 80)

        while True:
            try:
                user_input = input("\n👤 User: ")

                if user_input.lower() in ["quit", "exit", "q"]:
                    await self.container.cleanup()
                    print("🤖 Goodbye! Analytics saved.")
                    break

                if user_input.lower() == "report":
                    report = self.get_comprehensive_report()
                    print(json.dumps(report, indent=2, default=str))
                    continue

                result = await self.process_request_enhanced(user_input)
                print(f"\n🤖 Assistant:\n{result}")
                print(f"\n{'=' * 80}\n✨ REQUEST COMPLETED\n{'=' * 80}")

            except KeyboardInterrupt:
                await self.container.cleanup()
                print("\n🤖 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Unexpected error: {e}")


class EnhancedDirectHandlerRegistry:
    """Enhanced direct handler registry with analytics."""

    def __init__(self, container: Container):
        self.container = container
        self.handlers = {}
        self._register_enhanced_handlers()

    def _register_enhanced_handlers(self):
        """Register enhanced handlers with analytics tracking."""

        async def enhanced_bom_listing(user_input: str) -> str:
            operation_id = self.container.analytics.start_operation(
                OperationType.BOM_LISTING,
                {"user_input": user_input}
            )

            try:
                # Check cache first
                cached_result = await self.container.intelligent_cache.get("bom_listing")
                if cached_result:
                    print("📋 Using cached BOM data")
                    self.container.analytics.end_operation(operation_id, success=True)
                    return cached_result

                # Get fresh data
                bom_result = self.container.bom_service.get_boms()

                if bom_result.get("success"):
                    from Bom_Chatbot.models import BOMTreeResult
                    bom_tree = BOMTreeResult(**bom_result["bom_tree"])
                    formatted_result = self.container.formatter.format_bom_tree(bom_tree)

                    # Cache result
                    await self.container.intelligent_cache.set("bom_listing", formatted_result, ttl=300)

                    self.container.analytics.end_operation(operation_id, success=True, output_data=formatted_result)
                    return formatted_result
                else:
                    error_msg = "❌ Failed to retrieve BOMs"
                    self.container.analytics.end_operation(operation_id, success=False, error_message=error_msg)
                    return error_msg

            except Exception as e:
                error_msg = f"❌ Error retrieving BOMs: {str(e)}"
                self.container.analytics.end_operation(operation_id, success=False, error_message=str(e))
                return error_msg

        # Register enhanced handlers
        self.handlers = {
            "list_boms": enhanced_bom_listing,
            "show_boms": enhanced_bom_listing,
            "view_boms": enhanced_bom_listing
        }

    async def find_and_execute_handler(self, user_input: str):
        """Find and execute enhanced handler."""
        user_lower = user_input.lower()

        for keyword, handler in self.handlers.items():
            if keyword.replace("_", " ") in user_lower:
                return await handler(user_input)

        return None


def main():
    """Enhanced main entry point."""
    try:
        config = AppConfig.from_env()
        agent = BOMAgent(config)
        asyncio.run(agent.run_interactive_enhanced())
    except (ConfigurationError, Exception) as e:
        print(f"❌ Fatal Error: {e}")
        return 1
    return 0


if __name__ == "__main__":
    exit(main())