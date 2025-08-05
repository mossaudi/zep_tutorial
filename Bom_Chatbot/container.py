# container.py
"""Manages service dependencies for the application."""

from langchain_google_genai import ChatGoogleGenerativeAI

from Bom_Chatbot.services.formatter import ComponentTableFormatter
from config import AppConfig
from clients.silicon_expert import SiliconExpertClient
from services.analysis import ComponentAnalysisService
from services.workflow import BOMWorkflowService
from services.bom_management import BOMManagementService
from services.parsing import ParsingService
from security.input_validator import SecurityValidator
from services.memory_manager import ConversationMemory
from services.resilience import CircuitBreaker, FallbackHandler
from llm.router import IntelligentLLMRouter
from cache.enhanced_cache import IntelligentCache, CachePolicy
from processor.concurrent_processor import ConcurrentProcessor, ProcessingConfig
from analytics.agent_analytics import AgentAnalytics
from learning.adaptive_responses import AdaptiveLearningSystem
from intelligence.smart_routing import AdvancedIntentClassifier


class Container:
    """A container for managing and injecting service dependencies."""

    def __init__(self, config: AppConfig, llm: ChatGoogleGenerativeAI):
        self.config = config
        self.llm = llm

        # --- Security & Resilience ---
        self.security_validator = SecurityValidator()
        self.memory = ConversationMemory()
        self.circuit_breaker = CircuitBreaker()
        self.fallback_handler = FallbackHandler(self.memory)

        # --- Performance & LLM Optimization ---
        cache_policy = CachePolicy(
            default_ttl=300,
            hot_cache_size=100,
            enable_redis=True,
            redis_prefix="bom_agent:"
        )
        self.intelligent_cache = IntelligentCache(
            policy=cache_policy,
            redis_url=config.redis_url if hasattr(config, 'redis_url') else None
        )

        # LLM Router with multiple providers
        llm_config = {
            "google_api_key": config.llm.google_api_key,
            "openai_api_key": getattr(config, 'openai_api_key', None)
        }
        self.llm_router = IntelligentLLMRouter(llm_config)

        # Concurrent processing
        processing_config = ProcessingConfig(
            max_concurrent=10,
            batch_size=50,
            timeout_per_item=30.0,
            enable_batching=True
        )
        self.concurrent_processor = ConcurrentProcessor(processing_config)

        # --- Intelligence & Analytics ---
        self.analytics = AgentAnalytics(enable_detailed_logging=True)
        self.learning_system = AdaptiveLearningSystem(learning_window_days=30)
        self.intent_classifier = AdvancedIntentClassifier(use_ml_models=True)

        # --- Core Services (Enhanced) ---
        self.silicon_expert_client = SiliconExpertClient(config.silicon_expert)
        self.parsing_service = ParsingService()
        self.analysis_service = ComponentAnalysisService(llm, self.silicon_expert_client)
        self.bom_service = BOMManagementService(self.silicon_expert_client)
        self.workflow_service = BOMWorkflowService(
            analysis_service=self.analysis_service,
            parsing_service=self.parsing_service,
            silicon_expert_client=self.silicon_expert_client
        )
        self.formatter = ComponentTableFormatter()

        print("✅ Container initialized")

    async def cleanup(self):
        """Cleanup resources."""
        await self.intelligent_cache.invalidate_pattern("")
        await self.concurrent_processor.close()