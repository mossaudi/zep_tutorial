import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv


@dataclass
class SiliconExpertConfig:
    """Silicon Expert configuration."""
    username: str
    api_key: str
    base_url: str = "https://api.siliconexpert.com/ProductAPI"
    timeout: int = 30
    rate_limit_per_minute: int = 60

    def is_valid(self) -> bool:
        """Check if configuration is valid."""
        return bool(self.username and self.api_key)


@dataclass
class CacheConfig:
    """Cache configuration"""
    redis_url: Optional[str] = None
    default_ttl: int = 300
    hot_cache_size: int = 100
    enable_compression: bool = True


@dataclass
class LLMConfig:
    """LLM configuration for routing."""
    google_api_key: str
    openai_api_key: Optional[str] = None
    azure_openai_key: Optional[str] = None
    enable_local_models: bool = True
    default_temperature: float = 0.1
    max_tokens: int = 4096


@dataclass
class SecurityConfig:
    """Security configuration."""
    max_url_length: int = 2048
    max_json_size: int = 1024 * 1024  # 1MB
    allowed_domains: list = None
    enable_input_sanitization: bool = True

    def __post_init__(self):
        if self.allowed_domains is None:
            self.allowed_domains = [
                'imgur.com', 'github.com', 'githubusercontent.com',
                'drive.google.com', 'dropbox.com', 's3.amazonaws.com'
            ]


@dataclass
class AnalyticsConfig:
    """Analytics configuration."""
    enable_detailed_logging: bool = True
    performance_thresholds: dict = None
    enable_ml_classification: bool = True
    learning_window_days: int = 30

    def __post_init__(self):
        if self.performance_thresholds is None:
            self.performance_thresholds = {
                "max_execution_time": 30.0,
                "max_error_rate": 0.15,
                "max_cost_per_hour": 5.0
            }


@dataclass
class ProcessingConfig:
    """Processing configuration."""
    max_concurrent_operations: int = 10
    batch_size: int = 50
    timeout_per_operation: float = 30.0
    enable_async_processing: bool = True
    thread_pool_size: int = 4


@dataclass
class AppConfig:
    """application configuration."""
    # Core configs
    llm: LLMConfig
    silicon_expert: SiliconExpertConfig

    # Security & Resilience
    security: SecurityConfig

    # Performance
    cache: CacheConfig
    processing: ProcessingConfig

    # Intelligence
    analytics: AnalyticsConfig

    @classmethod
    def from_env(cls) -> 'AppConfig':
        """Create enhanced configuration from environment variables."""
        load_dotenv()

        # Core configurations
        llm_config = LLMConfig(
            google_api_key=os.getenv("GOOGLE_API_KEY", ""),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            azure_openai_key=os.getenv("AZURE_OPENAI_KEY"),
            enable_local_models=os.getenv("ENABLE_LOCAL_MODELS", "true").lower() == "true",
            default_temperature=float(os.getenv("DEFAULT_TEMPERATURE", "0.1")),
            max_tokens=int(os.getenv("MAX_TOKENS", "4096"))
        )

        silicon_expert_config = SiliconExpertConfig(
            username=os.getenv("SILICON_EXPERT_USERNAME", ""),
            api_key=os.getenv("SILICON_EXPERT_API_KEY", ""),
            timeout=int(os.getenv("SILICON_EXPERT_TIMEOUT", "30")),
            rate_limit_per_minute=int(os.getenv("SILICON_EXPERT_RATE_LIMIT", "60"))
        )

        # Security
        security_config = SecurityConfig(
            max_url_length=int(os.getenv("MAX_URL_LENGTH", "2048")),
            max_json_size=int(os.getenv("MAX_JSON_SIZE", str(1024 * 1024))),
            enable_input_sanitization=os.getenv("ENABLE_INPUT_SANITIZATION", "true").lower() == "true"
        )

        # Performance
        cache_config = CacheConfig(
            redis_url=os.getenv("REDIS_URL"),
            default_ttl=int(os.getenv("CACHE_DEFAULT_TTL", "300")),
            hot_cache_size=int(os.getenv("HOT_CACHE_SIZE", "100")),
            enable_compression=os.getenv("ENABLE_CACHE_COMPRESSION", "true").lower() == "true"
        )

        processing_config = ProcessingConfig(
            max_concurrent_operations=int(os.getenv("MAX_CONCURRENT_OPS", "10")),
            batch_size=int(os.getenv("BATCH_SIZE", "50")),
            timeout_per_operation=float(os.getenv("OPERATION_TIMEOUT", "30.0")),
            enable_async_processing=os.getenv("ENABLE_ASYNC", "true").lower() == "true",
            thread_pool_size=int(os.getenv("THREAD_POOL_SIZE", "4"))
        )

        # Intelligence
        analytics_config = AnalyticsConfig(
            enable_detailed_logging=os.getenv("ENABLE_DETAILED_LOGGING", "true").lower() == "true",
            enable_ml_classification=os.getenv("ENABLE_ML_CLASSIFICATION", "true").lower() == "true",
            learning_window_days=int(os.getenv("LEARNING_WINDOW_DAYS", "30"))
        )

        return cls(
            llm=llm_config,
            silicon_expert=silicon_expert_config,
            security=security_config,
            cache=cache_config,
            processing=processing_config,
            analytics=analytics_config
        )

    def validate(self) -> list[str]:
        """Validate enhanced configuration and return list of errors."""
        errors = []

        # Core validation
        if not self.llm.google_api_key:
            errors.append("GOOGLE_API_KEY is required")

        if not self.silicon_expert.is_valid():
            errors.append("Silicon Expert credentials are required")

        # validation
        if self.security.max_url_length < 100:
            errors.append("MAX_URL_LENGTH too small (minimum 100)")

        if self.security.max_json_size < 1024:
            errors.append("MAX_JSON_SIZE too small (minimum 1KB)")

        # validation
        if self.processing.max_concurrent_operations < 1:
            errors.append("MAX_CONCURRENT_OPS must be at least 1")

        if self.processing.batch_size < 1:
            errors.append("BATCH_SIZE must be at least 1")

        # validation
        if self.analytics.learning_window_days < 1:
            errors.append("LEARNING_WINDOW_DAYS must be at least 1")

        return errors

    def get_feature_summary(self) -> dict:
        """Get summary of enabled features."""
        return {
            "core_features": {
                "google_gemini": bool(self.llm.google_api_key),
                "openai_models": bool(self.llm.openai_api_key),
                "local_models": self.llm.enable_local_models,
                "silicon_expert": self.silicon_expert.is_valid()
            },
            "security": {
                "input_validation": self.security.enable_input_sanitization,
                "url_length_limit": self.security.max_url_length,
                "json_size_limit": self.security.max_json_size
            },
            "performance": {
                "redis_cache": bool(self.cache.redis_url),
                "async_processing": self.processing.enable_async_processing,
                "max_concurrent": self.processing.max_concurrent_operations,
                "batch_processing": self.processing.batch_size > 1
            },
            "intelligence": {
                "detailed_analytics": self.analytics.enable_detailed_logging,
                "ml_classification": self.analytics.enable_ml_classification,
                "adaptive_learning": True,
                "learning_window": f"{self.analytics.learning_window_days} days"
            }
        }