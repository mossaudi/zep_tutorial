# config.py
"""Configuration management for the LangGraph agent."""

import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv


@dataclass
class SiliconExpertConfig:
    """Configuration for Silicon Expert API."""
    username: str
    api_key: str
    base_url: str = "https://api.siliconexpert.com/ProductAPI"

    def is_valid(self) -> bool:
        """Check if configuration is valid."""
        return bool(self.username and self.api_key)


@dataclass
class AppConfig:
    """Main application configuration."""
    google_api_key: str
    silicon_expert: SiliconExpertConfig
    access_token: Optional[str] = None
    base_url: Optional[str] = None

    @classmethod
    def from_env(cls) -> 'AppConfig':
        """Create configuration from environment variables."""
        load_dotenv()

        silicon_expert_config = SiliconExpertConfig(
            username=os.getenv("SILICON_EXPERT_USERNAME", ""),
            api_key=os.getenv("SILICON_EXPERT_API_KEY", "")
        )

        return cls(
            google_api_key=os.getenv("GOOGLE_API_KEY", ""),
            silicon_expert=silicon_expert_config,
            access_token=os.getenv("ACCESS_TOKEN"),
            base_url=os.getenv("BASE_URL")
        )

    def validate(self) -> list[str]:
        """Validate configuration and return list of errors."""
        errors = []

        if not self.google_api_key:
            errors.append("GOOGLE_API_KEY is required")

        if not self.silicon_expert.is_valid():
            errors.append("Silicon Expert credentials are required")

        return errors