# exceptions.py
"""Custom exceptions for the LangGraph agent."""

from typing import Optional, Any, Dict


class AgentError(Exception):
    """Base exception for all agent-related errors."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ConfigurationError(AgentError):
    """Raised when configuration is invalid or missing."""
    pass


class AuthenticationError(AgentError):
    """Raised when authentication fails."""

    def __init__(self, message: str, provider: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.provider = provider
        self.status_code = status_code


class APIError(AgentError):
    """Base class for API-related errors."""

    def __init__(self, message: str, status_code: Optional[int] = None,
                 response_data: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_data = response_data or {}


class SiliconExpertError(APIError):
    """Raised when Silicon Expert API operations fail."""
    pass


class ComponentSearchError(SiliconExpertError):
    """Raised when component search fails."""

    def __init__(self, message: str, component_name: str, search_query: str, **kwargs):
        super().__init__(message, **kwargs)
        self.component_name = component_name
        self.search_query = search_query


class BOMError(SiliconExpertError):
    """Raised when BOM operations fail."""

    def __init__(self, message: str, bom_name: Optional[str] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.bom_name = bom_name


class SchematicAnalysisError(AgentError):
    """Raised when schematic analysis fails."""

    def __init__(self, message: str, image_url: str, **kwargs):
        super().__init__(message, **kwargs)
        self.image_url = image_url


class DataValidationError(AgentError):
    """Raised when data validation fails."""

    def __init__(self, message: str, data_type: str, validation_errors: list[str]):
        super().__init__(message)
        self.data_type = data_type
        self.validation_errors = validation_errors


class JSONProcessingError(AgentError):
    """Raised when JSON processing fails."""

    def __init__(self, message: str, raw_data: str, parsing_stage: str):
        super().__init__(message)
        self.raw_data = raw_data[:200] + "..." if len(raw_data) > 200 else raw_data
        self.parsing_stage = parsing_stage