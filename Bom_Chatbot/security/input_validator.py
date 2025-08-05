# security/input_validator.py
"""Enhanced input validation and security for the BOM agent."""

import json
import re
from typing import Dict, Any
from urllib.parse import urlparse

from Bom_Chatbot.exceptions import DataValidationError


class SecurityValidator:
    """Security-focused input validation."""

    ALLOWED_IMAGE_DOMAINS = {
        'imgur.com', 'github.com', 'githubusercontent.com',
        'drive.google.com', 'dropbox.com', 's3.amazonaws.com'
    }

    ALLOWED_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
    MAX_URL_LENGTH = 2048
    MAX_JSON_SIZE = 1024 * 1024  # 1MB

    @classmethod
    def validate_image_url(cls, url: str) -> bool:
        """Validate image URL for security and format."""
        if not url or len(url) > cls.MAX_URL_LENGTH:
            raise DataValidationError("URL too long or empty", "image_url", ["Invalid length"])

        try:
            parsed = urlparse(url)

            # Check scheme
            if parsed.scheme not in ['http', 'https']:
                raise DataValidationError("Invalid URL scheme", "image_url", ["Must be http/https"])

            # Check domain (optional - remove for open access)
            # if parsed.netloc.lower() not in cls.ALLOWED_IMAGE_DOMAINS:
            #     raise DataValidationError("Domain not allowed", "image_url", [f"Allowed: {cls.ALLOWED_IMAGE_DOMAINS}"])

            # Check file extension
            path_lower = parsed.path.lower()
            if not any(path_lower.endswith(ext) for ext in cls.ALLOWED_IMAGE_EXTENSIONS):
                raise DataValidationError("Invalid image format", "image_url",
                                          [f"Allowed: {cls.ALLOWED_IMAGE_EXTENSIONS}"])

            return True

        except Exception as e:
            raise DataValidationError(f"URL validation failed: {str(e)}", "image_url", [str(e)])

    @classmethod
    def sanitize_component_data(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize component data input."""
        sanitized = {}

        # Define allowed fields and their max lengths
        field_limits = {
            'name': 100,
            'part_number': 50,
            'manufacturer': 50,
            'description': 500,
            'value': 50,
            'features': 200,
            'quantity': 10,
            'designator': 20,
            'functional_block': 100,
            'notes': 500
        }

        for field, max_length in field_limits.items():
            if field in data:
                value = str(data[field])[:max_length]  # Truncate
                # Remove potential XSS characters
                value = re.sub(r'[<>"\']', '', value)
                sanitized[field] = value.strip()

        return sanitized

    @classmethod
    def validate_json_input(cls, json_string: str) -> Dict[str, Any]:
        """Validate and parse JSON input safely."""
        if len(json_string.encode('utf-8')) > cls.MAX_JSON_SIZE:
            raise DataValidationError("JSON too large", "json_input", ["Max 1MB allowed"])

        try:
            data = json.loads(json_string)
            return data
        except json.JSONDecodeError as e:
            raise DataValidationError(f"Invalid JSON: {str(e)}", "json_input", [str(e)])
