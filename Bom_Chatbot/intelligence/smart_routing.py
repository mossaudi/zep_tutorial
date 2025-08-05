# intelligence/smart_routing.py
"""Advanced intent classification and smart routing system."""

import re
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Any, Optional

# Try to import transformers for advanced classification
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from Bom_Chatbot.services.progress import get_progress_tracker


class IntentCategory(Enum):
    """High-level intent categories."""
    ANALYSIS = "analysis"  # Schematic analysis, component extraction
    SEARCH = "search"  # Component search, data lookup
    MANAGEMENT = "management"  # BOM creation, parts addition
    INFORMATION = "information"  # Listing, viewing, status
    TROUBLESHOOTING = "troubleshooting"  # Error handling, help


@dataclass
class ClassificationResult:
    """Result of intent classification."""
    intent: str
    category: IntentCategory
    confidence: float
    suggested_route: str
    parameters: Dict[str, Any]
    context_requirements: List[str]


class AdvancedIntentClassifier:
    """Advanced intent classification using multiple techniques."""

    def __init__(self, use_ml_models: bool = True):
        self.progress = get_progress_tracker()
        self.use_ml_models = use_ml_models and TRANSFORMERS_AVAILABLE

        # Initialize ML model if available
        self.ml_classifier = None
        if self.use_ml_models:
            try:
                self._initialize_ml_classifier()
            except Exception as e:
                self.progress.warning("Intent Classification", f"ML model unavailable: {e}")
                self.use_ml_models = False

        # Rule-based patterns for high-precision classification
        self.intent_patterns = {
            "schematic_analysis": {
                "patterns": [
                    r"analyz\w*\s+(schematic|image|circuit|diagram)",
                    r"extract\s+(component|part).*from.*(image|schematic)",
                    r"(schematic|circuit|diagram).*analysis",
                    r"identify.*component.*in.*image"
                ],
                "category": IntentCategory.ANALYSIS,
                "route": "analyze_schematic_tool",
                "required_params": ["image_url"],
                "context_requirements": []
            },
            "component_search": {
                "patterns": [
                    r"search\s+(component|part)",
                    r"find\s+(part|component).*number",
                    r"look.*up.*part.*\w+",
                    r"get.*data.*for.*component"
                ],
                "category": IntentCategory.SEARCH,
                "route": "search_component_data_tool",
                "required_params": ["component_data"],
                "context_requirements": []
            },
            "bom_creation": {
                "patterns": [
                    r"create\s+(new\s+)?bom",
                    r"make\s+(new\s+)?bom",
                    r"(start|begin).*(new\s+)?bom",
                    r"empty.*bom"
                ],
                "category": IntentCategory.MANAGEMENT,
                "route": "create_empty_bom_tool",
                "required_params": ["name"],
                "context_requirements": []
            },
            "add_parts_to_bom": {
                "patterns": [
                    r"add\s+(part|component).*to.*bom",
                    r"insert.*component.*bom",
                    r"put.*part.*in.*bom"
                ],
                "category": IntentCategory.MANAGEMENT,
                "route": "add_parts_to_bom_tool",
                "required_params": ["bom_name", "parts_data"],
                "context_requirements": ["existing_bom"]
            },
            "list_boms": {
                "patterns": [
                    r"(list|show|display|view)\s+(my\s+)?bom",
                    r"what.*bom.*do.*i.*have",
                    r"see.*my.*bom",
                    r"bom.*overview"
                ],
                "category": IntentCategory.INFORMATION,
                "route": "direct_handler",
                "required_params": [],
                "context_requirements": []
            },
            "help_request": {
                "patterns": [
                    r"help",
                    r"what.*can.*you.*do",
                    r"how.*do.*i",
                    r"instructions",
                    r"guide"
                ],
                "category": IntentCategory.TROUBLESHOOTING,
                "route": "help_handler",
                "required_params": [],
                "context_requirements": []
            }
        }

    def _initialize_ml_classifier(self):
        """Initialize ML-based intent classifier."""
        try:
            # Use a lightweight model for intent classification
            model_name = "microsoft/DialoGPT-medium"  # Placeholder - use appropriate model
            self.ml_classifier = pipeline(
                "text-classification",
                model=model_name,
                return_all_scores=True
            )
            self.progress.success("Intent Classification", "ML classifier initialized")
        except Exception as e:
            self.progress.warning("Intent Classification", f"ML initialization failed: {e}")
            raise

    def classify_intent(self, user_input: str, context: Dict[str, Any] = None) -> ClassificationResult:
        """Classify user intent using multiple techniques."""
        context = context or {}

        # First try rule-based classification (high precision)
        rule_result = self._classify_with_rules(user_input, context)
        if rule_result and rule_result.confidence > 0.8:
            return rule_result

        # Then try ML classification if available
        if self.use_ml_models:
            ml_result = self._classify_with_ml(user_input, context)
            if ml_result and ml_result.confidence > 0.7:
                return ml_result

        # Fallback to contextual classification
        context_result = self._classify_with_context(user_input, context)
        if context_result:
            return context_result

        # Default classification
        return ClassificationResult(
            intent="general_query",
            category=IntentCategory.INFORMATION,
            confidence=0.3,
            suggested_route="llm_agent",
            parameters={},
            context_requirements=[]
        )

    def _classify_with_rules(self, user_input: str, context: Dict[str, Any]) -> Optional[ClassificationResult]:
        """Rule-based intent classification."""
        user_input_lower = user_input.lower()

        for intent, config in self.intent_patterns.items():
            for pattern in config["patterns"]:
                if re.search(pattern, user_input_lower):
                    # Extract parameters from input
                    parameters = self._extract_parameters(user_input, intent)

                    # Check context requirements
                    context_satisfied = self._check_context_requirements(
                        context, config["context_requirements"]
                    )

                    confidence = 0.9 if context_satisfied else 0.7

                    return ClassificationResult(
                        intent=intent,
                        category=config["category"],
                        confidence=confidence,
                        suggested_route=config["route"],
                        parameters=parameters,
                        context_requirements=config["context_requirements"]
                    )

        return None

    def _classify_with_ml(self, user_input: str, context: Dict[str, Any]) -> Optional[ClassificationResult]:
        """ML-based intent classification."""
        if not self.ml_classifier:
            return None

        try:
            # Get ML predictions
            results = self.ml_classifier(user_input)

            # Find best prediction
            best_result = max(results, key=lambda x: x['score'])

            if best_result['score'] > 0.7:
                # Map ML label to our intent system
                intent = self._map_ml_label_to_intent(best_result['label'])

                if intent in self.intent_patterns:
                    config = self.intent_patterns[intent]
                    parameters = self._extract_parameters(user_input, intent)

                    return ClassificationResult(
                        intent=intent,
                        category=config["category"],
                        confidence=best_result['score'],
                        suggested_route=config["route"],
                        parameters=parameters,
                        context_requirements=config["context_requirements"]
                    )

        except Exception as e:
            self.progress.warning("ML Classification", f"Error: {e}")

        return None

    def _classify_with_context(self, user_input: str, context: Dict[str, Any]) -> Optional[ClassificationResult]:
        """Context-based classification for ambiguous inputs."""

        # If user has recent schematic analysis, they might want to create BOM
        if (context.get("session_components") and
                any(word in user_input.lower() for word in ["bom", "create", "make", "add"])):
            return ClassificationResult(
                intent="create_bom_from_analysis",
                category=IntentCategory.MANAGEMENT,
                confidence=0.6,
                suggested_route="create_empty_bom_tool",
                parameters={"use_session_components": True},
                context_requirements=["session_components"]
            )

        # If user has existing BOMs and mentions adding
        if (context.get("recent_boms") and
                any(word in user_input.lower() for word in ["add", "insert", "put"])):
            return ClassificationResult(
                intent="add_to_existing_bom",
                category=IntentCategory.MANAGEMENT,
                confidence=0.6,
                suggested_route="add_parts_to_bom_tool",
                parameters={"suggest_recent_bom": True},
                context_requirements=["recent_boms"]
            )

        return None

    def _extract_parameters(self, user_input: str, intent: str) -> Dict[str, Any]:
        """Extract parameters from user input based on intent."""
        parameters = {}

        # URL extraction for schematic analysis
        if intent == "schematic_analysis":
            url_pattern = r'https?://[^\s]+'
            urls = re.findall(url_pattern, user_input)
            if urls:
                parameters["image_url"] = urls[0]

        # BOM name extraction
        if intent in ["bom_creation", "add_parts_to_bom"]:
            # Look for quoted names or names after "called", "named", etc.
            name_patterns = [
                r'["\']([^"\']+)["\']',
                r'(?:called|named|titled)\s+([a-zA-Z0-9_\-\s]+)',
                r'bom\s+([a-zA-Z0-9_\-]+)'
            ]

            for pattern in name_patterns:
                matches = re.findall(pattern, user_input, re.IGNORECASE)
                if matches:
                    parameters["bom_name"] = matches[0].strip()
                    break

        # Part number extraction for search
        if intent == "component_search":
            # Common part number patterns
            part_patterns = [
                r'\b[A-Z0-9]{3,}-[A-Z0-9]+\b',  # ABC123-XYZ
                r'\b[A-Z]{2,}[0-9]{2,}[A-Z]*\b'  # ABC123, ABC123X
            ]

            for pattern in part_patterns:
                matches = re.findall(pattern, user_input.upper())
                if matches:
                    parameters["part_numbers"] = matches
                    break

        return parameters

    def _check_context_requirements(self, context: Dict[str, Any], requirements: List[str]) -> bool:
        """Check if context meets requirements."""
        for req in requirements:
            if req not in context or not context[req]:
                return False
        return True

    def _map_ml_label_to_intent(self, ml_label: str) -> str:
        """Map ML model labels to our intent system."""
        # Mapping would depend on the specific ML model used
        # This is a placeholder implementation
        label_mapping = {
            "LABEL_0": "schematic_analysis",
            "LABEL_1": "component_search",
            "LABEL_2": "bom_creation",
            # Add more mappings based on your model
        }

        return label_mapping.get(ml_label, "general_query")

    def get_classification_report(self) -> Dict[str, Any]:
        """Get classification performance report."""
        return {
            "ml_models_available": self.use_ml_models,
            "total_intents": len(self.intent_patterns),
            "intent_categories": {
                category.value: [
                    intent for intent, config in self.intent_patterns.items()
                    if config["category"] == category
                ]
                for category in IntentCategory
            },
            "classification_methods": [
                "rule_based",
                "ml_based" if self.use_ml_models else None,
                "context_based"
            ]
        }
