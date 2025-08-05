# learning/adaptive_responses.py
"""Adaptive learning system for intelligent agent responses."""

import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

from Bom_Chatbot.services.progress import get_progress_tracker


@dataclass
class UserInteraction:
    """Single user interaction record."""
    timestamp: datetime
    user_input: str
    agent_response: str
    user_feedback: Optional[str] = None  # positive, negative, neutral
    success_indicators: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResponsePattern:
    """Pattern for successful response strategies."""
    pattern_id: str
    input_pattern: str
    response_template: str
    success_rate: float
    usage_count: int
    last_used: datetime
    context_requirements: List[str] = field(default_factory=list)


class AdaptiveLearningSystem:
    """System that learns from user interactions to improve responses."""

    def __init__(self, learning_window_days: int = 30):
        self.learning_window_days = learning_window_days
        self.progress = get_progress_tracker()

        # Storage for learning data
        self.interaction_history: List[UserInteraction] = []
        self.successful_patterns: Dict[str, ResponsePattern] = {}
        self.failed_patterns: Dict[str, int] = defaultdict(int)

        # Learning parameters
        self.min_pattern_usage = 3  # Minimum usage before considering pattern reliable
        self.pattern_confidence_threshold = 0.7

        # Initialize with some base patterns
        self._initialize_base_patterns()

    def _initialize_base_patterns(self):
        """Initialize with successful patterns from domain knowledge."""
        base_patterns = [
            {
                "pattern_id": "schematic_analysis_request",
                "input_pattern": r"analyz\w*.*schematic|extract.*component.*from.*image",
                "response_template": "I'll analyze your schematic image. Please provide the URL and I'll extract all components with their specifications.",
                "success_rate": 0.9,
                "usage_count": 10,
                "context_requirements": ["image_url"]
            },
            {
                "pattern_id": "bom_creation_followup",
                "input_pattern": r"create.*bom|make.*bom",
                "response_template": "I'll create a BOM for you. Would you like me to use the components from your recent schematic analysis, or create an empty BOM structure?",
                "success_rate": 0.85,
                "usage_count": 8,
                "context_requirements": ["session_components"]
            },
            {
                "pattern_id": "component_search_request",
                "input_pattern": r"search.*component|find.*part.*number",
                "response_template": "I'll search for component data. Please provide the part number, manufacturer, or component description.",
                "success_rate": 0.8,
                "usage_count": 5,
                "context_requirements": []
            }
        ]

        for pattern_data in base_patterns:
            pattern = ResponsePattern(
                pattern_id=pattern_data["pattern_id"],
                input_pattern=pattern_data["input_pattern"],
                response_template=pattern_data["response_template"],
                success_rate=pattern_data["success_rate"],
                usage_count=pattern_data["usage_count"],
                last_used=datetime.now() - timedelta(days=1),
                context_requirements=pattern_data["context_requirements"]
            )
            self.successful_patterns[pattern.pattern_id] = pattern

    def record_interaction(self, user_input: str, agent_response: str,
                           context: Dict[str, Any] = None,
                           success_indicators: Dict[str, Any] = None):
        """Record a user-agent interaction for learning."""
        interaction = UserInteraction(
            timestamp=datetime.now(),
            user_input=user_input,
            agent_response=agent_response,
            context=context or {},
            success_indicators=success_indicators or {}
        )

        self.interaction_history.append(interaction)

        # Cleanup old interactions
        cutoff_date = datetime.now() - timedelta(days=self.learning_window_days)
        self.interaction_history = [i for i in self.interaction_history if i.timestamp > cutoff_date]

        self.progress.info("Learning", f"Recorded interaction: {user_input[:50]}...")

    def record_feedback(self, interaction_index: int, feedback: str):
        """Record user feedback on agent response."""
        if 0 <= interaction_index < len(self.interaction_history):
            self.interaction_history[interaction_index].user_feedback = feedback
            self._update_patterns_from_feedback(self.interaction_history[interaction_index])

    def suggest_response(self, user_input: str, context: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """Suggest response based on learned patterns."""
        context = context or {}

        # Find matching patterns
        matching_patterns = self._find_matching_patterns(user_input, context)

        if not matching_patterns:
            return None

        # Get best pattern
        best_pattern = max(matching_patterns, key=lambda p: p.success_rate * p.usage_count)

        # Check if pattern meets confidence threshold
        if (best_pattern.success_rate >= self.pattern_confidence_threshold and
                best_pattern.usage_count >= self.min_pattern_usage):
            # Update usage
            best_pattern.usage_count += 1
            best_pattern.last_used = datetime.now()

            return {
                "suggested_response": best_pattern.response_template,
                "confidence": best_pattern.success_rate,
                "pattern_id": best_pattern.pattern_id,
                "usage_count": best_pattern.usage_count
            }

        return None

    def _find_matching_patterns(self, user_input: str, context: Dict[str, Any]) -> List[ResponsePattern]:
        """Find patterns that match the user input and context."""
        matching = []
        user_input_lower = user_input.lower()

        for pattern in self.successful_patterns.values():
            # Check input pattern match
            if re.search(pattern.input_pattern, user_input_lower):
                # Check context requirements
                if self._context_meets_requirements(context, pattern.context_requirements):
                    matching.append(pattern)

        return matching

    def _context_meets_requirements(self, context: Dict[str, Any], requirements: List[str]) -> bool:
        """Check if context meets pattern requirements."""
        for req in requirements:
            if req not in context or not context[req]:
                return False
        return True

    def _update_patterns_from_feedback(self, interaction: UserInteraction):
        """Update patterns based on user feedback."""
        if not interaction.user_feedback:
            return

        feedback_positive = interaction.user_feedback in ['positive', 'good', 'helpful']

        # Try to identify which pattern was used
        matching_patterns = self._find_matching_patterns(interaction.user_input, interaction.context)

        for pattern in matching_patterns:
            if feedback_positive:
                # Increase success rate
                total_interactions = pattern.usage_count
                current_successes = pattern.success_rate * total_interactions
                new_success_rate = (current_successes + 1) / (total_interactions + 1)
                pattern.success_rate = min(new_success_rate, 1.0)
            else:
                # Decrease success rate
                total_interactions = pattern.usage_count
                current_successes = pattern.success_rate * total_interactions
                new_success_rate = current_successes / (total_interactions + 1)
                pattern.success_rate = max(new_success_rate, 0.0)

    def learn_new_patterns(self):
        """Analyze recent interactions to discover new successful patterns."""
        self.progress.info("Pattern Learning", "Analyzing interactions for new patterns...")

        # Get recent successful interactions
        recent_cutoff = datetime.now() - timedelta(days=7)
        recent_interactions = [
            i for i in self.interaction_history
            if i.timestamp > recent_cutoff and
               (i.user_feedback == 'positive' or
                i.success_indicators.get('task_completed', False))
        ]

        if len(recent_interactions) < 3:
            return

        # Group by similar input patterns
        pattern_groups = self._group_similar_interactions(recent_interactions)

        # Create new patterns for frequent successful groups
        for group_key, interactions in pattern_groups.items():
            if len(interactions) >= 3:  # Minimum frequency
                self._create_pattern_from_group(group_key, interactions)

    def _group_similar_interactions(self, interactions: List[UserInteraction]) -> Dict[str, List[UserInteraction]]:
        """Group interactions by similar input patterns."""
        groups = defaultdict(list)

        for interaction in interactions:
            # Extract key terms from user input
            key_terms = self._extract_key_terms(interaction.user_input)
            group_key = "_".join(sorted(key_terms[:3]))  # Use top 3 terms
            groups[group_key].append(interaction)

        return groups

    def _extract_key_terms(self, text: str) -> List[str]:
        """Extract key terms from text for pattern grouping."""
        # Simple extraction - in production, use more sophisticated NLP
        text_lower = text.lower()

        # Domain-specific important terms
        important_terms = [
            'schematic', 'component', 'bom', 'part', 'search', 'create',
            'analyze', 'add', 'list', 'find', 'extract', 'manufacturer'
        ]

        found_terms = []
        for term in important_terms:
            if term in text_lower:
                found_terms.append(term)

        return found_terms

    def _create_pattern_from_group(self, group_key: str, interactions: List[UserInteraction]):
        """Create a new pattern from a group of similar successful interactions."""
        if len(interactions) < 3:
            return

        # Generate pattern ID
        pattern_id = f"learned_{group_key}_{int(datetime.now().timestamp())}"

        # Create input pattern (simplified - use first interaction as template)
        first_input = interactions[0].user_input.lower()
        key_terms = self._extract_key_terms(first_input)
        input_pattern = "|".join([re.escape(term) for term in key_terms])

        # Create response template (use most common response structure)
        response_template = self._generate_response_template(interactions)

        # Calculate initial success rate
        success_rate = 1.0  # Start optimistic since these were successful

        # Identify common context requirements
        context_requirements = self._identify_context_requirements(interactions)

        new_pattern = ResponsePattern(
            pattern_id=pattern_id,
            input_pattern=input_pattern,
            response_template=response_template,
            success_rate=success_rate,
            usage_count=len(interactions),
            last_used=max(i.timestamp for i in interactions),
            context_requirements=context_requirements
        )

        self.successful_patterns[pattern_id] = new_pattern

        self.progress.success("Pattern Learning",
                              f"Created new pattern '{pattern_id}' from {len(interactions)} interactions")

    def _generate_response_template(self, interactions: List[UserInteraction]) -> str:
        """Generate response template from successful interactions."""
        # Simplified - in production, use more sophisticated template generation
        responses = [i.agent_response for i in interactions]

        # Find common response structure
        if len(responses) > 0:
            # For now, return the most common response
            response_counts = defaultdict(int)
            for response in responses:
                # Normalize response for comparison
                normalized = re.sub(r'\b\d+\b', 'NUMBER', response.lower())
                normalized = re.sub(r'https?://\S+', 'URL', normalized)
                response_counts[normalized] += 1

            most_common = max(response_counts.items(), key=lambda x: x[1])
            return most_common[0]

        return "I'll help you with that request."

    def _identify_context_requirements(self, interactions: List[UserInteraction]) -> List[str]:
        """Identify common context requirements from successful interactions."""
        context_keys = defaultdict(int)

        for interaction in interactions:
            for key in interaction.context.keys():
                if interaction.context[key]:  # Only count non-empty context
                    context_keys[key] += 1

        # Return context keys that appear in most interactions
        threshold = len(interactions) * 0.7  # 70% threshold
        requirements = [key for key, count in context_keys.items() if count >= threshold]

        return requirements

    def get_learning_report(self) -> Dict[str, Any]:
        """Generate report on learning progress and patterns."""
        total_patterns = len(self.successful_patterns)
        learned_patterns = len([p for p in self.successful_patterns.values()
                                if p.pattern_id.startswith('learned_')])

        # Calculate average success rates
        success_rates = [p.success_rate for p in self.successful_patterns.values()]
        avg_success_rate = sum(success_rates) / len(success_rates) if success_rates else 0

        # Most used patterns
        most_used = sorted(
            self.successful_patterns.values(),
            key=lambda p: p.usage_count,
            reverse=True
        )[:5]

        return {
            "total_patterns": total_patterns,
            "base_patterns": total_patterns - learned_patterns,
            "learned_patterns": learned_patterns,
            "average_success_rate": round(avg_success_rate, 3),
            "total_interactions": len(self.interaction_history),
            "learning_window_days": self.learning_window_days,
            "most_used_patterns": [
                {
                    "pattern_id": p.pattern_id,
                    "usage_count": p.usage_count,
                    "success_rate": p.success_rate
                }
                for p in most_used
            ],
            "recent_feedback": self._get_recent_feedback_summary()
        }

    def _get_recent_feedback_summary(self) -> Dict[str, int]:
        """Get summary of recent user feedback."""
        recent_cutoff = datetime.now() - timedelta(days=7)
        recent_feedback = [
            i.user_feedback for i in self.interaction_history
            if i.timestamp > recent_cutoff and i.user_feedback
        ]

        feedback_counts = defaultdict(int)
        for feedback in recent_feedback:
            feedback_counts[feedback] += 1

        return dict(feedback_counts)
