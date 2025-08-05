# services/memory_manager.py
"""Enhanced memory and context management."""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json


@dataclass
class UserPreference:
    """User preference data structure."""
    preferred_columns: List[str] = field(default_factory=list)
    default_project_path: str = ""
    preferred_manufacturers: List[str] = field(default_factory=list)
    search_history_limit: int = 50


@dataclass
class ProjectContext:
    """Current project context."""
    active_project: Optional[str] = None
    recent_boms: List[str] = field(default_factory=list)
    component_library: List[Dict[str, Any]] = field(default_factory=list)
    last_schematic_url: Optional[str] = None


@dataclass
class InteractionHistory:
    """Track user interaction patterns."""
    timestamp: datetime
    interaction_type: str  # 'schematic_analysis', 'bom_creation', 'component_search'
    input_data: Dict[str, Any]
    success: bool
    execution_time: float


class ConversationMemory:
    """Enhanced conversation memory with persistence."""

    def __init__(self, max_history: int = 100):
        self.user_preferences = UserPreference()
        self.project_context = ProjectContext()
        self.interaction_history: List[InteractionHistory] = []
        self.max_history = max_history
        self.session_components: List[Dict[str, Any]] = []

    def update_context(self, interaction_type: str, data: Dict[str, Any],
                       success: bool = True, execution_time: float = 0.0):
        """Update conversation context based on user interactions."""

        # Add to history
        interaction = InteractionHistory(
            timestamp=datetime.now(),
            interaction_type=interaction_type,
            input_data=data.copy(),
            success=success,
            execution_time=execution_time
        )

        self.interaction_history.append(interaction)

        # Trim history if needed
        if len(self.interaction_history) > self.max_history:
            self.interaction_history = self.interaction_history[-self.max_history:]

        # Update context based on interaction type
        if interaction_type == "schematic_analysis" and success:
            self.project_context.last_schematic_url = data.get('image_url')
            if 'components' in data:
                self.session_components.extend(data['components'])

        elif interaction_type == "bom_creation" and success:
            bom_name = data.get('name')
            if bom_name and bom_name not in self.project_context.recent_boms:
                self.project_context.recent_boms.insert(0, bom_name)
                # Keep only last 10 BOMs
                self.project_context.recent_boms = self.project_context.recent_boms[:10]

    def get_context_summary(self) -> Dict[str, Any]:
        """Get current context summary for LLM."""
        recent_interactions = self.interaction_history[-5:] if self.interaction_history else []

        return {
            "recent_activity": [
                {
                    "type": i.interaction_type,
                    "timestamp": i.timestamp.isoformat(),
                    "success": i.success
                } for i in recent_interactions
            ],
            "current_project": self.project_context.active_project,
            "recent_boms": self.project_context.recent_boms[:3],
            "session_components_count": len(self.session_components),
            "last_schematic": self.project_context.last_schematic_url
        }

    def suggest_next_actions(self) -> List[str]:
        """Suggest next actions based on context."""
        suggestions = []

        if self.session_components and not self.project_context.recent_boms:
            suggestions.append("Create a BOM from your analyzed components")

        if self.project_context.last_schematic_url and not self.session_components:
            suggestions.append("Re-analyze the last schematic for component updates")

        if len(self.interaction_history) > 5:
            failed_interactions = [i for i in self.interaction_history[-5:] if not i.success]
            if failed_interactions:
                suggestions.append("Review failed operations and retry")

        return suggestions
