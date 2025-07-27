"""Intelligent tool selection using LLM reasoning instead of manual mapping."""

import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from langchain.tools import Tool
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from Bom_Chatbot.services.progress import get_progress_tracker


@dataclass
class ToolRecommendation:
    """A tool recommendation with reasoning."""
    tool_name: str
    confidence: float
    reasoning: str
    suggested_parameters: Dict[str, Any]
    execution_order: int = 1


@dataclass
class ConversationContext:
    """Lightweight conversation context."""
    recent_messages: List[str]
    previous_tool_results: List[Dict[str, Any]]
    available_data: Dict[str, Any]  # Components, BOMs, etc.

    def to_context_string(self) -> str:
        """Convert context to string for LLM."""
        context_parts = []

        if self.recent_messages:
            context_parts.append(f"Recent conversation: {' -> '.join(self.recent_messages[-3:])}")

        if self.available_data:
            data_summary = []
            for key, value in self.available_data.items():
                if isinstance(value, list):
                    data_summary.append(f"{key}: {len(value)} items")
                else:
                    data_summary.append(f"{key}: {str(value)[:50]}...")
            context_parts.append(f"Available data: {', '.join(data_summary)}")

        if self.previous_tool_results:
            context_parts.append(f"Previous results: {len(self.previous_tool_results)} tool executions")

        return " | ".join(context_parts) if context_parts else "No prior context"


class IntelligentToolSelector:
    """Uses LLM to intelligently select tools based on descriptions and context."""

    def __init__(self, llm: ChatGoogleGenerativeAI, available_tools: List[Tool]):
        self.llm = llm
        self.available_tools = available_tools
        self.progress = get_progress_tracker()

        # Build tool knowledge base from descriptions
        self.tool_knowledge = self._build_tool_knowledge()

    def _build_tool_knowledge(self) -> str:
        """Extract tool information from their actual descriptions."""
        tool_info = []

        for tool in self.available_tools:
            # Extract info from tool
            tool_desc = {
                'name': tool.name,
                'description': tool.description,
                'parameters': self._extract_parameters_from_tool(tool)
            }

            # Format for LLM understanding
            param_info = ", ".join([f"{k}: {v}" for k, v in tool_desc['parameters'].items()])
            tool_info.append(
                f"Tool: {tool_desc['name']}\n"
                f"Description: {tool_desc['description']}\n"
                f"Parameters: {param_info}\n"
            )

        return "\n".join(tool_info)

    def _extract_parameters_from_tool(self, tool: Tool) -> Dict[str, str]:
        """Extract parameter information from tool definition."""
        parameters = {}

        # Try to get from tool's args_schema if available
        if hasattr(tool, 'args_schema') and tool.args_schema:
            try:
                schema = tool.args_schema.schema()
                properties = schema.get('properties', {})
                for param_name, param_info in properties.items():
                    param_type = param_info.get('type', 'string')
                    param_desc = param_info.get('description', '')
                    parameters[param_name] = f"{param_type} - {param_desc}"
            except:
                pass

        # Fallback: parse from docstring
        if not parameters and hasattr(tool, 'func') and tool.func.__doc__:
            parameters = self._parse_docstring_params(tool.func.__doc__)

        return parameters

    def _parse_docstring_params(self, docstring: str) -> Dict[str, str]:
        """Extract parameters from function docstring."""
        parameters = {}
        if not docstring:
            return parameters

        lines = docstring.split('\n')
        in_args_section = False

        for line in lines:
            line = line.strip()
            if line.lower().startswith('args:'):
                in_args_section = True
                continue
            elif line.lower().startswith(('returns:', 'return:')):
                break
            elif in_args_section and ':' in line:
                parts = line.split(':', 1)
                if len(parts) == 2:
                    param_name = parts[0].strip()
                    param_desc = parts[1].strip()
                    parameters[param_name] = param_desc

        return parameters

    def select_tools(self, user_message: str, context: ConversationContext) -> List[ToolRecommendation]:
        """Intelligently select tools using LLM reasoning."""
        self.progress.info("Intelligent Selection", "Analyzing user request...")

        # Create the selection prompt
        selection_prompt = self._create_selection_prompt(user_message, context)

        try:
            # Get LLM recommendation
            response = self.llm.invoke([HumanMessage(content=selection_prompt)])

            # Parse LLM response
            recommendations = self._parse_llm_response(response.content)

            self.progress.success(
                "Tool Selection",
                f"Selected {len(recommendations)} tools: {[r.tool_name for r in recommendations[:3]]}"
            )

            return recommendations

        except Exception as e:
            self.progress.error("Intelligent Selection", str(e))
            # Fallback to simple keyword matching
            return self._fallback_selection(user_message)

    def _create_selection_prompt(self, user_message: str, context: ConversationContext) -> str:
        """Create a comprehensive prompt for tool selection."""

        return f"""
You are an intelligent tool selector for a BOM (Bill of Materials) management system. 
Your job is to analyze the user's request and recommend the most appropriate tools.

USER REQUEST: "{user_message}"

CONVERSATION CONTEXT: {context.to_context_string()}

AVAILABLE TOOLS:
{self.tool_knowledge}

INSTRUCTIONS:
1. Analyze the user's intent and requirements
2. Consider the conversation context and available data
3. Select 1-3 most appropriate tools in logical execution order
4. For each selected tool, provide:
   - Tool name
   - Confidence score (0.0-1.0)
   - Clear reasoning for selection
   - Suggested parameter values (if extractable from user message)

RESPONSE FORMAT (JSON):
{{
  "recommendations": [
    {{
      "tool_name": "exact_tool_name",
      "confidence": 0.9,
      "reasoning": "Why this tool is appropriate",
      "suggested_parameters": {{"param1": "value1", "param2": "value2"}},
      "execution_order": 1
    }}
  ]
}}

IMPORTANT:
- Only recommend tools that directly address the user's request
- Consider the logical workflow sequence
- Extract parameter values from the user message when possible
- Higher confidence for exact matches, lower for inferential matches
- If unclear, ask for clarification rather than guessing

Respond with ONLY the JSON, no additional text.
"""

    def _parse_llm_response(self, response_content: str) -> List[ToolRecommendation]:
        """Parse LLM response into tool recommendations."""
        try:
            # Clean the response
            cleaned_response = response_content.strip()
            if cleaned_response.startswith('```json'):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.endswith('```'):
                cleaned_response = cleaned_response[:-3]

            # Parse JSON
            parsed = json.loads(cleaned_response)
            recommendations = []

            for rec_data in parsed.get('recommendations', []):
                # Validate tool exists
                tool_name = rec_data.get('tool_name')
                if not any(tool.name == tool_name for tool in self.available_tools):
                    self.progress.warning("Tool Validation", f"Unknown tool: {tool_name}")
                    continue

                recommendation = ToolRecommendation(
                    tool_name=tool_name,
                    confidence=float(rec_data.get('confidence', 0.5)),
                    reasoning=rec_data.get('reasoning', ''),
                    suggested_parameters=rec_data.get('suggested_parameters', {}),
                    execution_order=int(rec_data.get('execution_order', 1))
                )
                recommendations.append(recommendation)

            # Sort by execution order
            recommendations.sort(key=lambda x: x.execution_order)
            return recommendations

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            self.progress.error("Response Parsing", f"Failed to parse LLM response: {str(e)}")
            return []

    def _fallback_selection(self, user_message: str) -> List[ToolRecommendation]:
        """Simple fallback selection using keyword matching."""
        message_lower = user_message.lower()
        recommendations = []

        # Simple keyword-based fallback
        keyword_mappings = {
            'analyze_schematic': ['analyze', 'schematic', 'image', 'circuit'],
            'search_component_data': ['search', 'component', 'part', 'find'],
            'create_empty_bom': ['create', 'new', 'empty', 'bom'],
            'get_boms': ['get', 'list', 'show', 'existing', 'boms'],
            'parametric_search': ['parametric', 'filter', 'specifications']
        }

        for tool_name, keywords in keyword_mappings.items():
            if any(keyword in message_lower for keyword in keywords):
                recommendations.append(ToolRecommendation(
                    tool_name=tool_name,
                    confidence=0.6,
                    reasoning=f"Keyword match for {tool_name}",
                    suggested_parameters={},
                    execution_order=1
                ))

        return recommendations[:2]  # Limit fallback results