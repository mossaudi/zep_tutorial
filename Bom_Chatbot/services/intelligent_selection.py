# Enhanced intelligent_selection.py
"""Enhanced intelligent tool selection with parametric search optimization."""

import json
import re
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
    """Enhanced conversation context with tool output analysis."""
    recent_messages: List[str]
    previous_tool_results: List[Dict[str, Any]]
    available_data: Dict[str, Any]
    last_tool_output: Optional[str] = None

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

        # Add last tool output analysis
        if self.last_tool_output:
            output_type = self._analyze_tool_output_type(self.last_tool_output)
            context_parts.append(f"Last output type: {output_type}")

        return " | ".join(context_parts) if context_parts else "No prior context"

    def _analyze_tool_output_type(self, output: str) -> str:
        """Analyze the type of tool output to suggest next actions."""
        if not output:
            return "empty"
        
        # Check for parametric search format
        if self._contains_parametric_format(output):
            return "parametric_search_ready"
        
        # Check for component data
        if "COMPONENT_SEARCH_COMPLETE:" in output:
            return "component_data_available"
        
        # Check for BOM data
        if any(keyword in output.lower() for keyword in ["bom", "bill of materials"]):
            return "bom_related"
        
        return "general_data"

    def _contains_parametric_format(self, output: str) -> bool:
        """Check if output contains parametric search format."""
        try:
            # Look for JSON with plName and selectedFilters structure
            if '"plName"' in output and '"selectedFilters"' in output:
                return True
            
            # Look for component data with technical specifications
            parametric_indicators = [
                '"fetName"',
                '"values"',
                'Package Type',
                'Supply Voltage',
                'MOSFETs',
                'Microcontrollers',
                'Maximum'
            ]
            
            return sum(1 for indicator in parametric_indicators if indicator in output) >= 3
        except:
            return False


class EnhancedIntelligentToolSelector:
    """Enhanced tool selector with parametric search optimization."""

    def __init__(self, llm: ChatGoogleGenerativeAI, available_tools: List[Tool]):
        self.llm = llm
        self.available_tools = available_tools
        self.progress = get_progress_tracker()
        
        # Build enhanced tool knowledge base
        self.tool_knowledge = self._build_enhanced_tool_knowledge()
        self.tool_relationships = self._build_tool_relationships()

    def _build_enhanced_tool_knowledge(self) -> str:
        """Build enhanced tool knowledge with use cases and relationships."""
        tool_info = []

        tool_enhancements = {
            'analyze_schematic': {
                'primary_use': 'Extract component data from schematic images',
                'output_format': 'JSON with plName and selectedFilters for parametric search',
                'next_tools': ['parametric_search', 'create_bom_from_schematic'],
                'confidence_indicators': ['image', 'schematic', 'circuit', 'analyze']
            },
            'parametric_search': {
                'primary_use': 'Search components by technical specifications using structured filters',
                'input_format': 'product_line and selectedFilters JSON array',
                'best_for': 'Technical component specifications, precise filtering',
                'confidence_indicators': ['specifications', 'technical', 'filter', 'parametric', 'plName', 'selectedFilters']
            },
            'search_component_data': {
                'primary_use': 'General component search using keywords and part numbers',
                'best_for': 'Known part numbers, basic component information',
                'confidence_indicators': ['part number', 'search', 'component', 'basic']
            },
            'create_bom_from_schematic': {
                'primary_use': 'Complete workflow from schematic to BOM',
                'best_for': 'End-to-end automation',
                'confidence_indicators': ['complete', 'workflow', 'end-to-end', 'automatic']
            }
        }

        for tool in self.available_tools:
            enhancement = tool_enhancements.get(tool.name, {})
            
            # Extract basic info
            tool_desc = {
                'name': tool.name,
                'description': tool.description,
                'parameters': self._extract_parameters_from_tool(tool)
            }

            # Add enhancement info
            param_info = ", ".join([f"{k}: {v}" for k, v in tool_desc['parameters'].items()])
            
            tool_info.append(
                f"Tool: {tool_desc['name']}\n"
                f"Description: {tool_desc['description']}\n"
                f"Parameters: {param_info}\n"
                f"Primary Use: {enhancement.get('primary_use', 'General tool')}\n"
                f"Best For: {enhancement.get('best_for', 'Various tasks')}\n"
                f"Input Format: {enhancement.get('input_format', 'Standard parameters')}\n"
                f"Output Format: {enhancement.get('output_format', 'Standard response')}\n"
                f"Recommended Next Tools: {', '.join(enhancement.get('next_tools', []))}\n"
            )

        return "\n" + "="*50 + "\n".join(tool_info)

    def _build_tool_relationships(self) -> Dict[str, List[str]]:
        """Build tool relationship mapping."""
        return {
            'analyze_schematic': ['parametric_search', 'create_bom_from_schematic'],
            'parametric_search': ['create_empty_bom', 'add_parts_to_bom'],
            'search_component_data': ['create_empty_bom', 'add_parts_to_bom'],
            'create_empty_bom': ['add_parts_to_bom'],
            'get_boms': ['add_parts_to_bom']
        }

    def select_tools(self, user_message: str, context: ConversationContext) -> List[ToolRecommendation]:
        """Enhanced tool selection with parametric search optimization."""
        self.progress.info("Enhanced Tool Selection", "Analyzing request with parametric search awareness...")

        # First, check if we have parametric-ready data from previous tool
        if self._should_use_parametric_search(user_message, context):
            return self._create_parametric_recommendations(user_message, context)

        # Use LLM for complex decisions
        recommendations = self._llm_tool_selection(user_message, context)
        
        # Post-process recommendations
        recommendations = self._optimize_tool_sequence(recommendations, context)
        
        return recommendations

    def _should_use_parametric_search(self, user_message: str, context: ConversationContext) -> bool:
        """Determine if parametric search should be prioritized."""
        
        # Check if last tool output contains parametric-ready data
        if context.last_tool_output and context._contains_parametric_format(context.last_tool_output):
            self.progress.info("Parametric Detection", "Found parametric-ready data in previous output")
            return True
        
        # Check if user message explicitly requests parametric search
        parametric_keywords = [
            'parametric', 'filter', 'specifications', 'technical specs',
            'plName', 'selectedFilters', 'fetName', 'values'
        ]
        
        return any(keyword in user_message.lower() for keyword in parametric_keywords)

    def _create_parametric_recommendations(self, user_message: str, context: ConversationContext) -> List[ToolRecommendation]:
        """Create parametric search recommendations."""
        recommendations = []
        
        # Extract parametric data from context if available
        parametric_data = self._extract_parametric_data(context.last_tool_output or "")
        
        if parametric_data:
            for component in parametric_data:
                recommendation = ToolRecommendation(
                    tool_name='parametric_search',
                    confidence=0.95,
                    reasoning=f"Found structured component data for {component.get('plName', 'component')} with technical specifications",
                    suggested_parameters=component,
                    execution_order=1
                )
                recommendations.append(recommendation)
        else:
            # Generic parametric search recommendation
            recommendation = ToolRecommendation(
                tool_name='parametric_search',
                confidence=0.8,
                reasoning="User request indicates need for parametric component search",
                suggested_parameters={},
                execution_order=1
            )
            recommendations.append(recommendation)
        
        return recommendations

    def _extract_parametric_data(self, output: str) -> List[Dict[str, Any]]:
        """Extract parametric search data from tool output."""
        try:
            # Look for JSON data with parametric structure
            json_match = re.search(r'\{.*"components":\s*\[(.*?)\]', output, re.DOTALL)
            if json_match:
                # Try to parse the full JSON
                try:
                    data = json.loads(output[json_match.start():])
                    if 'components' in data:
                        components = data['components']
                        parametric_components = []
                        
                        for comp in components:
                            if 'plName' in comp and 'selectedFilters' in comp:
                                parametric_components.append({
                                    'product_line': comp['plName'],
                                    'selected_filters': json.dumps(comp['selectedFilters'])
                                })
                        
                        return parametric_components
                except json.JSONDecodeError:
                    pass
            
            return []
        except Exception as e:
            self.progress.warning("Parametric Extraction", f"Failed to extract parametric data: {str(e)}")
            return []

    def _llm_tool_selection(self, user_message: str, context: ConversationContext) -> List[ToolRecommendation]:
        """LLM-based tool selection with enhanced prompting."""
        
        selection_prompt = f"""
            You are an intelligent tool selector for a BOM management system with SPECIAL EXPERTISE in parametric component searching.
            
            USER REQUEST: "{user_message}"
            
            CONVERSATION CONTEXT: {context.to_context_string()}
            
            AVAILABLE TOOLS:
            {self.tool_knowledge}
            
            CRITICAL PARAMETRIC SEARCH RULES:
            1. If you see JSON with "plName" and "selectedFilters" structure, ALWAYS prioritize parametric_search
            2. If previous output contains technical specifications (voltage ranges, package types, etc.), use parametric_search
            3. If user mentions component categories (MOSFETs, Microcontrollers, etc.), use parametric_search
            4. Only use search_component_data for basic part number lookups or when no technical specifications are available
            
            TOOL SELECTION PRIORITY:
            1. analyze_schematic → parametric_search (if technical specs available)
            2. analyze_schematic → create_bom_from_schematic (for complete workflow)
            3. parametric_search → create_empty_bom/add_parts_to_bom
            4. search_component_data (only for basic searches)
            
            RESPONSE FORMAT (JSON):
            {{
              "recommendations": [
                {{
                  "tool_name": "exact_tool_name",
                  "confidence": 0.9,
                  "reasoning": "Specific reason focusing on parametric search optimization",
                  "suggested_parameters": {{"param1": "value1"}},
                  "execution_order": 1
                }}
              ]
            }}
            
            Analyze and respond with JSON only.
            """

        try:
            response = self.llm.invoke([HumanMessage(content=selection_prompt)])
            return self._parse_llm_response(response.content)
        except Exception as e:
            self.progress.error("LLM Tool Selection", str(e))
            return self._fallback_selection_with_parametric(user_message, context)

    def _optimize_tool_sequence(self, recommendations: List[ToolRecommendation], 
                               context: ConversationContext) -> List[ToolRecommendation]:
        """Optimize tool sequence based on relationships and context."""
        
        if not recommendations:
            return recommendations
        
        # If we have parametric-ready data but LLM didn't choose parametric search, inject it
        if (context.last_tool_output and 
            context._contains_parametric_format(context.last_tool_output) and
            not any(rec.tool_name == 'parametric_search' for rec in recommendations)):
            
            parametric_rec = ToolRecommendation(
                tool_name='parametric_search',
                confidence=0.9,
                reasoning="Injected: Previous output contains parametric-ready component data",
                suggested_parameters={},
                execution_order=1
            )
            
            # Insert at the beginning
            recommendations.insert(0, parametric_rec)
            # Re-order execution sequence
            for i, rec in enumerate(recommendations[1:], 2):
                rec.execution_order = i
        
        return recommendations

    def _fallback_selection_with_parametric(self, user_message: str, 
                                          context: ConversationContext) -> List[ToolRecommendation]:
        """Enhanced fallback with parametric search awareness."""
        message_lower = user_message.lower()
        recommendations = []

        # Parametric search indicators
        parametric_keywords = ['parametric', 'filter', 'specs', 'technical', 'mosfet', 'microcontroller']
        if any(keyword in message_lower for keyword in parametric_keywords):
            recommendations.append(ToolRecommendation(
                tool_name='parametric_search',
                confidence=0.8,
                reasoning="Fallback: Detected parametric search keywords",
                suggested_parameters={},
                execution_order=1
            ))

        # Standard fallback mappings
        keyword_mappings = {
            'analyze_schematic': ['analyze', 'schematic', 'image', 'circuit'],
            'search_component_data': ['search', 'component', 'part', 'find'],
            'create_empty_bom': ['create', 'new', 'empty', 'bom'],
            'get_boms': ['get', 'list', 'show', 'existing', 'boms']
        }

        for tool_name, keywords in keyword_mappings.items():
            if any(keyword in message_lower for keyword in keywords):
                recommendations.append(ToolRecommendation(
                    tool_name=tool_name,
                    confidence=0.6,
                    reasoning=f"Fallback: Keyword match for {tool_name}",
                    suggested_parameters={},
                    execution_order=len(recommendations) + 1
                ))

        return recommendations[:3]  # Limit fallback results

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