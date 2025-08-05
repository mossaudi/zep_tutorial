# llm/router.py
"""Intelligent LLM routing for cost and performance optimization."""

import asyncio
from typing import Any, Dict, Optional, Union
from enum import Enum
from dataclasses import dataclass
import re

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaLLM

from Bom_Chatbot.services.progress import get_progress_tracker


class TaskComplexity(Enum):
    """Task complexity levels for LLM routing."""
    SIMPLE = "simple"  # Rule-based, no LLM needed
    LIGHT = "light"  # Local LLM sufficient
    MEDIUM = "medium"  # Efficient cloud LLM
    COMPLEX = "complex"  # Advanced reasoning required
    VISION = "vision"  # Image analysis required


@dataclass
class LLMConfig:
    """Configuration for different LLM providers."""
    name: str
    model: str
    cost_per_1k_tokens: float
    max_tokens: int
    temperature: float
    capabilities: list


class TaskClassifier:
    """Classify tasks to determine optimal LLM routing."""

    def __init__(self):
        self.patterns = {
            TaskComplexity.SIMPLE: [
                r'^(list|show|get|display)\s+(bom|component)',
                r'^(create|add)\s+empty\s+bom',
                r'^\w+\s*=\s*\w+',  # Simple assignments
            ],
            TaskComplexity.VISION: [
                r'analyz\w*\s+(schematic|image|circuit)',
                r'extract\s+(component|part).*from.*image',
                r'(schematic|circuit|diagram).*analysis',
            ],
            TaskComplexity.LIGHT: [
                r'search\s+(component|part)',
                r'find\s+(datasheet|specification)',
                r'(validate|check)\s+part\s+number',
            ],
            TaskComplexity.MEDIUM: [
                r'compare\s+(component|alternative)',
                r'suggest\s+(replacement|alternative)',
                r'optimization.*bom',
            ],
            TaskComplexity.COMPLEX: [
                r'design\s+(recommendation|analysis)',
                r'cost\s+analysis.*optimization',
                r'supply\s+chain.*analysis',
            ]
        }

    def classify(self, user_input: str, context: Dict[str, Any] = None) -> TaskComplexity:
        """Classify task complexity based on user input and context."""
        user_input_lower = user_input.lower()

        # Check patterns in order of complexity
        for complexity, patterns in self.patterns.items():
            for pattern in patterns:
                if re.search(pattern, user_input_lower):
                    return complexity

        # Context-based classification
        if context:
            if 'image_url' in context or 'schematic' in user_input_lower:
                return TaskComplexity.VISION

            if len(user_input.split()) > 20:  # Long complex queries
                return TaskComplexity.COMPLEX

        # Default to medium complexity
        return TaskComplexity.MEDIUM


class IntelligentLLMRouter:
    """Routes tasks to optimal LLM based on complexity and cost."""

    def __init__(self, config: Dict[str, Any]):
        self.progress = get_progress_tracker()
        self.classifier = TaskClassifier()
        self.usage_stats = {"calls": 0, "total_cost": 0.0, "tokens_used": 0}

        # Initialize LLM configurations
        self.llm_configs = {
            TaskComplexity.VISION: LLMConfig(
                name="gemini-vision",
                model="gemini-2.0-flash-lite",
                cost_per_1k_tokens=0.0025,  # Vision pricing
                max_tokens=30000,
                temperature=0.1,
                capabilities=["vision", "reasoning"]
            ),
            TaskComplexity.COMPLEX: LLMConfig(
                name="gemini-2.0-flash-lite",
                model="gemini-2.0-flash-lite",
                cost_per_1k_tokens=0.005,
                max_tokens=4096,
                temperature=0.2,
                capabilities=["reasoning", "analysis"]
            ),
            TaskComplexity.MEDIUM: LLMConfig(
                name="gpt-3.5-turbo",
                model="gpt-3.5-turbo",
                cost_per_1k_tokens=0.00015,
                max_tokens=4096,
                temperature=0.1,
                capabilities=["reasoning"]
            ),
            TaskComplexity.LIGHT: LLMConfig(
                name="llama-local",
                model="llama3.1:8b",
                cost_per_1k_tokens=0.0,  # Local model
                max_tokens=2048,
                temperature=0.0,
                capabilities=["text"]
            )
        }

        # Initialize LLM instances
        self._initialize_llms(config)

    def _initialize_llms(self, config: Dict[str, Any]):
        """Initialize LLM instances based on configuration."""
        self.llms = {}

        try:
            # Gemini for vision tasks
            self.llms[TaskComplexity.VISION] = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash-lite",
                google_api_key=config.get("google_api_key"),
                temperature=0.1,
                max_output_tokens=30000
            )

            # OpenAI models for reasoning
            openai_key = config.get("openai_api_key")
            if openai_key:
                self.llms[TaskComplexity.COMPLEX] = ChatOpenAI(
                    model="gpt-3.5-turbo",
                    api_key=openai_key,
                    temperature=0.2,
                    max_tokens=4096
                )

                self.llms[TaskComplexity.MEDIUM] = ChatOpenAI(
                    model="gpt-3.5-turbo",
                    api_key=openai_key,
                    temperature=0.1,
                    max_tokens=4096
                )

            # Local Ollama for light tasks
            try:
                self.llms[TaskComplexity.LIGHT] = OllamaLLM(
                    model="llama3.1:8b",
                    temperature=0.0
                )
            except Exception as e:
                self.progress.warning("LLM Router", f"Local Ollama not available: {e}")
                # Fallback to OpenAI mini for light tasks
                if openai_key:
                    self.llms[TaskComplexity.LIGHT] = self.llms[TaskComplexity.MEDIUM]

        except Exception as e:
            self.progress.error("LLM Router", f"Failed to initialize LLMs: {e}")
            raise

    async def route_and_execute(self, user_input: str, context: Dict[str, Any] = None,
                                messages: list = None) -> Dict[str, Any]:
        """Route task to optimal LLM and execute."""

        # Classify task complexity
        complexity = self.classifier.classify(user_input, context)

        # Handle simple tasks without LLM
        if complexity == TaskComplexity.SIMPLE:
            return await self._handle_simple_task(user_input, context)

        # Get appropriate LLM
        llm = self._get_llm_for_complexity(complexity)
        if not llm:
            # Fallback to any available LLM
            llm = self._get_fallback_llm()

        # Track usage and cost
        config = self.llm_configs[complexity]
        self.progress.info("LLM Router", f"Using {config.name} for {complexity.value} task")

        try:
            # Execute with selected LLM
            if messages:
                response = await llm.ainvoke(messages)
            else:
                response = await llm.ainvoke(user_input)

            # Update usage statistics
            estimated_tokens = len(user_input.split()) * 1.3  # Rough estimate
            estimated_cost = (estimated_tokens / 1000) * config.cost_per_1k_tokens

            self.usage_stats["calls"] += 1
            self.usage_stats["total_cost"] += estimated_cost
            self.usage_stats["tokens_used"] += estimated_tokens

            return {
                "response": response,
                "llm_used": config.name,
                "complexity": complexity.value,
                "estimated_cost": estimated_cost,
                "estimated_tokens": estimated_tokens
            }

        except Exception as e:
            self.progress.error("LLM Execution", f"Failed with {config.name}: {e}")

            # Try fallback LLM
            fallback_llm = self._get_fallback_llm()
            if fallback_llm and fallback_llm != llm:
                self.progress.info("LLM Router", "Attempting fallback LLM")
                response = await fallback_llm.ainvoke(messages or user_input)
                return {
                    "response": response,
                    "llm_used": "fallback",
                    "complexity": complexity.value,
                    "estimated_cost": 0.001,  # Fallback cost
                    "estimated_tokens": estimated_tokens
                }

            raise

    async def _handle_simple_task(self, user_input: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle simple tasks without LLM."""
        self.progress.info("Simple Handler", "Processing without LLM")

        # Pattern matching for common simple tasks
        user_lower = user_input.lower()

        if 'list bom' in user_lower or 'show bom' in user_lower:
            return {
                "response": "DIRECT_HANDLER:list_boms",
                "llm_used": "none",
                "complexity": "simple",
                "estimated_cost": 0.0,
                "estimated_tokens": 0
            }

        return {
            "response": f"Simple task identified: {user_input}",
            "llm_used": "pattern_matching",
            "complexity": "simple",
            "estimated_cost": 0.0,
            "estimated_tokens": 0
        }

    def _get_llm_for_complexity(self, complexity: TaskComplexity):
        """Get LLM instance for given complexity."""
        return self.llms.get(complexity)

    def _get_fallback_llm(self):
        """Get any available LLM as fallback."""
        for complexity in [TaskComplexity.MEDIUM, TaskComplexity.LIGHT, TaskComplexity.COMPLEX]:
            if complexity in self.llms:
                return self.llms[complexity]
        return None

    def get_usage_report(self) -> Dict[str, Any]:
        """Get usage statistics and cost report."""
        return {
            "total_calls": self.usage_stats["calls"],
            "total_cost": round(self.usage_stats["total_cost"], 4),
            "total_tokens": int(self.usage_stats["tokens_used"]),
            "average_cost_per_call": round(
                self.usage_stats["total_cost"] / max(self.usage_stats["calls"], 1), 4
            ),
            "llm_configurations": {
                complexity.value: {
                    "model": config.model,
                    "cost_per_1k": config.cost_per_1k_tokens
                }
                for complexity, config in self.llm_configs.items()
            }
        }

