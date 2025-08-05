# analytics/agent_analytics.py
"""Comprehensive analytics and monitoring for the BOM agent."""

import time
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Any, Optional

import structlog


class OperationType(Enum):
    """Types of operations tracked."""
    SCHEMATIC_ANALYSIS = "schematic_analysis"
    COMPONENT_SEARCH = "component_search"
    BOM_CREATION = "bom_creation"
    BOM_LISTING = "bom_listing"
    PARTS_ADDITION = "parts_addition"


@dataclass
class OperationMetrics:
    """Metrics for a single operation."""
    operation_type: str
    start_time: datetime
    end_time: Optional[datetime] = None
    success: bool = False
    error_message: Optional[str] = None
    execution_time: float = 0.0
    input_size: int = 0
    output_size: int = 0
    llm_used: Optional[str] = None
    cost_estimate: float = 0.0
    user_context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceStats:
    """Aggregated performance statistics."""
    total_operations: int = 0
    successful_operations: int = 0
    failed_operations: int = 0
    average_execution_time: float = 0.0
    total_cost: float = 0.0
    operations_by_type: Dict[str, int] = field(default_factory=dict)
    error_patterns: Dict[str, int] = field(default_factory=dict)
    llm_usage: Dict[str, Dict[str, Any]] = field(default_factory=dict)


class AgentAnalytics:
    """Comprehensive analytics and monitoring system."""

    def __init__(self, enable_detailed_logging: bool = True):
        self.enable_detailed_logging = enable_detailed_logging
        self.logger = structlog.get_logger("agent_analytics")

        # Metrics storage
        self.operation_history: List[OperationMetrics] = []
        self.performance_stats = PerformanceStats()
        self.real_time_metrics = {
            "active_operations": 0,
            "current_hour_operations": 0,
            "last_operation_time": None
        }

        # User behavior tracking
        self.user_patterns = {
            "preferred_operations": Counter(),
            "peak_usage_hours": Counter(),
            "session_lengths": [],
            "error_recovery_success": 0,
            "error_recovery_attempts": 0
        }

        # Performance thresholds for alerting
        self.thresholds = {
            "max_execution_time": 30.0,  # seconds
            "max_error_rate": 0.15,  # 15%
            "max_cost_per_hour": 5.0  # dollars
        }

    def start_operation(self, operation_type: OperationType,
                        input_data: Dict[str, Any] = None) -> str:
        """Start tracking an operation."""
        operation_id = f"{operation_type.value}_{int(time.time() * 1000)}"

        metrics = OperationMetrics(
            operation_type=operation_type.value,
            start_time=datetime.now(),
            input_size=len(str(input_data)) if input_data else 0,
            user_context=input_data or {}
        )

        self.operation_history.append(metrics)
        self.real_time_metrics["active_operations"] += 1
        self.real_time_metrics["current_hour_operations"] += 1
        self.real_time_metrics["last_operation_time"] = datetime.now()

        # Track user patterns
        self.user_patterns["preferred_operations"][operation_type.value] += 1
        current_hour = datetime.now().hour
        self.user_patterns["peak_usage_hours"][current_hour] += 1

        if self.enable_detailed_logging:
            self.logger.info("operation_started",
                             operation_id=operation_id,
                             operation_type=operation_type.value,
                             input_size=metrics.input_size)

        return operation_id

    def end_operation(self, operation_id: str, success: bool = True,
                      output_data: Any = None, error_message: str = None,
                      llm_used: str = None, cost_estimate: float = 0.0):
        """End tracking an operation and record results."""

        # Find the operation
        operation = None
        for op in reversed(self.operation_history):
            if f"{op.operation_type}_{int(op.start_time.timestamp() * 1000)}" == operation_id:
                operation = op
                break

        if not operation:
            self.logger.warning("operation_not_found", operation_id=operation_id)
            return

        # Update operation metrics
        end_time = datetime.now()
        operation.end_time = end_time
        operation.success = success
        operation.error_message = error_message
        operation.execution_time = (end_time - operation.start_time).total_seconds()
        operation.output_size = len(str(output_data)) if output_data else 0
        operation.llm_used = llm_used
        operation.cost_estimate = cost_estimate

        # Update real-time metrics
        self.real_time_metrics["active_operations"] -= 1

        # Update performance stats
        self._update_performance_stats(operation)

        # Check for performance alerts
        self._check_performance_alerts(operation)

        if self.enable_detailed_logging:
            self.logger.info("operation_completed",
                             operation_id=operation_id,
                             success=success,
                             execution_time=operation.execution_time,
                             llm_used=llm_used,
                             cost=cost_estimate)

    def _update_performance_stats(self, operation: OperationMetrics):
        """Update aggregated performance statistics."""
        stats = self.performance_stats

        # Basic counts
        stats.total_operations += 1
        if operation.success:
            stats.successful_operations += 1
        else:
            stats.failed_operations += 1

        # Operation type tracking
        op_type = operation.operation_type
        stats.operations_by_type[op_type] = stats.operations_by_type.get(op_type, 0) + 1

        # Average execution time (running average)
        if stats.total_operations == 1:
            stats.average_execution_time = operation.execution_time
        else:
            stats.average_execution_time = (
                    (stats.average_execution_time * (stats.total_operations - 1) +
                     operation.execution_time) / stats.total_operations
            )

        # Cost tracking
        stats.total_cost += operation.cost_estimate

        # Error pattern tracking
        if not operation.success and operation.error_message:
            # Extract error type from message
            error_type = self._classify_error(operation.error_message)
            stats.error_patterns[error_type] = stats.error_patterns.get(error_type, 0) + 1

        # LLM usage tracking
        if operation.llm_used:
            if operation.llm_used not in stats.llm_usage:
                stats.llm_usage[operation.llm_used] = {
                    "calls": 0,
                    "total_cost": 0.0,
                    "avg_execution_time": 0.0,
                    "success_rate": 0.0
                }

            llm_stats = stats.llm_usage[operation.llm_used]
            llm_stats["calls"] += 1
            llm_stats["total_cost"] += operation.cost_estimate

            # Update LLM-specific success rate
            if operation.success:
                llm_stats["success_rate"] = (
                        (llm_stats["success_rate"] * (llm_stats["calls"] - 1) + 1) / llm_stats["calls"]
                )
            else:
                llm_stats["success_rate"] = (
                        llm_stats["success_rate"] * (llm_stats["calls"] - 1) / llm_stats["calls"]
                )

    def _classify_error(self, error_message: str) -> str:
        """Classify error message into categories."""
        error_lower = error_message.lower()

        if "quota" in error_lower or "rate limit" in error_lower:
            return "quota_exceeded"
        elif "timeout" in error_lower or "connection" in error_lower:
            return "network_error"
        elif "authentication" in error_lower or "401" in error_lower:
            return "auth_error"
        elif "validation" in error_lower or "invalid" in error_lower:
            return "validation_error"
        elif "404" in error_lower or "not found" in error_lower:
            return "not_found"
        else:
            return "unknown_error"

    def _check_performance_alerts(self, operation: OperationMetrics):
        """Check if operation triggers any performance alerts."""
        alerts = []

        # Execution time alert
        if operation.execution_time > self.thresholds["max_execution_time"]:
            alerts.append({
                "type": "slow_operation",
                "message": f"Operation took {operation.execution_time:.1f}s (threshold: {self.thresholds['max_execution_time']}s)",
                "operation_type": operation.operation_type
            })

        # Error rate alert
        if self.performance_stats.total_operations >= 10:  # Only check after sufficient data
            error_rate = self.performance_stats.failed_operations / self.performance_stats.total_operations
            if error_rate > self.thresholds["max_error_rate"]:
                alerts.append({
                    "type": "high_error_rate",
                    "message": f"Error rate: {error_rate:.1%} (threshold: {self.thresholds['max_error_rate']:.1%})",
                    "current_error_rate": error_rate
                })

        # Cost alert (hourly)
        hour_ago = datetime.now() - timedelta(hours=1)
        recent_operations = [op for op in self.operation_history
                             if op.start_time > hour_ago and op.cost_estimate > 0]
        hourly_cost = sum(op.cost_estimate for op in recent_operations)

        if hourly_cost > self.thresholds["max_cost_per_hour"]:
            alerts.append({
                "type": "high_cost",
                "message": f"Hourly cost: ${hourly_cost:.2f} (threshold: ${self.thresholds['max_cost_per_hour']})",
                "hourly_cost": hourly_cost
            })

        # Log alerts
        for alert in alerts:
            self.logger.warning("performance_alert", **alert)

    def get_performance_report(self, time_window_hours: int = 24) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        cutoff_time = datetime.now() - timedelta(hours=time_window_hours)
        recent_operations = [op for op in self.operation_history if op.start_time > cutoff_time]

        if not recent_operations:
            return {"message": "No operations in specified time window"}

        # Calculate metrics for time window
        successful_ops = [op for op in recent_operations if op.success]
        failed_ops = [op for op in recent_operations if not op.success]

        report = {
            "time_window_hours": time_window_hours,
            "summary": {
                "total_operations": len(recent_operations),
                "successful_operations": len(successful_ops),
                "failed_operations": len(failed_ops),
                "success_rate": len(successful_ops) / len(recent_operations) * 100,
                "average_execution_time": sum(op.execution_time for op in recent_operations) / len(recent_operations),
                "total_cost": sum(op.cost_estimate for op in recent_operations)
            },
            "operation_breakdown": dict(Counter(op.operation_type for op in recent_operations)),
            "llm_usage": self._calculate_llm_usage_report(recent_operations),
            "error_analysis": self._analyze_errors(failed_ops),
            "performance_trends": self._calculate_performance_trends(recent_operations),
            "user_behavior": self._analyze_user_behavior(recent_operations),
            "recommendations": self._generate_recommendations(recent_operations)
        }

        return report

    def _calculate_llm_usage_report(self, operations: List[OperationMetrics]) -> Dict[str, Any]:
        """Calculate LLM usage statistics."""
        llm_ops = [op for op in operations if op.llm_used]
        if not llm_ops:
            return {}

        usage_by_llm = defaultdict(list)
        for op in llm_ops:
            usage_by_llm[op.llm_used].append(op)

        report = {}
        for llm, ops in usage_by_llm.items():
            report[llm] = {
                "call_count": len(ops),
                "total_cost": sum(op.cost_estimate for op in ops),
                "success_rate": len([op for op in ops if op.success]) / len(ops) * 100,
                "avg_execution_time": sum(op.execution_time for op in ops) / len(ops)
            }

        return report

    def _analyze_errors(self, failed_operations: List[OperationMetrics]) -> Dict[str, Any]:
        """Analyze error patterns and trends."""
        if not failed_operations:
            return {"message": "No errors in time window"}

        error_types = [self._classify_error(op.error_message) for op in failed_operations
                       if op.error_message]

        return {
            "total_errors": len(failed_operations),
            "error_types": dict(Counter(error_types)),
            "most_common_error": Counter(error_types).most_common(1)[0] if error_types else None,
            "operations_with_most_errors": dict(Counter(op.operation_type for op in failed_operations))
        }

    def _calculate_performance_trends(self, operations: List[OperationMetrics]) -> Dict[str, Any]:
        """Calculate performance trends over time."""
        if len(operations) < 5:
            return {"message": "Insufficient data for trend analysis"}

        # Sort by time
        sorted_ops = sorted(operations, key=lambda x: x.start_time)

        # Split into early and late periods
        mid_point = len(sorted_ops) // 2
        early_ops = sorted_ops[:mid_point]
        late_ops = sorted_ops[mid_point:]

        early_avg_time = sum(op.execution_time for op in early_ops) / len(early_ops)
        late_avg_time = sum(op.execution_time for op in late_ops) / len(late_ops)

        early_success_rate = len([op for op in early_ops if op.success]) / len(early_ops)
        late_success_rate = len([op for op in late_ops if op.success]) / len(late_ops)

        return {
            "execution_time_trend": {
                "early_period_avg": early_avg_time,
                "late_period_avg": late_avg_time,
                "change_percent": ((late_avg_time - early_avg_time) / early_avg_time) * 100
            },
            "success_rate_trend": {
                "early_period": early_success_rate * 100,
                "late_period": late_success_rate * 100,
                "change_percent": ((late_success_rate - early_success_rate) / early_success_rate) * 100
            }
        }

    def _analyze_user_behavior(self, operations: List[OperationMetrics]) -> Dict[str, Any]:
        """Analyze user behavior patterns."""
        if not operations:
            return {}

        # Time-based patterns
        hour_distribution = Counter(op.start_time.hour for op in operations)

        # Operation sequences
        op_sequence = [op.operation_type for op in sorted(operations, key=lambda x: x.start_time)]
        common_sequences = self._find_common_sequences(op_sequence)

        return {
            "peak_usage_hours": dict(hour_distribution.most_common(3)),
            "common_operation_sequences": common_sequences,
            "average_operations_per_session": len(operations) / max(1,
                                                                    len(set(op.start_time.date() for op in operations)))
        }

    def _find_common_sequences(self, sequence: List[str], window_size: int = 3) -> List[tuple]:
        """Find common operation sequences."""
        if len(sequence) < window_size:
            return []

        sequences = []
        for i in range(len(sequence) - window_size + 1):
            sequences.append(tuple(sequence[i:i + window_size]))

        return Counter(sequences).most_common(3)

    def _generate_recommendations(self, operations: List[OperationMetrics]) -> List[str]:
        """Generate optimization recommendations based on analysis."""
        recommendations = []

        if not operations:
            return recommendations

        # Performance recommendations
        avg_time = sum(op.execution_time for op in operations) / len(operations)
        if avg_time > 10:
            recommendations.append("Consider implementing request caching to reduce execution times")

        # Cost optimization
        total_cost = sum(op.cost_estimate for op in operations)
        if total_cost > 2.0:  # $2 threshold
            recommendations.append("Evaluate LLM usage patterns for potential cost optimization")

        # Error rate recommendations
        error_rate = len([op for op in operations if not op.success]) / len(operations)
        if error_rate > 0.1:
            recommendations.append("High error rate detected - review error handling and retry mechanisms")

        # LLM usage recommendations
        llm_ops = [op for op in operations if op.llm_used]
        if llm_ops:
            gemini_usage = len([op for op in llm_ops if 'gemini' in op.llm_used.lower()])
            if gemini_usage / len(llm_ops) > 0.7:
                recommendations.append("Consider using cheaper LLMs for non-vision tasks")

        return recommendations
