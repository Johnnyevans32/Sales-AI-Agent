from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict

from app.models.base import ToolStatus, ToolUsageLog


class MetricsService:
    def __init__(self, logs_dir: str = "logs"):
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(exist_ok=True)

        self.metrics = {
            "tool_invocations": {
                "total": 0,
                "success": 0,
                "error": 0,
                "human_review": 0,
            },
            "latency": {"total": 0, "count": 0, "min": float("inf"), "max": 0},
            "confidence_scores": {
                "total": 0,
                "count": 0,
                "min": float("inf"),
                "max": 0,
            },
            "clarification_requests": {"total": 0, "reasons": {}},
        }

    def log_tool_usage(self, log: ToolUsageLog):
        """Log tool usage to file and update metrics."""
        log_file = (
            self.logs_dir / f"tool_usage_{datetime.now().strftime('%Y-%m-%d')}.jsonl"
        )
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(log.model_dump_json() + "\n")

        if log.status:
            # Update metrics
            self.metrics["tool_invocations"]["total"] += 1
            if log.status == ToolStatus.SUCCESS:
                self.metrics["tool_invocations"]["success"] += 1
            elif log.status == ToolStatus.ERROR:
                self.metrics["tool_invocations"]["error"] += 1
            elif log.status == ToolStatus.FLAG_FOR_REVIEW:
                self.metrics["tool_invocations"]["human_review"] += 1

        # Update latency metrics
        self.metrics["latency"]["total"] += log.latency_ms
        self.metrics["latency"]["count"] += 1
        self.metrics["latency"]["min"] = min(
            self.metrics["latency"]["min"], log.latency_ms
        )
        self.metrics["latency"]["max"] = max(
            self.metrics["latency"]["max"], log.latency_ms
        )

        if log.confidence_score:
            # Update confidence score metrics
            self.metrics["confidence_scores"]["total"] += log.confidence_score
            self.metrics["confidence_scores"]["count"] += 1
            self.metrics["confidence_scores"]["min"] = min(
                self.metrics["confidence_scores"]["min"], log.confidence_score
            )
            self.metrics["confidence_scores"]["max"] = max(
                self.metrics["confidence_scores"]["max"], log.confidence_score
            )

        # Update clarification metrics
        if log.clarification_needed:
            self.metrics["clarification_requests"]["total"] += 1
            if log.clarification_reason:
                self.metrics["clarification_requests"]["reasons"][
                    log.clarification_reason
                ] = (
                    self.metrics["clarification_requests"]["reasons"].get(
                        log.clarification_reason, 0
                    )
                    + 1
                )

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Calculate and return current performance metrics."""
        metrics = {
            "tool_invocation_rate": {
                "success_rate": self.metrics["tool_invocations"]["success"]
                / max(1, self.metrics["tool_invocations"]["total"]),
                "error_rate": self.metrics["tool_invocations"]["error"]
                / max(1, self.metrics["tool_invocations"]["total"]),
                "human_review_rate": self.metrics["tool_invocations"]["human_review"]
                / max(1, self.metrics["tool_invocations"]["total"]),
            },
            "latency": {
                "avg_ms": self.metrics["latency"]["total"]
                / max(1, self.metrics["latency"]["count"]),
                "min_ms": (
                    self.metrics["latency"]["min"]
                    if self.metrics["latency"]["min"] != float("inf")
                    else 0
                ),
                "max_ms": self.metrics["latency"]["max"],
            },
            "confidence": {
                "avg_score": self.metrics["confidence_scores"]["total"]
                / max(1, self.metrics["confidence_scores"]["count"]),
                "min_score": (
                    self.metrics["confidence_scores"]["min"]
                    if self.metrics["confidence_scores"]["min"] != float("inf")
                    else 0
                ),
                "max_score": self.metrics["confidence_scores"]["max"],
            },
            "clarification_requests": {
                "total": self.metrics["clarification_requests"]["total"],
                "reasons": self.metrics["clarification_requests"]["reasons"],
            },
        }

        # Calculate composite LLM Performance Score
        weights = {"success_rate": 0.4, "confidence": 0.3, "latency": 0.3}

        # Normalize latency (lower is better)
        max_latency = 5000  # 5 seconds
        latency_score = 1 - min(1, metrics["latency"]["avg_ms"] / max_latency)

        performance_score = (
            weights["success_rate"] * metrics["tool_invocation_rate"]["success_rate"]
            + weights["confidence"] * metrics["confidence"]["avg_score"]
            + weights["latency"] * latency_score
        )

        metrics["llm_performance_score"] = performance_score

        return metrics

    def get_historical_metrics(self, days: int = 7) -> Dict[str, Any]:
        """Get historical metrics for the specified number of days."""
        historical_metrics = []

        for i in range(days):
            date = datetime.now().date() - timedelta(days=i)
            log_file = self.logs_dir / f"tool_usage_{date.strftime('%Y-%m-%d')}.jsonl"

            if not log_file.exists():
                continue

            daily_metrics = {
                "date": date.isoformat(),
                "tool_invocations": 0,
                "success_rate": 0,
                "error_rate": 0,
                "avg_latency": 0,
                "avg_confidence": 0,
            }

            with open(log_file, "r", encoding="utf-8") as f:
                logs = [ToolUsageLog.model_validate_json(line) for line in f]

                if not logs:
                    continue

                daily_metrics["tool_invocations"] = len(logs)
                daily_metrics["success_rate"] = sum(
                    1 for log in logs if log.status == ToolStatus.SUCCESS
                ) / len(logs)
                daily_metrics["error_rate"] = sum(
                    1 for log in logs if log.status == ToolStatus.ERROR
                ) / len(logs)
                daily_metrics["avg_latency"] = sum(
                    log.latency_ms for log in logs
                ) / len(logs)
                daily_metrics["avg_confidence"] = sum(
                    log.confidence_score for log in logs
                ) / len(logs)

            historical_metrics.append(daily_metrics)

        return {
            "historical_metrics": historical_metrics,
            "summary": {
                "total_days": len(historical_metrics),
                "avg_daily_invocations": sum(
                    m["tool_invocations"] for m in historical_metrics
                )
                / max(1, len(historical_metrics)),
                "avg_success_rate": sum(m["success_rate"] for m in historical_metrics)
                / max(1, len(historical_metrics)),
                "avg_error_rate": sum(m["error_rate"] for m in historical_metrics)
                / max(1, len(historical_metrics)),
                "avg_latency": sum(m["avg_latency"] for m in historical_metrics)
                / max(1, len(historical_metrics)),
                "avg_confidence": sum(m["avg_confidence"] for m in historical_metrics)
                / max(1, len(historical_metrics)),
            },
        }
