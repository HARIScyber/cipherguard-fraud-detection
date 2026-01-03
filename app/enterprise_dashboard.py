"""
Enterprise Dashboard Module - Phase 6: Enterprise Integration
Real-time monitoring and management dashboard for enterprise fraud detection
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, asdict
import hashlib
import uuid
import os
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor
import time

logger = logging.getLogger(__name__)

class DashboardRole(Enum):
    """Dashboard user roles."""
    VIEWER = "viewer"
    EDITOR = "editor"
    ADMIN = "admin"

class AlertSeverity(Enum):
    """Alert severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class MetricType(Enum):
    """Types of metrics."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"

@dataclass
class DashboardWidget:
    """Dashboard widget configuration."""
    widget_id: str
    title: str
    type: str
    position: Dict[str, int]  # x, y, width, height
    config: Dict[str, Any]
    data_source: str
    refresh_interval: int  # seconds
    visible: bool = True

@dataclass
class Dashboard:
    """Dashboard configuration."""
    dashboard_id: str
    name: str
    description: str
    tenant_id: Optional[str]
    widgets: List[DashboardWidget]
    created_by: str
    created_at: datetime
    updated_at: datetime
    is_public: bool = False
    tags: List[str] = None

@dataclass
class AlertRule:
    """Alert rule configuration."""
    rule_id: str
    name: str
    description: str
    condition: str  # Expression to evaluate
    severity: str
    enabled: bool
    tenant_id: Optional[str]
    notification_channels: List[str]
    cooldown_period: int  # seconds
    last_triggered: Optional[datetime]

@dataclass
class Alert:
    """Alert instance."""
    alert_id: str
    rule_id: str
    message: str
    severity: str
    status: str  # active, resolved, acknowledged
    tenant_id: Optional[str]
    created_at: datetime
    resolved_at: Optional[datetime]
    acknowledged_at: Optional[datetime]
    acknowledged_by: Optional[str]
    details: Dict[str, Any]

@dataclass
class Metric:
    """Metric data point."""
    name: str
    value: Union[int, float]
    timestamp: datetime
    labels: Dict[str, str]
    metric_type: str

class MetricsCollector:
    """Collects and stores metrics for dashboard."""

    def __init__(self, retention_hours: int = 168):  # 7 days
        self.metrics: Dict[str, List[Metric]] = {}
        self.retention_hours = retention_hours
        self._lock = threading.RLock()
        self._cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self._cleanup_thread.start()

    def record_metric(self, name: str, value: Union[int, float],
                     labels: Dict[str, str] = None, metric_type: MetricType = MetricType.GAUGE):
        """Record a metric."""
        metric = Metric(
            name=name,
            value=value,
            timestamp=datetime.utcnow(),
            labels=labels or {},
            metric_type=metric_type.value
        )

        with self._lock:
            if name not in self.metrics:
                self.metrics[name] = []

            self.metrics[name].append(metric)

            # Keep only recent metrics
            cutoff = datetime.utcnow() - timedelta(hours=self.retention_hours)
            self.metrics[name] = [
                m for m in self.metrics[name]
                if m.timestamp > cutoff
            ]

    def get_metric(self, name: str, start_time: Optional[datetime] = None,
                  end_time: Optional[datetime] = None, labels: Dict[str, str] = None) -> List[Metric]:
        """Get metrics for a specific name and time range."""
        if name not in self.metrics:
            return []

        metrics = self.metrics[name]

        if start_time:
            metrics = [m for m in metrics if m.timestamp >= start_time]

        if end_time:
            metrics = [m for m in metrics if m.timestamp <= end_time]

        if labels:
            metrics = [
                m for m in metrics
                if all(m.labels.get(k) == v for k, v in labels.items())
            ]

        return metrics

    def get_metric_stats(self, name: str, hours: int = 24) -> Dict[str, Any]:
        """Get statistics for a metric over a time period."""
        start_time = datetime.utcnow() - timedelta(hours=hours)
        metrics = self.get_metric(name, start_time=start_time)

        if not metrics:
            return {"count": 0, "avg": 0, "min": 0, "max": 0, "latest": 0}

        values = [m.value for m in metrics]

        return {
            "count": len(values),
            "avg": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
            "latest": values[-1] if values else 0
        }

    def _cleanup_loop(self):
        """Background cleanup of old metrics."""
        while True:
            time.sleep(3600)  # Clean up every hour
            self._cleanup_old_metrics()

    def _cleanup_old_metrics(self):
        """Remove metrics older than retention period."""
        cutoff = datetime.utcnow() - timedelta(hours=self.retention_hours)

        with self._lock:
            for name in list(self.metrics.keys()):
                self.metrics[name] = [
                    m for m in self.metrics[name]
                    if m.timestamp > cutoff
                ]

                # Remove empty metric lists
                if not self.metrics[name]:
                    del self.metrics[name]

class AlertManager:
    """Manages alerts and alert rules."""

    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self._lock = threading.RLock()

    def create_rule(self, name: str, description: str, condition: str,
                   severity: AlertSeverity, tenant_id: Optional[str] = None,
                   notification_channels: List[str] = None,
                   cooldown_period: int = 300) -> str:
        """Create a new alert rule."""
        rule_id = str(uuid.uuid4())

        rule = AlertRule(
            rule_id=rule_id,
            name=name,
            description=description,
            condition=condition,
            severity=severity.value,
            enabled=True,
            tenant_id=tenant_id,
            notification_channels=notification_channels or ["email"],
            cooldown_period=cooldown_period,
            last_triggered=None
        )

        with self._lock:
            self.rules[rule_id] = rule

        logger.info(f"Created alert rule: {rule_id} ({name})")
        return rule_id

    def evaluate_rules(self):
        """Evaluate all alert rules and trigger alerts if needed."""
        with self._lock:
            for rule in self.rules.values():
                if not rule.enabled:
                    continue

                # Check cooldown period
                if rule.last_triggered and \
                   datetime.utcnow() - rule.last_triggered < timedelta(seconds=rule.cooldown_period):
                    continue

                try:
                    if self._evaluate_condition(rule.condition):
                        self._trigger_alert(rule)
                        rule.last_triggered = datetime.utcnow()
                except Exception as e:
                    logger.error(f"Error evaluating rule {rule.rule_id}: {e}")

    def _evaluate_condition(self, condition: str) -> bool:
        """Evaluate an alert condition expression."""
        # Simple expression evaluator - in production, use a proper expression parser
        # Example conditions: "fraud_detections > 10", "api_errors_rate > 0.05"

        try:
            # Parse simple conditions like "metric_name > value"
            if ">" in condition:
                metric_name, threshold = condition.split(">", 1)
                metric_name = metric_name.strip()
                threshold = float(threshold.strip())

                stats = self.metrics_collector.get_metric_stats(metric_name, hours=1)
                return stats["latest"] > threshold

            elif "<" in condition:
                metric_name, threshold = condition.split("<", 1)
                metric_name = metric_name.strip()
                threshold = float(threshold.strip())

                stats = self.metrics_collector.get_metric_stats(metric_name, hours=1)
                return stats["latest"] < threshold

            elif "==" in condition:
                metric_name, threshold = condition.split("==", 1)
                metric_name = metric_name.strip()
                threshold = float(threshold.strip())

                stats = self.metrics_collector.get_metric_stats(metric_name, hours=1)
                return abs(stats["latest"] - threshold) < 0.001

        except Exception as e:
            logger.error(f"Error parsing condition '{condition}': {e}")
            return False

        return False

    def _trigger_alert(self, rule: AlertRule):
        """Trigger an alert for a rule."""
        alert_id = str(uuid.uuid4())

        alert = Alert(
            alert_id=alert_id,
            rule_id=rule.rule_id,
            message=f"Alert: {rule.name} - {rule.description}",
            severity=rule.severity,
            status="active",
            tenant_id=rule.tenant_id,
            created_at=datetime.utcnow(),
            resolved_at=None,
            acknowledged_at=None,
            acknowledged_by=None,
            details={
                "rule_condition": rule.condition,
                "triggered_at": datetime.utcnow().isoformat()
            }
        )

        self.active_alerts[alert_id] = alert

        # Send notifications
        self._send_notifications(alert, rule.notification_channels)

        logger.warning(f"Alert triggered: {alert_id} - {alert.message}")

    def acknowledge_alert(self, alert_id: str, user_id: str):
        """Acknowledge an alert."""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.status = "acknowledged"
            alert.acknowledged_at = datetime.utcnow()
            alert.acknowledged_by = user_id

            logger.info(f"Alert acknowledged: {alert_id} by {user_id}")

    def resolve_alert(self, alert_id: str):
        """Resolve an alert."""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.status = "resolved"
            alert.resolved_at = datetime.utcnow()

            logger.info(f"Alert resolved: {alert_id}")

    def get_active_alerts(self, tenant_id: Optional[str] = None) -> List[Alert]:
        """Get active alerts."""
        alerts = [
            alert for alert in self.active_alerts.values()
            if alert.status in ["active", "acknowledged"]
        ]

        if tenant_id:
            alerts = [a for a in alerts if a.tenant_id == tenant_id]

        return alerts

    def _send_notifications(self, alert: Alert, channels: List[str]):
        """Send alert notifications through specified channels."""
        # Implementation would integrate with email, Slack, webhook services
        for channel in channels:
            if channel == "email":
                self._send_email_notification(alert)
            elif channel == "slack":
                self._send_slack_notification(alert)
            elif channel == "webhook":
                self._send_webhook_notification(alert)

    def _send_email_notification(self, alert: Alert):
        """Send email notification."""
        # Placeholder for email integration
        logger.info(f"Email notification sent for alert {alert.alert_id}")

    def _send_slack_notification(self, alert: Alert):
        """Send Slack notification."""
        # Placeholder for Slack integration
        logger.info(f"Slack notification sent for alert {alert.alert_id}")

    def _send_webhook_notification(self, alert: Alert):
        """Send webhook notification."""
        # Placeholder for webhook integration
        logger.info(f"Webhook notification sent for alert {alert.alert_id}")

class DashboardManager:
    """Manages dashboards and widgets."""

    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.dashboards: Dict[str, Dashboard] = {}
        self._lock = threading.RLock()

    def create_dashboard(self, name: str, description: str, tenant_id: Optional[str],
                        created_by: str, widgets: List[DashboardWidget] = None) -> str:
        """Create a new dashboard."""
        dashboard_id = str(uuid.uuid4())

        dashboard = Dashboard(
            dashboard_id=dashboard_id,
            name=name,
            description=description,
            tenant_id=tenant_id,
            widgets=widgets or [],
            created_by=created_by,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            is_public=False,
            tags=[]
        )

        with self._lock:
            self.dashboards[dashboard_id] = dashboard

        logger.info(f"Created dashboard: {dashboard_id} ({name})")
        return dashboard_id

    def get_dashboard(self, dashboard_id: str) -> Optional[Dashboard]:
        """Get a dashboard by ID."""
        return self.dashboards.get(dashboard_id)

    def update_dashboard(self, dashboard_id: str, updates: Dict[str, Any]):
        """Update a dashboard."""
        if dashboard_id not in self.dashboards:
            return False

        with self._lock:
            dashboard = self.dashboards[dashboard_id]
            for key, value in updates.items():
                if hasattr(dashboard, key):
                    setattr(dashboard, key, value)
            dashboard.updated_at = datetime.utcnow()

        logger.info(f"Updated dashboard: {dashboard_id}")
        return True

    def add_widget(self, dashboard_id: str, widget: DashboardWidget):
        """Add a widget to a dashboard."""
        if dashboard_id not in self.dashboards:
            return False

        with self._lock:
            self.dashboards[dashboard_id].widgets.append(widget)
            self.dashboards[dashboard_id].updated_at = datetime.utcnow()

        logger.info(f"Added widget {widget.widget_id} to dashboard {dashboard_id}")
        return True

    def remove_widget(self, dashboard_id: str, widget_id: str):
        """Remove a widget from a dashboard."""
        if dashboard_id not in self.dashboards:
            return False

        with self._lock:
            dashboard = self.dashboards[dashboard_id]
            dashboard.widgets = [w for w in dashboard.widgets if w.widget_id != widget_id]
            dashboard.updated_at = datetime.utcnow()

        logger.info(f"Removed widget {widget_id} from dashboard {dashboard_id}")
        return True

    def get_dashboard_data(self, dashboard_id: str) -> Dict[str, Any]:
        """Get dashboard with current widget data."""
        dashboard = self.get_dashboard(dashboard_id)
        if not dashboard:
            return None

        widget_data = {}
        for widget in dashboard.widgets:
            if widget.visible:
                widget_data[widget.widget_id] = self._get_widget_data(widget)

        return {
            "dashboard": asdict(dashboard),
            "widget_data": widget_data
        }

    def _get_widget_data(self, widget: DashboardWidget) -> Dict[str, Any]:
        """Get data for a widget."""
        try:
            if widget.type == "metric_chart":
                return self._get_metric_chart_data(widget)
            elif widget.type == "alert_list":
                return self._get_alert_list_data(widget)
            elif widget.type == "fraud_overview":
                return self._get_fraud_overview_data(widget)
            elif widget.type == "system_health":
                return self._get_system_health_data(widget)
            else:
                return {"error": f"Unknown widget type: {widget.type}"}
        except Exception as e:
            logger.error(f"Error getting widget data for {widget.widget_id}: {e}")
            return {"error": str(e)}

    def _get_metric_chart_data(self, widget: DashboardWidget) -> Dict[str, Any]:
        """Get data for metric chart widget."""
        metric_name = widget.config.get("metric_name")
        hours = widget.config.get("hours", 24)

        if not metric_name:
            return {"error": "No metric_name specified"}

        metrics = self.metrics_collector.get_metric(
            metric_name,
            start_time=datetime.utcnow() - timedelta(hours=hours)
        )

        return {
            "metric_name": metric_name,
            "data_points": [
                {
                    "timestamp": m.timestamp.isoformat(),
                    "value": m.value,
                    "labels": m.labels
                }
                for m in metrics
            ]
        }

    def _get_alert_list_data(self, widget: DashboardWidget) -> Dict[str, Any]:
        """Get data for alert list widget."""
        # This would integrate with AlertManager
        return {"alerts": [], "note": "Alert integration not implemented"}

    def _get_fraud_overview_data(self, widget: DashboardWidget) -> Dict[str, Any]:
        """Get fraud detection overview data."""
        # Aggregate fraud-related metrics
        fraud_detections = self.metrics_collector.get_metric_stats("fraud_detections", hours=24)
        false_positives = self.metrics_collector.get_metric_stats("false_positives", hours=24)
        blocked_transactions = self.metrics_collector.get_metric_stats("blocked_transactions", hours=24)

        return {
            "fraud_detections": fraud_detections,
            "false_positives": false_positives,
            "blocked_transactions": blocked_transactions,
            "accuracy": 1 - (false_positives["avg"] / max(fraud_detections["avg"], 1))
        }

    def _get_system_health_data(self, widget: DashboardWidget) -> Dict[str, Any]:
        """Get system health data."""
        api_response_time = self.metrics_collector.get_metric_stats("api_response_time", hours=1)
        error_rate = self.metrics_collector.get_metric_stats("api_errors", hours=1)
        cpu_usage = self.metrics_collector.get_metric_stats("cpu_usage", hours=1)
        memory_usage = self.metrics_collector.get_metric_stats("memory_usage", hours=1)

        return {
            "api_response_time": api_response_time,
            "error_rate": error_rate,
            "cpu_usage": cpu_usage,
            "memory_usage": memory_usage,
            "overall_health": self._calculate_health_score(api_response_time, error_rate, cpu_usage, memory_usage)
        }

    def _calculate_health_score(self, response_time: Dict, error_rate: Dict,
                               cpu: Dict, memory: Dict) -> str:
        """Calculate overall system health score."""
        score = 100

        # Response time penalty
        if response_time["avg"] > 1000:  # > 1 second
            score -= 20
        elif response_time["avg"] > 500:  # > 0.5 second
            score -= 10

        # Error rate penalty
        if error_rate["avg"] > 0.05:  # > 5%
            score -= 30
        elif error_rate["avg"] > 0.01:  # > 1%
            score -= 15

        # CPU usage penalty
        if cpu["avg"] > 90:
            score -= 25
        elif cpu["avg"] > 75:
            score -= 10

        # Memory usage penalty
        if memory["avg"] > 90:
            score -= 25
        elif memory["avg"] > 75:
            score -= 10

        if score >= 90:
            return "excellent"
        elif score >= 75:
            return "good"
        elif score >= 60:
            return "warning"
        else:
            return "critical"

class RealTimeUpdater:
    """Handles real-time updates for dashboard clients."""

    def __init__(self, dashboard_manager: DashboardManager, alert_manager: AlertManager):
        self.dashboard_manager = dashboard_manager
        self.alert_manager = alert_manager
        self.subscribers: Dict[str, List[Callable]] = {}
        self._lock = threading.RLock()
        self._update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self._update_thread.start()

    def subscribe_to_dashboard(self, dashboard_id: str, callback: Callable):
        """Subscribe to real-time updates for a dashboard."""
        with self._lock:
            if dashboard_id not in self.subscribers:
                self.subscribers[dashboard_id] = []
            self.subscribers[dashboard_id].append(callback)

    def unsubscribe_from_dashboard(self, dashboard_id: str, callback: Callable):
        """Unsubscribe from dashboard updates."""
        with self._lock:
            if dashboard_id in self.subscribers:
                self.subscribers[dashboard_id].remove(callback)
                if not self.subscribers[dashboard_id]:
                    del self.subscribers[dashboard_id]

    def _update_loop(self):
        """Background update loop."""
        while True:
            try:
                # Evaluate alert rules
                self.alert_manager.evaluate_rules()

                # Send updates to subscribers
                self._send_updates()

            except Exception as e:
                logger.error(f"Error in update loop: {e}")

            time.sleep(30)  # Update every 30 seconds

    def _send_updates(self):
        """Send updates to all subscribers."""
        with self._lock:
            for dashboard_id, callbacks in self.subscribers.items():
                try:
                    data = self.dashboard_manager.get_dashboard_data(dashboard_id)
                    if data:
                        for callback in callbacks:
                            try:
                                callback(data)
                            except Exception as e:
                                logger.error(f"Error in dashboard callback: {e}")
                except Exception as e:
                    logger.error(f"Error getting dashboard data for {dashboard_id}: {e}")

# Global instances
_metrics_collector = None
_alert_manager = None
_dashboard_manager = None
_real_time_updater = None

def get_metrics_collector() -> MetricsCollector:
    """Get global metrics collector."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector

def get_alert_manager() -> AlertManager:
    """Get global alert manager."""
    global _alert_manager
    if _alert_manager is None:
        _alert_manager = AlertManager(get_metrics_collector())
    return _alert_manager

def get_dashboard_manager() -> DashboardManager:
    """Get global dashboard manager."""
    global _dashboard_manager
    if _dashboard_manager is None:
        _dashboard_manager = DashboardManager(get_metrics_collector())
    return _dashboard_manager

def get_real_time_updater() -> RealTimeUpdater:
    """Get global real-time updater."""
    global _real_time_updater
    if _real_time_updater is None:
        _real_time_updater = RealTimeUpdater(get_dashboard_manager(), get_alert_manager())
    return _real_time_updater