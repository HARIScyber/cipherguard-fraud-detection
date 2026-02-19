"""
Enterprise Monitoring and Metrics Collection System
"""

import time
import threading
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import statistics
import logging
from concurrent.futures import ThreadPoolExecutor
import json

from app.core.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)


@dataclass
class MetricPoint:
    """Single metric data point."""
    timestamp: datetime
    value: float
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass  
class RequestMetrics:
    """Request metrics container."""
    endpoint: str
    method: str
    status_code: int
    duration_ms: float
    timestamp: datetime


@dataclass
class SystemMetrics:
    """System performance metrics."""
    cpu_usage: float
    memory_usage_mb: float
    active_connections: int
    requests_per_second: float
    error_rate: float
    average_response_time: float
    timestamp: datetime


class MetricsBuffer:
    """Thread-safe circular buffer for metrics."""
    
    def __init__(self, maxsize: int = 10000):
        self.buffer = deque(maxlen=maxsize)
        self._lock = threading.Lock()
    
    def append(self, item: Any):
        """Add item to buffer."""
        with self._lock:
            self.buffer.append(item)
    
    def get_recent(self, count: int = None) -> List[Any]:
        """Get recent items from buffer."""
        with self._lock:
            if count is None:
                return list(self.buffer)
            return list(self.buffer)[-count:]
    
    def get_since(self, since: datetime) -> List[Any]:
        """Get items since a specific time."""
        with self._lock:
            return [item for item in self.buffer if getattr(item, 'timestamp', None) and item.timestamp >= since]


class MetricsCollector:
    """Centralized metrics collection and aggregation."""
    
    def __init__(self):
        self.request_metrics = MetricsBuffer(maxsize=50000)
        self.system_metrics = MetricsBuffer(maxsize=10000)
        self.fraud_metrics = MetricsBuffer(maxsize=20000)
        self.custom_metrics = defaultdict(lambda: MetricsBuffer(maxsize=10000))
        
        # Counters
        self._counters = defaultdict(int)
        self._gauges = defaultdict(float)
        self._lock = threading.Lock()
        
        # Background metrics collection
        self._collection_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="metrics")
        self._collection_active = True
        
        # Start background collection
        self._start_background_collection()
    
    def record_request(
        self, 
        endpoint: str, 
        method: str, 
        status_code: int, 
        duration_ms: float
    ):
        """Record HTTP request metrics."""
        
        metrics = RequestMetrics(
            endpoint=endpoint,
            method=method,
            status_code=status_code,
            duration_ms=duration_ms,
            timestamp=datetime.utcnow()
        )
        
        self.request_metrics.append(metrics)
        
        # Update counters
        with self._lock:
            self._counters['total_requests'] += 1
            self._counters[f'requests_{status_code}'] += 1
            self._counters[f'requests_{method}'] += 1
            
            if status_code >= 400:
                self._counters['error_requests'] += 1
        
        logger.debug(f"Recorded request: {method} {endpoint} - {status_code} - {duration_ms:.1f}ms")
    
    def record_fraud_detection(
        self, 
        transaction_id: str,
        fraud_score: float,
        is_fraud: bool,
        risk_level: str,
        processing_time_ms: float,
        model_version: str
    ):
        """Record fraud detection metrics."""
        
        fraud_data = {
            'transaction_id': transaction_id,
            'fraud_score': fraud_score,
            'is_fraud': is_fraud,
            'risk_level': risk_level,
            'processing_time_ms': processing_time_ms,
            'model_version': model_version,
            'timestamp': datetime.utcnow()
        }
        
        self.fraud_metrics.append(fraud_data)
        
        # Update fraud counters
        with self._lock:
            self._counters['total_fraud_checks'] += 1
            self._counters[f'fraud_risk_{risk_level.lower()}'] += 1
            
            if is_fraud:
                self._counters['fraud_detected'] += 1
            
            # Track processing time
            self.record_gauge('fraud_processing_time_ms', processing_time_ms)
    
    def increment_counter(self, name: str, value: int = 1, tags: Dict[str, str] = None):
        """Increment a named counter."""
        with self._lock:
            counter_key = self._build_metric_key(name, tags)
            self._counters[counter_key] += value
        
        logger.debug(f"Counter {name} incremented by {value}")
    
    def record_gauge(self, name: str, value: float, tags: Dict[str, str] = None):
        """Record a gauge value."""
        with self._lock:
            gauge_key = self._build_metric_key(name, tags)
            self._gauges[gauge_key] = value
        
        # Also store in time series
        metric_point = MetricPoint(
            timestamp=datetime.utcnow(),
            value=value,
            tags=tags or {}
        )
        self.custom_metrics[name].append(metric_point)
    
    def _build_metric_key(self, name: str, tags: Dict[str, str] = None) -> str:
        """Build metric key with tags."""
        if not tags:
            return name
        
        tag_str = ",".join(f"{k}={v}" for k, v in sorted(tags.items()))
        return f"{name}[{tag_str}]"
    
    def get_request_stats(self, time_window: timedelta = None) -> Dict[str, Any]:
        """Get request statistics for a time window."""
        
        if time_window:
            since = datetime.utcnow() - time_window
            requests = self.request_metrics.get_since(since)
        else:
            requests = self.request_metrics.get_recent()
        
        if not requests:
            return {}
        
        # Calculate statistics
        total_requests = len(requests)
        response_times = [req.duration_ms for req in requests]
        status_codes = [req.status_code for req in requests]
        
        error_count = sum(1 for code in status_codes if code >= 400)
        error_rate = error_count / total_requests if total_requests > 0 else 0
        
        stats = {
            'total_requests': total_requests,
            'error_rate': error_rate,
            'avg_response_time': statistics.mean(response_times) if response_times else 0,
            'median_response_time': statistics.median(response_times) if response_times else 0,
            'p95_response_time': self._percentile(response_times, 0.95) if response_times else 0,
            'p99_response_time': self._percentile(response_times, 0.99) if response_times else 0,
            'requests_per_second': total_requests / time_window.total_seconds() if time_window and time_window.total_seconds() > 0 else 0
        }
        
        # Status code breakdown
        status_breakdown = defaultdict(int)
        for code in status_codes:
            status_breakdown[str(code)] += 1
        
        stats['status_codes'] = dict(status_breakdown)
        
        return stats
    
    def get_fraud_stats(self, time_window: timedelta = None) -> Dict[str, Any]:
        """Get fraud detection statistics."""
        
        if time_window:
            since = datetime.utcnow() - time_window
            fraud_data = self.fraud_metrics.get_since(since)
        else:
            fraud_data = self.fraud_metrics.get_recent()
        
        if not fraud_data:
            return {}
        
        total_checks = len(fraud_data)
        fraud_detected = sum(1 for data in fraud_data if data.get('is_fraud', False))
        fraud_scores = [data.get('fraud_score', 0) for data in fraud_data]
        processing_times = [data.get('processing_time_ms', 0) for data in fraud_data]
        
        # Risk level breakdown
        risk_breakdown = defaultdict(int)
        for data in fraud_data:
            risk_level = data.get('risk_level', 'UNKNOWN')
            risk_breakdown[risk_level] += 1
        
        return {
            'total_fraud_checks': total_checks,
            'fraud_detected': fraud_detected,
            'fraud_rate': fraud_detected / total_checks if total_checks > 0 else 0,
            'avg_fraud_score': statistics.mean(fraud_scores) if fraud_scores else 0,
            'avg_processing_time': statistics.mean(processing_times) if processing_times else 0,
            'risk_distribution': dict(risk_breakdown),
            'p95_processing_time': self._percentile(processing_times, 0.95) if processing_times else 0
        }
    
    def get_counters(self) -> Dict[str, int]:
        """Get all counters."""
        with self._lock:
            return dict(self._counters)
    
    def get_gauges(self) -> Dict[str, float]:
        """Get all gauges."""
        with self._lock:
            return dict(self._gauges)
    
    def _percentile(self, data: List[float], percentile: float) -> float:
        """Calculate percentile of data."""
        if not data:
            return 0
        
        sorted_data = sorted(data)
        index = int(percentile * len(sorted_data))
        
        if index >= len(sorted_data):
            index = len(sorted_data) - 1
        
        return sorted_data[index]
    
    def _start_background_collection(self):
        """Start background system metrics collection."""
        
        def collect_system_metrics():
            while self._collection_active:
                try:
                    # Collect system metrics (simplified for this example)
                    # In production, you'd use psutil or similar
                    
                    metrics = SystemMetrics(
                        cpu_usage=0.0,  # Would get actual CPU usage
                        memory_usage_mb=0.0,  # Would get actual memory usage
                        active_connections=0,  # Would get actual connection count
                        requests_per_second=self.get_request_stats(timedelta(seconds=60)).get('requests_per_second', 0),
                        error_rate=self.get_request_stats(timedelta(minutes=5)).get('error_rate', 0),
                        average_response_time=self.get_request_stats(timedelta(minutes=5)).get('avg_response_time', 0),
                        timestamp=datetime.utcnow()
                    )
                    
                    self.system_metrics.append(metrics)
                    
                    # Sleep for collection interval
                    time.sleep(settings.monitoring.collection_interval)
                    
                except Exception as e:
                    logger.error(f"System metrics collection failed: {e}")
                    time.sleep(10)  # Longer sleep on error
        
        # Start background collection
        self._collection_executor.submit(collect_system_metrics)
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get current system statistics."""
        
        recent_metrics = self.system_metrics.get_recent(10)
        
        if not recent_metrics:
            return {}
        
        latest = recent_metrics[-1]
        
        return {
            'cpu_usage': latest.cpu_usage,
            'memory_usage_mb': latest.memory_usage_mb,
            'active_connections': latest.active_connections,
            'requests_per_second': latest.requests_per_second,
            'error_rate': latest.error_rate,
            'average_response_time': latest.average_response_time,
            'timestamp': latest.timestamp.isoformat()
        }
    
    def generate_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive health report."""
        
        now = datetime.utcnow()
        
        return {
            'timestamp': now.isoformat(),
            'request_stats_5m': self.get_request_stats(timedelta(minutes=5)),
            'request_stats_1h': self.get_request_stats(timedelta(hours=1)),
            'fraud_stats_5m': self.get_fraud_stats(timedelta(minutes=5)),
            'fraud_stats_1h': self.get_fraud_stats(timedelta(hours=1)),
            'system_stats': self.get_system_stats(),
            'counters': self.get_counters(),
            'gauges': self.get_gauges()
        }
    
    def shutdown(self):
        """Shutdown metrics collection."""
        self._collection_active = False
        self._collection_executor.shutdown(wait=True)
        logger.info("Metrics collection shutdown complete")


class PerformanceTracker:
    """Track and analyze performance patterns."""
    
    def __init__(self):
        self.operation_times = defaultdict(list)
        self.operation_counts = defaultdict(int)
        self._lock = threading.Lock()
    
    def track_operation(self, operation: str, duration_ms: float, success: bool = True):
        """Track operation performance."""
        
        with self._lock:
            self.operation_times[operation].append({
                'duration_ms': duration_ms,
                'success': success,
                'timestamp': datetime.utcnow()
            })
            
            self.operation_counts[operation] += 1
            
            # Keep only recent entries to prevent memory growth
            if len(self.operation_times[operation]) > 1000:
                self.operation_times[operation] = self.operation_times[operation][-1000:]
    
    def get_operation_stats(self, operation: str, time_window: timedelta = None) -> Dict[str, Any]:
        """Get performance statistics for an operation."""
        
        with self._lock:
            data = self.operation_times.get(operation, [])
            
            if time_window:
                cutoff = datetime.utcnow() - time_window
                data = [item for item in data if item['timestamp'] >= cutoff]
            
            if not data:
                return {}
            
            durations = [item['duration_ms'] for item in data]
            success_count = sum(1 for item in data if item['success'])
            
            return {
                'operation': operation,
                'total_calls': len(data),
                'success_rate': success_count / len(data) if data else 0,
                'avg_duration': statistics.mean(durations) if durations else 0,
                'median_duration': statistics.median(durations) if durations else 0,
                'min_duration': min(durations) if durations else 0,
                'max_duration': max(durations) if durations else 0,
                'p95_duration': self._percentile(durations, 0.95) if durations else 0,
                'p99_duration': self._percentile(durations, 0.99) if durations else 0
            }
    
    def get_all_stats(self, time_window: timedelta = None) -> Dict[str, Dict[str, Any]]:
        """Get stats for all tracked operations."""
        
        stats = {}
        
        for operation in self.operation_times.keys():
            stats[operation] = self.get_operation_stats(operation, time_window)
        
        return stats
    
    def _percentile(self, data: List[float], percentile: float) -> float:
        """Calculate percentile."""
        if not data:
            return 0
        
        sorted_data = sorted(data)
        index = int(percentile * len(sorted_data))
        
        if index >= len(sorted_data):
            index = len(sorted_data) - 1
        
        return sorted_data[index]


class AlertManager:
    """Manage system alerts and notifications."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.alert_thresholds = {
            'error_rate': 0.05,  # 5% error rate
            'avg_response_time': 5000,  # 5 seconds
            'fraud_processing_time': 1000,  # 1 second
            'memory_usage': 80.0  # 80% memory usage
        }
        self.alert_history = deque(maxlen=1000)
        self._lock = threading.Lock()
    
    def check_alerts(self) -> List[Dict[str, Any]]:
        """Check for alert conditions."""
        
        alerts = []
        now = datetime.utcnow()
        
        # Check request metrics
        request_stats = self.metrics_collector.get_request_stats(timedelta(minutes=5))
        
        if request_stats.get('error_rate', 0) > self.alert_thresholds['error_rate']:
            alerts.append({
                'type': 'HIGH_ERROR_RATE',
                'message': f"Error rate {request_stats['error_rate']:.1%} exceeds threshold {self.alert_thresholds['error_rate']:.1%}",
                'severity': 'WARNING',
                'timestamp': now,
                'value': request_stats['error_rate']
            })
        
        if request_stats.get('avg_response_time', 0) > self.alert_thresholds['avg_response_time']:
            alerts.append({
                'type': 'HIGH_RESPONSE_TIME',
                'message': f"Average response time {request_stats['avg_response_time']:.1f}ms exceeds threshold {self.alert_thresholds['avg_response_time']:.1f}ms",
                'severity': 'WARNING',
                'timestamp': now,
                'value': request_stats['avg_response_time']
            })
        
        # Check fraud detection metrics
        fraud_stats = self.metrics_collector.get_fraud_stats(timedelta(minutes=5))
        
        if fraud_stats.get('avg_processing_time', 0) > self.alert_thresholds['fraud_processing_time']:
            alerts.append({
                'type': 'SLOW_FRAUD_DETECTION',
                'message': f"Fraud detection processing time {fraud_stats['avg_processing_time']:.1f}ms exceeds threshold {self.alert_thresholds['fraud_processing_time']:.1f}ms",
                'severity': 'WARNING',
                'timestamp': now,
                'value': fraud_stats['avg_processing_time']
            })
        
        # Store alerts
        with self._lock:
            for alert in alerts:
                self.alert_history.append(alert)
        
        return alerts
    
    def get_recent_alerts(self, time_window: timedelta = None) -> List[Dict[str, Any]]:
        """Get recent alerts."""
        
        with self._lock:
            alerts = list(self.alert_history)
        
        if time_window:
            cutoff = datetime.utcnow() - time_window
            alerts = [alert for alert in alerts if alert['timestamp'] >= cutoff]
        
        return alerts