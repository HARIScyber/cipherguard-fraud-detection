"""
Monitoring Module - Phase 4: Production Deployment
Prometheus metrics and health monitoring for production deployment
"""

import time
import psutil
import logging
from typing import Dict, Any, Optional
from prometheus_client import Counter, Histogram, Gauge, Info, generate_latest
from functools import wraps
import threading
import os

logger = logging.getLogger(__name__)

class MetricsCollector:
    """Collects and exposes Prometheus metrics for the fraud detection system."""

    def __init__(self):
        # Request metrics
        self.requests_total = Counter(
            'fraud_detection_requests_total',
            'Total number of fraud detection requests',
            ['method', 'endpoint', 'status']
        )

        self.requests_duration = Histogram(
            'fraud_detection_request_duration_seconds',
            'Request duration in seconds',
            ['method', 'endpoint'],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        )

        # Model metrics
        self.model_predictions_total = Counter(
            'fraud_detection_model_predictions_total',
            'Total number of model predictions',
            ['model_name', 'model_type']
        )

        self.model_prediction_duration = Histogram(
            'fraud_detection_model_prediction_duration_seconds',
            'Model prediction duration in seconds',
            ['model_name', 'model_type'],
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
        )

        # Fraud detection metrics
        self.fraud_detected_total = Counter(
            'fraud_detection_fraud_detected_total',
            'Total number of fraud cases detected',
            ['risk_level', 'model_used']
        )

        self.fraud_score_distribution = Histogram(
            'fraud_detection_fraud_score',
            'Distribution of fraud scores',
            ['model_used'],
            buckets=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        )

        # System metrics
        self.memory_usage = Gauge(
            'fraud_detection_memory_usage_bytes',
            'Current memory usage in bytes'
        )

        self.cpu_usage = Gauge(
            'fraud_detection_cpu_usage_percent',
            'Current CPU usage percentage'
        )

        self.active_connections = Gauge(
            'fraud_detection_active_connections',
            'Number of active connections'
        )

        # Kafka metrics
        self.kafka_messages_processed = Counter(
            'fraud_detection_kafka_messages_processed_total',
            'Total number of Kafka messages processed',
            ['topic', 'status']
        )

        self.kafka_message_processing_duration = Histogram(
            'fraud_detection_kafka_message_processing_duration_seconds',
            'Kafka message processing duration',
            ['topic'],
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
        )

        # Model health metrics
        self.model_accuracy = Gauge(
            'fraud_detection_model_accuracy',
            'Model accuracy score',
            ['model_name']
        )

        self.model_last_updated = Gauge(
            'fraud_detection_model_last_updated_timestamp',
            'Timestamp of last model update',
            ['model_name']
        )

        # Service info
        self.service_info = Info(
            'fraud_detection_service_info',
            'Service information'
        )
        self.service_info.info({
            'version': '4.0.0',
            'phase': 'production_deployment',
            'service': 'fraud-detection'
        })

        # Start system metrics collection
        self._start_system_metrics_collection()

    def _start_system_metrics_collection(self):
        """Start background thread to collect system metrics."""
        def collect_system_metrics():
            while True:
                try:
                    # Memory usage
                    memory = psutil.virtual_memory()
                    self.memory_usage.set(memory.used)

                    # CPU usage
                    cpu_percent = psutil.cpu_percent(interval=1)
                    self.cpu_usage.set(cpu_percent)

                    # Disk usage (optional)
                    # disk = psutil.disk_usage('/')
                    # self.disk_usage.set(disk.percent)

                except Exception as e:
                    logger.error(f"Failed to collect system metrics: {e}")

                time.sleep(30)  # Collect every 30 seconds

        thread = threading.Thread(target=collect_system_metrics, daemon=True)
        thread.start()

    def record_request(self, method: str, endpoint: str, status: str, duration: float):
        """Record an API request."""
        self.requests_total.labels(method=method, endpoint=endpoint, status=status).inc()
        self.requests_duration.labels(method=method, endpoint=endpoint).observe(duration)

    def record_model_prediction(self, model_name: str, model_type: str, duration: float):
        """Record a model prediction."""
        self.model_predictions_total.labels(model_name=model_name, model_type=model_type).inc()
        self.model_prediction_duration.labels(model_name=model_name, model_type=model_type).observe(duration)

    def record_fraud_detection(self, fraud_score: float, risk_level: str, model_used: str):
        """Record fraud detection result."""
        self.fraud_detected_total.labels(risk_level=risk_level, model_used=model_used).inc()
        self.fraud_score_distribution.labels(model_used=model_used).observe(fraud_score)

    def record_kafka_message(self, topic: str, status: str, duration: Optional[float] = None):
        """Record Kafka message processing."""
        self.kafka_messages_processed.labels(topic=topic, status=status).inc()
        if duration is not None:
            self.kafka_message_processing_duration.labels(topic=topic).observe(duration)

    def update_model_metrics(self, model_name: str, accuracy: Optional[float] = None):
        """Update model performance metrics."""
        if accuracy is not None:
            self.model_accuracy.labels(model_name=model_name).set(accuracy)

        self.model_last_updated.labels(model_name=model_name).set(time.time())

    def get_metrics(self) -> str:
        """Get all metrics in Prometheus format."""
        return generate_latest().decode('utf-8')

# Global metrics collector
metrics_collector = MetricsCollector()

def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector instance."""
    return metrics_collector

# Decorators for automatic metrics collection
def track_request(method: str, endpoint: str):
    """Decorator to track API requests."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                metrics_collector.record_request(method, endpoint, "success", duration)
                return result
            except Exception as e:
                duration = time.time() - start_time
                metrics_collector.record_request(method, endpoint, "error", duration)
                raise e
        return wrapper
    return decorator

def track_model_prediction(model_name: str, model_type: str):
    """Decorator to track model predictions."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                metrics_collector.record_model_prediction(model_name, model_type, duration)
                return result
            except Exception as e:
                duration = time.time() - start_time
                metrics_collector.record_model_prediction(model_name, model_type, duration)
                raise e
        return wrapper
    return decorator

def track_fraud_detection(model_used: str):
    """Decorator to track fraud detection results."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            if isinstance(result, dict) and 'fraud_score' in result:
                metrics_collector.record_fraud_detection(
                    result['fraud_score'],
                    result.get('risk_level', 'unknown'),
                    model_used
                )
            return result
        return wrapper
    return decorator