"""
Production monitoring and performance optimization
"""

import time
import asyncio
import psutil
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import logging
from dataclasses import dataclass
import numpy as np
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('http_request_duration_seconds', 'HTTP request duration', ['method', 'endpoint'])
FRAUD_DETECTION_DURATION = Histogram('fraud_detection_duration_seconds', 'Fraud detection processing time')
ACTIVE_CONNECTIONS = Gauge('active_connections', 'Number of active connections')
MEMORY_USAGE = Gauge('memory_usage_bytes', 'Memory usage in bytes')
CPU_USAGE = Gauge('cpu_usage_percent', 'CPU usage percentage')

# Custom metrics
FRAUD_DETECTIONS = Counter('fraud_detections_total', 'Total fraud detections', ['risk_level'])
MODEL_PREDICTIONS = Counter('model_predictions_total', 'Model prediction counts', ['model_type'])
CACHE_HITS = Counter('cache_hits_total', 'Cache hit counts', ['cache_type'])
CACHE_MISSES = Counter('cache_misses_total', 'Cache miss counts', ['cache_type'])

@dataclass
class PerformanceMetrics:
    """Performance metrics data structure."""
    timestamp: datetime
    request_count: int
    avg_response_time_ms: float
    memory_usage_mb: float
    cpu_usage_percent: float
    active_connections: int
    fraud_detection_rate: float
    error_rate: float

class MetricsCollector:
    """Collects and manages application metrics."""
    
    def __init__(self):
        self.start_time = datetime.utcnow()
        self.request_times: List[float] = []
        self.error_count = 0
        self.total_requests = 0
        
    async def collect_system_metrics(self):
        """Collect system-level metrics using psutil."""
        try:
            # Memory metrics
            memory = psutil.virtual_memory()
            MEMORY_USAGE.set(memory.used)
            
            # CPU metrics  
            cpu_percent = psutil.cpu_percent(interval=1)
            CPU_USAGE.set(cpu_percent)
            
            # Network connections (approximate)
            connections = len(psutil.net_connections())
            ACTIVE_CONNECTIONS.set(connections)
            
            logger.debug(f"System metrics - Memory: {memory.used//1024//1024}MB, CPU: {cpu_percent}%")
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
    
    def record_request(self, method: str, endpoint: str, status_code: int, duration: float):
        """Record request metrics."""
        REQUEST_COUNT.labels(method=method, endpoint=endpoint, status=str(status_code)).inc()
        REQUEST_DURATION.labels(method=method, endpoint=endpoint).observe(duration)
        
        self.total_requests += 1
        self.request_times.append(duration)
        
        if status_code >= 400:
            self.error_count += 1
    
    def record_fraud_detection(self, duration: float, risk_level: str, model_type: str):
        """Record fraud detection specific metrics."""
        FRAUD_DETECTION_DURATION.observe(duration)
        FRAUD_DETECTIONS.labels(risk_level=risk_level).inc() 
        MODEL_PREDICTIONS.labels(model_type=model_type).inc()
    
    def record_cache_operation(self, cache_type: str, hit: bool):
        """Record cache hit/miss metrics."""
        if hit:
            CACHE_HITS.labels(cache_type=cache_type).inc()
        else:
            CACHE_MISSES.labels(cache_type=cache_type).inc()
    
    def get_performance_summary(self) -> PerformanceMetrics:
        """Get current performance summary."""
        now = datetime.utcnow()
        uptime = (now - self.start_time).total_seconds()
        
        avg_response_time = np.mean(self.request_times[-1000:]) if self.request_times else 0
        error_rate = (self.error_count / max(self.total_requests, 1)) * 100
        
        return PerformanceMetrics(
            timestamp=now,
            request_count=self.total_requests,
            avg_response_time_ms=avg_response_time * 1000,
            memory_usage_mb=psutil.virtual_memory().used // 1024 // 1024,
            cpu_usage_percent=psutil.cpu_percent(),
            active_connections=len(psutil.net_connections()),
            fraud_detection_rate=0.0,  # Calculate from recent detections
            error_rate=error_rate
        )

class PerformanceOptimizer:
    """Handles performance optimization strategies."""
    
    def __init__(self):
        self.cache = {}
        self.cache_ttl = {}
        self.feature_cache = {}
        
    async def cached_feature_extraction(self, transaction_data: Dict) -> np.ndarray:
        """Cache feature extraction results."""
        # Create cache key from transaction data
        cache_key = self._create_cache_key(transaction_data)
        
        if cache_key in self.feature_cache:
            cache_time = self.cache_ttl.get(cache_key, 0)
            if time.time() - cache_time < 300:  # 5 minute TTL
                CACHE_HITS.labels(cache_type='features').inc()
                return self.feature_cache[cache_key]
        
        # Cache miss - calculate features
        CACHE_MISSES.labels(cache_type='features').inc()
        
        from .feature_extraction import FeatureExtractor
        extractor = FeatureExtractor()
        features = extractor.extract_features(transaction_data)
        
        # Cache the result
        self.feature_cache[cache_key] = features
        self.cache_ttl[cache_key] = time.time()
        
        return features
    
    def _create_cache_key(self, data: Dict) -> str:
        """Create deterministic cache key from transaction data."""
        # Sort keys for consistent hashing
        sorted_data = sorted(data.items())
        key_str = str(sorted_data)
        return str(hash(key_str))
    
    async def optimize_model_inference(self, features: np.ndarray):
        """Optimize model inference with batching and caching."""
        # Implement model inference optimizations
        # - Batch processing for multiple requests
        # - Model output caching
        # - ONNX runtime optimization
        pass
    
    def cleanup_cache(self):
        """Clean up expired cache entries."""
        current_time = time.time()
        expired_keys = [
            key for key, cache_time in self.cache_ttl.items()
            if current_time - cache_time > 300
        ]
        
        for key in expired_keys:
            self.feature_cache.pop(key, None)
            self.cache_ttl.pop(key, None)

class HealthChecker:
    """Health check implementation with detailed status."""
    
    def __init__(self):
        self.checks = {}
        
    def register_check(self, name: str, check_func):
        """Register a health check function."""
        self.checks[name] = check_func
    
    async def run_all_checks(self) -> Dict[str, any]:
        """Run all registered health checks."""
        results = {}
        overall_healthy = True
        
        for name, check_func in self.checks.items():
            try:
                start_time = time.time()
                result = await check_func() if asyncio.iscoroutinefunction(check_func) else check_func()
                duration = time.time() - start_time
                
                results[name] = {
                    "status": "healthy" if result else "unhealthy",
                    "duration_ms": round(duration * 1000, 2),
                    "timestamp": datetime.utcnow().isoformat()
                }
                
                if not result:
                    overall_healthy = False
                    
            except Exception as e:
                results[name] = {
                    "status": "error", 
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat()
                }
                overall_healthy = False
        
        return {
            "overall_status": "healthy" if overall_healthy else "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "checks": results
        }

# Database connection pool optimization
class DatabaseOptimizer:
    """Database performance optimization utilities."""
    
    @staticmethod
    def create_optimized_connection_pool(database_url: str, **kwargs):
        """Create optimized SQLAlchemy connection pool."""
        from sqlalchemy import create_engine
        from sqlalchemy.pool import QueuePool
        
        return create_engine(
            database_url,
            poolclass=QueuePool,
            pool_size=kwargs.get('pool_size', 10),
            max_overflow=kwargs.get('max_overflow', 20),
            pool_timeout=kwargs.get('pool_timeout', 30),
            pool_recycle=kwargs.get('pool_recycle', 3600),
            pool_pre_ping=True,  # Validate connections
            echo=kwargs.get('echo', False)
        )

# Async context manager for performance tracking
@asynccontextmanager
async def track_performance(operation_name: str, metrics_collector: MetricsCollector):
    """Context manager to track operation performance."""
    start_time = time.time()
    try:
        yield
    finally:
        duration = time.time() - start_time
        logger.debug(f"Operation '{operation_name}' took {duration:.3f}s")

# Global metrics collector instance
metrics_collector = MetricsCollector()
performance_optimizer = PerformanceOptimizer()
health_checker = HealthChecker()

# Background task for metrics collection
async def metrics_collection_task():
    """Background task to collect system metrics periodically."""
    while True:
        try:
            await metrics_collector.collect_system_metrics()
            performance_optimizer.cleanup_cache()
            await asyncio.sleep(30)  # Collect every 30 seconds
        except Exception as e:
            logger.error(f"Metrics collection error: {e}")
            await asyncio.sleep(60)  # Retry after 1 minute on error