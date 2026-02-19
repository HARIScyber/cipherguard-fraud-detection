"""
Centralized Logging Configuration for Enterprise Fraud Detection System
"""

import logging
import logging.config
from pathlib import Path
import sys
from typing import Dict, Any
import json
from datetime import datetime

from app.core.config import get_settings

settings = get_settings()


class JSONFormatter(logging.Formatter):
    """
    Custom JSON formatter for structured logging.
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        
        # Build log entry dictionary
        log_entry = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'line': record.lineno,
            'function': record.funcName,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields if present
        if hasattr(record, 'transaction_id'):
            log_entry['transaction_id'] = record.transaction_id
        
        if hasattr(record, 'user_id'):
            log_entry['user_id'] = record.user_id
            
        if hasattr(record, 'fraud_score'):
            log_entry['fraud_score'] = record.fraud_score
            
        if hasattr(record, 'processing_time_ms'):
            log_entry['processing_time_ms'] = record.processing_time_ms
        
        return json.dumps(log_entry)


class ServiceContextFilter(logging.Filter):
    """Add service context to log records."""
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Add service information to log record."""
        record.service_name = "fraud-detection-api"
        record.service_version = "2.0.0"
        record.environment = settings.general.environment
        return True


def setup_logging():
    """
    Setup centralized logging configuration.
    """
    
    # Ensure log directory exists
    log_dir = Path(settings.logging.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine log level
    log_level = getattr(logging, settings.logging.level.upper(), logging.INFO)
    
    # Build logging configuration
    logging_config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'json': {
                '()': JSONFormatter,
            },
            'detailed': {
                'format': '{asctime} - {name} - {levelname} - {module}:{lineno} - {message}',
                'style': '{',
                'datefmt': '%Y-%m-%d %H:%M:%S'
            },
            'simple': {
                'format': '{levelname}: {message}',
                'style': '{'
            }
        },
        'filters': {
            'service_context': {
                '()': ServiceContextFilter,
            }
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': log_level,
                'formatter': 'detailed' if settings.logging.format == 'text' else 'json',
                'stream': sys.stdout,
                'filters': ['service_context']
            },
            'file': {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': log_level,
                'formatter': 'json',
                'filename': str(log_dir / 'fraud-detection.log'),
                'maxBytes': settings.logging.max_file_size_mb * 1024 * 1024,  # Convert to bytes
                'backupCount': settings.logging.backup_count,
                'encoding': 'utf8',
                'filters': ['service_context']
            },
            'error_file': {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': 'ERROR',
                'formatter': 'json',
                'filename': str(log_dir / 'fraud-detection-errors.log'),
                'maxBytes': settings.logging.max_file_size_mb * 1024 * 1024,
                'backupCount': settings.logging.backup_count,
                'encoding': 'utf8',
                'filters': ['service_context']
            },
            'audit_file': {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': 'INFO',
                'formatter': 'json',
                'filename': str(log_dir / 'fraud-detection-audit.log'),
                'maxBytes': settings.logging.max_file_size_mb * 1024 * 1024,
                'backupCount': settings.logging.backup_count,
                'encoding': 'utf8',
                'filters': ['service_context']
            }
        },
        'loggers': {
            # Application loggers
            'app': {
                'level': log_level,
                'handlers': ['console', 'file'],
                'propagate': False
            },
            'database': {
                'level': log_level,
                'handlers': ['console', 'file'],
                'propagate': False
            },
            'ml': {
                'level': log_level,
                'handlers': ['console', 'file'],
                'propagate': False
            },
            
            # Audit logger for security events
            'audit': {
                'level': 'INFO',
                'handlers': ['audit_file'],
                'propagate': False
            },
            
            # Third-party library loggers
            'sqlalchemy': {
                'level': 'WARNING',
                'handlers': ['file'],
                'propagate': False
            },
            'sqlalchemy.engine': {
                'level': 'INFO' if settings.logging.enable_sql_debug else 'WARNING',
                'handlers': ['file'],
                'propagate': False
            },
            'uvicorn': {
                'level': 'INFO',
                'handlers': ['console', 'file'],
                'propagate': False
            },
            'uvicorn.access': {
                'level': 'INFO' if settings.logging.enable_access_logs else 'WARNING',
                'handlers': ['file'],
                'propagate': False
            },
            'fastapi': {
                'level': 'INFO',
                'handlers': ['console', 'file'],
                'propagate': False
            }
        },
        'root': {
            'level': log_level,
            'handlers': ['console', 'error_file']
        }
    }
    
    # Apply logging configuration
    logging.config.dictConfig(logging_config)
    
    # Log startup message
    logger = logging.getLogger('app.core.logging_config')
    logger.info(f"Logging configured - Level: {settings.logging.level}, Format: {settings.logging.format}")
    logger.info(f"Log directory: {log_dir}")


def get_audit_logger():
    """Get audit logger for security events."""
    return logging.getLogger('audit')


def log_transaction_event(
    transaction_id: str,
    event_type: str,
    user_id: str = None,
    fraud_score: float = None,
    risk_level: str = None,
    additional_data: Dict[str, Any] = None
):
    """
    Log transaction-related events for audit trail.
    
    Args:
        transaction_id: Transaction identifier
        event_type: Type of event (e.g., 'fraud_detection', 'feedback_received')
        user_id: User identifier
        fraud_score: Fraud probability score
        risk_level: Risk assessment level
        additional_data: Additional event data
    """
    
    audit_logger = get_audit_logger()
    
    event_data = {
        'event_type': event_type,
        'transaction_id': transaction_id,
        'timestamp': datetime.utcnow().isoformat() + 'Z'
    }
    
    if user_id:
        event_data['user_id'] = user_id
    
    if fraud_score is not None:
        event_data['fraud_score'] = fraud_score
        
    if risk_level:
        event_data['risk_level'] = risk_level
    
    if additional_data:
        event_data.update(additional_data)
    
    audit_logger.info(json.dumps(event_data))


def log_security_event(
    event_type: str,
    user_id: str = None,
    ip_address: str = None,
    user_agent: str = None,
    success: bool = True,
    additional_data: Dict[str, Any] = None
):
    """
    Log security-related events.
    
    Args:
        event_type: Type of security event
        user_id: User identifier (if applicable)
        ip_address: Client IP address
        user_agent: User agent string
        success: Whether the event was successful
        additional_data: Additional event data
    """
    
    audit_logger = get_audit_logger()
    
    event_data = {
        'event_type': event_type,
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'success': success
    }
    
    if user_id:
        event_data['user_id'] = user_id
        
    if ip_address:
        event_data['ip_address'] = ip_address
        
    if user_agent:
        event_data['user_agent'] = user_agent
    
    if additional_data:
        event_data.update(additional_data)
    
    audit_logger.info(json.dumps(event_data))


def get_performance_logger():
    """Get logger for performance metrics."""
    return logging.getLogger('app.performance')


def log_performance_metrics(
    operation: str,
    duration_ms: float,
    success: bool = True,
    additional_metrics: Dict[str, Any] = None
):
    """
    Log performance metrics for monitoring.
    
    Args:
        operation: Name of the operation
        duration_ms: Duration in milliseconds
        success: Whether operation was successful
        additional_metrics: Additional performance data
    """
    
    perf_logger = get_performance_logger()
    
    metrics_data = {
        'operation': operation,
        'duration_ms': duration_ms,
        'success': success,
        'timestamp': datetime.utcnow().isoformat() + 'Z'
    }
    
    if additional_metrics:
        metrics_data.update(additional_metrics)
    
    perf_logger.info(json.dumps(metrics_data))


class PerformanceContext:
    """Context manager for performance logging."""
    
    def __init__(self, operation: str, logger: logging.Logger = None):
        self.operation = operation
        self.logger = logger or get_performance_logger()
        self.start_time = None
        
    def __enter__(self):
        self.start_time = datetime.utcnow()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = (datetime.utcnow() - self.start_time).total_seconds() * 1000
        success = exc_type is None
        
        log_performance_metrics(
            operation=self.operation,
            duration_ms=duration,
            success=success
        )
        
        return False  # Don't suppress exceptions