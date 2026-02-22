"""
Enterprise-grade logging configuration with structured output, audit trails,
correlation tracking, and comprehensive monitoring capabilities.
"""

import logging
import logging.config
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
import sys
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import traceback
import uuid
from contextvars import ContextVar
from functools import wraps

from .config import get_settings

# Context variable for request correlation
correlation_id: ContextVar[Optional[str]] = ContextVar('correlation_id', default=None)


class CorrelationIDFormatter(logging.Formatter):
    """Custom formatter that includes correlation IDs in log records."""
    
    def format(self, record: logging.LogRecord) -> str:
        # Add correlation ID to record
        record.correlation_id = correlation_id.get() or 'no-correlation'
        return super().format(record)


class JSONFormatter(CorrelationIDFormatter):
    """
    Custom JSON formatter for structured logging with enterprise requirements.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hostname = self._get_hostname()
    
    def _get_hostname(self) -> str:
        """Get hostname for logging context."""
        import socket
        try:
            return socket.gethostname()
        except Exception:
            return "unknown-host"
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON."""
        
        # Build the base log entry
        log_entry = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line_number': record.lineno,
            'process_id': record.process,
            'thread_id': record.thread,
            'correlation_id': getattr(record, 'correlation_id', None) or correlation_id.get(),
            'hostname': self.hostname,
        }
        
        # Add exception information if present
        if record.exc_info:
            log_entry['exception'] = {
                'type': record.exc_info[0].__name__ if record.exc_info[0] else None,
                'message': str(record.exc_info[1]) if record.exc_info[1] else None,
                'traceback': self.formatException(record.exc_info)
            }
        
        # Add extra fields from record
        extra_fields = {}
        for key, value in record.__dict__.items():
            if key not in {
                'name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 'filename',
                'module', 'lineno', 'funcName', 'created', 'msecs', 'relativeCreated',
                'thread', 'threadName', 'processName', 'process', 'message', 'exc_info',
                'exc_text', 'stack_info', 'correlation_id'
            }:
                extra_fields[key] = value
        
        if extra_fields:
            log_entry['extra'] = extra_fields
        
        return json.dumps(log_entry, default=str, ensure_ascii=False)


class TextFormatter(CorrelationIDFormatter):
    """
    Human-readable text formatter with correlation IDs.
    """
    
    def __init__(self):
        super().__init__(
            fmt='%(asctime)s | %(levelname)-8s | %(correlation_id)s | %(name)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )


class AuditLogger:
    """
    Specialized audit logger for security and business events.
    """
    
    def __init__(self):
        self.logger = logging.getLogger('audit')
    
    def log_fraud_decision(
        self, 
        transaction_id: str,
        user_id: str,
        fraud_score: float,
        is_fraud: bool,
        model_version: str,
        risk_factors: list = None,
        additional_data: dict = None
    ):
        """Log fraud detection decision for audit trail."""
        audit_data = {
            'event_type': 'fraud_decision',
            'transaction_id': transaction_id,
            'user_id': user_id,
            'fraud_score': fraud_score,
            'is_fraud': is_fraud,
            'model_version': model_version,
            'risk_factors': risk_factors or [],
            'timestamp': datetime.utcnow().isoformat() + 'Z',
        }
        
        if additional_data:
            audit_data['additional_data'] = additional_data
        
        self.logger.info(json.dumps(audit_data))
    
    def log_model_training(
        self,
        model_type: str,
        training_data_size: int,
        performance_metrics: dict,
        model_version: str,
        training_duration_seconds: float = None
    ):
        """Log model training events."""
        audit_data = {
            'event_type': 'model_training',
            'model_type': model_type,
            'training_data_size': training_data_size,
            'performance_metrics': performance_metrics,
            'model_version': model_version,
            'training_duration_seconds': training_duration_seconds,
            'timestamp': datetime.utcnow().isoformat() + 'Z',
        }
        
        self.logger.info(json.dumps(audit_data))
    
    def log_authentication(
        self,
        user_id: str = None,
        api_key_used: bool = False,
        jwt_used: bool = False,
        success: bool = True,
        failure_reason: str = None,
        ip_address: str = None,
        user_agent: str = None
    ):
        """Log authentication events."""
        audit_data = {
            'event_type': 'authentication',
            'user_id': user_id,
            'api_key_used': api_key_used,
            'jwt_used': jwt_used,
            'success': success,
            'failure_reason': failure_reason,
            'ip_address': ip_address,
            'user_agent': user_agent,
            'timestamp': datetime.utcnow().isoformat() + 'Z',
        }
        
        self.logger.info(json.dumps(audit_data))
    
    def log_system_event(
        self,
        event_type: str,
        description: str,
        severity: str = 'info',
        additional_data: dict = None
    ):
        """Log general system events."""
        audit_data = {
            'event_type': 'system_event',
            'system_event_type': event_type,
            'description': description,
            'severity': severity,
            'timestamp': datetime.utcnow().isoformat() + 'Z',
        }
        
        if additional_data:
            audit_data['additional_data'] = additional_data
        
        # Use appropriate log level based on severity
        log_method = getattr(self.logger, severity.lower(), self.logger.info)
        log_method(json.dumps(audit_data))


class PerformanceLogger:
    """
    Specialized logger for performance metrics and timing.
    """
    
    def __init__(self):
        self.logger = logging.getLogger('performance')
    
    def log_request_performance(
        self,
        endpoint: str,
        method: str,
        response_time_ms: float,
        status_code: int,
        request_size_bytes: int = None,
        response_size_bytes: int = None,
        user_id: str = None
    ):
        """Log API request performance metrics."""
        perf_data = {
            'metric_type': 'api_request_performance',
            'endpoint': endpoint,
            'method': method,
            'response_time_ms': response_time_ms,
            'status_code': status_code,
            'request_size_bytes': request_size_bytes,
            'response_size_bytes': response_size_bytes,
            'user_id': user_id,
            'timestamp': datetime.utcnow().isoformat() + 'Z',
        }
        
        self.logger.info(json.dumps(perf_data))
    
    def log_ml_performance(
        self,
        model_type: str,
        inference_time_ms: float,
        feature_extraction_time_ms: float = None,
        preprocessing_time_ms: float = None,
        transaction_id: str = None
    ):
        """Log ML model performance metrics."""
        perf_data = {
            'metric_type': 'ml_performance',
            'model_type': model_type,
            'inference_time_ms': inference_time_ms,
            'feature_extraction_time_ms': feature_extraction_time_ms,
            'preprocessing_time_ms': preprocessing_time_ms,
            'transaction_id': transaction_id,
            'timestamp': datetime.utcnow().isoformat() + 'Z',
        }
        
        self.logger.info(json.dumps(perf_data))


def setup_logging() -> None:
    """
    Set up comprehensive logging configuration for enterprise deployment.
    """
    settings = get_settings()
    
    # Create log directory
    log_dir = Path(settings.logging.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine formatters
    if settings.logging.format == 'json':
        formatter_class = JSONFormatter
        console_formatter = JSONFormatter()
    else:
        formatter_class = TextFormatter
        console_formatter = TextFormatter()
    
    file_formatter = JSONFormatter()  # Always use JSON for files
    
    # Configure logging
    logging_config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'console': {
                '()': console_formatter.__class__,
            },
            'file': {
                '()': JSONFormatter,
            },
            'audit': {
                '()': JSONFormatter,
            },
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': settings.logging.level,
                'formatter': 'console',
                'stream': sys.stdout,
            },
            'file': {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': settings.logging.level,
                'formatter': 'file',
                'filename': str(log_dir / 'cipherguard.log'),
                'maxBytes': settings.logging.max_file_size_mb * 1024 * 1024,
                'backupCount': settings.logging.backup_count,
                'encoding': 'utf8',
            },
            'error_file': {
                'class': 'logging.handlers.RotatingFileHandler', 
                'level': 'ERROR',
                'formatter': 'file',
                'filename': str(log_dir / 'errors.log'),
                'maxBytes': settings.logging.max_file_size_mb * 1024 * 1024,
                'backupCount': settings.logging.backup_count,
                'encoding': 'utf8',
            },
        },
        'loggers': {
            '': {  # Root logger
                'level': settings.logging.level,
                'handlers': ['console', 'file', 'error_file'],
                'propagate': False,
            },
            'uvicorn': {
                'level': 'INFO',
                'handlers': ['console', 'file'],
                'propagate': False,
            },
            'uvicorn.access': {
                'level': 'INFO',
                'handlers': ['file'],
                'propagate': False,
            },
            'sqlalchemy': {
                'level': 'WARNING',
                'handlers': ['file'],
                'propagate': False,
            },
            'sqlalchemy.engine': {
                'level': 'WARNING',
                'handlers': ['file'],
                'propagate': False,
            },
        },
    }
    
    # Add audit logging if enabled
    if settings.logging.enable_audit_logging:
        logging_config['handlers']['audit'] = {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': 'INFO',
            'formatter': 'audit',
            'filename': str(log_dir / settings.logging.audit_log_file),
            'maxBytes': settings.logging.max_file_size_mb * 1024 * 1024,
            'backupCount': settings.logging.backup_count,
            'encoding': 'utf8',
        }
        
        logging_config['loggers']['audit'] = {
            'level': 'INFO',
            'handlers': ['audit'],
            'propagate': False,
        }
        
        logging_config['loggers']['performance'] = {
            'level': 'INFO',
            'handlers': ['file', 'audit'],
            'propagate': False,
        }
    
    # Apply configuration
    logging.config.dictConfig(logging_config)
    
    # Log setup completion
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured - Level: {settings.logging.level}, Format: {settings.logging.format}")


def get_correlation_id() -> str:
    """Get current request correlation ID."""
    current_id = correlation_id.get()
    if not current_id:
        current_id = generate_correlation_id()
        correlation_id.set(current_id)
    return current_id


def generate_correlation_id() -> str:
    """Generate a new correlation ID."""
    return str(uuid.uuid4())


def set_correlation_id(correlation_id_value: str) -> None:
    """Set correlation ID for current context."""
    correlation_id.set(correlation_id_value)


def log_with_correlation(logger_name: str = None):
    """
    Decorator to automatically add correlation ID to log messages.
    
    Usage:
        @log_with_correlation('my.logger')
        def my_function():
            pass
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = logging.getLogger(logger_name or func.__module__)
            
            # Ensure correlation ID is set
            if not correlation_id.get():
                set_correlation_id(generate_correlation_id())
            
            try:
                result = func(*args, **kwargs)
                logger.debug(f"Function {func.__name__} completed successfully")
                return result
            except Exception as e:
                logger.error(f"Function {func.__name__} failed with error: {str(e)}")
                raise
        
        return wrapper
    return decorator


# Global logger instances
audit_logger = AuditLogger()
performance_logger = PerformanceLogger()


# Export key components
__all__ = [
    'setup_logging',
    'get_correlation_id', 
    'set_correlation_id',
    'generate_correlation_id',
    'log_with_correlation',
    'AuditLogger',
    'PerformanceLogger',
    'audit_logger',
    'performance_logger',
    'JSONFormatter',
    'TextFormatter'
]