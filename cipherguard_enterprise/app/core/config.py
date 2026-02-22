"""
Enterprise-grade configuration management with environment-based settings,
security, and comprehensive validation for production deployment.
"""

from functools import lru_cache
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseSettings, Field, validator, root_validator
from pydantic.env_settings import SettingsSourceCallable
import os
from pathlib import Path
import logging.config
import json


class SecuritySettings(BaseSettings):
    """Security configuration settings."""
    
    # Authentication & Authorization
    secret_key: str = Field(..., env="SECRET_KEY")
    jwt_secret_key: str = Field(..., env="JWT_SECRET_KEY") 
    jwt_algorithm: str = Field(default="HS256", env="JWT_ALGORITHM")
    jwt_access_token_expire_minutes: int = Field(default=30, env="JWT_ACCESS_TOKEN_EXPIRE_MINUTES")
    
    # API Security
    api_key: str = Field(..., env="API_KEY")
    api_key_header_name: str = Field(default="X-API-Key", env="API_KEY_HEADER_NAME")
    
    # Rate limiting
    rate_limit_enabled: bool = Field(default=True, env="RATE_LIMIT_ENABLED")
    rate_limit_calls: int = Field(default=100, env="RATE_LIMIT_CALLS")
    rate_limit_period: int = Field(default=60, env="RATE_LIMIT_PERIOD")
    
    # CORS
    cors_origins: List[str] = Field(default=["*"], env="CORS_ORIGINS")
    cors_allow_credentials: bool = Field(default=True, env="CORS_ALLOW_CREDENTIALS")
    cors_allow_methods: List[str] = Field(default=["*"], env="CORS_ALLOW_METHODS")
    cors_allow_headers: List[str] = Field(default=["*"], env="CORS_ALLOW_HEADERS")
    
    @validator('secret_key', 'jwt_secret_key', 'api_key')
    def validate_security_keys(cls, v: str) -> str:
        if len(v) < 32:
            raise ValueError("Security keys must be at least 32 characters long")
        return v
    
    class Config:
        env_prefix = "SECURITY_"
        case_sensitive = False


class DatabaseSettings(BaseSettings):
    """Database configuration settings."""
    
    # Connection
    url: str = Field(..., env="DATABASE_URL")
    driver: str = Field(default="postgresql+asyncpg", env="DATABASE_DRIVER")
    
    # Connection pooling
    pool_size: int = Field(default=20, env="DATABASE_POOL_SIZE")
    max_overflow: int = Field(default=30, env="DATABASE_MAX_OVERFLOW")
    pool_timeout: int = Field(default=30, env="DATABASE_POOL_TIMEOUT")
    pool_recycle: int = Field(default=3600, env="DATABASE_POOL_RECYCLE")
    
    # Query optimization
    echo: bool = Field(default=False, env="DATABASE_ECHO")
    echo_pool: bool = Field(default=False, env="DATABASE_ECHO_POOL")
    
    # Connection health
    pool_pre_ping: bool = Field(default=True, env="DATABASE_POOL_PRE_PING")
    
    @validator('url')
    def validate_database_url(cls, v: str) -> str:
        if not v.startswith(('postgresql://', 'postgresql+asyncpg://', 'sqlite:///')):
            raise ValueError("Database URL must use postgresql or sqlite")
        return v
    
    class Config:
        env_prefix = "DATABASE_"
        case_sensitive = False


class MLSettings(BaseSettings):
    """Machine Learning configuration settings."""
    
    # Model paths and storage
    models_dir: str = Field(default="./models", env="ML_MODELS_DIR")
    data_dir: str = Field(default="./data", env="ML_DATA_DIR")
    
    # Training configuration
    train_test_split: float = Field(default=0.2, env="ML_TRAIN_TEST_SPLIT")
    random_state: int = Field(default=42, env="ML_RANDOM_STATE")
    n_jobs: int = Field(default=-1, env="ML_N_JOBS")
    
    # Model selection
    enable_isolation_forest: bool = Field(default=True, env="ML_ENABLE_ISOLATION_FOREST")
    enable_random_forest: bool = Field(default=True, env="ML_ENABLE_RANDOM_FOREST")
    enable_xgboost: bool = Field(default=True, env="ML_ENABLE_XGBOOST")
    enable_ensemble: bool = Field(default=True, env="ML_ENABLE_ENSEMBLE")
    
    # Hyperparameter tuning
    enable_hyperparameter_tuning: bool = Field(default=True, env="ML_ENABLE_HYPERPARAMETER_TUNING")
    hyperparameter_cv_folds: int = Field(default=5, env="ML_HYPERPARAMETER_CV_FOLDS")
    hyperparameter_n_jobs: int = Field(default=-1, env="ML_HYPERPARAMETER_N_JOBS")
    
    # Feature engineering
    feature_selection_enabled: bool = Field(default=True, env="ML_FEATURE_SELECTION_ENABLED")
    feature_importance_threshold: float = Field(default=0.01, env="ML_FEATURE_IMPORTANCE_THRESHOLD")
    
    # Model performance thresholds
    min_accuracy_threshold: float = Field(default=0.85, env="ML_MIN_ACCURACY_THRESHOLD")
    min_f1_threshold: float = Field(default=0.80, env="ML_MIN_F1_THRESHOLD")
    
    # Fraud detection thresholds
    fraud_threshold: float = Field(default=0.5, env="ML_FRAUD_THRESHOLD")
    high_risk_threshold: float = Field(default=0.7, env="ML_HIGH_RISK_THRESHOLD")
    low_risk_threshold: float = Field(default=0.3, env="ML_LOW_RISK_THRESHOLD")
    
    @validator('train_test_split', 'fraud_threshold', 'high_risk_threshold', 'low_risk_threshold')
    def validate_thresholds(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("Threshold values must be between 0.0 and 1.0")
        return v
    
    class Config:
        env_prefix = "ML_"
        case_sensitive = False


class APISettings(BaseSettings):
    """API server configuration settings."""
    
    # Server
    host: str = Field(default="0.0.0.0", env="API_HOST")
    port: int = Field(default=8000, env="API_PORT")
    reload: bool = Field(default=False, env="API_RELOAD")
    workers: int = Field(default=1, env="API_WORKERS")
    
    # Application
    title: str = Field(default="CipherGuard Fraud Detection API", env="API_TITLE")
    description: str = Field(default="Enterprise-grade fraud detection platform", env="API_DESCRIPTION")
    version: str = Field(default="2.0.0", env="API_VERSION")
    
    # Features
    enable_docs: bool = Field(default=True, env="API_ENABLE_DOCS")
    enable_openapi: bool = Field(default=True, env="API_ENABLE_OPENAPI")
    docs_url: Optional[str] = Field(default="/docs", env="API_DOCS_URL")
    redoc_url: Optional[str] = Field(default="/redoc", env="API_REDOC_URL")
    openapi_url: Optional[str] = Field(default="/openapi.json", env="API_OPENAPI_URL")
    
    # Request handling
    max_request_size: int = Field(default=1024 * 1024, env="API_MAX_REQUEST_SIZE")  # 1MB
    request_timeout: int = Field(default=30, env="API_REQUEST_TIMEOUT")
    
    class Config:
        env_prefix = "API_"
        case_sensitive = False


class LoggingSettings(BaseSettings):
    """Logging configuration settings."""
    
    # Basic settings
    level: str = Field(default="INFO", env="LOG_LEVEL")
    format: str = Field(default="json", env="LOG_FORMAT")  # json or text
    
    # File logging
    enable_file_logging: bool = Field(default=True, env="LOG_ENABLE_FILE_LOGGING")
    log_dir: str = Field(default="./logs", env="LOG_DIR")
    max_file_size_mb: int = Field(default=100, env="LOG_MAX_FILE_SIZE_MB")
    backup_count: int = Field(default=5, env="LOG_BACKUP_COUNT")
    
    # Request correlation
    enable_correlation_id: bool = Field(default=True, env="LOG_ENABLE_CORRELATION_ID")
    correlation_id_header: str = Field(default="X-Correlation-ID", env="LOG_CORRELATION_ID_HEADER")
    
    # Audit logging
    enable_audit_logging: bool = Field(default=True, env="LOG_ENABLE_AUDIT_LOGGING")
    audit_log_file: str = Field(default="audit.log", env="LOG_AUDIT_LOG_FILE")
    
    @validator('level')
    def validate_log_level(cls, v: str) -> str:
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of: {valid_levels}")
        return v.upper()
    
    @validator('format')
    def validate_log_format(cls, v: str) -> str:
        valid_formats = ['json', 'text']
        if v.lower() not in valid_formats:
            raise ValueError(f"Log format must be one of: {valid_formats}")
        return v.lower()
    
    class Config:
        env_prefix = "LOG_"
        case_sensitive = False


class MonitoringSettings(BaseSettings):
    """Monitoring and metrics configuration."""
    
    # Metrics collection
    enable_metrics: bool = Field(default=True, env="MONITORING_ENABLE_METRICS")
    metrics_port: int = Field(default=9090, env="MONITORING_METRICS_PORT")
    
    # Health checks
    enable_health_checks: bool = Field(default=True, env="MONITORING_ENABLE_HEALTH_CHECKS")
    health_check_interval: int = Field(default=30, env="MONITORING_HEALTH_CHECK_INTERVAL")
    
    # Performance tracking  
    enable_performance_tracking: bool = Field(default=True, env="MONITORING_ENABLE_PERFORMANCE_TRACKING")
    slow_request_threshold_ms: int = Field(default=1000, env="MONITORING_SLOW_REQUEST_THRESHOLD_MS")
    
    # Alerting
    enable_alerting: bool = Field(default=True, env="MONITORING_ENABLE_ALERTING")
    alert_threshold_error_rate: float = Field(default=0.05, env="MONITORING_ALERT_THRESHOLD_ERROR_RATE")
    alert_threshold_response_time_ms: int = Field(default=2000, env="MONITORING_ALERT_THRESHOLD_RESPONSE_TIME_MS")
    
    class Config:
        env_prefix = "MONITORING_"
        case_sensitive = False


class Settings(BaseSettings):
    """Main application settings combining all configuration sections."""
    
    # Environment
    environment: str = Field(default="development", env="ENVIRONMENT")
    debug: bool = Field(default=False, env="DEBUG")
    testing: bool = Field(default=False, env="TESTING")
    
    # Configuration sections
    security: SecuritySettings = SecuritySettings()
    database: DatabaseSettings = DatabaseSettings()
    ml: MLSettings = MLSettings()
    api: APISettings = APISettings()
    logging: LoggingSettings = LoggingSettings()
    monitoring: MonitoringSettings = MonitoringSettings()
    
    # Application metadata
    app_name: str = Field(default="CipherGuard Fraud Detection", env="APP_NAME")
    app_version: str = Field(default="2.0.0", env="APP_VERSION")
    
    @validator('environment')
    def validate_environment(cls, v: str) -> str:
        valid_environments = ['development', 'staging', 'production']
        if v.lower() not in valid_environments:
            raise ValueError(f"Environment must be one of: {valid_environments}")
        return v.lower()
    
    @root_validator
    def validate_environment_specific_settings(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Validate environment-specific configuration rules."""
        environment = values.get('environment', 'development')
        
        if environment == 'production':
            # Production safety checks
            if values.get('debug', False):
                raise ValueError("Debug mode cannot be enabled in production")
            
            # Ensure secure configuration in production
            api_settings = values.get('api', {})
            if isinstance(api_settings, APISettings):
                if api_settings.enable_docs and environment == 'production':
                    # Warning, but don't fail - let ops decide
                    pass
        
        return values
    
    def get_database_url(self, async_driver: bool = True) -> str:
        """Get the appropriate database URL for sync/async usage."""
        if async_driver and 'postgresql' in self.database.url:
            return self.database.url.replace('postgresql://', 'postgresql+asyncpg://')
        return self.database.url
    
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.environment == 'development'
    
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.environment == 'production'
    
    def get_cors_origins(self) -> List[str]:
        """Get CORS origins with environment-specific defaults."""
        if self.is_development():
            return ["http://localhost:3000", "http://127.0.0.1:3000"] + self.security.cors_origins
        return self.security.cors_origins
    
    class Config:
        case_sensitive = False
        env_file = ".env"
        env_file_encoding = "utf-8"
        
        @classmethod
        def customise_sources(
            cls,
            init_settings: SettingsSourceCallable,
            env_settings: SettingsSourceCallable,
            file_secret_settings: SettingsSourceCallable,
        ) -> tuple[SettingsSourceCallable, ...]:
            """Customize settings sources priority."""
            return (
                init_settings,
                env_settings,
                file_secret_settings,
            )


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached application settings instance.
    
    Uses LRU cache to ensure settings are loaded only once per process.
    This is the recommended approach for FastAPI dependency injection.
    """
    return Settings()


def create_directories(settings: Settings) -> None:
    """Create required directories if they don't exist."""
    directories = [
        settings.logging.log_dir,
        settings.ml.models_dir,
        settings.ml.data_dir,
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)


def validate_configuration(settings: Settings) -> None:
    """
    Validate configuration for common issues and environment-specific requirements.
    
    Args:
        settings: Application settings instance
        
    Raises:
        ValueError: If configuration validation fails
    """
    # Validate required directories exist or can be created
    create_directories(settings)
    
    # Environment-specific validations
    if settings.is_production():
        # Production-specific validations
        if not settings.security.secret_key or len(settings.security.secret_key) < 32:
            raise ValueError("Production requires a strong secret key (32+ characters)")
        
        if settings.debug:
            raise ValueError("Debug mode must be disabled in production")
        
        if not settings.database.url.startswith('postgresql'):
            raise ValueError("Production requires PostgreSQL database")
    
    # Database connection validation
    try:
        from urllib.parse import urlparse
        parsed_url = urlparse(settings.database.url)
        if not parsed_url.scheme:
            raise ValueError("Invalid database URL format")
    except Exception as e:
        raise ValueError(f"Database URL validation failed: {e}")
    
    # ML configuration validation
    if not any([
        settings.ml.enable_isolation_forest,
        settings.ml.enable_random_forest,
        settings.ml.enable_xgboost
    ]):
        raise ValueError("At least one ML model must be enabled")


# Export main settings function and validation
__all__ = ['get_settings', 'Settings', 'validate_configuration', 'create_directories']