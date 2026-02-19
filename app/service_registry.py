"""
Improved dependency management with proper service locator pattern
"""

from typing import Protocol, Optional, Dict, Any
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ServiceConfig:
    """Configuration for a service dependency."""
    name: str
    required: bool = False
    fallback: Optional[Any] = None

class ServiceRegistry:
    """Centralized service dependency registry."""
    
    def __init__(self):
        self._services: Dict[str, Any] = {}
        self._configs: Dict[str, ServiceConfig] = {}
        
    def register_config(self, service_name: str, config: ServiceConfig):
        """Register service configuration."""
        self._configs[service_name] = config
        
    def register_service(self, service_name: str, service: Any):
        """Register a service instance."""
        self._services[service_name] = service
        logger.info(f"Service '{service_name}' registered successfully")
        
    def get_service(self, service_name: str) -> Optional[Any]:
        """Get a service instance with fallback handling."""
        if service_name in self._services:
            return self._services[service_name]
            
        config = self._configs.get(service_name)
        if config and not config.required:
            logger.warning(f"Service '{service_name}' not available, using fallback")
            return config.fallback
            
        if config and config.required:
            raise RuntimeError(f"Required service '{service_name}' not available")
            
        return None

# Global service registry
services = ServiceRegistry()

def init_services():
    """Initialize all services with proper error handling."""
    
    # Register configurations
    services.register_config("cyborg_client", ServiceConfig("cyborg_client", required=False))
    services.register_config("metrics", ServiceConfig("metrics", required=False))
    services.register_config("security", ServiceConfig("security", required=True))
    
    # Try to initialize each service
    _init_cyborg_client()
    _init_metrics()  
    _init_security()

def _init_cyborg_client():
    """Initialize CyborgDB client."""
    try:
        from .cyborg_client import get_cyborg_client
        client = get_cyborg_client()
        services.register_service("cyborg_client", client)
    except ImportError:
        logger.warning("CyborgDB not available - using mock implementation")
        services.register_service("cyborg_client", None)

def _init_metrics():
    """Initialize metrics collection."""
    try:
        from .monitoring import get_metrics_collector
        collector = get_metrics_collector()
        services.register_service("metrics", collector)
    except ImportError:
        logger.warning("Metrics collection not available")

def _init_security():
    """Initialize security services."""
    try:
        from .enterprise_security import get_security_manager
        security = get_security_manager()
        services.register_service("security", security)
    except ImportError as e:
        logger.error(f"Security services failed to initialize: {e}")
        # Security is required, so we should fail fast
        raise