"""
Multi-Tenant Architecture Module - Phase 6: Enterprise Integration
Scalable multi-tenant architecture for enterprise fraud detection
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
from contextvars import ContextVar
import threading
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

# Context variable for current tenant
current_tenant: ContextVar[Optional[str]] = ContextVar('current_tenant', default=None)

class TenantStatus(Enum):
    """Tenant status enumeration."""
    ACTIVE = "active"
    SUSPENDED = "suspended"
    INACTIVE = "inactive"
    PENDING = "pending"

class TenantTier(Enum):
    """Tenant service tiers."""
    BASIC = "basic"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"
    CUSTOM = "custom"

@dataclass
class TenantConfig:
    """Tenant configuration."""
    tenant_id: str
    name: str
    status: str
    tier: str
    created_at: datetime
    updated_at: datetime
    settings: Dict[str, Any]
    limits: Dict[str, Any]
    features: List[str]
    domains: List[str]
    admin_users: List[str]

@dataclass
class TenantMetrics:
    """Tenant usage metrics."""
    tenant_id: str
    period_start: datetime
    period_end: datetime
    api_calls: int
    data_processed_gb: float
    fraud_detections: int
    alerts_sent: int
    storage_used_gb: float
    active_users: int
    response_time_avg_ms: float

@dataclass
class ResourceQuota:
    """Resource quota for tenant."""
    resource_type: str
    limit: Union[int, float]
    used: Union[int, float]
    reset_time: Optional[datetime]

class TenantManager:
    """Manages multi-tenant operations."""

    def __init__(self, config_file: str = "tenants.json"):
        self.config_file = config_file
        self.tenants: Dict[str, TenantConfig] = {}
        self.metrics: Dict[str, List[TenantMetrics]] = {}
        self.quotas: Dict[str, Dict[str, ResourceQuota]] = {}
        self._lock = threading.RLock()
        self._load_tenants()

    def create_tenant(self, name: str, tier: TenantTier = TenantTier.BASIC,
                     admin_email: str = "", settings: Dict[str, Any] = None) -> str:
        """Create a new tenant."""
        tenant_id = str(uuid.uuid4())

        config = TenantConfig(
            tenant_id=tenant_id,
            name=name,
            status=TenantStatus.PENDING.value,
            tier=tier.value,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            settings=settings or self._get_default_settings(tier),
            limits=self._get_tier_limits(tier),
            features=self._get_tier_features(tier),
            domains=[],
            admin_users=[admin_email] if admin_email else []
        )

        with self._lock:
            self.tenants[tenant_id] = config
            self.quotas[tenant_id] = self._initialize_quotas(tenant_id, tier)
            self.metrics[tenant_id] = []

        self._save_tenants()
        logger.info(f"Created tenant: {tenant_id} ({name})")

        return tenant_id

    def get_tenant(self, tenant_id: str) -> Optional[TenantConfig]:
        """Get tenant configuration."""
        return self.tenants.get(tenant_id)

    def update_tenant(self, tenant_id: str, updates: Dict[str, Any]) -> bool:
        """Update tenant configuration."""
        if tenant_id not in self.tenants:
            return False

        with self._lock:
            config = self.tenants[tenant_id]
            for key, value in updates.items():
                if hasattr(config, key):
                    setattr(config, key, value)
            config.updated_at = datetime.utcnow()

        self._save_tenants()
        logger.info(f"Updated tenant: {tenant_id}")
        return True

    def activate_tenant(self, tenant_id: str) -> bool:
        """Activate a tenant."""
        return self.update_tenant(tenant_id, {"status": TenantStatus.ACTIVE.value})

    def suspend_tenant(self, tenant_id: str) -> bool:
        """Suspend a tenant."""
        return self.update_tenant(tenant_id, {"status": TenantStatus.SUSPENDED.value})

    def check_tenant_access(self, tenant_id: str, resource: str = None) -> bool:
        """Check if tenant has access to a resource."""
        config = self.get_tenant(tenant_id)
        if not config:
            return False

        if config.status != TenantStatus.ACTIVE.value:
            return False

        if resource and resource not in config.features:
            return False

        return True

    def get_tenant_quota(self, tenant_id: str, resource_type: str) -> Optional[ResourceQuota]:
        """Get tenant resource quota."""
        tenant_quotas = self.quotas.get(tenant_id, {})
        return tenant_quotas.get(resource_type)

    def update_quota_usage(self, tenant_id: str, resource_type: str, usage: Union[int, float]) -> bool:
        """Update quota usage for a tenant."""
        if tenant_id not in self.quotas:
            return False

        quota = self.quotas[tenant_id].get(resource_type)
        if not quota:
            return False

        with self._lock:
            quota.used += usage

            # Check if quota exceeded
            if quota.used > quota.limit:
                logger.warning(f"Quota exceeded for tenant {tenant_id}: {resource_type}")
                return False

        return True

    def reset_quotas(self, tenant_id: str):
        """Reset quotas for a tenant (typically monthly)."""
        if tenant_id not in self.quotas:
            return

        with self._lock:
            for quota in self.quotas[tenant_id].values():
                quota.used = 0
                quota.reset_time = datetime.utcnow() + timedelta(days=30)

        logger.info(f"Reset quotas for tenant: {tenant_id}")

    def record_metrics(self, tenant_id: str, metrics: TenantMetrics):
        """Record tenant usage metrics."""
        if tenant_id not in self.metrics:
            self.metrics[tenant_id] = []

        with self._lock:
            self.metrics[tenant_id].append(metrics)

            # Keep only last 12 months of metrics
            cutoff = datetime.utcnow() - timedelta(days=365)
            self.metrics[tenant_id] = [
                m for m in self.metrics[tenant_id]
                if m.period_end > cutoff
            ]

    def get_tenant_metrics(self, tenant_id: str, months: int = 3) -> List[TenantMetrics]:
        """Get tenant metrics for specified months."""
        if tenant_id not in self.metrics:
            return []

        cutoff = datetime.utcnow() - timedelta(days=months * 30)
        return [m for m in self.metrics[tenant_id] if m.period_end > cutoff]

    def get_all_tenants(self, status: Optional[TenantStatus] = None) -> List[TenantConfig]:
        """Get all tenants, optionally filtered by status."""
        tenants = list(self.tenants.values())

        if status:
            tenants = [t for t in tenants if t.status == status.value]

        return tenants

    def _get_default_settings(self, tier: TenantTier) -> Dict[str, Any]:
        """Get default settings for a tier."""
        defaults = {
            "fraud_detection": {
                "model_version": "latest",
                "sensitivity": "medium",
                "auto_block": False
            },
            "notifications": {
                "email_alerts": True,
                "webhooks": False,
                "slack_integration": False
            },
            "security": {
                "two_factor_required": False,
                "session_timeout": 3600,
                "password_policy": "standard"
            }
        }

        # Tier-specific overrides
        if tier == TenantTier.ENTERPRISE:
            defaults["fraud_detection"]["auto_block"] = True
            defaults["notifications"]["slack_integration"] = True
            defaults["security"]["two_factor_required"] = True

        return defaults

    def _get_tier_limits(self, tier: TenantTier) -> Dict[str, Any]:
        """Get resource limits for a tier."""
        limits = {
            TenantTier.BASIC: {
                "api_calls_per_month": 10000,
                "data_processing_gb": 1,
                "storage_gb": 5,
                "active_users": 5
            },
            TenantTier.PROFESSIONAL: {
                "api_calls_per_month": 100000,
                "data_processing_gb": 10,
                "storage_gb": 50,
                "active_users": 25
            },
            TenantTier.ENTERPRISE: {
                "api_calls_per_month": 1000000,
                "data_processing_gb": 100,
                "storage_gb": 500,
                "active_users": 100
            },
            TenantTier.CUSTOM: {
                "api_calls_per_month": 10000,
                "data_processing_gb": 1,
                "storage_gb": 5,
                "active_users": 5
            }
        }

        return limits.get(tier, limits[TenantTier.BASIC])

    def _get_tier_features(self, tier: TenantTier) -> List[str]:
        """Get features available for a tier."""
        features = {
            TenantTier.BASIC: [
                "fraud_detection",
                "basic_reporting",
                "email_alerts"
            ],
            TenantTier.PROFESSIONAL: [
                "fraud_detection",
                "advanced_reporting",
                "email_alerts",
                "webhooks",
                "api_access"
            ],
            TenantTier.ENTERPRISE: [
                "fraud_detection",
                "advanced_reporting",
                "email_alerts",
                "webhooks",
                "api_access",
                "custom_models",
                "white_label",
                "priority_support",
                "sla_guarantee"
            ],
            TenantTier.CUSTOM: [
                "fraud_detection",
                "basic_reporting",
                "email_alerts"
            ]
        }

        return features.get(tier, features[TenantTier.BASIC])

    def _initialize_quotas(self, tenant_id: str, tier: TenantTier) -> Dict[str, ResourceQuota]:
        """Initialize quotas for a new tenant."""
        limits = self._get_tier_limits(tier)
        quotas = {}

        for resource_type, limit in limits.items():
            quotas[resource_type] = ResourceQuota(
                resource_type=resource_type,
                limit=limit,
                used=0,
                reset_time=datetime.utcnow() + timedelta(days=30)
            )

        return quotas

    def _load_tenants(self):
        """Load tenant configurations from file."""
        if not os.path.exists(self.config_file):
            return

        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            for tenant_data in data.get("tenants", []):
                # Convert timestamp strings to datetime
                tenant_data["created_at"] = datetime.fromisoformat(tenant_data["created_at"])
                tenant_data["updated_at"] = datetime.fromisoformat(tenant_data["updated_at"])
                tenant = TenantConfig(**tenant_data)
                self.tenants[tenant.tenant_id] = tenant

                # Load quotas
                if tenant.tenant_id in data.get("quotas", {}):
                    quota_data = data["quotas"][tenant.tenant_id]
                    self.quotas[tenant.tenant_id] = {}
                    for resource_type, quota_info in quota_data.items():
                        if "reset_time" in quota_info and quota_info["reset_time"]:
                            quota_info["reset_time"] = datetime.fromisoformat(quota_info["reset_time"])
                        self.quotas[tenant.tenant_id][resource_type] = ResourceQuota(**quota_info)

        except Exception as e:
            logger.error(f"Failed to load tenants: {e}")

    def _save_tenants(self):
        """Save tenant configurations to file."""
        try:
            data = {
                "tenants": [asdict(t) for t in self.tenants.values()],
                "quotas": {}
            }

            for tenant_id, tenant_quotas in self.quotas.items():
                data["quotas"][tenant_id] = {
                    resource_type: asdict(quota) for resource_type, quota in tenant_quotas.items()
                }

            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, default=str, indent=2, ensure_ascii=False)

        except Exception as e:
            logger.error(f"Failed to save tenants: {e}")

class TenantIsolation:
    """Handles tenant data isolation and context management."""

    def __init__(self, tenant_manager: TenantManager):
        self.tenant_manager = tenant_manager
        self.isolation_level = "database"  # or "schema" or "row"

    def set_tenant_context(self, tenant_id: str) -> bool:
        """Set the current tenant context."""
        if not self.tenant_manager.check_tenant_access(tenant_id):
            logger.warning(f"Access denied for tenant: {tenant_id}")
            return False

        current_tenant.set(tenant_id)
        logger.debug(f"Set tenant context: {tenant_id}")
        return True

    def get_current_tenant(self) -> Optional[str]:
        """Get the current tenant ID from context."""
        return current_tenant.get()

    def require_tenant_context(self) -> str:
        """Require and return current tenant context."""
        tenant_id = self.get_current_tenant()
        if not tenant_id:
            raise ValueError("No tenant context set")
        return tenant_id

    def clear_tenant_context(self):
        """Clear the current tenant context."""
        current_tenant.set(None)

    def isolate_data_query(self, base_query: str, tenant_field: str = "tenant_id") -> str:
        """Modify query to include tenant isolation."""
        tenant_id = self.get_current_tenant()
        if not tenant_id:
            raise ValueError("No tenant context for data isolation")

        # Add tenant filter to query
        if "WHERE" in base_query.upper():
            return f"{base_query} AND {tenant_field} = '{tenant_id}'"
        else:
            return f"{base_query} WHERE {tenant_field} = '{tenant_id}'"

    def isolate_storage_path(self, base_path: str) -> str:
        """Isolate storage path by tenant."""
        tenant_id = self.get_current_tenant()
        if not tenant_id:
            raise ValueError("No tenant context for storage isolation")

        return os.path.join(base_path, tenant_id)

class TenantRateLimiter:
    """Rate limiting per tenant."""

    def __init__(self, tenant_manager: TenantManager):
        self.tenant_manager = tenant_manager
        self.requests: Dict[str, List[datetime]] = {}
        self._lock = threading.RLock()

    def check_rate_limit(self, tenant_id: str, resource_type: str = "api_calls",
                        window_seconds: int = 60, max_requests: int = 100) -> bool:
        """Check if tenant is within rate limits."""
        now = datetime.utcnow()
        window_start = now - timedelta(seconds=window_seconds)

        with self._lock:
            if tenant_id not in self.requests:
                self.requests[tenant_id] = []

            # Remove old requests outside the window
            self.requests[tenant_id] = [
                req_time for req_time in self.requests[tenant_id]
                if req_time > window_start
            ]

            # Check current request count
            if len(self.requests[tenant_id]) >= max_requests:
                logger.warning(f"Rate limit exceeded for tenant {tenant_id}")
                return False

            # Add current request
            self.requests[tenant_id].append(now)

        return True

    def get_remaining_requests(self, tenant_id: str, window_seconds: int = 60,
                             max_requests: int = 100) -> int:
        """Get remaining requests in current window."""
        now = datetime.utcnow()
        window_start = now - timedelta(seconds=window_seconds)

        with self._lock:
            if tenant_id not in self.requests:
                return max_requests

            # Count requests in current window
            current_requests = len([
                req_time for req_time in self.requests[tenant_id]
                if req_time > window_start
            ])

        return max(0, max_requests - current_requests)

class TenantBilling:
    """Handles tenant billing and usage tracking."""

    def __init__(self, tenant_manager: TenantManager):
        self.tenant_manager = tenant_manager
        self.pricing: Dict[str, Dict[str, float]] = self._load_pricing()

    def calculate_bill(self, tenant_id: str, period_start: datetime,
                      period_end: datetime) -> Dict[str, Any]:
        """Calculate billing for a tenant for a specific period."""
        metrics = self.tenant_manager.get_tenant_metrics(tenant_id, months=1)

        # Filter metrics for the period
        period_metrics = [
            m for m in metrics
            if period_start <= m.period_start <= period_end
        ]

        if not period_metrics:
            return {"total_amount": 0, "breakdown": {}, "period": "no_data"}

        tenant = self.tenant_manager.get_tenant(tenant_id)
        if not tenant:
            return {"total_amount": 0, "breakdown": {}, "period": "invalid_tenant"}

        tier_pricing = self.pricing.get(tenant.tier, self.pricing["basic"])

        # Calculate usage-based charges
        breakdown = {}

        # API calls
        total_api_calls = sum(m.api_calls for m in period_metrics)
        api_overage = max(0, total_api_calls - tenant.limits["api_calls_per_month"])
        breakdown["api_calls"] = {
            "included": tenant.limits["api_calls_per_month"],
            "used": total_api_calls,
            "overage": api_overage,
            "rate": tier_pricing.get("api_overage_per_1000", 0),
            "amount": (api_overage / 1000) * tier_pricing.get("api_overage_per_1000", 0)
        }

        # Data processing
        total_data_gb = sum(m.data_processed_gb for m in period_metrics)
        data_overage = max(0, total_data_gb - tenant.limits["data_processing_gb"])
        breakdown["data_processing"] = {
            "included": tenant.limits["data_processing_gb"],
            "used": total_data_gb,
            "overage": data_overage,
            "rate": tier_pricing.get("data_processing_per_gb", 0),
            "amount": data_overage * tier_pricing.get("data_processing_per_gb", 0)
        }

        # Storage
        total_storage_gb = sum(m.storage_used_gb for m in period_metrics)
        storage_overage = max(0, total_storage_gb - tenant.limits["storage_gb"])
        breakdown["storage"] = {
            "included": tenant.limits["storage_gb"],
            "used": total_storage_gb,
            "overage": storage_overage,
            "rate": tier_pricing.get("storage_per_gb", 0),
            "amount": storage_overage * tier_pricing.get("storage_per_gb", 0)
        }

        # Base fee
        base_amount = tier_pricing.get("base_monthly", 0)

        total_amount = base_amount + sum(item["amount"] for item in breakdown.values())

        return {
            "tenant_id": tenant_id,
            "period": {
                "start": period_start.isoformat(),
                "end": period_end.isoformat()
            },
            "tier": tenant.tier,
            "base_amount": base_amount,
            "breakdown": breakdown,
            "total_amount": round(total_amount, 2),
            "currency": "USD"
        }

    def _load_pricing(self) -> Dict[str, Dict[str, float]]:
        """Load pricing configuration."""
        return {
            "basic": {
                "base_monthly": 99,
                "api_overage_per_1000": 5,
                "data_processing_per_gb": 10,
                "storage_per_gb": 2
            },
            "professional": {
                "base_monthly": 499,
                "api_overage_per_1000": 3,
                "data_processing_per_gb": 7,
                "storage_per_gb": 1.5
            },
            "enterprise": {
                "base_monthly": 1999,
                "api_overage_per_1000": 2,
                "data_processing_per_gb": 5,
                "storage_per_gb": 1
            },
            "custom": {
                "base_monthly": 0,
                "api_overage_per_1000": 10,
                "data_processing_per_gb": 15,
                "storage_per_gb": 3
            }
        }

# Global instances
_tenant_manager = None
_tenant_isolation = None
_tenant_rate_limiter = None
_tenant_billing = None

def get_tenant_manager() -> TenantManager:
    """Get global tenant manager."""
    global _tenant_manager
    if _tenant_manager is None:
        _tenant_manager = TenantManager()
    return _tenant_manager

def get_tenant_isolation() -> TenantIsolation:
    """Get global tenant isolation manager."""
    global _tenant_isolation
    if _tenant_isolation is None:
        _tenant_isolation = TenantIsolation(get_tenant_manager())
    return _tenant_isolation

def get_tenant_rate_limiter() -> TenantRateLimiter:
    """Get global tenant rate limiter."""
    global _tenant_rate_limiter
    if _tenant_rate_limiter is None:
        _tenant_rate_limiter = TenantRateLimiter(get_tenant_manager())
    return _tenant_rate_limiter

def get_tenant_billing() -> TenantBilling:
    """Get global tenant billing manager."""
    global _tenant_billing
    if _tenant_billing is None:
        _tenant_billing = TenantBilling(get_tenant_manager())
    return _tenant_billing

# Context manager for tenant context
class tenant_context:
    """Context manager for tenant operations."""

    def __init__(self, tenant_id: str):
        self.tenant_id = tenant_id
        self.isolation = get_tenant_isolation()
        self.previous_tenant = None

    def __enter__(self):
        self.previous_tenant = self.isolation.get_current_tenant()
        if not self.isolation.set_tenant_context(self.tenant_id):
            raise ValueError(f"Cannot set tenant context: {self.tenant_id}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.previous_tenant:
            self.isolation.set_tenant_context(self.previous_tenant)
        else:
            self.isolation.clear_tenant_context()