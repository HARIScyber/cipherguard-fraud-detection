"""
Enhanced models with comprehensive validation and security
"""

from pydantic import BaseModel, Field, validator, EmailStr
from typing import Dict, List, Optional, Any
from datetime import datetime
import re
from decimal import Decimal
from enum import Enum

class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM" 
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

class DeviceType(str, Enum):
    MOBILE = "mobile"
    DESKTOP = "desktop"  
    TABLET = "tablet"
    UNKNOWN = "unknown"

class Transaction(BaseModel):
    """Enhanced transaction model with strict validation."""
    
    amount: Decimal = Field(
        ..., 
        gt=0, 
        le=1000000,  # Max $1M transaction
        description="Transaction amount in USD"
    )
    merchant: str = Field(
        ..., 
        min_length=1, 
        max_length=100,
        description="Merchant identifier"
    )
    device: DeviceType = Field(..., description="Device type")
    country: str = Field(
        ..., 
        regex=r'^[A-Z]{2}$',  # ISO country code
        description="ISO 3166-1 alpha-2 country code"
    )
    customer_id: Optional[str] = Field(
        None,
        min_length=1,
        max_length=50,
        regex=r'^[a-zA-Z0-9_-]+$'
    )
    timestamp: Optional[datetime] = Field(
        None,
        description="Transaction timestamp (UTC)"
    )
    ip_address: Optional[str] = Field(
        None,
        description="Client IP address for geo-validation"
    )
    session_id: Optional[str] = Field(
        None,
        min_length=1,
        max_length=100
    )
    
    @validator('merchant')
    def validate_merchant(cls, v):
        """Sanitize merchant name."""
        # Remove special characters, normalize
        sanitized = re.sub(r'[^\w\s-]', '', v).strip()
        if not sanitized:
            raise ValueError("Merchant name cannot be empty after sanitization")
        return sanitized[:100]  # Truncate if too long
    
    @validator('ip_address')
    def validate_ip(cls, v):
        """Validate IP address format."""
        if v is None:
            return v
        try:
            import ipaddress
            ipaddress.ip_address(v)
            return v
        except ValueError:
            raise ValueError("Invalid IP address format")
    
    @validator('timestamp')
    def validate_timestamp(cls, v):
        """Ensure timestamp is not too far in past/future."""
        if v is None:
            return datetime.utcnow()
        
        now = datetime.utcnow()
        max_past = now.timestamp() - (365 * 24 * 3600)  # 1 year
        max_future = now.timestamp() + (24 * 3600)      # 1 day
        
        if v.timestamp() < max_past:
            raise ValueError("Timestamp too far in the past")
        if v.timestamp() > max_future:
            raise ValueError("Timestamp too far in the future")
            
        return v

class DetectionResult(BaseModel):
    """Enhanced detection result with additional context."""
    
    transaction_id: str = Field(..., min_length=1)
    is_fraud: bool
    fraud_score: float = Field(..., ge=0.0, le=1.0)
    confidence: float = Field(..., ge=0.0, le=1.0)  # Model confidence
    risk_level: RiskLevel
    risk_factors: List[str] = Field(default_factory=list)  # Reasons for risk
    similar_transactions: List[str] = Field(default_factory=list)
    timestamp: datetime
    processing_time_ms: Optional[float] = None
    model_version: Optional[str] = None
    
class AuthenticatedRequest(BaseModel):
    """Base class for authenticated requests."""
    
    def validate_auth_token(self, token: str) -> bool:
        """Validate authentication token."""
        # Implement JWT validation
        pass

class RateLimitedRequest(BaseModel):
    """Base class for rate-limited requests."""
    
    client_id: str = Field(..., min_length=1, max_length=50)
    
    @validator('client_id')
    def validate_client_id(cls, v):
        """Validate client ID format."""
        if not re.match(r'^[a-zA-Z0-9_-]+$', v):
            raise ValueError("Client ID contains invalid characters")
        return v