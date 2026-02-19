"""
Production-ready security middleware and utilities
"""

from fastapi import Request, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import time
import asyncio
from typing import Dict, List, Optional, Callable
import redis
import jwt
import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# Rate limiting using Redis
class RedisRateLimiter:
    """Redis-backed rate limiter with sliding window."""
    
    def __init__(self, redis_client, window_size: int = 3600):
        self.redis = redis_client
        self.window_size = window_size
        
    async def is_allowed(
        self, 
        key: str, 
        limit: int, 
        window: int = None
    ) -> tuple[bool, Dict[str, int]]:
        """Check if request is within rate limit."""
        window = window or self.window_size
        now = int(time.time())
        
        # Use sliding window counter
        pipe = self.redis.pipeline()
        pipe.zremrangebyscore(key, 0, now - window)
        pipe.zadd(key, {str(now): now})
        pipe.zcount(key, now - window, now)
        pipe.expire(key, window)
        
        results = await asyncio.to_thread(pipe.execute)
        current_requests = results[2]
        
        return current_requests <= limit, {
            "remaining": max(0, limit - current_requests),
            "reset_time": now + window,
            "current_requests": current_requests
        }

# In-memory fallback for development
class InMemoryRateLimiter:
    """In-memory rate limiter for development."""
    
    def __init__(self):
        self.requests: Dict[str, List[float]] = {}
        
    async def is_allowed(
        self, 
        key: str, 
        limit: int, 
        window: int = 3600
    ) -> tuple[bool, Dict[str, int]]:
        """Check rate limit using in-memory storage."""
        now = time.time()
        
        if key not in self.requests:
            self.requests[key] = []
            
        # Clean old requests
        self.requests[key] = [
            req_time for req_time in self.requests[key] 
            if now - req_time < window
        ]
        
        current_count = len(self.requests[key])
        
        if current_count < limit:
            self.requests[key].append(now)
            return True, {
                "remaining": limit - current_count - 1,
                "reset_time": int(now + window),
                "current_requests": current_count + 1
            }
        
        return False, {
            "remaining": 0,
            "reset_time": int(now + window),
            "current_requests": current_count
        }

class SecurityMiddleware:
    """Comprehensive security middleware."""
    
    def __init__(self, rate_limiter, jwt_secret: str):
        self.rate_limiter = rate_limiter
        self.jwt_secret = jwt_secret
        self.security = HTTPBearer()
        
    def create_rate_limit_middleware(
        self, 
        requests_per_minute: int = 60,
        requests_per_hour: int = 1000
    ):
        """Create rate limiting middleware."""
        
        async def rate_limit_middleware(request: Request, call_next):
            client_ip = self._get_client_ip(request)
            
            # Check minute limit
            allowed, info = await self.rate_limiter.is_allowed(
                f"rate_limit:minute:{client_ip}", 
                requests_per_minute, 
                60
            )
            
            if not allowed:
                raise HTTPException(
                    status_code=429, 
                    detail="Rate limit exceeded",
                    headers={"Retry-After": "60"}
                )
                
            # Check hourly limit
            allowed, _ = await self.rate_limiter.is_allowed(
                f"rate_limit:hour:{client_ip}", 
                requests_per_hour, 
                3600
            )
            
            if not allowed:
                raise HTTPException(
                    status_code=429, 
                    detail="Hourly rate limit exceeded",
                    headers={"Retry-After": "3600"}
                )
            
            # Add rate limit headers
            response = await call_next(request)
            response.headers["X-RateLimit-Remaining"] = str(info["remaining"])
            response.headers["X-RateLimit-Reset"] = str(info["reset_time"])
            
            return response
            
        return rate_limit_middleware
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract real client IP considering proxies."""
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")  
        if real_ip:
            return real_ip
            
        return request.client.host
    
    async def verify_jwt_token(
        self, 
        credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer())
    ) -> Dict[str, any]:
        """Verify JWT token and extract claims."""
        try:
            payload = jwt.decode(
                credentials.credentials, 
                self.jwt_secret, 
                algorithms=["HS256"]
            )
            
            # Check expiration
            if payload.get("exp", 0) < time.time():
                raise HTTPException(
                    status_code=401, 
                    detail="Token has expired"
                )
                
            return payload
            
        except jwt.InvalidTokenError:
            raise HTTPException(
                status_code=401,
                detail="Invalid authentication token"
            )

# Request validation middleware
async def validate_request_size_middleware(request: Request, call_next):
    """Limit request body size to prevent DoS."""
    if request.method in ["POST", "PUT", "PATCH"]:
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > 1024 * 1024:  # 1MB limit
            raise HTTPException(
                status_code=413, 
                detail="Request body too large"
            )
    
    return await call_next(request)

# Security headers middleware  
async def security_headers_middleware(request: Request, call_next):
    """Add security headers."""
    response = await call_next(request)
    
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    response.headers["Content-Security-Policy"] = "default-src 'self'"
    
    return response

# Request logging for audit
@dataclass
class AuditLogEntry:
    timestamp: datetime
    method: str
    path: str
    client_ip: str
    user_agent: str
    user_id: Optional[str]
    response_status: int
    processing_time_ms: float

class AuditLogger:
    """Audit logging for security events."""
    
    def __init__(self):
        self.logger = logging.getLogger("audit")
        
    async def log_request(self, entry: AuditLogEntry):
        """Log request for audit trail."""
        self.logger.info(
            "Request audit",
            extra={
                "timestamp": entry.timestamp.isoformat(),
                "method": entry.method,
                "path": entry.path,
                "client_ip": entry.client_ip,
                "user_agent": entry.user_agent,
                "user_id": entry.user_id,
                "response_status": entry.response_status,
                "processing_time_ms": entry.processing_time_ms
            }
        )

async def audit_middleware(request: Request, call_next):
    """Middleware for audit logging."""
    start_time = time.time()
    
    response = await call_next(request)
    
    processing_time = (time.time() - start_time) * 1000
    
    audit_logger = AuditLogger()
    entry = AuditLogEntry(
        timestamp=datetime.utcnow(),
        method=request.method,
        path=request.url.path,
        client_ip=request.client.host,
        user_agent=request.headers.get("user-agent", ""),
        user_id=getattr(request.state, "user_id", None),
        response_status=response.status_code,
        processing_time_ms=processing_time
    )
    
    await audit_logger.log_request(entry)
    
    return response