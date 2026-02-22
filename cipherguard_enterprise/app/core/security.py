"""
Enterprise security middleware and authentication system with JWT tokens,
API keys, rate limiting, CORS, and comprehensive security headers.
"""

from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
import jwt
from passlib.context import CryptContext
from fastapi import HTTPException, status, Request, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response, JSONResponse
import time
import hashlib
import logging
from collections import defaultdict, deque
from threading import Lock
import ipaddress
from urllib.parse import urlparse

from .config import get_settings
from .logging import get_correlation_id, set_correlation_id, generate_correlation_id, audit_logger

logger = logging.getLogger(__name__)
settings = get_settings()

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Security schemes
security_bearer = HTTPBearer(auto_error=False)
security_api_key = APIKeyHeader(name=settings.security.api_key_header_name, auto_error=False)


class User:
    """User model for authentication."""
    
    def __init__(self, user_id: str, username: str, email: str, is_active: bool = True, scopes: List[str] = None):
        self.user_id = user_id
        self.username = username
        self.email = email
        self.is_active = is_active
        self.scopes = scopes or []


class TokenData:
    """Token data model."""
    
    def __init__(self, user_id: str = None, scopes: List[str] = None):
        self.user_id = user_id
        self.scopes = scopes or []


class RateLimiter:
    """
    Advanced rate limiter with multiple strategies and distributed support.
    """
    
    def __init__(self):
        self.requests = defaultdict(lambda: deque())
        self.lock = Lock()
        self.blocked_ips = set()
        self.whitelist_ips = set()
        
        # Load configuration
        self.enabled = settings.security.rate_limit_enabled
        self.calls_limit = settings.security.rate_limit_calls
        self.time_window = settings.security.rate_limit_period
        
        # Advanced rate limiting
        self.burst_limit = self.calls_limit * 2  # Allow burst
        self.block_duration = 300  # 5 minutes block
        
    def _get_client_key(self, request: Request) -> str:
        """Get unique client identifier for rate limiting."""
        # Use X-Forwarded-For or client IP
        client_ip = request.headers.get("X-Forwarded-For", "").split(",")[0].strip()
        if not client_ip:
            client_ip = request.client.host if request.client else "unknown"
        
        # Include user ID if available (for authenticated users)
        user_id = getattr(request.state, 'user_id', None)
        if user_id:
            return f"user:{user_id}:{client_ip}"
        
        return f"ip:{client_ip}"
    
    def _is_whitelisted(self, client_ip: str) -> bool:
        """Check if IP is whitelisted."""
        try:
            ip = ipaddress.ip_address(client_ip)
            for whitelist_ip in self.whitelist_ips:
                if ip in ipaddress.ip_network(whitelist_ip, strict=False):
                    return True
        except ValueError:
            pass
        return False
    
    def _is_blocked(self, client_ip: str) -> bool:
        """Check if IP is currently blocked."""
        return client_ip in self.blocked_ips
    
    def is_allowed(self, request: Request) -> tuple[bool, Dict[str, Any]]:
        """
        Check if request is allowed based on rate limiting rules.
        
        Returns:
            Tuple of (is_allowed, info_dict)
        """
        if not self.enabled:
            return True, {}
        
        client_key = self._get_client_key(request)
        client_ip = client_key.split(":")[-1]  # Extract IP from key
        
        # Check whitelist
        if self._is_whitelisted(client_ip):
            return True, {"status": "whitelisted"}
        
        # Check blacklist
        if self._is_blocked(client_ip):
            return False, {
                "status": "blocked",
                "reason": "IP temporarily blocked due to rate limiting violations",
                "retry_after": self.block_duration
            }
        
        now = time.time()
        
        with self.lock:
            # Get request history for this client
            client_requests = self.requests[client_key]
            
            # Remove old requests outside the time window
            cutoff_time = now - self.time_window
            while client_requests and client_requests[0] < cutoff_time:
                client_requests.popleft()
            
            # Check rate limit
            current_requests = len(client_requests)
            
            if current_requests >= self.burst_limit:
                # Block this IP temporarily
                self.blocked_ips.add(client_ip)
                
                # Log security event
                audit_logger.log_system_event(
                    event_type="rate_limit_exceeded",
                    description=f"Rate limit exceeded for {client_key}",
                    severity="warning",
                    additional_data={
                        "client_key": client_key,
                        "requests_count": current_requests,
                        "limit": self.calls_limit,
                        "time_window": self.time_window
                    }
                )
                
                return False, {
                    "status": "rate_limited",
                    "reason": f"Rate limit exceeded: {current_requests} requests in {self.time_window} seconds",
                    "limit": self.calls_limit,
                    "window_seconds": self.time_window,
                    "retry_after": self.time_window
                }
            
            elif current_requests >= self.calls_limit:
                return False, {
                    "status": "rate_limited", 
                    "reason": f"Rate limit exceeded: {current_requests} requests in {self.time_window} seconds",
                    "limit": self.calls_limit,
                    "window_seconds": self.time_window,
                    "retry_after": self.time_window
                }
            
            # Add current request
            client_requests.append(now)
            
            return True, {
                "status": "allowed",
                "remaining": self.calls_limit - current_requests - 1,
                "reset_time": int(now + self.time_window)
            }


class SecurityMiddleware(BaseHTTPMiddleware):
    """
    Comprehensive security middleware with CORS, rate limiting, 
    security headers, and request correlation.
    """
    
    def __init__(self, app, rate_limiter: RateLimiter = None):
        super().__init__(app)
        self.rate_limiter = rate_limiter or RateLimiter()
    
    async def dispatch(self, request: Request, call_next):
        """Process request with security checks."""
        start_time = time.time()
        
        # Generate/set correlation ID
        correlation_id_value = request.headers.get(
            settings.logging.correlation_id_header,
            generate_correlation_id()
        )
        set_correlation_id(correlation_id_value)
        
        # Store correlation ID in request state
        request.state.correlation_id = correlation_id_value
        
        # Rate limiting
        if hasattr(self.rate_limiter, 'is_allowed'):
            is_allowed, rate_info = self.rate_limiter.is_allowed(request)
            if not is_allowed:
                response = JSONResponse(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    content={
                        "detail": rate_info.get("reason", "Rate limit exceeded"),
                        "retry_after": rate_info.get("retry_after", 60)
                    }
                )
                if "retry_after" in rate_info:
                    response.headers["Retry-After"] = str(rate_info["retry_after"])
                return response
        
        # Process request
        try:
            response = await call_next(request)
        except Exception as e:
            logger.error(f"Request processing failed: {str(e)}", exc_info=True)
            response = JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"detail": "Internal server error"}
            )
        
        # Add security headers
        self._add_security_headers(response)
        
        # Add correlation ID to response
        response.headers[settings.logging.correlation_id_header] = correlation_id_value
        
        # Add rate limiting headers if applicable
        if hasattr(self.rate_limiter, 'is_allowed') and 'remaining' in rate_info:
            response.headers["X-RateLimit-Limit"] = str(settings.security.rate_limit_calls)
            response.headers["X-RateLimit-Remaining"] = str(rate_info.get("remaining", 0))
            response.headers["X-RateLimit-Reset"] = str(rate_info.get("reset_time", 0))
        
        # Add processing time
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        
        return response
    
    def _add_security_headers(self, response: Response) -> None:
        """Add comprehensive security headers."""
        # Content security
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        
        # Transport security (if HTTPS)
        if settings.is_production():
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        
        # Information disclosure
        response.headers["Server"] = "CipherGuard"
        response.headers["X-Powered-By"] = "FastAPI"
        
        # Cache control for sensitive data
        if "/api/" in str(getattr(response, 'url', '')):
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"


# Authentication and authorization functions
def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Get password hash."""
    return pwd_context.hash(password)


def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """
    Create JWT access token with optional expiration.
    
    Args:
        data: Token payload data
        expires_delta: Token expiration time
        
    Returns:
        Encoded JWT token
    """
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.security.jwt_access_token_expire_minutes)
    
    to_encode.update({
        "exp": expire,
        "iat": datetime.utcnow(),
        "type": "access"
    })
    
    encoded_jwt = jwt.encode(
        to_encode, 
        settings.security.jwt_secret_key, 
        algorithm=settings.security.jwt_algorithm
    )
    
    return encoded_jwt


def verify_token(token: str) -> Optional[TokenData]:
    """
    Verify and decode JWT token.
    
    Args:
        token: JWT token string
        
    Returns:
        TokenData if valid, None if invalid
    """
    try:
        payload = jwt.decode(
            token, 
            settings.security.jwt_secret_key, 
            algorithms=[settings.security.jwt_algorithm]
        )
        
        user_id: str = payload.get("sub")
        if user_id is None:
            return None
        
        scopes: List[str] = payload.get("scopes", [])
        token_data = TokenData(user_id=user_id, scopes=scopes)
        
        return token_data
    
    except jwt.PyJWTError as e:
        logger.warning(f"JWT token verification failed: {str(e)}")
        return None


def verify_api_key(api_key: str) -> bool:
    """
    Verify API key.
    
    Args:
        api_key: API key to verify
        
    Returns:
        True if valid, False otherwise
    """
    # In production, this should check against a database
    # For now, compare with configured API key
    return api_key == settings.security.api_key


async def get_current_user_jwt(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security_bearer)) -> Optional[User]:
    """
    Get current user from JWT token.
    
    Returns:
        User object if authenticated, None otherwise
    """
    if not credentials:
        return None
    
    token_data = verify_token(credentials.credentials)
    if not token_data:
        audit_logger.log_authentication(
            jwt_used=True,
            success=False,
            failure_reason="Invalid JWT token"
        )
        return None
    
    # In production, fetch user from database
    # For now, create a mock user
    user = User(
        user_id=token_data.user_id,
        username=f"user_{token_data.user_id}",
        email=f"user_{token_data.user_id}@example.com",
        scopes=token_data.scopes
    )
    
    audit_logger.log_authentication(
        user_id=user.user_id,
        jwt_used=True,
        success=True
    )
    
    return user


async def get_current_user_api_key(api_key: Optional[str] = Depends(security_api_key)) -> Optional[User]:
    """
    Get current user from API key.
    
    Returns:
        User object if authenticated, None otherwise
    """
    if not api_key:
        return None
    
    if not verify_api_key(api_key):
        audit_logger.log_authentication(
            api_key_used=True,
            success=False, 
            failure_reason="Invalid API key"
        )
        return None
    
    # Create API key user (in production, get from database)
    user = User(
        user_id="api_user",
        username="api_user", 
        email="api@cipherguard.com",
        scopes=["read", "write", "admin"]
    )
    
    audit_logger.log_authentication(
        user_id=user.user_id,
        api_key_used=True,
        success=True
    )
    
    return user


async def get_current_user(
    jwt_user: Optional[User] = Depends(get_current_user_jwt),
    api_user: Optional[User] = Depends(get_current_user_api_key)
) -> Optional[User]:
    """
    Get current authenticated user from any supported method.
    
    Returns JWT user if available, otherwise API key user.
    """
    return jwt_user or api_user


async def require_authentication(
    current_user: Optional[User] = Depends(get_current_user)
) -> User:
    """
    Require user authentication.
    
    Raises HTTPException if user is not authenticated.
    """
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    
    return current_user


def require_scopes(required_scopes: List[str]):
    """
    Dependency factory to require specific scopes.
    
    Args:
        required_scopes: List of required scopes
        
    Returns:
        Dependency function
    """
    def scope_checker(current_user: User = Depends(require_authentication)) -> User:
        if not all(scope in current_user.scopes for scope in required_scopes):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions"
            )
        return current_user
    
    return scope_checker


def setup_security_middleware(app):
    """
    Setup security middleware for FastAPI application.
    
    Args:
        app: FastAPI application instance
    """
    # Rate limiter
    rate_limiter = RateLimiter()
    
    # Security middleware (should be first)
    app.add_middleware(SecurityMiddleware, rate_limiter=rate_limiter)
    
    # CORS middleware
    if settings.security.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.get_cors_origins(),
            allow_credentials=settings.security.cors_allow_credentials,
            allow_methods=settings.security.cors_allow_methods,
            allow_headers=settings.security.cors_allow_headers,
        )
    
    # Trusted host middleware (for production)
    if settings.is_production():
        # In production, configure trusted hosts
        trusted_hosts = ["cipherguard.com", "*.cipherguard.com", "localhost"]
        app.add_middleware(TrustedHostMiddleware, allowed_hosts=trusted_hosts)
    
    logger.info("Security middleware configured successfully")


# Export key components
__all__ = [
    'User',
    'TokenData',
    'RateLimiter', 
    'SecurityMiddleware',
    'create_access_token',
    'verify_token',
    'verify_api_key',
    'get_current_user',
    'require_authentication',
    'require_scopes',
    'setup_security_middleware',
    'verify_password',
    'get_password_hash'
]