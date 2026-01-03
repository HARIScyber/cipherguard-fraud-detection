"""
Enterprise Security Module - Phase 6: Enterprise Integration
Advanced security features for enterprise-grade fraud detection
"""

import hashlib
import hmac
import secrets
import jwt
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import os
import json
from dataclasses import dataclass, asdict
import ipaddress
import re

logger = logging.getLogger(__name__)

@dataclass
class SecurityEvent:
    """Security event for audit logging."""
    event_id: str
    event_type: str
    user_id: Optional[str]
    ip_address: str
    user_agent: str
    timestamp: datetime
    severity: str
    details: Dict[str, Any]
    risk_score: float

@dataclass
class UserSession:
    """User session information."""
    session_id: str
    user_id: str
    created_at: datetime
    expires_at: datetime
    ip_address: str
    user_agent: str
    is_active: bool = True

class EncryptionManager:
    """Manages encryption/decryption operations."""

    def __init__(self, master_key: Optional[bytes] = None):
        if master_key:
            self.master_key = master_key
        else:
            # Generate a new key
            self.master_key = Fernet.generate_key()

        self.fernet = Fernet(self.master_key)

    def encrypt_data(self, data: str) -> str:
        """Encrypt string data."""
        return self.fernet.encrypt(data.encode()).decode()

    def decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt string data."""
        return self.fernet.decrypt(encrypted_data.encode()).decode()

    def encrypt_dict(self, data: Dict[str, Any]) -> str:
        """Encrypt dictionary data."""
        json_data = json.dumps(data)
        return self.encrypt_data(json_data)

    def decrypt_dict(self, encrypted_data: str) -> Dict[str, Any]:
        """Decrypt dictionary data."""
        json_data = self.decrypt_data(encrypted_data)
        return json.loads(json_data)

    def hash_password(self, password: str, salt: Optional[bytes] = None) -> Tuple[str, bytes]:
        """Hash password with PBKDF2."""
        if salt is None:
            salt = secrets.token_bytes(16)

        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        return key.decode(), salt

    def verify_password(self, password: str, hashed: str, salt: bytes) -> bool:
        """Verify password against hash."""
        expected_key, _ = self.hash_password(password, salt)
        return secrets.compare_digest(hashed, expected_key)

class JWTManager:
    """Manages JWT token operations."""

    def __init__(self, secret_key: str, algorithm: str = "HS256"):
        self.secret_key = secret_key
        self.algorithm = algorithm

    def create_token(self, payload: Dict[str, Any], expires_in: int = 3600) -> str:
        """Create JWT token."""
        payload_copy = payload.copy()
        payload_copy['exp'] = datetime.utcnow() + timedelta(seconds=expires_in)
        payload_copy['iat'] = datetime.utcnow()

        return jwt.encode(payload_copy, self.secret_key, algorithm=self.algorithm)

    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify and decode JWT token."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("Token has expired")
            return None
        except jwt.InvalidTokenError:
            logger.warning("Invalid token")
            return None

    def refresh_token(self, token: str, expires_in: int = 3600) -> Optional[str]:
        """Refresh JWT token."""
        payload = self.verify_token(token)
        if payload:
            # Remove old timestamps
            payload.pop('exp', None)
            payload.pop('iat', None)
            return self.create_token(payload, expires_in)
        return None

class SessionManager:
    """Manages user sessions."""

    def __init__(self, encryption_manager: EncryptionManager):
        self.encryption = encryption_manager
        self.sessions: Dict[str, UserSession] = {}
        self.max_sessions_per_user = 5

    def create_session(self, user_id: str, ip_address: str, user_agent: str) -> str:
        """Create a new user session."""
        session_id = secrets.token_urlsafe(32)

        # Clean up expired sessions for this user
        self._cleanup_user_sessions(user_id)

        # Check session limit
        user_sessions = [s for s in self.sessions.values() if s.user_id == user_id and s.is_active]
        if len(user_sessions) >= self.max_sessions_per_user:
            # Deactivate oldest session
            oldest_session = min(user_sessions, key=lambda s: s.created_at)
            oldest_session.is_active = False

        session = UserSession(
            session_id=session_id,
            user_id=user_id,
            created_at=datetime.utcnow(),
            expires_at=datetime.utcnow() + timedelta(hours=24),
            ip_address=ip_address,
            user_agent=user_agent
        )

        self.sessions[session_id] = session
        logger.info(f"Created session for user {user_id}")
        return session_id

    def validate_session(self, session_id: str) -> Optional[UserSession]:
        """Validate session and return session info."""
        session = self.sessions.get(session_id)
        if session and session.is_active and session.expires_at > datetime.utcnow():
            return session
        elif session:
            session.is_active = False
        return None

    def invalidate_session(self, session_id: str):
        """Invalidate a session."""
        if session_id in self.sessions:
            self.sessions[session_id].is_active = False
            logger.info(f"Invalidated session {session_id}")

    def invalidate_user_sessions(self, user_id: str):
        """Invalidate all sessions for a user."""
        for session in self.sessions.values():
            if session.user_id == user_id:
                session.is_active = False
        logger.info(f"Invalidated all sessions for user {user_id}")

    def _cleanup_user_sessions(self, user_id: str):
        """Clean up expired sessions for a user."""
        current_time = datetime.utcnow()
        for session in list(self.sessions.values()):
            if (session.user_id == user_id and
                (not session.is_active or session.expires_at < current_time)):
                del self.sessions[session.session_id]

class SecurityAuditor:
    """Audits security events and maintains compliance logs."""

    def __init__(self):
        self.events: List[SecurityEvent] = []
        self.max_events = 10000  # Keep last 10k events in memory

    def log_event(self, event_type: str, user_id: Optional[str], ip_address: str,
                  user_agent: str, severity: str, details: Dict[str, Any],
                  risk_score: float = 0.0):
        """Log a security event."""
        event = SecurityEvent(
            event_id=secrets.token_hex(16),
            event_type=event_type,
            user_id=user_id,
            ip_address=ip_address,
            user_agent=user_agent,
            timestamp=datetime.utcnow(),
            severity=severity,
            details=details,
            risk_score=risk_score
        )

        self.events.append(event)

        # Maintain max events limit
        if len(self.events) > self.max_events:
            self.events = self.events[-self.max_events:]

        logger.info(f"Security event logged: {event_type} (severity: {severity})")

    def get_events(self, user_id: Optional[str] = None, event_type: Optional[str] = None,
                   severity: Optional[str] = None, limit: int = 100) -> List[SecurityEvent]:
        """Get filtered security events."""
        filtered_events = self.events

        if user_id:
            filtered_events = [e for e in filtered_events if e.user_id == user_id]
        if event_type:
            filtered_events = [e for e in filtered_events if e.event_type == event_type]
        if severity:
            filtered_events = [e for e in filtered_events if e.severity == severity]

        return filtered_events[-limit:]

    def get_security_stats(self) -> Dict[str, Any]:
        """Get security statistics."""
        if not self.events:
            return {"total_events": 0}

        recent_events = [e for e in self.events if e.timestamp > datetime.utcnow() - timedelta(hours=24)]

        return {
            "total_events": len(self.events),
            "recent_events_24h": len(recent_events),
            "severity_breakdown": {
                "low": len([e for e in recent_events if e.severity == "low"]),
                "medium": len([e for e in recent_events if e.severity == "medium"]),
                "high": len([e for e in recent_events if e.severity == "high"]),
                "critical": len([e for e in recent_events if e.severity == "critical"])
            },
            "high_risk_events": len([e for e in recent_events if e.risk_score > 0.7])
        }

class InputValidator:
    """Validates and sanitizes user inputs."""

    @staticmethod
    def validate_email(email: str) -> bool:
        """Validate email format."""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))

    @staticmethod
    def validate_ip_address(ip: str) -> bool:
        """Validate IP address."""
        try:
            ipaddress.ip_address(ip)
            return True
        except ValueError:
            return False

    @staticmethod
    def sanitize_string(input_str: str, max_length: int = 255) -> str:
        """Sanitize string input."""
        # Remove potentially dangerous characters
        sanitized = re.sub(r'[<>]', '', input_str)
        return sanitized[:max_length].strip()

    @staticmethod
    def validate_amount(amount: float, min_amount: float = 0.01, max_amount: float = 1000000.0) -> bool:
        """Validate transaction amount."""
        return min_amount <= amount <= max_amount

    @staticmethod
    def detect_suspicious_input(input_str: str) -> Dict[str, Any]:
        """Detect potentially suspicious input patterns."""
        issues = []

        # Check for SQL injection patterns
        sql_patterns = [r'union.*select', r'--', r'/\*', r'\*/', r'xp_', r'sp_']
        for pattern in sql_patterns:
            if re.search(pattern, input_str, re.IGNORECASE):
                issues.append("potential_sql_injection")

        # Check for XSS patterns
        xss_patterns = [r'<script', r'javascript:', r'on\w+\s*=', r'<iframe']
        for pattern in xss_patterns:
            if re.search(pattern, input_str, re.IGNORECASE):
                issues.append("potential_xss")

        # Check for unusual characters
        if len(re.findall(r'[^\w\s@.-]', input_str)) > len(input_str) * 0.3:
            issues.append("high_special_characters")

        return {
            "is_suspicious": len(issues) > 0,
            "issues": issues,
            "risk_score": min(len(issues) * 0.2, 1.0)
        }

class RateLimiter:
    """Implements rate limiting for API endpoints."""

    def __init__(self):
        self.requests: Dict[str, List[datetime]] = {}
        self.max_requests_per_minute = 60
        self.max_requests_per_hour = 1000

    def is_allowed(self, identifier: str) -> bool:
        """Check if request is allowed based on rate limits."""
        now = datetime.utcnow()
        minute_ago = now - timedelta(minutes=1)
        hour_ago = now - timedelta(hours=1)

        if identifier not in self.requests:
            self.requests[identifier] = []

        # Clean old requests
        self.requests[identifier] = [
            req_time for req_time in self.requests[identifier]
            if req_time > hour_ago
        ]

        # Check limits
        recent_minute = [req for req in self.requests[identifier] if req > minute_ago]

        if len(recent_minute) >= self.max_requests_per_minute:
            return False

        if len(self.requests[identifier]) >= self.max_requests_per_hour:
            return False

        # Add current request
        self.requests[identifier].append(now)
        return True

    def get_remaining_requests(self, identifier: str) -> Dict[str, int]:
        """Get remaining requests for an identifier."""
        now = datetime.utcnow()
        minute_ago = now - timedelta(minutes=1)
        hour_ago = now - timedelta(hours=1)

        if identifier not in self.requests:
            return {"minute": self.max_requests_per_minute, "hour": self.max_requests_per_hour}

        recent_minute = [req for req in self.requests[identifier] if req > minute_ago]

        return {
            "minute": max(0, self.max_requests_per_minute - len(recent_minute)),
            "hour": max(0, self.max_requests_per_hour - len(self.requests[identifier]))
        }

class SecurityManager:
    """Central security manager coordinating all security components."""

    def __init__(self):
        self.encryption_manager = get_encryption_manager()
        self.jwt_manager = get_jwt_manager()
        self.session_manager = get_session_manager()
        self.security_auditor = get_security_auditor()
        self.rate_limiter = get_rate_limiter()

    def authenticate_user(self, username: str, password: str) -> Optional[str]:
        """Authenticate user and return session token."""
        # Simplified authentication - in production, verify against user database
        if username and password:  # Mock authentication
            user_id = f"user_{username}"
            session = self.session_manager.create_session(
                user_id, "127.0.0.1", "API Client"
            )
            token = self.jwt_manager.generate_token(user_id, session_id=session.session_id)
            return token
        return None

    def validate_request(self, token: str, ip_address: str, user_agent: str) -> Tuple[bool, Optional[str]]:
        """Validate API request."""
        try:
            # Validate JWT
            payload = self.jwt_manager.validate_token(token)
            user_id = payload.get("user_id")

            # Check rate limits
            if not self.rate_limiter.check_rate_limit(user_id or ip_address):
                return False, "Rate limit exceeded"

            # Log security event
            self.security_auditor.log_event(
                "api_access", user_id, ip_address, user_agent,
                {"endpoint": "validate_request"}, "low"
            )

            return True, user_id

        except Exception as e:
            logger.warning(f"Request validation failed: {e}")
            return False, str(e)

    def encrypt_sensitive_data(self, data: str) -> str:
        """Encrypt sensitive data."""
        return self.encryption_manager.encrypt(data)

    def decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data."""
        return self.encryption_manager.decrypt(encrypted_data)

# Global instances
_encryption_manager = None
_jwt_manager = None
_session_manager = None
_security_auditor = None
_rate_limiter = None
_security_manager = None

def get_encryption_manager() -> EncryptionManager:
    """Get global encryption manager."""
    global _encryption_manager
    if _encryption_manager is None:
        _encryption_manager = EncryptionManager()
    return _encryption_manager

def get_jwt_manager() -> JWTManager:
    """Get global JWT manager."""
    global _jwt_manager
    if _jwt_manager is None:
        secret_key = os.getenv('JWT_SECRET_KEY', secrets.token_hex(32))
        _jwt_manager = JWTManager(secret_key)
    return _jwt_manager

def get_session_manager() -> SessionManager:
    """Get global session manager."""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager(get_encryption_manager())
    return _session_manager

def get_security_auditor() -> SecurityAuditor:
    """Get global security auditor."""
    global _security_auditor
    if _security_auditor is None:
        _security_auditor = SecurityAuditor()
    return _security_auditor

def get_rate_limiter() -> RateLimiter:
    """Get global rate limiter."""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter()
    return _rate_limiter

def get_security_manager() -> SecurityManager:
    """Get global security manager."""
    global _security_manager
    if _security_manager is None:
        _security_manager = SecurityManager()
    return _security_manager