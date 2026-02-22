"""
API routes package.
Imports all route modules for easy access.
"""

from .auth import router as auth_router
from .comments import router as comments_router
from .health import router as health_router

__all__ = [
    "auth_router",
    "comments_router", 
    "health_router"
]
