"""
Database configuration and session management for Comment Sentiment Analysis API.
Production-ready PostgreSQL setup with SQLAlchemy.
"""

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
import os
import logging
from typing import Generator

logger = logging.getLogger(__name__)

# Database URL from environment variables
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:password@localhost:5432/sentiment_analysis"
)

# For development, you can use SQLite as fallback
if os.getenv("USE_SQLITE", "false").lower() == "true":
    DATABASE_URL = "sqlite:///./sentiment_analysis.db"
    logger.warning("Using SQLite database for development")

# Create SQLAlchemy engine
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,  # Enable connection health checks
    pool_recycle=300,    # Recycle connections every 5 minutes
    echo=os.getenv("DEBUG", "false").lower() == "true"  # Log SQL queries in debug mode
)

# Create SessionLocal class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create Base class for models
Base = declarative_base()

def get_db() -> Generator[Session, None, None]:
    """
    Dependency function to get database session.
    
    Yields:
        Database session
    """
    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        logger.error(f"Database session error: {e}")
        db.rollback()
        raise
    finally:
        db.close()

def create_tables():
    """Create all database tables."""
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Failed to create database tables: {e}")
        raise

def check_database_connection() -> bool:
    """
    Check if database connection is healthy.
    
    Returns:
        True if connection is successful, False otherwise
    """
    try:
        db = SessionLocal()
        db.execute("SELECT 1")
        db.close()
        return True
    except Exception as e:
        logger.error(f"Database connection check failed: {e}")
        return False


def init_db():
    """Initialize database tables."""
    create_tables()


def create_admin_user():
    """Create admin user from environment variables."""
    import os
    from .models import User
    from .auth import hash_password
    
    # Get admin credentials from environment
    admin_username = os.getenv("ADMIN_USERNAME", "admin")
    admin_email = os.getenv("ADMIN_EMAIL", "admin@example.com")
    admin_password = os.getenv("ADMIN_PASSWORD", "admin123")
    admin_full_name = os.getenv("ADMIN_FULL_NAME", "System Administrator")
    
    db = SessionLocal()
    
    try:
        # Check if admin user already exists
        existing_admin = db.query(User).filter(User.username == admin_username).first()
        
        if existing_admin:
            logger.info(f"Admin user '{admin_username}' already exists")
            return False  # Admin already exists
        
        # Create admin user
        hashed_password = hash_password(admin_password)
        admin_user = User(
            username=admin_username,
            email=admin_email,
            full_name=admin_full_name,
            hashed_password=hashed_password,
            is_active=True,
            is_admin=True
        )
        
        db.add(admin_user)
        db.commit()
        logger.info(f"Admin user '{admin_username}' created successfully")
        return True
    
    except Exception as e:
        logger.error(f"Failed to create admin user: {e}")
        db.rollback()
        raise e
    finally:
        db.close()