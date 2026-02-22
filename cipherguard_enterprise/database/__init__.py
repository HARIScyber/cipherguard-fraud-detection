"""
Database configuration and session management with SQLAlchemy.
Provides connection pooling, async support, and migration capabilities.
"""

from sqlalchemy import create_engine, MetaData, event
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
from contextlib import contextmanager
from typing import Generator
import logging

from ..core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Create the SQLAlchemy engine
engine = create_engine(
    settings.database.database_url,
    pool_size=settings.database.pool_size,
    max_overflow=settings.database.max_overflow,
    pool_timeout=settings.database.pool_timeout,
    pool_recycle=settings.database.pool_recycle,
    echo=settings.database.echo_sql,
    # Additional engine options for production
    pool_pre_ping=True,  # Verify connections before use
    connect_args={
        "check_same_thread": False,  # For SQLite compatibility
        "connect_timeout": 30,  # Connection timeout
    } if "sqlite" in settings.database.database_url else {}
)

# Create SessionLocal class
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

# Create declarative base
Base = declarative_base()

# Metadata for migrations
metadata = MetaData()


@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """
    Context manager for database sessions.
    
    Yields:
        Database session
    """
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_db() -> Generator[Session, None, None]:
    """
    Dependency to get database session for FastAPI.
    
    Yields:
        Database session
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_tables():
    """Create all database tables."""
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Failed to create database tables: {str(e)}")
        raise


def drop_tables():
    """Drop all database tables."""
    try:
        Base.metadata.drop_all(bind=engine)
        logger.info("Database tables dropped successfully")
    except Exception as e:
        logger.error(f"Failed to drop database tables: {str(e)}")
        raise


# Database event listeners for monitoring and optimization
@event.listens_for(engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    """Set SQLite pragmas for better performance."""
    if "sqlite" in settings.database.database_url:
        cursor = dbapi_connection.cursor()
        # Enable foreign keys
        cursor.execute("PRAGMA foreign_keys=ON")
        # Set journal mode to WAL for better concurrency
        cursor.execute("PRAGMA journal_mode=WAL")
        # Set synchronous mode for better performance
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.close()


@event.listens_for(engine, "checkout")
def receive_checkout(dbapi_connection, connection_record, connection_proxy):
    """Log database connection checkout."""
    logger.debug("Database connection checked out from pool")


@event.listens_for(engine, "checkin")
def receive_checkin(dbapi_connection, connection_record):
    """Log database connection checkin."""
    logger.debug("Database connection returned to pool")


# Health check function
def check_database_health() -> dict:
    """
    Check database connection health.
    
    Returns:
        Health status dictionary
    """
    try:
        with get_db_session() as db:
            # Simple query to test connection
            result = db.execute("SELECT 1").scalar()
            
            # Get pool status
            pool = engine.pool
            pool_status = {
                "size": pool.size(),
                "checked_out": pool.checkedout(),
                "overflow": pool.overflow(),
                "checked_in": pool.checkedin()
            }
            
            return {
                "status": "healthy",
                "connection_test": result == 1,
                "pool_status": pool_status,
                "database_url": settings.database.database_url.split("@")[-1] if "@" in settings.database.database_url else settings.database.database_url
            }
    except Exception as e:
        logger.error(f"Database health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }


# Initialize database
def init_database():
    """Initialize database with tables and initial data."""
    try:
        logger.info("Initializing database...")
        create_tables()
        logger.info("Database initialization completed successfully")
    except Exception as e:
        logger.error(f"Database initialization failed: {str(e)}")
        raise


# Export key components
__all__ = [
    'engine',
    'SessionLocal', 
    'Base',
    'metadata',
    'get_db_session',
    'get_db',
    'create_tables',
    'drop_tables', 
    'check_database_health',
    'init_database'
]