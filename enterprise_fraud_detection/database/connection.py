"""
Database connection and session management
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
from contextlib import contextmanager
from typing import Generator
import logging

from app.core.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)


class DatabaseManager:
    """Database connection and session manager."""
    
    def __init__(self):
        self.engine = None
        self.session_local = None
        self._setup_database()
    
    def _setup_database(self):
        """Setup database engine and session factory."""
        try:
            # Create engine with connection pooling
            self.engine = create_engine(
                settings.database.url,
                poolclass=QueuePool,
                pool_size=settings.database.pool_size,
                max_overflow=settings.database.max_overflow,
                pool_timeout=settings.database.pool_timeout,
                pool_recycle=settings.database.pool_recycle,
                pool_pre_ping=True,  # Validate connections before use
                echo=settings.database.echo,  # Log SQL statements if enabled
            )
            
            # Create session factory
            self.session_local = sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=self.engine
            )
            
            logger.info("Database connection established successfully")
            
        except Exception as e:
            logger.error(f"Failed to setup database: {e}")
            raise
    
    def get_session(self) -> Session:
        """Get a new database session."""
        if not self.session_local:
            raise RuntimeError("Database not initialized")
        return self.session_local()
    
    @contextmanager
    def get_session_context(self) -> Generator[Session, None, None]:
        """Get database session with automatic cleanup."""
        session = self.get_session()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
    
    def create_tables(self):
        """Create all database tables."""
        try:
            from database.models import create_tables
            create_tables(self.engine)
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Failed to create tables: {e}")
            raise
    
    def health_check(self) -> bool:
        """Check database connection health."""
        try:
            with self.get_session_context() as session:
                session.execute("SELECT 1")
            return True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False
    
    def close(self):
        """Close database connections."""
        if self.engine:
            self.engine.dispose()
            logger.info("Database connections closed")


# Global database manager instance
db_manager = DatabaseManager()


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency for database sessions."""
    with db_manager.get_session_context() as session:
        yield session


def init_database():
    """Initialize database and create tables."""
    try:
        db_manager.create_tables()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise