"""
Base repository pattern for data access abstraction.
Provides common CRUD operations and query patterns.
"""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar, List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from uuid import UUID
import logging

from ..database import get_db_session

logger = logging.getLogger(__name__)

# Generic type for model classes
ModelType = TypeVar("ModelType")


class BaseRepository(Generic[ModelType], ABC):
    """
    Abstract base repository class with common CRUD operations.
    """
    
    def __init__(self, model_class: type):
        self.model_class = model_class
    
    def create(self, db: Session, *, obj_in: Dict[str, Any]) -> ModelType:
        """
        Create a new record.
        
        Args:
            db: Database session
            obj_in: Dictionary of model attributes
            
        Returns:
            Created model instance
        """
        try:
            db_obj = self.model_class(**obj_in)
            db.add(db_obj)
            db.commit()
            db.refresh(db_obj)
            logger.debug(f"Created {self.model_class.__name__} with ID: {getattr(db_obj, 'id', 'unknown')}")
            return db_obj
        except SQLAlchemyError as e:
            db.rollback()
            logger.error(f"Failed to create {self.model_class.__name__}: {str(e)}")
            raise
    
    def get(self, db: Session, id: Any) -> Optional[ModelType]:
        """
        Get a record by ID.
        
        Args:
            db: Database session
            id: Record ID
            
        Returns:
            Model instance or None if not found
        """
        try:
            return db.query(self.model_class).filter(self.model_class.id == id).first()
        except SQLAlchemyError as e:
            logger.error(f"Failed to get {self.model_class.__name__} by ID {id}: {str(e)}")
            raise
    
    def get_multi(
        self, 
        db: Session, 
        *, 
        skip: int = 0, 
        limit: int = 100,
        filters: Dict[str, Any] = None
    ) -> List[ModelType]:
        """
        Get multiple records with pagination and filtering.
        
        Args:
            db: Database session
            skip: Number of records to skip
            limit: Maximum number of records to return
            filters: Dictionary of filter criteria
            
        Returns:
            List of model instances
        """
        try:
            query = db.query(self.model_class)
            
            # Apply filters if provided
            if filters:
                for field, value in filters.items():
                    if hasattr(self.model_class, field):
                        query = query.filter(getattr(self.model_class, field) == value)
            
            return query.offset(skip).limit(limit).all()
        except SQLAlchemyError as e:
            logger.error(f"Failed to get multiple {self.model_class.__name__}: {str(e)}")
            raise
    
    def update(self, db: Session, *, db_obj: ModelType, obj_in: Dict[str, Any]) -> ModelType:
        """
        Update a record.
        
        Args:
            db: Database session
            db_obj: Existing model instance
            obj_in: Dictionary of attributes to update
            
        Returns:
            Updated model instance
        """
        try:
            for field, value in obj_in.items():
                if hasattr(db_obj, field):
                    setattr(db_obj, field, value)
            
            db.add(db_obj)
            db.commit()
            db.refresh(db_obj)
            logger.debug(f"Updated {self.model_class.__name__} with ID: {getattr(db_obj, 'id', 'unknown')}")
            return db_obj
        except SQLAlchemyError as e:
            db.rollback()
            logger.error(f"Failed to update {self.model_class.__name__}: {str(e)}")
            raise
    
    def delete(self, db: Session, *, id: Any) -> ModelType:
        """
        Delete a record by ID.
        
        Args:
            db: Database session
            id: Record ID
            
        Returns:
            Deleted model instance
            
        Raises:
            ValueError: If record not found
        """
        try:
            obj = db.query(self.model_class).get(id)
            if not obj:
                raise ValueError(f"{self.model_class.__name__} with ID {id} not found")
            
            db.delete(obj)
            db.commit()
            logger.debug(f"Deleted {self.model_class.__name__} with ID: {id}")
            return obj
        except (SQLAlchemyError, ValueError) as e:
            db.rollback()
            logger.error(f"Failed to delete {self.model_class.__name__} with ID {id}: {str(e)}")
            raise
    
    def count(self, db: Session, filters: Dict[str, Any] = None) -> int:
        """
        Count records with optional filtering.
        
        Args:
            db: Database session
            filters: Dictionary of filter criteria
            
        Returns:
            Number of records
        """
        try:
            query = db.query(self.model_class)
            
            if filters:
                for field, value in filters.items():
                    if hasattr(self.model_class, field):
                        query = query.filter(getattr(self.model_class, field) == value)
            
            return query.count()
        except SQLAlchemyError as e:
            logger.error(f"Failed to count {self.model_class.__name__}: {str(e)}")
            raise
    
    def exists(self, db: Session, id: Any) -> bool:
        """
        Check if a record exists by ID.
        
        Args:
            db: Database session
            id: Record ID
            
        Returns:
            True if record exists, False otherwise
        """
        try:
            return db.query(self.model_class.id).filter(self.model_class.id == id).scalar() is not None
        except SQLAlchemyError as e:
            logger.error(f"Failed to check existence of {self.model_class.__name__} with ID {id}: {str(e)}")
            raise


class TransactionalRepository:
    """
    Mixin class for transactional operations across multiple repositories.
    """
    
    @staticmethod
    def execute_transaction(operations: List[callable]):
        """
        Execute multiple operations in a single transaction.
        
        Args:
            operations: List of callable operations
        """
        with get_db_session() as db:
            try:
                results = []
                for operation in operations:
                    result = operation(db)
                    results.append(result)
                
                db.commit()
                logger.debug(f"Successfully executed {len(operations)} transactional operations")
                return results
            except Exception as e:
                db.rollback()
                logger.error(f"Transaction failed: {str(e)}")
                raise


# Context manager for repository operations
class RepositoryContext:
    """
    Context manager for repository operations with automatic session management.
    """
    
    def __init__(self):
        self.session = None
    
    def __enter__(self) -> Session:
        from ..database import SessionLocal
        self.session = SessionLocal()
        return self.session
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.session.rollback()
            logger.error(f"Repository operation failed: {exc_val}")
        else:
            self.session.commit()
        
        self.session.close()


# Query builder for complex queries
class QueryBuilder:
    """
    Builder pattern for constructing complex database queries.
    """
    
    def __init__(self, session: Session, model_class: type):
        self.session = session
        self.model_class = model_class
        self.query = session.query(model_class)
    
    def filter_by(self, **kwargs) -> 'QueryBuilder':
        """Add filter conditions."""
        self.query = self.query.filter_by(**kwargs)
        return self
    
    def filter(self, *conditions) -> 'QueryBuilder':
        """Add filter conditions using SQLAlchemy expressions."""
        self.query = self.query.filter(*conditions)
        return self
    
    def order_by(self, *columns) -> 'QueryBuilder':
        """Add ordering."""
        self.query = self.query.order_by(*columns)
        return self
    
    def limit(self, limit: int) -> 'QueryBuilder':
        """Add limit."""
        self.query = self.query.limit(limit)
        return self
    
    def offset(self, offset: int) -> 'QueryBuilder':
        """Add offset."""
        self.query = self.query.offset(offset)
        return self
    
    def join(self, *args, **kwargs) -> 'QueryBuilder':
        """Add join."""
        self.query = self.query.join(*args, **kwargs)
        return self
    
    def group_by(self, *columns) -> 'QueryBuilder':
        """Add grouping."""
        self.query = self.query.group_by(*columns)
        return self
    
    def having(self, *conditions) -> 'QueryBuilder':
        """Add having conditions."""
        self.query = self.query.having(*conditions)
        return self
    
    def all(self) -> List:
        """Execute query and return all results."""
        return self.query.all()
    
    def first(self) -> Optional[Any]:
        """Execute query and return first result."""
        return self.query.first()
    
    def count(self) -> int:
        """Execute query and return count."""
        return self.query.count()
    
    def paginate(self, page: int, per_page: int) -> Dict[str, Any]:
        """Execute query with pagination."""
        total = self.query.count()
        items = self.query.offset((page - 1) * per_page).limit(per_page).all()
        
        return {
            "items": items,
            "total": total,
            "page": page,
            "per_page": per_page,
            "pages": (total + per_page - 1) // per_page
        }


# Export key components
__all__ = [
    'BaseRepository',
    'TransactionalRepository', 
    'RepositoryContext',
    'QueryBuilder',
    'ModelType'
]