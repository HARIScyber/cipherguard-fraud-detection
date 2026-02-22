"""
Specific repository implementations for fraud detection models.
Provides specialized queries and operations for each entity type.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import func, and_, or_, desc, asc
from uuid import UUID

from .base import BaseRepository, QueryBuilder
from ..models import (
    User, Transaction, Prediction, Feedback, 
    ModelMetrics, AuditLog, FeedbackType, RiskLevel
)


class UserRepository(BaseRepository[User]):
    """Repository for User model."""
    
    def __init__(self):
        super().__init__(User)
    
    def get_by_username(self, db: Session, username: str) -> Optional[User]:
        """Get user by username."""
        return db.query(User).filter(User.username == username).first()
    
    def get_by_email(self, db: Session, email: str) -> Optional[User]:
        """Get user by email."""
        return db.query(User).filter(User.email == email).first()
    
    def get_by_api_key(self, db: Session, api_key: str) -> Optional[User]:
        """Get user by API key."""
        return db.query(User).filter(User.api_key == api_key).first()
    
    def update_last_login(self, db: Session, user_id: UUID) -> Optional[User]:
        """Update user's last login timestamp."""
        user = self.get(db, user_id)
        if user:
            return self.update(db, db_obj=user, obj_in={"last_login_at": datetime.utcnow()})
        return None
    
    def get_active_users(self, db: Session, skip: int = 0, limit: int = 100) -> List[User]:
        """Get active users."""
        return self.get_multi(db, skip=skip, limit=limit, filters={"is_active": True})


class TransactionRepository(BaseRepository[Transaction]):
    """Repository for Transaction model."""
    
    def __init__(self):
        super().__init__(Transaction)
    
    def get_by_user(
        self, 
        db: Session, 
        user_id: UUID, 
        skip: int = 0, 
        limit: int = 100
    ) -> List[Transaction]:
        """Get transactions by user ID."""
        return (db.query(Transaction)
                .filter(Transaction.user_id == user_id)
                .order_by(desc(Transaction.transaction_time))
                .offset(skip)
                .limit(limit)
                .all())
    
    def get_by_time_range(
        self, 
        db: Session, 
        start_time: datetime, 
        end_time: datetime,
        skip: int = 0,
        limit: int = 1000
    ) -> List[Transaction]:
        """Get transactions within time range."""
        return (db.query(Transaction)
                .filter(and_(
                    Transaction.transaction_time >= start_time,
                    Transaction.transaction_time <= end_time
                ))
                .order_by(desc(Transaction.transaction_time))
                .offset(skip)
                .limit(limit)
                .all())
    
    def get_by_amount_range(
        self, 
        db: Session, 
        min_amount: float, 
        max_amount: float,
        skip: int = 0,
        limit: int = 100
    ) -> List[Transaction]:
        """Get transactions within amount range."""
        return (db.query(Transaction)
                .filter(and_(
                    Transaction.amount >= min_amount,
                    Transaction.amount <= max_amount
                ))
                .order_by(desc(Transaction.transaction_time))
                .offset(skip)
                .limit(limit)
                .all())
    
    def get_recent_by_user(
        self, 
        db: Session, 
        user_id: UUID, 
        hours: int = 24,
        limit: int = 10
    ) -> List[Transaction]:
        """Get recent transactions for a user."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        return (db.query(Transaction)
                .filter(and_(
                    Transaction.user_id == user_id,
                    Transaction.transaction_time >= cutoff_time
                ))
                .order_by(desc(Transaction.transaction_time))
                .limit(limit)
                .all())
    
    def get_by_merchant(
        self, 
        db: Session, 
        merchant_name: str,
        skip: int = 0,
        limit: int = 100
    ) -> List[Transaction]:
        """Get transactions by merchant."""
        return (db.query(Transaction)
                .filter(Transaction.merchant_name == merchant_name)
                .order_by(desc(Transaction.transaction_time))
                .offset(skip)
                .limit(limit)
                .all())
    
    def get_by_ip_address(
        self, 
        db: Session, 
        ip_address: str,
        skip: int = 0,
        limit: int = 100
    ) -> List[Transaction]:
        """Get transactions by IP address."""
        return (db.query(Transaction)
                .filter(Transaction.ip_address == ip_address)
                .order_by(desc(Transaction.transaction_time))
                .offset(skip)
                .limit(limit)
                .all())
    
    def get_aggregate_stats(
        self, 
        db: Session, 
        start_time: datetime, 
        end_time: datetime
    ) -> Dict[str, Any]:
        """Get aggregate transaction statistics."""
        result = (db.query(
                    func.count(Transaction.id).label('total_transactions'),
                    func.sum(Transaction.amount).label('total_amount'),
                    func.avg(Transaction.amount).label('avg_amount'),
                    func.min(Transaction.amount).label('min_amount'),
                    func.max(Transaction.amount).label('max_amount')
                )
                .filter(and_(
                    Transaction.transaction_time >= start_time,
                    Transaction.transaction_time <= end_time
                ))
                .first())
        
        return {
            "total_transactions": result.total_transactions or 0,
            "total_amount": float(result.total_amount or 0),
            "avg_amount": float(result.avg_amount or 0),
            "min_amount": float(result.min_amount or 0),
            "max_amount": float(result.max_amount or 0)
        }


class PredictionRepository(BaseRepository[Prediction]):
    """Repository for Prediction model."""
    
    def __init__(self):
        super().__init__(Prediction)
    
    def get_by_transaction(self, db: Session, transaction_id: UUID) -> List[Prediction]:
        """Get predictions for a transaction."""
        return (db.query(Prediction)
                .filter(Prediction.transaction_id == transaction_id)
                .order_by(desc(Prediction.created_at))
                .all())
    
    def get_latest_by_transaction(self, db: Session, transaction_id: UUID) -> Optional[Prediction]:
        """Get latest prediction for a transaction."""
        return (db.query(Prediction)
                .filter(Prediction.transaction_id == transaction_id)
                .order_by(desc(Prediction.created_at))
                .first())
    
    def get_by_model(
        self, 
        db: Session, 
        model_name: str,
        skip: int = 0,
        limit: int = 100
    ) -> List[Prediction]:
        """Get predictions by model."""
        return (db.query(Prediction)
                .filter(Prediction.model_name == model_name)
                .order_by(desc(Prediction.created_at))
                .offset(skip)
                .limit(limit)
                .all())
    
    def get_by_risk_level(
        self, 
        db: Session, 
        risk_level: RiskLevel,
        skip: int = 0,
        limit: int = 100
    ) -> List[Prediction]:
        """Get predictions by risk level."""
        return (db.query(Prediction)
                .filter(Prediction.risk_level == risk_level.value)
                .order_by(desc(Prediction.created_at))
                .offset(skip)
                .limit(limit)
                .all())
    
    def get_fraud_predictions(
        self, 
        db: Session, 
        start_time: datetime, 
        end_time: datetime
    ) -> List[Prediction]:
        """Get fraud predictions within time range."""
        return (db.query(Prediction)
                .filter(and_(
                    Prediction.is_fraud == True,
                    Prediction.created_at >= start_time,
                    Prediction.created_at <= end_time
                ))
                .order_by(desc(Prediction.created_at))
                .all())
    
    def get_prediction_stats(
        self, 
        db: Session, 
        start_time: datetime, 
        end_time: datetime,
        model_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get prediction statistics."""
        query = (db.query(
                    func.count(Prediction.id).label('total_predictions'),
                    func.sum(func.cast(Prediction.is_fraud, db.dialect.name == 'sqlite' and int or int)).label('fraud_predictions'),
                    func.avg(Prediction.fraud_probability).label('avg_fraud_prob'),
                    func.avg(Prediction.prediction_latency_ms).label('avg_latency_ms')
                )
                .filter(and_(
                    Prediction.created_at >= start_time,
                    Prediction.created_at <= end_time
                )))
        
        if model_name:
            query = query.filter(Prediction.model_name == model_name)
        
        result = query.first()
        
        return {
            "total_predictions": result.total_predictions or 0,
            "fraud_predictions": result.fraud_predictions or 0,
            "fraud_rate": (result.fraud_predictions / result.total_predictions * 100) if result.total_predictions else 0,
            "avg_fraud_probability": float(result.avg_fraud_prob or 0),
            "avg_latency_ms": float(result.avg_latency_ms or 0)
        }


class FeedbackRepository(BaseRepository[Feedback]):
    """Repository for Feedback model."""
    
    def __init__(self):
        super().__init__(Feedback)
    
    def get_by_transaction(self, db: Session, transaction_id: UUID) -> List[Feedback]:
        """Get feedback for a transaction."""
        return (db.query(Feedback)
                .filter(Feedback.transaction_id == transaction_id)
                .order_by(desc(Feedback.created_at))
                .all())
    
    def get_by_type(
        self, 
        db: Session, 
        feedback_type: FeedbackType,
        skip: int = 0,
        limit: int = 100
    ) -> List[Feedback]:
        """Get feedback by type."""
        return (db.query(Feedback)
                .filter(Feedback.feedback_type == feedback_type.value)
                .order_by(desc(Feedback.created_at))
                .offset(skip)
                .limit(limit)
                .all())
    
    def get_pending_review(
        self, 
        db: Session,
        skip: int = 0,
        limit: int = 100
    ) -> List[Feedback]:
        """Get feedback pending review."""
        return (db.query(Feedback)
                .filter(Feedback.review_status == "pending")
                .order_by(asc(Feedback.created_at))
                .offset(skip)
                .limit(limit)
                .all())
    
    def get_feedback_stats(
        self, 
        db: Session, 
        start_time: datetime, 
        end_time: datetime
    ) -> Dict[str, Any]:
        """Get feedback statistics."""
        result = (db.query(
                    Feedback.feedback_type,
                    func.count(Feedback.id).label('count')
                )
                .filter(and_(
                    Feedback.created_at >= start_time,
                    Feedback.created_at <= end_time
                ))
                .group_by(Feedback.feedback_type)
                .all())
        
        stats = {ft.value: 0 for ft in FeedbackType}
        for row in result:
            stats[row.feedback_type] = row.count
        
        total = sum(stats.values())
        return {
            "feedback_counts": stats,
            "total_feedback": total,
            "feedback_distribution": {k: (v / total * 100) if total else 0 for k, v in stats.items()}
        }


class ModelMetricsRepository(BaseRepository[ModelMetrics]):
    """Repository for ModelMetrics model."""
    
    def __init__(self):
        super().__init__(ModelMetrics)
    
    def get_by_model(
        self, 
        db: Session, 
        model_name: str,
        skip: int = 0,
        limit: int = 100
    ) -> List[ModelMetrics]:
        """Get metrics by model name."""
        return (db.query(ModelMetrics)
                .filter(ModelMetrics.model_name == model_name)
                .order_by(desc(ModelMetrics.end_time))
                .offset(skip)
                .limit(limit)
                .all())
    
    def get_latest_metrics(self, db: Session, model_name: str) -> Optional[ModelMetrics]:
        """Get latest metrics for a model."""
        return (db.query(ModelMetrics)
                .filter(ModelMetrics.model_name == model_name)
                .order_by(desc(ModelMetrics.end_time))
                .first())
    
    def get_metrics_trend(
        self, 
        db: Session, 
        model_name: str, 
        days: int = 30
    ) -> List[ModelMetrics]:
        """Get metrics trend for a model."""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        return (db.query(ModelMetrics)
                .filter(and_(
                    ModelMetrics.model_name == model_name,
                    ModelMetrics.end_time >= cutoff_date
                ))
                .order_by(asc(ModelMetrics.end_time))
                .all())


class AuditLogRepository(BaseRepository[AuditLog]):
    """Repository for AuditLog model."""
    
    def __init__(self):
        super().__init__(AuditLog)
    
    def get_by_user(
        self, 
        db: Session, 
        user_id: UUID,
        skip: int = 0,
        limit: int = 100
    ) -> List[AuditLog]:
        """Get audit logs by user."""
        return (db.query(AuditLog)
                .filter(AuditLog.user_id == user_id)
                .order_by(desc(AuditLog.created_at))
                .offset(skip)
                .limit(limit)
                .all())
    
    def get_by_event_type(
        self, 
        db: Session, 
        event_type: str,
        skip: int = 0,
        limit: int = 100
    ) -> List[AuditLog]:
        """Get audit logs by event type."""
        return (db.query(AuditLog)
                .filter(AuditLog.event_type == event_type)
                .order_by(desc(AuditLog.created_at))
                .offset(skip)
                .limit(limit)
                .all())
    
    def get_security_events(
        self, 
        db: Session, 
        start_time: datetime, 
        end_time: datetime
    ) -> List[AuditLog]:
        """Get security-related audit logs."""
        return (db.query(AuditLog)
                .filter(and_(
                    AuditLog.event_category == "security",
                    AuditLog.created_at >= start_time,
                    AuditLog.created_at <= end_time
                ))
                .order_by(desc(AuditLog.created_at))
                .all())
    
    def get_failed_events(
        self, 
        db: Session, 
        start_time: datetime, 
        end_time: datetime
    ) -> List[AuditLog]:
        """Get failed events."""
        return (db.query(AuditLog)
                .filter(and_(
                    AuditLog.success == False,
                    AuditLog.created_at >= start_time,
                    AuditLog.created_at <= end_time
                ))
                .order_by(desc(AuditLog.created_at))
                .all())


# Repository factory
class RepositoryManager:
    """Factory class for repository instances."""
    
    def __init__(self):
        self._repositories = {}
    
    def get_user_repository(self) -> UserRepository:
        """Get user repository instance."""
        if 'user' not in self._repositories:
            self._repositories['user'] = UserRepository()
        return self._repositories['user']
    
    def get_transaction_repository(self) -> TransactionRepository:
        """Get transaction repository instance."""
        if 'transaction' not in self._repositories:
            self._repositories['transaction'] = TransactionRepository()
        return self._repositories['transaction']
    
    def get_prediction_repository(self) -> PredictionRepository:
        """Get prediction repository instance."""
        if 'prediction' not in self._repositories:
            self._repositories['prediction'] = PredictionRepository()
        return self._repositories['prediction']
    
    def get_feedback_repository(self) -> FeedbackRepository:
        """Get feedback repository instance."""
        if 'feedback' not in self._repositories:
            self._repositories['feedback'] = FeedbackRepository()
        return self._repositories['feedback']
    
    def get_model_metrics_repository(self) -> ModelMetricsRepository:
        """Get model metrics repository instance."""
        if 'model_metrics' not in self._repositories:
            self._repositories['model_metrics'] = ModelMetricsRepository()
        return self._repositories['model_metrics']
    
    def get_audit_log_repository(self) -> AuditLogRepository:
        """Get audit log repository instance."""
        if 'audit_log' not in self._repositories:
            self._repositories['audit_log'] = AuditLogRepository()
        return self._repositories['audit_log']


# Global repository manager instance
repository_manager = RepositoryManager()

# Export key components
__all__ = [
    'UserRepository',
    'TransactionRepository',
    'PredictionRepository', 
    'FeedbackRepository',
    'ModelMetricsRepository',
    'AuditLogRepository',
    'RepositoryManager',
    'repository_manager'
]