"""
Repository layer for data access operations
"""

from sqlalchemy.orm import Session
from sqlalchemy import func, desc, and_, or_
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
from decimal import Decimal
import json
import logging

from database.models import (
    Transaction, FraudPrediction, UserBehavior, 
    UserFeedback, ModelPerformance, SystemHealth
)

logger = logging.getLogger(__name__)


class TransactionRepository:
    """Repository for transaction data operations."""
    
    def __init__(self, session: Session):
        self.session = session
    
    def create_transaction(self, transaction_data: Dict[str, Any]) -> Transaction:
        """Create a new transaction record."""
        transaction = Transaction(
            transaction_id=transaction_data['transaction_id'],
            amount=Decimal(str(transaction_data['amount'])),
            merchant=transaction_data['merchant'],
            device_type=transaction_data['device'],
            country_code=transaction_data['country'],
            customer_id=transaction_data.get('customer_id'),
            ip_address=transaction_data.get('ip_address'),
            user_agent=transaction_data.get('user_agent'),
            session_id=transaction_data.get('session_id'),
            transaction_time=transaction_data.get('timestamp', datetime.utcnow()),
            raw_data=json.dumps(transaction_data)
        )
        
        self.session.add(transaction)
        self.session.flush()
        return transaction
    
    def get_transaction(self, transaction_id: str) -> Optional[Transaction]:
        """Get transaction by ID."""
        return self.session.query(Transaction).filter(
            Transaction.transaction_id == transaction_id
        ).first()
    
    def get_user_transactions(
        self, 
        customer_id: str, 
        days: int = 30, 
        limit: int = 100
    ) -> List[Transaction]:
        """Get recent transactions for a user."""
        since_date = datetime.utcnow() - timedelta(days=days)
        
        return self.session.query(Transaction).filter(
            and_(
                Transaction.customer_id == customer_id,
                Transaction.transaction_time >= since_date
            )
        ).order_by(desc(Transaction.transaction_time)).limit(limit).all()
    
    def get_user_velocity_metrics(
        self, 
        customer_id: str, 
        window_minutes: int = 60
    ) -> Dict[str, Any]:
        """Get user transaction velocity metrics."""
        since_time = datetime.utcnow() - timedelta(minutes=window_minutes)
        
        result = self.session.query(
            func.count(Transaction.id).label('transaction_count'),
            func.sum(Transaction.amount).label('total_amount'),
            func.avg(Transaction.amount).label('avg_amount'),
            func.count(func.distinct(Transaction.merchant)).label('unique_merchants')
        ).filter(
            and_(
                Transaction.customer_id == customer_id,
                Transaction.transaction_time >= since_time
            )
        ).first()
        
        return {
            'transaction_count': result.transaction_count or 0,
            'total_amount': float(result.total_amount or 0),
            'avg_amount': float(result.avg_amount or 0),
            'unique_merchants': result.unique_merchants or 0,
            'window_minutes': window_minutes
        }
    
    def get_merchant_statistics(self, merchant: str, days: int = 7) -> Dict[str, Any]:
        """Get transaction statistics for a merchant."""
        since_date = datetime.utcnow() - timedelta(days=days)
        
        result = self.session.query(
            func.count(Transaction.id).label('transaction_count'),
            func.avg(Transaction.amount).label('avg_amount'),
            func.min(Transaction.amount).label('min_amount'),
            func.max(Transaction.amount).label('max_amount'),
            func.count(func.distinct(Transaction.customer_id)).label('unique_customers')
        ).filter(
            and_(
                Transaction.merchant == merchant,
                Transaction.transaction_time >= since_date
            )
        ).first()
        
        return {
            'transaction_count': result.transaction_count or 0,
            'avg_amount': float(result.avg_amount or 0),
            'min_amount': float(result.min_amount or 0),
            'max_amount': float(result.max_amount or 0),
            'unique_customers': result.unique_customers or 0,
            'days': days
        }


class FraudPredictionRepository:
    """Repository for fraud prediction data operations."""
    
    def __init__(self, session: Session):
        self.session = session
    
    def create_prediction(self, prediction_data: Dict[str, Any]) -> FraudPrediction:
        """Create a fraud prediction record."""
        prediction = FraudPrediction(
            transaction_id=prediction_data['transaction_id'],
            fraud_score=prediction_data['fraud_score'],
            is_fraud=prediction_data['is_fraud'],
            risk_level=prediction_data['risk_level'],
            confidence_score=prediction_data['confidence_score'],
            model_version=prediction_data['model_version'],
            model_type=prediction_data['model_type'],
            isolation_forest_score=prediction_data.get('isolation_forest_score'),
            random_forest_score=prediction_data.get('random_forest_score'),
            xgboost_score=prediction_data.get('xgboost_score'),
            top_risk_factors=json.dumps(prediction_data.get('top_risk_factors', [])),
            shap_values=json.dumps(prediction_data.get('shap_values', [])),
            processing_time_ms=prediction_data['processing_time_ms'],
            feature_vector=json.dumps(prediction_data.get('feature_vector', []))
        )
        
        self.session.add(prediction)
        self.session.flush()
        return prediction
    
    def get_prediction(self, transaction_id: str) -> Optional[FraudPrediction]:
        """Get fraud prediction by transaction ID."""
        return self.session.query(FraudPrediction).filter(
            FraudPrediction.transaction_id == transaction_id
        ).first()
    
    def get_fraud_statistics(self, days: int = 7) -> Dict[str, Any]:
        """Get fraud detection statistics."""
        since_date = datetime.utcnow() - timedelta(days=days)
        
        total_predictions = self.session.query(FraudPrediction).filter(
            FraudPrediction.created_at >= since_date
        ).count()
        
        fraud_predictions = self.session.query(FraudPrediction).filter(
            and_(
                FraudPrediction.created_at >= since_date,
                FraudPrediction.is_fraud == True
            )
        ).count()
        
        risk_distribution = self.session.query(
            FraudPrediction.risk_level,
            func.count(FraudPrediction.id).label('count')
        ).filter(
            FraudPrediction.created_at >= since_date
        ).group_by(FraudPrediction.risk_level).all()
        
        avg_processing_time = self.session.query(
            func.avg(FraudPrediction.processing_time_ms)
        ).filter(
            FraudPrediction.created_at >= since_date
        ).scalar() or 0
        
        return {
            'total_predictions': total_predictions,
            'fraud_predictions': fraud_predictions,
            'fraud_rate': (fraud_predictions / max(total_predictions, 1)) * 100,
            'risk_distribution': {level: count for level, count in risk_distribution},
            'avg_processing_time_ms': float(avg_processing_time),
            'days': days
        }


class UserBehaviorRepository:
    """Repository for user behavior tracking."""
    
    def __init__(self, session: Session):
        self.session = session
    
    def get_or_create_user_behavior(self, customer_id: str) -> UserBehavior:
        """Get existing user behavior record or create new one."""
        behavior = self.session.query(UserBehavior).filter(
            UserBehavior.customer_id == customer_id
        ).first()
        
        if not behavior:
            behavior = UserBehavior(customer_id=customer_id)
            self.session.add(behavior)
            self.session.flush()
        
        return behavior
    
    def update_user_behavior(self, customer_id: str, transaction: Transaction):
        """Update user behavior metrics after a new transaction."""
        behavior = self.get_or_create_user_behavior(customer_id)
        
        # Update totals
        behavior.total_transactions += 1
        behavior.total_amount += transaction.amount
        behavior.avg_transaction_amount = behavior.total_amount / behavior.total_transactions
        
        # Update timestamps
        if not behavior.first_transaction:
            behavior.first_transaction = transaction.transaction_time
        behavior.last_transaction = transaction.transaction_time
        
        # Update velocity metrics (these would be calculated by a background job)
        # For now, we'll update them with recent transaction counts
        hour_ago = datetime.utcnow() - timedelta(hours=1)
        day_ago = datetime.utcnow() - timedelta(days=1)
        
        behavior.transactions_last_hour = self.session.query(Transaction).filter(
            and_(
                Transaction.customer_id == customer_id,
                Transaction.transaction_time >= hour_ago
            )
        ).count()
        
        behavior.transactions_last_day = self.session.query(Transaction).filter(
            and_(
                Transaction.customer_id == customer_id,
                Transaction.transaction_time >= day_ago
            )
        ).count()
        
        self.session.flush()


class FeedbackRepository:
    """Repository for user feedback operations."""
    
    def __init__(self, session: Session):
        self.session = session
    
    def create_feedback(self, feedback_data: Dict[str, Any]) -> UserFeedback:
        """Create a new feedback record."""
        feedback = UserFeedback(
            transaction_id=feedback_data['transaction_id'],
            is_actual_fraud=feedback_data['is_actual_fraud'],
            feedback_type=feedback_data['feedback_type'],
            analyst_id=feedback_data.get('analyst_id'),
            comments=feedback_data.get('comments'),
            confidence=feedback_data.get('confidence'),
            original_fraud_score=feedback_data.get('original_fraud_score'),
            original_prediction=feedback_data.get('original_prediction')
        )
        
        self.session.add(feedback)
        self.session.flush()
        return feedback
    
    def get_feedback_for_retraining(self, limit: int = 1000) -> List[Tuple[str, bool]]:
        """Get feedback data for model retraining."""
        feedback_records = self.session.query(
            UserFeedback.transaction_id,
            UserFeedback.is_actual_fraud
        ).limit(limit).all()
        
        return [(record.transaction_id, record.is_actual_fraud) for record in feedback_records]


class ModelPerformanceRepository:
    """Repository for model performance tracking."""
    
    def __init__(self, session: Session):
        self.session = session
    
    def create_performance_record(self, performance_data: Dict[str, Any]) -> ModelPerformance:
        """Create a model performance record."""
        performance = ModelPerformance(
            model_version=performance_data['model_version'],
            model_type=performance_data['model_type'],
            training_date=performance_data['training_date'],
            accuracy=performance_data['accuracy'],
            precision=performance_data['precision'],
            recall=performance_data['recall'],
            f1_score=performance_data['f1_score'],
            roc_auc=performance_data['roc_auc'],
            training_samples=performance_data['training_samples'],
            test_samples=performance_data['test_samples'],
            training_time_seconds=performance_data['training_time_seconds'],
            cv_mean_accuracy=performance_data.get('cv_mean_accuracy'),
            cv_std_accuracy=performance_data.get('cv_std_accuracy'),
            hyperparameters=json.dumps(performance_data.get('hyperparameters', {})),
            feature_importance=json.dumps(performance_data.get('feature_importance', {}))
        )
        
        self.session.add(performance)
        self.session.flush()
        return performance
    
    def get_latest_performance(self, model_type: str) -> Optional[ModelPerformance]:
        """Get latest performance metrics for a model type."""
        return self.session.query(ModelPerformance).filter(
            ModelPerformance.model_type == model_type
        ).order_by(desc(ModelPerformance.training_date)).first()