"""
Analytics API endpoints for fraud statistics and performance tracking.
Provides insights, metrics, and reporting capabilities.
"""

from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from sqlalchemy import func, and_
from datetime import datetime, timedelta
import logging

from ...core.security import User, require_authentication, require_scopes
from ...core.logging import audit_logger
from ...database import get_db
from ...repositories import repository_manager
from ...models import Transaction, Prediction, Feedback, ModelMetrics, RiskLevel
from ...schemas import (
    AnalyticsResponse, FraudStatistics, ModelPerformance, 
    TimeRange, APIResponse, ComponentHealth
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/analytics", tags=["analytics"])


@router.get(
    "/fraud-statistics",
    response_model=FraudStatistics,
    summary="Get fraud statistics",
    description="Retrieve comprehensive fraud statistics for a given time period"
)
async def get_fraud_statistics(
    start_time: datetime = Query(..., description="Start time for statistics"),
    end_time: datetime = Query(..., description="End time for statistics"),
    current_user: User = Depends(require_authentication),
    db: Session = Depends(get_db)
) -> FraudStatistics:
    """
    Get comprehensive fraud statistics for analysis and reporting.
    
    Returns:
    - Transaction volumes and fraud rates
    - Amount-based statistics
    - Risk level distribution
    - Top fraud merchants and countries
    """
    try:
        # Validate time range
        if end_time <= start_time:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="end_time must be after start_time"
            )
        
        # Get repositories
        transaction_repo = repository_manager.get_transaction_repository()
        prediction_repo = repository_manager.get_prediction_repository()
        
        # Get basic transaction statistics
        transaction_stats = transaction_repo.get_aggregate_stats(db, start_time, end_time)
        
        # Get fraud predictions in time range
        fraud_predictions = prediction_repo.get_fraud_predictions(db, start_time, end_time)
        fraud_transaction_ids = [p.transaction_id for p in fraud_predictions]
        
        # Calculate fraud amounts
        if fraud_transaction_ids:
            fraud_transactions = (
                db.query(Transaction)
                .filter(Transaction.id.in_(fraud_transaction_ids))
                .all()
            )
            fraud_amount = sum(t.amount for t in fraud_transactions)
        else:
            fraud_transactions = []
            fraud_amount = 0.0
        
        # Calculate fraud rate
        total_transactions = transaction_stats["total_transactions"]
        fraud_count = len(fraud_predictions)
        fraud_rate = (fraud_count / total_transactions * 100) if total_transactions > 0 else 0.0
        
        # Get risk level distribution
        risk_distribution = (
            db.query(
                Prediction.risk_level,
                func.count(Prediction.id).label('count')
            )
            .filter(and_(
                Prediction.created_at >= start_time,
                Prediction.created_at <= end_time
            ))
            .group_by(Prediction.risk_level)
            .all()
        )
        
        risk_level_distribution = {risk.value: 0 for risk in RiskLevel}
        for row in risk_distribution:
            risk_level_distribution[row.risk_level] = row.count
        
        # Get top fraud merchants
        fraud_merchant_stats = (
            db.query(
                Transaction.merchant_name,
                func.count(Transaction.id).label('fraud_count'),
                func.sum(Transaction.amount).label('fraud_amount')
            )
            .join(Prediction, Transaction.id == Prediction.transaction_id)
            .filter(and_(
                Prediction.is_fraud == True,
                Transaction.transaction_time >= start_time,
                Transaction.transaction_time <= end_time,
                Transaction.merchant_name.isnot(None)
            ))
            .group_by(Transaction.merchant_name)
            .order_by(func.count(Transaction.id).desc())
            .limit(10)
            .all()
        )
        
        top_fraud_merchants = [
            {
                "merchant_name": row.merchant_name,
                "fraud_count": row.fraud_count,
                "fraud_amount": float(row.fraud_amount or 0)
            }
            for row in fraud_merchant_stats
        ]
        
        # Get top fraud countries
        fraud_country_stats = (
            db.query(
                Transaction.country_code,
                func.count(Transaction.id).label('fraud_count'),
                func.sum(Transaction.amount).label('fraud_amount')
            )
            .join(Prediction, Transaction.id == Prediction.transaction_id)
            .filter(and_(
                Prediction.is_fraud == True,
                Transaction.transaction_time >= start_time,
                Transaction.transaction_time <= end_time,
                Transaction.country_code.isnot(None)
            ))
            .group_by(Transaction.country_code)
            .order_by(func.count(Transaction.id).desc())
            .limit(10)
            .all()
        )
        
        top_fraud_countries = [
            {
                "country_code": row.country_code,
                "fraud_count": row.fraud_count,
                "fraud_amount": float(row.fraud_amount or 0)
            }
            for row in fraud_country_stats
        ]
        
        # Log analytics access
        audit_logger.log_api_access(
            user_id=str(current_user.user_id),
            endpoint="/api/v1/analytics/fraud-statistics",
            method="GET",
            success=True,
            additional_data={
                "time_range_hours": (end_time - start_time).total_seconds() / 3600,
                "total_transactions": total_transactions,
                "fraud_rate": fraud_rate
            }
        )
        
        return FraudStatistics(
            time_range=TimeRange(start_time=start_time, end_time=end_time),
            total_transactions=total_transactions,
            fraud_transactions=fraud_count,
            fraud_rate=fraud_rate,
            total_amount=transaction_stats["total_amount"],
            fraud_amount=fraud_amount,
            risk_level_distribution=risk_level_distribution,
            top_fraud_merchants=top_fraud_merchants,
            top_fraud_countries=top_fraud_countries
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get fraud statistics: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve fraud statistics: {str(e)}"
        )


@router.get(
    "/model-performance",
    response_model=List[ModelPerformance],
    summary="Get model performance metrics",
    description="Retrieve performance metrics for all models"
)
async def get_model_performance(
    start_time: datetime = Query(..., description="Start time for metrics"),
    end_time: datetime = Query(..., description="End time for metrics"),
    model_name: Optional[str] = Query(None, description="Specific model name"),
    current_user: User = Depends(require_authentication),
    db: Session = Depends(get_db)
) -> List[ModelPerformance]:
    """Get performance metrics for fraud detection models."""
    try:
        # Get model metrics repository
        metrics_repo = repository_manager.get_model_metrics_repository()
        
        # Query model metrics
        query = (
            db.query(ModelMetrics)
            .filter(and_(
                ModelMetrics.start_time >= start_time,
                ModelMetrics.end_time <= end_time
            ))
        )
        
        if model_name:
            query = query.filter(ModelMetrics.model_name == model_name)
        
        metrics = query.order_by(ModelMetrics.end_time.desc()).all()
        
        if not metrics:
            # Generate metrics from recent predictions if no stored metrics exist
            return await _generate_model_performance(db, start_time, end_time, model_name)
        
        # Convert to response format
        performance_list = []
        for metric in metrics:
            performance = ModelPerformance(
                model_name=metric.model_name,
                model_version=metric.model_version,
                time_range=TimeRange(start_time=metric.start_time, end_time=metric.end_time),
                accuracy=metric.accuracy or 0.0,
                precision=metric.precision or 0.0,
                recall=metric.recall or 0.0,
                f1_score=metric.f1_score or 0.0,
                roc_auc=metric.auc_roc or 0.0,
                true_positives=metric.true_positives or 0,
                true_negatives=metric.true_negatives or 0,
                false_positives=metric.false_positives or 0,
                false_negatives=metric.false_negatives or 0,
                total_predictions=metric.total_predictions or 0,
                avg_prediction_latency_ms=metric.avg_prediction_latency_ms or 0.0
            )
            performance_list.append(performance)
        
        logger.info(f"Retrieved performance metrics for {len(performance_list)} model versions")
        return performance_list
        
    except Exception as e:
        logger.error(f"Failed to get model performance: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve model performance: {str(e)}"
        )


async def _generate_model_performance(
    db: Session,
    start_time: datetime,
    end_time: datetime,
    model_name: Optional[str] = None
) -> List[ModelPerformance]:
    """Generate model performance metrics from prediction and feedback data."""
    try:
        # Get predictions in time range
        prediction_repo = repository_manager.get_prediction_repository()
        feedback_repo = repository_manager.get_feedback_repository()
        
        # Get prediction statistics
        prediction_stats = prediction_repo.get_prediction_stats(db, start_time, end_time, model_name)
        
        # Get feedback statistics for ground truth
        feedback_stats = feedback_repo.get_feedback_stats(db, start_time, end_time)
        
        # Calculate confusion matrix from feedback data
        # This is simplified - in production, you'd need to match predictions with feedback
        tp = feedback_stats["feedback_counts"].get("true_positive", 0)
        fp = feedback_stats["feedback_counts"].get("false_positive", 0)
        tn = feedback_stats["feedback_counts"].get("true_negative", 0)
        fn = feedback_stats["feedback_counts"].get("false_negative", 0)
        
        # Calculate metrics
        total_with_feedback = tp + fp + tn + fn
        if total_with_feedback > 0:
            accuracy = (tp + tn) / total_with_feedback
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        else:
            accuracy = precision = recall = f1_score = 0.0
        
        # Create performance metrics
        performance = ModelPerformance(
            model_name=model_name or "ensemble",
            model_version="current",
            time_range=TimeRange(start_time=start_time, end_time=end_time),
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            roc_auc=0.0,  # Would need actual predictions and labels to calculate
            true_positives=tp,
            true_negatives=tn,
            false_positives=fp,
            false_negatives=fn,
            total_predictions=prediction_stats["total_predictions"],
            avg_prediction_latency_ms=prediction_stats["avg_latency_ms"]
        )
        
        return [performance]
        
    except Exception as e:
        logger.error(f"Failed to generate model performance: {str(e)}")
        return []


@router.get(
    "/dashboard",
    response_model=AnalyticsResponse,
    summary="Get analytics dashboard data",
    description="Retrieve comprehensive analytics data for dashboard"
)
async def get_analytics_dashboard(
    hours: int = Query(default=24, ge=1, le=168, description="Time window in hours"),
    current_user: User = Depends(require_authentication),
    db: Session = Depends(get_db)
) -> AnalyticsResponse:
    """Get comprehensive analytics data for dashboard display."""
    try:
        # Calculate time range
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=hours)
        
        # Get fraud statistics
        fraud_stats = await get_fraud_statistics(start_time, end_time, current_user, db)
        
        # Get model performance
        model_performance = await get_model_performance(start_time, end_time, None, current_user, db)
        
        # Get system health
        system_health = await _get_system_health(db)
        
        return AnalyticsResponse(
            fraud_statistics=fraud_stats,
            model_performance=model_performance,
            system_health=system_health,
            message="Analytics data retrieved successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get dashboard analytics: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve dashboard analytics: {str(e)}"
        )


async def _get_system_health(db: Session) -> Dict[str, Any]:
    """Get system health metrics."""
    try:
        # Database health
        from ...database import check_database_health
        db_health = check_database_health()
        
        # ML pipeline health
        from ...ml.inference.pipeline import fraud_detection_pipeline
        ml_health = {
            "status": "healthy" if fraud_detection_pipeline.is_trained else "not_ready",
            "model_version": fraud_detection_pipeline.model_version,
            "is_trained": fraud_detection_pipeline.is_trained
        }
        
        # Recent prediction volume (last hour)
        recent_predictions = (
            db.query(func.count(Prediction.id))
            .filter(Prediction.created_at >= datetime.utcnow() - timedelta(hours=1))
            .scalar()
        )
        
        return {
            "database": db_health,
            "ml_pipeline": ml_health,
            "recent_predictions_per_hour": recent_predictions or 0,
            "system_uptime_hours": 0,  # Would be calculated from application start time
            "last_check": datetime.utcnow()
        }
        
    except Exception as e:
        logger.error(f"Failed to get system health: {str(e)}")
        return {
            "status": "error",
            "error": str(e),
            "last_check": datetime.utcnow()
        }


@router.get(
    "/trends/fraud-rate",
    response_model=List[Dict[str, Any]],
    summary="Get fraud rate trends",
    description="Get fraud rate trends over time for visualization"
)
async def get_fraud_rate_trends(
    days: int = Query(default=30, ge=1, le=365, description="Number of days for trend"),
    interval_hours: int = Query(default=24, ge=1, le=168, description="Time interval in hours"),
    current_user: User = Depends(require_authentication),
    db: Session = Depends(get_db)
) -> List[Dict[str, Any]]:
    """Get fraud rate trends over time for trend analysis and visualization."""
    try:
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=days)
        
        # Calculate time intervals
        interval_delta = timedelta(hours=interval_hours)
        intervals = []
        current_time = start_time
        
        while current_time < end_time:
            interval_end = min(current_time + interval_delta, end_time)
            intervals.append((current_time, interval_end))
            current_time = interval_end
        
        # Get fraud rate for each interval
        trends = []
        transaction_repo = repository_manager.get_transaction_repository()
        prediction_repo = repository_manager.get_prediction_repository()
        
        for interval_start, interval_end in intervals:
            # Get transaction stats for interval
            stats = transaction_repo.get_aggregate_stats(db, interval_start, interval_end)
            
            # Get fraud predictions for interval
            fraud_predictions = prediction_repo.get_fraud_predictions(db, interval_start, interval_end)
            fraud_count = len(fraud_predictions)
            
            # Calculate fraud rate
            total_transactions = stats["total_transactions"]
            fraud_rate = (fraud_count / total_transactions * 100) if total_transactions > 0 else 0.0
            
            trends.append({
                "timestamp": interval_start,
                "interval_start": interval_start,
                "interval_end": interval_end,
                "total_transactions": total_transactions,
                "fraud_transactions": fraud_count,
                "fraud_rate": round(fraud_rate, 2),
                "total_amount": stats["total_amount"],
                "avg_amount": stats["avg_amount"]
            })
        
        logger.info(f"Retrieved fraud rate trends for {days} days with {len(trends)} data points")
        return trends
        
    except Exception as e:
        logger.error(f"Failed to get fraud rate trends: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve fraud rate trends: {str(e)}"
        )


@router.get(
    "/alerts",
    response_model=List[Dict[str, Any]],
    summary="Get fraud alerts",
    description="Get recent high-risk transactions and alerts"
)
async def get_fraud_alerts(
    hours: int = Query(default=24, ge=1, le=168, description="Time window for alerts"),
    risk_level: Optional[str] = Query(None, description="Minimum risk level (high/critical)"),
    current_user: User = Depends(require_scopes(["alerts"])),
    db: Session = Depends(get_db)
) -> List[Dict[str, Any]]:
    """Get recent fraud alerts for monitoring and response."""
    try:
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=hours)
        
        # Build query for high-risk predictions
        query = (
            db.query(Prediction, Transaction)
            .join(Transaction, Prediction.transaction_id == Transaction.id)
            .filter(and_(
                Prediction.created_at >= start_time,
                Prediction.created_at <= end_time
            ))
        )
        
        # Filter by risk level
        if risk_level:
            if risk_level.lower() == "critical":
                query = query.filter(Prediction.risk_level == "critical")
            elif risk_level.lower() == "high":
                query = query.filter(Prediction.risk_level.in_(["high", "critical"]))
        else:
            # Default to high and critical
            query = query.filter(Prediction.risk_level.in_(["high", "critical"]))
        
        # Order by fraud probability (highest first)
        query = query.order_by(Prediction.fraud_probability.desc())
        
        results = query.limit(100).all()  # Limit to prevent overload
        
        # Format alerts
        alerts = []
        for prediction, transaction in results:
            alert = {
                "alert_id": str(prediction.id),
                "transaction_id": str(transaction.id),
                "user_id": str(transaction.user_id),
                "fraud_probability": prediction.fraud_probability,
                "risk_level": prediction.risk_level,
                "amount": transaction.amount,
                "currency": transaction.currency,
                "merchant_name": transaction.merchant_name,
                "country_code": transaction.country_code,
                "transaction_time": transaction.transaction_time,
                "prediction_time": prediction.created_at,
                "model_version": prediction.model_version,
                "confidence_score": prediction.confidence_score,
                "top_risk_factors": prediction.top_risk_factors or [],
                "severity": "critical" if prediction.risk_level == "critical" else "high"
            }
            alerts.append(alert)
        
        logger.info(f"Retrieved {len(alerts)} fraud alerts for the last {hours} hours")
        return alerts
        
    except Exception as e:
        logger.error(f"Failed to get fraud alerts: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve fraud alerts: {str(e)}"
        )