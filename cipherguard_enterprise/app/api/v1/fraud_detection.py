"""
Main fraud detection API endpoints.
Handles fraud detection requests, feedback, and predictions.
"""

from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
import logging
from uuid import UUID, uuid4

from ...core.security import User, require_authentication, require_scopes
from ...core.logging import audit_logger, performance_logger
from ...database import get_db
from ...repositories import repository_manager
from ...models import Transaction, Prediction, Feedback, RiskLevel, FeedbackType
from ...schemas import (
    FraudDetectionRequest, FraudDetectionResponse, RiskFactor,
    FeedbackCreate, FeedbackResponse, APIResponse, ErrorResponse
)
from ...ml.inference.pipeline import fraud_detection_pipeline

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["fraud-detection"])


@router.post(
    "/detect",
    response_model=FraudDetectionResponse,
    summary="Detect fraud in a transaction",
    description="Analyze a transaction for fraud using advanced ML ensemble models"
)
async def detect_fraud(
    request: FraudDetectionRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(require_authentication),
    db: Session = Depends(get_db)
) -> FraudDetectionResponse:
    """
    Detect fraud in a transaction using advanced ML models.
    
    This endpoint:
    1. Validates the transaction data
    2. Applies feature engineering
    3. Runs ensemble ML models (IsolationForest, RandomForest, XGBoost)
    4. Returns fraud probability, risk level, and explanations
    5. Stores the prediction for future analysis
    """
    start_time = datetime.utcnow()
    
    try:
        # Log the fraud detection request
        audit_logger.log_api_access(
            user_id=str(current_user.user_id),
            endpoint="/api/v1/detect",
            method="POST",
            success=True,
            additional_data={
                "transaction_amount": request.transaction.amount,
                "transaction_type": request.transaction.transaction_type,
                "correlation_id": request.correlation_id
            }
        )
        
        # Store transaction in database
        transaction_repo = repository_manager.get_transaction_repository()
        
        transaction_data = {
            **request.transaction.dict(),
            "status": "pending",
            "processing_time_ms": 0  # Will be updated later
        }
        
        db_transaction = transaction_repo.create(db, obj_in=transaction_data)
        
        # Run fraud detection
        logger.info(f"Running fraud detection for transaction {db_transaction.id}")
        
        prediction_result = fraud_detection_pipeline.predict_fraud(
            transaction_data=request.transaction.dict(),
            return_explanation=request.return_explanation
        )
        
        # Calculate processing time
        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        # Store prediction in database
        prediction_repo = repository_manager.get_prediction_repository()
        
        prediction_data = {
            "transaction_id": db_transaction.id,
            "fraud_probability": prediction_result["fraud_probability"],
            "risk_level": prediction_result["risk_level"],
            "is_fraud": prediction_result["is_fraud"],
            "model_name": "ensemble",
            "model_version": prediction_result.get("model_version", "1.0"),
            "confidence_score": prediction_result.get("confidence_score", 0.5),
            "prediction_latency_ms": int(processing_time),
            "fraud_threshold": prediction_result["thresholds_used"]["fraud"],
            "high_risk_threshold": prediction_result["thresholds_used"].get("high_risk", 0.7),
            "medium_risk_threshold": prediction_result["thresholds_used"].get("medium_risk", 0.3),
            "correlation_id": request.correlation_id,
            "api_version": "v1"
        }
        
        # Add explanations if available
        if request.return_explanation:
            prediction_data.update({
                "feature_importance": prediction_result.get("feature_importance"),
                "top_risk_factors": [
                    {"feature": rf["feature"], "importance": rf["importance"]}
                    for rf in prediction_result.get("top_risk_factors", [])
                ],
                "shap_values": prediction_result.get("shap_values")
            })
        
        db_prediction = prediction_repo.create(db, obj_in=prediction_data)
        
        # Update transaction status and processing time
        transaction_repo.update(
            db,
            db_obj=db_transaction,
            obj_in={
                "status": "flagged" if prediction_result["is_fraud"] else "approved",
                "processing_time_ms": int(processing_time)
            }
        )
        
        # Prepare risk factors for response
        risk_factors = []
        if request.return_explanation and prediction_result.get("top_risk_factors"):
            risk_factors = [
                RiskFactor(
                    feature=rf["feature"],
                    importance=rf["importance"],
                    value=None  # Could be populated with actual feature values
                )
                for rf in prediction_result["top_risk_factors"]
            ]
        
        # Log performance metrics in background
        background_tasks.add_task(
            performance_logger.log_performance_metric,
            operation="fraud_detection",
            duration_seconds=processing_time / 1000,
            record_count=1,
            additional_data={
                "fraud_probability": prediction_result["fraud_probability"],
                "risk_level": prediction_result["risk_level"],
                "model_version": prediction_result.get("model_version")
            }
        )
        
        # Prepare response
        response = FraudDetectionResponse(
            transaction_id=db_transaction.id,
            fraud_probability=prediction_result["fraud_probability"],
            is_fraud=prediction_result["is_fraud"],
            risk_level=RiskLevel(prediction_result["risk_level"]),
            confidence_score=prediction_result.get("confidence_score", 0.5),
            model_name="ensemble",
            model_version=prediction_result.get("model_version", "1.0"),
            prediction_latency_ms=processing_time,
            fraud_threshold=prediction_result["thresholds_used"]["fraud"],
            thresholds=prediction_result["thresholds_used"],
            top_risk_factors=risk_factors if request.return_explanation else None,
            feature_importance=prediction_result.get("feature_importance") if request.return_explanation else None,
            shap_values=prediction_result.get("shap_values") if request.return_explanation else None,
            correlation_id=request.correlation_id,
            message="Fraud detection completed successfully"
        )
        
        logger.info(
            f"Fraud detection completed for transaction {db_transaction.id}: "
            f"probability={prediction_result['fraud_probability']:.3f}, "
            f"risk={prediction_result['risk_level']}, "
            f"processing_time={processing_time:.1f}ms"
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Fraud detection failed: {str(e)}", exc_info=True)
        
        # Log the error
        audit_logger.log_api_access(
            user_id=str(current_user.user_id),
            endpoint="/api/v1/detect",
            method="POST",
            success=False,
            error_message=str(e)
        )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Fraud detection failed: {str(e)}"
        )


@router.post(
    "/feedback",
    response_model=FeedbackResponse,
    summary="Provide feedback on fraud prediction",
    description="Submit feedback to improve model accuracy"
)
async def submit_feedback(
    feedback: FeedbackCreate,
    current_user: User = Depends(require_authentication),
    db: Session = Depends(get_db)
) -> FeedbackResponse:
    """
    Submit feedback on a fraud prediction to improve model performance.
    
    This endpoint allows users to provide ground truth labels for predictions,
    which can be used to retrain and improve the ML models.
    """
    try:
        # Verify transaction exists
        transaction_repo = repository_manager.get_transaction_repository()
        transaction = transaction_repo.get(db, feedback.transaction_id)
        
        if not transaction:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Transaction {feedback.transaction_id} not found"
            )
        
        # Check if feedback already exists for this transaction
        feedback_repo = repository_manager.get_feedback_repository()
        existing_feedback = feedback_repo.get_by_transaction(db, feedback.transaction_id)
        
        if existing_feedback:
            logger.warning(f"Feedback already exists for transaction {feedback.transaction_id}")
        
        # Create feedback record
        feedback_data = {
            **feedback.dict(),
            "user_id": current_user.user_id,
            "review_status": "pending"
        }
        
        db_feedback = feedback_repo.create(db, obj_in=feedback_data)
        
        # Log audit event
        audit_logger.log_api_access(
            user_id=str(current_user.user_id),
            endpoint="/api/v1/feedback",
            method="POST",
            success=True,
            additional_data={
                "transaction_id": str(feedback.transaction_id),
                "feedback_type": feedback.feedback_type.value,
                "is_fraud_actual": feedback.is_fraud_actual
            }
        )
        
        logger.info(f"Feedback submitted for transaction {feedback.transaction_id} by user {current_user.user_id}")
        
        return FeedbackResponse(
            feedback_id=db_feedback.id,
            transaction_id=db_feedback.transaction_id,
            feedback_type=FeedbackType(db_feedback.feedback_type),
            is_fraud_actual=db_feedback.is_fraud_actual,
            review_status=db_feedback.review_status,
            created_at=db_feedback.created_at,
            message="Feedback submitted successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to submit feedback: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to submit feedback: {str(e)}"
        )


@router.get(
    "/predictions/{transaction_id}",
    response_model=List[Dict[str, Any]],
    summary="Get predictions for a transaction",
    description="Retrieve all predictions made for a specific transaction"
)
async def get_predictions(
    transaction_id: UUID,
    current_user: User = Depends(require_authentication),
    db: Session = Depends(get_db)
) -> List[Dict[str, Any]]:
    """Get all predictions for a specific transaction."""
    try:
        # Verify transaction exists
        transaction_repo = repository_manager.get_transaction_repository()
        transaction = transaction_repo.get(db, transaction_id)
        
        if not transaction:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Transaction {transaction_id} not found"
            )
        
        # Get predictions
        prediction_repo = repository_manager.get_prediction_repository()
        predictions = prediction_repo.get_by_transaction(db, transaction_id)
        
        # Format response
        response = []
        for pred in predictions:
            response.append({
                "prediction_id": pred.id,
                "fraud_probability": pred.fraud_probability,
                "is_fraud": pred.is_fraud,
                "risk_level": pred.risk_level,
                "model_name": pred.model_name,
                "model_version": pred.model_version,
                "confidence_score": pred.confidence_score,
                "prediction_latency_ms": pred.prediction_latency_ms,
                "created_at": pred.created_at,
                "top_risk_factors": pred.top_risk_factors or []
            })
        
        logger.info(f"Retrieved {len(predictions)} predictions for transaction {transaction_id}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get predictions: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve predictions: {str(e)}"
        )


@router.get(
    "/transactions/{user_id}/recent",
    response_model=List[Dict[str, Any]],
    summary="Get recent transactions for a user",
    description="Retrieve recent transactions and their fraud analysis"
)
async def get_recent_transactions(
    user_id: UUID,
    limit: int = 10,
    current_user: User = Depends(require_authentication),
    db: Session = Depends(get_db)
) -> List[Dict[str, Any]]:
    """Get recent transactions for a user with fraud analysis."""
    try:
        # Get recent transactions
        transaction_repo = repository_manager.get_transaction_repository()
        transactions = transaction_repo.get_recent_by_user(db, user_id, hours=72, limit=limit)
        
        if not transactions:
            return []
        
        # Get predictions for these transactions
        prediction_repo = repository_manager.get_prediction_repository()
        response = []
        
        for transaction in transactions:
            # Get latest prediction
            latest_prediction = prediction_repo.get_latest_by_transaction(db, transaction.id)
            
            transaction_data = {
                "transaction_id": transaction.id,
                "amount": transaction.amount,
                "currency": transaction.currency,
                "transaction_type": transaction.transaction_type,
                "merchant_name": transaction.merchant_name,
                "transaction_time": transaction.transaction_time,
                "status": transaction.status,
                "fraud_analysis": None
            }
            
            if latest_prediction:
                transaction_data["fraud_analysis"] = {
                    "fraud_probability": latest_prediction.fraud_probability,
                    "is_fraud": latest_prediction.is_fraud,
                    "risk_level": latest_prediction.risk_level,
                    "confidence_score": latest_prediction.confidence_score,
                    "model_version": latest_prediction.model_version
                }
            
            response.append(transaction_data)
        
        logger.info(f"Retrieved {len(transactions)} recent transactions for user {user_id}")
        return response
        
    except Exception as e:
        logger.error(f"Failed to get recent transactions: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve recent transactions: {str(e)}"
        )


@router.post(
    "/batch-detect",
    response_model=List[FraudDetectionResponse],
    summary="Batch fraud detection",
    description="Analyze multiple transactions for fraud in a single request"
)
async def batch_detect_fraud(
    requests: List[FraudDetectionRequest],
    background_tasks: BackgroundTasks,
    current_user: User = Depends(require_scopes(["batch_operations"])),
    db: Session = Depends(get_db)
) -> List[FraudDetectionResponse]:
    """
    Batch fraud detection for multiple transactions.
    
    This endpoint is optimized for processing multiple transactions efficiently.
    Requires 'batch_operations' scope.
    """
    if len(requests) > 100:  # Configurable limit
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Batch size exceeds maximum limit of 100 transactions"
        )
    
    try:
        results = []
        start_time = datetime.utcnow()
        
        for request in requests:
            try:
                # Process each transaction individually
                # In production, this could be optimized for true batch processing
                result = await detect_fraud(request, background_tasks, current_user, db)
                results.append(result)
                
            except Exception as e:
                # Continue processing other transactions even if one fails
                logger.error(f"Failed to process transaction in batch: {str(e)}")
                # Add error result
                error_result = FraudDetectionResponse(
                    transaction_id=uuid4(),  # Placeholder
                    fraud_probability=0.0,
                    is_fraud=False,
                    risk_level=RiskLevel.LOW,
                    confidence_score=0.0,
                    model_name="ensemble",
                    prediction_latency_ms=0,
                    fraud_threshold=0.5,
                    thresholds={},
                    success=False,
                    message=f"Processing failed: {str(e)}"
                )
                results.append(error_result)
        
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        # Log batch processing metrics
        background_tasks.add_task(
            performance_logger.log_performance_metric,
            operation="batch_fraud_detection",
            duration_seconds=processing_time,
            record_count=len(requests),
            additional_data={
                "batch_size": len(requests),
                "successful_predictions": len([r for r in results if r.success]),
                "failed_predictions": len([r for r in results if not r.success])
            }
        )
        
        logger.info(f"Batch fraud detection completed: {len(results)}/{len(requests)} transactions processed")
        return results
        
    except Exception as e:
        logger.error(f"Batch fraud detection failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch fraud detection failed: {str(e)}"
        )