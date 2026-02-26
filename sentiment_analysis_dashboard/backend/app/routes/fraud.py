"""
Fraud Detection API Routes
Endpoints for fraud detection analysis and transaction management.
"""

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session
from sqlalchemy import func, desc
from typing import Optional, List
from datetime import datetime, timedelta
import time
import logging

from ..database import get_db
from ..models import FraudTransaction, CustomerAlert
from ..schemas import (
    FraudDetectionRequest, 
    FraudDetectionResponse, 
    FraudListResponse,
    FraudAnalytics,
    AlertResponse,
    AlertHistoryResponse,
    AlertStats,
    APIResponse
)
from ..services.fraud_detector import get_fraud_detector
from ..services.alert_service import get_alert_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/fraud", tags=["Fraud Detection"])


def generate_transaction_id() -> str:
    """Generate a unique transaction ID."""
    import uuid
    return f"txn_{uuid.uuid4().hex[:12]}"


@router.post("/detect", response_model=APIResponse[FraudDetectionResponse])
async def detect_fraud(
    request: FraudDetectionRequest,
    db: Session = Depends(get_db)
):
    """
    Analyze a transaction for potential fraud and send alerts if suspicious.
    
    - **amount**: Transaction amount (required)
    - **merchant**: Merchant name (required)
    - **device**: Device type - mobile, desktop, tablet (default: desktop)
    - **country**: Country code (default: US)
    - **customer_name**: Customer name for alerts (optional)
    - **customer_email**: Customer email for alerts (optional)
    - **customer_phone**: Customer phone for SMS alerts (optional)
    - **send_alert**: Whether to send alert if fraud detected (default: true)
    """
    try:
        start_time = time.time()
        
        # Get fraud detector
        detector = get_fraud_detector()
        if not detector.is_loaded():
            detector.load_model_sync()
        
        # Prepare transaction data
        transaction = {
            "amount": request.amount,
            "merchant": request.merchant,
            "device": request.device,
            "country": request.country
        }
        
        # Detect fraud
        is_fraud, fraud_score, risk_level = detector.detect_fraud(transaction)
        
        # Generate transaction ID
        transaction_id = generate_transaction_id()
        
        # Send alert if fraud detected and customer info provided
        alert_sent = False
        alert_channels = []
        alert_record = None
        
        if is_fraud and request.send_alert and (request.customer_email or request.customer_phone):
            try:
                alert_service = get_alert_service()
                
                customer_info = {
                    "name": request.customer_name or "Valued Customer",
                    "email": request.customer_email,
                    "phone": request.customer_phone,
                    "device_token": None  # Would come from mobile app
                }
                
                fraud_result = {
                    "transaction_id": transaction_id,
                    "fraud_score": fraud_score,
                    "risk_level": risk_level,
                    "is_fraud": is_fraud
                }
                
                alert_data = alert_service.send_alert(
                    customer_info=customer_info,
                    transaction=transaction,
                    fraud_result=fraud_result
                )
                
                alert_sent = True
                alert_channels = alert_data.get("channels_used", [])
                
                # Save alert to database
                alert_record = CustomerAlert(
                    alert_id=alert_data.get("alert_id"),
                    transaction_id=transaction_id,
                    customer_name=request.customer_name,
                    customer_email=request.customer_email,
                    customer_phone=request.customer_phone,
                    alert_type=alert_data.get("alert_type", "suspicious_transaction"),
                    channels_used=",".join(alert_channels),
                    sms_status=alert_data.get("results", {}).get("sms", {}).get("status", "skipped"),
                    email_status=alert_data.get("results", {}).get("email", {}).get("status", "skipped"),
                    push_status=alert_data.get("results", {}).get("push", {}).get("status", "skipped")
                )
                db.add(alert_record)
                
                logger.info(f"Alert sent for transaction {transaction_id} via {alert_channels}")
                
            except Exception as e:
                logger.error(f"Failed to send alert: {e}")
        
        # Store fraud transaction in database
        fraud_record = FraudTransaction(
            transaction_id=transaction_id,
            amount=request.amount,
            merchant=request.merchant,
            device=request.device,
            country=request.country,
            customer_name=request.customer_name,
            customer_email=request.customer_email,
            customer_phone=request.customer_phone,
            is_fraud=is_fraud,
            fraud_score=fraud_score,
            risk_level=risk_level,
            alert_sent=alert_sent,
            alert_channels=",".join(alert_channels) if alert_channels else None,
            model_version=detector.model_version
        )
        
        db.add(fraud_record)
        db.commit()
        db.refresh(fraud_record)
        
        processing_time = (time.time() - start_time) * 1000
        logger.info(f"Fraud detection completed: {transaction_id}, is_fraud={is_fraud}, score={fraud_score:.3f}, alert_sent={alert_sent}, time={processing_time:.2f}ms")
        
        return APIResponse(
            success=True,
            message="Fraud detection completed" + (" - Alert sent to customer!" if alert_sent else ""),
            data=FraudDetectionResponse(
                transaction_id=fraud_record.transaction_id,
                amount=fraud_record.amount,
                merchant=fraud_record.merchant,
                device=fraud_record.device,
                country=fraud_record.country,
                is_fraud=fraud_record.is_fraud,
                fraud_score=fraud_record.fraud_score,
                risk_level=fraud_record.risk_level,
                created_at=fraud_record.created_at,
                model_version=fraud_record.model_version,
                alert_sent=alert_sent,
                alert_channels=alert_channels if alert_channels else None
            )
        )
        
    except Exception as e:
        logger.error(f"Fraud detection error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Fraud detection failed: {str(e)}"
        )


@router.get("/transactions", response_model=APIResponse[FraudListResponse])
async def get_fraud_transactions(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(10, ge=1, le=100, description="Items per page"),
    is_fraud: Optional[bool] = Query(None, description="Filter by fraud status"),
    risk_level: Optional[str] = Query(None, description="Filter by risk level"),
    min_amount: Optional[float] = Query(None, ge=0, description="Minimum amount"),
    max_amount: Optional[float] = Query(None, ge=0, description="Maximum amount"),
    days: int = Query(30, ge=1, le=365, description="Days to look back"),
    db: Session = Depends(get_db)
):
    """
    Get paginated list of fraud transactions with optional filters.
    """
    try:
        # Base query
        query = db.query(FraudTransaction)
        
        # Apply filters
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        query = query.filter(FraudTransaction.created_at >= cutoff_date)
        
        if is_fraud is not None:
            query = query.filter(FraudTransaction.is_fraud == is_fraud)
        
        if risk_level:
            query = query.filter(FraudTransaction.risk_level == risk_level.upper())
        
        if min_amount is not None:
            query = query.filter(FraudTransaction.amount >= min_amount)
        
        if max_amount is not None:
            query = query.filter(FraudTransaction.amount <= max_amount)
        
        # Get total count
        total_count = query.count()
        total_pages = (total_count + page_size - 1) // page_size
        
        # Get paginated results
        transactions = query.order_by(desc(FraudTransaction.created_at)) \
            .offset((page - 1) * page_size) \
            .limit(page_size) \
            .all()
        
        return APIResponse(
            success=True,
            message=f"Retrieved {len(transactions)} transactions",
            data=FraudListResponse(
                transactions=[
                    FraudDetectionResponse(
                        transaction_id=t.transaction_id,
                        amount=t.amount,
                        merchant=t.merchant,
                        device=t.device,
                        country=t.country,
                        is_fraud=t.is_fraud,
                        fraud_score=t.fraud_score,
                        risk_level=t.risk_level,
                        created_at=t.created_at,
                        model_version=t.model_version
                    ) for t in transactions
                ],
                total_count=total_count,
                page=page,
                page_size=page_size,
                total_pages=total_pages
            )
        )
        
    except Exception as e:
        logger.error(f"Error fetching transactions: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/analytics", response_model=APIResponse[FraudAnalytics])
async def get_fraud_analytics(
    days: int = Query(30, ge=1, le=365, description="Days to analyze"),
    db: Session = Depends(get_db)
):
    """
    Get fraud analytics and statistics.
    """
    try:
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        # Get total counts
        total_query = db.query(FraudTransaction).filter(
            FraudTransaction.created_at >= cutoff_date
        )
        
        total_transactions = total_query.count()
        
        if total_transactions == 0:
            return APIResponse(
                success=True,
                message="No transactions found in the specified period",
                data=FraudAnalytics(
                    total_transactions=0,
                    fraud_count=0,
                    legitimate_count=0,
                    fraud_rate=0.0,
                    average_fraud_score=0.0,
                    risk_distribution={
                        "CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0, "VERY_LOW": 0
                    },
                    total_amount=0.0,
                    fraud_amount=0.0
                )
            )
        
        # Fraud counts
        fraud_count = total_query.filter(FraudTransaction.is_fraud == True).count()
        legitimate_count = total_transactions - fraud_count
        fraud_rate = (fraud_count / total_transactions) * 100
        
        # Average fraud score
        avg_score = db.query(func.avg(FraudTransaction.fraud_score)).filter(
            FraudTransaction.created_at >= cutoff_date
        ).scalar() or 0.0
        
        # Risk distribution
        risk_dist = {}
        for level in ["CRITICAL", "HIGH", "MEDIUM", "LOW", "VERY_LOW"]:
            count = total_query.filter(FraudTransaction.risk_level == level).count()
            risk_dist[level] = count
        
        # Amount statistics
        total_amount = db.query(func.sum(FraudTransaction.amount)).filter(
            FraudTransaction.created_at >= cutoff_date
        ).scalar() or 0.0
        
        fraud_amount = db.query(func.sum(FraudTransaction.amount)).filter(
            FraudTransaction.created_at >= cutoff_date,
            FraudTransaction.is_fraud == True
        ).scalar() or 0.0
        
        return APIResponse(
            success=True,
            message="Analytics retrieved successfully",
            data=FraudAnalytics(
                total_transactions=total_transactions,
                fraud_count=fraud_count,
                legitimate_count=legitimate_count,
                fraud_rate=round(fraud_rate, 2),
                average_fraud_score=round(float(avg_score), 3),
                risk_distribution=risk_dist,
                total_amount=round(float(total_amount), 2),
                fraud_amount=round(float(fraud_amount), 2)
            )
        )
        
    except Exception as e:
        logger.error(f"Error calculating analytics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/recent", response_model=APIResponse[List[FraudDetectionResponse]])
async def get_recent_frauds(
    limit: int = Query(10, ge=1, le=50, description="Number of recent frauds"),
    hours: int = Query(24, ge=1, le=168, description="Hours to look back"),
    db: Session = Depends(get_db)
):
    """
    Get recent fraudulent transactions.
    """
    try:
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        transactions = db.query(FraudTransaction).filter(
            FraudTransaction.created_at >= cutoff_time,
            FraudTransaction.is_fraud == True
        ).order_by(desc(FraudTransaction.created_at)).limit(limit).all()
        
        return APIResponse(
            success=True,
            message=f"Retrieved {len(transactions)} recent fraud transactions",
            data=[
                FraudDetectionResponse(
                    transaction_id=t.transaction_id,
                    amount=t.amount,
                    merchant=t.merchant,
                    device=t.device,
                    country=t.country,
                    is_fraud=t.is_fraud,
                    fraud_score=t.fraud_score,
                    risk_level=t.risk_level,
                    created_at=t.created_at,
                    model_version=t.model_version
                ) for t in transactions
            ]
        )
        
    except Exception as e:
        logger.error(f"Error fetching recent frauds: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


# ======================= ALERT ENDPOINTS =======================

@router.get("/alerts", response_model=APIResponse[AlertHistoryResponse])
async def get_alert_history(
    limit: int = Query(50, ge=1, le=200, description="Number of alerts to retrieve"),
    days: int = Query(30, ge=1, le=365, description="Days to look back"),
    db: Session = Depends(get_db)
):
    """
    Get customer alert history.
    """
    try:
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        alerts = db.query(CustomerAlert).filter(
            CustomerAlert.created_at >= cutoff_date
        ).order_by(desc(CustomerAlert.created_at)).limit(limit).all()
        
        return APIResponse(
            success=True,
            message=f"Retrieved {len(alerts)} alerts",
            data=AlertHistoryResponse(
                alerts=[
                    AlertResponse(
                        alert_id=a.alert_id,
                        transaction_id=a.transaction_id,
                        customer_email=a.customer_email,
                        customer_phone=a.customer_phone,
                        alert_type=a.alert_type,
                        channels_used=a.channels_used.split(",") if a.channels_used else [],
                        timestamp=a.created_at
                    ) for a in alerts
                ],
                total_count=len(alerts)
            )
        )
        
    except Exception as e:
        logger.error(f"Error fetching alert history: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/alerts/stats", response_model=APIResponse[AlertStats])
async def get_alert_stats(
    days: int = Query(30, ge=1, le=365, description="Days to analyze"),
    db: Session = Depends(get_db)
):
    """
    Get alert statistics.
    """
    try:
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        alerts = db.query(CustomerAlert).filter(
            CustomerAlert.created_at >= cutoff_date
        ).all()
        
        total_alerts = len(alerts)
        
        if total_alerts == 0:
            return APIResponse(
                success=True,
                message="No alerts in the specified period",
                data=AlertStats(
                    total_alerts=0,
                    by_type={},
                    by_channel={"sms": 0, "email": 0, "push": 0},
                    success_rate=0.0
                )
            )
        
        # Count by type
        by_type = {}
        by_channel = {"sms": 0, "email": 0, "push": 0}
        successful = 0
        
        for alert in alerts:
            # Count by type
            by_type[alert.alert_type] = by_type.get(alert.alert_type, 0) + 1
            
            # Count by channel
            if alert.sms_status == "sent":
                by_channel["sms"] += 1
                successful += 1
            if alert.email_status == "sent":
                by_channel["email"] += 1
                successful += 1
            if alert.push_status == "sent":
                by_channel["push"] += 1
                successful += 1
        
        success_rate = (successful / (total_alerts * 3)) * 100 if total_alerts > 0 else 0
        
        return APIResponse(
            success=True,
            message="Alert statistics retrieved",
            data=AlertStats(
                total_alerts=total_alerts,
                by_type=by_type,
                by_channel=by_channel,
                success_rate=round(success_rate, 2)
            )
        )
        
    except Exception as e:
        logger.error(f"Error calculating alert stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
