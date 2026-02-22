"""
Comment analysis routes for sentiment analysis operations.
"""

from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session

from ..database import get_db
from ..schemas import (
    CommentAnalysisRequest,
    CommentAnalysisResponse,
    CommentResponse,
    SentimentAnalytics,
    TimeSeriesAnalytics,
    APIResponse,
    PaginatedResponse
)
from ..auth import get_current_user
from ..models import User
from ..services.comment_service import CommentService
from ..services.sentiment_analyzer import SentimentAnalyzer

# Router
router = APIRouter(prefix="/comments", tags=["Comment Analysis"])


def get_comment_service():
    """Dependency to get comment service instance."""
    sentiment_analyzer = SentimentAnalyzer()
    return CommentService(sentiment_analyzer)


@router.post(
    "/analyze",
    response_model=CommentAnalysisResponse,
    summary="Analyze Comment Sentiment",
    description="Analyze the sentiment of a comment and store the result"
)
async def analyze_comment(
    request: CommentAnalysisRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    comment_service: CommentService = Depends(get_comment_service)
):
    """
    Analyze the sentiment of a comment text.
    
    **Authentication:** Bearer token required
    
    **Parameters:**
    - **comment_text**: The comment text to analyze (max 5000 characters)
    - **store_result**: Whether to store the analysis result in database (default: true)
    
    **Returns:**
    - **sentiment**: Detected sentiment (positive, negative, neutral)
    - **confidence_score**: Confidence level (0.0 - 1.0)
    - **processing_time_ms**: Processing time in milliseconds
    - **comment_id**: Database ID if stored
    
    **Example Request:**
    ```json
    {
        "comment_text": "This product is absolutely amazing! I love it!",
        "store_result": true
    }
    ```
    
    **Example Response:**
    ```json
    {
        "id": 123,
        "comment_text": "This product is absolutely amazing! I love it!",
        "sentiment": "positive",
        "confidence_score": 0.94,
        "processing_time_ms": 15.2,
        "user_id": "user123",
        "created_at": "2024-01-15T10:30:00Z"
    }
    ```
    """
    try:
        # Validate comment length
        if len(request.comment_text.strip()) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Comment text cannot be empty"
            )
        
        if len(request.comment_text) > 5000:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Comment text too long (maximum 5000 characters)"
            )
        
        # Analyze and store comment
        comment = await comment_service.analyze_and_store_comment(
            db=db,
            user_id=str(current_user.id),
            comment_text=request.comment_text.strip()
        )
        
        # Convert to response model
        response = CommentAnalysisResponse(
            id=comment.id,
            comment_text=comment.comment_text,
            sentiment=comment.sentiment,
            confidence_score=comment.confidence_score,
            processing_time_ms=comment.processing_time_ms,
            user_id=comment.user_id,
            created_at=comment.created_at
        )
        
        return response
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to analyze comment: {str(e)}"
        )


@router.get(
    "/",
    response_model=PaginatedResponse[CommentResponse],
    summary="Get Comments",
    description="Get paginated list of analyzed comments with optional filters"
)
async def get_comments(
    page: int = Query(1, ge=1, description="Page number (1-based)"),
    page_size: int = Query(10, ge=1, le=100, description="Items per page (max 100)"),
    sentiment: Optional[str] = Query(None, regex="^(positive|negative|neutral)$", description="Filter by sentiment"),
    user_id: Optional[str] = Query(None, description="Filter by user ID"),
    search: Optional[str] = Query(None, description="Search in comment text"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    comment_service: CommentService = Depends(get_comment_service)
):
    """
    Get paginated list of analyzed comments.
    
    **Authentication:** Bearer token required
    
    **Query Parameters:**
    - **page**: Page number (default: 1)
    - **page_size**: Items per page (default: 10, max: 100)
    - **sentiment**: Filter by sentiment (positive, negative, neutral)
    - **user_id**: Filter by user ID
    - **search**: Search text in comments
    
    **Returns:**
    - **items**: List of comments
    - **total**: Total number of comments
    - **page**: Current page
    - **page_size**: Items per page
    - **total_pages**: Total number of pages
    
    **Example Response:**
    ```json
    {
        "items": [
            {
                "id": 123,
                "comment_text": "Great product!",
                "sentiment": "positive",
                "confidence_score": 0.92,
                "user_id": "user123",
                "created_at": "2024-01-15T10:30:00Z"
            }
        ],
        "total": 250,
        "page": 1,
        "page_size": 10,
        "total_pages": 25
    }
    ```
    """
    try:
        # Get comments with filters
        comments, total_count = comment_service.get_comments_paginated(
            db=db,
            page=page,
            page_size=page_size,
            sentiment_filter=sentiment,
            user_id_filter=user_id,
            search_text=search
        )
        
        # Convert to response models
        comment_responses = []
        for comment in comments:
            comment_response = CommentResponse(
                id=comment.id,
                comment_text=comment.comment_text,
                sentiment=comment.sentiment,
                confidence_score=comment.confidence_score,
                text_length=comment.text_length,
                processing_time_ms=comment.processing_time_ms,
                user_id=comment.user_id,
                model_version=comment.model_version,
                created_at=comment.created_at
            )
            comment_responses.append(comment_response)
        
        # Calculate pagination
        total_pages = (total_count + page_size - 1) // page_size
        
        return PaginatedResponse(
            items=comment_responses,
            total=total_count,
            page=page,
            page_size=page_size,
            total_pages=total_pages
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch comments: {str(e)}"
        )


@router.get(
    "/{comment_id}",
    response_model=APIResponse[CommentResponse],
    summary="Get Comment by ID",
    description="Get a specific comment by its ID"
)
async def get_comment(
    comment_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get a specific comment by ID.
    
    **Authentication:** Bearer token required
    
    **Parameters:**
    - **comment_id**: The comment ID
    
    **Returns:**
    - **Comment details**
    
    **Example Response:**
    ```json
    {
        "success": true,
        "data": {
            "id": 123,
            "comment_text": "Great product!",
            "sentiment": "positive",
            "confidence_score": 0.92,
            "user_id": "user123",
            "created_at": "2024-01-15T10:30:00Z"
        },
        "message": "Comment retrieved successfully"
    }
    ```
    """
    try:
        from ..models import Comment
        
        comment = db.query(Comment).filter(Comment.id == comment_id).first()
        
        if not comment:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Comment not found"
            )
        
        comment_response = CommentResponse(
            id=comment.id,
            comment_text=comment.comment_text,
            sentiment=comment.sentiment,
            confidence_score=comment.confidence_score,
            text_length=comment.text_length,
            processing_time_ms=comment.processing_time_ms,
            user_id=comment.user_id,
            model_version=comment.model_version,
            created_at=comment.created_at
        )
        
        return APIResponse(
            success=True,
            data=comment_response,
            message="Comment retrieved successfully"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch comment: {str(e)}"
        )


@router.get(
    "/analytics/sentiment",
    response_model=APIResponse[SentimentAnalytics],
    summary="Get Sentiment Analytics",
    description="Get sentiment analytics for the specified time period"
)
async def get_sentiment_analytics(
    days: int = Query(30, ge=1, le=365, description="Number of days to analyze (1-365)"),
    use_cache: bool = Query(True, description="Use cached results if available"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    comment_service: CommentService = Depends(get_comment_service)
):
    """
    Get sentiment analytics for the specified time period.
    
    **Authentication:** Bearer token required
    
    **Query Parameters:**
    - **days**: Number of days to analyze (default: 30, max: 365)
    - **use_cache**: Use cached results if available (default: true)
    
    **Returns:**
    - **total_comments**: Total number of comments analyzed
    - **positive_count**: Number of positive comments
    - **negative_count**: Number of negative comments
    - **neutral_count**: Number of neutral comments
    - **positive_percentage**: Percentage of positive comments
    - **negative_percentage**: Percentage of negative comments  
    - **neutral_percentage**: Percentage of neutral comments
    - **avg_confidence_score**: Average confidence score
    
    **Example Response:**
    ```json
    {
        "success": true,
        "data": {
            "total_comments": 1500,
            "positive_count": 850,
            "negative_count": 300,
            "neutral_count": 350,
            "positive_percentage": 56.67,
            "negative_percentage": 20.00,
            "neutral_percentage": 23.33,
            "avg_confidence_score": 0.847
        },
        "message": "Sentiment analytics retrieved successfully"
    }
    ```
    """
    try:
        analytics = comment_service.get_sentiment_analytics(
            db=db,
            days=days,
            use_cache=use_cache
        )
        
        return APIResponse(
            success=True,
            data=analytics,
            message="Sentiment analytics retrieved successfully"
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get sentiment analytics: {str(e)}"
        )


@router.get(
    "/analytics/timeseries",
    response_model=APIResponse[TimeSeriesAnalytics],
    summary="Get Time-Series Analytics",
    description="Get time-series sentiment analytics"
)
async def get_timeseries_analytics(
    days: int = Query(7, ge=1, le=90, description="Number of days to analyze (1-90)"),
    interval_hours: int = Query(24, ge=1, le=168, description="Time interval in hours (1-168)"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    comment_service: CommentService = Depends(get_comment_service)
):
    """
    Get time-series sentiment analytics.
    
    **Authentication:** Bearer token required
    
    **Query Parameters:**
    - **days**: Number of days to analyze (default: 7, max: 90)
    - **interval_hours**: Time interval in hours (default: 24, max: 168)
    
    **Returns:**
    - **data_points**: List of time-series data points
    - **interval**: Interval description
    - **start_date**: Analysis start date
    - **end_date**: Analysis end date
    
    **Example Response:**
    ```json
    {
        "success": true,
        "data": {
            "data_points": [
                {
                    "timestamp": "2024-01-15T00:00:00Z",
                    "positive_count": 45,
                    "negative_count": 12,
                    "neutral_count": 18,
                    "total_count": 75
                }
            ],
            "interval": "daily",
            "start_date": "2024-01-08T00:00:00Z",
            "end_date": "2024-01-15T00:00:00Z"
        },
        "message": "Time-series analytics retrieved successfully"
    }
    ```
    """
    try:
        analytics = comment_service.get_time_series_analytics(
            db=db,
            days=days,
            interval_hours=interval_hours
        )
        
        return APIResponse(
            success=True,
            data=analytics,
            message="Time-series analytics retrieved successfully"
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get time-series analytics: {str(e)}"
        )


@router.get(
    "/recent/negative",
    response_model=APIResponse[List[CommentResponse]],
    summary="Get Recent Negative Comments",
    description="Get recent negative comments for monitoring purposes"
)
async def get_recent_negative_comments(
    limit: int = Query(10, ge=1, le=50, description="Maximum number of comments (1-50)"),
    hours: int = Query(24, ge=1, le=168, description="Hours to look back (1-168)"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    comment_service: CommentService = Depends(get_comment_service)
):
    """
    Get recent negative comments for monitoring.
    
    **Authentication:** Bearer token required
    
    **Query Parameters:**
    - **limit**: Maximum number of comments (default: 10, max: 50)
    - **hours**: Hours to look back (default: 24, max: 168)
    
    **Returns:**
    - **List of recent negative comments**
    
    **Example Response:**
    ```json
    {
        "success": true,
        "data": [
            {
                "id": 456,
                "comment_text": "This service is terrible",
                "sentiment": "negative",
                "confidence_score": 0.89,
                "user_id": "user789",
                "created_at": "2024-01-15T09:45:00Z"
            }
        ],
        "message": "Recent negative comments retrieved successfully"
    }
    ```
    """
    try:
        comments = comment_service.get_recent_negative_comments(
            db=db,
            limit=limit,
            hours=hours
        )
        
        # Convert to response models
        comment_responses = []
        for comment in comments:
            comment_response = CommentResponse(
                id=comment.id,
                comment_text=comment.comment_text,
                sentiment=comment.sentiment,
                confidence_score=comment.confidence_score,
                text_length=comment.text_length,
                processing_time_ms=comment.processing_time_ms,
                user_id=comment.user_id,
                model_version=comment.model_version,
                created_at=comment.created_at
            )
            comment_responses.append(comment_response)
        
        return APIResponse(
            success=True,
            data=comment_responses,
            message="Recent negative comments retrieved successfully"
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get recent negative comments: {str(e)}"
        )