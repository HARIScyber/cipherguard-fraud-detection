"""
Comment service for business logic and database operations.
Handles comment analysis, storage, and retrieval with caching.
"""

import time
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import func, desc, and_
import logging

from ..models import Comment, AnalyticsCache
from ..schemas import SentimentAnalytics, TimeSeriesAnalytics, TimeSeriesPoint
from .sentiment_analyzer import SentimentAnalyzer

logger = logging.getLogger(__name__)


class CommentService:
    """
    Service class for comment-related business logic and database operations.
    """
    
    def __init__(self, sentiment_analyzer: SentimentAnalyzer):
        """
        Initialize the comment service.
        
        Args:
            sentiment_analyzer: Sentiment analysis service instance
        """
        self.sentiment_analyzer = sentiment_analyzer
    
    async def analyze_and_store_comment(
        self, 
        db: Session, 
        user_id: str, 
        comment_text: str
    ) -> Comment:
        """
        Analyze sentiment of a comment and store it in the database.
        
        Args:
            db: Database session
            user_id: User identifier
            comment_text: Comment text to analyze
            
        Returns:
            Stored comment with sentiment analysis
        """
        try:
            start_time = time.time()
            
            # Analyze sentiment
            sentiment, confidence_score = await self.sentiment_analyzer.predict_sentiment(comment_text)
            
            processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            
            # Create comment record
            comment = Comment(
                user_id=user_id,
                comment_text=comment_text,
                sentiment=sentiment,
                confidence_score=confidence_score,
                text_length=len(comment_text),
                processing_time_ms=processing_time,
                model_version=self.sentiment_analyzer.model_version
            )
            
            # Save to database
            db.add(comment)
            db.commit()
            db.refresh(comment)
            
            logger.info(f"Comment {comment.id} analyzed and stored: {sentiment} ({confidence_score:.3f})")
            
            return comment
        
        except Exception as e:
            logger.error(f"Failed to analyze and store comment: {e}")
            db.rollback()
            raise
    
    def get_comments_paginated(
        self,
        db: Session,
        page: int = 1,
        page_size: int = 10,
        sentiment_filter: Optional[str] = None,
        user_id_filter: Optional[str] = None,
        search_text: Optional[str] = None
    ) -> Tuple[List[Comment], int]:
        """
        Get paginated comments with optional filters.
        
        Args:
            db: Database session
            page: Page number (1-based)
            page_size: Number of comments per page
            sentiment_filter: Filter by sentiment type
            user_id_filter: Filter by user ID
            search_text: Search in comment text
            
        Returns:
            Tuple of (comments_list, total_count)
        """
        try:
            # Build query
            query = db.query(Comment)
            
            # Apply filters
            if sentiment_filter:
                query = query.filter(Comment.sentiment == sentiment_filter)
            
            if user_id_filter:
                query = query.filter(Comment.user_id == user_id_filter)
            
            if search_text:
                query = query.filter(Comment.comment_text.ilike(f"%{search_text}%"))
            
            # Get total count
            total_count = query.count()
            
            # Apply pagination and ordering
            comments = (
                query
                .order_by(desc(Comment.created_at))
                .offset((page - 1) * page_size)
                .limit(page_size)
                .all()
            )
            
            return comments, total_count
        
        except Exception as e:
            logger.error(f"Failed to get paginated comments: {e}")
            raise
    
    def get_sentiment_analytics(
        self,
        db: Session,
        days: int = 30,
        use_cache: bool = True
    ) -> SentimentAnalytics:
        """
        Get sentiment analytics for the specified number of days.
        
        Args:
            db: Database session
            days: Number of days to analyze
            use_cache: Whether to use cached results
            
        Returns:
            Sentiment analytics data
        """
        try:
            cache_key = f"sentiment_analytics_{days}d"
            
            # Check cache first
            if use_cache:
                cached_result = self._get_cached_analytics(db, cache_key)
                if cached_result:
                    return cached_result
            
            # Calculate date range
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=days)
            
            # Query sentiment counts
            sentiment_counts = (
                db.query(
                    Comment.sentiment,
                    func.count(Comment.id).label('count'),
                    func.avg(Comment.confidence_score).label('avg_confidence')
                )
                .filter(Comment.created_at >= start_date)
                .group_by(Comment.sentiment)
                .all()
            )
            
            # Process results
            total_comments = sum(row.count for row in sentiment_counts)
            positive_count = 0
            negative_count = 0
            neutral_count = 0
            total_confidence = 0.0
            
            for row in sentiment_counts:
                if row.sentiment == 'positive':
                    positive_count = row.count
                elif row.sentiment == 'negative':
                    negative_count = row.count
                elif row.sentiment == 'neutral':
                    neutral_count = row.count
                
                total_confidence += row.avg_confidence * row.count
            
            # Calculate percentages
            if total_comments > 0:
                positive_percentage = (positive_count / total_comments) * 100
                negative_percentage = (negative_count / total_comments) * 100
                neutral_percentage = (neutral_count / total_comments) * 100
                avg_confidence_score = total_confidence / total_comments
            else:
                positive_percentage = negative_percentage = neutral_percentage = 0.0
                avg_confidence_score = 0.0
            
            # Create analytics object
            analytics = SentimentAnalytics(
                total_comments=total_comments,
                positive_count=positive_count,
                negative_count=negative_count,
                neutral_count=neutral_count,
                positive_percentage=round(positive_percentage, 2),
                negative_percentage=round(negative_percentage, 2),
                neutral_percentage=round(neutral_percentage, 2),
                avg_confidence_score=round(avg_confidence_score, 3)
            )
            
            # Cache the result
            if use_cache:
                self._cache_analytics(db, cache_key, analytics, expire_minutes=15)
            
            return analytics
        
        except Exception as e:
            logger.error(f"Failed to get sentiment analytics: {e}")
            raise
    
    def get_time_series_analytics(
        self,
        db: Session,
        days: int = 7,
        interval_hours: int = 24
    ) -> TimeSeriesAnalytics:
        """
        Get time-series sentiment analytics.
        
        Args:
            db: Database session
            days: Number of days to analyze
            interval_hours: Time interval in hours
            
        Returns:
            Time-series analytics data
        """
        try:
            # Calculate date range
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=days)
            
            # Generate time intervals
            intervals = []
            current_time = start_date
            interval_delta = timedelta(hours=interval_hours)
            
            while current_time < end_date:
                interval_end = min(current_time + interval_delta, end_date)
                intervals.append((current_time, interval_end))
                current_time = interval_end
            
            # Get data for each interval
            data_points = []
            
            for interval_start, interval_end in intervals:
                # Query sentiment counts for this interval
                sentiment_counts = (
                    db.query(
                        Comment.sentiment,
                        func.count(Comment.id).label('count')
                    )
                    .filter(and_(
                        Comment.created_at >= interval_start,
                        Comment.created_at < interval_end
                    ))
                    .group_by(Comment.sentiment)
                    .all()
                )
                
                # Process counts
                positive_count = 0
                negative_count = 0
                neutral_count = 0
                
                for row in sentiment_counts:
                    if row.sentiment == 'positive':
                        positive_count = row.count
                    elif row.sentiment == 'negative':
                        negative_count = row.count
                    elif row.sentiment == 'neutral':
                        neutral_count = row.count
                
                total_count = positive_count + negative_count + neutral_count
                
                # Create data point
                data_point = TimeSeriesPoint(
                    timestamp=interval_start,
                    positive_count=positive_count,
                    negative_count=negative_count,
                    neutral_count=neutral_count,
                    total_count=total_count
                )
                
                data_points.append(data_point)
            
            # Determine interval label
            if interval_hours == 1:
                interval_label = "hourly"
            elif interval_hours == 24:
                interval_label = "daily"
            else:
                interval_label = f"{interval_hours}h"
            
            return TimeSeriesAnalytics(
                data_points=data_points,
                interval=interval_label,
                start_date=start_date,
                end_date=end_date
            )
        
        except Exception as e:
            logger.error(f"Failed to get time-series analytics: {e}")
            raise
    
    def get_recent_negative_comments(
        self,
        db: Session,
        limit: int = 10,
        hours: int = 24
    ) -> List[Comment]:
        """
        Get recent negative comments for monitoring.
        
        Args:
            db: Database session
            limit: Maximum number of comments to return
            hours: Number of hours to look back
            
        Returns:
            List of recent negative comments
        """
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)
            
            comments = (
                db.query(Comment)
                .filter(and_(
                    Comment.sentiment == 'negative',
                    Comment.created_at >= cutoff_time
                ))
                .order_by(desc(Comment.created_at))
                .limit(limit)
                .all()
            )
            
            return comments
        
        except Exception as e:
            logger.error(f"Failed to get recent negative comments: {e}")
            raise
    
    def _get_cached_analytics(self, db: Session, cache_key: str) -> Optional[SentimentAnalytics]:
        """Get analytics from cache if available and not expired."""
        try:
            cache_entry = (
                db.query(AnalyticsCache)
                .filter(and_(
                    AnalyticsCache.cache_key == cache_key,
                    AnalyticsCache.expires_at > datetime.utcnow()
                ))
                .first()
            )
            
            if cache_entry:
                import json
                data = json.loads(cache_entry.cache_value)
                return SentimentAnalytics(**data)
            
            return None
        
        except Exception as e:
            logger.error(f"Failed to get cached analytics: {e}")
            return None
    
    def _cache_analytics(
        self,
        db: Session,
        cache_key: str,
        analytics: SentimentAnalytics,
        expire_minutes: int = 15
    ) -> None:
        """Cache analytics data."""
        try:
            import json
            
            # Delete existing cache entry
            db.query(AnalyticsCache).filter(AnalyticsCache.cache_key == cache_key).delete()
            
            # Create new cache entry
            cache_entry = AnalyticsCache(
                cache_key=cache_key,
                cache_value=analytics.json(),
                expires_at=datetime.utcnow() + timedelta(minutes=expire_minutes)
            )
            
            db.add(cache_entry)
            db.commit()
            
        except Exception as e:
            logger.error(f"Failed to cache analytics: {e}")
            # Don't raise - caching failure shouldn't break the request