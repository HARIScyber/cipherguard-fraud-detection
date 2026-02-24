"""
Sentiment Analysis Service for Comment Classification.
Uses TF-IDF vectorization with Logistic Regression for sentiment prediction.
"""

import pickle
import os
import logging
import time
from typing import Tuple, Dict, Any
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import joblib
import re

logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """
    Sentiment analysis service using TF-IDF and Logistic Regression.
    Provides training, prediction, and model management capabilities.
    """
    
    def __init__(self, model_path: str = None):
        """
        Initialize the sentiment analyzer.
        
        Args:
            model_path: Path to the trained model file
        """
        self.model_path = model_path or os.path.join(
            os.path.dirname(__file__), "../ml/sentiment_model.pkl"
        )
        self.model = None
        self.vectorizer = None
        self.pipeline = None
        self.model_version = "1.0"
        self.is_loaded = False
        
        # Text preprocessing regex patterns
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.mention_pattern = re.compile(r'@\w+')
        self.hashtag_pattern = re.compile(r'#\w+')
        
    async def load_model(self) -> bool:
        """
        Load the trained sentiment analysis model (async version).
        
        Returns:
            True if model loaded successfully, False otherwise
        """
        return self.load_model_sync()
    
    def load_model_sync(self) -> bool:
        """
        Load the trained sentiment analysis model (synchronous version).
        
        Returns:
            True if model loaded successfully, False otherwise
        """
        try:
            if os.path.exists(self.model_path):
                logger.info(f"Loading sentiment model from {self.model_path}")
                
                with open(self.model_path, 'rb') as f:
                    model_data = pickle.load(f)
                
                self.pipeline = model_data['pipeline']
                self.model_version = model_data.get('version', '1.0')
                self.is_loaded = True
                
                logger.info(f"Sentiment model v{self.model_version} loaded successfully")
                return True
            else:
                logger.warning(f"Model file not found at {self.model_path}, creating new model")
                self.train_default_model_sync()
                return True
                
        except Exception as e:
            logger.error(f"Failed to load sentiment model: {e}")
            self.is_loaded = False
            return False
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for sentiment analysis.
        
        Args:
            text: Raw text to preprocess
            
        Returns:
            Preprocessed text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = self.url_pattern.sub(' ', text)
        
        # Remove mentions and hashtags (but keep the content)
        text = self.mention_pattern.sub(' ', text)
        text = self.hashtag_pattern.sub(' ', text)
        
        # Remove special characters but keep alphanumeric and spaces
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    async def predict_sentiment(self, text: str) -> Tuple[str, float]:
        """
        Predict sentiment for a given text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Tuple of (sentiment, confidence_score)
        """
        if not self.is_loaded or self.pipeline is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        try:
            start_time = time.time()
            
            # Preprocess text
            processed_text = self.preprocess_text(text)
            
            # Get prediction and probabilities
            prediction = self.pipeline.predict([processed_text])[0]
            probabilities = self.pipeline.predict_proba([processed_text])[0]
            
            # Get confidence score (max probability)
            confidence_score = max(probabilities)
            
            # Map prediction to sentiment label
            sentiment_mapping = {0: 'negative', 1: 'neutral', 2: 'positive'}
            sentiment = sentiment_mapping.get(prediction, 'neutral')
            
            processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            
            logger.debug(f"Sentiment prediction: '{sentiment}' (confidence: {confidence_score:.3f}, time: {processing_time:.1f}ms)")
            
            return sentiment, float(confidence_score)
        
        except Exception as e:
            logger.error(f"Sentiment prediction failed: {e}")
            # Return neutral sentiment with low confidence as fallback
            return "neutral", 0.5
    
    def train_default_model_sync(self) -> bool:
        """
        Train a default sentiment analysis model with sample data (synchronous version).
        
        Returns:
            True if training successful, False otherwise
        """
        try:
            logger.info("Training default sentiment analysis model...")
            
            # Sample training data (in production, use a larger dataset)
            sample_texts = [
                # Positive samples
                "I love this product! It's amazing and works perfectly.",
                "This is the best service I've ever experienced. Highly recommend!",
                "Fantastic quality and fast delivery. Very satisfied!",
                "Excellent customer support and great value for money.",
                "Outstanding performance and user-friendly interface.",
                "Perfect solution for my needs. Couldn't be happier!",
                "Great job on this feature. It works flawlessly.",
                "Impressed by the quality and attention to detail.",
                "Wonderful experience from start to finish.",
                "Top-notch product with excellent build quality.",
                
                # Negative samples
                "This product is terrible and doesn't work at all.",
                "Worst customer service ever. Very disappointed.",
                "Poor quality and overpriced. Not recommended.",
                "Completely useless and waste of money.",
                "Horrible experience. Nothing worked as expected.",
                "Failed to meet expectations. Very poor performance.",
                "Disappointing results and unreliable functionality.",
                "Bad design and confusing interface.",
                "Not worth the price. Very poor value.",
                "Frustrated with the poor quality and service.",
                
                # Neutral samples
                "The product works as described. Nothing special.",
                "Average quality for the price. Could be better.",
                "It's okay, but there's room for improvement.",
                "Standard functionality. Does what it says.",
                "Neither good nor bad. Just average.",
                "Decent product with basic features.",
                "Works fine but not outstanding.",
                "Acceptable quality for the price range.",
                "Normal performance as expected.",
                "It's an adequate solution for basic needs."
            ]
            
            # Labels (0: negative, 1: neutral, 2: positive)
            labels = [2] * 10 + [0] * 10 + [1] * 10
            
            # Preprocess texts
            processed_texts = [self.preprocess_text(text) for text in sample_texts]
            
            # Create and train pipeline
            self.pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(
                    max_features=5000,
                    stop_words='english',
                    ngram_range=(1, 2)
                )),
                ('classifier', LogisticRegression(
                    random_state=42,
                    max_iter=1000
                ))
            ])
            
            # Train the model
            self.pipeline.fit(processed_texts, labels)
            
            # Calculate accuracy on training data (for demonstration)
            predictions = self.pipeline.predict(processed_texts)
            accuracy = accuracy_score(labels, predictions)
            
            logger.info(f"Model trained with accuracy: {accuracy:.3f}")
            
            # Save the model
            model_data = {
                'pipeline': self.pipeline,
                'version': self.model_version,
                'training_accuracy': accuracy,
                'trained_at': time.time()
            }
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            
            with open(self.model_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            self.is_loaded = True
            logger.info(f"Default model saved to {self.model_path}")
            
            return True
        
        except Exception as e:
            logger.error(f"Failed to train default model: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        return {
            'version': self.model_version,
            'is_loaded': self.is_loaded,
            'model_path': self.model_path,
            'model_type': 'TF-IDF + Logistic Regression'
        }
    
    async def train_default_model(self) -> bool:
        """Async wrapper for train_default_model_sync."""
        return self.train_default_model_sync()
    
    def predict_sentiment_sync(self, text: str) -> Tuple[str, float]:
        """
        Predict sentiment for a given text (synchronous version).
        
        Args:
            text: Text to analyze
            
        Returns:
            Tuple of (sentiment, confidence_score)
        """
        if not self.is_loaded or self.pipeline is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        try:
            start_time = time.time()
            
            # Preprocess text
            processed_text = self.preprocess_text(text)
            
            # Get prediction and probabilities
            prediction = self.pipeline.predict([processed_text])[0]
            probabilities = self.pipeline.predict_proba([processed_text])[0]
            
            # Get confidence score (max probability)
            confidence = float(max(probabilities))
            
            # Map prediction to label
            sentiment_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
            sentiment = sentiment_map.get(prediction, 'neutral')
            
            processing_time = (time.time() - start_time) * 1000
            logger.debug(f"Sentiment prediction: {sentiment} ({confidence:.3f}) in {processing_time:.2f}ms")
            
            return sentiment, confidence
            
        except Exception as e:
            logger.error(f"Sentiment prediction failed: {e}")
            return "neutral", 0.5