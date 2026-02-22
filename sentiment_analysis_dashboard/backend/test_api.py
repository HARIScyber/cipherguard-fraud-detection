#!/usr/bin/env python3
"""
Test script for the Sentiment Analysis Dashboard backend.
Tests API endpoints and core functionality.
"""

import asyncio
import json
import sys
import os
from pathlib import Path

# Add app directory to path
current_dir = Path(__file__).parent
app_dir = current_dir / "app"
sys.path.insert(0, str(app_dir))

import httpx
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test configuration
BASE_URL = f"http://localhost:{os.getenv('API_PORT', 8000)}"
TEST_USER = {
    "username": "testuser",
    "email": "test@example.com", 
    "password": "testpass123",
    "full_name": "Test User"
}

class APITester:
    def __init__(self):
        self.client = httpx.AsyncClient()
        self.access_token = None
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()
    
    async def test_health_check(self):
        """Test basic health check endpoint."""
        logger.info("Testing health check...")
        
        try:
            response = await self.client.get(f"{BASE_URL}/api/v1/health/")
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"✅ Health check passed: {data['data']['status']}")
                return True
            else:
                logger.error(f"❌ Health check failed: {response.status_code}")
                return False
        
        except Exception as e:
            logger.error(f"❌ Health check error: {e}")
            return False
    
    async def test_user_registration(self):
        """Test user registration."""
        logger.info("Testing user registration...")
        
        try:
            response = await self.client.post(
                f"{BASE_URL}/api/v1/auth/register",
                json=TEST_USER
            )
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"✅ User registration passed: {data['data']['username']}")
                return True
            elif response.status_code == 400 and "already registered" in response.json()["detail"]:
                logger.info("✅ User registration passed (user already exists)")
                return True
            else:
                logger.error(f"❌ User registration failed: {response.status_code} - {response.text}")
                return False
        
        except Exception as e:
            logger.error(f"❌ User registration error: {e}")
            return False
    
    async def test_user_login(self):
        """Test user login and get access token."""
        logger.info("Testing user login...")
        
        try:
            login_data = {
                "username": TEST_USER["username"],
                "password": TEST_USER["password"]
            }
            
            response = await self.client.post(
                f"{BASE_URL}/api/v1/auth/login",
                json=login_data
            )
            
            if response.status_code == 200:
                data = response.json()
                self.access_token = data["data"]["access_token"]
                logger.info("✅ User login passed")
                return True
            else:
                logger.error(f"❌ User login failed: {response.status_code} - {response.text}")
                return False
        
        except Exception as e:
            logger.error(f"❌ User login error: {e}")
            return False
    
    async def test_protected_endpoint(self):
        """Test protected endpoint with authentication."""
        logger.info("Testing protected endpoint...")
        
        if not self.access_token:
            logger.error("❌ No access token available")
            return False
        
        try:
            headers = {"Authorization": f"Bearer {self.access_token}"}
            response = await self.client.get(
                f"{BASE_URL}/api/v1/auth/me",
                headers=headers
            )
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"✅ Protected endpoint passed: {data['data']['username']}")
                return True
            else:
                logger.error(f"❌ Protected endpoint failed: {response.status_code} - {response.text}")
                return False
        
        except Exception as e:
            logger.error(f"❌ Protected endpoint error: {e}")
            return False
    
    async def test_sentiment_analysis(self):
        """Test sentiment analysis endpoint."""
        logger.info("Testing sentiment analysis...")
        
        if not self.access_token:
            logger.error("❌ No access token available")
            return False
        
        try:
            headers = {"Authorization": f"Bearer {self.access_token}"}
            
            test_comments = [
                "This product is absolutely amazing! I love it!",
                "This service is terrible and I hate it.",
                "It's okay, nothing special about it."
            ]
            
            results = []
            
            for comment in test_comments:
                request_data = {
                    "comment_text": comment,
                    "store_result": True
                }
                
                response = await self.client.post(
                    f"{BASE_URL}/api/v1/comments/analyze",
                    json=request_data,
                    headers=headers
                )
                
                if response.status_code == 200:
                    data = response.json()
                    sentiment = data["sentiment"]
                    confidence = data["confidence_score"]
                    results.append((comment[:30], sentiment, confidence))
                    logger.info(f"  Comment: '{comment[:30]}...' -> {sentiment} ({confidence:.2f})")
                else:
                    logger.error(f"❌ Sentiment analysis failed for comment: {response.status_code}")
                    return False
            
            if results:
                logger.info("✅ Sentiment analysis passed")
                return True
            else:
                logger.error("❌ No sentiment analysis results")
                return False
        
        except Exception as e:
            logger.error(f"❌ Sentiment analysis error: {e}")
            return False
    
    async def test_analytics(self):
        """Test analytics endpoints."""
        logger.info("Testing analytics...")
        
        if not self.access_token:
            logger.error("❌ No access token available")
            return False
        
        try:
            headers = {"Authorization": f"Bearer {self.access_token}"}
            
            # Test sentiment analytics
            response = await self.client.get(
                f"{BASE_URL}/api/v1/comments/analytics/sentiment?days=7",
                headers=headers
            )
            
            if response.status_code == 200:
                data = response.json()
                analytics = data["data"]
                logger.info(f"  Total comments: {analytics['total_comments']}")
                logger.info(f"  Positive: {analytics['positive_percentage']:.1f}%")
                logger.info(f"  Negative: {analytics['negative_percentage']:.1f}%")
                logger.info(f"  Neutral: {analytics['neutral_percentage']:.1f}%")
                logger.info("✅ Analytics passed")
                return True
            else:
                logger.error(f"❌ Analytics failed: {response.status_code} - {response.text}")
                return False
        
        except Exception as e:
            logger.error(f"❌ Analytics error: {e}")
            return False

async def run_all_tests():
    """Run all API tests."""
    logger.info("🚀 Starting API tests...")
    logger.info(f"Base URL: {BASE_URL}")
    
    async with APITester() as tester:
        tests = [
            ("Health Check", tester.test_health_check),
            ("User Registration", tester.test_user_registration),
            ("User Login", tester.test_user_login),
            ("Protected Endpoint", tester.test_protected_endpoint),
            ("Sentiment Analysis", tester.test_sentiment_analysis),
            ("Analytics", tester.test_analytics),
        ]
        
        results = []
        
        for test_name, test_func in tests:
            logger.info(f"\n--- {test_name} ---")
            result = await test_func()
            results.append((test_name, result))
        
        # Summary
        logger.info("\n" + "="*50)
        logger.info("TEST SUMMARY")
        logger.info("="*50)
        
        passed = 0
        total = len(results)
        
        for test_name, result in results:
            status = "✅ PASSED" if result else "❌ FAILED"
            logger.info(f"{test_name:<25} {status}")
            if result:
                passed += 1
        
        logger.info(f"\nTotal: {passed}/{total} tests passed")
        
        if passed == total:
            logger.info("🎉 All tests passed!")
            return True
        else:
            logger.error("💥 Some tests failed!")
            return False

async def test_database_setup():
    """Test database setup."""
    logger.info("Testing database setup...")
    
    try:
        from app.database import check_database_connection, create_admin_user
        
        # Test database connection
        if check_database_connection():
            logger.info("✅ Database connection successful")
        else:
            logger.error("❌ Database connection failed")
            return False
        
        # Test admin user creation
        try:
            create_admin_user()
            logger.info("✅ Admin user setup completed")
        except Exception as e:
            logger.error(f"❌ Admin user setup failed: {e}")
            return False
        
        return True
    
    except Exception as e:
        logger.error(f"❌ Database setup error: {e}")
        return False

def main():
    """Main test function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Sentiment Analysis API")
    parser.add_argument("--db-only", action="store_true", help="Test database setup only")
    parser.add_argument("--api-only", action="store_true", help="Test API endpoints only")
    args = parser.parse_args()
    
    if args.db_only:
        success = asyncio.run(test_database_setup())
    elif args.api_only:
        success = asyncio.run(run_all_tests())
    else:
        # Test both database and API
        db_success = asyncio.run(test_database_setup())
        if db_success:
            api_success = asyncio.run(run_all_tests())
            success = db_success and api_success
        else:
            success = False
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()