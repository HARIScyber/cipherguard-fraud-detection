#!/usr/bin/env python
"""
Quick test script for CipherGuard API
Tests fraud detection endpoint with sample transactions
"""

import requests
import json
from typing import Dict

API_URL = "http://localhost:8001"

# Sample transactions for testing
SAMPLE_TRANSACTIONS = {
    "normal": {
        "amount": 150.00,
        "merchant": "Amazon",
        "device": "desktop",
        "country": "US"
    },
    "suspicious_high_amount": {
        "amount": 25000.00,
        "merchant": "Unknown",
        "device": "mobile",
        "country": "CN"
    },
    "risky_mobile": {
        "amount": 5000.00,
        "merchant": "Apple",
        "device": "mobile",
        "country": "Other"
    },
    "legitimate_online": {
        "amount": 89.99,
        "merchant": "Walmart",
        "device": "mobile",
        "country": "US"
    }
}


def test_health():
    """Test health endpoint."""
    print("\n🏥 Testing Health Endpoint...")
    try:
        response = requests.get(f"{API_URL}/health")
        data = response.json()
        print(f"✅ Status: {data['status']}")
        print(f"   Vectors: {data['cyborg_vectors_count']}")
        print(f"   Model: {data['model_status']}")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_fraud_detection(name: str, transaction: Dict):
    """Test fraud detection on a transaction."""
    print(f"\n🔍 Testing: {name}")
    print(f"   Transaction: {json.dumps(transaction, indent=2)}")
    
    try:
        response = requests.post(
            f"{API_URL}/detect",
            json=transaction,
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            
            # Color-coded risk level
            risk_colors = {
                "LOW": "🟢",
                "MEDIUM": "🟡",
                "HIGH": "🟠",
                "CRITICAL": "🔴"
            }
            
            risk_icon = risk_colors.get(data['risk_level'], "⚪")
            fraud_icon = "🚨" if data['is_fraud'] else "✅"
            
            print(f"   {fraud_icon} Fraud: {data['is_fraud']}")
            print(f"   {risk_icon} Risk Level: {data['risk_level']}")
            print(f"   📊 Fraud Score: {data['fraud_score']:.2%}")
            print(f"   🔗 Similar Txns: {len(data['similar_transactions'])}")
            
            return True
        else:
            print(f"❌ Error {response.status_code}: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Connection Error: {e}")
        return False


def test_stats():
    """Test stats endpoint."""
    print("\n📈 Testing Stats Endpoint...")
    try:
        response = requests.get(f"{API_URL}/stats")
        data = response.json()
        print(f"✅ Vectors Stored: {data['count']}")
        print(f"   Vector Dim: {data['vector_dim']}")
        print(f"   Model Trained: {data.get('model_trained', False)}")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("🛡️  CipherGuard Fraud Detection - API Test Suite")
    print("=" * 60)
    
    # Check connection
    try:
        response = requests.get(f"{API_URL}/", timeout=5)
        print(f"✅ Connected to API at {API_URL}")
        root_data = response.json()
        print(f"   Backend: {root_data.get('backend', 'Unknown')}")
    except Exception as e:
        print(f"❌ Cannot connect to API at {API_URL}")
        print(f"   Make sure API is running: python -m uvicorn app.main:app --reload")
        return
    
    # Run tests
    tests_passed = 0
    tests_total = 0
    
    # Health check
    tests_total += 1
    if test_health():
        tests_passed += 1
    
    # Test fraud detection on each sample
    for name, transaction in SAMPLE_TRANSACTIONS.items():
        tests_total += 1
        if test_fraud_detection(name.replace("_", " ").title(), transaction):
            tests_passed += 1
    
    # Stats check
    tests_total += 1
    if test_stats():
        tests_passed += 1
    
    # Summary
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {tests_passed}/{tests_total} passed")
    print("=" * 60)
    
    if tests_passed == tests_total:
        print("✅ All tests passed! API is working correctly.\n")
    else:
        print(f"⚠️  {tests_total - tests_passed} test(s) failed.\n")


if __name__ == "__main__":
    main()
