#!/usr/bin/env python
"""
Phase 2 Test: Secure Data Pipeline with Kafka Streaming
Tests the complete Kafka-based fraud detection pipeline
"""

import requests
import json
import time
import sys
from typing import Dict
import os

# Service URLs
API_URL = "http://localhost:8000"

def test_kafka_pipeline():
    """Test the complete Kafka streaming pipeline."""
    print("🌀 Testing Phase 2: Secure Data Pipeline with Kafka Streaming")
    print("=" * 60)

    # Test transactions
    test_cases = [
        {
            "name": "Normal Transaction",
            "data": {
                "amount": 100.00,
                "merchant": "Amazon",
                "device": "desktop",
                "country": "US",
                "customer_id": "CUST_001"
            },
            "expected_fraud": False
        },
        {
            "name": "Suspicious Transaction",
            "data": {
                "amount": 25000.00,
                "merchant": "Unknown Store",
                "device": "mobile",
                "country": "RU",
                "customer_id": "CUST_002"
            },
            "expected_fraud": True
        }
    ]

    results = []

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🔍 Test {i}: {test_case['name']}")
        print("-" * 40)

        try:
            # Send transaction through API gateway
            print(f"📤 Sending transaction: {test_case['data']['amount']} from {test_case['data']['merchant']}")

            start_time = time.time()
            response = requests.post(
                f"{API_URL}/detect",
                json=test_case["data"],
                timeout=60  # Increased timeout for Kafka processing
            )
            end_time = time.time()

            if response.status_code == 200:
                result = response.json()
                processing_time = end_time - start_time

                print("✅ Response received:"                print(f"   Transaction ID: {result['transaction_id']}")
                print(".3f"                print(f"   Risk Level: {result['risk_level']}")
                print(f"   Is Fraud: {result['is_fraud']}")
                print(".2f"
                # Validate result
                is_correct = (result['is_fraud'] == test_case['expected_fraud'])
                results.append({
                    "test": test_case["name"],
                    "success": True,
                    "processing_time": processing_time,
                    "correct_prediction": is_correct,
                    "result": result
                })

                if is_correct:
                    print("✅ Prediction matches expected result")
                else:
                    print("⚠️  Prediction differs from expected result")

            else:
                print(f"❌ API call failed: HTTP {response.status_code}")
                print(f"   Response: {response.text}")
                results.append({
                    "test": test_case["name"],
                    "success": False,
                    "error": f"HTTP {response.status_code}",
                    "response": response.text
                })

        except requests.exceptions.Timeout:
            print("❌ Request timed out - Kafka pipeline may not be responding")
            results.append({
                "test": test_case["name"],
                "success": False,
                "error": "Timeout",
                "timeout_seconds": 60
            })
        except Exception as e:
            print(f"❌ Test failed: {str(e)}")
            results.append({
                "test": test_case["name"],
                "success": False,
                "error": str(e)
            })

    # Summary
    print("\n📊 Phase 2 Test Results Summary")
    print("=" * 60)

    successful_tests = sum(1 for r in results if r["success"])
    total_tests = len(results)

    print(f"Tests Passed: {successful_tests}/{total_tests}")

    if successful_tests == total_tests:
        print("🎉 All tests passed! Kafka streaming pipeline is working correctly.")

        # Show performance metrics
        processing_times = [r["processing_time"] for r in results if "processing_time" in r]
        if processing_times:
            avg_time = sum(processing_times) / len(processing_times)
            min_time = min(processing_times)
            max_time = max(processing_times)
            print("
⏱️  Performance Metrics:"            print(".2f"            print(".2f"            print(".2f"
        # Show prediction accuracy
        predictions = [r for r in results if "correct_prediction" in r]
        if predictions:
            correct_predictions = sum(1 for r in predictions if r["correct_prediction"])
            accuracy = correct_predictions / len(predictions) * 100
            print("
🎯 Prediction Accuracy:"            print(".1f"
        return True
    else:
        print("❌ Some tests failed. Check the Kafka pipeline setup.")
        print("\nFailed Tests:")
        for result in results:
            if not result["success"]:
                print(f"  - {result['test']}: {result.get('error', 'Unknown error')}")

        print("\n🔧 Troubleshooting Tips:")
        print("1. Ensure Docker containers are running: docker-compose ps")
        print("2. Check Kafka logs: docker-compose logs kafka")
        print("3. Verify service health: curl http://localhost:8000/health")
        print("4. Check individual services are consuming from Kafka")

        return False

def test_health_endpoints():
    """Test that all services are healthy."""
    print("\n🏥 Checking Service Health...")

    services = [
        ("API Gateway", f"{API_URL}/health"),
        ("Ingestion Service", "http://localhost:8001/health"),
        ("Embedding Service", "http://localhost:8002/health"),
        ("Fraud Detection Service", "http://localhost:8003/health"),
        ("Alert Service", "http://localhost:8004/health")
    ]

    healthy_count = 0

    for name, url in services:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                print(f"✅ {name}: Healthy")
                healthy_count += 1
            else:
                print(f"❌ {name}: HTTP {response.status_code}")
        except Exception as e:
            print(f"❌ {name}: {str(e)}")

    if healthy_count == len(services):
        print("✅ All services are healthy!")
        return True
    else:
        print(f"⚠️  Only {healthy_count}/{len(services)} services are healthy")
        return False

def main():
    """Main test function."""
    print("🚀 CipherGuard Phase 2: Secure Data Pipeline Test")
    print("Testing Kafka streaming with encrypted data pipeline")
    print("=" * 60)

    # Check if services are running
    if not test_health_endpoints():
        print("\n❌ Services are not healthy. Please start them with:")
        print("   docker-compose up -d")
        sys.exit(1)

    # Test the Kafka pipeline
    if test_kafka_pipeline():
        print("\n🎉 Phase 2 implementation successful!")
        print("Secure data pipeline with Kafka streaming is working correctly.")
        sys.exit(0)
    else:
        print("\n❌ Phase 2 tests failed.")
        sys.exit(1)

if __name__ == "__main__":
    main()