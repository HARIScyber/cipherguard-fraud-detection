#!/usr/bin/env python
"""
Microservices Integration Test
Tests the full CipherGuard microservices pipeline
"""

import requests
import json
import time
import sys
from typing import Dict

# Service URLs
INGESTION_URL = "http://localhost:8001"
EMBEDDING_URL = "http://localhost:8002"
FRAUD_URL = "http://localhost:8003"
ALERT_URL = "http://localhost:8004"
API_URL = "http://localhost:8000"

def test_service_health(name: str, url: str) -> bool:
    """Test if a service is healthy."""
    try:
        resp = requests.get(f"{url}/health", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            print(f"✅ {name}: {data.get('status', 'unknown')}")
            return True
        else:
            print(f"❌ {name}: HTTP {resp.status_code}")
            return False
    except Exception as e:
        print(f"❌ {name}: {str(e)}")
        return False

def test_full_pipeline():
    """Test the complete fraud detection pipeline."""
    print("\n🧪 Testing Full Microservices Pipeline...")

    # Sample transaction
    transaction = {
        "amount": 25000.00,
        "merchant": "Suspicious Store",
        "device": "mobile",
        "country": "RU",
        "customer_id": "TEST001"
    }

    try:
        # Step 1: Ingest transaction
        print("\n📥 Step 1: Ingesting transaction...")
        ingest_resp = requests.post(f"{INGESTION_URL}/ingest", json=transaction, timeout=10)
        if ingest_resp.status_code != 200:
            print(f"❌ Ingestion failed: {ingest_resp.text}")
            return False

        ingest_data = ingest_resp.json()
        transaction_id = ingest_data["transaction_id"]
        print(f"✅ Transaction ingested: {transaction_id}")

        # Step 2: Get embedding (wait a moment for processing)
        print("\n🧮 Step 2: Getting transaction embedding...")
        time.sleep(1)

        embed_resp = requests.get(f"{EMBEDDING_URL}/vectors/{transaction_id}", timeout=10)
        if embed_resp.status_code != 200:
            print(f"❌ Embedding retrieval failed: {embed_resp.text}")
            return False

        embed_data = embed_resp.json()
        vector = embed_data["vector"]
        print(f"✅ Vector created: {len(vector)} dimensions")

        # Step 3: Run fraud detection
        print("\n🔍 Step 3: Running fraud detection...")
        detect_payload = {
            "transaction_id": transaction_id,
            "vector": vector,
            "metadata": transaction
        }

        detect_resp = requests.post(f"{FRAUD_URL}/detect", json=detect_payload, timeout=10)
        if detect_resp.status_code != 200:
            print(f"❌ Fraud detection failed: {detect_resp.text}")
            return False

        detect_data = detect_resp.json()
        print(f"✅ Fraud analysis complete:")
        print(f"   - Fraud Score: {detect_data['fraud_score']:.3f}")
        print(f"   - Risk Level: {detect_data['risk_level']}")
        print(f"   - Is Fraud: {detect_data['is_fraud']}")

        # Step 4: Create alert if fraudulent
        if detect_data['is_fraud']:
            print("\n🚨 Step 4: Creating fraud alert...")
            alert_payload = {
                "transaction_id": transaction_id,
                "fraud_score": detect_data['fraud_score'],
                "risk_level": detect_data['risk_level'],
                "is_fraud": detect_data['is_fraud'],
                "customer_id": transaction['customer_id'],
                "amount": transaction['amount'],
                "merchant": transaction['merchant'],
                "timestamp": detect_data['timestamp']
            }

            alert_resp = requests.post(f"{ALERT_URL}/alert", json=alert_payload, timeout=10)
            if alert_resp.status_code != 200:
                print(f"❌ Alert creation failed: {alert_resp.text}")
                return False

            alert_data = alert_resp.json()
            print(f"✅ Alert created: {alert_data['alert_id']}")

        # Step 5: Test main API gateway
        print("\n🌐 Step 5: Testing API Gateway...")
        gateway_resp = requests.post(f"{API_URL}/detect", json=transaction, timeout=15)
        if gateway_resp.status_code != 200:
            print(f"❌ API Gateway failed: {gateway_resp.text}")
            return False

        gateway_data = gateway_resp.json()
        print(f"✅ API Gateway response:")
        print(f"   - Transaction ID: {gateway_data['transaction_id']}")
        print(f"   - Fraud Score: {gateway_data['fraud_score']:.3f}")
        print(f"   - Risk Level: {gateway_data['risk_level']}")

        print("\n🎉 Full pipeline test PASSED!")
        return True

    except Exception as e:
        print(f"❌ Pipeline test failed: {str(e)}")
        return False

def main():
    """Main test function."""
    print("🚀 CipherGuard Microservices Integration Test")
    print("=" * 50)

    # Test service health
    print("\n🏥 Checking Service Health...")
    services = [
        ("Ingestion Service", INGESTION_URL),
        ("Embedding Service", EMBEDDING_URL),
        ("Fraud Detection Service", FRAUD_URL),
        ("Alert Service", ALERT_URL),
        ("API Gateway", API_URL)
    ]

    healthy_services = 0
    for name, url in services:
        if test_service_health(name, url):
            healthy_services += 1

    if healthy_services < len(services):
        print(f"\n⚠️  Only {healthy_services}/{len(services)} services are healthy")
        print("Make sure all services are running:")
        print("docker-compose up -d")
        sys.exit(1)

    # Test full pipeline
    if test_full_pipeline():
        print("\n✅ All tests passed! Microservices are working correctly.")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Check service logs for details.")
        sys.exit(1)

if __name__ == "__main__":
    main()
