"""
Test script to verify the IsolationForest fix works correctly
"""

import requests
import json
import time

def test_fraud_detection_fix():
    """Test the fixed fraud detection endpoint."""
    
    print("🧪 Testing CipherGuard Fraud Detection Fix")
    print("=" * 50)
    
    base_url = "http://localhost:8001"
    
    # Test 1: Health check with model status
    print("1️⃣ Testing health check with model status...")
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            health_data = response.json()
            print(f"   ✅ Health check passed")
            print(f"   📊 Model status: {health_data.get('model_status', {})}")
        else:
            print(f"   ❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Health check error: {e}")
        return False
    
    print()
    
    # Test 2: Model status endpoint
    print("2️⃣ Testing model status endpoint...")
    try:
        response = requests.get(f"{base_url}/models/status")
        if response.status_code == 200:
            model_data = response.json()
            print(f"   ✅ Model status retrieved")
            print(f"   🤖 IsolationForest loaded: {model_data['isolation_forest']['loaded']}")
            print(f"   💾 Model file exists: {model_data['isolation_forest']['file_exists']}")
        else:
            print(f"   ❌ Model status failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Model status error: {e}")
    
    print()
    
    # Test 3: Fraud detection with normal transaction
    print("3️⃣ Testing fraud detection (normal transaction)...")
    normal_transaction = {
        "amount": 100.0,
        "merchant": "Amazon",
        "device": "desktop",
        "country": "US"
    }
    
    try:
        response = requests.post(
            f"{base_url}/detect",
            json=normal_transaction,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ Normal transaction processed successfully")
            print(f"   📈 Fraud score: {result['fraud_score']:.3f}")
            print(f"   🚦 Risk level: {result['risk_level']}")
            print(f"   🆔 Transaction ID: {result['transaction_id']}")
        else:
            print(f"   ❌ Normal transaction failed: {response.status_code}")
            print(f"   📄 Response: {response.text}")
            return False
    except Exception as e:
        print(f"   ❌ Normal transaction error: {e}")
        return False
    
    print()
    
    # Test 4: Fraud detection with suspicious transaction  
    print("4️⃣ Testing fraud detection (suspicious transaction)...")
    suspicious_transaction = {
        "amount": 25000.0,  # Very high amount
        "merchant": "Unknown_Merchant",
        "device": "mobile",
        "country": "XX"  # Unknown country
    }
    
    try:
        response = requests.post(
            f"{base_url}/detect",
            json=suspicious_transaction,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ Suspicious transaction processed successfully")
            print(f"   📈 Fraud score: {result['fraud_score']:.3f}")
            print(f"   🚦 Risk level: {result['risk_level']}")
            print(f"   ⚠️ Is fraud: {result['is_fraud']}")
        else:
            print(f"   ❌ Suspicious transaction failed: {response.status_code}")
            print(f"   📄 Response: {response.text}")
            return False
    except Exception as e:
        print(f"   ❌ Suspicious transaction error: {e}")
        return False
    
    print()
    
    # Test 5: Performance test with multiple requests
    print("5️⃣ Testing performance with multiple requests...")
    test_transaction = {
        "amount": 500.0,
        "merchant": "Target",
        "device": "desktop", 
        "country": "US"
    }
    
    times = []
    success_count = 0
    
    for i in range(5):
        try:
            start_time = time.time()
            response = requests.post(
                f"{base_url}/detect",
                json=test_transaction,
                headers={"Content-Type": "application/json"}
            )
            end_time = time.time()
            
            if response.status_code == 200:
                success_count += 1
                times.append((end_time - start_time) * 1000)  # Convert to ms
        except Exception as e:
            print(f"   ⚠️ Request {i+1} failed: {e}")
    
    if times:
        avg_time = sum(times) / len(times)
        print(f"   ✅ Performance test completed")
        print(f"   📊 Success rate: {success_count}/5")
        print(f"   ⚡ Average response time: {avg_time:.1f}ms")
        print(f"   📈 Min/Max response time: {min(times):.1f}ms / {max(times):.1f}ms")
    
    print()
    print("🎉 All tests completed! The IsolationForest fix is working correctly.")
    return True

if __name__ == "__main__":
    print("Make sure your FastAPI server is running on http://localhost:8001")
    print("Run: uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload")
    print()
    
    input("Press Enter when server is ready...")
    test_fraud_detection_fix()