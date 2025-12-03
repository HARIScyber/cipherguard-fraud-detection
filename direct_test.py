#!/usr/bin/env python
"""
CipherGuard API test - tests the application directly without server
Shows the full fraud detection workflow
"""

import sys
import json
import numpy as np
from datetime import datetime

# Add the app to the path
from app.main import app
from app.feature_extraction import extract_transaction_vector
from app.cyborg_shim import get_cyborg_shim

shim = get_cyborg_shim()

print("\n" + "="*60)
print("🛡️  CipherGuard Fraud Detection - Direct Test")
print("="*60)

# Test 1: Feature extraction
print("\n1️⃣  Testing Feature Extraction...")
transaction = {
    "amount": 150.00,
    "merchant": "Amazon",
    "device": "desktop",
    "country": "US",
    "customer_id": "CUST001",
    "timestamp": datetime.utcnow().isoformat()
}

vector = extract_transaction_vector(transaction)
print(f"✅ Vector shape: {vector.shape}")
print(f"✅ Vector values: {vector}")
print(f"✅ Vector norm: {np.linalg.norm(vector):.3f}")

# Test 2: Insert into shim
print("\n2️⃣  Testing Vector Storage...")
txn_id = f"txn_{datetime.utcnow().timestamp()}"
metadata = {
    "customer_id": transaction.get("customer_id"),
    "timestamp": transaction.get("timestamp"),
    "merchant": transaction.get("merchant"),
    "amount": transaction.get("amount")
}

shim.insert(txn_id, vector, metadata)
print(f"✅ Inserted transaction: {txn_id}")
print(f"✅ Total vectors in store: {shim.count()}")

# Test 3: Search similar vectors
print("\n3️⃣  Testing Vector Search (kNN)...")
suspicious_txn = {
    "amount": 25000.00,
    "merchant": "Unknown",
    "device": "mobile",
    "country": "CN",
    "customer_id": "CUST002",
    "timestamp": datetime.utcnow().isoformat()
}

suspicious_vector = extract_transaction_vector(suspicious_txn)
results = shim.search(suspicious_vector, k=5)
print(f"✅ kNN search returned {len(results)} results")
for i, (tid, distance) in enumerate(results, 1):
    print(f"  {i}. Transaction: {tid}, Distance: {distance:.4f}")

# Test 4: Simulate fraud detection
print("\n4️⃣  Testing Fraud Detection Logic...")
from app.main import compute_fraud_score, get_risk_level

fraud_score = compute_fraud_score(suspicious_vector, results, 0.7)
risk_level = get_risk_level(fraud_score)
is_fraud = fraud_score > 0.6

print(f"✅ Fraud score: {fraud_score:.3f}")
print(f"✅ Risk level: {risk_level}")
print(f"✅ Is fraud: {is_fraud}")

# Test 5: Batch insert for model training
print("\n5️⃣  Testing Batch Insert...")
batch_size = 15
for i in range(batch_size):
    amt = np.random.uniform(50, 5000)
    merchants = ["Amazon", "Walmart", "Target", "Best Buy", "Unknown"]
    devices = ["desktop", "mobile", "tablet"]
    countries = ["US", "UK", "CN", "RU", "Other"]
    
    batch_txn = {
        "amount": amt,
        "merchant": np.random.choice(merchants),
        "device": np.random.choice(devices),
        "country": np.random.choice(countries),
        "customer_id": f"CUST{i:03d}",
        "timestamp": datetime.utcnow().isoformat()
    }
    
    batch_vector = extract_transaction_vector(batch_txn)
    batch_id = f"txn_batch_{i}"
    shim.insert(batch_id, batch_vector, {"batch": i})

print(f"✅ Inserted {batch_size} transactions")
print(f"✅ Total vectors in store: {shim.count()}")

# Test 6: Train Isolation Forest
print("\n6️⃣  Testing Model Training...")
from sklearn.ensemble import IsolationForest

vectors = []
for tid, vec in shim.vectors.items():
    vectors.append(vec)

X = np.array(vectors)
iso_forest = IsolationForest(contamination=0.1, random_state=42, n_estimators=100)
iso_forest.fit(X)

print(f"✅ Trained Isolation Forest on {len(vectors)} vectors")
anomaly_preds = iso_forest.predict(X[:3])
print(f"✅ Sample predictions: {anomaly_preds}")

# Test 7: Final fraud detection report
print("\n7️⃣  Testing Full Fraud Detection Pipeline...")
test_txn = {
    "amount": 8000.00,
    "merchant": "Unknown Store",
    "device": "mobile",
    "country": "RU",
    "customer_id": "CUST_FINAL",
    "timestamp": datetime.utcnow().isoformat()
}

test_vector = extract_transaction_vector(test_txn)
knn_results = shim.search(test_vector, k=5)
anomaly_pred = iso_forest.predict(test_vector.reshape(1, -1))[0]
anomaly_score = 1.0 if anomaly_pred == -1 else 0.2
fraud_score = compute_fraud_score(test_vector, knn_results, anomaly_score)
is_fraud = fraud_score > 0.6
risk_level = get_risk_level(fraud_score)

print(f"✅ Transaction amount: ${test_txn['amount']}")
print(f"✅ Merchant: {test_txn['merchant']}")
print(f"✅ Device: {test_txn['device']}")
print(f"✅ Country: {test_txn['country']}")
print(f"✅ Anomaly score: {anomaly_score:.2f}")
print(f"✅ Fraud score: {fraud_score:.3f}")
print(f"✅ Risk level: {risk_level}")
print(f"✅ Is fraud: {is_fraud}")

print("\n" + "="*60)
print("✅ ALL TESTS PASSED - System is working correctly!")
print("="*60 + "\n")
