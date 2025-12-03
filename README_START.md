# 🛡️ CipherGuard: Encrypted Real-Time Fraud Detection System

> **A production-ready fraud detection platform using CyborgDB for privacy-preserving encrypted vector storage.**

**API Key:** `cyborg_e3652dfedfa64a2392d9a927211ffd77` ✅

---

## 🚀 Quick Start (5 minutes)

### Step 1: Install CyborgDB SDK

```bash
pip install cyborgdb cyborgdb-service
```

### Step 2: Set API Key

**Windows (cmd):**
```cmd
set CYBORGDB_API_KEY=cyborg_e3652dfedfa64a2392d9a927211ffd77
```

**Linux/macOS:**
```bash
export CYBORGDB_API_KEY="cyborg_e3652dfedfa64a2392d9a927211ffd77"
```

### Step 3: Install & Run

```bash
# Install dependencies
pip install -r requirements.txt

# Start API server
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8001
```

### Step 4: Test Fraud Detection

```bash
# Normal transaction
curl -X POST http://localhost:8001/detect \
  -H "Content-Type: application/json" \
  -d '{"amount": 100, "merchant": "Amazon", "device": "desktop", "country": "US"}'

# Suspicious transaction
curl -X POST http://localhost:8001/detect \
  -H "Content-Type: application/json" \
  -d '{"amount": 15000, "merchant": "Unknown", "device": "mobile", "country": "CN"}'
```

**Expected Response:**
```json
{
  "transaction_id": "txn_1701619200.123",
  "is_fraud": false,
  "fraud_score": 0.35,
  "risk_level": "LOW",
  "similar_transactions": [],
  "timestamp": "2025-12-03T10:30:00"
}
```

---

## 📌 Project Overview

**Problem:** Financial institutions need real-time fraud detection that protects customer privacy.

**Solution:** CipherGuard uses **CyborgDB** (encrypted vector database) to:
- ✅ Detect fraud without exposing sensitive data
- ✅ Prevent embedding inversion attacks
- ✅ Maintain GDPR/PCI-DSS compliance
- ✅ Achieve sub-15ms fraud detection latency

---

## 📁 Project Structure

```
cipherguard-fraud-poc/
├── app/
│   ├── __init__.py                 # Package init
│   ├── main.py                     # FastAPI app & endpoints
│   ├── feature_extraction.py       # Transaction → Vector
│   ├── cyborg_client.py            # CyborgDB SDK client (encrypted)
│   └── cyborg_shim.py              # Local mock (fallback)
├── requirements.txt                # Dependencies
├── .env                            # Configuration (with API key)
├── .env.example                    # Config template
├── SETUP.md                        # Detailed setup guide
├── test_api.py                     # Test script
└── README_START.md                 # This file
```

---

## 🔐 CyborgDB Integration

### What is CyborgDB?

**Encrypted Vector Database** that enables:
- 🔒 **Client-side encryption** - Data encrypted before leaving client
- 🔍 **Encrypted kNN search** - Find similar patterns without decrypting
- 🛡️ **Privacy-preserving** - Protects against embedding inversion attacks
- ⚡ **Fast** - Millisecond-level encrypted searches

### API Key Credentials

```
API Key: cyborg_e3652dfedfa64a2392d9a927211ffd77
```

This enables:
- Encrypted storage of transaction vectors
- Client-side encryption with automatic key management
- Secure nearest neighbor search
- Compliance with privacy regulations

---

## 🏗️ System Architecture

---

## 🚀 API Endpoints

### 🔍 Fraud Detection
```bash
POST /detect
```

**Request:**
```json
{
  "amount": 950.00,
  "merchant": "Amazon",
  "device": "mobile",
  "country": "US"
}
```

**Response:**
```json
{
  "transaction_id": "txn_1701619200.123",
  "is_fraud": false,
  "fraud_score": 0.35,
  "risk_level": "LOW",
  "similar_transactions": ["txn_001"],
  "timestamp": "2025-12-03T10:30:00"
}
```

### 📝 Analyst Feedback
```bash
POST /feedback
```

Submit human review to retrain model:
```json
{
  "transaction_id": "txn_1701619200.123",
  "was_fraud": true,
  "feedback_text": "Confirmed fraudulent - unauthorized charge"
}
```

### 🔄 Model Retraining
```bash
POST /train
```

Trigger Isolation Forest retraining on stored vectors.

### 📊 Statistics
```bash
GET /stats
```

Get system metrics:
```json
{
  "count": 42,
  "vector_dim": 6,
  "model_trained": true,
  "timestamp": "2025-12-03T10:30:00"
}
```

### 🏥 Health Check
```bash
GET /health
```

Check API and backend status:
```json
{
  "status": "operational",
  "cyborg_vectors_count": 42,
  "model_status": "trained",
  "timestamp": "2025-12-03T10:30:00"
}
```

---

## 📊 Workflow Overview

```
┌─────────────────────┐
│ Incoming Transaction│
└──────────┬──────────┘
           │
    (1) Feature Extraction
           │ [6-dim vector]
           ▼
┌──────────────────────┐
│  CyborgDB Client     │ ← Client-side encryption
├──────────────────────┤
│ • Encrypt vector     │
│ • Insert encrypted   │
│ • Search encrypted   │
└──────────┬───────────┘
           │ [encrypted]
           ▼
┌──────────────────────┐
│ CyborgDB Service     │ ← Encrypted vector store
├──────────────────────┤
│ • PostgreSQL backend │
│ • kNN search         │
│ • Vector indexing    │
└──────────┬───────────┘
           │ [similar vectors]
           ▼
┌──────────────────────┐
│ Fraud Scoring Model  │
├──────────────────────┤
│ • Isolation Forest   │
│ • kNN distances      │
│ • Risk calculation   │
└──────────┬───────────┘
           │ [fraud_score]
           ▼
┌──────────────────────┐
│ Risk Assessment      │
├──────────────────────┤
│ • Decision: Fraud?   │
│ • Risk Level         │
│ • Alert if needed    │
└──────────────────────┘
```
