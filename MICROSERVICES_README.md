# CipherGuard Fraud Detection - Microservices Architecture

## Overview

CipherGuard is a production-ready, encrypted fraud detection system built with a microservices architecture. The system provides real-time fraud detection using advanced machine learning and encrypted vector storage.

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  API Gateway    │    │ Ingestion       │    │ Embedding       │
│  (Port 8000)    │◄──►│ Service         │◄──►│ Service         │
│                 │    │ (Port 8001)     │    │ (Port 8002)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Fraud Detection │    │ Alert Service   │    │   CyborgDB      │
│ Service         │◄──►│ (Port 8004)     │    │   (Encrypted)   │
│ (Port 8003)     │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Services

### 1. API Gateway (Port 8000)
- **Purpose**: Main entry point, orchestrates microservices
- **Technology**: FastAPI, Python
- **Endpoints**:
  - `POST /detect` - Full fraud detection pipeline
  - `GET /health` - System health check
  - `GET /stats` - System statistics

### 2. Ingestion Service (Port 8001)
- **Purpose**: Transaction data ingestion and validation
- **Technology**: FastAPI, Python
- **Endpoints**:
  - `POST /ingest` - Ingest transaction data
  - `GET /health` - Service health
  - `GET /queue` - View transaction queue

### 3. Embedding Service (Port 8002)
- **Purpose**: Feature extraction and vector embedding
- **Technology**: FastAPI, NumPy, Python
- **Features**:
  - 6-dimensional transaction vectors
  - L2 normalization
  - Encrypted vector storage preparation

### 4. Fraud Detection Service (Port 8003)
- **Purpose**: Core ML fraud detection logic
- **Technology**: FastAPI, scikit-learn, Python
- **Models**:
  - Isolation Forest for anomaly detection
  - kNN similarity search
  - Hybrid scoring algorithm

### 5. Alert Service (Port 8004)
- **Purpose**: Alert management and audit logging
- **Technology**: FastAPI, Python
- **Features**:
  - Fraud alert creation
  - Analyst feedback collection
  - Audit trail logging

## Quick Start

### Using Docker Compose (Recommended)

1. **Clone and setup**:
```bash
git clone https://github.com/HARIScyber/cipherguard-fraud-detection.git
cd cipherguard-fraud-detection
```

2. **Create environment file**:
```bash
cp .env.example .env
# Edit .env with your CyborgDB API key
```

3. **Start all services**:
```bash
docker-compose up -d
```

4. **Check health**:
```bash
curl http://localhost:8000/health
```

### Manual Setup

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Start individual services** (in separate terminals):
```bash
# Ingestion Service
cd services/ingestion && python main.py

# Embedding Service
cd services/embedding && python main.py

# Fraud Detection Service
cd services/fraud-detection && python main.py

# Alert Service
cd services/alert && python main.py

# Main API Gateway
python run_server.py
```

## API Usage Examples

### Full Fraud Detection Pipeline
```bash
curl -X POST "http://localhost:8000/detect" \
  -H "Content-Type: application/json" \
  -d '{
    "amount": 25000.00,
    "merchant": "Unknown Store",
    "device": "mobile",
    "country": "RU",
    "customer_id": "CUST001"
  }'
```

### Microservice Direct Access
```bash
# Ingest transaction
curl -X POST "http://localhost:8001/ingest" \
  -H "Content-Type: application/json" \
  -d '{"amount": 150.00, "merchant": "Amazon", "device": "desktop", "country": "US"}'

# Get embedding
curl "http://localhost:8002/vectors"

# Run detection
curl -X POST "http://localhost:8003/detect" \
  -H "Content-Type: application/json" \
  -d '{"transaction_id": "txn_123", "vector": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]}'

# Create alert
curl -X POST "http://localhost:8004/alert" \
  -H "Content-Type: application/json" \
  -d '{"transaction_id": "txn_123", "fraud_score": 0.85, "risk_level": "HIGH", "is_fraud": true}'
```

## Development

### Adding New Services
1. Create service directory under `services/`
2. Add `main.py`, `requirements.txt`, `Dockerfile`
3. Update `docker-compose.yml`
4. Add service discovery/routing in API Gateway

### Testing
```bash
# Run direct pipeline test
python direct_test.py

# Test individual services
python quick_test.py
```

## Security Features

- **Client-side encryption** via CyborgDB
- **Encrypted vector storage** with kNN search
- **Microservices isolation** for attack surface reduction
- **Health checks** and service monitoring
- **Audit logging** for compliance

## Future Roadmap

### Phase 2: Secure Data Pipeline
- Apache Kafka for streaming transactions
- Key Management Service (KMS)
- End-to-end encryption

### Phase 3: Advanced ML
- XGBoost/DNN models
- SHAP explainability
- Hybrid similarity + classification

### Phase 4: MLOps
- MLflow model versioning
- Automated retraining
- Data drift detection

### Phase 5: Enterprise Deployment
- Kubernetes orchestration
- Prometheus/Grafana monitoring
- Multi-region deployment

### Phase 6: Compliance & Governance
- PCI-DSS compliance
- GDPR data protection
- SIEM integration

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.