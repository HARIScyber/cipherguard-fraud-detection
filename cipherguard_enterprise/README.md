# CipherGuard Enterprise Fraud Detection Platform

[![License](https://img.shields.io/badge/license-Proprietary-red.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-13+-blue.svg)](https://postgresql.org)

## 🚀 Overview

CipherGuard is a production-ready, enterprise-grade fraud detection platform built with advanced machine learning capabilities. It provides real-time fraud detection, comprehensive analytics, and enterprise security features suitable for financial institutions and e-commerce platforms.

### ⭐ Key Features

- **🤖 Advanced ML Pipeline**: Ensemble models with IsolationForest, RandomForest, and XGBoost
- **⚡ Real-time Detection**: Sub-second fraud prediction with explainable AI (SHAP)
- **🔐 Enterprise Security**: JWT authentication, API keys, rate limiting, and comprehensive audit trails
- **📊 Analytics Dashboard**: Fraud statistics, trends, alerts, and model performance metrics
- **🏗️ Scalable Architecture**: Async FastAPI with PostgreSQL, proper caching, and horizontal scaling
- **🐳 Production Ready**: Docker containers, Kubernetes deployment, monitoring, and CI/CD
- **🧪 Comprehensive Testing**: Unit tests, integration tests, and performance testing
- **📖 Complete Documentation**: API docs, deployment guides, and architectural documentation

## 🏛️ Architecture

```mermaid
graph TB
    subgraph "API Layer"
        A[FastAPI Application] --> B[Authentication]
        A --> C[Rate Limiting]
        A --> D[Request Validation]
    end
    
    subgraph "Business Logic"
        E[Fraud Detection Service] --> F[Feature Engineering]
        E --> G[ML Pipeline]
        H[Analytics Service] --> I[Statistics]
        H --> J[Reporting]
    end
    
    subgraph "ML Layer"
        G --> K[Isolation Forest]
        G --> L[Random Forest]  
        G --> M[XGBoost]
        G --> N[Ensemble Model]
        N --> O[SHAP Explainer]
    end
    
    subgraph "Data Layer"
        P[Repository Pattern] --> Q[PostgreSQL]
        P --> R[SQLAlchemy ORM]
        S[Feature Store] --> Q
    end
    
    subgraph "Infrastructure"
        T[Docker Containers] --> U[Kubernetes]
        V[Monitoring] --> W[Prometheus]
        V --> X[Grafana]
        Y[Logging] --> Z[Structured JSON]
    end
    
    A --> E
    A --> H
    E --> P
    H --> P
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- PostgreSQL 13+
- Redis (optional, for caching)
- Docker & Docker Compose (for containerized deployment)

### 1. Clone and Setup

```bash
# Clone the repository
git clone https://github.com/your-org/cipherguard-fraud-detection.git
cd cipherguard-fraud-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit configuration (see Configuration section)
nano .env
```

### 3. Database Setup

```bash
# Create database
createdb cipherguard_fraud

# Run migrations
alembic upgrade head

# Optional: Load sample data
python scripts/load_sample_data.py
```

### 4. Start the Application

```bash
# Development server
python -m app.main

# Or using uvicorn directly
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Production server
gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker
```

### 5. Access the API

- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **Metrics**: http://localhost:8000/metrics

## 🔧 Configuration

The platform uses environment-based configuration with comprehensive validation:

```env
# Application Settings
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=info

# Database Configuration
DATABASE_URL=postgresql://user:password@localhost:5432/cipherguard_fraud
DATABASE_POOL_SIZE=10
DATABASE_MAX_OVERFLOW=20

# Security Settings
JWT_SECRET_KEY=your-super-secret-key-here
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=30
API_KEY=your-api-key-here
RATE_LIMIT_CALLS=100
RATE_LIMIT_PERIOD=60

# ML Configuration
FRAUD_THRESHOLD=0.7
HIGH_RISK_THRESHOLD=0.5
MEDIUM_RISK_THRESHOLD=0.3
HYPERPARAMETER_TUNING=true

# API Settings
API_HOST=0.0.0.0
API_PORT=8000
CORS_ORIGINS=["http://localhost:3000"]
```

## 🛡️ Security Features

### Authentication
- **JWT Tokens**: Industry-standard JSON Web Tokens with configurable expiration
- **API Keys**: Secure API key authentication for service-to-service communication  
- **Multi-method Support**: Flexible authentication supporting both methods simultaneously

### Security Middleware
- **Rate Limiting**: Configurable per-user/IP rate limiting with burst allowance
- **CORS Protection**: Comprehensive Cross-Origin Resource Sharing configuration
- **Security Headers**: Modern security headers (HSTS, CSP, X-Frame-Options)
- **Request Correlation**: Unique correlation IDs for request tracking and debugging

### Audit & Monitoring
- **Comprehensive Audit Logs**: All API access, authentication events, and system changes
- **Structured Logging**: JSON-formatted logs with correlation tracking
- **Performance Monitoring**: Request latency, throughput, and system health metrics

## 🤖 Machine Learning Pipeline

### Feature Engineering
The platform generates 50+ sophisticated features including:

- **Velocity Features**: Transaction frequency patterns over multiple time windows
- **Behavioral Features**: User spending patterns and deviation from normal behavior
- **Contextual Features**: Location, device, merchant, and time-based signals
- **Network Features**: IP-based risk signals and device fingerprinting

### Ensemble Models

1. **Isolation Forest**: Unsupervised anomaly detection for novel fraud patterns
2. **Random Forest**: Robust ensemble classifier with feature importance
3. **XGBoost**: Gradient boosting with hyperparameter optimization via Optuna

### Model Explainability
- **SHAP Values**: Shapley values for local feature importance
- **Global Feature Importance**: Model-level feature rankings
- **Risk Factors**: Human-readable explanations for fraud predictions

### Performance Optimization
- **Hyperparameter Tuning**: Automated optimization using Optuna and GridSearchCV
- **Model Versioning**: Track model versions and performance over time
- **A/B Testing**: Support for model comparison and gradual rollout

## 📊 API Endpoints

### Core Fraud Detection

#### `POST /api/v1/detect`
Real-time fraud detection for individual transactions.

```json
{
  "transaction": {
    "user_id": "123e4567-e89b-12d3-a456-426614174000",
    "amount": 250.00,
    "currency": "USD",
    "transaction_type": "purchase",
    "merchant_name": "Amazon",
    "ip_address": "192.168.1.1",
    "country_code": "US"
  },
  "return_explanation": true
}
```

**Response:**
```json
{
  "transaction_id": "987fcdeb-51a2-4bc3-8d9e-87654321098f",
  "fraud_probability": 0.85,
  "is_fraud": true,
  "risk_level": "high",
  "confidence_score": 0.92,
  "model_version": "v1.2.0",
  "prediction_latency_ms": 45,
  "top_risk_factors": [
    {
      "feature": "amount_zscore",
      "importance": 0.34,
      "value": 3.2
    }
  ]
}
```

#### `POST /api/v1/batch-detect`
Batch fraud detection for multiple transactions (up to 100 per request).

#### `POST /api/v1/feedback`
Submit feedback to improve model accuracy.

```json
{
  "transaction_id": "987fcdeb-51a2-4bc3-8d9e-87654321098f",
  "feedback_type": "false_positive",
  "is_fraud_actual": false,
  "confidence": "high",
  "notes": "Legitimate large purchase verified with customer"
}
```

### Analytics & Reporting

#### `GET /api/v1/analytics/fraud-statistics`
Comprehensive fraud statistics for specified time periods.

#### `GET /api/v1/analytics/model-performance`
Model performance metrics including accuracy, precision, recall, F1-score.

#### `GET /api/v1/analytics/dashboard`
Complete analytics dashboard data with fraud trends and system health.

#### `GET /api/v1/analytics/alerts`
Recent high-risk transactions and fraud alerts.

### System Management

#### `GET /health`
Comprehensive health check including database, ML pipeline, and system status.

#### `GET /metrics`
Prometheus-compatible metrics for monitoring and alerting.

## 📈 Analytics Dashboard

The platform provides comprehensive analytics and reporting:

### Fraud Statistics
- **Transaction Volumes**: Total transactions and fraud counts over time
- **Fraud Rates**: Percentage of fraudulent transactions with trend analysis
- **Amount Analysis**: Total amounts processed and lost to fraud
- **Risk Distribution**: Breakdown by risk levels (low, medium, high, critical)

### Geographic Analysis
- **Country Risk**: Fraud rates by country with interactive maps
- **City Analysis**: Detailed city-level fraud statistics
- **IP Intelligence**: Suspicious IP addresses and network analysis

### Merchant Analysis
- **Top Fraud Merchants**: Merchants with highest fraud rates
- **Category Analysis**: Fraud patterns by merchant category
- **Volume vs. Risk**: Correlation between transaction volume and fraud risk

### Model Performance
- **Accuracy Metrics**: Precision, recall, F1-score, AUC-ROC over time
- **Confusion Matrix**: True/false positive and negative rates
- **Feature Importance**: Most predictive features for fraud detection
- **Model Comparison**: Performance comparison across different model versions

## 🐳 Deployment

### Docker Development

```bash
# Build and run with Docker Compose
docker-compose up --build

# Run specific services
docker-compose up db redis
docker-compose up api
```

### Kubernetes Production

```bash
# Deploy to Kubernetes
kubectl apply -f k8s/

# Check deployment status
kubectl get pods -l app=cipherguard

# View logs
kubectl logs -f deployment/cipherguard-api
```

### Helm Chart
```bash
# Install with Helm
helm install cipherguard ./helm/cipherguard

# Upgrade deployment
helm upgrade cipherguard ./helm/cipherguard
```

## 🧪 Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov-report=html

# Run specific test categories
pytest tests/unit/        # Unit tests
pytest tests/integration/ # Integration tests
pytest tests/performance/ # Performance tests
```

### Test Categories

- **Unit Tests**: Individual component testing with mocking
- **Integration Tests**: End-to-end API testing with test database
- **Performance Tests**: Load testing and benchmarking
- **Security Tests**: Authentication, authorization, and security validation

## 📊 Monitoring & Observability

### Metrics Collection
- **Application Metrics**: Request rates, response times, error rates
- **Business Metrics**: Fraud detection rates, model accuracy, alert volumes
- **Infrastructure Metrics**: CPU, memory, database, and network utilization

### Alerting
- **High Fraud Rate**: Alert when fraud rate exceeds threshold
- **Model Performance**: Alert on model accuracy degradation
- **System Health**: Alert on service downtime or errors
- **Security Events**: Alert on authentication failures or suspicious activity

### Dashboards
- **Operational Dashboard**: System health, performance, and error tracking
- **Business Dashboard**: Fraud statistics, trends, and financial impact
- **Security Dashboard**: Authentication events, rate limiting, and audit trails

## 🔧 Development

### Project Structure

```
cipherguard_enterprise/
├── app/
│   ├── api/v1/           # API endpoints
│   ├── core/            # Core configuration and utilities
│   ├── models/          # Database models
│   ├── repositories/    # Data access layer
│   ├── schemas/         # Pydantic schemas
│   └── main.py         # FastAPI application
├── ml/
│   ├── training/        # ML model training
│   ├── inference/       # ML inference pipeline
│   └── feature_engineering/ # Feature extraction
├── database/           # Database configuration
├── deployment/         # Docker and Kubernetes configs
├── tests/             # Test suite
├── docs/              # Documentation
└── scripts/           # Utility scripts
```

### Code Quality

```bash
# Format code
black app/ ml/ tests/
isort app/ ml/ tests/

# Lint code  
flake8 app/ ml/ tests/
mypy app/

# Security scanning
bandit -r app/
```

### Contributing

1. **Fork the repository** and create a feature branch
2. **Write tests** for new functionality
3. **Ensure code quality** with linting and formatting
4. **Update documentation** for API changes
5. **Submit a pull request** with detailed description

## 📚 Documentation

### API Documentation
- **Interactive Docs**: Available at `/docs` (Swagger UI)
- **ReDoc**: Available at `/redoc` (alternative API docs)
- **OpenAPI Spec**: Available at `/openapi.json`

### Architecture Documentation
- **System Design**: High-level architecture and component overview
- **Database Schema**: Complete ERD and table descriptions  
- **ML Pipeline**: Model architecture and feature engineering details
- **Deployment Guide**: Production deployment and scaling strategies

### Security Documentation
- **Authentication Guide**: JWT and API key implementation details
- **Security Best Practices**: Recommended security configurations
- **Audit Trail**: Logging and monitoring setup guide

## 🚨 Troubleshooting

### Common Issues

#### Database Connection
```bash
# Check database connectivity
python -c "from app.database import check_database_health; print(check_database_health())"

# Reset database
alembic downgrade base && alembic upgrade head
```

#### ML Pipeline Issues
```bash
# Check model status
python -c "from app.ml.inference.pipeline import fraud_detection_pipeline; print(fraud_detection_pipeline.is_trained)"

# Retrain models
python scripts/train_models.py
```

#### Performance Issues
```bash
# Monitor system resources
docker stats

# Check application metrics
curl http://localhost:8000/metrics
```

### Logs Analysis
```bash
# View application logs
docker-compose logs -f api

# Search for errors
docker-compose logs api | grep ERROR

# Filter by correlation ID
docker-compose logs api | grep "correlation_id=abc123"
```

## 📋 Roadmap

### Phase 1: Core Platform ✅
- [x] Advanced ML pipeline with ensemble models
- [x] Real-time fraud detection API
- [x] Enterprise authentication and security
- [x] Comprehensive analytics and reporting
- [x] Production-ready deployment

### Phase 2: Advanced Features
- [ ] Graph neural networks for network fraud detection
- [ ] Real-time streaming with Apache Kafka
- [ ] Advanced visualization and dashboards
- [ ] Multi-tenant architecture
- [ ] Mobile SDK for fraud detection

### Phase 3: Enterprise Integration  
- [ ] Enterprise SSO integration (SAML, OAuth2)
- [ ] Advanced compliance features (PCI DSS, GDPR)
- [ ] White-label solution capabilities
- [ ] Advanced ML model management (MLOps)
- [ ] Horizontal auto-scaling

## 📄 License

This project is proprietary software owned by CipherGuard Security Solutions. 

**All rights reserved.** Unauthorized copying, distribution, or modification of this software is strictly prohibited.

For licensing inquiries, contact: licensing@cipherguard.com

## 🤝 Support

### Enterprise Support
- **Email**: support@cipherguard.com
- **Phone**: +1 (555) 123-FRAUD
- **Documentation**: https://docs.cipherguard.com
- **Status Page**: https://status.cipherguard.com

### Community
- **GitHub Issues**: [Report bugs and feature requests](https://github.com/cipherguard/fraud-detection/issues)
- **Discussions**: [Community discussions and Q&A](https://github.com/cipherguard/fraud-detection/discussions)
- **Security**: security@cipherguard.com (for security vulnerabilities)

---

**CipherGuard Enterprise Fraud Detection Platform** - Protecting your business from fraud with advanced AI and enterprise security.

Built with ❤️ by the CipherGuard Security Team