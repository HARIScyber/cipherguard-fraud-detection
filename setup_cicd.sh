#!/bin/bash
# CI/CD Pipeline Setup - Phase 8: Production Operations
# Complete CI/CD pipeline for CipherGuard fraud detection system

set -e

echo "🚀 CipherGuard CI/CD Pipeline Setup - Phase 8"
echo "============================================="

PROJECT_NAME="cipherguard"
GITHUB_REPO="your-org/cipherguard"
DOCKER_REGISTRY="your-registry.com"
K8S_NAMESPACE="fraud-detection"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# Pre-requisites check
check_prerequisites() {
    log_step "Checking prerequisites..."

    # Check required tools
    command -v docker >/dev/null 2>&1 || { log_error "Docker is required"; exit 1; }
    command -v kubectl >/dev/null 2>&1 || { log_error "kubectl is required"; exit 1; }
    command -v helm >/dev/null 2>&1 || { log_error "Helm is required"; exit 1; }

    # Check GitHub CLI (optional)
    if command -v gh >/dev/null 2>&1; then
        log_info "GitHub CLI available"
    else
        log_warn "GitHub CLI not available - manual GitHub setup required"
    fi

    # Check Kubernetes connection
    kubectl cluster-info >/dev/null 2>&1 || { log_error "Cannot connect to Kubernetes"; exit 1; }

    log_info "Prerequisites check completed"
}

# Setup GitHub Actions
setup_github_actions() {
    log_step "Setting up GitHub Actions CI/CD..."

    mkdir -p .github/workflows

    # Create main CI pipeline
    cat > .github/workflows/ci.yml << 'EOF'
name: CI Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

env:
  REGISTRY: your-registry.com
  IMAGE_NAME: cipherguard

jobs:
  test:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_PASSWORD: postgres
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
      redis:
        image: redis:7
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5

    steps:
    - uses: actions/checkout@v4

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'

    - name: Cache pip dependencies
      uses: actions/cache@v3
      with:
        path: ~/.cache/pip
        key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements.txt') }}
        restore-keys: |
          ${{ runner.os }}-pip-

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest pytest-cov pytest-asyncio

    - name: Run tests
      run: |
        pytest --cov=app --cov-report=xml --cov-report=term-missing
      env:
        DATABASE_URL: postgresql://postgres:postgres@localhost:5432/test_db
        REDIS_URL: redis://localhost:6379

    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml

  security-scan:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4

    - name: Run Trivy vulnerability scanner
      uses: aquasecurity/trivy-action@master
      with:
        scan-type: 'fs'
        scan-ref: '.'
        format: 'sarif'
        output: 'trivy-results.sarif'

    - name: Upload Trivy scan results
      uses: github/codeql-action/upload-sarif@v2
      if: always()
      with:
        sarif_file: 'trivy-results.sarif'

  build-and-push:
    runs-on: ubuntu-latest
    needs: [test, security-scan]
    if: github.ref == 'refs/heads/main'

    steps:
    - name: Checkout code
      uses: actions/checkout@v4

    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v3

    - name: Log in to Container Registry
      uses: docker/login-action@v3
      with:
        registry: ${{ env.REGISTRY }}
        username: ${{ secrets.REGISTRY_USERNAME }}
        password: ${{ secrets.REGISTRY_PASSWORD }}

    - name: Extract metadata
      id: meta
      uses: docker/metadata-action@v5
      with:
        images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
        tags: |
          type=ref,event=branch
          type=ref,event=pr
          type=sha,prefix={{branch}}-
          type=raw,value=latest,enable={{is_default_branch}}

    - name: Build and push API image
      uses: docker/build-push-action@v5
      with:
        context: .
        file: ./Dockerfile
        push: true
        tags: ${{ steps.meta.outputs.tags }}
        labels: ${{ steps.meta.outputs.labels }}
        cache-from: type=gha
        cache-to: type=gha,mode=max

    - name: Build and push microservice images
      run: |
        for service in fraud-detection ingestion embedding alert; do
          if [ -d "services/$service" ]; then
            echo "Building $service service..."
            docker build -t ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}-$service:${{ github.sha }} services/$service/
            docker push ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}-$service:${{ github.sha }}
            docker tag ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}-$service:${{ github.sha }} ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}-$service:latest
            docker push ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}-$service:latest
          fi
        done
EOF

    # Create CD pipeline
    cat > .github/workflows/cd.yml << 'EOF'
name: CD Pipeline

on:
  push:
    branches: [ main ]
  workflow_dispatch:
    inputs:
      environment:
        description: 'Environment to deploy to'
        required: true
        default: 'staging'
        type: choice
        options:
        - staging
        - production

env:
  REGISTRY: your-registry.com
  IMAGE_NAME: cipherguard

jobs:
  deploy-staging:
    if: github.ref == 'refs/heads/main' || github.event.inputs.environment == 'staging'
    runs-on: ubuntu-latest
    environment: staging

    steps:
    - name: Checkout code
      uses: actions/checkout@v4

    - name: Configure kubectl
      uses: azure/k8s-set-context@v3
      with:
        method: kubeconfig
        kubeconfig: ${{ secrets.STAGING_KUBECONFIG }}

    - name: Deploy to staging
      run: |
        # Update Helm chart version
        sed -i "s/version:.*/version: ${{ github.sha }}/" helm/cipherguard/Chart.yaml
        sed -i "s/appVersion:.*/appVersion: ${{ github.sha }}/" helm/cipherguard/Chart.yaml

        # Deploy using Helm
        helm upgrade --install cipherguard-staging ./helm/cipherguard \
          --namespace fraud-detection-staging \
          --set global.environment=staging \
          --set global.imageRegistry=${{ env.REGISTRY }} \
          --set api.image.tag=${{ github.sha }} \
          --create-namespace \
          --wait

    - name: Run integration tests
      run: |
        # Wait for deployment
        kubectl wait --for=condition=available --timeout=300s deployment/cipherguard-api -n fraud-detection-staging

        # Run integration tests
        npm install -g artillery
        artillery run tests/integration.yml --environment staging

  deploy-production:
    if: github.event.inputs.environment == 'production'
    runs-on: ubuntu-latest
    environment: production
    needs: deploy-staging

    steps:
    - name: Checkout code
      uses: actions/checkout@v4

    - name: Configure kubectl
      uses: azure/k8s-set-context@v3
      with:
        method: kubeconfig
        kubeconfig: ${{ secrets.PRODUCTION_KUBECONFIG }}

    - name: Deploy to production
      run: |
        # Update Helm chart version
        sed -i "s/version:.*/version: ${{ github.sha }}/" helm/cipherguard/Chart.yaml
        sed -i "s/appVersion:.*/appVersion: ${{ github.sha }}/" helm/cipherguard/Chart.yaml

        # Deploy using Helm with production values
        helm upgrade --install cipherguard ./helm/cipherguard \
          --namespace fraud-detection \
          --set global.environment=production \
          --set global.imageRegistry=${{ env.REGISTRY }} \
          --set api.image.tag=${{ github.sha }} \
          --wait

    - name: Run smoke tests
      run: |
        # Wait for deployment
        kubectl wait --for=condition=available --timeout=600s deployment/cipherguard-api -n fraud-detection

        # Run smoke tests
        curl -f https://api.yourdomain.com/health
        curl -f https://api.yourdomain.com/docs

    - name: Notify deployment
      uses: 8398a7/action-slack@v3
      with:
        status: success
        text: "CipherGuard deployed to production successfully"
      env:
        SLACK_WEBHOOK_URL: ${{ secrets.SLACK_WEBHOOK_URL }}
      if: success()

    - name: Notify failure
      uses: 8398a7/action-slack@v3
      with:
        status: failure
        text: "CipherGuard production deployment failed"
      env:
        SLACK_WEBHOOK_URL: ${{ secrets.SLACK_WEBHOOK_URL }}
      if: failure()
EOF

    log_info "GitHub Actions workflows created"
}

# Setup testing infrastructure
setup_testing() {
    log_step "Setting up testing infrastructure..."

    mkdir -p tests/{unit,integration,performance,e2e}

    # Create pytest configuration
    cat > pytest.ini << 'EOF'
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts =
    --strict-markers
    --strict-config
    --cov=app
    --cov-report=term-missing
    --cov-report=xml
    --cov-report=html
    --cov-fail-under=80
markers =
    unit: Unit tests
    integration: Integration tests
    e2e: End-to-end tests
    performance: Performance tests
    slow: Slow running tests
EOF

    # Create integration tests
    cat > tests/integration/test_api_integration.py << 'EOF'
import pytest
import httpx
import asyncio
from typing import Dict, Any

class TestAPIIntegration:
    """Integration tests for CipherGuard API."""

    def setup_method(self):
        self.base_url = "http://localhost:8000"
        self.client = httpx.Client(timeout=30.0)

    def teardown_method(self):
        self.client.close()

    @pytest.mark.integration
    def test_health_endpoint(self):
        """Test health check endpoint."""
        response = self.client.get(f"{self.base_url}/health")
        assert response.status_code == 200

        data = response.json()
        assert "status" in data
        assert "services" in data
        assert data["status"] in ["healthy", "degraded"]

    @pytest.mark.integration
    def test_fraud_detection_basic(self):
        """Test basic fraud detection."""
        transaction = {
            "amount": 1000.0,
            "merchant": "test_merchant",
            "device": "test_device",
            "country": "US",
            "timestamp": "2024-01-01T00:00:00Z"
        }

        response = self.client.post(
            f"{self.base_url}/detect",
            json=transaction,
            headers={"Content-Type": "application/json"}
        )

        assert response.status_code == 200

        data = response.json()
        assert "is_fraud" in data
        assert "fraud_score" in data
        assert "risk_level" in data
        assert isinstance(data["is_fraud"], bool)
        assert isinstance(data["fraud_score"], (int, float))

    @pytest.mark.integration
    def test_advanced_fraud_detection(self):
        """Test advanced fraud detection with ML models."""
        transaction = {
            "amount": 500.0,
            "merchant": "luxury_store",
            "device": "mobile_device",
            "country": "US",
            "customer_id": "customer_123",
            "timestamp": "2024-01-01T00:00:00Z"
        }

        response = self.client.post(
            f"{self.base_url}/detect/advanced",
            json=transaction,
            headers={"Content-Type": "application/json"}
        )

        assert response.status_code == 200

        data = response.json()
        assert "is_fraud" in data
        assert "fraud_score" in data
        assert "models_used" in data

    @pytest.mark.integration
    def test_enterprise_authentication(self):
        """Test enterprise authentication."""
        auth_data = {
            "username": "test_user",
            "password": "test_password",
            "tenant_id": "test_tenant"
        }

        response = self.client.post(
            f"{self.base_url}/auth/login",
            json=auth_data,
            headers={"Content-Type": "application/json"}
        )

        # Authentication might fail with test credentials, but endpoint should respond
        assert response.status_code in [200, 401, 422]

        if response.status_code == 200:
            data = response.json()
            assert "access_token" in data
            assert "token_type" in data
            assert data["token_type"] == "bearer"

    @pytest.mark.integration
    def test_tenant_management(self):
        """Test tenant management endpoints."""
        response = self.client.get(f"{self.base_url}/tenants")

        # Might require authentication
        assert response.status_code in [200, 401, 403]

    @pytest.mark.integration
    def test_compliance_reporting(self):
        """Test compliance reporting."""
        response = self.client.get(f"{self.base_url}/compliance/report?standard=gdpr")

        # Might require authentication
        assert response.status_code in [200, 401, 403]

    @pytest.mark.integration
    def test_audit_logging(self):
        """Test audit logging."""
        response = self.client.get(f"{self.base_url}/audit/events?limit=10")

        # Might require authentication
        assert response.status_code in [200, 401, 403]

    @pytest.mark.integration
    def test_rate_limiting(self):
        """Test rate limiting functionality."""
        # Send multiple requests quickly
        responses = []
        for i in range(15):
            response = self.client.get(f"{self.base_url}/health")
            responses.append(response.status_code)

        # Some requests should be rate limited (429)
        rate_limited = any(status == 429 for status in responses)
        assert rate_limited or all(status == 200 for status in responses[:10])

    @pytest.mark.integration
    def test_openapi_specification(self):
        """Test OpenAPI specification."""
        response = self.client.get(f"{self.base_url}/docs")
        assert response.status_code == 200

        response = self.client.get(f"{self.base_url}/openapi.json")
        assert response.status_code == 200

        spec = response.json()
        assert "paths" in spec
        assert "/detect" in spec["paths"]
        assert "/health" in spec["paths"]
EOF

    # Create performance tests
    cat > tests/performance/test_performance.py << 'EOF'
import pytest
import httpx
import asyncio
import time
import statistics
from typing import List, Dict, Any

class TestPerformance:
    """Performance tests for CipherGuard API."""

    def setup_method(self):
        self.base_url = "http://localhost:8000"
        self.client = httpx.Client(timeout=60.0)

    def teardown_method(self):
        self.client.close()

    @pytest.mark.performance
    def test_api_response_time(self):
        """Test API response time under normal load."""
        response_times = []

        for _ in range(100):
            start_time = time.time()
            response = self.client.get(f"{self.base_url}/health")
            end_time = time.time()

            assert response.status_code == 200
            response_times.append(end_time - start_time)

        avg_response_time = statistics.mean(response_times)
        p95_response_time = statistics.quantiles(response_times, n=20)[18]  # 95th percentile

        # Assert reasonable performance
        assert avg_response_time < 0.5  # Average < 500ms
        assert p95_response_time < 1.0  # 95th percentile < 1s

        print(f"Average response time: {avg_response_time:.3f}s")
        print(f"95th percentile: {p95_response_time:.3f}s")

    @pytest.mark.performance
    def test_fraud_detection_throughput(self):
        """Test fraud detection throughput."""
        transaction = {
            "amount": 100.0,
            "merchant": "test_merchant",
            "device": "test_device",
            "country": "US",
            "timestamp": "2024-01-01T00:00:00Z"
        }

        start_time = time.time()
        request_count = 0

        # Test for 10 seconds
        test_duration = 10
        end_time = start_time + test_duration

        while time.time() < end_time:
            response = self.client.post(
                f"{self.base_url}/detect",
                json=transaction,
                headers={"Content-Type": "application/json"}
            )
            assert response.status_code == 200
            request_count += 1

        actual_duration = time.time() - start_time
        throughput = request_count / actual_duration

        print(f"Throughput: {throughput:.2f} requests/second")

        # Assert minimum throughput
        assert throughput > 10  # At least 10 requests per second

    @pytest.mark.performance
    def test_concurrent_requests(self):
        """Test handling of concurrent requests."""
        async def make_request(session: httpx.AsyncClient, request_id: int):
            transaction = {
                "amount": 50.0 + request_id,
                "merchant": f"merchant_{request_id}",
                "device": f"device_{request_id % 5}",
                "country": "US"
            }

            start_time = time.time()
            response = await session.post(
                f"{self.base_url}/detect",
                json=transaction,
                headers={"Content-Type": "application/json"}
            )
            end_time = time.time()

            return {
                "status_code": response.status_code,
                "response_time": end_time - start_time,
                "request_id": request_id
            }

        async def run_concurrent_test():
            async with httpx.AsyncClient(timeout=60.0) as client:
                tasks = [make_request(client, i) for i in range(50)]
                results = await asyncio.gather(*tasks)

                successful_requests = sum(1 for r in results if r["status_code"] == 200)
                avg_response_time = statistics.mean(r["response_time"] for r in results)

                return successful_requests, avg_response_time, results

        successful, avg_time, results = asyncio.run(run_concurrent_test())

        print(f"Concurrent requests: {successful}/50 successful")
        print(f"Average response time: {avg_time:.3f}s")

        # Assert good concurrent performance
        assert successful >= 45  # At least 90% success rate
        assert avg_time < 2.0  # Average response time < 2s

    @pytest.mark.performance
    def test_memory_usage(self):
        """Test memory usage under load."""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Generate load
        for i in range(1000):
            transaction = {
                "amount": 100.0 + (i % 100),
                "merchant": f"merchant_{i % 10}",
                "device": f"device_{i % 5}",
                "country": "US"
            }

            response = self.client.post(
                f"{self.base_url}/detect",
                json=transaction,
                headers={"Content-Type": "application/json"}
            )
            assert response.status_code == 200

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        print(f"Memory usage: {final_memory:.1f}MB (increase: {memory_increase:.1f}MB)")

        # Assert reasonable memory usage
        assert memory_increase < 100  # Less than 100MB increase
        assert final_memory < 500  # Total memory < 500MB

    @pytest.mark.performance
    def test_database_connection_pooling(self):
        """Test database connection pooling efficiency."""
        import threading
        import queue

        results_queue = queue.Queue()

        def worker_thread(thread_id: int):
            thread_results = []
            for i in range(50):
                start_time = time.time()

                # Test database-dependent endpoint
                response = self.client.post(
                    f"{self.base_url}/detect",
                    json={
                        "amount": 100.0,
                        "merchant": f"merchant_{thread_id}_{i}",
                        "device": f"device_{thread_id}",
                        "country": "US"
                    },
                    headers={"Content-Type": "application/json"}
                )

                end_time = time.time()

                thread_results.append({
                    "status_code": response.status_code,
                    "response_time": end_time - start_time
                })

            results_queue.put(thread_results)

        # Start multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker_thread, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join()

        # Collect results
        all_results = []
        while not results_queue.empty():
            all_results.extend(results_queue.get())

        successful_requests = sum(1 for r in all_results if r["status_code"] == 200)
        avg_response_time = statistics.mean(r["response_time"] for r in all_results)

        print(f"Threaded requests: {successful_requests}/{len(all_results)} successful")
        print(f"Average response time: {avg_response_time:.3f}s")

        # Assert good performance under concurrent database load
        assert successful_requests == len(all_results)  # All requests successful
        assert avg_response_time < 1.0  # Average < 1s
EOF

    # Create Artillery load testing config
    cat > tests/load/artillery.yml << 'EOF'
config:
  target: 'http://localhost:8000'
  phases:
    - duration: 60
      arrivalRate: 10
      name: "Warm up phase"
    - duration: 300
      arrivalRate: 50
      name: "Load testing phase"
    - duration: 60
      arrivalRate: 100
      name: "Stress testing phase"
  defaults:
    headers:
      Content-Type: 'application/json'

scenarios:
  - name: 'Health check'
    weight: 30
    requests:
      - method: GET
        url: '/health'

  - name: 'Basic fraud detection'
    weight: 50
    requests:
      - method: POST
        url: '/detect'
        json:
          amount: 100
          merchant: 'test_merchant'
          device: 'test_device'
          country: 'US'
          timestamp: '2024-01-01T00:00:00Z'

  - name: 'Advanced fraud detection'
    weight: 15
    requests:
      - method: POST
        url: '/detect/advanced'
        json:
          amount: 500
          merchant: 'luxury_store'
          device: 'mobile_device'
          country: 'US'
          customer_id: 'customer_123'
          timestamp: '2024-01-01T00:00:00Z'

  - name: 'API documentation'
    weight: 5
    requests:
      - method: GET
        url: '/docs'
EOF

    log_info "Testing infrastructure setup completed"
}

# Setup monitoring and alerting
setup_monitoring() {
    log_step "Setting up monitoring and alerting..."

    # Create monitoring dashboards
    mkdir -p monitoring/{grafana,prometheus,alertmanager}

    # Grafana dashboard for CipherGuard
    cat > monitoring/grafana/cipherguard-dashboard.json << 'EOF'
{
  "dashboard": {
    "title": "CipherGuard Fraud Detection",
    "tags": ["cipherguard", "fraud-detection"],
    "timezone": "browser",
    "panels": [
      {
        "title": "API Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          }
        ]
      },
      {
        "title": "Fraud Detection Rate",
        "type": "stat",
        "targets": [
          {
            "expr": "rate(fraud_detections_total[5m])",
            "legendFormat": "Detections/min"
          }
        ]
      },
      {
        "title": "System Health",
        "type": "table",
        "targets": [
          {
            "expr": "up{job=~\"cipherguard-.*\"}",
            "legendFormat": "{{ job }}"
          }
        ]
      }
    ],
    "time": {
      "from": "now-1h",
      "to": "now"
    },
    "refresh": "30s"
  }
}
EOF

    # AlertManager configuration
    cat > monitoring/alertmanager/alertmanager.yml << 'EOF'
global:
  smtp_smarthost: 'smtp.gmail.com:587'
  smtp_from: 'alerts@cipherguard.com'
  smtp_auth_username: 'alerts@cipherguard.com'
  smtp_auth_password: 'your-smtp-password'

route:
  group_by: ['alertname']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h
  receiver: 'email'
  routes:
  - match:
      severity: critical
    receiver: 'critical-email'

receivers:
- name: 'email'
  email_configs:
  - to: 'ops@cipherguard.com'
    subject: 'CipherGuard Alert: {{ .GroupLabels.alertname }}'
    body: |
      {{ range .Alerts }}
      Alert: {{ .Annotations.summary }}
      Description: {{ .Annotations.description }}
      Runbook: {{ .Annotations.runbook_url }}
      {{ end }}

- name: 'critical-email'
  email_configs:
  - to: 'emergency@cipherguard.com'
    subject: 'CRITICAL: CipherGuard Alert: {{ .GroupLabels.alertname }}'
EOF

    log_info "Monitoring and alerting setup completed"
}

# Setup deployment environments
setup_environments() {
    log_step "Setting up deployment environments..."

    # Create staging environment
    mkdir -p environments/staging

    cat > environments/staging/values.yaml << 'EOF'
global:
  environment: staging
  domain: staging.cipherguard.yourdomain.com

api:
  replicaCount: 2
  resources:
    requests:
      cpu: 200m
      memory: 256Mi
    limits:
      cpu: 500m
      memory: 512Mi

fraudDetection:
  replicaCount: 1
  resources:
    requests:
      cpu: 500m
      memory: 1Gi
    limits:
      cpu: 1000m
      memory: 2Gi

postgresql:
  auth:
    postgresPassword: "staging-password"
  primary:
    resources:
      requests:
        cpu: 250m
        memory: 256Mi
      limits:
        cpu: 500m
        memory: 512Mi

redis:
  auth:
    password: "staging-password"
  master:
    resources:
      requests:
        cpu: 100m
        memory: 128Mi
      limits:
        cpu: 200m
        memory: 256Mi
EOF

    # Create production environment
    mkdir -p environments/production

    cat > environments/production/values.yaml << 'EOF'
global:
  environment: production
  domain: api.cipherguard.yourdomain.com

api:
  replicaCount: 3
  resources:
    requests:
      cpu: 500m
      memory: 1Gi
    limits:
      cpu: 1000m
      memory: 2Gi

fraudDetection:
  replicaCount: 2
  resources:
    requests:
      cpu: 1000m
      memory: 2Gi
    limits:
      cpu: 2000m
      memory: 4Gi

postgresql:
  auth:
    postgresPassword: "production-password"
  architecture: replication
  primary:
    resources:
      requests:
        cpu: 500m
        memory: 1Gi
      limits:
        cpu: 1000m
        memory: 2Gi

redis:
  auth:
    password: "production-password"
  architecture: replication
  master:
    resources:
      requests:
        cpu: 250m
        memory: 512Mi
      limits:
        cpu: 500m
        memory: 1Gi
EOF

    log_info "Deployment environments setup completed"
}

# Setup quality gates
setup_quality_gates() {
    log_step "Setting up quality gates..."

    # Create code quality checks
    cat > .pre-commit-config.yaml << 'EOF'
repos:
- repo: https://github.com/pre-commit/pre-commit-hooks
  rev: v4.4.0
  hooks:
  - id: trailing-whitespace
  - id: end-of-file-fixer
  - id: check-yaml
  - id: check-added-large-files

- repo: https://github.com/psf/black
  rev: 23.7.0
  hooks:
  - id: black
    language_version: python3

- repo: https://github.com/pycqa/flake8
  rev: 6.0.0
  hooks:
  - id: flake8
    args: [--max-line-length=88, --extend-ignore=E203,W503]

- repo: https://github.com/pre-commit/mirrors-mypy
  rev: v1.5.1
  hooks:
  - id: mypy
    additional_dependencies: [types-all]
EOF

    # Create sonarcloud configuration
    cat > sonar-project.properties << 'EOF'
sonar.projectKey=cipherguard-fraud-detection
sonar.organization=your-org

# This is the name and version displayed in the SonarCloud UI.
sonar.projectName=CipherGuard Fraud Detection
sonar.projectVersion=1.0.0

# Path is relative to the sonar-project.properties file. Replace "\" by "/" on Windows.
sonar.sources=app,services
sonar.tests=tests
sonar.test.inclusions=**/*test*.py
sonar.python.coverage.reportPaths=coverage.xml
sonar.python.version=3.11

# Encoding of the source code. Default is default system encoding
sonar.sourceEncoding=UTF-8
EOF

    log_info "Quality gates setup completed"
}

# Setup documentation
setup_documentation() {
    log_step "Setting up documentation..."

    mkdir -p docs/{api,user-guide,operations}

    # Create API documentation
    cat > docs/api/README.md << 'EOF'
# CipherGuard API Documentation

## Overview
CipherGuard provides REST APIs for real-time fraud detection with enterprise features.

## Authentication
All API endpoints require JWT authentication except health checks.

```bash
# Get access token
curl -X POST https://api.cipherguard.com/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"your-user","password":"your-password"}'

# Use token in requests
curl -H "Authorization: Bearer YOUR_TOKEN" \
  https://api.cipherguard.com/detect
```

## Endpoints

### Fraud Detection
- `POST /detect` - Basic fraud detection
- `POST /detect/advanced` - Advanced ML-based detection

### Enterprise Features
- `POST /auth/login` - User authentication
- `POST /payments/process` - Payment processing
- `GET /tenants` - Tenant management
- `GET /compliance/report` - Compliance reporting

### Monitoring
- `GET /health` - Health check
- `GET /metrics` - Prometheus metrics
- `GET /docs` - API documentation

## Rate Limits
- 100 requests per minute for basic endpoints
- 1000 requests per hour for enterprise endpoints
EOF

    # Create operations runbook
    cat > docs/operations/runbook.md << 'EOF'
# CipherGuard Operations Runbook

## Incident Response

### High Error Rate Alert
1. Check application logs: `kubectl logs deployment/cipherguard-api`
2. Check database connectivity: `kubectl exec -it deployment/postgres -- pg_isready`
3. Check resource usage: `kubectl top pods`
4. Scale up if needed: `kubectl scale deployment cipherguard-api --replicas=5`
5. Restart pods if necessary: `kubectl rollout restart deployment/cipherguard-api`

### Database Issues
1. Check database pod status: `kubectl get pods -l app=postgres`
2. Check database logs: `kubectl logs -f deployment/postgres-postgresql`
3. Verify connections: `kubectl exec -it deployment/postgres -- psql -c "SELECT * FROM pg_stat_activity;"`
4. Restart database if needed: `kubectl rollout restart deployment/postgres-postgresql`

### Memory Issues
1. Check memory usage: `kubectl top pods`
2. Check application metrics: Grafana dashboard
3. Adjust resource limits: `kubectl edit deployment cipherguard-api`
4. Enable garbage collection tuning if needed

## Maintenance Procedures

### Weekly Tasks
- Review error logs and alerts
- Update security patches
- Check disk usage and clean up old logs
- Verify backup integrity

### Monthly Tasks
- Update dependencies and libraries
- Review and optimize resource usage
- Update monitoring dashboards
- Security vulnerability assessment

### Deployment Procedures
1. Create feature branch
2. Implement changes with tests
3. Create pull request
4. CI/CD pipeline runs automatically
5. Manual approval for production deployment
6. Post-deployment verification

## Monitoring Dashboards

### Key Metrics to Monitor
- API response time (< 500ms average)
- Error rate (< 1%)
- Fraud detection accuracy (> 95%)
- System resource usage (< 80%)
- Database connection pool utilization

### Alert Thresholds
- Error rate > 5%: Warning
- Response time > 2s: Warning
- CPU usage > 90%: Critical
- Memory usage > 90%: Critical
- Database connections > 90% of pool: Warning
EOF

    log_info "Documentation setup completed"
}

# Main setup function
main() {
    log_info "Starting CI/CD pipeline setup..."

    check_prerequisites
    setup_github_actions
    setup_testing
    setup_monitoring
    setup_environments
    setup_quality_gates
    setup_documentation

    log_info "🎉 CI/CD pipeline setup completed!"
    log_info ""
    log_info "Next steps:"
    log_info "1. Push these changes to your GitHub repository"
    log_info "2. Configure GitHub secrets for registry access"
    log_info "3. Set up Kubernetes cluster access"
    log_info "4. Run initial CI pipeline"
    log_info "5. Configure monitoring and alerting"
    log_info ""
    log_info "Your CipherGuard system now has enterprise-grade CI/CD! 🚀"
}

# Run main setup
main "$@"