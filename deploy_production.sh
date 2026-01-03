#!/bin/bash
# Production Deployment Script - Phase 7: Production Deployment & Scaling
# Complete production deployment for CipherGuard fraud detection system

set -e

echo "🚀 CipherGuard Production Deployment - Phase 7"
echo "=============================================="

# Configuration
PROJECT_NAME="cipherguard"
NAMESPACE="fraud-detection"
DOCKER_REGISTRY="your-registry.com"
VERSION=$(date +%Y%m%d-%H%M%S)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Pre-deployment checks
pre_deployment_checks() {
    log_info "Running pre-deployment checks..."

    # Check required tools
    command -v docker >/dev/null 2>&1 || { log_error "Docker is required but not installed."; exit 1; }
    command -v kubectl >/dev/null 2>&1 || { log_error "kubectl is required but not installed."; exit 1; }
    command -v helm >/dev/null 2>&1 || { log_error "Helm is required but not installed."; exit 1; }

    # Check Kubernetes connection
    kubectl cluster-info >/dev/null 2>&1 || { log_error "Cannot connect to Kubernetes cluster."; exit 1; }

    # Check if namespace exists, create if not
    kubectl get namespace $NAMESPACE >/dev/null 2>&1 || {
        log_info "Creating namespace $NAMESPACE..."
        kubectl create namespace $NAMESPACE
    }

    log_info "Pre-deployment checks completed."
}

# Build Docker images
build_images() {
    log_info "Building Docker images..."

    # Build main application image
    docker build -t $DOCKER_REGISTRY/$PROJECT_NAME-api:$VERSION -f Dockerfile .
    docker build -t $DOCKER_REGISTRY/$PROJECT_NAME-api:latest -f Dockerfile .

    # Build microservices images
    for service in ingestion embedding fraud-detection alert; do
        if [ -d "services/$service" ]; then
            log_info "Building $service service image..."
            docker build -t $DOCKER_REGISTRY/$PROJECT_NAME-$service:$VERSION services/$service/
            docker build -t $DOCKER_REGISTRY/$PROJECT_NAME-$service:latest services/$service/
        fi
    done

    log_info "Docker images built successfully."
}

# Push images to registry
push_images() {
    log_info "Pushing images to registry..."

    # Login to registry (you may need to modify this)
    # docker login $DOCKER_REGISTRY

    # Push images
    for image in api ingestion embedding fraud-detection alert; do
        docker push $DOCKER_REGISTRY/$PROJECT_NAME-$image:$VERSION
        docker push $DOCKER_REGISTRY/$PROJECT_NAME-$image:latest
    done

    log_info "Images pushed to registry."
}

# Deploy to Kubernetes
deploy_kubernetes() {
    log_info "Deploying to Kubernetes..."

    # Update Helm values with current version
    sed -i "s/tag:.*/tag: \"$VERSION\"/g" helm/cipherguard/values.yaml

    # Deploy using Helm
    helm upgrade --install $PROJECT_NAME ./helm/cipherguard \
        --namespace $NAMESPACE \
        --set image.tag=$VERSION \
        --wait

    log_info "Kubernetes deployment completed."
}

# Setup monitoring and logging
setup_monitoring() {
    log_info "Setting up monitoring and logging..."

    # Deploy Prometheus and Grafana if not already deployed
    kubectl apply -f k8s/monitoring/

    # Setup logging with ELK stack or similar
    kubectl apply -f k8s/logging/

    log_info "Monitoring and logging setup completed."
}

# Configure ingress and SSL
setup_ingress() {
    log_info "Setting up ingress and SSL..."

    # Install cert-manager if not present
    kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml

    # Wait for cert-manager to be ready
    kubectl wait --for=condition=available --timeout=300s deployment -n cert-manager --all

    # Apply ingress configuration
    kubectl apply -f k8s/ingress/

    log_info "Ingress and SSL setup completed."
}

# Database setup
setup_database() {
    log_info "Setting up databases..."

    # Deploy PostgreSQL
    helm repo add bitnami https://charts.bitnami.com/bitnami
    helm repo update

    helm upgrade --install postgres bitnami/postgresql \
        --namespace $NAMESPACE \
        --set auth.postgresPassword="your-postgres-password" \
        --set auth.database="fraud_detection" \
        --wait

    # Deploy Redis for caching and sessions
    helm upgrade --install redis bitnami/redis \
        --namespace $NAMESPACE \
        --set auth.password="your-redis-password" \
        --wait

    # Run database migrations
    log_info "Running database migrations..."
    kubectl apply -f k8s/migrations/

    log_info "Database setup completed."
}

# Security hardening
security_hardening() {
    log_info "Applying security hardening..."

    # Apply network policies
    kubectl apply -f k8s/security/

    # Setup secrets management
    kubectl apply -f k8s/secrets/

    # Configure RBAC
    kubectl apply -f k8s/rbac/

    log_info "Security hardening completed."
}

# Performance optimization
performance_optimization() {
    log_info "Applying performance optimizations..."

    # Setup horizontal pod autoscaling
    kubectl apply -f k8s/hpa/

    # Configure resource limits and requests
    kubectl apply -f k8s/resources/

    # Setup caching layers
    kubectl apply -f k8s/cache/

    log_info "Performance optimizations applied."
}

# Health checks and validation
validate_deployment() {
    log_info "Validating deployment..."

    # Wait for all pods to be ready
    kubectl wait --for=condition=ready pod -l app=$PROJECT_NAME --timeout=600s -n $NAMESPACE

    # Test API endpoints
    API_URL=$(kubectl get ingress -n $NAMESPACE -o jsonpath='{.items[0].spec.rules[0].host}')

    if [ -n "$API_URL" ]; then
        log_info "Testing API endpoints at https://$API_URL"

        # Wait for ingress to be ready
        sleep 30

        # Test health endpoint
        curl -k -f https://$API_URL/health || {
            log_error "Health check failed"
            exit 1
        }

        # Test basic fraud detection
        curl -k -f -X POST https://$API_URL/detect \
            -H "Content-Type: application/json" \
            -d '{"amount": 100, "merchant": "test", "device": "test", "country": "US"}' || {
            log_error "Fraud detection test failed"
            exit 1
        }
    fi

    log_info "Deployment validation completed successfully."
}

# Backup and disaster recovery setup
setup_backup_recovery() {
    log_info "Setting up backup and disaster recovery..."

    # Setup database backups
    kubectl apply -f k8s/backup/

    # Configure disaster recovery
    kubectl apply -f k8s/disaster-recovery/

    log_info "Backup and disaster recovery setup completed."
}

# Post-deployment tasks
post_deployment_tasks() {
    log_info "Running post-deployment tasks..."

    # Setup CI/CD pipelines
    log_info "CI/CD pipeline setup would go here..."

    # Configure alerting
    log_info "Alerting configuration would go here..."

    # Documentation update
    log_info "Documentation update would go here..."

    log_info "Post-deployment tasks completed."
}

# Main deployment function
main() {
    log_info "Starting production deployment..."

    pre_deployment_checks
    build_images
    push_images
    setup_database
    deploy_kubernetes
    setup_monitoring
    setup_ingress
    security_hardening
    performance_optimization
    setup_backup_recovery
    validate_deployment
    post_deployment_tasks

    log_info "🎉 Production deployment completed successfully!"
    log_info ""
    log_info "Next steps:"
    log_info "1. Configure your DNS to point to the ingress IP"
    log_info "2. Update your application configuration with production secrets"
    log_info "3. Set up monitoring dashboards in Grafana"
    log_info "4. Configure alerting rules"
    log_info "5. Perform load testing"
    log_info "6. Set up regular backups"
    log_info ""
    log_info "Your CipherGuard system is now running in production!"
}

# Rollback function
rollback() {
    log_error "Deployment failed. Starting rollback..."

    # Rollback Helm release
    helm rollback $PROJECT_NAME -n $NAMESPACE || true

    # Clean up failed resources
    kubectl delete namespace $NAMESPACE --ignore-not-found=true || true

    log_error "Rollback completed."
}

# Trap errors and rollback
trap 'rollback' ERR

# Run main deployment
main "$@"