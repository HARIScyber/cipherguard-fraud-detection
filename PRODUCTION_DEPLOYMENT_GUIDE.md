# CipherGuard Production Deployment Guide
## Phase 7: Production Deployment & Scaling

This guide covers the complete production deployment of CipherGuard fraud detection system.

## Prerequisites

### Infrastructure Requirements
- Kubernetes cluster (EKS, GKE, AKS, or self-managed)
- NGINX Ingress Controller
- cert-manager for SSL certificates
- Docker registry access
- PostgreSQL and Redis databases
- Monitoring stack (Prometheus/Grafana)

### System Requirements
- Minimum 3 worker nodes
- 16+ CPU cores total
- 64GB+ RAM total
- 500GB+ storage
- GPU support (optional, for ML acceleration)

## Pre-Deployment Checklist

### 1. Infrastructure Setup
```bash
# Install NGINX Ingress Controller
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.8.1/deploy/static/provider/cloud/deploy.yaml

# Install cert-manager
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml

# Verify installations
kubectl get pods -n ingress-nginx
kubectl get pods -n cert-manager
```

### 2. DNS Configuration
Update your DNS records:
- `api.cipherguard.yourdomain.com` → Load balancer IP
- `dashboard.cipherguard.yourdomain.com` → Load balancer IP
- `monitoring.cipherguard.yourdomain.com` → Load balancer IP

### 3. Docker Registry Setup
```bash
# Login to your registry
docker login your-registry.com

# Update deployment script with your registry
sed -i 's/your-registry.com/your-actual-registry.com/g' deploy_production.sh
```

### 4. Secrets Configuration
Create production secrets:
```bash
# Create namespace
kubectl create namespace fraud-detection

# Create secrets
kubectl create secret generic cipherguard-secrets \
  --namespace fraud-detection \
  --from-literal=jwt-secret="$(openssl rand -hex 32)" \
  --from-literal=database-url="postgresql://user:pass@postgres:5432/fraud_detection" \
  --from-literal=redis-url="redis://redis:6379" \
  --from-literal=smtp-server="smtp.gmail.com:587" \
  --from-literal=alert-email-from="alerts@yourdomain.com"
```

## Deployment Steps

### 1. Run Automated Deployment
```bash
# Make deployment script executable
chmod +x deploy_production.sh

# Run deployment (will prompt for confirmation at key steps)
./deploy_production.sh
```

### 2. Manual Deployment (Alternative)
```bash
# Build and push images
docker build -t your-registry.com/cipherguard-api:latest -f Dockerfile .
docker push your-registry.com/cipherguard-api:latest

# Deploy using Helm
helm upgrade --install cipherguard ./helm/cipherguard \
  --namespace fraud-detection \
  --set global.imageRegistry="your-registry.com" \
  --set global.domain="cipherguard.yourdomain.com" \
  --wait
```

### 3. Verify Deployment
```bash
# Check pod status
kubectl get pods -n fraud-detection

# Check services
kubectl get services -n fraud-detection

# Check ingress
kubectl get ingress -n fraud-detection

# Test API endpoint
curl -k https://api.cipherguard.yourdomain.com/health
```

## Post-Deployment Configuration

### 1. SSL Certificate Setup
```bash
# Verify certificate status
kubectl get certificate -n fraud-detection

# Check certificate details
kubectl describe certificate cipherguard-api-tls -n fraud-detection
```

### 2. Database Initialization
```bash
# Run database migrations
kubectl apply -f k8s/migrations/

# Verify database connection
kubectl logs -f deployment/cipherguard-api -n fraud-detection
```

### 3. Monitoring Setup
```bash
# Access Grafana
kubectl port-forward svc/grafana 3000:3000 -n fraud-detection

# Open http://localhost:3000 (admin/admin)

# Import CipherGuard dashboards
# - Fraud Detection Overview
# - System Performance
# - API Metrics
# - Alert Dashboard
```

### 4. Alert Configuration
Configure alerting rules in Prometheus:
```bash
# Edit alert rules
kubectl edit prometheusrules -n fraud-detection

# Configure notification channels (Email, Slack, PagerDuty)
```

## Scaling Configuration

### Horizontal Pod Autoscaling
The system is configured with HPA for automatic scaling:

- **API Gateway**: 2-20 replicas based on CPU (70%), Memory (80%), Requests/sec
- **Fraud Detection**: 1-10 replicas based on CPU (75%), Memory (85%)
- **Ingestion**: 1-8 replicas based on CPU (70%), Kafka consumer lag
- **Embedding**: 1-6 replicas based on CPU (80%), Memory (85%)
- **Alert**: 1-4 replicas based on CPU (60%), Queue size

### Manual Scaling
```bash
# Scale specific service
kubectl scale deployment cipherguard-api --replicas=5 -n fraud-detection

# Scale all services
kubectl scale deployment --selector=app.kubernetes.io/name=cipherguard --replicas=3 -n fraud-detection
```

## Security Hardening

### Network Policies
Network policies are automatically applied to:
- Isolate services from each other
- Allow only necessary communication
- Block external access except through ingress

### RBAC Configuration
RBAC roles are configured for:
- **Admin**: Full access to all resources
- **Developer**: Read-only access to deployments and logs
- **Monitoring**: Access to metrics and monitoring data

### Secrets Management
Sensitive data is stored in Kubernetes secrets:
- Database credentials
- JWT secrets
- API keys
- SMTP credentials

## Backup and Recovery

### Automated Backups
- **Database**: Daily backups at 2 AM, 30-day retention
- **Logs**: Continuous log shipping to centralized storage
- **Models**: ML model versioning and backup

### Disaster Recovery
```bash
# Create backup
kubectl apply -f k8s/backup/

# Restore from backup
kubectl apply -f k8s/disaster-recovery/
```

## Monitoring and Observability

### Key Metrics to Monitor
- **API Performance**: Response time, error rate, throughput
- **Fraud Detection**: Accuracy, false positive rate, processing time
- **System Resources**: CPU, memory, disk usage
- **Business Metrics**: Transaction volume, fraud alerts, user activity

### Alert Rules
Pre-configured alerts for:
- High error rates (>5%)
- High latency (>2 seconds)
- Low fraud detection accuracy (<85%)
- Resource exhaustion (>90% usage)
- Service unavailability

## Performance Optimization

### Resource Allocation
- **API Gateway**: 500m-1000m CPU, 1-2Gi RAM
- **Fraud Detection**: 1000m-2000m CPU, 2-4Gi RAM
- **Ingestion**: 500m-1000m CPU, 1-2Gi RAM
- **Embedding**: 1000m-2000m CPU, 4-8Gi RAM
- **Alert**: 250m-500m CPU, 512Mi-1Gi RAM

### Caching Strategy
- Redis for session storage and API response caching
- In-memory caching for frequently accessed data
- CDN integration for static assets

## Troubleshooting

### Common Issues

#### Pods Not Starting
```bash
# Check pod status
kubectl describe pod <pod-name> -n fraud-detection

# Check logs
kubectl logs <pod-name> -n fraud-detection

# Check resource constraints
kubectl describe hpa -n fraud-detection
```

#### Database Connection Issues
```bash
# Check database pod
kubectl logs deployment/postgres-postgresql -n fraud-detection

# Test connection
kubectl exec -it deployment/cipherguard-api -n fraud-detection -- nc -zv postgres 5432
```

#### SSL Certificate Issues
```bash
# Check certificate status
kubectl describe certificate cipherguard-api-tls -n fraud-detection

# Renew certificate
kubectl delete certificate cipherguard-api-tls -n fraud-detection
kubectl apply -f k8s/ingress/ingress.yaml
```

#### High Resource Usage
```bash
# Check resource usage
kubectl top pods -n fraud-detection

# Adjust resource limits
kubectl edit deployment <deployment-name> -n fraud-detection
```

## Maintenance Procedures

### Regular Maintenance
1. **Weekly**: Review logs and alerts
2. **Monthly**: Update dependencies and security patches
3. **Quarterly**: Performance testing and optimization

### Updates and Upgrades
```bash
# Update application version
helm upgrade cipherguard ./helm/cipherguard \
  --namespace fraud-detection \
  --set api.image.tag="v1.1.0" \
  --set fraudDetection.image.tag="v1.1.0"

# Update Kubernetes version (follow cluster provider guidelines)
```

## Support and Documentation

### Documentation Resources
- API Documentation: `https://api.cipherguard.yourdomain.com/docs`
- Monitoring Dashboard: `https://dashboard.cipherguard.yourdomain.com`
- System Metrics: `https://monitoring.cipherguard.yourdomain.com`

### Support Contacts
- **Technical Support**: support@cipherguard.yourdomain.com
- **Security Issues**: security@cipherguard.yourdomain.com
- **Emergency**: +1-800-CIPHERGUARD

---

## Next Steps

1. **Load Testing**: Perform comprehensive load testing
2. **Security Audit**: Conduct security assessment
3. **Compliance Review**: Verify regulatory compliance
4. **User Training**: Train operations and development teams
5. **Go-Live**: Schedule production go-live with rollback plan

Your CipherGuard system is now production-ready! 🚀