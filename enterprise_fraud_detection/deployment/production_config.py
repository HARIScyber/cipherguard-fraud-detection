"""
Production Deployment Configuration and Management
"""

import os
import subprocess
from pathlib import Path
from typing import Dict, List, Any
import json
import logging
import time
import psutil
import signal
from dataclasses import dataclass
from datetime import datetime

from app.core.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)


@dataclass
class DeploymentConfig:
    """Configuration for production deployment."""
    environment: str
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    max_workers: int = 8
    worker_class: str = "uvicorn.workers.UvicornWorker"
    timeout: int = 120
    keepalive: int = 2
    max_requests: int = 1000
    max_requests_jitter: int = 100
    preload_app: bool = True
    enable_ssl: bool = False
    ssl_cert_path: str = None
    ssl_key_path: str = None


class ProductionServer:
    """Production server management."""
    
    def __init__(self, config: DeploymentConfig = None):
        self.config = config or DeploymentConfig(environment=settings.general.environment)
        self.pid_file = Path("fraud_detection.pid")
        self.log_file = Path("logs/fraud_detection_server.log")
        
    def generate_gunicorn_config(self) -> str:
        """Generate Gunicorn configuration."""
        
        # Calculate optimal worker count
        if self.config.workers == 0:
            # Auto-calculate based on CPU count
            cpu_count = os.cpu_count() or 1
            self.config.workers = min(cpu_count * 2 + 1, self.config.max_workers)
        
        config_content = f'''
# Gunicorn configuration for Fraud Detection API
import multiprocessing
import os

# Server bind
bind = "{self.config.host}:{self.config.port}"

# Worker processes
workers = {self.config.workers}
worker_class = "{self.config.worker_class}"
worker_connections = 1000
max_requests = {self.config.max_requests}
max_requests_jitter = {self.config.max_requests_jitter}

# Timeout
timeout = {self.config.timeout}
keepalive = {self.config.keepalive}
graceful_timeout = 30

# Application
preload_app = {str(self.config.preload_app).lower()}
wsgi_module = "app.main:app"

# Logging
accesslog = "logs/access.log"
errorlog = "logs/error.log"
loglevel = "{settings.logging.level.lower()}"
access_log_format = '%%(h)s %%(l)s %%(u)s %%(t)s "%%(r)s" %%(s)s %%(b)s "%%(f)s" "%%(a)s" %%(D)s'

# Process naming
proc_name = "fraud_detection_api"

# Security
limit_request_line = 8190
limit_request_fields = 100
limit_request_field_size = 8190

# SSL Configuration (if enabled)
{"keyfile = '" + self.config.ssl_key_path + "'" if self.config.ssl_key_path else "# keyfile = None"}
{"certfile = '" + self.config.ssl_cert_path + "'" if self.config.ssl_cert_path else "# certfile = None"}

# Worker configuration
worker_tmp_dir = "/dev/shm" if os.path.exists("/dev/shm") else None
tmp_upload_dir = None

# Restart workers after this many requests (with jitter)
max_requests = {self.config.max_requests}

# Restart workers after this much time
max_requests_jitter = {self.config.max_requests_jitter}

# Environment variables
raw_env = [
    "ENVIRONMENT={self.config.environment}",
    "LOG_LEVEL={settings.logging.level}"
]

# PID file
pidfile = "fraud_detection.pid"

# User and group (for production)
# user = "fraud_api"
# group = "fraud_api"

# Enable hot reload in development
reload = {"true" if self.config.environment == "development" else "false"}
'''
        
        return config_content
    
    def generate_systemd_service(self) -> str:
        """Generate systemd service file for production."""
        
        current_path = Path.cwd()
        venv_path = current_path / "venv"
        
        service_content = f'''[Unit]
Description=Fraud Detection API
After=network.target
Wants=postgresql.service

[Service]
Type=notify
User=fraud_api
Group=fraud_api
WorkingDirectory={current_path}
Environment=PATH={venv_path}/bin
Environment=ENVIRONMENT={self.config.environment}
Environment=DATABASE_URL={settings.database.url}
Environment=LOG_LEVEL={settings.logging.level}
ExecStart={venv_path}/bin/gunicorn --config gunicorn.conf.py app.main:app
ExecReload=/bin/kill -s HUP $MAINPID
KillSignal=SIGTERM
Restart=always
RestartSec=3
StandardOutput=journal
StandardError=journal

# Security settings
NoNewPrivileges=yes
PrivateTmp=yes
ProtectSystem=strict
ProtectHome=yes
ReadWritePaths={current_path}/logs {current_path}/models {current_path}/data

# Resource limits
LimitNOFILE=65536
LimitCORE=0

[Install]
WantedBy=multi-user.target
'''
        
        return service_content
    
    def generate_nginx_config(self) -> str:
        """Generate Nginx reverse proxy configuration."""
        
        nginx_config = f'''# Nginx configuration for Fraud Detection API

upstream fraud_detection_backend {{
    server 127.0.0.1:{self.config.port} fail_timeout=0;
    keepalive 32;
}}

# Rate limiting
limit_req_zone $binary_remote_addr zone=api_limit:10m rate=100r/m;
limit_req_zone $binary_remote_addr zone=api_burst:10m rate=20r/s;

# Main server block
server {{
    listen 80;
    listen [::]:80;
    server_name fraud-api.company.com;
    
    # Security headers
    add_header X-Frame-Options DENY always;
    add_header X-Content-Type-Options nosniff always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    add_header Referrer-Policy "strict-origin-when-cross-origin" always;
    
    # Disable server tokens
    server_tokens off;
    
    # Client settings
    client_max_body_size 10M;
    client_body_timeout 30s;
    client_header_timeout 30s;
    
    # Proxy settings
    proxy_connect_timeout 30s;
    proxy_send_timeout 30s;
    proxy_read_timeout 30s;
    
    # API endpoints
    location /api/ {{
        # Rate limiting
        limit_req zone=api_limit burst=50 nodelay;
        limit_req zone=api_burst burst=100 nodelay;
        
        # Proxy to application
        proxy_pass http://fraud_detection_backend;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;
        
        # Timeouts
        proxy_connect_timeout 30s;
        proxy_send_timeout 30s;
        proxy_read_timeout 30s;
        
        # Buffer settings
        proxy_buffering on;
        proxy_buffer_size 8k;
        proxy_buffers 16 8k;
    }}
    
    # Health check endpoint (no rate limiting)
    location /api/v1/health {{
        proxy_pass http://fraud_detection_backend;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        access_log off;
    }}
    
    # Documentation (if enabled)
    location /docs {{
        proxy_pass http://fraud_detection_backend;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
    }}
    
    location /redoc {{
        proxy_pass http://fraud_detection_backend;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
    }}
    
    # Block access to sensitive paths
    location ~ /(\.git|\.env|\.py|requirements\.txt|Dockerfile) {{
        deny all;
        return 404;
    }}
    
    # Logging
    access_log /var/log/nginx/fraud_api_access.log combined;
    error_log /var/log/nginx/fraud_api_error.log warn;
}}

# SSL/HTTPS server block (uncomment and configure for production)
# server {{
#     listen 443 ssl http2;
#     listen [::]:443 ssl http2;
#     server_name fraud-api.company.com;
#     
#     ssl_certificate /path/to/ssl/certificate.pem;
#     ssl_certificate_key /path/to/ssl/private.key;
#     ssl_protocols TLSv1.2 TLSv1.3;
#     ssl_ciphers ECDHE-RSA-AES128-GCM-SHA256:ECDHE-RSA-AES256-GCM-SHA384:ECDHE-RSA-AES128-SHA256:ECDHE-RSA-AES256-SHA384;
#     ssl_prefer_server_ciphers off;
#     ssl_session_cache shared:SSL:10m;
#     ssl_session_timeout 10m;
#     
#     # Include same location blocks as HTTP server
#     include /etc/nginx/conf.d/fraud_api_locations.conf;
# }}
'''
        
        return nginx_config
    
    def generate_docker_config(self) -> Dict[str, str]:
        """Generate Docker configuration files."""
        
        dockerfile_content = f'''# Multi-stage production Dockerfile for Fraud Detection API
FROM python:3.11-slim as builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    libpq-dev \\
    && rm -rf /var/lib/apt/lists/*

# Create and activate virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Production image
FROM python:3.11-slim as production

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PATH="/opt/venv/bin:$PATH"

# Install runtime dependencies
RUN apt-get update && apt-get install -y \\
    libpq5 \\
    curl \\
    && rm -rf /var/lib/apt/lists/* \\
    && groupadd -r fraud_api \\
    && useradd -r -g fraud_api fraud_api

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Create application directory
WORKDIR /app

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p logs models data \\
    && chown -R fraud_api:fraud_api /app

# Switch to non-root user
USER fraud_api

# Expose port
EXPOSE {self.config.port}

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \\
    CMD curl -f http://localhost:{self.config.port}/api/v1/health || exit 1

# Start command
CMD ["gunicorn", "--config", "gunicorn.conf.py", "app.main:app"]
'''

        docker_compose_content = f'''version: '3.8'

services:
  fraud-detection-api:
    build: .
    container_name: fraud-detection-api
    restart: unless-stopped
    env_file:
      - .env.production
    environment:
      - ENVIRONMENT=production
      - DATABASE_URL=postgresql://fraud_user:fraud_password@postgres:5432/fraud_db
    ports:
      - "{self.config.port}:{self.config.port}"
    volumes:
      - ./logs:/app/logs
      - ./models:/app/models
      - ./data:/app/data
    depends_on:
      - postgres
      - redis
    networks:
      - fraud-detection-network
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:{self.config.port}/api/v1/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s

  postgres:
    image: postgres:15
    container_name: fraud-detection-postgres
    restart: unless-stopped
    environment:
      POSTGRES_DB: fraud_db
      POSTGRES_USER: fraud_user
      POSTGRES_PASSWORD: fraud_password
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./database/init.sql:/docker-entrypoint-initdb.d/init.sql
    ports:
      - "5432:5432"
    networks:
      - fraud-detection-network
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U fraud_user -d fraud_db"]
      interval: 30s
      timeout: 10s
      retries: 3

  redis:
    image: redis:7-alpine
    container_name: fraud-detection-redis
    restart: unless-stopped
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data
    ports:
      - "6379:6379"
    networks:
      - fraud-detection-network
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 30s
      timeout: 10s
      retries: 3

  nginx:
    image: nginx:alpine
    container_name: fraud-detection-nginx
    restart: unless-stopped
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/conf.d/default.conf
      - ./ssl:/etc/ssl/certs
    depends_on:
      - fraud-detection-api
    networks:
      - fraud-detection-network

networks:
  fraud-detection-network:
    driver: bridge

volumes:
  postgres_data:
  redis_data:
'''
        
        return {
            "Dockerfile": dockerfile_content,
            "docker-compose.yml": docker_compose_content
        }
    
    def write_production_configs(self, output_dir: Path = None):
        """Write all production configuration files."""
        
        output_dir = output_dir or Path("./deployment")
        output_dir.mkdir(exist_ok=True)
        
        configs = {
            "gunicorn.conf.py": self.generate_gunicorn_config(),
            "fraud-detection-api.service": self.generate_systemd_service(),
            "nginx.conf": self.generate_nginx_config()
        }
        
        # Add Docker configs
        docker_configs = self.generate_docker_config()
        configs.update(docker_configs)
        
        # Write files
        for filename, content in configs.items():
            file_path = output_dir / filename
            with open(file_path, 'w') as f:
                f.write(content)
            
            logger.info(f"Generated {filename} -> {file_path}")
        
        # Generate environment files
        self._generate_env_files(output_dir)
        
        # Generate deployment scripts
        self._generate_deployment_scripts(output_dir)
        
        logger.info(f"Production configuration files written to {output_dir}")
    
    def _generate_env_files(self, output_dir: Path):
        """Generate environment configuration files."""
        
        # Production environment file
        prod_env = f'''# Production Environment Configuration
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO

# Database
DATABASE_URL=postgresql://fraud_user:fraud_password@localhost:5432/fraud_db
DATABASE_POOL_SIZE=20
DATABASE_MAX_OVERFLOW=30

# API Configuration
API_HOST=0.0.0.0
API_PORT={self.config.port}
API_KEY=your-super-secure-api-key-here
CORS_ORIGINS=["https://your-frontend-domain.com"]
ENABLE_DOCS=false

# ML Configuration
MODELS_DIR=/app/models
ENABLE_MODEL_TRAINING=true
RETRAIN_INTERVAL_HOURS=24

# Security
SECRET_KEY=your-super-secret-key-for-jwt-signing
TRUSTED_HOSTS=["your-api-domain.com"]

# Monitoring
ENABLE_METRICS=true
METRICS_PORT=9090
'''
        
        # Development environment file
        dev_env = f'''# Development Environment Configuration
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=DEBUG

# Database
DATABASE_URL=postgresql://fraud_user:fraud_password@localhost:5432/fraud_db_dev

# API Configuration
API_HOST=127.0.0.1
API_PORT=8000
API_KEY=dev-api-key
CORS_ORIGINS=["http://localhost:3000", "http://127.0.0.1:3000"]
ENABLE_DOCS=true

# ML Configuration
MODELS_DIR=./models
ENABLE_MODEL_TRAINING=true

# Security
SECRET_KEY=dev-secret-key
'''
        
        with open(output_dir / ".env.production", 'w') as f:
            f.write(prod_env)
        
        with open(output_dir / ".env.development", 'w') as f:
            f.write(dev_env)
    
    def _generate_deployment_scripts(self, output_dir: Path):
        """Generate deployment and management scripts."""
        
        deploy_script = '''#!/bin/bash
# Production Deployment Script for Fraud Detection API

set -e

echo "Starting Fraud Detection API deployment..."

# Check for required files
if [ ! -f ".env.production" ]; then
    echo "Error: .env.production file not found"
    exit 1
fi

# Load environment variables
export $(cat .env.production | xargs)

# Create necessary directories
mkdir -p logs models data

# Install system dependencies (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install -y postgresql postgresql-contrib nginx supervisor

# Set up database
sudo -u postgres psql -c "CREATE DATABASE fraud_db;"
sudo -u postgres psql -c "CREATE USER fraud_user WITH ENCRYPTED PASSWORD 'fraud_password';"
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE fraud_db TO fraud_user;"

# Install Python dependencies
pip install -r requirements.txt

# Run database migrations
python -c "from database.init_db import create_tables; create_tables()"

# Copy configuration files
sudo cp fraud-detection-api.service /etc/systemd/system/
sudo cp nginx.conf /etc/nginx/sites-available/fraud-detection-api
sudo ln -sf /etc/nginx/sites-available/fraud-detection-api /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default

# Create application user
sudo useradd -r -s /bin/false fraud_api || true
sudo chown -R fraud_api:fraud_api logs models data

# Start services
sudo systemctl daemon-reload
sudo systemctl enable fraud-detection-api
sudo systemctl start fraud-detection-api
sudo systemctl enable nginx
sudo systemctl restart nginx

# Test deployment
echo "Waiting for service to start..."
sleep 10
curl -f http://localhost/api/v1/health || {
    echo "Health check failed!"
    sudo systemctl status fraud-detection-api
    exit 1
}

echo "Deployment completed successfully!"
echo "API is available at: http://your-domain.com/api/v1/"
'''
        
        docker_deploy_script = '''#!/bin/bash
# Docker Deployment Script for Fraud Detection API

set -e

echo "Starting Docker deployment..."

# Build and start services
docker-compose build
docker-compose up -d

# Wait for services to be healthy
echo "Waiting for services to be ready..."
timeout 120s bash -c 'until docker-compose ps | grep -q "healthy"; do sleep 5; done'

# Run database migrations
docker-compose exec fraud-detection-api python -c "from database.init_db import create_tables; create_tables()"

# Test deployment
echo "Testing API..."
curl -f http://localhost/api/v1/health || {
    echo "Health check failed!"
    docker-compose logs fraud-detection-api
    exit 1
}

echo "Docker deployment completed successfully!"
echo "Services are running and healthy."
'''
        
        status_script = '''#!/bin/bash
# Status Check Script for Fraud Detection API

echo "=== Fraud Detection API Status ==="
echo

# System service status
if systemctl is-active --quiet fraud-detection-api; then
    echo "✓ API Service: Running"
    echo "  Workers: $(ps aux | grep 'fraud_detection_api' | grep -v grep | wc -l)"
    echo "  PID: $(systemctl show --property MainPID --value fraud-detection-api)"
else
    echo "✗ API Service: Not running"
fi

# Nginx status
if systemctl is-active --quiet nginx; then
    echo "✓ Nginx: Running"
else
    echo "✗ Nginx: Not running"
fi

# Database connection
if pg_isready -h localhost -p 5432 -U fraud_user >/dev/null 2>&1; then
    echo "✓ Database: Connected"
else
    echo "✗ Database: Connection failed"
fi

# API health check
if curl -sf http://localhost/api/v1/health >/dev/null; then
    echo "✓ API Health: OK"
else
    echo "✗ API Health: Failed"
fi

echo
echo "=== Recent Logs ==="
sudo journalctl -u fraud-detection-api --lines=5 --no-pager
'''
        
        # Write scripts
        scripts = {
            "deploy.sh": deploy_script,
            "docker-deploy.sh": docker_deploy_script,
            "status.sh": status_script
        }
        
        for script_name, content in scripts.items():
            script_path = output_dir / script_name
            with open(script_path, 'w') as f:
                f.write(content)
            
            # Make executable
            os.chmod(script_path, 0o755)
            
            logger.info(f"Generated {script_name} -> {script_path}")


def create_production_deployment():
    """Create complete production deployment configuration."""
    
    # Determine deployment configuration
    config = DeploymentConfig(
        environment="production",
        host="0.0.0.0",
        port=settings.api.port,
        workers=0,  # Auto-calculate
        timeout=120
    )
    
    # Create server manager
    server = ProductionServer(config)
    
    # Generate all configuration files
    server.write_production_configs()
    
    print("✓ Production deployment configuration generated successfully!")
    print("\nNext steps:")
    print("1. Review and customize the generated configuration files")
    print("2. Set up your production environment variables in .env.production")
    print("3. Run ./deployment/deploy.sh to deploy to a server")
    print("4. Or run ./deployment/docker-deploy.sh to deploy with Docker")
    print("\nFiles generated in ./deployment/ directory:")
    print("- gunicorn.conf.py (WSGI server configuration)")
    print("- nginx.conf (Reverse proxy configuration)")
    print("- fraud-detection-api.service (Systemd service)")
    print("- Dockerfile & docker-compose.yml (Container deployment)")
    print("- deploy.sh & docker-deploy.sh (Deployment scripts)")
    print("- .env.production & .env.development (Environment configs)")


if __name__ == "__main__":
    create_production_deployment()