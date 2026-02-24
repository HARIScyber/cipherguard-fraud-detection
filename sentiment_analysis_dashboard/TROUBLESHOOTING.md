# 🔧 Troubleshooting Guide

## Common Docker Compose Issues

### Issue 1: "docker-compose command not found"
**Solution:**
```bash
# Use modern Docker syntax instead
docker compose up -d

# OR install Docker Desktop (includes docker-compose)
# Download from: https://www.docker.com/products/docker-desktop/
```

### Issue 2: Port Already in Use
**Error:** `bind: address already in use`
**Solution:**
```bash
# Check what's using the ports
netstat -ano | findstr :8000  # Windows
lsof -i :8000                 # Linux/Mac

# Kill the process
taskkill /PID <process_id> /F  # Windows
kill -9 <process_id>           # Linux/Mac

# Or change ports in docker-compose.yml
ports:
  - "8001:8000"  # Use different external port
```

### Issue 3: Database Connection Failures
**Error:** "could not connect to server"
**Solutions:**
```bash
# Wait longer for database to initialize
docker-compose up -d
sleep 60  # Wait 60 seconds

# Check database logs
docker-compose logs database

# Restart database service
docker-compose restart database

# Reset database completely
docker-compose down -v
docker-compose up -d
```

### Issue 4: Backend Import Errors
**Error:** ModuleNotFoundError
**Solutions:**
```bash
# Rebuild backend without cache
docker-compose build --no-cache backend

# Check if requirements.txt is complete
docker-compose exec backend pip list

# Manual dependency install
docker-compose exec backend pip install missing-package
```

### Issue 5: Dashboard Cannot Connect to Backend
**Error:** Connection refused on localhost:8000
**Check:**
1. Backend service is running: `docker-compose ps`
2. Backend health: `curl http://localhost:8000/api/v1/health/`
3. Network connectivity: `docker-compose exec dashboard ping backend`

### Issue 6: NLTK Data Download Failures
**Error:** NLTK download errors
**Solution:**
```bash
# Manual NLTK data download
docker-compose exec backend python -c "
import nltk
nltk.download('punkt')
nltk.download('stopwords') 
nltk.download('wordnet')
"
```

### Issue 7: Permission Denied (Linux/Mac)
**Solution:**
```bash
# Give execute permission to scripts
chmod +x deploy.sh

# Fix Docker socket permissions
sudo chmod 666 /var/run/docker.sock
```

## Diagnostic Commands

### Service Status
```bash
# Check all services
docker-compose ps

# Check specific service
docker-compose ps backend

# View service logs
docker-compose logs -f backend
docker-compose logs -f dashboard
docker-compose logs -f database
```

### Health Checks
```bash
# Backend API health
curl http://localhost:8000/api/v1/health/

# Database health
docker-compose exec database pg_isready -U sentiment_user -d sentiment_db

# Dashboard health
curl http://localhost:8501/_stcore/health
```

### Network Debugging
```bash
# List Docker networks
docker network ls

# Inspect sentiment network
docker network inspect sentiment_analysis_dashboard_sentiment_network

# Test inter-service connectivity
docker-compose exec dashboard ping backend
docker-compose exec backend ping database
```

### Database Access
```bash
# Connect to PostgreSQL
docker-compose exec database psql -U sentiment_user -d sentiment_db

# Inside PostgreSQL:
\dt                    # List tables
\d comments           # Describe comments table
SELECT * FROM comments LIMIT 5;
SELECT * FROM users;
\q                    # Exit
```

### Complete Reset
```bash
# Nuclear option - reset everything
docker-compose down -v
docker system prune -f
docker volume prune -f
docker-compose build --no-cache
docker-compose up -d
```

## Performance Optimization

### Resource Limits
Add to docker-compose.yml services:
```yaml
deploy:
  resources:
    limits:
      cpus: '1.0'
      memory: 1G
    reservations:
      memory: 512M
```

### Database Optimization
```bash
# Increase shared_buffers for PostgreSQL
docker-compose exec database psql -U sentiment_user -d sentiment_db -c "
ALTER SYSTEM SET shared_buffers = '256MB';
ALTER SYSTEM SET effective_cache_size = '1GB';
SELECT pg_reload_conf();
"
```

## Security Hardening

### Change Default Passwords
Edit .env file:
```bash
ADMIN_PASSWORD=YourSecurePassword123!
POSTGRES_PASSWORD=YourSecureDbPassword456!
JWT_SECRET_KEY=YourSuperSecretJWTKey789!
```

### Enable HTTPS (Production)
1. Uncomment nginx service in docker-compose.yml
2. Add SSL certificates to nginx/ssl/
3. Configure nginx.conf for HTTPS

### Firewall Configuration
```bash
# Allow only necessary ports
ufw allow 8000    # Backend API
ufw allow 8501    # Dashboard
ufw deny 5432     # Block direct database access
```

## Monitoring Setup

### Health Check Dashboard
```bash
# Check all service health
curl -s http://localhost:8000/api/v1/health/ | jq .
```

### Log Aggregation
```bash
# Centralized logging
docker-compose logs --tail=100 -f | tee application.log
```

### Performance Monitoring
```bash
# Resource usage
docker stats

# Disk usage
docker system df
```

## Backup & Recovery

### Database Backup
```bash
# Create backup
docker-compose exec database pg_dump -U sentiment_user sentiment_db > backup_$(date +%Y%m%d).sql

# Restore backup
cat backup_20240223.sql | docker-compose exec -T database psql -U sentiment_user sentiment_db
```

### Volume Backup
```bash
# Backup volumes
docker run --rm -v sentiment_analysis_dashboard_postgres_data:/data -v $(pwd):/backup alpine tar czf /backup/postgres_backup.tar.gz /data
```