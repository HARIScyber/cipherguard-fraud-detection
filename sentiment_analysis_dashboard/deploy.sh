#!/bin/bash
# Deployment script for Sentiment Analysis Dashboard

echo "🚀 Starting Sentiment Analysis Dashboard Deployment..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Check if .env file exists
if [ ! -f .env ]; then
    echo "📝 Creating .env file from template..."
    cp .env.example .env
    echo "✅ Please edit .env file with your configuration before continuing."
    echo "   Key settings to update:"
    echo "   - JWT_SECRET_KEY (use a strong random key)"
    echo "   - ADMIN_PASSWORD (change from default)"
    echo "   - POSTGRES_PASSWORD (change from default)"
    read -p "Press Enter after editing .env file to continue..."
fi

echo "🧹 Cleaning up existing containers..."
docker-compose down -v

echo "🔨 Building images..."
docker-compose build --no-cache

echo "🚀 Starting services..."
docker-compose up -d

echo "⏳ Waiting for services to be ready..."
sleep 30

# Check service health
echo "🔍 Checking service health..."

# Check database
echo "📊 Database status:"
docker-compose exec -T database pg_isready -U sentiment_user -d sentiment_db

# Check backend
echo "🔧 Backend API status:"
curl -f http://localhost:8000/api/v1/health/ || echo "❌ Backend health check failed"

# Check dashboard
echo "🎨 Dashboard status:"
curl -f http://localhost:8501/_stcore/health || echo "❌ Dashboard health check failed"

echo ""
echo "✅ Deployment complete!"
echo ""
echo "🌐 Access your applications:"
echo "   📊 Admin Dashboard:  http://localhost:8501"
echo "   🔧 Backend API:      http://localhost:8000"
echo "   📖 API Docs:         http://localhost:8000/docs"
echo ""
echo "🔐 Default Login Credentials:"
echo "   Username: admin"
echo "   Password: (check ADMIN_PASSWORD in .env file)"
echo ""
echo "📋 Useful Commands:"
echo "   View logs:           docker-compose logs -f"
echo "   Stop services:       docker-compose down"
echo "   Restart service:     docker-compose restart [service_name]"
echo ""