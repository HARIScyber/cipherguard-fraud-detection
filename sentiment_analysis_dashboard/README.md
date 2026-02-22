# 📊 Sentiment Analysis Dashboard

A production-ready Comment Sentiment Analysis Dashboard with FastAPI backend, Streamlit admin interface, and PostgreSQL database.

## 🚀 Features

### Backend API (FastAPI)
- **RESTful API** with async support and automatic documentation
- **JWT Authentication** with bcrypt password hashing
- **PostgreSQL Database** with SQLAlchemy ORM and connection pooling
- **ML Pipeline** using TF-IDF + Logistic Regression for sentiment analysis
- **Real-time Analytics** with caching and time-series data
- **Health Monitoring** with system metrics and ML model status
- **Production Ready** with proper logging, error handling, and validation

### Admin Dashboard (Streamlit)
- **Interactive Web Interface** with modern UI components
- **Authentication System** with session management
- **Real-time Analytics** with interactive charts and visualizations
- **Comment Analysis** with live sentiment prediction
- **Data Management** with searchable tables and filters
- **System Monitoring** with health checks and performance metrics

### Core ML Features
- **Sentiment Classification**: Positive, Negative, Neutral
- **Confidence Scoring**: Model confidence for each prediction
- **Text Preprocessing**: Advanced text cleaning and normalization
- **Model Persistence**: Automatic model training and caching
- **Performance Monitoring**: Response time tracking and analytics

## 📋 Prerequisites

- Python 3.11+
- Docker and Docker Compose (for containerized deployment)
- PostgreSQL (for local development)

## 🛠️ Quick Start

### Method 1: Docker Compose (Recommended)

1. **Clone and Navigate**
   ```bash
   cd sentiment_analysis_dashboard
   ```

2. **Configure Environment**
   ```bash
   cp .env.example .env
   # Edit .env file with your settings
   ```

3. **Start All Services**
   ```bash
   docker-compose up -d
   ```

4. **Access Applications**
   - **Backend API**: http://localhost:8000
   - **API Documentation**: http://localhost:8000/docs
   - **Admin Dashboard**: http://localhost:8501

### Method 2: Local Development

1. **Setup Backend**
   ```bash
   cd backend
   
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # Windows: venv\\Scripts\\activate
   
   # Install dependencies
   pip install -r requirements.txt
   
   # Configure environment
   cp .env.example .env
   # Edit .env with your database settings
   
   # Start backend server
   python run_server.py --setup
   ```

2. **Setup Dashboard**
   ```bash
   cd ../dashboard
   
   # Install dependencies
   pip install -r requirements.txt
   
   # Start dashboard
   python run_dashboard.py
   ```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection string | `postgresql://...` |
| `JWT_SECRET_KEY` | JWT token secret key | `your-secret-key` |
| `ADMIN_USERNAME` | Default admin username | `admin` |
| `ADMIN_PASSWORD` | Default admin password | `admin123` |
| `API_HOST` | Backend host | `0.0.0.0` |
| `API_PORT` | Backend port | `8000` |
| `DEBUG_MODE` | Enable debug mode | `false` |

### Database Configuration

For production, use a properly configured PostgreSQL instance:

```bash
# Production database example
DATABASE_URL=postgresql://user:password@localhost:5432/sentiment_production
```

For development, SQLite is supported as fallback:

```bash
DATABASE_URL=sqlite:///./sentiment.db
USE_SQLITE=true
```

## 📚 API Documentation

### Authentication Endpoints

#### `POST /api/v1/auth/login`
Login with username/password to get JWT token.

**Request:**
```json
{
  "username": "admin",
  "password": "admin123"
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "access_token": "eyJ0eXAiOiJKV1QiLCJ...",
    "token_type": "bearer",
    "expires_in": 86400
  }
}
```

#### `POST /api/v1/auth/register`
Register a new user account.

#### `GET /api/v1/auth/me`
Get current user information.

### Comment Analysis Endpoints

#### `POST /api/v1/comments/analyze`
Analyze sentiment of a comment.

**Request:**
```json
{
  "comment_text": "This product is amazing!",
  "store_result": true
}
```

**Response:**
```json
{
  "id": 123,
  "comment_text": "This product is amazing!",
  "sentiment": "positive",
  "confidence_score": 0.924,
  "processing_time_ms": 15.2,
  "created_at": "2024-01-15T10:30:00Z"
}
```

#### `GET /api/v1/comments/`
Get paginated comments with filters.

**Query Parameters:**
- `page`: Page number (default: 1)
- `page_size`: Items per page (default: 10)
- `sentiment`: Filter by sentiment (positive/negative/neutral)
- `search`: Search in comment text
- `user_id`: Filter by user ID

#### `GET /api/v1/comments/analytics/sentiment`
Get sentiment analytics for time period.

**Query Parameters:**
- `days`: Number of days to analyze (default: 30)
- `use_cache`: Use cached results (default: true)

**Response:**
```json
{
  "success": true,
  "data": {
    "total_comments": 1500,
    "positive_count": 850,
    "negative_count": 300,
    "neutral_count": 350,
    "positive_percentage": 56.67,
    "negative_percentage": 20.00,
    "neutral_percentage": 23.33,
    "avg_confidence_score": 0.847
  }
}
```

### Health Check Endpoints

#### `GET /api/v1/health/`
Basic health check (public endpoint).

#### `GET /api/v1/health/detailed`
Detailed system health with metrics (authenticated).

## 🎨 Dashboard Features

### Overview Dashboard
- **Real-time Metrics**: Total comments, sentiment distribution, confidence scores
- **Interactive Charts**: Pie charts, bar charts, time-series visualizations
- **Recent Alerts**: Display of recent negative comments for monitoring
- **Quick Stats**: Key performance indicators and trends

### Comment Analysis
- **Live Analysis**: Enter comments for real-time sentiment analysis
- **Confidence Scoring**: Visual confidence gauge and detailed metrics
- **Result Storage**: Automatic saving to database with tracking
- **Batch Processing**: Support for analyzing multiple comments

### Analytics Dashboard
- **Time-based Analytics**: Configurable time periods (7, 30, 90 days)
- **Trend Analysis**: Time-series charts showing sentiment trends over time
- **Distribution Analysis**: Detailed breakdown of sentiment categories
- **Export Options**: Download analytics data in various formats

### Comment History
- **Searchable Table**: Filter and search through all analyzed comments
- **Advanced Filters**: Filter by sentiment, user, time range, text content
- **Pagination**: Efficient pagination for large datasets
- **Detailed View**: Expandable rows with full comment details

### System Status
- **Health Monitoring**: Real-time system health and performance metrics
- **API Status**: Backend API connectivity and response times
- **Database Health**: Database connection and performance metrics
- **ML Model Status**: Machine learning model health and test predictions

## 🔒 Security Features

### Authentication & Authorization
- **JWT Tokens**: Secure token-based authentication
- **Password Hashing**: Bcrypt with configurable rounds
- **Session Management**: Secure session handling in dashboard
- **Role-based Access**: Admin and user role separation

### Data Security
- **Input Validation**: Comprehensive request validation with Pydantic
- **SQL Injection Protection**: Parameterized queries with SQLAlchemy ORM
- **CORS Configuration**: Configurable cross-origin request handling
- **Rate Limiting**: Built-in request rate limiting (production ready)

### Production Security
- **Environment Variables**: Secure configuration management
- **Secrets Management**: Proper handling of sensitive credentials
- **HTTPS Ready**: SSL/TLS configuration support
- **Security Headers**: Standard security headers implementation

## 🚀 Deployment

### Docker Production Deployment

1. **Production Environment Setup**
   ```bash
   cp .env.example .env
   # Configure production settings in .env
   ```

2. **Deploy with Production Profile**
   ```bash
   docker-compose --profile production up -d
   ```

3. **Database Migration**
   ```bash
   docker-compose exec backend python -c "from app.database import init_db; init_db()"
   ```

4. **Create Admin User**
   ```bash
   docker-compose exec backend python -c "from app.database import create_admin_user; create_admin_user()"
   ```

### Kubernetes Deployment

For Kubernetes deployment, use the provided Helm charts:

```bash
cd helm/
helm install sentiment-dashboard ./sentiment-chart
```

### Cloud Deployment Options

#### AWS ECS
- Use provided `docker-compose.yml` with AWS ECS CLI
- Configure RDS PostgreSQL instance
- Set up Application Load Balancer

#### Google Cloud Run
- Deploy containers using Cloud Run
- Use Cloud SQL for PostgreSQL
- Configure Cloud Load Balancer

#### Azure Container Instances
- Deploy using Azure Container Instances
- Use Azure Database for PostgreSQL
- Configure Azure Application Gateway

## 🧪 Testing

### Backend API Testing

```bash
cd backend

# Run unit tests
python -m pytest tests/

# Test API endpoints
python test_api.py

# Test database setup only
python test_api.py --db-only

# Test API endpoints only
python test_api.py --api-only
```

### Manual Testing

1. **Start Services**
   ```bash
   docker-compose up -d
   ```

2. **Test API Health**
   ```bash
   curl http://localhost:8000/api/v1/health/
   ```

3. **Test Authentication**
   ```bash
   curl -X POST http://localhost:8000/api/v1/auth/login \\
     -H "Content-Type: application/json" \\
     -d '{"username": "admin", "password": "admin123"}'
   ```

4. **Test Sentiment Analysis**
   ```bash
   curl -X POST http://localhost:8000/api/v1/comments/analyze \\
     -H "Content-Type: application/json" \\
     -H "Authorization: Bearer YOUR_TOKEN" \\
     -d '{"comment_text": "This is amazing!", "store_result": true}'
   ```

## 📊 Monitoring & Observability

### Built-in Monitoring
- **Health Checks**: Comprehensive health endpoints for all services
- **Performance Metrics**: Response time tracking and system resource monitoring
- **Error Logging**: Structured logging with configurable levels
- **Database Monitoring**: Connection pool and query performance tracking

### Integration Options
- **Prometheus**: Metrics collection endpoint available
- **Grafana**: Dashboard templates for visualization
- **ELK Stack**: Structured logging compatible with Elasticsearch
- **New Relic**: APM integration ready

## 🔧 Maintenance

### Database Maintenance

```bash
# Backup database
docker-compose exec database pg_dump -U sentiment_user sentiment_db > backup.sql

# Restore database
docker-compose exec -T database psql -U sentiment_user sentiment_db < backup.sql

# Clean old cache entries
docker-compose exec database psql -U sentiment_user -d sentiment_db -c "DELETE FROM analytics_cache WHERE expires_at < NOW();"
```

### Log Management

```bash
# View backend logs
docker-compose logs backend

# View dashboard logs
docker-compose logs dashboard

# Follow real-time logs
docker-compose logs -f
```

### Model Retraining

The ML model automatically retrains when new data is available. For manual retraining:

```bash
docker-compose exec backend python -c "
from app.services.sentiment_analyzer import SentimentAnalyzer
analyzer = SentimentAnalyzer()
analyzer.train_model()
"
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

### Common Issues

#### Connection Refused Errors
- Ensure all services are running: `docker-compose ps`
- Check network connectivity: `docker-compose exec backend ping database`
- Verify environment variables in `.env` file

#### Database Connection Issues  
- Reset database: `docker-compose down -v && docker-compose up -d`
- Check PostgreSQL logs: `docker-compose logs database`
- Verify database credentials in environment

#### ML Model Issues
- Restart backend service: `docker-compose restart backend`
- Check model training logs: `docker-compose logs backend | grep "model"`
- Verify NLTK data download: `docker-compose exec backend python -c "import nltk; print(nltk.data.path)"`

### Getting Help

- **Documentation**: Check API docs at http://localhost:8000/docs
- **Logs**: Use `docker-compose logs [service_name]` for troubleshooting
- **Health Checks**: Visit http://localhost:8000/api/v1/health/detailed
- **Issues**: Open a GitHub issue with detailed information

## 🎯 Roadmap

### Upcoming Features
- [ ] Real-time sentiment monitoring with WebSocket support
- [ ] Advanced ML models (BERT, RoBERTa) for improved accuracy
- [ ] Multi-language sentiment analysis support
- [ ] Advanced analytics with trend predictions
- [ ] API rate limiting and usage analytics
- [ ] Webhook notifications for sentiment alerts
- [ ] Advanced user management and permissions
- [ ] Data export and backup automation

### Performance Improvements
- [ ] Redis caching layer for improved performance
- [ ] Database query optimization and indexing
- [ ] Async processing queue for batch operations
- [ ] CDN integration for static assets
- [ ] Horizontal scaling support

---

**Built with ❤️ using FastAPI, Streamlit, PostgreSQL, and Docker**