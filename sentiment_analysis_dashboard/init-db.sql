-- Database initialization script for Sentiment Analysis Dashboard
-- This script sets up the PostgreSQL database with proper permissions and indexes

-- Create the database (if not already created by Docker)
-- CREATE DATABASE sentiment_db;

-- Connect to sentiment database
\c sentiment_db;

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Create indexes for better performance (these will be created by SQLAlchemy too)
-- But we can pre-create some for optimization

-- Sample data for testing (optional)
-- Note: The application will create the actual tables via SQLAlchemy

-- Create a function to generate UUID v4
CREATE OR REPLACE FUNCTION generate_uuid()
RETURNS UUID AS $$
BEGIN
    RETURN uuid_generate_v4();
END;
$$ LANGUAGE plpgsql;

-- Grant permissions to the sentiment_user
GRANT ALL PRIVILEGES ON DATABASE sentiment_db TO sentiment_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO sentiment_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO sentiment_user;

-- Set default permissions for future tables
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL PRIVILEGES ON TABLES TO sentiment_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL PRIVILEGES ON SEQUENCES TO sentiment_user;

-- Create a function to clean old analytics cache (cleanup job)
CREATE OR REPLACE FUNCTION cleanup_expired_cache()
RETURNS void AS $$
BEGIN
    -- This will be used to clean expired cache entries
    -- The actual table will be created by SQLAlchemy
    -- DELETE FROM analytics_cache WHERE expires_at < NOW();
    NULL;
END;
$$ LANGUAGE plpgsql;

-- Example of creating a materialized view for analytics (advanced optimization)
-- This would be created after the tables exist
-- CREATE MATERIALIZED VIEW daily_sentiment_stats AS
-- SELECT 
--     DATE(created_at) as date,
--     sentiment,
--     COUNT(*) as count,
--     AVG(confidence_score) as avg_confidence
-- FROM comments 
-- GROUP BY DATE(created_at), sentiment
-- ORDER BY date DESC;

COMMIT;