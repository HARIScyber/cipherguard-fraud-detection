#!/usr/bin/env python3
"""
Quick start script for the Sentiment Analysis Dashboard backend.
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

# Add the app directory to Python path
current_dir = Path(__file__).parent
app_dir = current_dir / "app"
sys.path.insert(0, str(app_dir))

def setup_logging():
    """Setup basic logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def check_environment():
    """Check if environment is properly set up."""
    logger = logging.getLogger(__name__)
    
    # Check if .env file exists
    env_file = current_dir / ".env"
    if not env_file.exists():
        logger.error(".env file not found. Please copy .env.example to .env and configure it.")
        return False
    
    # Check if virtual environment is activated (optional but recommended)
    if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        logger.warning("Virtual environment not detected. It's recommended to use a virtual environment.")
    
    return True

def install_dependencies():
    """Install Python dependencies."""
    logger = logging.getLogger(__name__)
    requirements_file = current_dir / "requirements.txt"
    
    if not requirements_file.exists():
        logger.error("requirements.txt not found.")
        return False
    
    try:
        logger.info("Installing dependencies...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(requirements_file)])
        logger.info("Dependencies installed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install dependencies: {e}")
        return False

def setup_database():
    """Initialize database."""
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Setting up database...")
        from app.database import init_db, create_admin_user
        
        # Initialize database tables
        init_db()
        logger.info("Database tables created.")
        
        # Create admin user
        admin_created = create_admin_user()
        if admin_created:
            logger.info("Admin user created successfully.")
        else:
            logger.info("Admin user already exists or creation skipped.")
        
        return True
    except Exception as e:
        logger.error(f"Failed to setup database: {e}")
        return False

def download_nltk_data():
    """Download required NLTK data."""
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Downloading NLTK data...")
        import nltk
        
        # Download required NLTK data
        nltk_data = ['punkt', 'stopwords', 'wordnet']
        for data in nltk_data:
            try:
                nltk.download(data, quiet=True)
            except:
                pass
        
        logger.info("NLTK data downloaded.")
        return True
    except Exception as e:
        logger.error(f"Failed to download NLTK data: {e}")
        return False

def start_server():
    """Start the FastAPI server."""
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Starting FastAPI server...")
        
        # Import environment variables
        from dotenv import load_dotenv
        load_dotenv()
        
        host = os.getenv("API_HOST", "0.0.0.0")
        port = int(os.getenv("API_PORT", 8000))
        reload = os.getenv("API_RELOAD", "true").lower() == "true"
        
        # Start server using uvicorn
        import uvicorn
        uvicorn.run(
            "app.main:app",
            host=host,
            port=port,
            reload=reload,
            log_level="info"
        )
        
    except KeyboardInterrupt:
        logger.info("Server stopped by user.")
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        sys.exit(1)

def main():
    """Main function."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("Starting Sentiment Analysis Dashboard Backend...")
    
    # Check environment
    if not check_environment():
        sys.exit(1)
    
    # Install dependencies
    if "--install-deps" in sys.argv or "--setup" in sys.argv:
        if not install_dependencies():
            sys.exit(1)
    
    # Setup database
    if "--setup-db" in sys.argv or "--setup" in sys.argv:
        if not setup_database():
            sys.exit(1)
    
    # Download NLTK data
    if "--nltk-data" in sys.argv or "--setup" in sys.argv:
        download_nltk_data()
    
    # Start server
    if "--no-server" not in sys.argv:
        start_server()

if __name__ == "__main__":
    main()