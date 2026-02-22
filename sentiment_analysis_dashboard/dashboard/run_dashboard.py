#!/usr/bin/env python3
"""
Start script for the Sentiment Analysis Streamlit Dashboard.
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

def setup_logging():
    """Setup basic logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def check_dependencies():
    """Check if required dependencies are installed."""
    logger = logging.getLogger(__name__)
    
    try:
        import streamlit
        import pandas
        import plotly
        import requests
        logger.info("All required dependencies are available")
        return True
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        return False

def install_dependencies():
    """Install dashboard dependencies."""
    logger = logging.getLogger(__name__)
    current_dir = Path(__file__).parent
    requirements_file = current_dir / "requirements.txt"
    
    if not requirements_file.exists():
        logger.error("requirements.txt not found")
        return False
    
    try:
        logger.info("Installing dependencies...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(requirements_file)])
        logger.info("Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install dependencies: {e}")
        return False

def start_dashboard():
    """Start the Streamlit dashboard."""
    logger = logging.getLogger(__name__)
    current_dir = Path(__file__).parent
    app_file = current_dir / "app.py"
    
    if not app_file.exists():
        logger.error("app.py not found")
        return False
    
    try:
        logger.info("Starting Streamlit dashboard...")
        
        # Configure Streamlit
        env = os.environ.copy()
        env['STREAMLIT_SERVER_PORT'] = str(os.getenv('DASHBOARD_PORT', 8501))
        env['STREAMLIT_SERVER_ADDRESS'] = str(os.getenv('DASHBOARD_HOST', '0.0.0.0'))
        env['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'
        
        # Start Streamlit
        cmd = [
            sys.executable, '-m', 'streamlit', 'run', 
            str(app_file),
            '--server.port', str(env['STREAMLIT_SERVER_PORT']),
            '--server.address', str(env['STREAMLIT_SERVER_ADDRESS']),
            '--browser.gatherUsageStats', 'false'
        ]
        
        subprocess.run(cmd, env=env)
        
    except KeyboardInterrupt:
        logger.info("Dashboard stopped by user")
    except Exception as e:
        logger.error(f"Failed to start dashboard: {e}")
        return False

def main():
    """Main function."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("Starting Sentiment Analysis Dashboard...")
    
    # Install dependencies if requested
    if "--install-deps" in sys.argv:
        if not install_dependencies():
            sys.exit(1)
    
    # Check dependencies
    if not check_dependencies():
        logger.error("Dependencies not satisfied. Run with --install-deps to install them.")
        sys.exit(1)
    
    # Start dashboard
    if "--no-start" not in sys.argv:
        start_dashboard()

if __name__ == "__main__":
    main()