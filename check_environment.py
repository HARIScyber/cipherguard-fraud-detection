#!/usr/bin/env python3
"""
Quick diagnostic script for CipherGuard
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def check_environment():
    print('🔍 Checking CipherGuard Environment...')
    print('=' * 50)

    # Check environment variables
    api_key = os.getenv('CYBORGDB_API_KEY')
    service_url = os.getenv('CYBORGDB_SERVICE_URL')

    print(f'CYBORGDB_API_KEY: {api_key or "NOT SET"}')
    print(f'CYBORGDB_SERVICE_URL: {service_url or "NOT SET"}')
    print()

    # Check Python version
    print(f'Python version: {sys.version}')
    print()

    # Test imports
    print('🔍 Testing imports...')
    try:
        import fastapi
        print('✅ FastAPI imported')
    except ImportError:
        print('❌ FastAPI not available')

    try:
        import uvicorn
        print('✅ Uvicorn imported')
    except ImportError:
        print('❌ Uvicorn not available')

    try:
        import cyborgdb
        print('✅ CyborgDB imported')

        # Test client creation
        if api_key:
            try:
                client = cyborgdb.Client(api_key=api_key, base_url=service_url or 'http://localhost:8080')
                print('✅ CyborgDB client created')
            except Exception as e:
                print(f'⚠️  CyborgDB client failed: {e}')
        else:
            print('⚠️  No API key for CyborgDB client')

    except ImportError:
        print('❌ CyborgDB not available')

    print()
    print('🎯 Ready to start server!')
    print('Run: python run_server.py')

if __name__ == "__main__":
    check_environment()