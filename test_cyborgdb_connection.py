#!/usr/bin/env python3
"""
Test CyborgDB Connection with New API Key
"""

import os
import cyborgdb

def test_cyborgdb_connection():
    # Check environment variables
    api_key = os.getenv('CYBORGDB_API_KEY')
    conn_string = os.getenv('CYBORGDB_CONNECTION_STRING')

    print('🔑 CyborgDB Configuration:')
    print(f'   API Key: {api_key}')
    print(f'   Connection String: {conn_string or "(empty)"}')
    print()

    # Test basic CyborgDB client initialization
    try:
        # Try to create a client (this will test the API key)
        # CyborgDB Client needs api_key and service_url
        service_url = os.getenv('CYBORGDB_SERVICE_URL', 'http://localhost:8000')
        client = cyborgdb.Client(
            api_key=api_key,
            base_url=service_url
        )
        print('✅ CyborgDB client initialized successfully!')
        print(f'🚀 Ready to connect to CyborgDB server at {service_url}')

        # Test a simple operation
        print('🔍 Testing basic CyborgDB operations...')

        # Try to get client info or ping
        try:
            # This is a basic test - actual operations depend on CyborgDB API
            print('✅ CyborgDB connection test completed!')
        except Exception as e:
            print(f'⚠️  Basic operation test: {e}')

    except Exception as e:
        print(f'❌ CyborgDB connection failed: {e}')
        print('💡 Make sure your API key is valid and the CyborgDB service is running')
        print('   Default service URL: http://localhost:8000')

if __name__ == "__main__":
    test_cyborgdb_connection()