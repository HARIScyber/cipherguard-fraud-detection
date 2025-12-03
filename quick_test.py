#!/usr/bin/env python
"""Simple test to verify API is running"""

import requests
import json
import time
import sys

print("\n🧪 Testing CipherGuard API on port 8000...")
time.sleep(1)

try:
    url = f'http://localhost:8000/'
    print(f'\n📍 Testing GET /...')
    resp = requests.get(url, timeout=5)
    print(f'✅ Status: {resp.status_code}')
    data = resp.json()
    print(f"✅ Response: {data['name']}")
    print(f"✅ Version: {data['version']}")
    print(f"✅ Backend: {data['backend']}")
    
except Exception as e:
    print(f'❌ Error: {str(e)[:200]}')
    sys.exit(1)

print("\n✅ API is running successfully!")
