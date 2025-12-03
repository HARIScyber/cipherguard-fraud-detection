#!/usr/bin/env python
"""
CipherGuard Setup Verification Script
Checks all dependencies and configuration before running
"""

import sys
import os
from pathlib import Path


def print_section(title):
    """Print a section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def print_check(passed, message):
    """Print check result."""
    icon = "✅" if passed else "❌"
    print(f"{icon} {message}")
    return passed


def check_python_version():
    """Check Python version."""
    print_section("Python Version Check")
    version = sys.version_info
    passed = version.major >= 3 and version.minor >= 9
    print_check(passed, f"Python {version.major}.{version.minor}.{version.micro}")
    if not passed:
        print("  ⚠️  Requires Python 3.9+")
    return passed


def check_dependencies():
    """Check if required packages are installed."""
    print_section("Dependencies Check")
    
    packages = [
        "fastapi",
        "uvicorn",
        "pydantic",
        "numpy",
        "scikit-learn",
        "httpx",
    ]
    
    optional_packages = [
        "cyborgdb",
        "cyborgdb-service",
    ]
    
    all_passed = True
    
    print("\nRequired:")
    for package in packages:
        try:
            __import__(package.replace("-", "_"))
            print_check(True, f"{package}")
        except ImportError:
            print_check(False, f"{package}")
            all_passed = False
    
    print("\nOptional (for full CyborgDB integration):")
    for package in optional_packages:
        try:
            __import__(package.replace("-", "_"))
            print_check(True, f"{package}")
        except ImportError:
            print_check(False, f"{package} (will use local shim)")
    
    return all_passed


def check_files():
    """Check if all required files exist."""
    print_section("Project Files Check")
    
    files = {
        "app/__init__.py": "Package init",
        "app/main.py": "FastAPI application",
        "app/feature_extraction.py": "Feature extraction",
        "app/cyborg_client.py": "CyborgDB client",
        "app/cyborg_shim.py": "Local mock shim",
        "requirements.txt": "Dependencies",
        ".env": "Configuration",
    }
    
    all_passed = True
    
    for filepath, description in files.items():
        exists = Path(filepath).exists()
        print_check(exists, f"{filepath} ({description})")
        all_passed = all_passed and exists
    
    return all_passed


def check_configuration():
    """Check if configuration is set."""
    print_section("Configuration Check")
    
    # Check API Key
    api_key = os.getenv("CYBORGDB_API_KEY") or check_env_file("CYBORGDB_API_KEY")
    passed = print_check(
        api_key is not None,
        f"CYBORGDB_API_KEY: {api_key[:20] + '...' if api_key else 'NOT SET'}"
    )
    
    # Check .env file
    env_exists = Path(".env").exists()
    print_check(env_exists, ".env file exists")
    
    return passed


def check_env_file(key):
    """Read value from .env file."""
    env_path = Path(".env")
    if not env_path.exists():
        return None
    
    with open(env_path, "r") as f:
        for line in f:
            if line.startswith(key):
                return line.split("=", 1)[1].strip()
    
    return None


def check_ports():
    """Check if required ports are available."""
    print_section("Port Availability Check")
    
    import socket
    
    ports = {
        8001: "API Server",
        5432: "PostgreSQL (optional)",
        8000: "CyborgDB Service (optional)",
    }
    
    for port, service in ports.items():
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(("127.0.0.1", port))
        available = result != 0
        
        if port == 8001:
            # Port 8001 should be available
            print_check(available, f"Port {port} ({service}) - {['IN USE' if not available else 'Available']}")
        else:
            # Optional ports
            status = "In use" if not available else "Available"
            print_check(True, f"Port {port} ({service}) - {status}")


def main():
    """Run all checks."""
    print("\n" + "="*60)
    print("  🛡️  CipherGuard Setup Verification")
    print("="*60)
    
    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("Project Files", check_files),
        ("Configuration", check_configuration),
        ("Ports", check_ports),
    ]
    
    results = []
    for name, check_func in checks:
        try:
            results.append(check_func())
        except Exception as e:
            print(f"❌ Error checking {name}: {e}")
            results.append(False)
    
    # Summary
    print_section("Summary")
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"✅ All {total} checks passed!")
        print("\n🚀 Ready to run:")
        print("   python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8001")
    else:
        print(f"⚠️  {passed}/{total} checks passed")
        print("\n📖 Review errors above and run setup:")
        print("   pip install -r requirements.txt")
        print("   pip install cyborgdb cyborgdb-service")
    
    print("\n" + "="*60)
    print("  For more info, see:")
    print("  - README_START.md")
    print("  - SETUP.md")
    print("  - QUICK_REFERENCE.md")
    print("="*60 + "\n")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
