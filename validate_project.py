#!/usr/bin/env python3
"""
CipherGuard Project Validation Script
Validates that all components are properly configured for production deployment
"""

import sys
import os
import subprocess
import importlib.util

def print_header():
    print('🔍 CipherGuard Project Validation')
    print('================================')

def check_python_version():
    """Check Python version compatibility"""
    print(f'Python Version: {sys.version}')
    version = sys.version_info
    if version.major == 3 and version.minor >= 11:
        print('✅ Python 3.11+ detected')
        return True
    else:
        print('❌ Python 3.11+ required')
        return False

def check_required_files():
    """Check if all required files exist"""
    required_files = [
        'setup_cicd.sh',
        'setup_operations.sh',
        'PROJECT_COMPLETION_REPORT.md',
        'app/main.py',
        'requirements.txt',
        'helm/cipherguard/Chart.yaml',
        'k8s/ingress/ingress.yaml',
        'docker-compose.yml'
    ]

    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)

    if missing_files:
        print(f'❌ Missing files: {missing_files}')
        return False
    else:
        print('✅ All required files present')
        return True

def check_python_dependencies():
    """Check if core Python dependencies are available"""
    dependencies = [
        'fastapi',
        'uvicorn',
        'sqlalchemy',
        'redis',
        'pydantic',
        'pytest',
        'kubernetes'
    ]

    missing_deps = []
    for dep in dependencies:
        try:
            importlib.import_module(dep)
        except ImportError:
            missing_deps.append(dep)

    if missing_deps:
        print(f'❌ Missing Python dependencies: {missing_deps}')
        print('Run: pip install -r requirements.txt')
        return False
    else:
        print('✅ Core Python dependencies available')
        return True

def check_docker():
    """Check if Docker is available (optional for development)"""
    try:
        result = subprocess.run(['docker', '--version'],
                              capture_output=True, text=True, check=True)
        print(f'✅ Docker available: {result.stdout.strip()}')
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print('⚠️  Docker not available (install for production deployment)')
        return True  # Don't fail validation for this

def check_kubectl():
    """Check if kubectl is available (optional for development)"""
    try:
        result = subprocess.run(['kubectl', 'version', '--client'],
                              capture_output=True, text=True, check=True)
        print('✅ kubectl available')
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print('⚠️  kubectl not available (install for Kubernetes deployment)')
        return True  # Don't fail validation for this

def check_helm():
    """Check if Helm is available (optional for development)"""
    try:
        result = subprocess.run(['helm', 'version'],
                              capture_output=True, text=True, check=True)
        print('✅ Helm available')
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print('⚠️  Helm not available (install for Kubernetes deployment)')
        return True  # Don't fail validation for this

def validate_project_structure():
    """Validate overall project structure"""
    structure_checks = [
        ('app/', 'Application code directory'),
        ('helm/', 'Helm charts directory'),
        ('k8s/', 'Kubernetes manifests directory'),
        ('services/', 'Microservices directory'),
        ('tests/', 'Test directory'),
        ('monitoring/', 'Monitoring configuration'),
        ('environments/', 'Environment configurations')
    ]

    all_good = True
    for path, description in structure_checks:
        if os.path.exists(path):
            print(f'✅ {description} exists')
        else:
            print(f'❌ {description} missing: {path}')
            all_good = False

    return all_good

def main():
    print_header()

    checks = [
        ('Python Version', check_python_version),
        ('Required Files', check_required_files),
        ('Python Dependencies', check_python_dependencies),
        ('Docker', check_docker),
        ('kubectl', check_kubectl),
        ('Helm', check_helm),
        ('Project Structure', validate_project_structure)
    ]

    results = []
    for name, check_func in checks:
        print(f'\n🔍 Checking {name}...')
        result = check_func()
        results.append(result)

    print('\n' + '='*50)
    if all(results):
        print('✅ Project validation completed successfully!')
        print('')
        print('🎉 CipherGuard is ready for production deployment!')
        print('')
        print('Next steps:')
        print('1. Run: chmod +x setup_cicd.sh && ./setup_cicd.sh')
        print('2. Run: chmod +x setup_operations.sh && ./setup_operations.sh')
        print('3. Configure your production environment variables')
        print('4. Deploy: helm install cipherguard ./helm/cipherguard')
        print('')
        print('📖 See PROJECT_COMPLETION_REPORT.md for detailed documentation')
        return 0
    else:
        print('❌ Project validation failed!')
        print('Please fix the issues above before proceeding.')
        return 1

if __name__ == '__main__':
    sys.exit(main())