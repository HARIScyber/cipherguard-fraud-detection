"""
Production Readiness Validation Script
Run this to check if critical improvements are implemented
"""

import asyncio
import time
import requests
import json
from typing import Dict, List
from datetime import datetime

class ProductionReadinessChecker:
    """Validates production readiness of CipherGuard API."""
    
    def __init__(self, base_url: str = "http://localhost:8001"):
        self.base_url = base_url
        self.results = []
    
    def check_security_headers(self):
        """Check if security headers are implemented."""
        try:
            response = requests.get(f"{self.base_url}/health")
            
            required_headers = [
                'X-Content-Type-Options',
                'X-Frame-Options', 
                'X-XSS-Protection',
                'Strict-Transport-Security'
            ]
            
            missing_headers = []
            for header in required_headers:
                if header not in response.headers:
                    missing_headers.append(header)
            
            if missing_headers:
                return False, f"Missing security headers: {missing_headers}"
            else:
                return True, "All security headers present"
                
        except Exception as e:
            return False, f"Error checking security headers: {e}"
    
    def check_rate_limiting(self):
        """Check if rate limiting is working."""
        try:
            # Make rapid requests to trigger rate limiting
            for i in range(5):
                response = requests.get(f"{self.base_url}/health")
            
            # Check for rate limit headers
            if 'X-RateLimit-Remaining' in response.headers:
                return True, "Rate limiting headers found"
            else:
                return False, "No rate limiting headers detected"
                
        except Exception as e:
            return False, f"Error checking rate limiting: {e}"
    
    def check_input_validation(self):
        """Check enhanced input validation."""
        try:
            # Test invalid transaction
            invalid_transaction = {
                "amount": -100,  # Negative amount should be rejected
                "merchant": "",   # Empty merchant should be rejected
                "device": "invalid_device",
                "country": "INVALID"  # Invalid country code
            }
            
            response = requests.post(
                f"{self.base_url}/detect",
                json=invalid_transaction
            )
            
            # Should return 400 for validation error
            if response.status_code == 422:  # Pydantic validation error
                return True, "Input validation working (rejected invalid input)"
            else:
                return False, f"Invalid input accepted (status: {response.status_code})"
                
        except Exception as e:
            return False, f"Error checking input validation: {e}"
    
    def check_performance(self):
        """Check API performance."""
        try:
            valid_transaction = {
                "amount": 100.0,
                "merchant": "Amazon", 
                "device": "desktop",
                "country": "US"
            }
            
            # Measure response time
            start_time = time.time()
            response = requests.post(
                f"{self.base_url}/detect",
                json=valid_transaction
            )
            end_time = time.time()
            
            response_time_ms = (end_time - start_time) * 1000
            
            if response.status_code == 200 and response_time_ms < 100:
                return True, f"Performance good: {response_time_ms:.1f}ms"
            elif response.status_code == 200:
                return False, f"Performance slow: {response_time_ms:.1f}ms (target: <100ms)"
            else:
                return False, f"API error: {response.status_code}"
                
        except Exception as e:
            return False, f"Error checking performance: {e}"
    
    def check_health_endpoint(self):
        """Check enhanced health endpoint."""
        try:
            response = requests.get(f"{self.base_url}/health")
            
            if response.status_code != 200:
                return False, f"Health check failed: {response.status_code}"
            
            data = response.json()
            
            # Check if enhanced health info is present
            required_fields = ['status', 'timestamp', 'version']
            missing_fields = [field for field in required_fields if field not in data]
            
            if missing_fields:
                return False, f"Basic health endpoint (missing: {missing_fields})"
            else:
                return True, f"Enhanced health endpoint working"
                
        except Exception as e:
            return False, f"Error checking health endpoint: {e}"
    
    def run_all_checks(self) -> Dict[str, any]:
        """Run all production readiness checks."""
        checks = [
            ("Security Headers", self.check_security_headers),
            ("Rate Limiting", self.check_rate_limiting), 
            ("Input Validation", self.check_input_validation),
            ("API Performance", self.check_performance),
            ("Health Endpoint", self.check_health_endpoint)
        ]
        
        results = {}
        passed = 0
        total = len(checks)
        
        print("🛡️  CipherGuard Production Readiness Check")
        print("=" * 50)
        
        for check_name, check_func in checks:
            try:
                success, message = check_func()
                status = "✅ PASS" if success else "❌ FAIL"
                results[check_name] = {"passed": success, "message": message}
                
                print(f"{status} {check_name}: {message}")
                
                if success:
                    passed += 1
                    
            except Exception as e:
                results[check_name] = {"passed": False, "message": f"Error: {e}"}
                print(f"❌ FAIL {check_name}: Error: {e}")
        
        print("=" * 50)
        print(f"📊 Results: {passed}/{total} checks passed")
        
        if passed == total:
            print("🎉 Production ready!")
        elif passed >= total * 0.8:
            print("⚠️  Mostly ready - fix remaining issues")  
        else:
            print("🚨 Not production ready - critical issues need fixing")
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "passed": passed,
            "total": total,
            "percentage": round((passed / total) * 100, 1),
            "checks": results
        }

def main():
    """Main function to run all checks."""
    checker = ProductionReadinessChecker()
    results = checker.run_all_checks()
    
    # Save results to file
    with open("production_readiness_report.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📄 Detailed results saved to: production_readiness_report.json")

if __name__ == "__main__":
    main()