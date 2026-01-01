#!/usr/bin/env python3
"""
Production Deployment Script for CipherGuard Fraud Detection System
Automates the deployment process with validation and rollback capabilities
"""

import subprocess
import sys
import os
import json
import time
from datetime import datetime
from typing import Dict, List, Optional
import requests
import argparse

class ProductionDeployer:
    def __init__(self, namespace: str = "production", helm_chart_path: str = "./helm/cipherguard"):
        self.namespace = namespace
        self.helm_chart_path = helm_chart_path
        self.release_name = f"cipherguard-{namespace}"
        self.deployment_log = []

    def log(self, message: str, level: str = "INFO"):
        """Log deployment messages"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {level}: {message}"
        print(log_entry)
        self.deployment_log.append(log_entry)

    def run_command(self, command: str, check: bool = True) -> subprocess.CompletedProcess:
        """Execute shell command with logging"""
        self.log(f"Executing: {command}")
        try:
            result = subprocess.run(
                command.split(),
                capture_output=True,
                text=True,
                check=check
            )
            if result.stdout:
                self.log(f"Output: {result.stdout.strip()}")
            return result
        except subprocess.CalledProcessError as e:
            self.log(f"Command failed: {e.stderr}", "ERROR")
            raise

    def check_prerequisites(self) -> bool:
        """Check deployment prerequisites"""
        self.log("🔍 Checking deployment prerequisites...")

        checks = [
            ("kubectl", "kubectl version --client"),
            ("helm", "helm version"),
            ("docker", "docker --version"),
            ("kubernetes cluster", "kubectl cluster-info")
        ]

        for tool, command in checks:
            try:
                self.run_command(command)
                self.log(f"✅ {tool} is available")
            except:
                self.log(f"❌ {tool} is not available", "ERROR")
                return False

        # Check namespace exists
        try:
            self.run_command(f"kubectl get namespace {self.namespace}")
            self.log(f"✅ Namespace '{self.namespace}' exists")
        except:
            self.log(f"Creating namespace '{self.namespace}'")
            self.run_command(f"kubectl create namespace {self.namespace}")

        return True

    def validate_helm_chart(self) -> bool:
        """Validate Helm chart"""
        self.log("📋 Validating Helm chart...")

        try:
            self.run_command(f"helm lint {self.helm_chart_path}")
            self.log("✅ Helm chart validation passed")
            return True
        except:
            self.log("❌ Helm chart validation failed", "ERROR")
            return False

    def backup_current_deployment(self) -> Optional[str]:
        """Create backup of current deployment"""
        self.log("💾 Creating deployment backup...")

        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = f"backup_{self.release_name}_{timestamp}.yaml"

            # Export current deployment
            self.run_command(f"helm get manifest {self.release_name} -n {self.namespace} > {backup_file}")
            self.log(f"✅ Backup created: {backup_file}")
            return backup_file
        except:
            self.log("⚠️  Could not create backup", "WARNING")
            return None

    def deploy_helm_chart(self, values_file: Optional[str] = None, dry_run: bool = False) -> bool:
        """Deploy Helm chart"""
        self.log(f"🚀 Deploying Helm chart (dry-run: {dry_run})...")

        command = f"helm upgrade --install {self.release_name} {self.helm_chart_path} -n {self.namespace}"

        if values_file:
            command += f" -f {values_file}"

        if dry_run:
            command += " --dry-run"

        command += " --wait --timeout=600s"

        try:
            self.run_command(command)
            self.log("✅ Helm deployment successful")
            return True
        except:
            self.log("❌ Helm deployment failed", "ERROR")
            return False

    def wait_for_deployment(self, timeout_seconds: int = 300) -> bool:
        """Wait for deployment to be ready"""
        self.log(f"⏳ Waiting for deployment to be ready (timeout: {timeout_seconds}s)...")

        start_time = time.time()

        while time.time() - start_time < timeout_seconds:
            try:
                # Check pod status
                result = self.run_command(f"kubectl get pods -n {self.namespace} -l app.kubernetes.io/instance={self.release_name}")
                pods_output = result.stdout

                # Check if all pods are ready
                if "Running" in pods_output and "Error" not in pods_output and "CrashLoopBackOff" not in pods_output:
                    # Check API health
                    if self.check_api_health():
                        self.log("✅ All services are healthy")
                        return True

                time.sleep(10)

            except Exception as e:
                self.log(f"Error checking deployment status: {e}", "WARNING")
                time.sleep(10)

        self.log("❌ Deployment timeout or unhealthy", "ERROR")
        return False

    def check_api_health(self) -> bool:
        """Check API service health"""
        try:
            # Get service URL
            result = self.run_command(f"kubectl get svc -n {self.namespace} -l app.kubernetes.io/component=api -o jsonpath='{{.items[0].spec.clusterIP}}'")
            service_ip = result.stdout.strip()

            if not service_ip:
                return False

            # Check health endpoint
            response = requests.get(f"http://{service_ip}:8000/health", timeout=10)
            return response.status_code == 200

        except:
            return False

    def run_post_deployment_tests(self) -> bool:
        """Run post-deployment validation tests"""
        self.log("🧪 Running post-deployment tests...")

        tests_passed = 0
        total_tests = 0

        # Test 1: API health check
        total_tests += 1
        if self.check_api_health():
            self.log("✅ API health check passed")
            tests_passed += 1
        else:
            self.log("❌ API health check failed")

        # Test 2: Model prediction endpoint
        total_tests += 1
        try:
            # Get service URL
            result = self.run_command(f"kubectl get svc -n {self.namespace} -l app.kubernetes.io/component=api -o jsonpath='{{.items[0].spec.clusterIP}}'")
            service_ip = result.stdout.strip()

            test_payload = {
                "amount": 100.0,
                "merchant_id": "test_merchant",
                "user_id": "test_user",
                "timestamp": datetime.now().isoformat(),
                "features": {
                    "velocity_1h": 5.0,
                    "velocity_24h": 20.0,
                    "amount_deviation": 1.2,
                    "location_anomaly": 0.1,
                    "device_fingerprint": "test_fp"
                }
            }

            response = requests.post(
                f"http://{service_ip}:8000/predict",
                json=test_payload,
                timeout=30
            )

            if response.status_code == 200 and "fraud_probability" in response.json():
                self.log("✅ Model prediction test passed")
                tests_passed += 1
            else:
                self.log("❌ Model prediction test failed")

        except Exception as e:
            self.log(f"❌ Model prediction test error: {e}")

        # Test 3: Metrics endpoint
        total_tests += 1
        try:
            result = self.run_command(f"kubectl get svc -n {self.namespace} -l app.kubernetes.io/component=api -o jsonpath='{{.items[0].spec.clusterIP}}'")
            service_ip = result.stdout.strip()

            response = requests.get(f"http://{service_ip}:8000/metrics", timeout=10)
            if response.status_code == 200 and "prometheus" in response.text.lower():
                self.log("✅ Metrics endpoint test passed")
                tests_passed += 1
            else:
                self.log("❌ Metrics endpoint test failed")

        except Exception as e:
            self.log(f"❌ Metrics endpoint test error: {e}")

        success_rate = tests_passed / total_tests
        self.log(f"Test Results: {tests_passed}/{total_tests} passed ({success_rate:.1%})")

        return success_rate >= 0.8  # Require 80% success rate

    def rollback_deployment(self, backup_file: Optional[str] = None) -> bool:
        """Rollback deployment"""
        self.log("🔄 Rolling back deployment...")

        try:
            if backup_file and os.path.exists(backup_file):
                # Apply backup manifest
                self.run_command(f"kubectl apply -f {backup_file} -n {self.namespace}")
            else:
                # Helm rollback
                self.run_command(f"helm rollback {self.release_name} -n {self.namespace}")

            self.log("✅ Rollback successful")
            return True
        except Exception as e:
            self.log(f"❌ Rollback failed: {e}", "ERROR")
            return False

    def cleanup_old_resources(self):
        """Clean up old resources"""
        self.log("🧹 Cleaning up old resources...")

        try:
            # Remove completed jobs
            self.run_command(f"kubectl delete jobs -n {self.namespace} --field-selector status.successful=1")

            # Remove failed pods
            self.run_command(f"kubectl delete pods -n {self.namespace} --field-selector status.phase=Failed")

            self.log("✅ Cleanup completed")
        except:
            self.log("⚠️  Cleanup partially failed", "WARNING")

    def deploy(self, values_file: Optional[str] = None, skip_tests: bool = False) -> bool:
        """Execute full deployment pipeline"""
        self.log("🎯 Starting CipherGuard Production Deployment")
        self.log("=" * 60)

        success = False
        backup_file = None

        try:
            # Phase 1: Prerequisites
            if not self.check_prerequisites():
                return False

            # Phase 2: Validation
            if not self.validate_helm_chart():
                return False

            # Phase 3: Backup
            backup_file = self.backup_current_deployment()

            # Phase 4: Dry run
            if not self.deploy_helm_chart(values_file, dry_run=True):
                return False

            # Phase 5: Actual deployment
            if not self.deploy_helm_chart(values_file, dry_run=False):
                return False

            # Phase 6: Wait for readiness
            if not self.wait_for_deployment():
                return False

            # Phase 7: Post-deployment tests
            if not skip_tests and not self.run_post_deployment_tests():
                self.log("❌ Post-deployment tests failed, rolling back...")
                self.rollback_deployment(backup_file)
                return False

            # Phase 8: Cleanup
            self.cleanup_old_resources()

            success = True
            self.log("🎉 Deployment completed successfully!")

        except Exception as e:
            self.log(f"💥 Deployment failed with error: {e}", "ERROR")
            if not success:
                self.rollback_deployment(backup_file)

        finally:
            # Save deployment log
            log_filename = f"deployment_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(log_filename, 'w') as f:
                f.write('\n'.join(self.deployment_log))
            self.log(f"📄 Deployment log saved to {log_filename}")

        return success

def main():
    parser = argparse.ArgumentParser(description="CipherGuard Production Deployment")
    parser.add_argument("--namespace", default="production", help="Kubernetes namespace")
    parser.add_argument("--values", help="Custom values file")
    parser.add_argument("--chart-path", default="./helm/cipherguard", help="Helm chart path")
    parser.add_argument("--skip-tests", action="store_true", help="Skip post-deployment tests")
    parser.add_argument("--dry-run", action="store_true", help="Perform dry run only")

    args = parser.parse_args()

    deployer = ProductionDeployer(args.namespace, args.chart_path)

    if args.dry_run:
        deployer.log("🔍 Performing dry run deployment...")
        success = deployer.deploy_helm_chart(args.values, dry_run=True)
    else:
        success = deployer.deploy(args.values, args.skip_tests)

    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()