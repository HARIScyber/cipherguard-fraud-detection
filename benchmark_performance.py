#!/usr/bin/env python3
"""
Performance Benchmarking Suite for CipherGuard Fraud Detection System
Tests model inference performance, throughput, and resource utilization
"""

import asyncio
import time
import statistics
import json
import requests
import psutil
import os
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any
import numpy as np
from datetime import datetime

class PerformanceBenchmark:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.results = {}

    def generate_test_transaction(self) -> Dict[str, Any]:
        """Generate a realistic test transaction for benchmarking"""
        return {
            "amount": np.random.uniform(10, 10000),
            "merchant_id": f"merchant_{np.random.randint(1, 1000)}",
            "user_id": f"user_{np.random.randint(1, 10000)}",
            "timestamp": datetime.now().isoformat(),
            "location": {
                "country": np.random.choice(["US", "UK", "DE", "FR", "CA"]),
                "city": f"city_{np.random.randint(1, 100)}"
            },
            "device_info": {
                "device_type": np.random.choice(["mobile", "desktop", "tablet"]),
                "browser": np.random.choice(["chrome", "firefox", "safari", "edge"])
            },
            "payment_method": np.random.choice(["credit_card", "debit_card", "paypal", "bank_transfer"]),
            "features": {
                "velocity_1h": np.random.uniform(0, 50),
                "velocity_24h": np.random.uniform(0, 200),
                "amount_deviation": np.random.uniform(0, 5),
                "location_anomaly": np.random.uniform(0, 1),
                "device_fingerprint": f"fp_{np.random.randint(1, 1000000)}"
            }
        }

    async def benchmark_inference_latency(self, num_requests: int = 1000) -> Dict[str, Any]:
        """Benchmark model inference latency"""
        print(f"🔬 Benchmarking inference latency with {num_requests} requests...")

        latencies = []
        errors = 0

        async def single_request():
            nonlocal errors
            transaction = self.generate_test_transaction()
            start_time = time.time()

            try:
                response = requests.post(
                    f"{self.base_url}/predict",
                    json=transaction,
                    timeout=30
                )
                latency = time.time() - start_time

                if response.status_code == 200:
                    latencies.append(latency)
                else:
                    errors += 1
            except Exception as e:
                errors += 1
                print(f"Request error: {e}")

        # Run requests concurrently
        semaphore = asyncio.Semaphore(50)  # Limit concurrent requests

        async def limited_request():
            async with semaphore:
                await single_request()

        tasks = [limited_request() for _ in range(num_requests)]
        await asyncio.gather(*tasks)

        if latencies:
            return {
                "total_requests": num_requests,
                "successful_requests": len(latencies),
                "error_rate": errors / num_requests,
                "mean_latency": statistics.mean(latencies),
                "median_latency": statistics.median(latencies),
                "p95_latency": np.percentile(latencies, 95),
                "p99_latency": np.percentile(latencies, 99),
                "min_latency": min(latencies),
                "max_latency": max(latencies),
                "requests_per_second": len(latencies) / sum(latencies)
            }
        else:
            return {"error": "No successful requests"}

    def benchmark_throughput(self, duration_seconds: int = 60) -> Dict[str, Any]:
        """Benchmark system throughput"""
        print(f"📊 Benchmarking throughput for {duration_seconds} seconds...")

        successful_requests = 0
        errors = 0
        latencies = []
        start_time = time.time()

        def worker():
            nonlocal successful_requests, errors
            while time.time() - start_time < duration_seconds:
                transaction = self.generate_test_transaction()
                req_start = time.time()

                try:
                    response = requests.post(
                        f"{self.base_url}/predict",
                        json=transaction,
                        timeout=10
                    )

                    if response.status_code == 200:
                        successful_requests += 1
                        latencies.append(time.time() - req_start)
                    else:
                        errors += 1
                except:
                    errors += 1

        # Run with multiple threads
        num_threads = min(20, os.cpu_count() * 2)
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(worker) for _ in range(num_threads)]
            for future in futures:
                future.result()

        total_time = time.time() - start_time
        total_requests = successful_requests + errors

        return {
            "duration_seconds": total_time,
            "total_requests": total_requests,
            "successful_requests": successful_requests,
            "error_rate": errors / total_requests if total_requests > 0 else 0,
            "requests_per_second": successful_requests / total_time,
            "mean_latency": statistics.mean(latencies) if latencies else 0,
            "p95_latency": np.percentile(latencies, 95) if latencies else 0
        }

    def benchmark_resource_utilization(self, duration_seconds: int = 30) -> Dict[str, Any]:
        """Benchmark resource utilization during load"""
        print(f"💻 Benchmarking resource utilization for {duration_seconds} seconds...")

        cpu_percentages = []
        memory_percentages = []
        start_time = time.time()

        # Start background load
        def generate_load():
            while time.time() - start_time < duration_seconds:
                transaction = self.generate_test_transaction()
                try:
                    requests.post(f"{self.base_url}/predict", json=transaction, timeout=5)
                except:
                    pass

        with ThreadPoolExecutor(max_workers=10) as executor:
            load_future = executor.submit(generate_load)

            # Monitor resources
            while time.time() - start_time < duration_seconds:
                cpu_percentages.append(psutil.cpu_percent(interval=1))
                memory_percentages.append(psutil.virtual_memory().percent)
                time.sleep(1)

            load_future.result()

        return {
            "cpu_utilization": {
                "mean": statistics.mean(cpu_percentages),
                "max": max(cpu_percentages),
                "min": min(cpu_percentages)
            },
            "memory_utilization": {
                "mean": statistics.mean(memory_percentages),
                "max": max(memory_percentages),
                "min": min(memory_percentages)
            }
        }

    def benchmark_model_accuracy(self, num_samples: int = 1000) -> Dict[str, Any]:
        """Benchmark model accuracy on test data"""
        print(f"🎯 Benchmarking model accuracy with {num_samples} samples...")

        predictions = []
        ground_truth = []

        for _ in range(num_samples):
            transaction = self.generate_test_transaction()
            # Simulate ground truth (random for demo, use real labels in production)
            is_fraud = np.random.choice([0, 1], p=[0.95, 0.05])  # 5% fraud rate

            try:
                response = requests.post(
                    f"{self.base_url}/predict",
                    json=transaction,
                    timeout=10
                )

                if response.status_code == 200:
                    result = response.json()
                    predictions.append(result.get("fraud_probability", 0) > 0.5)
                    ground_truth.append(bool(is_fraud))
            except Exception as e:
                print(f"Accuracy test error: {e}")

        if predictions and ground_truth:
            correct_predictions = sum(p == g for p, g in zip(predictions, ground_truth))
            accuracy = correct_predictions / len(predictions)

            return {
                "total_samples": len(predictions),
                "accuracy": accuracy,
                "precision": sum(p and g for p, g in zip(predictions, ground_truth)) / sum(predictions) if sum(predictions) > 0 else 0,
                "recall": sum(p and g for p, g in zip(predictions, ground_truth)) / sum(ground_truth) if sum(ground_truth) > 0 else 0
            }
        else:
            return {"error": "No predictions collected"}

    async def run_full_benchmark(self) -> Dict[str, Any]:
        """Run complete performance benchmark suite"""
        print("🚀 Starting CipherGuard Performance Benchmark Suite")
        print("=" * 60)

        self.results = {
            "timestamp": datetime.now().isoformat(),
            "inference_latency": await self.benchmark_inference_latency(),
            "throughput": self.benchmark_throughput(),
            "resource_utilization": self.benchmark_resource_utilization(),
            "model_accuracy": self.benchmark_model_accuracy()
        }

        return self.results

    def save_results(self, filename: str = "benchmark_results.json"):
        """Save benchmark results to file"""
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"📄 Results saved to {filename}")

    def print_summary(self):
        """Print benchmark summary"""
        if not self.results:
            print("No results available. Run benchmark first.")
            return

        print("\n" + "=" * 60)
        print("📊 CIPHERGUARD PERFORMANCE BENCHMARK SUMMARY")
        print("=" * 60)

        # Inference Latency
        lat = self.results.get("inference_latency", {})
        if "mean_latency" in lat:
            print("\n🔬 Inference Latency:")
            print(".3f")
            print(".3f")
            print(".3f")
        # Throughput
        tp = self.results.get("throughput", {})
        if "requests_per_second" in tp:
            print("\n📈 Throughput:")
            print(".1f")
            print(".3f")
            print(".1f")
        # Resources
        res = self.results.get("resource_utilization", {})
        cpu = res.get("cpu_utilization", {})
        mem = res.get("memory_utilization", {})
        if cpu:
            print("\n💻 Resource Utilization:")
            print(".1f")
            print(".1f")
            print(".1f")
            print(".1f")
        # Accuracy
        acc = self.results.get("model_accuracy", {})
        if "accuracy" in acc:
            print("\n🎯 Model Accuracy:")
            print(".3f")
            print(".3f")
            print(".3f")
        print("\n" + "=" * 60)

async def main():
    """Main benchmark execution"""
    import argparse

    parser = argparse.ArgumentParser(description="CipherGuard Performance Benchmark")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--output", default="benchmark_results.json", help="Output file")
    args = parser.parse_args()

    benchmark = PerformanceBenchmark(args.url)

    try:
        results = await benchmark.run_full_benchmark()
        benchmark.save_results(args.output)
        benchmark.print_summary()

    except KeyboardInterrupt:
        print("\n⏹️  Benchmark interrupted by user")
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())