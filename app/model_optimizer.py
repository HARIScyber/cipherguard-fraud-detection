"""
Model Optimization Module - Phase 4: Production Deployment
ONNX conversion and model optimization for production inference
"""

import os
import numpy as np
import onnxruntime as ort
import onnx
from onnx import helper, TensorProto
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import xgboost as xgb
from sklearn.ensemble import IsolationForest, RandomForestClassifier
import joblib
import logging
from typing import Dict, Any, Optional, Tuple
import time

logger = logging.getLogger(__name__)

class ModelOptimizer:
    """Handles model optimization and ONNX conversion for production deployment."""

    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        self.onnx_dir = os.path.join(models_dir, "onnx")
        os.makedirs(self.onnx_dir, exist_ok=True)

        # Initialize ONNX Runtime sessions
        self.sessions = {}

    def convert_isolation_forest_to_onnx(self, model: IsolationForest, model_name: str) -> str:
        """Convert Isolation Forest to ONNX format."""
        try:
            # Create sample input for ONNX conversion
            sample_input = np.random.rand(1, 8).astype(np.float32)

            # Convert using skl2onnx
            initial_type = [('input', FloatTensorType([None, 8]))]
            onnx_model = convert_sklearn(model, initial_types=initial_type, target_opset=12)

            # Save ONNX model
            onnx_path = os.path.join(self.onnx_dir, f"{model_name}_isolation_forest.onnx")
            onnx.save(onnx_model, onnx_path)

            logger.info(f"Converted Isolation Forest to ONNX: {onnx_path}")
            return onnx_path

        except Exception as e:
            logger.error(f"Failed to convert Isolation Forest to ONNX: {e}")
            return None

    def convert_random_forest_to_onnx(self, model: RandomForestClassifier, model_name: str) -> str:
        """Convert Random Forest to ONNX format."""
        try:
            # Create sample input for ONNX conversion
            sample_input = np.random.rand(1, 8).astype(np.float32)

            # Convert using skl2onnx
            initial_type = [('input', FloatTensorType([None, 8]))]
            onnx_model = convert_sklearn(model, initial_types=initial_type, target_opset=12)

            # Save ONNX model
            onnx_path = os.path.join(self.onnx_dir, f"{model_name}_random_forest.onnx")
            onnx.save(onnx_model, onnx_path)

            logger.info(f"Converted Random Forest to ONNX: {onnx_path}")
            return onnx_path

        except Exception as e:
            logger.error(f"Failed to convert Random Forest to ONNX: {e}")
            return None

    def convert_xgboost_to_onnx(self, model: xgb.XGBClassifier, model_name: str) -> str:
        """Convert XGBoost to ONNX format."""
        try:
            # Create sample input for ONNX conversion
            sample_input = np.random.rand(1, 8).astype(np.float32)

            # Convert using skl2onnx
            initial_type = [('input', FloatTensorType([None, 8]))]
            onnx_model = convert_sklearn(model, initial_types=initial_type, target_opset=12)

            # Save ONNX model
            onnx_path = os.path.join(self.onnx_dir, f"{model_name}_xgboost.onnx")
            onnx.save(onnx_model, onnx_path)

            logger.info(f"Converted XGBoost to ONNX: {onnx_path}")
            return onnx_path

        except Exception as e:
            logger.error(f"Failed to convert XGBoost to ONNX: {e}")
            return None

    def create_ensemble_onnx_model(self, models: Dict[str, Any], model_name: str) -> str:
        """Create a custom ONNX ensemble model."""
        try:
            # This is a simplified ensemble - in production you'd create a proper ONNX graph
            # For now, we'll just save the model paths for runtime ensemble
            ensemble_config = {
                "model_name": model_name,
                "models": list(models.keys()),
                "weights": {
                    'isolation_forest': 0.25,
                    'random_forest': 0.375,
                    'xgboost': 0.375
                },
                "onnx_paths": {}
            }

            # Convert individual models
            for model_type, model in models.items():
                if model_type == 'isolation_forest':
                    path = self.convert_isolation_forest_to_onnx(model, model_name)
                elif model_type == 'random_forest':
                    path = self.convert_random_forest_to_onnx(model, model_name)
                elif model_type == 'xgboost':
                    path = self.convert_xgboost_to_onnx(model, model_name)
                else:
                    continue

                if path:
                    ensemble_config["onnx_paths"][model_type] = path

            # Save ensemble configuration
            config_path = os.path.join(self.onnx_dir, f"{model_name}_ensemble_config.json")
            import json
            with open(config_path, 'w') as f:
                json.dump(ensemble_config, f, indent=2)

            logger.info(f"Created ensemble ONNX config: {config_path}")
            return config_path

        except Exception as e:
            logger.error(f"Failed to create ensemble ONNX model: {e}")
            return None

    def load_onnx_session(self, model_path: str) -> Optional[ort.InferenceSession]:
        """Load ONNX model into inference session."""
        try:
            if model_path not in self.sessions:
                # Configure session options for performance
                sess_options = ort.SessionOptions()
                sess_options.intra_op_num_threads = 1  # Optimize for single-threaded inference
                sess_options.inter_op_num_threads = 1

                self.sessions[model_path] = ort.InferenceSession(
                    model_path,
                    sess_options=sess_options,
                    providers=['CPUExecutionProvider']  # Use CPU by default
                )

            return self.sessions[model_path]

        except Exception as e:
            logger.error(f"Failed to load ONNX session for {model_path}: {e}")
            return None

    def predict_with_onnx(self, session: ort.InferenceSession, input_data: np.ndarray) -> np.ndarray:
        """Run inference with ONNX model."""
        try:
            # Prepare input
            input_data = input_data.astype(np.float32)
            if input_data.ndim == 1:
                input_data = input_data.reshape(1, -1)

            # Get input name from model
            input_name = session.get_inputs()[0].name

            # Run inference
            result = session.run(None, {input_name: input_data})

            return np.array(result[0])

        except Exception as e:
            logger.error(f"ONNX inference failed: {e}")
            return None

    def benchmark_inference(self, model_func, input_data: np.ndarray, n_runs: int = 100) -> Dict[str, float]:
        """Benchmark inference performance."""
        try:
            # Warmup
            for _ in range(10):
                _ = model_func(input_data)

            # Benchmark
            start_time = time.time()
            for _ in range(n_runs):
                _ = model_func(input_data)
            end_time = time.time()

            avg_time = (end_time - start_time) / n_runs * 1000  # Convert to milliseconds
            throughput = n_runs / (end_time - start_time)  # inferences per second

            return {
                "avg_inference_time_ms": round(avg_time, 3),
                "throughput_ips": round(throughput, 2),
                "total_time_seconds": round(end_time - start_time, 3)
            }

        except Exception as e:
            logger.error(f"Benchmark failed: {e}")
            return {"error": str(e)}

    def optimize_model_size(self, model_path: str) -> str:
        """Optimize ONNX model size (simplified version)."""
        try:
            # Load and optimize model
            model = onnx.load(model_path)

            # Apply basic optimizations
            optimized_model = onnx.optimizer.optimize(model, [
                'eliminate_deadend',
                'eliminate_nop_transpose',
                'fuse_consecutive_transposes'
            ])

            # Save optimized model
            optimized_path = model_path.replace('.onnx', '_optimized.onnx')
            onnx.save(optimized_model, optimized_path)

            original_size = os.path.getsize(model_path)
            optimized_size = os.path.getsize(optimized_path)
            compression_ratio = optimized_size / original_size

            logger.info(f"Optimized model: {original_size} → {optimized_size} bytes ({compression_ratio:.2%})")

            return optimized_path

        except Exception as e:
            logger.error(f"Model optimization failed: {e}")
            return model_path

# Global optimizer instance
model_optimizer = ModelOptimizer()

def get_model_optimizer() -> ModelOptimizer:
    """Get the global model optimizer instance."""
    return model_optimizer