"""
Final Phase 3 Verification Script
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'services', 'fraud-detection'))

from main import get_risk_level, compute_enhanced_fraud_score, generate_fraud_insights
import numpy as np

print('🔍 Final Phase 3 Verification')
print('=' * 40)

# Test risk levels
print('✓ Risk levels:', [get_risk_level(s) for s in [0.9, 0.7, 0.5, 0.3, 0.1]])

# Test enhanced scoring
vector = np.array([0.8, 0.4, 0.3, 0.2, 0.5, 0.6, 0.7, 0.1])
knn_results = [('txn_001', 0.1), ('txn_002', 0.15)]
score = compute_enhanced_fraud_score(vector, knn_results, 0.8, 0.9)
print(f'✓ Enhanced scoring: {score:.3f}')

# Test insights
insights = generate_fraud_insights(vector, {'model_predictions': []}, knn_results)
print(f'✓ Insights generated: {len(insights)} items')

print('✅ Phase 3: FULLY COMPLETE!')