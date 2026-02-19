"""
Enterprise Integration Test - Phase 6 Validation
Test script to validate enterprise features integration
"""

import asyncio
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

async def test_enterprise_features():
    """Test enterprise features availability."""
    print("Testing Phase 6: Enterprise Integration Features")
    print("=" * 50)

    # Test imports
    features_status = {}

    # Payment Gateways
    try:
        from app.payment_gateways import get_payment_gateway_manager
        features_status["payment_gateways"] = True
        print("✓ Payment Gateways module imported successfully")
    except ImportError as e:
        features_status["payment_gateways"] = False
        print(f"✗ Payment Gateways failed: {e}")

    # Enterprise Security
    try:
        from app.enterprise_security import get_security_manager, get_jwt_manager
        features_status["enterprise_security"] = True
        print("✓ Enterprise Security module imported successfully")
    except ImportError as e:
        features_status["enterprise_security"] = False
        print(f"✗ Enterprise Security failed: {e}")

    # Compliance & Audit
    try:
        from app.compliance_audit import get_audit_logger, get_compliance_checker
        features_status["compliance_audit"] = True
        print("✓ Compliance & Audit module imported successfully")
    except ImportError as e:
        features_status["compliance_audit"] = False
        print(f"✗ Compliance & Audit failed: {e}")

    # Multi-Tenant
    try:
        from app.multi_tenant import get_tenant_manager, tenant_context
        features_status["multi_tenant"] = True
        print("✓ Multi-Tenant module imported successfully")
    except ImportError as e:
        features_status["multi_tenant"] = False
        print(f"✗ Multi-Tenant failed: {e}")

    # Enterprise Dashboard
    try:
        from app.enterprise_dashboard import get_dashboard_manager, get_alert_manager
        features_status["enterprise_dashboard"] = True
        print("✓ Enterprise Dashboard module imported successfully")
    except ImportError as e:
        features_status["enterprise_dashboard"] = False
        print(f"✗ Enterprise Dashboard failed: {e}")

    # Main App Integration
    try:
        from app.main import PHASE6_FEATURES_AVAILABLE
        features_status["main_integration"] = PHASE6_FEATURES_AVAILABLE
        if PHASE6_FEATURES_AVAILABLE:
            print("✓ Main application Phase 6 integration active")
        else:
            print("⚠ Main application Phase 6 integration inactive (missing dependencies)")
    except ImportError as e:
        features_status["main_integration"] = False
        print(f"✗ Main application integration failed: {e}")

    print("\n" + "=" * 50)
    print("Enterprise Features Summary:")
    print("=" * 50)

    total_features = len(features_status)
    working_features = sum(1 for status in features_status.values() if status)

    for feature, status in features_status.items():
        status_icon = "✓" if status else "✗"
        print(f"{status_icon} {feature.replace('_', ' ').title()}: {'Available' if status else 'Not Available'}")

    print(f"\nOverall Status: {working_features}/{total_features} features working")

    if working_features == total_features:
        print(" All enterprise features successfully integrated!")
        return True
    else:
        print("⚠️ Some enterprise features require dependency installation")
        print("Run: pip install aiohttp PyJWT stripe requests")
        return False

if __name__ == "__main__":
    asyncio.run(test_enterprise_features())