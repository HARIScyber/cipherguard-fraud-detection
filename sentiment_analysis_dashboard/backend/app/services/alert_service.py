"""
Customer Alert Notification Service
Sends real-time alerts to customers when suspicious transactions are detected.
Supports SMS, Email, and Push notifications (simulated for demo).
"""

import logging
import time
from datetime import datetime
from typing import Dict, Optional, List
from enum import Enum
import json

logger = logging.getLogger(__name__)


class AlertChannel(str, Enum):
    """Notification channels."""
    SMS = "sms"
    EMAIL = "email"
    PUSH = "push"
    ALL = "all"


class AlertType(str, Enum):
    """Types of fraud alerts."""
    SUSPICIOUS_TRANSACTION = "suspicious_transaction"
    HIGH_AMOUNT = "high_amount"
    UNUSUAL_LOCATION = "unusual_location"
    MULTIPLE_ATTEMPTS = "multiple_attempts"
    BLOCKED_TRANSACTION = "blocked_transaction"


class AlertService:
    """
    Service for sending real-time alerts to customers about suspicious transactions.
    In production, this would integrate with:
    - Twilio for SMS
    - SendGrid/AWS SES for Email
    - Firebase/APNs for Push notifications
    """
    
    def __init__(self):
        """Initialize the alert service."""
        self.alerts_sent: List[Dict] = []
        self.alert_templates = self._load_templates()
        logger.info("Alert Service initialized")
    
    def _load_templates(self) -> Dict[str, Dict[str, str]]:
        """Load notification templates."""
        return {
            AlertType.SUSPICIOUS_TRANSACTION: {
                "sms": "🚨 ALERT: Suspicious transaction of ${amount} at {merchant} detected on your account. If this wasn't you, reply BLOCK or call 1-800-SECURE immediately. - CipherGuard",
                "email_subject": "⚠️ Suspicious Transaction Alert - Action Required",
                "email_body": """
Dear {customer_name},

We detected a suspicious transaction on your account:

📋 Transaction Details:
• Amount: ${amount}
• Merchant: {merchant}
• Location: {country}
• Device: {device}
• Time: {timestamp}
• Risk Level: {risk_level}

🛡️ Fraud Score: {fraud_score}

If you DID NOT make this transaction:
1. Click here to BLOCK your card immediately: {block_link}
2. Call our 24/7 fraud hotline: 1-800-SECURE
3. Reply to this email with "BLOCK"

If you DID make this transaction:
• No action needed. You can mark it as safe in your app.

Stay protected,
CipherGuard Security Team
""",
                "push": "🚨 Suspicious ${amount} transaction at {merchant}. Tap to review."
            },
            AlertType.HIGH_AMOUNT: {
                "sms": "💰 ALERT: Large transaction of ${amount} detected at {merchant}. Confirm this is you by replying YES or call 1-800-SECURE. - CipherGuard",
                "email_subject": "💰 Large Transaction Alert - Please Verify",
                "email_body": """
Dear {customer_name},

A large transaction was detected on your account:

📋 Transaction Details:
• Amount: ${amount}
• Merchant: {merchant}
• Location: {country}

Please verify this transaction in your CipherGuard app or call us.

CipherGuard Security Team
""",
                "push": "💰 Large transaction: ${amount} at {merchant}. Tap to confirm."
            },
            AlertType.UNUSUAL_LOCATION: {
                "sms": "🌍 ALERT: Transaction from {country} detected. If you're not traveling, reply BLOCK. - CipherGuard",
                "email_subject": "🌍 Transaction from Unusual Location",
                "email_body": """
Dear {customer_name},

We noticed a transaction from an unusual location:

• Amount: ${amount}
• Location: {country}
• Your usual location: {usual_location}

If you're traveling, no action needed. Otherwise, please secure your account.

CipherGuard Security Team
""",
                "push": "🌍 Transaction from {country}. Is this you? Tap to verify."
            },
            AlertType.BLOCKED_TRANSACTION: {
                "sms": "🛑 BLOCKED: We stopped a ${amount} transaction at {merchant} that looked suspicious. Your card is safe. - CipherGuard",
                "email_subject": "🛑 Transaction Blocked for Your Protection",
                "email_body": """
Dear {customer_name},

Good news! We blocked a suspicious transaction:

• Amount: ${amount}
• Merchant: {merchant}
• Reason: {reason}

Your card remains active for legitimate purchases.

If this was actually you, please contact us to verify and retry.

CipherGuard Security Team
""",
                "push": "🛑 We blocked a ${amount} suspicious transaction. Your card is safe."
            }
        }
    
    def send_alert(
        self,
        customer_info: Dict,
        transaction: Dict,
        fraud_result: Dict,
        channel: AlertChannel = AlertChannel.ALL
    ) -> Dict:
        """
        Send fraud alert to customer.
        
        Args:
            customer_info: Customer details (name, email, phone)
            transaction: Transaction details
            fraud_result: Fraud detection results
            channel: Notification channel(s) to use
        
        Returns:
            Dict with alert status and details
        """
        start_time = time.time()
        
        # Determine alert type based on risk
        alert_type = self._determine_alert_type(transaction, fraud_result)
        
        # Prepare template data
        template_data = {
            "customer_name": customer_info.get("name", "Valued Customer"),
            "amount": f"{transaction.get('amount', 0):,.2f}",
            "merchant": transaction.get("merchant", "Unknown"),
            "country": transaction.get("country", "Unknown"),
            "device": transaction.get("device", "Unknown"),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "risk_level": fraud_result.get("risk_level", "UNKNOWN"),
            "fraud_score": f"{fraud_result.get('fraud_score', 0):.2f}",
            "transaction_id": fraud_result.get("transaction_id", "N/A"),
            "block_link": f"https://cipherguard.com/block/{fraud_result.get('transaction_id', 'N/A')}",
            "usual_location": customer_info.get("usual_location", "US"),
            "reason": self._get_block_reason(fraud_result)
        }
        
        # Send notifications
        results = {}
        
        if channel in [AlertChannel.SMS, AlertChannel.ALL]:
            results["sms"] = self._send_sms(
                customer_info.get("phone"),
                alert_type,
                template_data
            )
        
        if channel in [AlertChannel.EMAIL, AlertChannel.ALL]:
            results["email"] = self._send_email(
                customer_info.get("email"),
                alert_type,
                template_data
            )
        
        if channel in [AlertChannel.PUSH, AlertChannel.ALL]:
            results["push"] = self._send_push(
                customer_info.get("device_token"),
                alert_type,
                template_data
            )
        
        processing_time = (time.time() - start_time) * 1000
        
        # Create alert record
        alert_record = {
            "alert_id": f"alert_{int(time.time() * 1000)}",
            "transaction_id": fraud_result.get("transaction_id"),
            "customer_email": customer_info.get("email"),
            "customer_phone": customer_info.get("phone"),
            "alert_type": alert_type.value,
            "channels_used": list(results.keys()),
            "results": results,
            "timestamp": datetime.now().isoformat(),
            "processing_time_ms": processing_time
        }
        
        self.alerts_sent.append(alert_record)
        logger.info(f"Alert sent: {alert_record['alert_id']} via {list(results.keys())}")
        
        return alert_record
    
    def _determine_alert_type(self, transaction: Dict, fraud_result: Dict) -> AlertType:
        """Determine the type of alert based on transaction and fraud result."""
        risk_level = fraud_result.get("risk_level", "LOW")
        amount = transaction.get("amount", 0)
        country = transaction.get("country", "US")
        
        # Blocked transaction
        if risk_level == "CRITICAL":
            return AlertType.BLOCKED_TRANSACTION
        
        # High amount
        if amount > 5000:
            return AlertType.HIGH_AMOUNT
        
        # Unusual location
        if country not in ["US", "UK", "CA", "AU"]:
            return AlertType.UNUSUAL_LOCATION
        
        # Default suspicious
        return AlertType.SUSPICIOUS_TRANSACTION
    
    def _get_block_reason(self, fraud_result: Dict) -> str:
        """Get human-readable reason for blocking."""
        risk_level = fraud_result.get("risk_level", "UNKNOWN")
        score = fraud_result.get("fraud_score", 0)
        
        if risk_level == "CRITICAL":
            return "Transaction matched known fraud patterns"
        elif risk_level == "HIGH":
            return "Multiple risk indicators detected"
        elif score > 0.5:
            return "Unusual transaction behavior"
        else:
            return "Precautionary security measure"
    
    def _send_sms(self, phone: Optional[str], alert_type: AlertType, data: Dict) -> Dict:
        """
        Send SMS notification.
        In production: Integrate with Twilio, AWS SNS, or similar.
        """
        if not phone:
            return {"status": "skipped", "reason": "No phone number"}
        
        template = self.alert_templates.get(alert_type, {}).get("sms", "")
        message = template.format(**data)
        
        # Simulate SMS sending
        logger.info(f"📱 SMS to {phone}: {message[:50]}...")
        
        return {
            "status": "sent",
            "channel": "sms",
            "recipient": phone,
            "message_preview": message[:100] + "...",
            "timestamp": datetime.now().isoformat(),
            # In production, this would be the SMS provider's response
            "provider_response": {"message_id": f"sms_{int(time.time())}", "status": "delivered"}
        }
    
    def _send_email(self, email: Optional[str], alert_type: AlertType, data: Dict) -> Dict:
        """
        Send Email notification.
        In production: Integrate with SendGrid, AWS SES, or similar.
        """
        if not email:
            return {"status": "skipped", "reason": "No email address"}
        
        templates = self.alert_templates.get(alert_type, {})
        subject = templates.get("email_subject", "Security Alert").format(**data)
        body = templates.get("email_body", "").format(**data)
        
        # Simulate email sending
        logger.info(f"📧 Email to {email}: {subject}")
        
        return {
            "status": "sent",
            "channel": "email",
            "recipient": email,
            "subject": subject,
            "body_preview": body[:200] + "...",
            "timestamp": datetime.now().isoformat(),
            # In production, this would be the email provider's response
            "provider_response": {"message_id": f"email_{int(time.time())}", "status": "delivered"}
        }
    
    def _send_push(self, device_token: Optional[str], alert_type: AlertType, data: Dict) -> Dict:
        """
        Send Push notification.
        In production: Integrate with Firebase FCM, Apple APNs, or similar.
        """
        if not device_token:
            return {"status": "skipped", "reason": "No device token"}
        
        template = self.alert_templates.get(alert_type, {}).get("push", "")
        message = template.format(**data)
        
        # Simulate push notification
        logger.info(f"🔔 Push to device: {message}")
        
        return {
            "status": "sent",
            "channel": "push",
            "device_token": device_token[:20] + "...",
            "message": message,
            "timestamp": datetime.now().isoformat(),
            # In production, this would be the push provider's response
            "provider_response": {"message_id": f"push_{int(time.time())}", "status": "delivered"}
        }
    
    def get_alert_history(self, limit: int = 50) -> List[Dict]:
        """Get recent alert history."""
        return self.alerts_sent[-limit:][::-1]  # Return most recent first
    
    def get_alert_stats(self) -> Dict:
        """Get alert statistics."""
        total = len(self.alerts_sent)
        if total == 0:
            return {
                "total_alerts": 0,
                "by_type": {},
                "by_channel": {},
                "success_rate": 0
            }
        
        by_type = {}
        by_channel = {"sms": 0, "email": 0, "push": 0}
        successful = 0
        
        for alert in self.alerts_sent:
            # Count by type
            alert_type = alert.get("alert_type", "unknown")
            by_type[alert_type] = by_type.get(alert_type, 0) + 1
            
            # Count by channel
            for channel, result in alert.get("results", {}).items():
                if result.get("status") == "sent":
                    by_channel[channel] = by_channel.get(channel, 0) + 1
                    successful += 1
        
        return {
            "total_alerts": total,
            "by_type": by_type,
            "by_channel": by_channel,
            "success_rate": (successful / (total * 3)) * 100 if total > 0 else 0
        }


# Singleton instance
alert_service = AlertService()


def get_alert_service() -> AlertService:
    """Get the alert service instance."""
    return alert_service
