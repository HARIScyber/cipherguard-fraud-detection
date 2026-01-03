"""
Payment Gateway Connectors - Phase 6: Enterprise Integration
Real payment gateway integrations for production fraud detection
"""

import asyncio
import aiohttp
import json
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import hashlib
import hmac
import base64
from dataclasses import dataclass
import os

logger = logging.getLogger(__name__)

@dataclass
class PaymentTransaction:
    """Standardized payment transaction data."""
    transaction_id: str
    amount: float
    currency: str
    merchant_id: str
    customer_id: str
    payment_method: str
    status: str
    timestamp: datetime
    metadata: Dict[str, Any]
    risk_score: Optional[float] = None

class PaymentGatewayConnector:
    """Abstract base class for payment gateway connectors."""

    def __init__(self, api_key: str, api_secret: str, base_url: str):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url
        self.session: Optional[aiohttp.ClientSession] = None

    async def connect(self) -> bool:
        """Establish connection to payment gateway."""
        try:
            headers = self._get_auth_headers()
            self.session = aiohttp.ClientSession(headers=headers)
            # Test connection
            async with self.session.get(f"{self.base_url}/health") as response:
                return response.status == 200
        except Exception as e:
            logger.error(f"Failed to connect to payment gateway: {e}")
            return False

    async def fetch_transactions(self, start_time: datetime, end_time: datetime,
                               limit: int = 100) -> List[PaymentTransaction]:
        """Fetch transactions from payment gateway."""
        if not self.session:
            raise ConnectionError("Not connected to payment gateway")

        params = {
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'limit': limit
        }

        try:
            async with self.session.get(f"{self.base_url}/transactions", params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return [self._parse_transaction(tx) for tx in data.get('transactions', [])]
                else:
                    logger.error(f"Failed to fetch transactions: {response.status}")
                    return []
        except Exception as e:
            logger.error(f"Error fetching transactions: {e}")
            return []

    async def get_transaction_details(self, transaction_id: str) -> Optional[PaymentTransaction]:
        """Get detailed transaction information."""
        if not self.session:
            raise ConnectionError("Not connected to payment gateway")

        try:
            async with self.session.get(f"{self.base_url}/transactions/{transaction_id}") as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_transaction(data)
                else:
                    logger.error(f"Failed to get transaction details: {response.status}")
                    return None
        except Exception as e:
            logger.error(f"Error getting transaction details: {e}")
            return None

    def _get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers for API requests."""
        return {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }

    def _parse_transaction(self, data: Dict[str, Any]) -> PaymentTransaction:
        """Parse gateway-specific transaction data into standardized format."""
        raise NotImplementedError("Subclasses must implement _parse_transaction")

    async def close(self):
        """Close the connection."""
        if self.session:
            await self.session.close()

class StripeConnector(PaymentGatewayConnector):
    """Stripe payment gateway connector."""

    def __init__(self, api_key: str, webhook_secret: str = None):
        super().__init__(api_key, "", "https://api.stripe.com/v1")
        self.webhook_secret = webhook_secret

    def _get_auth_headers(self) -> Dict[str, str]:
        """Stripe uses API key in authorization header."""
        return {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }

    def _parse_transaction(self, data: Dict[str, Any]) -> PaymentTransaction:
        """Parse Stripe transaction data."""
        return PaymentTransaction(
            transaction_id=data.get('id', ''),
            amount=data.get('amount', 0) / 100,  # Convert from cents
            currency=data.get('currency', 'usd').upper(),
            merchant_id=data.get('metadata', {}).get('merchant_id', ''),
            customer_id=data.get('customer', ''),
            payment_method=data.get('payment_method_types', ['card'])[0],
            status=self._map_stripe_status(data.get('status', '')),
            timestamp=datetime.fromtimestamp(data.get('created', 0)),
            metadata=data,
            risk_score=self._extract_risk_score(data)
        )

    def _map_stripe_status(self, stripe_status: str) -> str:
        """Map Stripe status to standardized status."""
        status_mapping = {
            'succeeded': 'completed',
            'pending': 'pending',
            'failed': 'failed',
            'canceled': 'cancelled'
        }
        return status_mapping.get(stripe_status, 'unknown')

    def _extract_risk_score(self, data: Dict[str, Any]) -> Optional[float]:
        """Extract risk score from Stripe Radar data."""
        outcomes = data.get('outcome', {})
        risk_level = outcomes.get('risk_level', '')
        risk_score_mapping = {
            'normal': 0.1,
            'elevated': 0.5,
            'highest': 0.9
        }
        return risk_score_mapping.get(risk_level)

    def verify_webhook_signature(self, payload: bytes, signature: str) -> bool:
        """Verify Stripe webhook signature."""
        if not self.webhook_secret:
            return False

        try:
            timestamp, signatures = signature.split(',', 1)
            timestamp = timestamp.split('=')[1]

            signed_payload = f"{timestamp}.{payload.decode('utf-8')}"
            expected_signature = hmac.new(
                self.webhook_secret.encode('utf-8'),
                signed_payload.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()

            for sig in signatures.split(','):
                if sig.split('=')[1] == expected_signature:
                    return True
            return False
        except Exception as e:
            logger.error(f"Webhook signature verification failed: {e}")
            return False

class PayPalConnector(PaymentGatewayConnector):
    """PayPal payment gateway connector."""

    def __init__(self, client_id: str, client_secret: str, sandbox: bool = True):
        base_url = "https://api-m.sandbox.paypal.com" if sandbox else "https://api-m.paypal.com"
        super().__init__(client_id, client_secret, base_url)
        self.access_token = None
        self.token_expires = None

    async def connect(self) -> bool:
        """Establish connection and get access token."""
        try:
            # Get access token
            auth = base64.b64encode(f"{self.api_key}:{self.api_secret}".encode()).decode()
            headers = {
                'Authorization': f'Basic {auth}',
                'Content-Type': 'application/x-www-form-urlencoded'
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(f"{self.base_url}/v1/oauth2/token",
                                      headers=headers,
                                      data='grant_type=client_credentials') as response:
                    if response.status == 200:
                        token_data = await response.json()
                        self.access_token = token_data.get('access_token')
                        expires_in = token_data.get('expires_in', 3600)
                        self.token_expires = datetime.now() + timedelta(seconds=expires_in)

                        # Create session with token
                        self.session = aiohttp.ClientSession(
                            headers={'Authorization': f'Bearer {self.access_token}'}
                        )
                        return True
                    else:
                        logger.error(f"Failed to get PayPal access token: {response.status}")
                        return False
        except Exception as e:
            logger.error(f"Failed to connect to PayPal: {e}")
            return False

    def _parse_transaction(self, data: Dict[str, Any]) -> PaymentTransaction:
        """Parse PayPal transaction data."""
        transactions = data.get('transaction_info', [{}])
        tx_info = transactions[0] if transactions else {}

        return PaymentTransaction(
            transaction_id=data.get('id', ''),
            amount=float(tx_info.get('transaction_amount', {}).get('value', 0)),
            currency=tx_info.get('transaction_amount', {}).get('currency_code', 'USD'),
            merchant_id=data.get('merchant_id', ''),
            customer_id=data.get('payer_info', {}).get('payer_id', ''),
            payment_method='paypal',
            status=self._map_paypal_status(data.get('state', '')),
            timestamp=datetime.fromisoformat(data.get('create_time', '').replace('Z', '+00:00')),
            metadata=data
        )

    def _map_paypal_status(self, paypal_status: str) -> str:
        """Map PayPal status to standardized status."""
        status_mapping = {
            'completed': 'completed',
            'pending': 'pending',
            'failed': 'failed',
            'denied': 'failed',
            'expired': 'cancelled',
            'voided': 'cancelled'
        }
        return status_mapping.get(paypal_status, 'unknown')

class SquareConnector(PaymentGatewayConnector):
    """Square payment gateway connector."""

    def __init__(self, access_token: str, environment: str = 'sandbox'):
        base_url = f"https://{environment}.squareup.com"
        super().__init__(access_token, "", base_url)

    def _get_auth_headers(self) -> Dict[str, str]:
        """Square uses Bearer token authentication."""
        return {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
            'Square-Version': '2023-10-18'
        }

    def _parse_transaction(self, data: Dict[str, Any]) -> PaymentTransaction:
        """Parse Square transaction data."""
        return PaymentTransaction(
            transaction_id=data.get('id', ''),
            amount=data.get('amount_money', {}).get('amount', 0) / 100,  # Convert from cents
            currency=data.get('amount_money', {}).get('currency', 'USD'),
            merchant_id=data.get('merchant_id', ''),
            customer_id=data.get('customer_id', ''),
            payment_method='card',  # Square primarily uses cards
            status=self._map_square_status(data.get('status', '')),
            timestamp=datetime.fromisoformat(data.get('created_at', '').replace('Z', '+00:00')),
            metadata=data
        )

    def _map_square_status(self, square_status: str) -> str:
        """Map Square status to standardized status."""
        status_mapping = {
            'COMPLETED': 'completed',
            'PENDING': 'pending',
            'FAILED': 'failed',
            'CANCELED': 'cancelled'
        }
        return status_mapping.get(square_status, 'unknown')

class PaymentGatewayManager:
    """Manages multiple payment gateway connectors."""

    def __init__(self):
        self.connectors: Dict[str, PaymentGatewayConnector] = {}

    def add_connector(self, name: str, connector: PaymentGatewayConnector):
        """Add a payment gateway connector."""
        self.connectors[name] = connector
        logger.info(f"Added payment gateway connector: {name}")

    async def initialize_connectors(self) -> Dict[str, bool]:
        """Initialize all connectors."""
        results = {}
        for name, connector in self.connectors.items():
            results[name] = await connector.connect()
            status = "connected" if results[name] else "failed"
            logger.info(f"Payment gateway {name}: {status}")
        return results

    async def fetch_all_transactions(self, start_time: datetime, end_time: datetime,
                                   limit: int = 100) -> Dict[str, List[PaymentTransaction]]:
        """Fetch transactions from all connected gateways."""
        results = {}

        for name, connector in self.connectors.items():
            try:
                transactions = await connector.fetch_transactions(start_time, end_time, limit)
                results[name] = transactions
                logger.info(f"Fetched {len(transactions)} transactions from {name}")
            except Exception as e:
                logger.error(f"Error fetching from {name}: {e}")
                results[name] = []

        return results

    async def get_transaction_by_id(self, gateway_name: str, transaction_id: str) -> Optional[PaymentTransaction]:
        """Get transaction details from specific gateway."""
        if gateway_name not in self.connectors:
            logger.error(f"Unknown gateway: {gateway_name}")
            return None

        try:
            return await self.connectors[gateway_name].get_transaction_details(transaction_id)
        except Exception as e:
            logger.error(f"Error getting transaction from {gateway_name}: {e}")
            return None

    async def close_all(self):
        """Close all connector connections."""
        for connector in self.connectors.values():
            await connector.close()

# Global manager instance
_payment_gateway_manager = None

def get_payment_gateway_manager() -> PaymentGatewayManager:
    """Get or create the global payment gateway manager instance."""
    global _payment_gateway_manager
    if _payment_gateway_manager is None:
        _payment_gateway_manager = PaymentGatewayManager()
    return _payment_gateway_manager