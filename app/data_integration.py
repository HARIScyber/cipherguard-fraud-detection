"""
Data Integration Module - Phase 5: Real Data Pipeline
Handles real transaction data ingestion, validation, and preprocessing
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Iterator
import logging
import json
from datetime import datetime, timedelta
import asyncio
import aiohttp
import os
from dataclasses import dataclass
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

@dataclass
class TransactionData:
    """Standardized transaction data structure."""
    transaction_id: str
    amount: float
    merchant: str
    device: str
    country: str
    timestamp: datetime
    user_id: Optional[str] = None
    card_number: Optional[str] = None
    ip_address: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class DataSource(ABC):
    """Abstract base class for data sources."""

    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection to data source."""
        pass

    @abstractmethod
    async def fetch_transactions(self, start_time: datetime, end_time: datetime) -> Iterator[TransactionData]:
        """Fetch transactions within time range."""
        pass

    @abstractmethod
    async def validate_transaction(self, transaction: Dict[str, Any]) -> bool:
        """Validate transaction data."""
        pass

class PaymentGatewaySource(DataSource):
    """Integration with payment gateway APIs."""

    def __init__(self, api_key: str, api_secret: str, base_url: str):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url
        self.session: Optional[aiohttp.ClientSession] = None

    async def connect(self) -> bool:
        """Connect to payment gateway."""
        try:
            self.session = aiohttp.ClientSession(
                headers={
                    'Authorization': f'Bearer {self.api_key}',
                    'Content-Type': 'application/json'
                }
            )
            # Test connection
            async with self.session.get(f"{self.base_url}/health") as response:
                return response.status == 200
        except Exception as e:
            logger.error(f"Failed to connect to payment gateway: {e}")
            return False

    async def fetch_transactions(self, start_time: datetime, end_time: datetime) -> Iterator[TransactionData]:
        """Fetch transactions from payment gateway."""
        if not self.session:
            raise ConnectionError("Not connected to payment gateway")

        params = {
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'limit': 1000
        }

        try:
            async with self.session.get(f"{self.base_url}/transactions", params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    for tx in data.get('transactions', []):
                        if self.validate_transaction(tx):
                            yield self._parse_transaction(tx)
                else:
                    logger.error(f"Failed to fetch transactions: {response.status}")
        except Exception as e:
            logger.error(f"Error fetching transactions: {e}")

    def validate_transaction(self, transaction: Dict[str, Any]) -> bool:
        """Validate transaction data structure."""
        required_fields = ['id', 'amount', 'merchant', 'timestamp']
        return all(field in transaction for field in required_fields)

    def _parse_transaction(self, tx: Dict[str, Any]) -> TransactionData:
        """Parse raw transaction data into standardized format."""
        return TransactionData(
            transaction_id=str(tx['id']),
            amount=float(tx['amount']),
            merchant=tx.get('merchant', 'Unknown'),
            device=tx.get('device', 'desktop'),
            country=tx.get('country', 'US'),
            timestamp=datetime.fromisoformat(tx['timestamp']),
            user_id=tx.get('user_id'),
            card_number=tx.get('card_last_four'),
            ip_address=tx.get('ip_address'),
            metadata=tx.get('metadata', {})
        )

class DatabaseSource(DataSource):
    """Integration with database sources (PostgreSQL, MySQL, etc.)."""

    def __init__(self, connection_string: str, table_name: str):
        self.connection_string = connection_string
        self.table_name = table_name
        self.engine = None

    async def connect(self) -> bool:
        """Connect to database."""
        try:
            from sqlalchemy import create_engine
            self.engine = create_engine(self.connection_string)
            # Test connection
            with self.engine.connect() as conn:
                conn.execute("SELECT 1")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            return False

    async def fetch_transactions(self, start_time: datetime, end_time: datetime) -> Iterator[TransactionData]:
        """Fetch transactions from database."""
        if not self.engine:
            raise ConnectionError("Not connected to database")

        query = f"""
        SELECT * FROM {self.table_name}
        WHERE timestamp BETWEEN '{start_time}' AND '{end_time}'
        ORDER BY timestamp
        """

        try:
            df = pd.read_sql(query, self.engine)
            for _, row in df.iterrows():
                tx_dict = row.to_dict()
                if self.validate_transaction(tx_dict):
                    yield self._parse_transaction(tx_dict)
        except Exception as e:
            logger.error(f"Error fetching from database: {e}")

    def validate_transaction(self, transaction: Dict[str, Any]) -> bool:
        """Validate database transaction."""
        required_fields = ['transaction_id', 'amount', 'timestamp']
        return all(field in transaction for field in required_fields)

    def _parse_transaction(self, tx: Dict[str, Any]) -> TransactionData:
        """Parse database row into standardized format."""
        return TransactionData(
            transaction_id=str(tx['transaction_id']),
            amount=float(tx['amount']),
            merchant=tx.get('merchant', 'Unknown'),
            device=tx.get('device', 'desktop'),
            country=tx.get('country', 'US'),
            timestamp=pd.to_datetime(tx['timestamp']),
            user_id=tx.get('user_id'),
            card_number=tx.get('card_number'),
            ip_address=tx.get('ip_address'),
            metadata=tx.get('metadata', {})
        )

class FileSource(DataSource):
    """Integration with file-based data sources (CSV, JSON, Parquet)."""

    def __init__(self, file_path: str, file_format: str = 'csv'):
        self.file_path = file_path
        self.file_format = file_format.lower()

    async def connect(self) -> bool:
        """Check if file exists and is readable."""
        return os.path.exists(self.file_path) and os.access(self.file_path, os.R_OK)

    async def fetch_transactions(self, start_time: datetime, end_time: datetime) -> Iterator[TransactionData]:
        """Fetch transactions from file."""
        try:
            if self.file_format == 'csv':
                df = pd.read_csv(self.file_path)
            elif self.file_format == 'json':
                df = pd.read_json(self.file_path)
            elif self.file_format == 'parquet':
                df = pd.read_parquet(self.file_path)
            else:
                raise ValueError(f"Unsupported file format: {self.file_format}")

            # Filter by timestamp if available
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df[(df['timestamp'] >= start_time) & (df['timestamp'] <= end_time)]

            for _, row in df.iterrows():
                tx_dict = row.to_dict()
                if self.validate_transaction(tx_dict):
                    yield self._parse_transaction(tx_dict)

        except Exception as e:
            logger.error(f"Error reading file {self.file_path}: {e}")

    def validate_transaction(self, transaction: Dict[str, Any]) -> bool:
        """Validate file-based transaction."""
        required_fields = ['amount', 'timestamp']
        return all(field in transaction for field in required_fields)

    def _parse_transaction(self, tx: Dict[str, Any]) -> TransactionData:
        """Parse file row into standardized format."""
        return TransactionData(
            transaction_id=str(tx.get('transaction_id', tx.get('id', np.random.randint(1000000)))),
            amount=float(tx['amount']),
            merchant=tx.get('merchant', 'Unknown'),
            device=tx.get('device', 'desktop'),
            country=tx.get('country', 'US'),
            timestamp=pd.to_datetime(tx['timestamp']),
            user_id=tx.get('user_id'),
            card_number=tx.get('card_number'),
            ip_address=tx.get('ip_address'),
            metadata=tx.get('metadata', {})
        )

class DataPipeline:
    """Orchestrates data ingestion from multiple sources."""

    def __init__(self):
        self.sources: Dict[str, DataSource] = {}
        self.processed_transactions = 0
        self.failed_transactions = 0

    def add_source(self, name: str, source: DataSource):
        """Add a data source to the pipeline."""
        self.sources[name] = source
        logger.info(f"Added data source: {name}")

    async def initialize_sources(self) -> Dict[str, bool]:
        """Initialize all data sources."""
        results = {}
        for name, source in self.sources.items():
            results[name] = await source.connect()
            status = "connected" if results[name] else "failed"
            logger.info(f"Data source {name}: {status}")
        return results

    async def fetch_all_transactions(self, start_time: datetime, end_time: datetime,
                                   batch_size: int = 1000) -> Iterator[List[TransactionData]]:
        """Fetch transactions from all sources in batches."""
        all_transactions = []

        for source_name, source in self.sources.items():
            logger.info(f"Fetching from source: {source_name}")
            try:
                async for transaction in source.fetch_transactions(start_time, end_time):
                    all_transactions.append(transaction)
                    self.processed_transactions += 1

                    if len(all_transactions) >= batch_size:
                        yield all_transactions
                        all_transactions = []

            except Exception as e:
                logger.error(f"Error fetching from {source_name}: {e}")
                self.failed_transactions += 1

        # Yield remaining transactions
        if all_transactions:
            yield all_transactions

    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        return {
            'sources_count': len(self.sources),
            'processed_transactions': self.processed_transactions,
            'failed_transactions': self.failed_transactions,
            'success_rate': (self.processed_transactions /
                           (self.processed_transactions + self.failed_transactions)
                           if (self.processed_transactions + self.failed_transactions) > 0 else 0)
        }

class DataValidator:
    """Validates and cleans transaction data."""

    @staticmethod
    def validate_transaction_data(transaction: TransactionData) -> Dict[str, Any]:
        """Comprehensive validation of transaction data."""
        issues = []

        # Amount validation
        if transaction.amount <= 0:
            issues.append("Invalid amount: must be positive")
        elif transaction.amount > 100000:  # Configurable threshold
            issues.append("Suspiciously high amount")

        # Timestamp validation
        if transaction.timestamp > datetime.now():
            issues.append("Future timestamp")
        elif transaction.timestamp < datetime.now() - timedelta(days=365*2):
            issues.append("Too old transaction")

        # Required field validation
        if not transaction.merchant or transaction.merchant == "Unknown":
            issues.append("Missing merchant information")

        # IP address validation (if provided)
        if transaction.ip_address:
            if not DataValidator._is_valid_ip(transaction.ip_address):
                issues.append("Invalid IP address format")

        return {
            'is_valid': len(issues) == 0,
            'issues': issues,
            'severity': 'high' if len(issues) > 2 else 'medium' if len(issues) > 0 else 'low'
        }

    @staticmethod
    def _is_valid_ip(ip: str) -> bool:
        """Validate IP address format."""
        import ipaddress
        try:
            ipaddress.ip_address(ip)
            return True
        except ValueError:
            return False

    @staticmethod
    def clean_transaction_data(transaction: TransactionData) -> TransactionData:
        """Clean and normalize transaction data."""
        # Normalize merchant names
        merchant_mapping = {
            'amzn': 'Amazon',
            'goog': 'Google',
            'appl': 'Apple',
            'wlmrt': 'Walmart'
        }

        merchant = transaction.merchant.lower()
        for abbr, full in merchant_mapping.items():
            if abbr in merchant:
                transaction.merchant = full
                break

        # Normalize country codes
        country_mapping = {
            'usa': 'US',
            'uk': 'GB',
            'canada': 'CA',
            'australia': 'AU'
        }

        country = transaction.country.upper()
        transaction.country = country_mapping.get(country, transaction.country)

        return transaction

# Global pipeline instance
_data_pipeline = None

def get_data_pipeline() -> DataPipeline:
    """Get or create the global data pipeline instance."""
    global _data_pipeline
    if _data_pipeline is None:
        _data_pipeline = DataPipeline()
    return _data_pipeline</content>
<parameter name="filePath">d:\cipherguard-fraud-poc\app\data_integration.py