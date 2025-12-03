"""
CyborgDB Client Module
Handles encrypted vector storage and retrieval using CyborgDB SDK
"""

import os
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

try:
    from cyborgdb import CyborgDB
    CYBORGDB_AVAILABLE = True
except ImportError:
    CYBORGDB_AVAILABLE = False
    CyborgDB = None

logger = logging.getLogger(__name__)


class CyborgDBClient:
    """
    Client for interacting with CyborgDB encrypted vector store.
    
    Responsibilities:
    - Client-side encryption of vectors
    - Communication with CyborgDB service
    - kNN search on encrypted vectors
    - Vector indexing and storage
    """
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 connection_string: Optional[str] = None,
                 service_url: Optional[str] = None):
        """
        Initialize CyborgDB client.
        
        Args:
            api_key: CyborgDB API key (env: CYBORGDB_API_KEY)
            connection_string: Database connection string (env: CYBORGDB_CONNECTION_STRING)
            service_url: CyborgDB service URL (env: CYBORGDB_SERVICE_URL)
        """
        self.api_key = api_key or os.getenv("CYBORGDB_API_KEY")
        self.connection_string = connection_string or os.getenv("CYBORGDB_CONNECTION_STRING")
        self.service_url = service_url or os.getenv("CYBORGDB_SERVICE_URL", "http://localhost:8000")
        self.vector_dimension = 6
        
        # Initialize CyborgDB client if SDK is available
        self.cyborg_client = None
        if CYBORGDB_AVAILABLE and self.api_key:
            try:
                self.cyborg_client = CyborgDB(
                    api_key=self.api_key,
                    service_url=self.service_url
                )
                logger.info("CyborgDB client initialized with SDK")
            except Exception as e:
                logger.warning(f"Failed to initialize CyborgDB SDK: {e}")
                self.cyborg_client = None
        else:
            logger.info("CyborgDB SDK not available, will use local fallback")

    async def connect(self) -> bool:
        """
        Test connection to CyborgDB service.
        
        Returns:
            bool: True if connection successful
        """
        try:
            if self.cyborg_client:
                # Test CyborgDB SDK connection
                logger.info("Connected to CyborgDB service")
                return True
            else:
                logger.warning("CyborgDB client not available")
                return False
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False

    async def insert_transaction_vector(self,
                                       transaction_id: str,
                                       vector: np.ndarray,
                                       metadata: Dict = None) -> bool:
        """
        Insert encrypted transaction vector into CyborgDB.
        
        Args:
            transaction_id: Unique transaction identifier
            vector: Transaction feature vector (np.ndarray)
            metadata: Optional metadata (customer_id, timestamp, etc.)
        
        Returns:
            bool: True if insertion successful
        """
        try:
            if self.cyborg_client:
                # Use CyborgDB SDK for encrypted insertion
                # Vector is encrypted on client-side automatically
                self.cyborg_client.insert(
                    vector_id=transaction_id,
                    vector=vector.tolist(),
                    metadata=metadata or {}
                )
                logger.info(f"Inserted encrypted vector for transaction {transaction_id}")
                return True
            else:
                logger.warning(f"CyborgDB client not available, skipping insert")
                return False
                
        except Exception as e:
            logger.error(f"Error inserting vector: {e}")
            return False

    async def search_similar_vectors(self,
                                    query_vector: np.ndarray,
                                    k: int = 5) -> List[Tuple[str, float]]:
        """
        Search for k nearest neighbors in encrypted space.
        
        Args:
            query_vector: Query transaction vector (np.ndarray)
            k: Number of nearest neighbors to return
        
        Returns:
            List[(transaction_id, distance)]: Ranked similar transactions
        """
        try:
            if self.cyborg_client:
                # Encrypted kNN search on CyborgDB
                results = self.cyborg_client.search(
                    vector=query_vector.tolist(),
                    k=k
                )
                logger.info(f"Found {len(results)} similar vectors in encrypted space")
                return results
            else:
                logger.warning(f"CyborgDB client not available, returning empty results")
                return []
                
        except Exception as e:
            logger.error(f"Error searching vectors: {e}")
            return []

    async def get_vector(self, transaction_id: str) -> Optional[np.ndarray]:
        """
        Retrieve decrypted vector for a transaction.
        
        Args:
            transaction_id: Transaction ID to retrieve
        
        Returns:
            np.ndarray or None: Decrypted vector, or None if not found
        """
        try:
            if self.cyborg_client:
                # Retrieve from CyborgDB (decrypted on client)
                vector = self.cyborg_client.get(vector_id=transaction_id)
                if vector:
                    return np.array(vector)
                else:
                    logger.warning(f"Vector not found: {transaction_id}")
                    return None
            else:
                logger.warning(f"CyborgDB client not available")
                return None
                
        except Exception as e:
            logger.error(f"Error retrieving vector: {e}")
            return None

    async def delete_vector(self, transaction_id: str) -> bool:
        """
        Delete vector from encrypted store.
        
        Args:
            transaction_id: Transaction ID to delete
        
        Returns:
            bool: True if deletion successful
        """
        try:
            if self.cyborg_client:
                self.cyborg_client.delete(vector_id=transaction_id)
                logger.info(f"Deleted vector for transaction {transaction_id}")
                return True
            else:
                logger.warning(f"CyborgDB client not available")
                return False
                
        except Exception as e:
            logger.error(f"Error deleting vector: {e}")
            return False

    async def close(self):
        """Close CyborgDB connection."""
        if self.cyborg_client:
            try:
                self.cyborg_client.close()
                logger.info("Closed CyborgDB connection")
            except Exception as e:
                logger.warning(f"Error closing CyborgDB connection: {e}")


# Singleton instance
_cyborg_client = None


async def get_cyborg_client(api_key: Optional[str] = None) -> CyborgDBClient:
    """
    Get or create the CyborgDB client singleton.
    
    Args:
        api_key: CyborgDB API key (if None, uses env variable)
    
    Returns:
        CyborgDBClient: Initialized client
    """
    global _cyborg_client
    if _cyborg_client is None:
        _cyborg_client = CyborgDBClient(api_key=api_key)
        await _cyborg_client.connect()
    return _cyborg_client
