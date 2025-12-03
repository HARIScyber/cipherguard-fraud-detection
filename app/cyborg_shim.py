"""
CyborgDB Shim Module
Local mock/stub implementation of CyborgDB for testing without service
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class CyborgDBShim:
    """
    Local in-memory mock of CyborgDB for development/testing.
    
    Stores encrypted vectors locally and performs approximate kNN search.
    In production, replace with actual CyborgDB service.
    """
    
    def __init__(self, vector_dim: int = 6):
        """
        Initialize local vector store.
        
        Args:
            vector_dim: Dimension of vectors
        """
        self.vector_dim = vector_dim
        self.vectors: Dict[str, np.ndarray] = {}
        self.metadata: Dict[str, Dict] = {}
        self.created_at = datetime.now()
        logger.info("CyborgDB Shim initialized (local mode)")
    
    def insert(self,
               transaction_id: str,
               vector: np.ndarray,
               metadata: Dict = None) -> bool:
        """
        Insert vector with optional metadata.
        
        Args:
            transaction_id: Unique ID
            vector: Feature vector
            metadata: Associated metadata
        
        Returns:
            bool: Success status
        """
        try:
            if len(vector) != self.vector_dim:
                logger.error(f"Vector dimension mismatch: expected {self.vector_dim}, got {len(vector)}")
                return False
            
            self.vectors[transaction_id] = vector.astype(np.float32)
            self.metadata[transaction_id] = metadata or {}
            logger.debug(f"Inserted vector {transaction_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error inserting vector: {e}")
            return False
    
    def search(self,
               query_vector: np.ndarray,
               k: int = 5) -> List[Tuple[str, float]]:
        """
        Find k nearest neighbors.
        
        Args:
            query_vector: Query vector
            k: Number of results
        
        Returns:
            List[(transaction_id, distance)]: Ranked results
        """
        try:
            if len(self.vectors) == 0:
                logger.warning("No vectors in store")
                return []
            
            if len(query_vector) != self.vector_dim:
                logger.error(f"Query vector dimension mismatch")
                return []
            
            # Compute distances to all vectors
            distances = []
            for tid, vec in self.vectors.items():
                # L2 distance
                dist = np.linalg.norm(query_vector - vec)
                distances.append((tid, float(dist)))
            
            # Sort by distance and return top-k
            distances.sort(key=lambda x: x[1])
            results = distances[:k]
            
            logger.debug(f"Found {len(results)} nearest neighbors")
            return results
            
        except Exception as e:
            logger.error(f"Error searching vectors: {e}")
            return []
    
    def get(self, transaction_id: str) -> Optional[np.ndarray]:
        """
        Retrieve vector by ID.
        
        Args:
            transaction_id: Transaction ID
        
        Returns:
            np.ndarray or None: Vector or None if not found
        """
        return self.vectors.get(transaction_id)
    
    def delete(self, transaction_id: str) -> bool:
        """
        Delete vector by ID.
        
        Args:
            transaction_id: Transaction ID
        
        Returns:
            bool: Success status
        """
        try:
            if transaction_id in self.vectors:
                del self.vectors[transaction_id]
                if transaction_id in self.metadata:
                    del self.metadata[transaction_id]
                logger.debug(f"Deleted vector {transaction_id}")
                return True
            else:
                logger.warning(f"Vector not found: {transaction_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error deleting vector: {e}")
            return False
    
    def count(self) -> int:
        """Get count of stored vectors."""
        return len(self.vectors)
    
    def clear(self):
        """Clear all vectors from store."""
        self.vectors.clear()
        self.metadata.clear()
        logger.info("Cleared all vectors")
    
    def get_stats(self) -> Dict:
        """Get statistics about the vector store."""
        return {
            "count": self.count(),
            "vector_dim": self.vector_dim,
            "created_at": self.created_at.isoformat(),
            "uptime_seconds": (datetime.now() - self.created_at).total_seconds()
        }


# Global shim instance
_shim_instance: Optional[CyborgDBShim] = None


def get_cyborg_shim(vector_dim: int = 6) -> CyborgDBShim:
    """
    Get or create global CyborgDB shim instance.
    
    Args:
        vector_dim: Vector dimension
    
    Returns:
        CyborgDBShim: Global shim instance
    """
    global _shim_instance
    if _shim_instance is None:
        _shim_instance = CyborgDBShim(vector_dim=vector_dim)
    return _shim_instance
