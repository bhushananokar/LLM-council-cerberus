"""
Cache Service
=============
Redis-based caching for LLM responses and analysis results.

Features:
- Cache LLM responses by code hash to avoid redundant API calls
- TTL-based expiration
- Cost savings through response reuse
- Thread-safe operations
"""

import json
import hashlib
import logging
from typing import Optional, Any, Dict
from datetime import timedelta

try:
    import redis
    from redis.exceptions import RedisError, ConnectionError as RedisConnectionError
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None
    RedisError = Exception
    RedisConnectionError = Exception

from config.settings import settings

logger = logging.getLogger(__name__)


class CacheService:
    """Redis-based cache service for LLM responses."""
    
    def __init__(self, 
                 host: str = None,
                 port: int = None,
                 db: int = 0,
                 password: str = None,
                 enabled: bool = True):
        """
        Initialize cache service.
        
        Args:
            host: Redis host (default from settings)
            port: Redis port (default from settings)
            db: Redis database number
            password: Redis password
            enabled: Whether caching is enabled
        """
        self.enabled = enabled and REDIS_AVAILABLE
        self.client: Optional[Any] = None
        
        if not REDIS_AVAILABLE and enabled:
            logger.warning("Redis package not installed. Caching disabled. Install with: pip install redis")
            self.enabled = False
            return
        
        if not self.enabled:
            logger.info("Cache service disabled")
            return
        
        # Use settings if not provided
        self.host = host or settings.REDIS_HOST
        self.port = port or settings.REDIS_PORT
        self.db = db
        self.password = password or settings.REDIS_PASSWORD
        
        # Default TTLs
        self.default_ttl = settings.CACHE_TTL_SECONDS
        
        # Initialize Redis client
        try:
            self.client = redis.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password if self.password else None,
                decode_responses=True,
                socket_timeout=5,
                socket_connect_timeout=5
            )
            
            # Test connection
            self.client.ping()
            logger.info(f"Cache service connected to Redis at {self.host}:{self.port}")
            
        except RedisConnectionError as e:
            logger.error(f"Failed to connect to Redis: {str(e)}")
            logger.warning("Cache service will operate in pass-through mode (no caching)")
            self.enabled = False
            self.client = None
        except Exception as e:
            logger.error(f"Redis initialization error: {str(e)}")
            self.enabled = False
            self.client = None
    
    def _generate_key(self, prefix: str, data: Any) -> str:
        """
        Generate cache key from data.
        
        Args:
            prefix: Key prefix (e.g., "agent1", "council")
            data: Data to hash (string, dict, or any JSON-serializable object)
            
        Returns:
            Cache key string
        """
        if isinstance(data, str):
            hash_input = data
        elif isinstance(data, dict):
            # Sort dict keys for consistent hashing
            hash_input = json.dumps(data, sort_keys=True)
        else:
            hash_input = str(data)
        
        # Create SHA-256 hash
        hash_value = hashlib.sha256(hash_input.encode()).hexdigest()[:16]
        
        return f"{prefix}:{hash_value}"
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
        if not self.enabled or not self.client:
            return None
        
        try:
            value = self.client.get(key)
            if value:
                logger.debug(f"Cache HIT: {key}")
                # Try to parse as JSON
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    return value
            else:
                logger.debug(f"Cache MISS: {key}")
                return None
                
        except RedisError as e:
            logger.error(f"Redis GET error: {str(e)}")
            return None
    
    def set(self, key: str, value: Any, ttl: int = None) -> bool:
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (default from settings)
            
        Returns:
            True if successful, False otherwise
        """
        if not self.enabled or not self.client:
            return False
        
        try:
            # Serialize value if it's not a string
            if not isinstance(value, str):
                value = json.dumps(value)
            
            ttl = ttl or self.default_ttl
            
            self.client.setex(key, ttl, value)
            logger.debug(f"Cache SET: {key} (TTL: {ttl}s)")
            return True
            
        except RedisError as e:
            logger.error(f"Redis SET error: {str(e)}")
            return False
    
    def delete(self, key: str) -> bool:
        """
        Delete key from cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if successful, False otherwise
        """
        if not self.enabled or not self.client:
            return False
        
        try:
            self.client.delete(key)
            logger.debug(f"Cache DELETE: {key}")
            return True
        except RedisError as e:
            logger.error(f"Redis DELETE error: {str(e)}")
            return False
    
    def get_agent_response(self, agent_name: str, package_data: Dict) -> Optional[Dict]:
        """
        Get cached agent response.
        
        Args:
            agent_name: Name of the agent
            package_data: Package data used for analysis
            
        Returns:
            Cached agent response or None
        """
        key = self._generate_key(f"agent:{agent_name}", package_data)
        return self.get(key)
    
    def set_agent_response(self, agent_name: str, package_data: Dict, response: Dict, ttl: int = None) -> bool:
        """
        Cache agent response.
        
        Args:
            agent_name: Name of the agent
            package_data: Package data used for analysis
            response: Agent response to cache
            ttl: Time to live in seconds
            
        Returns:
            True if successful
        """
        key = self._generate_key(f"agent:{agent_name}", package_data)
        return self.set(key, response, ttl)
    
    def get_council_decision(self, package_data: Dict) -> Optional[Dict]:
        """
        Get cached council decision.
        
        Args:
            package_data: Package data
            
        Returns:
            Cached council decision or None
        """
        key = self._generate_key("council", package_data)
        return self.get(key)
    
    def set_council_decision(self, package_data: Dict, decision: Dict, ttl: int = None) -> bool:
        """
        Cache council decision.
        
        Args:
            package_data: Package data
            decision: Council decision to cache
            ttl: Time to live in seconds
            
        Returns:
            True if successful
        """
        key = self._generate_key("council", package_data)
        return self.set(key, decision, ttl)
    
    def get_by_code_hash(self, code: str, prefix: str = "code") -> Optional[Dict]:
        """
        Get cached result by code hash.
        
        Args:
            code: Code string to hash
            prefix: Key prefix
            
        Returns:
            Cached result or None
        """
        key = self._generate_key(prefix, code)
        return self.get(key)
    
    def set_by_code_hash(self, code: str, result: Dict, prefix: str = "code", ttl: int = None) -> bool:
        """
        Cache result by code hash.
        
        Args:
            code: Code string to hash
            result: Result to cache
            prefix: Key prefix
            ttl: Time to live in seconds
            
        Returns:
            True if successful
        """
        key = self._generate_key(prefix, code)
        return self.set(key, result, ttl)
    
    def clear_all(self) -> bool:
        """
        Clear all cache entries (use with caution).
        
        Returns:
            True if successful
        """
        if not self.enabled or not self.client:
            return False
        
        try:
            self.client.flushdb()
            logger.warning("Cache CLEARED: All entries deleted")
            return True
        except RedisError as e:
            logger.error(f"Redis FLUSHDB error: {str(e)}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache stats
        """
        if not self.enabled or not self.client:
            return {
                "enabled": False,
                "status": "disabled"
            }
        
        try:
            info = self.client.info("stats")
            return {
                "enabled": True,
                "status": "connected",
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0),
                "hit_rate": self._calculate_hit_rate(
                    info.get("keyspace_hits", 0),
                    info.get("keyspace_misses", 0)
                ),
                "keys": self.client.dbsize(),
                "memory_used": info.get("used_memory_human", "unknown")
            }
        except RedisError as e:
            logger.error(f"Redis INFO error: {str(e)}")
            return {
                "enabled": True,
                "status": "error",
                "error": str(e)
            }
    
    def _calculate_hit_rate(self, hits: int, misses: int) -> float:
        """Calculate cache hit rate percentage."""
        total = hits + misses
        if total == 0:
            return 0.0
        return round((hits / total) * 100, 2)
    
    def health_check(self) -> bool:
        """
        Check if Redis is healthy.
        
        Returns:
            True if Redis is accessible, False otherwise
        """
        if not self.enabled or not self.client:
            return False
        
        try:
            return self.client.ping()
        except RedisError:
            return False
    
    def close(self):
        """Close Redis connection."""
        if self.client:
            try:
                self.client.close()
                logger.info("Cache service connection closed")
            except Exception as e:
                logger.error(f"Error closing cache connection: {str(e)}")


# Singleton instance
_cache_instance: Optional[CacheService] = None


def get_cache_service() -> CacheService:
    """
    Get singleton cache service instance.
    
    Returns:
        CacheService instance
    """
    global _cache_instance
    
    if _cache_instance is None:
        _cache_instance = CacheService(
            enabled=settings.CACHE_ENABLED
        )
    
    return _cache_instance


# Convenience functions
def get_cached_response(agent_name: str, package_data: Dict) -> Optional[Dict]:
    """Get cached agent response."""
    cache = get_cache_service()
    return cache.get_agent_response(agent_name, package_data)


def cache_response(agent_name: str, package_data: Dict, response: Dict) -> bool:
    """Cache agent response."""
    cache = get_cache_service()
    return cache.set_agent_response(agent_name, package_data, response)


def get_cached_decision(package_data: Dict) -> Optional[Dict]:
    """Get cached council decision."""
    cache = get_cache_service()
    return cache.get_council_decision(package_data)


def cache_decision(package_data: Dict, decision: Dict) -> bool:
    """Cache council decision."""
    cache = get_cache_service()
    return cache.set_council_decision(package_data, decision)


def clear_cache() -> bool:
    """Clear all cache entries."""
    cache = get_cache_service()
    return cache.clear_all()


def get_cache_stats() -> Dict[str, Any]:
    """Get cache statistics."""
    cache = get_cache_service()
    return cache.get_stats()