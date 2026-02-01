"""
MongoDB database connection module using Motor (async driver).
Provides connection pooling, error handling, and retry logic.
"""

import logging
from typing import Optional
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
from app.core.config import settings

logger = logging.getLogger(__name__)


class MongoDB:
    """
    MongoDB connection manager with connection pooling and lifecycle management.
    """
    
    def __init__(self):
        self.client: Optional[AsyncIOMotorClient] = None
        self.db: Optional[AsyncIOMotorDatabase] = None
    
    async def connect(self) -> None:
        """
        Establish MongoDB connection with retry logic.
        
        Raises:
            ConnectionFailure: If connection cannot be established after retries.
        """
        try:
            logger.info("Connecting to MongoDB...")
            
            self.client = AsyncIOMotorClient(
                settings.mongodb_uri,
                maxPoolSize=settings.max_pool_size,
                minPoolSize=settings.min_pool_size,
                maxIdleTimeMS=settings.max_idle_time_ms,
                serverSelectionTimeoutMS=5000,  # 5 seconds timeout
            )
            
            # Verify connection
            await self.client.admin.command('ping')
            
            self.db = self.client[settings.database_name]
            logger.info(f"Successfully connected to MongoDB database: {settings.database_name}")
            
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise ConnectionFailure(
                f"Could not connect to MongoDB at {settings.mongodb_uri}. "
                f"Please check your connection string and network connectivity."
            )
        except Exception as e:
            logger.error(f"Unexpected error during MongoDB connection: {e}")
            raise
    
    async def close(self) -> None:
        """
        Close MongoDB connection and cleanup resources.
        """
        if self.client:
            logger.info("Closing MongoDB connection...")
            self.client.close()
            self.client = None
            self.db = None
            logger.info("MongoDB connection closed successfully")
    
    async def disconnect(self) -> None:
        """
        Alias for close() - disconnect from MongoDB.
        """
        await self.close()
    
    async def get_database(self) -> AsyncIOMotorDatabase:
        """
        Get database instance.
        
        Returns:
            AsyncIOMotorDatabase: The MongoDB database instance.
            
        Raises:
            RuntimeError: If database is not connected.
        """
        if self.db is None:
            raise RuntimeError(
                "Database is not connected. Call connect() first."
            )
        return self.db


# Global MongoDB instance
mongodb = MongoDB()


async def get_database() -> AsyncIOMotorDatabase:
    """
    Dependency injection function for FastAPI routes.
    
    Returns:
        AsyncIOMotorDatabase: The MongoDB database instance.
    """
    return await mongodb.get_database()
