"""
Generic repository pattern for MongoDB CRUD operations.
This base repository can be reused for any collection.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorCollection, AsyncIOMotorDatabase
from pymongo.errors import DuplicateKeyError
from fastapi import HTTPException, status


class BaseRepository:
    """
    Generic repository providing CRUD operations for any MongoDB collection.
    
    CUSTOMIZATION NOTE:
    - Instantiate this class with your collection name
    - Example: items_repo = BaseRepository(database, "items")
    - Override methods if you need custom behavior
    """
    
    def __init__(self, database: AsyncIOMotorDatabase, collection_name: str):
        """
        Initialize repository with database and collection name.
        
        Args:
            database: MongoDB database instance
            collection_name: Name of the collection to operate on
        """
        self.collection: AsyncIOMotorCollection = database[collection_name]
        self.collection_name = collection_name
    
    async def create(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a new document in the collection.
        
        Args:
            data: Document data to insert
            
        Returns:
            Created document with _id
            
        Raises:
            HTTPException: If duplicate key error occurs
        """
        try:
            # Add timestamps
            now = datetime.utcnow()
            data["created_at"] = now
            data["updated_at"] = now
            
            result = await self.collection.insert_one(data)
            created_document = await self.collection.find_one({"_id": result.inserted_id})
            return created_document
        
        except DuplicateKeyError as e:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Document with this unique field already exists: {str(e)}"
            )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error creating document: {str(e)}"
            )
    
    async def get_by_id(self, item_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a document by its ID.
        
        Args:
            item_id: String representation of ObjectId
            
        Returns:
            Document if found, None otherwise
            
        Raises:
            HTTPException: If invalid ObjectId format
        """
        try:
            if not ObjectId.is_valid(item_id):
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="Invalid ID format"
                )
            
            document = await self.collection.find_one({"_id": ObjectId(item_id)})
            return document
        
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error retrieving document: {str(e)}"
            )
    
    async def get_all(
        self,
        skip: int = 0,
        limit: int = 100,
        filters: Optional[Dict[str, Any]] = None,
        sort_by: Optional[str] = None,
        sort_order: int = 1
    ) -> tuple[List[Dict[str, Any]], int]:
        """
        Retrieve all documents with pagination, filtering, and sorting.
        
        Args:
            skip: Number of documents to skip
            limit: Maximum number of documents to return
            filters: MongoDB filter query
            sort_by: Field name to sort by
            sort_order: 1 for ascending, -1 for descending
            
        Returns:
            Tuple of (list of documents, total count)
        """
        try:
            query = filters or {}
            
            # Get total count
            total = await self.collection.count_documents(query)
            
            # Build query with pagination and sorting
            cursor = self.collection.find(query).skip(skip).limit(limit)
            
            if sort_by:
                cursor = cursor.sort(sort_by, sort_order)
            
            documents = await cursor.to_list(length=limit)
            return documents, total
        
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error retrieving documents: {str(e)}"
            )
    
    async def update(
        self,
        item_id: str,
        data: Dict[str, Any],
        partial: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        Update a document by ID.
        
        Args:
            item_id: String representation of ObjectId
            data: Updated data
            partial: If True, only update provided fields (PATCH). If False, replace document (PUT)
            
        Returns:
            Updated document if found, None otherwise
            
        Raises:
            HTTPException: If invalid ID or update fails
        """
        try:
            if not ObjectId.is_valid(item_id):
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="Invalid ID format"
                )
            
            # Remove None values for partial updates
            if partial:
                data = {k: v for k, v in data.items() if v is not None}
            
            # Update timestamp
            data["updated_at"] = datetime.utcnow()
            
            update_operation = {"$set": data} if partial else {"$set": data}
            
            result = await self.collection.find_one_and_update(
                {"_id": ObjectId(item_id)},
                update_operation,
                return_document=True
            )
            
            return result
        
        except HTTPException:
            raise
        except DuplicateKeyError as e:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Document with this unique field already exists: {str(e)}"
            )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error updating document: {str(e)}"
            )
    
    async def delete(self, item_id: str) -> bool:
        """
        Delete a document by ID.
        
        Args:
            item_id: String representation of ObjectId
            
        Returns:
            True if deleted, False if not found
            
        Raises:
            HTTPException: If invalid ID or delete fails
        """
        try:
            if not ObjectId.is_valid(item_id):
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="Invalid ID format"
                )
            
            result = await self.collection.delete_one({"_id": ObjectId(item_id)})
            return result.deleted_count > 0
        
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error deleting document: {str(e)}"
            )
    
    async def search(
        self,
        field: str,
        value: str,
        skip: int = 0,
        limit: int = 100
    ) -> tuple[List[Dict[str, Any]], int]:
        """
        Search documents by field value (case-insensitive regex).
        
        Args:
            field: Field name to search in
            value: Search value (supports partial matching)
            skip: Number of documents to skip
            limit: Maximum number of documents to return
            
        Returns:
            Tuple of (list of documents, total count)
        """
        try:
            query = {field: {"$regex": value, "$options": "i"}}
            return await self.get_all(skip=skip, limit=limit, filters=query)
        
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error searching documents: {str(e)}"
            )
