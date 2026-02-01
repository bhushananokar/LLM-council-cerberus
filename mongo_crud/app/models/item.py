"""
Database models representing MongoDB document structure.
"""

from datetime import datetime
from typing import Optional, Any, Dict
from bson import ObjectId


class PyObjectId(ObjectId):
    """
    Custom ObjectId type for Pydantic validation.
    """
    
    @classmethod
    def __get_validators__(cls):
        yield cls.validate
    
    @classmethod
    def validate(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid ObjectId")
        return ObjectId(v)
    
    @classmethod
    def __get_pydantic_core_schema__(cls, _source_type, _handler):
        from pydantic_core import core_schema
        return core_schema.no_info_after_validator_function(
            cls.validate,
            core_schema.str_schema(),
        )


class ItemModel:
    """
    Base item model representing a MongoDB document.
    
    CUSTOMIZATION NOTE:
    - Extend this class to add specific fields for your use case
    - The 'data' field can store any custom attributes as a dictionary
    - Add your own fields as class attributes
    """
    
    def __init__(
        self,
        name: str,
        description: Optional[str] = None,
        data: Optional[Dict[str, Any]] = None,
        created_at: Optional[datetime] = None,
        updated_at: Optional[datetime] = None,
        _id: Optional[ObjectId] = None
    ):
        self._id = _id
        self.name = name
        self.description = description
        self.data = data or {}
        self.created_at = created_at or datetime.utcnow()
        self.updated_at = updated_at or datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary for MongoDB insertion."""
        doc = {
            "name": self.name,
            "description": self.description,
            "data": self.data,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
        if self._id:
            doc["_id"] = self._id
        return doc
