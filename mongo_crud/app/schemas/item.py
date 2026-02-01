"""
Pydantic schemas for request/response validation.
"""

from datetime import datetime
from typing import Optional, Any, Dict
from pydantic import BaseModel, Field, ConfigDict
from bson import ObjectId


class PyObjectId(str):
    """
    Custom ObjectId field for Pydantic models.
    """
    
    @classmethod
    def __get_pydantic_core_schema__(cls, _source_type, _handler):
        from pydantic_core import core_schema
        return core_schema.no_info_after_validator_function(
            cls.validate,
            core_schema.str_schema(),
        )
    
    @classmethod
    def validate(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid ObjectId")
        return str(v)


class ItemBase(BaseModel):
    """
    Base schema with common fields.
    
    CUSTOMIZATION NOTE:
    - Add your custom fields here
    - Fields will be automatically validated by Pydantic
    """
    name: str = Field(..., min_length=1, max_length=200, description="Item name")
    description: Optional[str] = Field(None, max_length=1000, description="Item description")
    data: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional custom data")


class ItemCreate(ItemBase):
    """
    Schema for creating a new item.
    Add any required fields for creation here.
    """
    pass


class ItemUpdate(BaseModel):
    """
    Schema for full update (PUT) of an item.
    All fields are optional to allow partial updates in practice.
    """
    name: Optional[str] = Field(None, min_length=1, max_length=200)
    description: Optional[str] = Field(None, max_length=1000)
    data: Optional[Dict[str, Any]] = None


class ItemPartialUpdate(BaseModel):
    """
    Schema for partial update (PATCH) of an item.
    All fields are optional.
    """
    name: Optional[str] = Field(None, min_length=1, max_length=200)
    description: Optional[str] = Field(None, max_length=1000)
    data: Optional[Dict[str, Any]] = None


class ItemResponse(ItemBase):
    """
    Schema for item response with ID and timestamps.
    """
    id: PyObjectId = Field(alias="_id", description="Item ID")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    
    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
        json_encoders={ObjectId: str}
    )


class ItemListResponse(BaseModel):
    """
    Schema for paginated list response.
    """
    items: list[ItemResponse] = Field(..., description="List of items")
    total: int = Field(..., description="Total number of items matching the query")
    skip: int = Field(..., description="Number of items skipped")
    limit: int = Field(..., description="Maximum number of items returned")
    
    model_config = ConfigDict(
        json_encoders={ObjectId: str}
    )


class ErrorResponse(BaseModel):
    """
    Standard error response schema.
    """
    detail: str = Field(..., description="Error message")
    error_code: Optional[str] = Field(None, description="Error code for client handling")


class MessageResponse(BaseModel):
    """
    Standard message response schema.
    """
    message: str = Field(..., description="Response message")
