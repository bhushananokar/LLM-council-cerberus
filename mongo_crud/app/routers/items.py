"""
API routes for CRUD operations on items.

CUSTOMIZATION NOTE:
- Change 'items' to your entity name throughout this file
- Update collection_name in get_repository()
- Modify schemas if you added custom fields
"""

from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.database.mongodb import get_database
from app.database.repository import BaseRepository
from app.schemas.item import (
    ItemCreate,
    ItemUpdate,
    ItemPartialUpdate,
    ItemResponse,
    ItemListResponse,
    MessageResponse,
    ErrorResponse
)

router = APIRouter(
    prefix="/items",
    tags=["items"],
    responses={
        404: {"model": ErrorResponse, "description": "Item not found"},
        422: {"model": ErrorResponse, "description": "Validation error"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)


def get_repository(db: AsyncIOMotorDatabase = Depends(get_database)) -> BaseRepository:
    """
    Dependency to get items repository.
    
    CUSTOMIZATION: Change 'items' to your collection name
    """
    return BaseRepository(db, "items")


@router.post(
    "/",
    response_model=ItemResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new item",
    description="Create a new item with the provided data"
)
async def create_item(
    item: ItemCreate,
    repository: BaseRepository = Depends(get_repository)
) -> ItemResponse:
    """
    Create a new item.
    
    - **name**: Item name (required)
    - **description**: Item description (optional)
    - **data**: Additional custom data (optional)
    """
    item_dict = item.model_dump(exclude_unset=True)
    created_item = await repository.create(item_dict)
    return ItemResponse(**created_item)


@router.get(
    "/",
    response_model=ItemListResponse,
    summary="List all items",
    description="Retrieve a paginated list of items with optional filtering and sorting"
)
async def list_items(
    skip: int = Query(0, ge=0, description="Number of items to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of items to return"),
    name: Optional[str] = Query(None, description="Filter by name (case-insensitive partial match)"),
    sort_by: Optional[str] = Query("created_at", description="Field to sort by"),
    sort_order: int = Query(-1, ge=-1, le=1, description="Sort order: 1 for ascending, -1 for descending"),
    repository: BaseRepository = Depends(get_repository)
) -> ItemListResponse:
    """
    Retrieve a list of items with pagination, filtering, and sorting.
    
    - **skip**: Number of items to skip (for pagination)
    - **limit**: Maximum number of items to return
    - **name**: Filter by item name (case-insensitive partial match)
    - **sort_by**: Field to sort by (default: created_at)
    - **sort_order**: Sort order (1: ascending, -1: descending)
    """
    # Build filters
    filters = {}
    if name:
        filters["name"] = {"$regex": name, "$options": "i"}
    
    # Get items with pagination
    items, total = await repository.get_all(
        skip=skip,
        limit=limit,
        filters=filters,
        sort_by=sort_by,
        sort_order=sort_order
    )
    
    return ItemListResponse(
        items=[ItemResponse(**item) for item in items],
        total=total,
        skip=skip,
        limit=limit
    )


@router.get(
    "/{item_id}",
    response_model=ItemResponse,
    summary="Get item by ID",
    description="Retrieve a single item by its ID"
)
async def get_item(
    item_id: str,
    repository: BaseRepository = Depends(get_repository)
) -> ItemResponse:
    """
    Retrieve a single item by ID.
    
    - **item_id**: The ID of the item to retrieve
    """
    item = await repository.get_by_id(item_id)
    
    if not item:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Item with id '{item_id}' not found"
        )
    
    return ItemResponse(**item)


@router.put(
    "/{item_id}",
    response_model=ItemResponse,
    summary="Update entire item",
    description="Replace an entire item with new data (full update)"
)
async def update_item(
    item_id: str,
    item: ItemUpdate,
    repository: BaseRepository = Depends(get_repository)
) -> ItemResponse:
    """
    Update an entire item (PUT - full replacement).
    
    - **item_id**: The ID of the item to update
    - **item**: New item data
    """
    item_dict = item.model_dump(exclude_unset=True)
    
    updated_item = await repository.update(item_id, item_dict, partial=False)
    
    if not updated_item:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Item with id '{item_id}' not found"
        )
    
    return ItemResponse(**updated_item)


@router.patch(
    "/{item_id}",
    response_model=ItemResponse,
    summary="Partially update item",
    description="Update specific fields of an item (partial update)"
)
async def partial_update_item(
    item_id: str,
    item: ItemPartialUpdate,
    repository: BaseRepository = Depends(get_repository)
) -> ItemResponse:
    """
    Partially update an item (PATCH - update only provided fields).
    
    - **item_id**: The ID of the item to update
    - **item**: Fields to update (only provided fields will be updated)
    """
    item_dict = item.model_dump(exclude_unset=True)
    
    if not item_dict:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No fields provided for update"
        )
    
    updated_item = await repository.update(item_id, item_dict, partial=True)
    
    if not updated_item:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Item with id '{item_id}' not found"
        )
    
    return ItemResponse(**updated_item)


@router.delete(
    "/{item_id}",
    response_model=MessageResponse,
    summary="Delete item",
    description="Delete an item by its ID"
)
async def delete_item(
    item_id: str,
    repository: BaseRepository = Depends(get_repository)
) -> MessageResponse:
    """
    Delete an item by ID.
    
    - **item_id**: The ID of the item to delete
    """
    deleted = await repository.delete(item_id)
    
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Item with id '{item_id}' not found"
        )
    
    return MessageResponse(message=f"Item with id '{item_id}' successfully deleted")


@router.get(
    "/search/{field}",
    response_model=ItemListResponse,
    summary="Search items by field",
    description="Search items by a specific field value"
)
async def search_items(
    field: str,
    value: str = Query(..., description="Search value"),
    skip: int = Query(0, ge=0, description="Number of items to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of items to return"),
    repository: BaseRepository = Depends(get_repository)
) -> ItemListResponse:
    """
    Search items by field value (case-insensitive partial match).
    
    - **field**: Field name to search in (e.g., 'name', 'description')
    - **value**: Search value
    - **skip**: Number of items to skip
    - **limit**: Maximum number of items to return
    """
    items, total = await repository.search(
        field=field,
        value=value,
        skip=skip,
        limit=limit
    )
    
    return ItemListResponse(
        items=[ItemResponse(**item) for item in items],
        total=total,
        skip=skip,
        limit=limit
    )
