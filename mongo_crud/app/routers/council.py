"""
API Router for LLM Council analysis results.
Provides endpoints to query and retrieve stored council analyses and debate results.
"""

from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query, status, Depends
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.database.mongodb import MongoDB
from app.database.council_repository import CouncilRepository
from app.schemas.council import (
    CouncilAnalysisResponse,
    DebateResultResponse,
    CouncilAnalysisListResponse,
    DebateResultListResponse,
    CouncilStatsResponse
)


router = APIRouter(prefix="/council", tags=["Council Analytics"])


# Dependency to get database
async def get_db() -> AsyncIOMotorDatabase:
    """Get database dependency."""
    from app.main import mongodb
    if not mongodb.db:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database not connected"
        )
    return mongodb.db


# ============================================================================
# COUNCIL ANALYSIS ENDPOINTS
# ============================================================================

@router.get(
    "/analyses",
    response_model=CouncilAnalysisListResponse,
    summary="Get recent council analyses",
    description="Retrieve recent council analyses with pagination support"
)
async def get_analyses(
    limit: int = Query(20, ge=1, le=100, description="Number of results to return"),
    skip: int = Query(0, ge=0, description="Number of results to skip"),
    db: AsyncIOMotorDatabase = Depends(get_db)
):
    """Get recent council analyses."""
    repo = CouncilRepository(db)
    analyses = await repo.get_recent_analyses(limit=limit, skip=skip)
    
    # Convert ObjectId to string for JSON serialization
    for analysis in analyses:
        analysis["_id"] = str(analysis["_id"])
    
    return {
        "total": len(analyses),
        "analyses": analyses
    }


@router.get(
    "/analyses/decision/{decision_id}",
    response_model=CouncilAnalysisResponse,
    summary="Get analysis by decision ID",
    description="Retrieve a specific council analysis by its decision ID"
)
async def get_analysis_by_decision_id(
    decision_id: str,
    db: AsyncIOMotorDatabase = Depends(get_db)
):
    """Get analysis by decision ID."""
    repo = CouncilRepository(db)
    analysis = await repo.get_analysis_by_decision_id(decision_id)
    
    if not analysis:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Analysis with decision_id '{decision_id}' not found"
        )
    
    analysis["_id"] = str(analysis["_id"])
    return analysis


@router.get(
    "/analyses/package/{package_name}",
    response_model=CouncilAnalysisListResponse,
    summary="Get analyses by package name",
    description="Retrieve all analyses for a specific package"
)
async def get_analyses_by_package(
    package_name: str,
    limit: int = Query(10, ge=1, le=50, description="Number of results to return"),
    db: AsyncIOMotorDatabase = Depends(get_db)
):
    """Get analyses for a specific package."""
    repo = CouncilRepository(db)
    analyses = await repo.get_analyses_by_package(package_name, limit=limit)
    
    for analysis in analyses:
        analysis["_id"] = str(analysis["_id"])
    
    return {
        "total": len(analyses),
        "analyses": analyses
    }


@router.get(
    "/analyses/verdict/{verdict}",
    response_model=CouncilAnalysisListResponse,
    summary="Get analyses by verdict",
    description="Retrieve analyses filtered by verdict (malicious, benign, uncertain, error)"
)
async def get_analyses_by_verdict(
    verdict: str,
    limit: int = Query(50, ge=1, le=100),
    skip: int = Query(0, ge=0),
    db: AsyncIOMotorDatabase = Depends(get_db)
):
    """Get analyses filtered by verdict."""
    valid_verdicts = ["malicious", "benign", "uncertain", "error"]
    if verdict not in valid_verdicts:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid verdict. Must be one of: {', '.join(valid_verdicts)}"
        )
    
    repo = CouncilRepository(db)
    analyses = await repo.get_analyses_by_verdict(verdict, limit=limit, skip=skip)
    
    for analysis in analyses:
        analysis["_id"] = str(analysis["_id"])
    
    return {
        "total": len(analyses),
        "analyses": analyses
    }


@router.get(
    "/analyses/stats",
    response_model=CouncilStatsResponse,
    summary="Get council analysis statistics",
    description="Retrieve aggregate statistics about all council analyses"
)
async def get_analysis_stats(db: AsyncIOMotorDatabase = Depends(get_db)):
    """Get analysis statistics."""
    repo = CouncilRepository(db)
    stats = await repo.get_analysis_stats()
    return stats


# ============================================================================
# DEBATE RESULT ENDPOINTS
# ============================================================================

@router.get(
    "/debates",
    response_model=DebateResultListResponse,
    summary="Get recent debates",
    description="Retrieve recent debate results with pagination support"
)
async def get_debates(
    limit: int = Query(20, ge=1, le=100),
    skip: int = Query(0, ge=0),
    db: AsyncIOMotorDatabase = Depends(get_db)
):
    """Get recent debates."""
    repo = CouncilRepository(db)
    debates = await repo.get_recent_debates(limit=limit, skip=skip)
    
    for debate in debates:
        debate["_id"] = str(debate["_id"])
    
    return {
        "total": len(debates),
        "debates": debates
    }


@router.get(
    "/debates/debate/{debate_id}",
    response_model=DebateResultResponse,
    summary="Get debate by ID",
    description="Retrieve a specific debate result by its debate ID"
)
async def get_debate_by_id(
    debate_id: str,
    db: AsyncIOMotorDatabase = Depends(get_db)
):
    """Get debate by ID."""
    repo = CouncilRepository(db)
    debate = await repo.get_debate_by_id(debate_id)
    
    if not debate:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Debate with debate_id '{debate_id}' not found"
        )
    
    debate["_id"] = str(debate["_id"])
    return debate


@router.get(
    "/debates/package/{package_name}",
    response_model=DebateResultListResponse,
    summary="Get debates by package name",
    description="Retrieve all debate results for a specific package"
)
async def get_debates_by_package(
    package_name: str,
    limit: int = Query(10, ge=1, le=50),
    db: AsyncIOMotorDatabase = Depends(get_db)
):
    """Get debates for a specific package."""
    repo = CouncilRepository(db)
    debates = await repo.get_debates_by_package(package_name, limit=limit)
    
    for debate in debates:
        debate["_id"] = str(debate["_id"])
    
    return {
        "total": len(debates),
        "debates": debates
    }


@router.get(
    "/debates/stats",
    summary="Get debate statistics",
    description="Retrieve aggregate statistics about all debates"
)
async def get_debate_stats(db: AsyncIOMotorDatabase = Depends(get_db)):
    """Get debate statistics."""
    repo = CouncilRepository(db)
    stats = await repo.get_debate_stats()
    return stats
