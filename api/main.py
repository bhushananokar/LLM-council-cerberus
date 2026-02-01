"""
FastAPI Application
===================
REST API for the LLM Council supply chain security system.

Endpoints:
- POST /analyze - Analyze a package
- GET /health - Health check
- GET /stats - System statistics
- GET /config - System configuration
"""

import logging
import os
import time
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.models import (
    PackageData,
    CouncilDecision,
    AnalysisRequest,
    AnalysisResponse,
    HealthCheckResponse
)
from src.orchestrator import (
    get_orchestrator, 
    analyze_package, 
    health_check as system_health_check,
    set_council_repository
)
from src.debate_orchestrator import set_debate_repository
from src.utils import setup_logging, print_decision_summary
from config.settings import settings

# Import MongoDB components from mongo_crud
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'mongo_crud'))
from app.database.mongodb import MongoDB
from app.database.council_repository import CouncilRepository
from app.routers.council import router as council_router

# Setup logging
setup_logging(settings.LOG_LEVEL, settings.LOG_FILE)
logger = logging.getLogger(__name__)

# Global MongoDB instance
mongodb = MongoDB()
council_repo = None


# ============================================================================
# LIFESPAN CONTEXT MANAGER
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events.
    
    Args:
        app: FastAPI application
    """
    global council_repo
    
    # Startup
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info(f"Environment: {settings.ENVIRONMENT}")
    
    # Initialize MongoDB if configured
    mongodb_uri = os.getenv("MONGODB_URI")
    if mongodb_uri:
        try:
            logger.info("Connecting to MongoDB...")
            await mongodb.connect()
            council_repo = CouncilRepository(mongodb.db)
            
            # Set repository in orchestrators
            set_council_repository(council_repo)
            set_debate_repository(council_repo)
            
            logger.info("MongoDB connected and repositories configured")
        except Exception as e:
            logger.error(f"MongoDB connection failed: {str(e)}")
            logger.warning("Continuing without MongoDB storage")
    else:
        logger.info("MONGODB_URI not configured - running without MongoDB storage")
    
    # Initialize orchestrator
    orchestrator = get_orchestrator()
    logger.info("Orchestrator initialized")
    
    # Validate configuration
    try:
        settings.validate_settings()
        logger.info("Configuration validated successfully")
    except ValueError as e:
        logger.error(f"Configuration validation failed: {str(e)}")
        if settings.ENVIRONMENT == "production":
            raise
    
    logger.info(f"API server ready on {settings.API_HOST}:{settings.API_PORT}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down API server...")
    
    # Disconnect MongoDB
    if mongodb.client:
        await mongodb.disconnect()
        logger.info("MongoDB disconnected")
    
    logger.info("Cleanup complete")


# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(
    title=settings.APP_NAME,
    description="AI-powered supply chain security analysis using LLM consensus",
    version=settings.APP_VERSION,
    lifespan=lifespan
)

# Include council router for MongoDB analytics
app.include_router(council_router, prefix="/api/v1")

# ============================================================================
# MIDDLEWARE
# ============================================================================

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
    allow_methods=settings.CORS_ALLOW_METHODS,
    allow_headers=settings.CORS_ALLOW_HEADERS,
)


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests."""
    start_time = time.time()
    
    # Log request
    logger.info(f"Request: {request.method} {request.url.path}")
    
    # Process request
    response = await call_next(request)
    
    # Log response
    duration = time.time() - start_time
    logger.info(
        f"Response: {request.method} {request.url.path} - "
        f"Status: {response.status_code} - Duration: {duration:.2f}s"
    )
    
    return response


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class AnalyzePackageRequest(BaseModel):
    """Request model for package analysis."""
    package_data: PackageData
    force_reanalysis: bool = Field(default=False, description="Force reanalysis even if cached")
    skip_cache: bool = Field(default=False, description="Skip cache lookup and storage")
    
    class Config:
        json_schema_extra = {
            "example": {
                "package_data": {
                    "package_name": "suspicious-lib",
                    "version": "1.2.3",
                    "registry": "npm",
                    "description": "A utility library",
                    "code_segments": [
                        {
                            "code": "eval(atob('payload'))",
                            "location": "index.js:45",
                            "reason": "eval_with_encoded_string"
                        }
                    ]
                },
                "force_reanalysis": False,
                "skip_cache": False
            }
        }


class AnalyzePackageResponse(BaseModel):
    """Response model for package analysis."""
    success: bool
    decision: CouncilDecision
    cached: bool = False
    processing_time_seconds: float
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "decision": {
                    "package_name": "suspicious-lib",
                    "package_version": "1.2.3",
                    "verdict": "malicious",
                    "threat_level": "critical",
                    "final_risk_score": 96.0,
                    "final_confidence": 92.0
                },
                "cached": False,
                "processing_time_seconds": 4.2
            }
        }


class ErrorResponse(BaseModel):
    """Error response model."""
    success: bool = False
    error: str
    detail: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": False,
                "error": "Analysis failed",
                "detail": "Groq API connection timeout"
            }
        }


# ============================================================================
# ROUTES
# ============================================================================

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "operational",
        "documentation": "/docs",
        "health": "/health",
        "endpoints": {
            "analyze": "POST /analyze",
            "health": "GET /health",
            "stats": "GET /stats",
            "config": "GET /config"
        }
    }


@app.post(
    "/analyze",
    response_model=AnalyzePackageResponse,
    status_code=status.HTTP_200_OK,
    tags=["Analysis"],
    summary="Analyze a package for security threats"
)
async def analyze_package_endpoint(request: AnalyzePackageRequest):
    """
    Analyze a package using the LLM Council.
    
    **Request Body:**
    - package_data: Package information including code segments, metadata, etc.
    - force_reanalysis: Force new analysis even if cached result exists
    - skip_cache: Don't use or store cache
    
    **Returns:**
    - CouncilDecision with complete analysis results
    - Risk score (0-100)
    - Verdict (malicious/benign/uncertain)
    - Threat level (critical/high/medium/low)
    - Recommended actions
    
    **Example:**
```json
    {
      "package_data": {
        "package_name": "test-package",
        "version": "1.0.0",
        "description": "Test package",
        "code_segments": []
      }
    }
```
    """
    start_time = time.time()
    
    try:
        logger.info(f"Received analysis request for {request.package_data.package_name}")
        
        # Get orchestrator
        orchestrator = get_orchestrator()
        
        # Check cache first (unless disabled)
        cached = False
        if not request.skip_cache and not request.force_reanalysis:
            cached_decision = orchestrator._check_cache(request.package_data)
            if cached_decision:
                logger.info(f"Returning cached result for {request.package_data.package_name}")
                processing_time = time.time() - start_time
                return AnalyzePackageResponse(
                    success=True,
                    decision=cached_decision,
                    cached=True,
                    processing_time_seconds=round(processing_time, 2)
                )
        
        # Perform analysis
        decision = await analyze_package(
            request.package_data,
            force_reanalysis=request.force_reanalysis,
            skip_cache=request.skip_cache
        )
        
        processing_time = time.time() - start_time
        
        logger.info(
            f"Analysis complete for {request.package_data.package_name}: "
            f"Verdict={decision.verdict}, Risk={decision.final_risk_score}"
        )
        
        return AnalyzePackageResponse(
            success=True,
            decision=decision,
            cached=False,
            processing_time_seconds=round(processing_time, 2)
        )
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis failed: {str(e)}"
        )


@app.get(
    "/health",
    response_model=HealthCheckResponse,
    status_code=status.HTTP_200_OK,
    tags=["System"],
    summary="Health check endpoint"
)
async def health_check_endpoint():
    """
    Check system health status.
    
    **Returns:**
    - Overall status (healthy/degraded/unhealthy)
    - Component health (agents, cache, LLM APIs)
    - System statistics
    
    **Status Codes:**
    - 200: System healthy
    - 503: System degraded or unhealthy
    """
    try:
        health = system_health_check()
        
        # Determine HTTP status code
        if health["status"] == "healthy":
            status_code = status.HTTP_200_OK
        else:
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        
        response = HealthCheckResponse(
            status=health["status"],
            version=settings.APP_VERSION,
            components=health.get("components", {}),
            stats=health.get("stats", {})
        )
        
        return JSONResponse(
            status_code=status_code,
            content=response.dict()
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "unhealthy",
                "version": settings.APP_VERSION,
                "error": str(e)
            }
        )


@app.get(
    "/stats",
    status_code=status.HTTP_200_OK,
    tags=["System"],
    summary="Get system statistics"
)
async def get_stats_endpoint():
    """
    Get system statistics.
    
    **Returns:**
    - Total analyses performed
    - Cache hit rate
    - Token usage statistics
    - Cost statistics
    - Average analysis time
    """
    try:
        orchestrator = get_orchestrator()
        stats = orchestrator.get_stats()
        
        return {
            "success": True,
            "stats": stats
        }
        
    except Exception as e:
        logger.error(f"Failed to get stats: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve statistics: {str(e)}"
        )


@app.get(
    "/config",
    status_code=status.HTTP_200_OK,
    tags=["System"],
    summary="Get system configuration"
)
async def get_config_endpoint():
    """
    Get system configuration (non-sensitive).
    
    **Returns:**
    - Model configurations
    - Agent weights
    - Consensus thresholds
    - Feature flags
    
    **Note:** API keys and sensitive data are excluded.
    """
    try:
        config = settings.to_dict()
        
        return {
            "success": True,
            "config": config
        }
        
    except Exception as e:
        logger.error(f"Failed to get config: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve configuration: {str(e)}"
        )


@app.post(
    "/cache/clear",
    status_code=status.HTTP_200_OK,
    tags=["Admin"],
    summary="Clear cache (admin only)"
)
async def clear_cache_endpoint():
    """
    Clear all cached analysis results.
    
    **⚠️ Warning:** This will clear all cached decisions.
    Use with caution in production.
    
    **Returns:**
    - Success status
    """
    try:
        from src.cache import clear_cache
        
        success = clear_cache()
        
        if success:
            logger.warning("Cache cleared via API endpoint")
            return {
                "success": True,
                "message": "Cache cleared successfully"
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to clear cache"
            )
            
    except Exception as e:
        logger.error(f"Failed to clear cache: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clear cache: {str(e)}"
        )


@app.get(
    "/cache/stats",
    status_code=status.HTTP_200_OK,
    tags=["Admin"],
    summary="Get cache statistics"
)
async def cache_stats_endpoint():
    """
    Get cache statistics.
    
    **Returns:**
    - Cache hit rate
    - Total keys
    - Memory usage
    """
    try:
        from src.cache import get_cache_stats
        
        stats = get_cache_stats()
        
        return {
            "success": True,
            "cache_stats": stats
        }
        
    except Exception as e:
        logger.error(f"Failed to get cache stats: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve cache statistics: {str(e)}"
        )


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": exc.detail,
            "status_code": exc.status_code
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "error": "Internal server error",
            "detail": str(exc) if settings.ENVIRONMENT != "production" else None
        }
    )


# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info(f"Environment: {settings.ENVIRONMENT}")
    logger.info(f"Host: {settings.API_HOST}:{settings.API_PORT}")
    
    uvicorn.run(
        "api.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.API_RELOAD,
        workers=settings.API_WORKERS,
        log_level=settings.LOG_LEVEL.lower()
    )