"""
Main FastAPI application with middleware, CORS, and lifecycle events.
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pymongo.errors import PyMongoError

from app.core.config import settings
from app.database.mongodb import mongodb
from app.routers import items
from app.middleware.logging import RequestLoggingMiddleware

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("app.log")
    ]
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events.
    """
    # Startup
    logger.info("Starting application...")
    try:
        await mongodb.connect()
        logger.info("Application started successfully")
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down application...")
    await mongodb.close()
    logger.info("Application shut down successfully")


# Initialize FastAPI application
app = FastAPI(
    title=settings.project_name,
    description=settings.description,
    version=settings.version,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Add request logging middleware
app.add_middleware(RequestLoggingMiddleware)


# Exception handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Handle Pydantic validation errors with detailed error messages.
    """
    logger.error(f"Validation error: {exc.errors()}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "detail": "Validation error",
            "errors": exc.errors()
        }
    )


@app.exception_handler(PyMongoError)
async def mongodb_exception_handler(request: Request, exc: PyMongoError):
    """
    Handle MongoDB-specific errors.
    """
    logger.error(f"MongoDB error: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "Database error occurred",
            "error_code": "DB_ERROR"
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """
    Handle all other unexpected exceptions.
    """
    logger.error(f"Unexpected error: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "An unexpected error occurred",
            "error_code": "INTERNAL_ERROR"
        }
    )


# Include routers
app.include_router(
    items.router,
    prefix=settings.api_v1_prefix,
    tags=["items"]
)


# Root endpoint
@app.get("/", tags=["root"])
async def root():
    """
    Root endpoint with API information.
    """
    return {
        "message": "Welcome to MongoDB CRUD API",
        "version": settings.version,
        "docs": "/docs",
        "redoc": "/redoc"
    }


# Health check endpoint
@app.get("/health", tags=["health"])
async def health_check():
    """
    Health check endpoint to verify API and database status.
    """
    try:
        # Verify database connection
        db = await mongodb.get_database()
        await db.command("ping")
        
        return {
            "status": "healthy",
            "database": "connected",
            "environment": settings.environment
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "unhealthy",
                "database": "disconnected",
                "error": str(e)
            }
        )


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=settings.port,
        reload=(settings.environment == "development"),
        log_level="info"
    )
