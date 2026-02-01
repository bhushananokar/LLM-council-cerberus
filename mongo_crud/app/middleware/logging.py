"""
Custom middleware for request logging.
"""

import time
import logging
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware to log all incoming requests and their response times.
    """
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Log request details and measure response time.
        
        Args:
            request: Incoming request
            call_next: Next middleware or route handler
            
        Returns:
            Response from the next handler
        """
        start_time = time.time()
        
        # Log request
        logger.info(
            f"Incoming request: {request.method} {request.url.path} "
            f"from {request.client.host if request.client else 'unknown'}"
        )
        
        # Process request
        response = await call_next(request)
        
        # Calculate processing time
        process_time = time.time() - start_time
        
        # Log response
        logger.info(
            f"Request completed: {request.method} {request.url.path} "
            f"- Status: {response.status_code} - Time: {process_time:.3f}s"
        )
        
        # Add custom header with process time
        response.headers["X-Process-Time"] = str(process_time)
        
        return response
