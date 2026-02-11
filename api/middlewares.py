from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from typing import Callable, Any
import logging

from utils.get_env import get_can_change_keys_env
from utils.user_config import update_env_with_user_config
from services.rate_limiter import check_rate_limit

logger = logging.getLogger(__name__)

# Endpoints that should have rate limiting applied
RATE_LIMITED_ENDPOINTS = [
    "/api/v1/ppt/generate",
    "/api/v1/ppt/generate_outline",
    "/api/v1/ppt/generate_slides",
    "/api/v1/ppt/generate_images",
    "/api/v1/ppt/image",
]


class UserConfigEnvUpdateMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if get_can_change_keys_env() != "false":
            update_env_with_user_config()
        return await call_next(request)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Middleware to apply rate limiting to LLM endpoints"""

    async def dispatch(self, request: Request, call_next: Callable) -> Any:
        """Apply rate limiting based on endpoint and client identifier"""
        
        # Check if endpoint should be rate limited
        path = request.url.path
        should_limit = any(path.startswith(ep) for ep in RATE_LIMITED_ENDPOINTS)
        
        if not should_limit:
            return await call_next(request)
        
        # Get client identifier (user_id from query/body, IP, etc.)
        identifier = self._get_identifier(request)
        
        # Check rate limit
        is_allowed, headers = check_rate_limit(identifier, tokens_cost=1)
        
        if not is_allowed:
            logger.warning(f"Rate limit exceeded for {identifier} on {path}")
            return JSONResponse(
                status_code=429,
                content={
                    "error": "rate_limit_exceeded",
                    "message": f"Too many API requests. Limit: {headers['X-RateLimit-Limit']} calls per {headers['X-RateLimit-Window-Seconds']} seconds",
                    "retry_after": headers.get("X-RateLimit-Reset"),
                },
                headers=headers,
            )
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers to response
        for key, value in headers.items():
            response.headers[key] = value
        
        return response
    
    def _get_identifier(self, request: Request) -> str:
        """Extract unique identifier for rate limiting"""
        # Priority: user_id in query params -> session/user from headers -> IP address
        
        # Check query parameters
        user_id = request.query_params.get("user_id")
        if user_id:
            return f"user:{user_id}"
        
        # Check headers
        session_id = request.headers.get("X-Session-ID")
        if session_id:
            return f"session:{session_id}"
        
        user_header = request.headers.get("X-User-ID")
        if user_header:
            return f"user:{user_header}"
        
        # Fall back to IP address
        client_ip = request.client.host if request.client else "unknown"
        return f"ip:{client_ip}"
