"""
Rate limiter service for AI API calls to prevent excessive usage
Implements token bucket algorithm with configurable limits per time window
"""

import time
from typing import Dict, Optional
from datetime import datetime, timedelta
from collections import defaultdict
import logging
import os

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Token bucket rate limiter for tracking LLM API usage
    
    Configuration via environment variables:
    - RATE_LIMIT_CALLS: Number of LLM API calls allowed (default: 100)
    - RATE_LIMIT_WINDOW: Time window in seconds (default: 3600 = 1 hour)
    - RATE_LIMIT_ENABLED: Enable/disable rate limiting (default: true)
    """
    
    def __init__(
        self,
        calls_per_window: int = None,
        window_seconds: int = None,
        enabled: bool = None,
    ):
        """
        Initialize rate limiter
        
        Args:
            calls_per_window: Max API calls allowed per window
            window_seconds: Time window in seconds
            enabled: Whether rate limiting is enabled
        """
        self.calls_per_window = calls_per_window or int(
            os.getenv("RATE_LIMIT_CALLS", "100")
        )
        self.window_seconds = window_seconds or int(
            os.getenv("RATE_LIMIT_WINDOW", "3600")
        )
        self.enabled = (
            enabled 
            if enabled is not None 
            else os.getenv("RATE_LIMIT_ENABLED", "true").lower() == "true"
        )
        
        # Track tokens per identifier (user id, IP, session, etc.)
        self.buckets: Dict[str, Dict] = defaultdict(
            lambda: {"tokens": self.calls_per_window, "last_refill": time.time()}
        )
        
        logger.info(
            f"Rate limiter initialized: {self.calls_per_window} calls per {self.window_seconds}s window"
        )
    
    def _refill_bucket(self, identifier: str) -> None:
        """Refill tokens based on elapsed time"""
        bucket = self.buckets[identifier]
        now = time.time()
        elapsed = now - bucket["last_refill"]
        
        # Calculate tokens to add (refill rate)
        refill_rate = self.calls_per_window / self.window_seconds
        tokens_to_add = elapsed * refill_rate
        
        # Cap tokens at max
        bucket["tokens"] = min(
            self.calls_per_window,
            bucket["tokens"] + tokens_to_add
        )
        bucket["last_refill"] = now
    
    def is_allowed(self, identifier: str, tokens_cost: int = 1) -> bool:
        """
        Check if API call is allowed under rate limit
        
        Args:
            identifier: Unique identifier (user_id, IP, session_id, etc.)
            tokens_cost: Number of tokens this operation costs (default: 1)
            
        Returns:
            True if allowed, False if rate limited
        """
        if not self.enabled:
            return True
        
        self._refill_bucket(identifier)
        bucket = self.buckets[identifier]
        
        if bucket["tokens"] >= tokens_cost:
            bucket["tokens"] -= tokens_cost
            logger.debug(
                f"Rate limit OK for {identifier}: {bucket['tokens']:.1f} tokens remaining"
            )
            return True
        
        logger.warning(
            f"Rate limit exceeded for {identifier}: {bucket['tokens']:.1f} tokens available, need {tokens_cost}"
        )
        return False
    
    def get_remaining_calls(self, identifier: str) -> float:
        """Get remaining calls for identifier"""
        if not self.enabled:
            return float("inf")
        
        self._refill_bucket(identifier)
        return self.buckets[identifier]["tokens"]
    
    def get_reset_time(self, identifier: str) -> Optional[datetime]:
        """Get when rate limit will reset for identifier"""
        if not self.enabled:
            return None
        
        bucket = self.buckets[identifier]
        elapsed = time.time() - bucket["last_refill"]
        if elapsed < self.window_seconds:
            seconds_until_reset = self.window_seconds - elapsed
            return datetime.now() + timedelta(seconds=seconds_until_reset)
        return None
    
    def reset(self, identifier: str = None) -> None:
        """Reset rate limit for identifier or all if identifier is None"""
        if identifier:
            self.buckets[identifier] = {
                "tokens": self.calls_per_window,
                "last_refill": time.time()
            }
            logger.info(f"Rate limit reset for {identifier}")
        else:
            self.buckets.clear()
            logger.info("Rate limit reset for all identifiers")


# Global rate limiter instance
_rate_limiter: Optional[RateLimiter] = None


def get_rate_limiter() -> RateLimiter:
    """Get or create global rate limiter instance"""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter()
    return _rate_limiter


def check_rate_limit(identifier: str, tokens_cost: int = 1) -> tuple[bool, Dict]:
    """
    Check rate limit and return status with headers
    
    Returns:
        (is_allowed, headers_dict)
    """
    limiter = get_rate_limiter()
    is_allowed = limiter.is_allowed(identifier, tokens_cost)
    remaining = limiter.get_remaining_calls(identifier)
    reset_time = limiter.get_reset_time(identifier)
    
    headers = {
        "X-RateLimit-Limit": str(limiter.calls_per_window),
        "X-RateLimit-Remaining": str(int(max(0, remaining))),
        "X-RateLimit-Window-Seconds": str(limiter.window_seconds),
    }
    
    if reset_time:
        headers["X-RateLimit-Reset"] = str(int(reset_time.timestamp()))
    
    return is_allowed, headers
