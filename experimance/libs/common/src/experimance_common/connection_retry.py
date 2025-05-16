"""
Connection retry utilities.
"""

import asyncio
import logging
import random
import time
from typing import Any, Callable, TypeVar, Optional

# Setup logging
logger = logging.getLogger(__name__)

# Type variable for return type
T = TypeVar('T')


def retry_with_backoff(
    func: Callable[..., T],
    max_retries: int = 5,
    initial_backoff: float = 0.1,
    max_backoff: float = 30.0,
    backoff_factor: float = 2.0,
    jitter: bool = True,
    exceptions: tuple = (Exception,),
    on_retry: Optional[Callable[[Exception, int, float], None]] = None
) -> T:
    """
    Execute a function with retry and exponential backoff.
    
    Args:
        func: Function to execute
        max_retries: Maximum number of retries
        initial_backoff: Initial backoff time in seconds
        max_backoff: Maximum backoff time in seconds
        backoff_factor: Backoff multiplication factor
        jitter: Whether to add jitter to backoff time
        exceptions: Tuple of exceptions to catch and retry
        on_retry: Optional callback for retry events
        
    Returns:
        The return value of the function
        
    Raises:
        The last exception if all retries fail
    """
    retry_count = 0
    backoff = initial_backoff
    
    while True:
        try:
            return func()
        except exceptions as e:
            retry_count += 1
            
            if retry_count > max_retries:
                logger.error(f"Failed after {max_retries} retries: {e}")
                raise
            
            # Calculate backoff with jitter
            if jitter:
                backoff = min(backoff * (1 + 0.2 * random.random()), max_backoff)
            else:
                backoff = min(backoff, max_backoff)
                
            if on_retry:
                on_retry(e, retry_count, backoff)
                
            logger.warning(f"Attempt {retry_count} failed: {e}. Retrying in {backoff:.2f}s")
            time.sleep(backoff)
            
            # Increase backoff for next retry
            backoff = min(backoff * backoff_factor, max_backoff)


async def async_retry_with_backoff(
    func: Callable[..., Any],
    max_retries: int = 5,
    initial_backoff: float = 0.1,
    max_backoff: float = 30.0,
    backoff_factor: float = 2.0,
    jitter: bool = True,
    exceptions: tuple = (Exception,),
    on_retry: Optional[Callable[[Exception, int, float], None]] = None
) -> Any:
    """
    Execute an async function with retry and exponential backoff.
    
    Args:
        func: Async function to execute
        max_retries: Maximum number of retries
        initial_backoff: Initial backoff time in seconds
        max_backoff: Maximum backoff time in seconds
        backoff_factor: Backoff multiplication factor
        jitter: Whether to add jitter to backoff time
        exceptions: Tuple of exceptions to catch and retry
        on_retry: Optional callback for retry events
        
    Returns:
        The return value of the function
        
    Raises:
        The last exception if all retries fail
    """
    retry_count = 0
    backoff = initial_backoff
    
    while True:
        try:
            return await func()
        except exceptions as e:
            retry_count += 1
            
            if retry_count > max_retries:
                logger.error(f"Failed after {max_retries} retries: {e}")
                raise
            
            # Calculate backoff with jitter
            if jitter:
                backoff = min(backoff * (1 + 0.2 * random.random()), max_backoff)
            else:
                backoff = min(backoff, max_backoff)
                
            if on_retry:
                on_retry(e, retry_count, backoff)
                
            logger.warning(f"Attempt {retry_count} failed: {e}. Retrying in {backoff:.2f}s")
            await asyncio.sleep(backoff)
            
            # Increase backoff for next retry
            backoff = min(backoff * backoff_factor, max_backoff)
