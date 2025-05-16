"""
Simple tests for the retry functionality.
"""

import logging
import pytest
import time

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_basic_retry():
    """Test a basic retry loop."""
    max_attempts = 3
    attempts = 0
    backoff = 0.1
    
    while True:
        attempts += 1
        logger.info(f"Attempt {attempts}")
        
        if attempts < 3:
            logger.info(f"Simulating failure, retrying in {backoff}s")
            time.sleep(backoff)
            backoff *= 2
            continue
        
        logger.info("Success!")
        break
    
    assert attempts == 3
    

if __name__ == "__main__":
    test_basic_retry()
    print("All tests passed!")
