import logging

def configure_external_loggers(level=logging.WARNING):
    """Configure external library loggers to a specific level.
    
    This function sets common HTTP and networking libraries to the specified
    log level (WARNING by default) to suppress excessive INFO messages.
    
    Args:
        level: The logging level to set for external libraries
    """
    for logger_name in [
        "httpx",           # HTTP client library often used by FAL
        "aiohttp",         # Async HTTP client
        "requests",        # Synchronous HTTP client
        "urllib3",         # Used by requests
        "fal_client",      # FAL.AI client library
        "asyncio",         # Asyncio debugging messages
    ]:
        logging.getLogger(logger_name).setLevel(level)