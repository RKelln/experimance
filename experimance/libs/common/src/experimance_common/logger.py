import logging
import os
from pathlib import Path
from typing import Optional


def get_log_directory() -> Path:    
    """
    Determine the appropriate log directory based on environment.
    
    Returns:
        Path to log directory - uses /var/log/experimance/ in production,
        logs/ directory in development
    """
    # Check if we're in a production environment (systemd service or root)
    is_production = (
        os.geteuid() == 0 or  # Running as root
        os.environ.get("EXPERIMANCE_ENV") == "production" or
        Path("/etc/experimance").exists()  # Production marker
    )
    
    if is_production:
        log_dir = Path("/var/log/experimance")
    else:
        # Development environment - use local logs directory
        try:
            from experimance_common.constants import PROJECT_ROOT
            log_dir = PROJECT_ROOT / "logs"
        except ImportError:
            # Fallback if constants not available
            log_dir = Path.cwd() / "logs"
        
    # Create log directory if it doesn't exist
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
    except (PermissionError, OSError) as e:
        # Only fall back to local logs if we're not in production or if the directory doesn't exist yet
        # In production, the deploy script should have created /var/log/experimance
        if is_production:
            # In production, this is likely a deployment issue - fail with clear error
            raise RuntimeError(
                f"Cannot create production log directory {log_dir}. "
                f"Run 'sudo ./infra/scripts/deploy.sh {os.environ.get('PROJECT', 'experimance')} install prod' "
                f"to set up system directories. Error: {e}"
            )
        else:
            # Fallback to local logs in development
            log_dir = Path.cwd() / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir

def get_log_file_path(log_filename: str = "application.log") -> str:
    """
    Determine the appropriate log file path based on environment.
    
    Args:
        log_filename: Name of the log file (default: "application.log")
    
    Returns:
        Path to log file - uses /var/log/experimance/ in production,
        logs/ directory in development
    """
    log_file = get_log_directory() / log_filename
    if not log_file.exists():
        # Create the log file if it doesn't exist
        try:
            log_file.touch(exist_ok=True)
        except OSError as e:
            # If we can't create the file in the production directory, fall back to local logs
            # This handles read-only filesystem issues
            log_dir = Path.cwd() / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file = log_dir / log_filename
            log_file.touch(exist_ok=True)
    return str(log_file)

def setup_logging(
    name: str = __name__,
    level: int = logging.INFO,
    log_filename: Optional[str] = None,
    include_console: Optional[bool] = None,  # Changed to Optional[bool] for auto-detection
    external_level: int = logging.WARNING
) -> logging.Logger:
    """
    Setup logging with adaptive file location and external logger configuration.
    
    Args:
        name: Logger name (usually __name__)
        level: Logging level for the main logger
        log_filename: Log file name (auto-generated if None)
        include_console: Whether to include console output. If None, auto-detects:
                        - Development: True (console + file)
                        - Production: False (file only, systemd handles console)
        external_level: Level for external libraries
    
    Returns:
        Configured logger instance
    """
    # Auto-detect console logging preference if not specified
    if include_console is None:
        # Check if we're in a production environment
        is_production = (
            os.geteuid() == 0 or  # Running as root
            os.environ.get("EXPERIMANCE_ENV") == "production" or
            Path("/etc/experimance").exists()  # Production marker
        )
        # In production, let systemd handle console output (file-only logging)
        # In development, show console output for debugging
        include_console = not is_production
    # Generate log filename if not provided
    if log_filename is None:
        # Extract service name from module name
        service_name = name.split('.')[-1] if '.' in name else name
        log_filename = f"{service_name}.log"
    
    # Get adaptive log file path
    log_file = get_log_file_path(log_filename)
    
    # Configure the main logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    
    # Console handler (optional)
    handlers = []
    handlers.append(file_handler)
    
    if include_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        handlers.append(console_handler)
    
    # Format
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    for handler in handlers:
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    # Configure external loggers
    configure_external_loggers(external_level)
    
    return logger

def configure_external_loggers(level=logging.WARNING):
    """Configure external library loggers to a specific level.
    
    This function sets common HTTP and networking libraries to the specified
    log level (WARNING by default) to suppress excessive INFO messages.
    
    Args:
        level: The logging level to set for external libraries
    """
    for logger_name in [
        "httpx",           # HTTP client library often used by FAL
        "httpcore.http11", # HTTP/1.1 implementation used by httpx
        "httpcore.connection", 
        "aiohttp",         # Async HTTP client
        "requests",        # Synchronous HTTP client
        "urllib3",         # Used by requests
        "fal_client",      # FAL.AI client library
        "asyncio",         # Asyncio debugging messages
        "PIL",             # Python Imaging Library (Pillow)
        "PIL.PngImagePlugin",  # Specific PIL PNG plugin that's very verbose
        "websockets",      # WebSocket library
        "websockets.client",  # WebSocket client library
        "openai._base_client",  # OpenAI base client logging
    ]:
        logger = logging.getLogger(logger_name)
        logger.setLevel(level)
        # Set all handlers to the same level
        for handler in logger.handlers:
            handler.setLevel(level)
        # Prevent propagation to root logger if not desired
        logger.propagate = False