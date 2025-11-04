"""
Logging Configuration
Sets up logging for the application.
"""

import logging
import logging.handlers
from pathlib import Path
from typing import Optional
import sys

from ..config import get_config

def setup_logging(
    level: Optional[str] = None,
    log_file: Optional[str] = None,
    format_string: Optional[str] = None,
    max_file_size: Optional[int] = None,
    backup_count: Optional[int] = None
) -> None:
    """
    Setup logging configuration for the application.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file
        format_string: Log format string
        max_file_size: Maximum log file size in bytes
        backup_count: Number of backup log files to keep
    """
    # Get configuration values
    if level is None:
        level = get_config("logging.level", "INFO")
    if log_file is None:
        log_file = get_config("logging.file", "logs/ml_preprocessing.log")
    if format_string is None:
        format_string = get_config("logging.format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    if max_file_size is None:
        max_file_size = get_config("logging.max_file_size", 10485760)  # 10MB
    if backup_count is None:
        backup_count = get_config("logging.backup_count", 5)

    # Convert string level to logging level
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Create formatter
    formatter = logging.Formatter(format_string)

    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Remove existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler with rotation
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.handlers.RotatingFileHandler(
            log_path,
            maxBytes=max_file_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # Log the setup
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured - Level: {level}, File: {log_file}")

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the specified name.

    Args:
        name: Logger name (usually __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)

def log_function_call(logger: logging.Logger, level: int = logging.DEBUG):
    """
    Decorator to log function calls.

    Args:
        logger: Logger instance
        level: Logging level for the decorator

    Returns:
        Decorated function
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger.log(level, f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
            try:
                result = func(*args, **kwargs)
                logger.log(level, f"{func.__name__} completed successfully")
                return result
            except Exception as e:
                logger.error(f"{func.__name__} failed with error: {e}")
                raise
        return wrapper
    return decorator

class LoggerMixin:
    """
    Mixin class that provides a logger property to classes.
    """

    @property
    def logger(self) -> logging.Logger:
        """
        Get logger for this class.

        Returns:
            Logger instance
        """
        return logging.getLogger(self.__class__.__module__ + '.' + self.__class__.__name__)
