"""
ML Preprocessing Suite - Source Code
Main package initialization with enhanced ML capabilities.
"""

# Initialize logging and configuration on import
from .utils import setup_logging
from .config import get_config_manager

# Setup logging
setup_logging()

# Initialize configuration
config_manager = get_config_manager()

__version__ = "2.0.0"
__author__ = "Enhanced ML Suite Team"
