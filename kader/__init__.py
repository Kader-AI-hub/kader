"""
Main Kader module initialization.

This module sets up the required configuration when imported, including
creating the .kader directory in the user's home directory.
"""

from .config import ENV_FILE_PATH, KADER_DIR, initialize_kader_config
from .providers import *  # noqa: F401, F403
from .tools import *  # noqa: F401, F403

# Initialize the configuration when the module is imported
initialize_kader_config()

__version__ = "0.1.0"
__author__ = "Kader Project"
__all__ = [
    "KADER_DIR",
    "ENV_FILE_PATH",
    "initialize_kader_config",
    # Export everything from providers and tools
]
