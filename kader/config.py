"""
Kader configuration management module.

This module handles the creation and management of the .kader directory
in the user's home directory, including creating the required .env file
with OLLAMA_API_KEY and loading environment variables.
"""

import os
import sys
from pathlib import Path


def load_env_file(env_file_path):
    """
    Load environment variables from a .env file.
    This function reads the .env file and sets the variables in os.environ.
    """
    if not env_file_path.exists():
        return False

    try:
        with open(env_file_path, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                # Skip comments and empty lines
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip()

                    # Remove surrounding quotes if present
                    if value.startswith('"') and value.endswith('"'):
                        value = value[1:-1]
                    elif value.startswith("'") and value.endswith("'"):
                        value = value[1:-1]

                    # Set the environment variable if it's not already set
                    if key not in os.environ:
                        os.environ[key] = value
        return True
    except Exception as e:
        print(f"Error loading .env file: {str(e)}")
        return False


def get_kader_directory():
    """Get the path to the .kader directory in the user's home directory."""
    return Path.home() / ".kader"


def ensure_kader_directory():
    """
    Ensure that the .kader directory exists in the user's home directory.
    Creates it if it doesn't exist.
    """
    kader_dir = get_kader_directory()

    # Create the directory if it doesn't exist
    kader_dir.mkdir(exist_ok=True)

    # Ensure the directory has appropriate permissions on Unix-like systems
    if not sys.platform.startswith("win"):
        kader_dir.chmod(0o755)

    return kader_dir


def ensure_env_file(kader_dir):
    """
    Ensure that the .env file exists in the .kader directory with the
    required OLLAMA_API_KEY configuration.
    """
    env_file = kader_dir / ".env"

    # Create the .env file if it doesn't exist
    if not env_file.exists():
        env_file.write_text("OLLAMA_API_KEY=''\n", encoding="utf-8")

        # Set appropriate permissions for the .env file on Unix-like systems
        if not sys.platform.startswith("win"):
            env_file.chmod(0o644)

    return env_file


def initialize_kader_config():
    """
    Initialize the .kader directory in the user's home directory with required configuration files.
    This function creates the directory, sets up the .env file with OLLAMA_API_KEY,
    and loads all environment variables from the .env file.
    """
    try:
        # Ensure the .kader directory exists
        kader_dir = ensure_kader_directory()

        # Ensure the .env file exists with the required configuration
        ensure_env_file(kader_dir)

        # Load environment variables from the .env file
        env_file_path = kader_dir / ".env"
        load_env_file(env_file_path)

        # Optionally add the .kader directory to the Python path so it can be accessed
        kader_dir_str = str(kader_dir)
        if kader_dir_str not in sys.path:
            sys.path.insert(0, kader_dir_str)

        return kader_dir, True

    except PermissionError as e:
        print(
            f"Permission denied: Unable to create .kader directory in {Path.home()}. {str(e)}"
        )
        return None, False
    except Exception as e:
        print(f"Error initializing Kader config: {str(e)}")
        return None, False


# Initialize the configuration when the module is imported
kader_dir, success = initialize_kader_config()

if success:
    # Define constants for other modules to use
    KADER_DIR = kader_dir
    ENV_FILE_PATH = kader_dir / ".env"

__version__ = "0.1.0"
__author__ = "Kader Project"
__all__ = [
    "KADER_DIR",
    "ENV_FILE_PATH",
    "initialize_kader_config",
    "get_kader_directory",
    "ensure_kader_directory",
    "ensure_env_file",
    "load_env_file",
]
