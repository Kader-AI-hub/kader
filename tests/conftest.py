"""
Pytest configuration for the Kader test suite.
"""

import os
import sys

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)



# If we need any fixtures, we can add them here
import pytest
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch
from loguru import logger


@pytest.fixture(autouse=True)
def mock_kader_home():
    """
    Global fixture to ensure no test ever touches the real ~/.kader directory.
    This creates a temporary directory and mocks Path.home() to return it,
    effectively isolating all tests to a temporary environment.
    """
    # Create a temporary directory to act as the fake home
    temp_home = tempfile.mkdtemp()
    temp_home_path = Path(temp_home)
    
    # Mock Path.home() to return our temporary directory
    # We use a patcher so we can start/stop it cleanly
    patcher = patch('pathlib.Path.home', return_value=temp_home_path)
    mock_home = patcher.start()
    
    yield temp_home_path
    
    # Cleanup
    patcher.stop()
    
    # Remove all logger handlers to release file locks on Windows
    try:
        logger.remove()
    except Exception:
        pass
        
    shutil.rmtree(temp_home)


