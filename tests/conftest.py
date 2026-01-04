"""
Pytest configuration for the Kader test suite.
"""

import sys
import os

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


# If we need any fixtures, we can add them here
# For now, we just need the path configuration
