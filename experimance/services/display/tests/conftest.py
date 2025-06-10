#!/usr/bin/env python3
"""
Configuration for pytest to handle display tests properly.

This file ensures that tests that require a real display window
are skipped by default, but can be run with explicit commands.
"""

import pytest
import sys
import os


def pytest_addoption(parser):
    """Add the --display option to enable window-creating tests."""
    parser.addoption(
        "--display", action="store_true", default=False,
        help="Run tests that create real display windows"
    )


def pytest_collection_modifyitems(config, items):
    """Skip window-creating tests unless --display is given."""
    if config.getoption("--display"):
        # Run all tests when --display is specified
        return
    
    skip_display = pytest.mark.skip(reason="Test creates a real window. Use --display to run.")
    
    for item in items:
        # Skip tests that are already marked to be skipped
        if item.get_closest_marker('skip'):
            continue
        
        # Skip tests in files that create windows
        if any(name in item.nodeid for name in [
            'test_display.py',
            'test_display_service.py',
            'test_integration.py'
        ]):
            item.add_marker(skip_display)
