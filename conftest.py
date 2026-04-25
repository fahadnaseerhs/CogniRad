import sys
import pytest
import asyncio

# Make pytest-asyncio use auto mode so @pytest.mark.asyncio is not required
def pytest_configure(config):
    config.addinivalue_line(
        "markers", "asyncio: mark test as async"
    )

# Fix for newer pytest-asyncio: set default loop scope
import pytest_asyncio
