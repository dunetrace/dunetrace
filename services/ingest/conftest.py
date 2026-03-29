import pytest

# Enable auto asyncio mode so async fixtures and tests run without
# needing @pytest.mark.asyncio on every method.
def pytest_configure(config):
    config.addinivalue_line("markers", "asyncio: mark test as async")
