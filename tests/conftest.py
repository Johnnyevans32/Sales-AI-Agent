"""Configure pytest for the project."""


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "asyncio: mark test as async",
    )
