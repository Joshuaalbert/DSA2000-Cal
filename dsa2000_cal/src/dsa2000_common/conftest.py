import pylab as plt
import pytest


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_teardown(item, nextitem):
    """Hook to run plt.close('all') after each test."""
    yield  # Run the actual test teardown
    plt.close('all')  # Close all plots after each test
