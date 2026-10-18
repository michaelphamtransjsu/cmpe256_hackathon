from pathlib import Path

import pytest


def pytest_addoption(parser):
    parser.addoption("--run-real-data", action="store_true", help="Run optional tests against local source documents/CSV datasets")
    parser.addoption("--real-data-dir", type=Path, default=Path("data"), help="Extracted data/ root for optional source tests")


@pytest.fixture
def real_data_dir(request):
    if not request.config.getoption("--run-real-data"):
        pytest.skip("Requires --run-real-data and local source data")
    return request.config.getoption("--real-data-dir")
