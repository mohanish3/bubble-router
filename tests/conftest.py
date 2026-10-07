import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

FIXTURE_CONFIG = Path(__file__).parent / "fixtures" / "model-router.test.json"


import pytest


@pytest.fixture(autouse=True)
def _no_admission_wait(monkeypatch):
    """Rejection tests expect an immediate answer; the wait has its own test."""
    from model_router.mac import resources

    monkeypatch.setattr(resources, "ADMISSION_WAIT_SECONDS", 0.0)
