import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

FIXTURE_CONFIG = Path(__file__).parent / "fixtures" / "model-router.test.json"
