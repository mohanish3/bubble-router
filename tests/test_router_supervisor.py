import importlib.util
import subprocess
import sys
import time
from collections import deque
from pathlib import Path


SUPERVISOR_PATH = Path.home() / "AI" / "services" / "router-supervisor.py"
SPEC = importlib.util.spec_from_file_location("router_supervisor", SUPERVISOR_PATH)
assert SPEC and SPEC.loader
supervisor = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(supervisor)


def test_restart_backoff_caps_and_crash_window_expires():
    crashes = deque()
    observed = []
    for tick in range(5):
        crashes, count, allowed, delay = supervisor.record_crash(crashes, float(tick))
        observed.append((count, allowed, delay))
    assert observed == [
        (1, True, 2.0),
        (2, True, 4.0),
        (3, True, 8.0),
        (4, True, 16.0),
        (5, False, 0.0),
    ]
    assert supervisor.backoff_delay(20) == 30.0
    crashes, count, allowed, delay = supervisor.record_crash(crashes, 605.0)
    assert (count, allowed, delay) == (1, True, 2.0)


def test_graceful_stop_signals_only_the_owned_child():
    owned = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        supervisor.stop_owned_child(owned, grace_seconds=5)
        assert owned.poll() is not None
        assert unrelated.poll() is None
    finally:
        if owned.poll() is None:
            owned.kill()
            owned.wait()
        if unrelated.poll() is None:
            unrelated.kill()
            unrelated.wait()
