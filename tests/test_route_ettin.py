import os
from pathlib import Path

import pytest

from model_router.mac.route_ettin import EttinRouter

MODEL = Path.home() / 'AI/models/route-ettin-150m'


@pytest.mark.skipif(not MODEL.exists(), reason='route-ettin checkpoint not installed')
def test_real_checkpoint_routes_and_keeps_long_input():
    router = EttinRouter(MODEL, max_tokens=512)
    out = router.route_request('hey, thanks')
    assert set(out['scores']) == {'general', 'swift', 'reasoner'} and out['route'] in out['scores']
    assert abs(sum(out['scores'].values()) - 1) < 1e-3
    ids = router._ids('word ' * 5000)
    assert len(ids) == 512 and ids[0] == router._tok.cls_token_id and ids[-1] == router._tok.sep_token_id
    assert router.route_request('hey, thanks') == out  # cached
