import logging
import os

from .app import create_app

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

app = create_app(warm=os.getenv("BUBBLE_ROUTER_NO_WARM", "").lower() not in {"1", "true"})
