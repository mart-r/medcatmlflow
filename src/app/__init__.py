import logging

from .app import app

logger = logging.getLogger(__package__)

logger.debug("Loaded Flask app: %s", app)
