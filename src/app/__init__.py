import logging

# for use in starting
from .app import create_app

logger = logging.getLogger(__package__)

logger.debug("Ready to load Flask app with: %s", create_app)
