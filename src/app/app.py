# app.py (Backend)

from flask import Flask

import logging

from .main.models import setup_db

from .main.utils import setup_logging

# blueprints
from .main.views import main_bp
from .modelmanage.views import models_bp
from .performance.views import perf_bp
from .modelmanage.mlflow_integration import setup_mlflow

# setup logging for root logger
logger = logging.getLogger()


def create_app() -> Flask:
    setup_logging(logger)
    app = Flask(__name__)
    app.debug = True

    # setup the database
    setup_db(app)

    # setup mlflow
    setup_mlflow()

    # setup blueprints

    app.register_blueprint(main_bp)
    app.register_blueprint(models_bp)
    app.register_blueprint(perf_bp)
    return app
