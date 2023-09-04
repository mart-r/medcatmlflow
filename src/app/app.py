# app.py (Backend)

from flask import Flask

import logging

from .main.models import setup_db

from .main.utils import setup_logging

# blueprints
from .main.views import main_bp
from .modelmanage.views import models_bp

app = Flask(__name__)

# setup logging for root logger
logger = logging.getLogger()
setup_logging(logger)

# setup the database
setup_db(app)

# setup blueprints

app.register_blueprint(main_bp)
app.register_blueprint(models_bp)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
