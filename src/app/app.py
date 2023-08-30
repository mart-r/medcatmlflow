# app.py (Backend)

from flask import Flask

import logging

from .main.models import setup_db

from .main.utils import setup_logging

# blueprints
from .main.views import main_bp
from .modelmanage.views import models_bp
from .performance.views import perf_bp

# In docker, this is what we should have

app = Flask(__name__)

# setup logging
# Create a root logger
logger = logging.getLogger()
setup_logging(logger)

# setup the database
setup_db(app)

# setup blueprints

app.register_blueprint(main_bp)
app.register_blueprint(models_bp)
app.register_blueprint(perf_bp)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
