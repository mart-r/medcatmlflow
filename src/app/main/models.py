from flask_sqlalchemy import SQLAlchemy
from flask import Flask

from .envs import MEDCATMLFLOW_DB_URI

db = SQLAlchemy()


def setup_db(app: Flask):
    # Configure the database URI (for SQLite)
    app.config["SQLALCHEMY_DATABASE_URI"] = MEDCATMLFLOW_DB_URI
    # init DB
    db.init_app(app)
    # Create the tables
    with app.app_context():
        db.create_all()


class TestDataset(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    category_name = db.Column(db.String(100), nullable=False)
    name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.String(250))
    file_path = db.Column(db.String(200), nullable=False)


class ModelDatasetPerformanceResult(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    model_id = db.Column(db.String(100), nullable=False)
    dataset_id = db.Column(db.Integer, nullable=False)
    fp = db.Column(db.Integer, nullable=True)
    fn = db.Column(db.Integer, nullable=True)
    tp = db.Column(db.Integer, nullable=True)
    prec = db.Column(db.JSON, nullable=True)
    recall = db.Column(db.JSON, nullable=True)
    f1 = db.Column(db.JSON, nullable=True)
    counts = db.Column(db.JSON, nullable=True)
    exmaples = db.Column(db.JSON, nullable=True)
