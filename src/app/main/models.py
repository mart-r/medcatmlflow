from flask_sqlalchemy import SQLAlchemy
from flask import Flask

import json

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


class TestDataset(db.Model):  # type: ignore
    id = db.Column(db.Integer, primary_key=True)
    category_name = db.Column(db.String(100), nullable=False)
    name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.String(250))
    file_path = db.Column(db.String(200), nullable=False)


class ModelDatasetPerformanceResult(db.Model):  # type: ignore
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
    examples = db.Column(db.JSON, nullable=True)

    def to_dict(self):
        result_dict = {}
        for column in self.__table__.columns:
            field_name = column.name
            field_value = getattr(self, field_name)
            if isinstance(field_value, dict):
                result_dict[field_name] = field_value
            elif field_value is not None:
                try:
                    result_dict[field_name] = json.loads(field_value)
                except json.JSONDecodeError:
                    result_dict[field_name] = field_value
                except TypeError:
                    result_dict[field_name] = field_value
            else:
                result_dict[field_name] = None
        return result_dict

    @classmethod
    def from_dict(cls, data_dict):
        return cls(
            model_id=data_dict.get('model_id'),
            dataset_id=data_dict.get('dataset_id'),
            fp=data_dict.get('fp'),
            fn=data_dict.get('fn'),
            tp=data_dict.get('tp'),
            prec=json.dumps(data_dict.get('prec') or {}),
            recall=json.dumps(data_dict.get('recall') or {}),
            f1=json.dumps(data_dict.get('f1') or {}),
            counts=json.dumps(data_dict.get('counts') or {}),
            examples=json.dumps(data_dict.get('examples') or {}),
        )
