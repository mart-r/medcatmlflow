# import os

from flask_sqlalchemy import SQLAlchemy
# from flask import current_app

VERSION_STR_LEN = 16  # e.g fd194216cd19efa1

# with current_app.app_context():
db = SQLAlchemy()


class ModelData(db.Model):
    version = db.Column(db.String(VERSION_STR_LEN), primary_key=True)
    version_history = db.Column(db.String(VERSION_STR_LEN * 100),
                                nullable=False)
    model_file_name = db.Column(db.String(80), nullable=False)
    performance = db.Column(db.String(2000), nullable=False)

    def __repr__(self):
        return f"<ModelData for {self.version}>"

    def as_dict(self) -> dict:
        return {
            "version": self.version,
            "version_history": self.version_history,
            "model_file_name": self.model_file_name,
            "performance": self.performance
        }
