# app.py (Backend)

from flask import Flask, render_template, request, send_file

import mlflow
from mlflow.tracking import MlflowClient
import os

from medcat.cat import CAT

from .models import db, ModelData, VERSION_STR_LEN

app = Flask(__name__)

# Configure MLflow
DB_URI = os.environ.get("MLFLOW_DB_URI")
MLFLOW_CLIENT = MlflowClient(tracking_uri=DB_URI)

STORAGE_PATH = os.environ.get("MODEL_STORAGE_PATH")

SQL_DB_URI = os.environ.get("METADATA_DATABASE_URI")

app.config['SQLALCHEMY_DATABASE_URI'] = SQL_DB_URI
db.init_app(app)
with app.app_context():
    db.create_all()


def _get_meta(file_path: str, model_name: str) -> ModelData:
    print('Loading CAT to get model info')
    cat = CAT.load_model_pack(file_path)
    version = cat.config.version.id
    version_history = ",".join(cat.config.version.history)
    performance = str(cat.config.version.performance)
    print('Found the model info/data')
    return ModelData(model_file_name=model_name, version=version,
                     version_history=version_history, performance=performance)


# Endpoint to handle file uploads
@app.route("/upload", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        uploaded_file = request.files["file"]
        custom_info = request.form.get("custom_info")

        # Save the uploaded file to the desired location
        file_path = os.path.join(STORAGE_PATH, uploaded_file.filename)
        uploaded_file.save(file_path)

        experiment_name = custom_info
        # Check if the experiment exists, create it if it doesn't
        exp = mlflow.get_experiment_by_name(experiment_name)
        if exp:
            experiment_id = exp.experiment_id
        else:
            experiment_id = mlflow.create_experiment(experiment_name)

        run = MLFLOW_CLIENT.create_run(experiment_id=experiment_id)
        run_id = run.info.run_id
        MLFLOW_CLIENT.log_artifact(run_id, file_path)
        MLFLOW_CLIENT.log_param(run_id, "custom_info", custom_info)

        model_name = uploaded_file.filename
        _ = MLFLOW_CLIENT.create_registered_model(model_name)

        # Get the artifact URI for the logged file
        artifact_uri = "runs:/{}/{}".format(run_id, file_path)

        # Create a model version associated with the registered model and file
        MLFLOW_CLIENT.create_model_version(model_name, artifact_uri)
        # db stuff
        meta = _get_meta(file_path, model_name)
        db.session.add(meta)
        db.session.commit()
        return "File uploaded successfully!"
        # return perform_post()
    return render_template("upload.html")


# Endpoint to browse and download files along with custom information
@app.route("/files")
def browse_files():
    # Query for registered models
    models = MLFLOW_CLIENT.list_registered_models()

    # Print the list of models and their information
    files_with_info = []
    for model in models:
        cur_info = {
            "name": model.name,
            "version": model.latest_versions[0].version,
            "creation": model.creation_timestamp,
            "update": model.last_updated_timestamp,
            "description": model.description,
            "tags": model.tags,
        }
        files_with_info.append(cur_info)

    return render_template("browse_files.html", files=files_with_info)


def get_info(file_path: str) -> dict:
    basename = os.path.basename(file_path).rsplit(".", 1)[0]
    model_version = basename[-VERSION_STR_LEN:]
    saved_meta = ModelData.query.filter_by(
        version=model_version).first()
    if saved_meta:
        cur_info = saved_meta.as_dict()
    else:
        cur_info = {"ISSUES": "Metadata not saved",
                    "looked for": model_version,
                    "file path": file_path,
                    "basename": basename,
                    "SAVED META": str(saved_meta)}
    return cur_info


# Endpoint to download a specific file
@app.route("/info/<filename>")
def show_file_info(filename):
    file_path = os.path.join(STORAGE_PATH, filename)

    return render_template("file_info.html", info=get_info(file_path))


@app.route("/download/<filename>")
def download_file(filename):
    return send_file(filename, as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
