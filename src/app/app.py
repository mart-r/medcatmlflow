# app.py (Backend)

from flask import Flask, render_template, request, send_file, redirect, url_for

from mlflow.tracking import MlflowClient
from mlflow import MlflowException
import os
from typing import Optional

from medcat.cat import CAT

from .models import db, ModelData, VERSION_STR_LEN
from .utils import build_nodes, get_all_trees

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


def _get_experiment_id(experiment_name):
    exp = MLFLOW_CLIENT.get_experiment_by_name(experiment_name)
    if exp:
        return exp.experiment_id
    else:
        return MLFLOW_CLIENT.create_experiment(experiment_name)


# Endpoint to handle file uploads
@app.route("/upload", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        uploaded_file = request.files["file"]
        custom_info = request.form.get("custom_info")

        # Save the uploaded file to the desired location
        file_path = os.path.join(STORAGE_PATH, uploaded_file.filename)
        uploaded_file.save(file_path)

        experiment_id = _get_experiment_id(custom_info)

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


@app.route("/delete/<filename>")
def delete_file(filename):
    file_path = os.path.join(STORAGE_PATH, filename)

    # Remove the file from the filesystem
    if os.path.exists(file_path):
        os.remove(file_path)

    # Delete the corresponding MLflow data
    model_name = filename
    model = MLFLOW_CLIENT.get_registered_model(model_name)
    if model:
        arg = "name='{}'".format(model_name)
        model_versions = MLFLOW_CLIENT.search_model_versions(arg)
        for version in model_versions:
            run_id = version.run_id
            try:
                MLFLOW_CLIENT.delete_run(run_id)
            except MlflowException:
                pass
        MLFLOW_CLIENT.delete_registered_model(model_name)

    # Delete metadata from the database
    model_data = ModelData.query.filter_by(model_file_name=filename).first()
    if model_data:
        db.session.delete(model_data)
        db.session.commit()

    return redirect(url_for("browse_files"))


def _fix_performance(str_perf: str) -> dict:
    return eval(str_perf)


def get_info(file_path: str) -> dict:
    basename = os.path.basename(file_path).rsplit(".", 1)[0]
    model_version = basename[-VERSION_STR_LEN:]
    saved_meta = ModelData.query.filter_by(
        version=model_version).first()
    if saved_meta:
        cur_info = saved_meta.as_dict()
        if "performance" in cur_info:
            cur_info["performance"] = _fix_performance(cur_info["performance"])
    else:
        cur_info = {"ISSUES": "Metadata not saved",
                    "looked for": model_version,
                    "file path": file_path,
                    "basename": basename,
                    "SAVED META": str(saved_meta)}
    return cur_info


def _get_hist_link(version: str) -> str:
    saved: Optional[ModelData]
    saved = ModelData.query.filter_by(version=version).first()
    if saved:
        return f"/info/{saved.model_file_name}"
    return "N/A"


# Endpoint for info for specific file
@app.route("/info/<filename>")
def show_file_info(filename):
    file_path = os.path.join(STORAGE_PATH, filename)

    return render_template("file_info.html", info=get_info(file_path))


def get_history(file_path: str) -> dict:
    basename = os.path.basename(file_path)
    saved_meta = ModelData.query.filter_by(
        model_file_name=basename).first()
    if saved_meta:
        cur_info = saved_meta.as_dict()
        versions = cur_info["version_history"].split(",")
        history = [(version, _get_hist_link(version)) for version in versions
                   if version]
        return history
    else:
        return [("ISSUES", "Metadata not saved"),
                # ("looked for", model_version),
                ("file path", file_path),
                ("basename", basename),
                ("SAVED META", str(saved_meta))]


# Endpoint for history of a specific file
@app.route("/history/<filename>")
def show_file_history(filename):
    file_path = os.path.join(STORAGE_PATH, filename)
    return render_template("history.html", history=get_history(file_path))


@app.route("/download/<filename>")
def download_file(filename):
    return send_file(filename, as_attachment=True)


@app.route("/")
def landing_page():
    return render_template("landing.html")


@app.route("/all_trees")
def all_trees():
    # Query for registered models
    models = MLFLOW_CLIENT.list_registered_models()

    # Create a list of tuples containing tree representations and model links
    data = {}
    for model in models:
        model_name = model.name
        basename = os.path.basename(model_name)
        saved_meta = ModelData.query.filter_by(
            model_file_name=basename).first()
        if saved_meta:
            version = saved_meta.version
            versions = saved_meta.version_history.split(",")
            data[version] = versions
    nodes = build_nodes(data).values()
    # Pass the _get_hist_link function as the model_link_func parameter
    all_trees_with_links = get_all_trees(nodes, _get_hist_link)
    return render_template("all_trees.html",
                           all_trees_with_links=all_trees_with_links)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
