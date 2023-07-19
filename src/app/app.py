# app.py (Backend)

from flask import Flask, render_template, request, send_file

import mlflow
from mlflow.tracking import MlflowClient
import os

app = Flask(__name__)

# Configure MLflow
DB_URI = os.environ.get("MLFLOW_DB_URI")
MLFLOW_CLIENT = MlflowClient(tracking_uri=DB_URI)
# mlflow.set_tracking_uri(DB_URI)

STORAGE_PATH = os.environ.get("MODEL_STORAGE_PATH")


# def perform_post():


# Endpoint to handle file uploads
@app.route("/upload", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        uploaded_file = request.files["file"]
        custom_info = request.form.get("custom_info")

        # Save the uploaded file to the desired location
        file_path = os.path.join(STORAGE_PATH, uploaded_file.filename)
        uploaded_file.save(file_path)

        # Log the file and custom information in mlflow
        # with MLFLOW_CLIENT.get_run() as run:
        #     mlflow.log_artifact(file_path)
        #     mlflow.log_param("custom_info", custom_info)
        experiment_name = custom_info
        # Check if the experiment exists, create it if it doesn't
        exp = mlflow.get_experiment_by_name(experiment_name)
        if exp:
            experiment_id = exp.experiment_id
        else:
            experiment_id = mlflow.create_experiment(experiment_name)

        # from mlflow.entities import Run
        # r: Run = MLFLOW_CLIENT.get_run()
        # r.data
        # MLFLOW_CLIENT.model
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
        # print(f"Name: {model.name}")
        # print(f"Version: {model.latest_versions[0].version}")
        # print(f"Creation Time: {model.creation_timestamp}")
        # print(f"Last Updated Time: {model.last_updated_timestamp}")
        # print(f"Description: {model.description}")
        # print("-----")
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
    # TODO - load data
    return {"file_name": os.path.basename(file_path)}


# Endpoint to download a specific file
@app.route("/info/<filename>")
def show_file_info(filename):
    # Retrieve the file from MLflow
    # You can use mlflow.download_artifacts to get the file path

    # Mock file path for demonstration purposes
    file_path = os.path.join(STORAGE_PATH, filename)

    return render_template("file_info.html", info=get_info(file_path))


@app.route("/download/<filename>")
def download_file(filename):
    return send_file(filename, as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
