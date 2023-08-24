# app.py (Backend)

from flask import Flask, render_template, request, send_file, jsonify
from flask import redirect, url_for

import os
import logging

from .mlflow_integration import attempt_upload, get_files_with_info
from .mlflow_integration import delete_mlflow_file, get_info
from .mlflow_integration import get_history, get_all_trees_with_links
from .mlflow_integration import has_experiment, create_mlflow_experiment
from .mlflow_integration import get_all_experiment_names, delete_experiment
from .mlflow_integration import get_all_experiments, recalc_model_metedata
from .datasets import get_test_datasets, upload_test_dataset
from .datasets import delete_test_dataset
from .models import setup_db

from .utils import setup_logging

from .envs import STORAGE_PATH

# In docker, this is what we should have

app = Flask(__name__)

# setup logging
# Create a root logger
logger = logging.getLogger()
setup_logging(logger)

# setup the database
setup_db(app)


# Endpoint to handle file uploads
@app.route("/upload", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        file = request.files['file']
        issues = attempt_upload(file.filename,
                                file.save,
                                request.form.get("experiment"),
                                request.form.get("model_name"),
                                request.form.get("overwrite") == "1")
        if issues:
            return issues
        return "File uploaded successfully!"
    else:
        experiment_names = get_all_experiment_names()
        return render_template("upload.html",
                               experiment_names=experiment_names)


# Endpoint to browse and download files along with custom information
@app.route("/files")
def browse_files():
    files_with_info = get_files_with_info()
    return render_template("browse_files.html", files=files_with_info)


@app.route("/delete_file", methods=["POST"])
def delete_file():
    filename = request.form.get("filename")
    if not filename:
        return jsonify({"error": "Filename not provided"}), 400

    delete_mlflow_file(filename)
    return jsonify({"message": "File deleted successfully"}), 200


# Endpoint for info for specific file
@app.route("/info/<filename>")
def show_file_info(filename):
    file_path = os.path.join(STORAGE_PATH, filename)
    return render_template("file_info.html", info=get_info(file_path))


# Endpoint for recalculation of metadata
@app.route("/recalculate_metadata/<filename>")
def recalculate_metadata(filename):
    file_path = os.path.join(STORAGE_PATH, filename)
    recalc_model_metedata(file_path)
    # Redirect to the file information page with the updated metadata
    return redirect(url_for("show_file_info", filename=filename))


# Endpoint for history of a specific file
@app.route("/history/<filename>")
def show_file_history(filename):
    file_path = os.path.join(STORAGE_PATH, filename)
    return render_template("history.html", history=get_history(file_path))


@app.route("/download/<filename>")
def download_file(filename):
    full_path = os.path.join(STORAGE_PATH, filename)
    return send_file(full_path, as_attachment=True)


@app.route("/")
def landing_page():
    return render_template("landing.html")


@app.route("/all_trees")
def all_trees():
    all_trees_with_links = get_all_trees_with_links()
    return render_template("all_trees.html",
                           all_trees_with_links=all_trees_with_links)


@app.route('/create_experiment', methods=['GET', 'POST'])
def create_experiment():
    if request.method == 'POST':
        short_name = request.form['short_name']
        description = request.form['description']

        if has_experiment(short_name):
            return f"Experiment already exists: {short_name}"
        create_mlflow_experiment(short_name, description)

        return redirect(url_for('manage_experiments'))

    return render_template('create_experiment.html')


@app.route('/manage_experiments', methods=['GET', 'POST'])
def manage_experiments():
    if request.method == 'POST':
        short_name_to_remove = request.form['short_name_to_remove']

        # Remove the experiment with the given short_name from the list.
        delete_experiment(short_name_to_remove)

        # Redirect to refresh the page after removing an experiment.
        return redirect(url_for('manage_experiments'))

    return render_template('manage_experiments.html',
                           experiments=get_all_experiments())


@app.route("/manage_datasets", methods=['GET', 'POST'])
def manage_datasets():
    if request.method == 'POST':
        ds_to_delete = request.form['dataset_name_to_delete']

        # Remove the dataset given
        delete_test_dataset(ds_to_delete)

        # Redirect to refresh the page.
        return redirect(url_for('manage_datasets'))
    return render_template("manage_datasets.html",
                           datasets=get_test_datasets())


@app.route("/upload_dataset", methods=["GET", "POST"])
def upload_dataset():
    if request.method == "POST":
        dataset_name = request.form.get("dataset_name")
        dataset_description = request.form.get("dataset_description")
        overwrite = request.form.get("overwrite") == "1"
        file = request.files['file']
        issues = upload_test_dataset(file.save, dataset_name,
                                     dataset_description, overwrite)
        if issues:
            return issues
        return redirect("/manage_datasets")

    return render_template("upload_dataset.html")


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
