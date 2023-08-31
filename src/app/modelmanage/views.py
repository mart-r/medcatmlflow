from flask import Blueprint, render_template, request, send_file, jsonify
from flask import redirect, url_for

import os

from .mlflow_integration import (
    attempt_upload, get_files_with_info, delete_mlflow_file,
    get_info, get_history, get_all_experiment_names, recalc_model_metedata,
    get_all_trees_with_links, has_experiment, create_mlflow_experiment,
    delete_experiment, get_all_experiments
)

from ..main.envs import STORAGE_PATH


models_bp = Blueprint('modelmanage', __name__)


# Endpoint to handle file uploads
@models_bp.route("/upload", methods=["GET", "POST"])
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
        return render_template("modelmanage/upload.html",
                               experiment_names=experiment_names)


# Endpoint to browse and download files along with custom information
@models_bp.route("/files")
def browse_files():
    files_with_info = get_files_with_info()
    return render_template("modelmanage/browse_files.html",
                           files=files_with_info)


@models_bp.route("/delete_file", methods=["POST"])
def delete_file():
    filename = request.form.get("filename")
    if not filename:
        return jsonify({"error": "Filename not provided"}), 400

    delete_mlflow_file(filename)
    return jsonify({"message": "File deleted successfully"}), 200


# Endpoint for info for specific file
@models_bp.route("/info/<filename>")
def show_file_info(filename):
    file_path = os.path.join(STORAGE_PATH, filename)
    return render_template("modelmanage/file_info.html",
                           info=get_info(file_path))


# Endpoint for recalculation of metadata
@models_bp.route("/recalculate_metadata/<filename>")
def recalculate_metadata(filename):
    file_path = os.path.join(STORAGE_PATH, filename)
    recalc_model_metedata(file_path)
    # Redirect to the file information page with the updated metadata
    return redirect(url_for("modelmanage.show_file_info", filename=filename))


# Endpoint for history of a specific file
@models_bp.route("/history/<filename>")
def show_file_history(filename):
    file_path = os.path.join(STORAGE_PATH, filename)
    return render_template("modelmanage/history.html",
                           history=get_history(file_path))


@models_bp.route("/download/<filename>")
def download_file(filename):
    full_path = os.path.join(STORAGE_PATH, filename)
    return send_file(full_path, as_attachment=True)


@models_bp.route("/all_trees")
def all_trees():
    all_trees_with_links = get_all_trees_with_links()
    return render_template("modelmanage/all_trees.html",
                           all_trees_with_links=all_trees_with_links)


@models_bp.route('/create_experiment', methods=['GET', 'POST'])
def create_experiment():
    if request.method == 'POST':
        short_name = request.form['short_name']
        description = request.form['description']

        if has_experiment(short_name):
            return f"Experiment already exists: {short_name}"
        create_mlflow_experiment(short_name, description)

        return redirect(url_for('modelmanage.manage_experiments'))

    return render_template('modelmanage/create_experiment.html')


@models_bp.route('/manage_experiments', methods=['GET', 'POST'])
def manage_experiments():
    if request.method == 'POST':
        short_name_to_remove = request.form['short_name_to_remove']

        # Remove the experiment with the given short_name from the list.
        delete_experiment(short_name_to_remove)

        # Redirect to refresh the page after removing an experiment.
        return redirect(url_for('modelmanage.manage_experiments'))

    return render_template('modelmanage/manage_experiments.html',
                           experiments=get_all_experiments())
