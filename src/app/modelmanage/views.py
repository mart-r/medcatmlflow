from flask import Blueprint, render_template, request, send_file, jsonify
from flask import redirect, url_for

import os

from .mlflow_integration import (
    attempt_upload, get_all_models_dict, delete_mlflow_file,
    get_history, get_all_experiment_names, recalc_model_metedata,
    get_all_trees_with_links, has_experiment, create_mlflow_experiment,
    delete_experiment, get_all_experiments, get_model_from_id,
    get_mlflow_from_id
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
                                request.form.get("model_description"),
                                request.form.get("overwrite") == "1")
        if issues:
            return issues
        return redirect(url_for("files"))
    else:
        experiment_names = get_all_experiment_names()
        return render_template("modelmanage/upload.html",
                               experiment_names=experiment_names)


@models_bp.route("/files")
def browse_files():
    files_with_info = get_all_models_dict()
    return render_template("modelmanage/browse_files.html",
                           files=files_with_info)


@models_bp.route("/delete_file", methods=["POST"])
def delete_file():
    filename = request.form.get("file_id")
    if not filename:
        return jsonify({"error": "File ID not provided"}), 400

    delete_mlflow_file(filename)
    return redirect(url_for("files"))


@models_bp.route("/info/<file_id>")
def show_file_info(file_id):
    model = get_model_from_id(file_id)
    return render_template("modelmanage/file_info.html",
                           info=model.as_dict())


@models_bp.route("/recalculate_metadata/<file_id>")
def recalculate_metadata(file_id):
    model = get_mlflow_from_id(file_id)
    recalc_model_metedata(model)
    return redirect(url_for("modelmanage.show_file_info", file_id=file_id))


@models_bp.route("/history/<file_id>")
def show_file_history(file_id):
    meta = get_model_from_id(file_id)
    return render_template("modelmanage/history.html",
                           history=get_history(meta), name=meta.name)


@models_bp.route("/download/<file_id>")
def download_file(file_id):
    model = get_model_from_id(file_id)
    full_path = os.path.join(STORAGE_PATH, model.model_file_name)
    return send_file(full_path, as_attachment=True)


@models_bp.route("/all_trees")
def all_trees():
    all_trees_with_links = get_all_trees_with_links()
    all_trees_with_links = sorted(all_trees_with_links, key=lambda li: li[1])
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
