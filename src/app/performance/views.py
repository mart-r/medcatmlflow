from flask import Blueprint, render_template, request
from flask import redirect, url_for

import os

from ..modelmanage.mlflow_integration import (
    get_all_experiment_names, get_all_model_metadata,
    get_model_descr_from_file
)
from ..medcat_linkage.medcat_integration import get_performance
from .datasets import get_test_datasets, upload_test_dataset
from .datasets import delete_test_dataset

from ..main.envs import STORAGE_PATH


perf_bp = Blueprint('performance', __name__)


@perf_bp.route("/manage_datasets", methods=['GET', 'POST'])
def manage_datasets():
    if request.method == 'POST':
        ds_to_delete = request.form['dataset_name_to_delete']

        # Remove the dataset given
        delete_test_dataset(ds_to_delete)

        # Redirect to refresh the page.
        return redirect(url_for('performance.manage_datasets'))
    return render_template("performance/manage_datasets.html",
                           datasets=get_test_datasets())


@perf_bp.route("/upload_dataset", methods=["GET", "POST"])
def upload_dataset():
    if request.method == "POST":
        dataset_name = request.form.get("dataset_name")
        dataset_description = request.form.get("dataset_description")
        category = request.form.get("category")
        overwrite = request.form.get("overwrite") == "1"
        file = request.files['file']
        issues = upload_test_dataset(file.save, category, dataset_name,
                                     dataset_description, overwrite)
        if issues:
            return issues
        return redirect("/manage_datasets")

    return render_template("performance/upload_dataset.html",
                           categories=get_all_experiment_names())


@perf_bp.route('/show_performance', methods=['GET'])
def show_performance():
    available_models = [md.as_dict() for md in get_all_model_metadata()]
    available_datasets = get_test_datasets()
    available_categories = get_all_experiment_names()
    return render_template("performance/show_performance.html",
                           available_models=available_models,
                           available_datasets=available_datasets,
                           available_categories=available_categories)


@perf_bp.route('/calculate_performance', methods=['POST'])
def calculate_performance():
    selected_model_files = request.form.getlist('selected_models')
    selected_dataset_files = request.form.getlist('selected_datasets')

    models = [
        (get_model_descr_from_file(model_file),
         os.path.join(STORAGE_PATH, model_file))
        for model_file in selected_model_files
    ]

    performance_results = get_performance(models, selected_dataset_files)

    return render_template("performance/performance_result.html",
                           performance_results=performance_results)
