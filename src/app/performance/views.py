from flask import Blueprint, render_template, request
from flask import redirect, url_for

import logging

import os

from ..modelmanage.mlflow_integration import (
    get_all_experiment_names,
    get_all_model_metadata,
    get_model_from_id,
)
from ..medcat_linkage.medcat_integration import get_performance
from .datasets import get_test_datasets, upload_test_dataset
from .datasets import delete_test_dataset
from .imaging import get_buffers

from ..main.envs import STORAGE_PATH


perf_bp = Blueprint("performance", __name__)


logger = logging.getLogger(__name__)


@perf_bp.route("/manage_datasets", methods=["GET", "POST"])
def manage_datasets():
    if request.method == "POST":
        ds_to_delete = request.form["dataset_name_to_delete"]

        # Remove the dataset given
        delete_test_dataset(ds_to_delete)

        # Redirect to refresh the page.
        return redirect(url_for("performance.manage_datasets"))
    return render_template(
        "performance/manage_datasets.html", datasets=get_test_datasets()
    )


@perf_bp.route("/upload_dataset", methods=["GET", "POST"])
def upload_dataset():
    if request.method == "POST":
        dataset_name = request.form.get("dataset_name")
        dataset_description = request.form.get("dataset_description")
        category = request.form.get("category")
        overwrite = request.form.get("overwrite") == "1"
        file = request.files["file"]
        issues = upload_test_dataset(
            file.save, category, dataset_name, dataset_description, overwrite
        )
        if issues:
            return issues
        return redirect("/manage_datasets")

    return render_template("performance/upload_dataset.html",
                           categories=get_all_experiment_names())


@perf_bp.route("/show_performance", methods=["GET"])
def show_performance():
    available_models = [md.as_dict() for md in get_all_model_metadata()]
    available_datasets = get_test_datasets()
    available_categories = get_all_experiment_names()
    return render_template(
        "performance/show_performance.html",
        available_models=available_models,
        available_datasets=available_datasets,
        available_categories=available_categories,
    )


@perf_bp.route("/calculate_performance", methods=["POST"])
def calculate_performance():
    selected_model_ids = request.form.getlist("selected_models")
    selected_dataset_ids = request.form.getlist("selected_datasets")

    models = []
    for model_id in selected_model_ids:
        model = get_model_from_id(model_id)
        file_path = os.path.join(STORAGE_PATH, model.model_file_name)
        models.append((model.name, file_path))

    logger.info("Getting performance of %d models over %d datasets",
                len(models), len(selected_dataset_ids))
    performance_results = get_performance(models, selected_dataset_ids)

    graph_buffers = get_buffers(performance_results)

    return render_template(
        "performance/performance_result.html",
        performance_results=performance_results,
        graph_paths=graph_buffers,
    )
