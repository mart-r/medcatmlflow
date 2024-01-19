from flask import Blueprint, render_template, request
from flask import redirect, url_for

import logging

from ..modelmanage.mlflow_integration import (
    get_all_experiment_names,
    get_all_model_metadata,
    get_model_from_id,
    get_model_cui_counts,
    get_model_total_count,
)
from .datasets import get_test_datasets, upload_test_dataset
from .datasets import delete_test_dataset, find_or_load_performance
from .imaging import get_buffers, get_buffer_for_cui_count_train


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
    force_recalc = request.form.get("recalc_performance")
    if not selected_model_ids or not selected_dataset_ids:
        # TODO - add message about missing stuff
        return show_performance()

    models = [get_model_from_id(model_id) for model_id in selected_model_ids]

    logger.info("Getting performance of %d models over %d datasets",
                len(models), len(selected_dataset_ids))
    performance_results = find_or_load_performance(models,
                                                   selected_dataset_ids,
                                                   force_recalc=force_recalc)

    graph_buffers = get_buffers(performance_results)

    return render_template(
        "performance/performance_result.html",
        performance_results=performance_results,
        graph_paths=graph_buffers,
    )


@perf_bp.route('/check_cuis', methods=['GET', 'POST'])
def check_cuis():
    if request.method == 'POST':
        selected_models = request.form.getlist('selected_models')
        selected_cuis = request.form.get('cuis').split(',')
        selected_cuis = [cui.strip() for cui in selected_cuis]
        if 'cuis_file' in request.files:
            cuis_file = request.files['cuis_file']
        else:
            cuis_file = None
        if cuis_file:
            # Save the uploaded file to the 'uploads' folder
            file_contents = cuis_file.read().decode('utf-8')
            if "," in file_contents:
                file_cuis = file_contents.split(",")
            else:
                file_cuis = file_contents.split("\n")
            selected_cuis += [cui.strip() for cui in file_cuis if cui.strip()]
        model_cuis_counts = get_model_cui_counts(selected_models,
                                                 selected_cuis)
        total_counts = [get_model_total_count(model)
                        for model in selected_models]
        total_counts = [tc for tc in total_counts if tc is not None]
        buffer = get_buffer_for_cui_count_train(model_cuis_counts,
                                                total_counts)
        return render_template('performance/evaluate_cuis.html',
                               graphs={"Train counts": buffer})

    available_categories = get_all_experiment_names()

    available_models = [md.as_dict() for md in get_all_model_metadata()]

    return render_template('performance/check_cuis.html',
                           available_categories=available_categories,
                           available_models=available_models)
