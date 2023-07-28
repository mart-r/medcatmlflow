import os
from typing import Optional, Callable
import shutil
import re

import logging

from mlflow import MlflowClient, MlflowException
from mlflow.entities import Experiment
from mlflow.entities.model_registry import RegisteredModel

from .utils import ModelMetaData
from .utils import build_nodes, get_all_trees
from .medcat_integration import create_meta

from .envs import STORAGE_PATH

# Configure MLflow
from .envs import DB_URI

MLFLOW_CLIENT = MlflowClient(tracking_uri=DB_URI)

logger = logging.getLogger(__name__)


def has_experiment(name: str) -> bool:
    return bool(MLFLOW_CLIENT.get_experiment_by_name(name))


def create_mlflow_experiment(name: str, description: str) -> str:
    exp = MLFLOW_CLIENT.create_experiment(name,
                                          tags={'description': description})
    return exp


def _get_exp_run_ids(exp: Experiment) -> list[str]:
    return [run.info.run_id
            for run in MLFLOW_CLIENT.search_runs([exp.experiment_id, ])]


def get_all_experiment_names() -> list[str]:
    return [el[0] for el in get_all_experiments()]


def get_all_experiments() -> list[tuple[str, str]]:
    return [(exp.name, exp.tags.get('description',
                                    "Old experiment with no description"),
             len(_get_exp_run_ids(exp)))
            for exp in MLFLOW_CLIENT.search_experiments()]


def delete_experiment(name: str) -> None:
    exp = MLFLOW_CLIENT.get_experiment_by_name(name)
    MLFLOW_CLIENT.delete_experiment(exp.experiment_id)


def _get_experiment_id(experiment_name: str) -> str:
    return MLFLOW_CLIENT.get_experiment_by_name(experiment_name).experiment_id


def attempt_upload(file_name: str, file_saver: Callable[[str], None],
                   experiment_name: str,
                   model_description: str,
                   overwrite: bool):
    if not has_experiment(experiment_name):
        return f'Experiment not found: {experiment_name}'

    # Save the uploaded file to the desired location
    file_path = os.path.join(STORAGE_PATH, file_name)

    if os.path.exists(file_path) and not overwrite:
        return f"File already exists: {file_name}"

    experiment_id = _get_experiment_id(experiment_name)

    # save
    file_saver(file_path)

    run = MLFLOW_CLIENT.create_run(experiment_id=experiment_id)
    run_id = run.info.run_id
    MLFLOW_CLIENT.log_artifact(run_id, file_path)
    MLFLOW_CLIENT.log_param(run_id, "model_description", model_description)

    model_filename = file_name

    try:
        # db stuff
        meta = create_meta(file_path, model_filename, category=experiment_name)
        MLFLOW_CLIENT.create_registered_model(model_filename,
                                              tags=meta.as_dict(),
                                              description=model_description)
        # Get the artifact URI for the logged file
        artifact_uri = "runs:/{}/{}".format(run_id, file_path)
        # Create a model version associated with the registered model and file
        MLFLOW_CLIENT.create_model_version(model_filename, artifact_uri)
    except Exception as e:
        logger.error("Unable to store model %s", file_name,
                     exc_info=e)
        # do cleanup on disk
        os.remove(file_path)
        if file_path.endswith('.zip'):
            folder_path = file_path[:-4]
            if os.path.exists(folder_path):
                shutil.rmtree(folder_path)
        # delete MLFLOW stuff
        MLFLOW_CLIENT.delete_run(run_id)
        return f"Unable to store model {file_name}: {e}"


RUN_ID_PATTERN = re.compile(re.escape("runs:/") +
                            "(.*?)" +
                            re.escape("/") + ".*")


def _get_run_id(model: RegisteredModel,
                run_id_pattern: re.Pattern = RUN_ID_PATTERN
                ) -> str:
    ver = MLFLOW_CLIENT.search_model_versions(f"name='{model.name}'")[0]
    source = ver.source
    # e.g: runs:/5a5dad1636bf4d87bba373e10dcd99e8//app/models/smth.zip
    return run_id_pattern.match(source).group(1)


def get_files_with_info() -> list[dict]:
    # Query for registered models
    models = MLFLOW_CLIENT.search_registered_models()

    # Print the list of models and their information
    files_with_info = []
    for model in models:
        run_id = _get_run_id(model)
        cur_info = {
            "name": model.name,
            "version": model.tags['version'],
            "description": model.description,
            "experiment": model.tags['category'],
            "run_id": run_id,
        }
        files_with_info.append(cur_info)
    return files_with_info


def delete_mlflow_file(filename: str) -> None:

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


def get_info(file_path: str) -> dict:
    basename = os.path.basename(file_path)
    model = MLFLOW_CLIENT.get_registered_model(basename)
    if model:
        metadata = get_meta_model(model, os.path.dirname(file_path))
        cur_info = metadata.as_dict()
        if ('performance' in cur_info
                and not isinstance(cur_info['performance'], dict)):
            # TODO - make the below better
            cur_info["performance"] = eval(cur_info["performance"])
        cur_info['Internal ID'] = _get_run_id(model)
    else:
        cur_info = {"ISSUES": "Metadata not saved",
                    "file path": file_path,
                    "basename": basename
                    }
    return cur_info


def _get_file_name(version: str, _tag_key: str = 'version') -> str:
    all_models = MLFLOW_CLIENT.search_registered_models()
    model: Optional[RegisteredModel] = None
    for found_model in all_models:
        if (_tag_key in found_model.tags and
                found_model.tags[_tag_key] == version):
            model = found_model
            break
    if model:
        return model.tags['model_file_name']
    return None


def _get_hist_link(version: str, _tag_key: str = 'version') -> str:
    file_name = _get_file_name(version, _tag_key)
    if file_name:
        return f"/info/{file_name}"
    return file_name  # None


def _update_model_meta(model: RegisteredModel, meta: ModelMetaData) -> None:
    for new_key, new_value in meta.as_dict().items():
        if new_key in model.tags and model.tags[new_key] == new_value:
            continue
        try:
            MLFLOW_CLIENT.set_registered_model_tag(model.name,
                                                   new_key,
                                                   new_value)
        except MlflowException as e:
            logger.warning("Issue setting value of '%s' for model '%s':",
                           new_key, model.name, exc_info=e)
            MLFLOW_CLIENT.set_registered_model_tag(model.name,
                                                   new_key,
                                                   "N/A")
    model.tags.update(meta.as_dict())


def get_meta_model(model: RegisteredModel) -> ModelMetaData:
    try:
        meta = ModelMetaData.from_mlflow_model(model)
    except KeyError:  # old model data with not all the keys
        logger.warning("Recalculating meta - not everything was saved on disk")
        file_path = os.path.join(STORAGE_PATH, model.tags['model_file_name'])
        meta = create_meta(file_path, model.name,
                           # if no category saved, we can't re-create
                           category=model.tags['category'])
        _update_model_meta(model, meta)
    return meta


def get_history(file_path: str) -> list:
    basename = os.path.basename(file_path)
    model = MLFLOW_CLIENT.get_registered_model(basename)
    if model:
        meta = get_meta_model(model)
        cur_info = meta.as_dict()
        versions = cur_info["version_history"].split(",")
        history = []
        for version in versions:
            if not version:
                continue
            cur_name = _get_file_name(version)
            history.append((version, cur_name))
        return history
    else:
        return [("ISSUES", "Metadata not saved"),
                # ("looked for", model_version),
                ("file path", file_path),
                ("basename", basename),
                ("SAVED META", str(model))]


def get_all_trees_with_links():
    # Query for registered models
    models = MLFLOW_CLIENT.search_registered_models()

    # Create a list of tuples containing tree representations and model links
    data = {}
    for model in models:
        saved_meta = get_meta_model(model)
        if saved_meta:
            version = saved_meta.version
            # remove empty versions
            versions = [ver for ver in saved_meta.version_history.split(",")
                        if ver]
            data[version] = versions
    nodes = build_nodes(data).values()
    # Pass the _get_hist_link function as the model_link_func parameter
    return get_all_trees(nodes, _get_hist_link)
