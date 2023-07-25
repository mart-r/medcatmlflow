import os
from typing import Optional

import logging

from mlflow import MlflowClient, MlflowException
from mlflow.entities.model_registry import RegisteredModel

from werkzeug.datastructures import FileStorage  # used in Flask

from .utils import DuplciateUploadException, ModelMetaData
from .utils import build_nodes, get_all_trees
from .medcat_integration import create_meta

# Configure MLflow
DB_URI = os.environ.get("MLFLOW_DB_URI")
MLFLOW_CLIENT = MlflowClient(tracking_uri=DB_URI)

logger = logging.getLogger(__name__)


def _create_new_experiment(file_path: str,
                           model_description: str,
                           overwrite: bool) -> str:
    """Validate upload and get experiment ID if valid upload.
    """
    if os.path.exists(file_path) and not overwrite:
        raise DuplciateUploadException("File already exists - tick the "
                                       "overwrite tick box if you wish "
                                       "to overwrite")

    exp = MLFLOW_CLIENT.get_experiment_by_name(model_description)
    if exp and not overwrite:
        raise DuplciateUploadException("Model by this description exists - "
                                       "choose another description / name")
    elif exp:
        return exp.experiment_id
    else:
        return MLFLOW_CLIENT.create_experiment(model_description)


def attempt_upload(uploaded_file: FileStorage,
                   model_description: str,
                   overwrite: bool, storage_path: str):

    # Save the uploaded file to the desired location
    file_path = os.path.join(storage_path, uploaded_file.filename)
    try:
        experiment_id = _create_new_experiment(file_path,
                                               model_description,
                                               overwrite)
    except DuplciateUploadException as e:
        return e.msg
    uploaded_file.save(file_path)

    run = MLFLOW_CLIENT.create_run(experiment_id=experiment_id)
    run_id = run.info.run_id
    MLFLOW_CLIENT.log_artifact(run_id, file_path)
    MLFLOW_CLIENT.log_param(run_id, "model_description", model_description)

    model_filename = uploaded_file.filename

    # db stuff
    meta = create_meta(file_path, model_filename)
    MLFLOW_CLIENT.create_registered_model(model_filename,
                                          tags=meta.as_dict(),
                                          description=model_description)
    # Get the artifact URI for the logged file
    artifact_uri = "runs:/{}/{}".format(run_id, file_path)
    # Create a model version associated with the registered model and file
    MLFLOW_CLIENT.create_model_version(model_filename, artifact_uri)


def get_files_with_info() -> list[dict]:
    # Query for registered models
    models = MLFLOW_CLIENT.search_registered_models()

    # Print the list of models and their information
    files_with_info = []
    for model in models:
        cur_info = {
            "name": model.name,
            "version": model.tags['version'],
            "description": model.description,
        }
        files_with_info.append(cur_info)
    return files_with_info


def delete_mlflow_file(filename: str, storage_path: str) -> None:

    file_path = os.path.join(storage_path, filename)

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
    else:
        cur_info = {"ISSUES": "Metadata not saved",
                    "file path": file_path,
                    "basename": basename
                    }
    return cur_info


def _get_hist_link(version: str, _tag_key: str = 'version') -> str:
    all_models = MLFLOW_CLIENT.search_registered_models()
    model: Optional[RegisteredModel] = None
    for found_model in all_models:
        if (_tag_key in found_model.tags and
                found_model.tags[_tag_key] == version):
            model = found_model
            break
    if model:
        return f"/info/{model.tags['model_file_name']}"
    return "N/A"


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


def get_meta_model(model: RegisteredModel, storage_path: str) -> ModelMetaData:
    try:
        meta = ModelMetaData.from_mlflow_model(model)
    except KeyError:  # old model data with not all the keys
        logger.warning("Recalculating meta - not everything was saved on disk")
        file_path = os.path.join(storage_path, model.tags['model_file_name'])
        meta = create_meta(file_path, model.name)
        _update_model_meta(model, meta)
    return meta


def get_history(file_path: str) -> dict:
    basename = os.path.basename(file_path)
    model = MLFLOW_CLIENT.get_registered_model(basename)
    if model:
        meta = get_meta_model(model, os.path.dirname(file_path))
        cur_info = meta.as_dict()
        versions = cur_info["version_history"].split(",")
        history = [(version, _get_hist_link(version)) for version in versions
                   if version]
        return history
    else:
        return [("ISSUES", "Metadata not saved"),
                # ("looked for", model_version),
                ("file path", file_path),
                ("basename", basename),
                ("SAVED META", str(model))]


def get_all_trees_with_links(storage_path: str):
    # Query for registered models
    models = MLFLOW_CLIENT.list_registered_models()

    # Create a list of tuples containing tree representations and model links
    data = {}
    for model in models:
        saved_meta = get_meta_model(model, storage_path)
        if saved_meta:
            version = saved_meta.version
            # remove empty versions
            versions = [ver for ver in saved_meta.version_history.split(",")
                        if ver]
            data[version] = versions
    nodes = build_nodes(data).values()
    # Pass the _get_hist_link function as the model_link_func parameter
    return get_all_trees(nodes, _get_hist_link)
