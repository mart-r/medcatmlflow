import os
from typing import Optional, Callable, List, Tuple, Dict
import shutil
import re

import logging

from mlflow import MlflowClient, MlflowException
from mlflow.entities import Experiment
from mlflow.entities.model_registry import RegisteredModel

from ..medcat_linkage.metadata import ModelMetaData, create_meta
from ..medcat_linkage.medcat_integration import get_cui_counts_for_model
from ..main.utils import build_nodes, get_all_trees, NoSuchModelExcepton

from ..main.envs import STORAGE_PATH

# Configure MLflow
from ..main.envs import DB_URI

MLFLOW_CLIENT = MlflowClient(tracking_uri=DB_URI)

logger = logging.getLogger(__name__)


def has_experiment(name: str) -> bool:
    return bool(MLFLOW_CLIENT.get_experiment_by_name(name))


def create_mlflow_experiment(name: str, description: str) -> str:
    exp = MLFLOW_CLIENT.create_experiment(name,
                                          tags={'description': description})
    return exp


def _get_exp_run_ids(exp: Experiment) -> List[str]:
    return [run.info.run_id
            for run in MLFLOW_CLIENT.search_runs([exp.experiment_id, ])]


def get_all_experiment_names() -> List[str]:
    return [el[0] for el in get_all_experiments()]


def get_all_experiments() -> List[Tuple[str, str, int]]:
    return [(str(exp.name), str(exp.tags.get('description',
                                             "Old experiment with no "
                                             "description")),
             len(_get_exp_run_ids(exp)))
            for exp in MLFLOW_CLIENT.search_experiments()]


def get_experiment_by_name(name: str) -> Dict[str, str]:
    exp = MLFLOW_CLIENT.get_experiment_by_name(name)
    return {"short_name": exp.name,
            "description": exp.tags['description']}


def update_experiment_description(name: str, new_description: str) -> None:
    exp = MLFLOW_CLIENT.get_experiment_by_name(name)
    old_descr = exp.tags['description']
    if old_descr == new_description:
        # do nothing
        return
    exp.tags['description'] = new_description
    MLFLOW_CLIENT.set_experiment_tag(exp.experiment_id, 'description',
                                     new_description)


def delete_experiment(name: str) -> None:
    exp = MLFLOW_CLIENT.get_experiment_by_name(name)
    MLFLOW_CLIENT.delete_experiment(exp.experiment_id)


def _get_experiment_id(experiment_name: str) -> str:
    return MLFLOW_CLIENT.get_experiment_by_name(experiment_name).experiment_id


def _mlflow_pre_meta(experiment_name: str, file_path: str,
                     model_description: str) -> str:
    experiment_id = _get_experiment_id(experiment_name)
    run = MLFLOW_CLIENT.create_run(experiment_id=experiment_id)
    run_id = run.info.run_id
    MLFLOW_CLIENT.log_artifact(run_id, file_path)
    MLFLOW_CLIENT.log_param(run_id, "model_description", model_description)
    return run_id


def _perform_upload(file_path: str, model_name: str,
                    model_description: str, category: str, run_id: str):
    meta = create_meta(file_path, model_name=model_name,
                       description=model_description,
                       category=category,
                       run_id=run_id,
                       hash2mct_id=get_existing_hash2mctid())
    MLFLOW_CLIENT.create_registered_model(model_name,
                                          tags=meta.as_dict(),
                                          description=model_description)
    # Get the artifact URI for the logged file
    artifact_uri = "runs:/{}/{}".format(run_id, file_path)
    # Create a model version associated with the registered model and file
    MLFLOW_CLIENT.create_model_version(model_name, artifact_uri)


def _cleanup_upload(file_path: str, run_id: str):
    # do cleanup on disk
    os.remove(file_path)
    if file_path.endswith('.zip'):
        folder_path = file_path[:-4]
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
    # delete MLFLOW stuff
    MLFLOW_CLIENT.delete_run(run_id)


def attempt_upload(file_name: str, file_saver: Callable[[str], None],
                   experiment_name: str,
                   model_name: str,
                   model_description: str,
                   overwrite: bool):
    if not has_experiment(experiment_name):
        return f'Experiment not found: {experiment_name}'

    # Save the uploaded file to the desired location
    file_path = os.path.join(STORAGE_PATH, file_name)

    if os.path.exists(file_path) and not overwrite:
        return f"File already exists: {file_name}"

    # save file
    file_saver(file_path)

    run_id = _mlflow_pre_meta(experiment_name, file_path, model_description)

    try:
        _perform_upload(file_path, model_name, model_description,
                        experiment_name, run_id)
    except Exception as e:
        logger.error("Unable to store model %s", file_name,
                     exc_info=e)
        _cleanup_upload(file_path, run_id)
        return f"Unable to store model {file_name}: {e}"


RUN_ID_PATTERN = re.compile(re.escape("runs:/") +
                            "(.*?)" +
                            re.escape("/") + ".*")


def update_model_info(model: RegisteredModel,
                      new_name: str, new_descr: str) -> None:
    meta = get_meta_model(model)
    changed = False
    if new_name != meta.name:
        logger.debug("Changing model name from '%s' to '%s'",
                     meta.name, new_name)
        meta.name = new_name
        changed = True
    if new_descr != meta.description:
        logger.debug("Changing model description from '%s' to '%s'",
                     meta.description, new_descr)
        meta.description = new_descr
        changed = True
    if changed:
        _update_model_meta(model, meta)


def _get_run_id(model: RegisteredModel,
                run_id_pattern: re.Pattern = RUN_ID_PATTERN
                ) -> str:
    ver = MLFLOW_CLIENT.search_model_versions(f"name='{model.name}'")[0]
    source = ver.source
    # e.g: runs:/5a5dad1636bf4d87bba373e10dcd99e8//app/models/smth.zip
    matched = run_id_pattern.match(source)
    if matched:
        return matched.group(1)
    else:
        raise ValueError(f"Unable to get run ID for model {model}")


def get_all_models_dict() -> List[dict]:
    # Print the list of models and their information
    files_with_info = []
    for model in get_all_model_metadata():
        files_with_info.append(model.as_dict())
    return files_with_info


def get_model_from_id(model_id: str) -> Optional[ModelMetaData]:
    return _get_meta_model_from_tag(model_id, _tag_key='id')


def get_model_from_version(version: str) -> Optional[ModelMetaData]:
    return _get_meta_model_from_tag(version, _tag_key='version')


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


def _get_mlflow_from_tag(value: str,
                         _tag_key: str = 'version') -> RegisteredModel:
    models = MLFLOW_CLIENT.search_registered_models(
        f"tag.{_tag_key} = '{value}'")
    if len(models) == 0:
        raise NoSuchModelExcepton(_tag_key, value)
    if len(models) > 1:
        logger.warning("Found more than one model for '%s': '%s' "
                       "(%d) while - returning the first of all the "
                       "models found", _tag_key, value, len(models))
    return models[0]


def get_mlflow_from_id(model_id: str) -> RegisteredModel:
    return _get_mlflow_from_tag(model_id, 'id')


def _get_meta_model_from_tag(value: str,
                             _tag_key: str) -> Optional[ModelMetaData]:
    try:
        model = _get_mlflow_from_tag(value, _tag_key=_tag_key)
    except NoSuchModelExcepton:
        return None
    return ModelMetaData.from_mlflow_model(model, run_id=model.tags['run_id'])


def _get_hist_link(version: str) -> str:
    model = get_model_from_version(version)
    if model:
        return f"/info/{model.id}"
    return ''


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


def recalc_model_metadata(model: RegisteredModel) -> None:
    if 'cdb_hash' in model.tags:
        cdb_hash = model.tags['cdb_hash']
    else:
        cdb_hash = None
    if 'mct_cdb_id' in model.tags:
        mct_cdb_id = model.tags['mct_cdb_id']
    else:
        mct_cdb_id = None
    file_path = os.path.join(STORAGE_PATH, model.tags['model_file_name'])
    run_id = _get_run_id(model)
    meta = create_meta(file_path, model.name,
                       description=model.description,
                       # if no category saved, we can't re-create
                       category=model.tags['category'],
                       run_id=run_id,
                       hash2mct_id={cdb_hash: mct_cdb_id},
                       existing_id=model.tags.get("id", None))
    _update_model_meta(model, meta)


def get_meta_model(model: RegisteredModel) -> ModelMetaData:
    run_id = _get_run_id(model)
    try:
        meta = ModelMetaData.from_mlflow_model(model, run_id=run_id)
    except KeyError as e:  # old model data with not all the keys
        logger.warning("Recalculating meta - not everything was saved on disk"
                       " exception: %e", e)
        recalc_model_metadata(model)
        meta = ModelMetaData.from_mlflow_model(model, run_id=run_id)
    return meta


def get_history(meta_in: ModelMetaData) -> List[Tuple[str, Optional[dict]]]:
    versions = meta_in.version_history
    history = []
    for version in versions:
        if not version:
            continue
        _get_meta_model_from_tag
        meta = get_model_from_version(version)
        meta_dict = meta.as_dict() if meta else None
        history.append((version, meta_dict))
    return history


def get_all_trees_with_links(
) -> List[Tuple[List[Tuple[str, str, str, str]], str]]:
    data: Dict[str, Tuple[List[str], str]] = {}
    for saved_meta in get_all_model_metadata():
        if saved_meta:
            version = saved_meta.version
            # remove empty versions
            versions = [ver for ver in saved_meta.version_history if ver]
            data[version] = (versions, saved_meta.category)
    nodes = build_nodes(data).values()
    # Pass the _get_hist_link function as the model_link_func parameter
    return get_all_trees(nodes, _get_hist_link, get_model_name_from_version)


def get_existing_hash2mctid() -> dict:
    out = {}
    for saved_meta in get_all_model_metadata():
        if saved_meta.mct_cdb_id:
            out[saved_meta.cdb_hash] = saved_meta.mct_cdb_id
    return out


def get_all_model_metadata() -> List[ModelMetaData]:
    return [get_meta_model(model) for model
            in MLFLOW_CLIENT.search_registered_models()]


def get_model_from_file_name(model_file: str) -> Optional[ModelMetaData]:
    model = MLFLOW_CLIENT.get_registered_model(model_file)
    if model:
        return get_meta_model(model)
    return None


def get_model_name_from_version(version: str) -> str:
    model = get_model_from_version(version)
    if not model:
        return version
    return model.name


def get_model_cui_counts(model_ids: List[str], cuis: List[str]) -> dict:
    out = {}
    for model_id in model_ids:
        model_meta = get_model_from_id(model_id)
        file_path = os.path.join(STORAGE_PATH, model_meta.model_file_name)
        model_result = get_cui_counts_for_model(file_path, cuis)
        out[model_meta.name] = model_result
    return out
