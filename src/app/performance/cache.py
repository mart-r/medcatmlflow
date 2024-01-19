import logging

import sqlalchemy.exc

from ..medcat_linkage.medcat_integration import PerDatasetPerformanceResult
from ..medcat_linkage.medcat_integration import remap_to_perf_results
from ..medcat_linkage.medcat_integration import remap_from_perf_results
from ..main.models import ModelDatasetPerformanceResult, db

logger = logging.getLogger(__name__)


def get_cached(model_id: str, ds_id: str) -> PerDatasetPerformanceResult:
    try:
        perf_res = ModelDatasetPerformanceResult.query.filter_by(
            model_id=model_id, dataset_id=ds_id).first()
    except sqlalchemy.exc.OperationalError as e:
        logger.info("Did not find performance results in cache for model '%s'"
                    " and datset '%s'", model_id, ds_id)
        raise ValueError(f"No model cached for model '{model_id}' "
                         "and dataset '{ds_id}'") from e
    if not perf_res:
        logger.info("Did not find performance results in cache for model '%s'"
                    " and datset '%s'", model_id, ds_id)
        raise ValueError(f"No model cached for model '{model_id}' "
                         "and dataset '{ds_id}'")
    logger.info("Found performance results in cache for model '%s'"
                " and datset '%s'", model_id, ds_id)
    return remap_to_perf_results(perf_res.to_dict())


def add_to_cache(model_id: str, ds_id: str,
                 perf: PerDatasetPerformanceResult) -> None:
    mapping = remap_from_perf_results(perf)
    mapping["model_id"] = model_id
    mapping["dataset_id"] = ds_id
    logger.info("Adding performance results for model '%s'"
                " and datset '%s'", model_id, ds_id)
    perf_ds = ModelDatasetPerformanceResult.from_dict(mapping)
    db.session.add(perf_ds)
    db.session.commit()
