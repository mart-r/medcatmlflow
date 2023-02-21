from models.model import Model
from tracking.performance import RegressionSuite
import logging

import mlflow


logger = logging.getLogger(__name__)


class ModelTracker:
    def __init__(self) -> None:
        pass

    def _track_model_single_regression(
        self, model: Model, regression: RegressionSuite
    ) -> None:
        metrics = regression.get_result(model)
        logger.info("Looking at model %s", regression.get_name())
        total = metrics.success + metrics.fail
        sperc = metrics.success / total
        fperc = metrics.fail / total
        extra = metrics.extra[0]
        if len(extra) > 200:
            extra = extra[:200] + " ..."
        logger.info(
            "RESULTING METRICS:\n"
            + f"Successes: {metrics.success} / {total} ({100 * sperc:.2f}%)\n"
            + f"Failures : {metrics.fail} / {total} ({100 * fperc:.2f}%)\n"
            + f"Extra    : {extra}"
        )
        suite_name = regression.get_name()
        mlflow.log_param("model_id", model.get_model_tag())
        mlflow.log_param("model_name", model.get_model_name())
        mlflow.log_metric(f"metric_success_{suite_name}", sperc)

    def save_model(self, model: Model) -> None:
        mlflow.pyfunc.log_model(
            artifact_path=model.get_model_name(),
            python_model=model.get_mlflow_model(),
            artifacts={"model_path": model.get_model_path()},
        )

    def track_model(self, model: Model, regressions: list[RegressionSuite]):
        with mlflow.start_run():
            for regression in regressions:
                self._track_model_single_regression(model, regression)
            logger.info(f"Saving model: {model.get_model_name()}")
            self.save_model(model)
