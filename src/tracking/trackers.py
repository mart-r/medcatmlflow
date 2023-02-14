from models.model import Model
from tracking.performance import RegressionSuite

import mlflow


class ModelTracker:
    def __init__(self) -> None:
        pass

    def _track_model_single_regression(
        self, model: Model, regression: RegressionSuite
    ) -> None:
        metrics = regression.get_result(model)
        print("For", id(regression), regression.get_name())
        total = metrics.success + metrics.fail
        sperc = metrics.success / total
        fperc = metrics.fail / total
        extra = metrics.extra[0]
        if len(extra) > 200:
            extra = extra[:200] + " ..."
        print(
            "RESULTING METRICS:\n"
            + f"Successes: {metrics.success} / {total} ({100 * sperc:.2f}%)\n"
            + f"Failures : {metrics.fail} / {total} ({100 * fperc:.2f}%)\n"
            + f"Extra    : {extra}"
        )
        suite_name = regression.get_name()
        print("LOGGING")
        mlflow.log_param("model_id", model.get_model_tag())
        mlflow.log_param("model_name", model.get_model_name())
        mlflow.log_metric(f"metric_success_{suite_name}", sperc)
        pass

    def track_model(self, model: Model, regressions: list[RegressionSuite]):
        model_name = model.get_model_name()
        model_id = model.get_model_tag()
        with mlflow.start_run() as run:
            uri = run.info.run_id
            full_name = f"{model_name}_{model_id}"
            mlflow.register_model(f"runs:/{uri}", name=full_name)
            for regression in regressions:
                self._track_model_single_regression(model, regression)
