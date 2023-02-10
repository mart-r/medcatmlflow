from models.model import Model
from tracking.performance import RegressionSuite

from mlflow import log_param, log_metric


class ModelTracker:
    def __init__(self) -> None:
        pass

    def track_model(self, model: Model, regression: RegressionSuite):
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
        model_name = model.get_model_name()
        model_id = model.get_model_tag()
        suite_name = regression.get_name()
        print('LOGGING')
        log_param('model_id', model_id)
        log_param('model_name', model_name)
        log_metric(f'metric_success_{suite_name}', sperc)
