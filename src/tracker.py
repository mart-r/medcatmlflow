import argparse
import logging

from tracking.trackers import ModelTracker
from models.loader import ModelType, Loader


logger = logging.getLogger(__name__)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("modelpath", help="Path to the model being loaded")
    parser.add_argument(
        "--modeltype",
        help="Specify type of model." " Currently only 'medcat' is supported",
        default="medcat",
    )
    parser.add_argument(
        "regressionsuites",
        help="The paths to the regression suites to be tested",
        nargs="+",
    )
    return parser.parse_args()


def setup_logging(args: argparse.Namespace):
    pkg_logger = logging.getLogger(__package__)
    pkg_logger.addHandler(logging.StreamHandler())
    pkg_logger.setLevel("INFO")


def main():
    args = get_args()

    setup_logging(args)

    tracker = ModelTracker()
    loader = Loader(model_type=ModelType.get_type(args.modeltype))
    model = loader.load_model(args.modelpath)
    suites = loader.load_regr_suite(args.regressionsuites)
    tracker.track_model(model, suites)


if __name__ == "__main__":
    main()
