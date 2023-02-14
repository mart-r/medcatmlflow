import argparse

from tracking.trackers import ModelTracker
from models.loader import ModelType, Loader


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


def main():
    args = get_args()

    tracker = ModelTracker()
    loader = Loader(model_type=ModelType.get_type(args.modeltype))
    model = loader.load_model(args.modelpath)
    suites = loader.load_regr_suite(args.regressionsuites)
    tracker.track_model(model, suites)


if __name__ == "__main__":
    main()
