import argparse
import logging
from typing import Protocol

from tracking.trackers import ModelTracker
from models.loader import ModelType, Loader


logger = logging.getLogger(__name__)


class CLIArguments(Protocol):
    @property
    def modelpath(self) -> str:
        """The model pack path."""

    @property
    def modeltype(self) -> str:
        """The model type."""

    @property
    def regressionsuites(self) -> str:
        """The regression suite paths."""

    @property
    def silent(self) -> bool:
        """Whether or not the loggers should be silent."""

    @property
    def debug(self) -> bool:
        """Whether or not to output verbose/debug output."""


def get_args() -> CLIArguments:
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
    parser.add_argument(
        "--debug",
        help="Allow verbose output",
        action="store_true",  # for debug / verbose output
    )
    parser.add_argument(
        "--silent",
        help="Supress all output",
        action="store_true",  # for silent operation
    )
    return parser.parse_args()


def setup_logging(args: CLIArguments):
    pkg_logger = logging.getLogger(__package__)
    if not args.silent:
        pkg_logger.addHandler(logging.StreamHandler())
    if args.debug:
        pkg_logger.setLevel("DEBUG")


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
