from abc import ABC, abstractmethod

from mlflow.pyfunc import PythonModel

from models.annotations import Annotation


class Model(ABC):
    """Describes a model in a very limited sense of annotating documents."""

    @abstractmethod
    def get_mlflow_model(self) -> PythonModel:
        """Get the MLFlow model.

        Returns:
            PythonModel: The MLFlow model
        """

    @abstractmethod
    def get_model_name(self) -> str:
        """Get the name of the model.

        Returns:
            str: The name of the model.
        """

    @abstractmethod
    def get_model_path(self) -> str:
        """Get the path to the file on disk corresponding to this model.

        Returns:
            str: The file path.
        """

    @abstractmethod
    def get_model_tag(self) -> str:
        """Get the ID/tag of the model.

        Returns:
            str: The tag of the model.
        """

    @abstractmethod
    def annotate(self, document: str) -> list[Annotation]:
        """Get the annotations from this model.

        Args:
            document (str): The document to predict

        Returns:
            list[Annotation]: The resulting dictionary
        """
