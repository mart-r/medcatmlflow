from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from models.model import Model


@dataclass
class RegressionMetrics:
    success: int = 0
    fail: int = 0
    extra: list[Any] = field(default_factory=list)

    @property
    def total(self):
        """The total recorded results"""
        return self.success + self.fail

    @property
    def success_rate(self):
        """The fraction of successful results"""
        total = self.total
        if total == 0:
            raise ValueError("No results have been recorded!")
        return self.success / total

    @property
    def fail_rate(self):
        """The fraction of failing results"""
        total = self.total
        if total == 0:
            raise ValueError("No results have been recorded!")
        return self.fail / total


class RegressionSuite(ABC):
    @abstractmethod
    def get_name(self) -> str:
        """Get the name of this regression suite.

        Returns:
            str: The name
        """

    @abstractmethod
    def get_result(self, model: Model) -> RegressionMetrics:
        """Get the results from this regression suite.

        Args:
            model (Model): The model to test.

        Returns:
            RegressionMetrics: The resulting metrics
        """
