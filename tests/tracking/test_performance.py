import unittest

from src.tracking.performance import RegressionMetrics


class TestRegressionMetrics(unittest.TestCase):
    def test_metrics_construct(self):
        metrics = RegressionMetrics(success=0, fail=0)
        self.assertIsInstance(metrics, RegressionMetrics)

    def metrics_have_different_extras(self):
        m1 = RegressionMetrics(success=0, fail=0)
        m1.extra.append("-1")
        m2 = RegressionMetrics(success=0, fail=0)
        self.assertFalse(m1 == m2)
