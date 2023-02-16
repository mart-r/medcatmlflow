import unittest

from src.tracking.performance import RegressionMetrics


class TestRegressionMetrics(unittest.TestCase):
    def setUp(self) -> None:
        self.empty_metrics = RegressionMetrics(success=0, fail=0)
        self.full_win_metrics = RegressionMetrics(success=10, fail=0)
        self.full_loose_metrics = RegressionMetrics(success=0, fail=100)
        self.win_loose_metrics = RegressionMetrics(success=15, fail=20)
        self.all_metrics = [
            self.empty_metrics,
            self.full_loose_metrics,
            self.full_loose_metrics,
            self.win_loose_metrics,
        ]
        self.nonempty_metrics = [
            self.full_loose_metrics,
            self.full_loose_metrics,
            self.win_loose_metrics,
        ]

    def test_metrics_construct(self):
        self.assertIsInstance(self.empty_metrics, RegressionMetrics)

    def test_metrics_have_different_extras(self):
        self.empty_metrics.extra.append("-1")
        m2 = RegressionMetrics(
            success=self.empty_metrics.success, fail=self.empty_metrics.fail
        )
        self.assertFalse(self.empty_metrics == m2)

    def test_metrics_calculates_correct_total(self):
        for metrics in self.all_metrics:
            with self.subTest(f"{metrics}"):
                total = metrics.success + metrics.fail
                self.assertEqual(total, metrics.total)

    def test_empty_metrics_fail_calc_success_rate(self):
        with self.assertRaises(ValueError):
            self.empty_metrics.success_rate

    def test_empty_metrics_fail_calc_fail_rate(self):
        with self.assertRaises(ValueError):
            self.empty_metrics.fail_rate

    def test_nonempty_metrics_calculates_success_rate(self):
        for metrics in self.nonempty_metrics:
            with self.subTest(f"{metrics}"):
                success = metrics.success / metrics.total
                self.assertEqual(success, metrics.success_rate)

    def test_nonempty_metrics_calculates_fail_rate(self):
        for metrics in self.nonempty_metrics:
            with self.subTest(f"{metrics}"):
                fail = metrics.fail / metrics.total
                self.assertEqual(fail, metrics.fail_rate)
