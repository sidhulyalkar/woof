from __future__ import annotations

import unittest

from ml.evaluation.promotion_gate import Thresholds, evaluate_gate


def report(rows: int = 800, baseline_brier: float = 0.20, learned_brier: float = 0.18):
    return {
        "schemaVersion": "woof-calibration-report-v1",
        "models": {
            "baseline_score": {
                "rows": rows,
                "brier": baseline_brier,
                "ece10": 0.07,
                "rocAuc": 0.72,
            },
            "learned_score": {
                "rows": rows,
                "brier": learned_brier,
                "ece10": 0.045,
                "rocAuc": 0.75,
            },
        },
    }


class PromotionGateTests(unittest.TestCase):
    def test_promotes_only_when_future_cold_and_operational_gates_pass(self):
        receipt = evaluate_gate(
            report(),
            "baseline_score",
            "learned_score",
            Thresholds(),
            cold_pair_report=report(rows=180, baseline_brier=0.22, learned_brier=0.215),
            cold_owner_report=report(rows=120, baseline_brier=0.23, learned_brier=0.225),
            telemetry={
                "attempts": {"total": 500, "fallbacks": 8},
                "latencyMs": {"p95": 430},
            },
        )
        self.assertTrue(receipt["passed"])
        self.assertEqual(receipt["decision"], "promote")
        self.assertEqual(receipt["failures"], [])

    def test_holds_shadow_when_fallback_rate_is_too_high(self):
        receipt = evaluate_gate(
            report(),
            "baseline_score",
            "learned_score",
            Thresholds(),
            cold_pair_report=report(rows=180),
            cold_owner_report=report(rows=120),
            telemetry={
                "attempts": {"total": 500, "fallbacks": 60},
                "latencyMs": {"p95": 430},
            },
        )
        self.assertFalse(receipt["passed"])
        self.assertEqual(receipt["decision"], "hold_shadow")
        self.assertIn("fallback_rate_above_maximum", receipt["failures"])

    def test_holds_shadow_on_cold_owner_regression_even_if_average_improves(self):
        receipt = evaluate_gate(
            report(),
            "baseline_score",
            "learned_score",
            Thresholds(),
            cold_pair_report=report(rows=180),
            cold_owner_report=report(rows=120, baseline_brier=0.20, learned_brier=0.23),
            telemetry={
                "attempts": {"total": 500, "fallbacks": 2},
                "latencyMs": {"p95": 300},
            },
        )
        self.assertFalse(receipt["passed"])
        self.assertIn("cold_owner_brier_regression", receipt["failures"])

    def test_offline_research_can_explicitly_skip_service_telemetry(self):
        receipt = evaluate_gate(
            report(),
            "baseline_score",
            "learned_score",
            Thresholds(),
            cold_pair_report=report(rows=180),
            cold_owner_report=report(rows=120),
            require_telemetry=False,
        )
        self.assertTrue(receipt["passed"])


if __name__ == "__main__":
    unittest.main()
