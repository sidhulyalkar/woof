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


def uncertainty_slice(
    aggregate,
    *,
    name: str,
    cluster_column: str,
    clusters: int,
    improvement_lower: float = 0.01,
    regression_upper: float = 0.005,
    ece_upper: float = 0.06,
):
    baseline = aggregate["models"]["baseline_score"]
    learned = aggregate["models"]["learned_score"]
    improvement = baseline["brier"] - learned["brier"]
    return {
        "name": name,
        "available": True,
        "source": {
            "predictionSha256": "a" * 64,
            "clusterSourceSha256": "b" * 64,
            "clusterJoinKey": "outcome_id",
            "clusterLabelVerified": True,
            "clusterColumn": cluster_column,
            "baselineColumn": "baseline_score",
            "learnedColumn": "learned_score",
        },
        "point": {
            "baselineBrier": baseline["brier"],
            "learnedBrier": learned["brier"],
            "brierImprovement": improvement,
            "learnedMinusBaselineBrier": -improvement,
            "learnedEce10": learned["ece10"],
        },
        "bootstrap": {
            "method": "paired_cluster_percentile_bootstrap",
            "seed": 17,
            "resamples": 4000,
            "confidenceLevel": 0.95,
            "clusterCount": clusters,
            "rowCount": baseline["rows"],
            "brierImprovement": {
                "lower": improvement_lower,
                "upper": improvement + 0.01,
            },
            "learnedMinusBaselineBrier": {
                "lower": -improvement - 0.01,
                "upper": regression_upper,
            },
            "learnedEce10": {"lower": 0.03, "upper": ece_upper},
        },
    }


def uncertainty_report(main, cold_pair, cold_owner):
    return {
        "schemaVersion": "woof-compatibility-uncertainty-v1",
        "policy": {
            "pairedRows": True,
            "relationshipClusterBootstrap": True,
            "clusterIdentityFromEvaluationSplits": True,
            "splitLabelAgreementRequired": True,
            "resamples": 4000,
            "confidenceLevel": 0.95,
            "seed": 17,
        },
        "futureTest": uncertainty_slice(
            main,
            name="future_test",
            cluster_column="owner_pair_key",
            clusters=45,
            improvement_lower=0.008,
        ),
        "coldPair": uncertainty_slice(
            cold_pair,
            name="cold_pair",
            cluster_column="pair_key",
            clusters=18,
            regression_upper=0.006,
        ),
        "coldOwner": uncertainty_slice(
            cold_owner,
            name="cold_owner",
            cluster_column="owner_pair_key",
            clusters=14,
            regression_upper=0.006,
        ),
        "safety": {"name": "safety", "available": False},
    }


def telemetry(fallbacks: int = 8):
    return {
        "attempts": {"total": 500, "fallbacks": fallbacks},
        "latencyMs": {"p95": 430},
    }


class PromotionGateTests(unittest.TestCase):
    def setUp(self):
        self.main = report()
        self.cold_pair = report(rows=180, baseline_brier=0.22, learned_brier=0.215)
        self.cold_owner = report(rows=120, baseline_brier=0.23, learned_brier=0.225)

    def evaluate(self, **overrides):
        arguments = {
            "cold_pair_report": self.cold_pair,
            "cold_owner_report": self.cold_owner,
            "uncertainty_report": uncertainty_report(
                self.main,
                self.cold_pair,
                self.cold_owner,
            ),
            "telemetry": telemetry(),
        }
        arguments.update(overrides)
        return evaluate_gate(
            self.main,
            "baseline_score",
            "learned_score",
            Thresholds(),
            **arguments,
        )

    def test_promotes_only_when_point_uncertainty_and_operational_gates_pass(self):
        receipt = self.evaluate()
        self.assertTrue(receipt["passed"])
        self.assertTrue(receipt["authoritativeEligible"])
        self.assertEqual(receipt["schemaVersion"], "woof-model-promotion-receipt-v2")
        self.assertEqual(receipt["decision"], "promote")
        self.assertTrue(receipt["uncertaintyEvidence"]["passed"])
        self.assertEqual(receipt["failures"], [])

    def test_holds_shadow_when_point_improves_but_bootstrap_lower_bound_is_weak(self):
        uncertainty = uncertainty_report(self.main, self.cold_pair, self.cold_owner)
        uncertainty["futureTest"]["bootstrap"]["brierImprovement"]["lower"] = 0.001
        receipt = self.evaluate(uncertainty_report=uncertainty)
        self.assertFalse(receipt["passed"])
        self.assertIn(
            "future_test_brier_improvement_lower_below_minimum",
            receipt["failures"],
        )

    def test_holds_shadow_when_cold_owner_uncertainty_crosses_regression_limit(self):
        uncertainty = uncertainty_report(self.main, self.cold_pair, self.cold_owner)
        uncertainty["coldOwner"]["bootstrap"]["learnedMinusBaselineBrier"]["upper"] = 0.02
        receipt = self.evaluate(uncertainty_report=uncertainty)
        self.assertFalse(receipt["passed"])
        self.assertIn(
            "cold_owner_brier_regression_upper_above_maximum",
            receipt["failures"],
        )

    def test_holds_shadow_when_uncertainty_evidence_does_not_match_aggregate_report(self):
        uncertainty = uncertainty_report(self.main, self.cold_pair, self.cold_owner)
        uncertainty["futureTest"]["point"]["learnedBrier"] = 0.12
        receipt = self.evaluate(uncertainty_report=uncertainty)
        self.assertFalse(receipt["passed"])
        self.assertIn("future_test_learned_brier_evidence_mismatch", receipt["failures"])

    def test_holds_shadow_when_split_cluster_provenance_is_missing(self):
        uncertainty = uncertainty_report(self.main, self.cold_pair, self.cold_owner)
        del uncertainty["futureTest"]["source"]["clusterSourceSha256"]
        receipt = self.evaluate(uncertainty_report=uncertainty)
        self.assertFalse(receipt["passed"])
        self.assertIn("future_test_cluster_source_hash_missing", receipt["failures"])

    def test_holds_shadow_when_uncertainty_policy_allows_self_declared_clusters(self):
        uncertainty = uncertainty_report(self.main, self.cold_pair, self.cold_owner)
        uncertainty["policy"]["clusterIdentityFromEvaluationSplits"] = False
        receipt = self.evaluate(uncertainty_report=uncertainty)
        self.assertFalse(receipt["passed"])
        self.assertIn("uncertainty_split_cluster_authority_missing", receipt["failures"])

    def test_holds_shadow_when_uncertainty_report_is_missing(self):
        receipt = self.evaluate(uncertainty_report=None)
        self.assertFalse(receipt["passed"])
        self.assertEqual(receipt["decision"], "hold_shadow")
        self.assertIn("uncertainty_report_missing", receipt["failures"])

    def test_holds_shadow_when_fallback_rate_is_too_high(self):
        receipt = self.evaluate(telemetry=telemetry(fallbacks=60))
        self.assertFalse(receipt["passed"])
        self.assertEqual(receipt["decision"], "hold_shadow")
        self.assertIn("fallback_rate_above_maximum", receipt["failures"])

    def test_holds_shadow_on_point_cold_owner_regression_even_if_average_future_improves(self):
        bad_cold_owner = report(rows=120, baseline_brier=0.20, learned_brier=0.23)
        uncertainty = uncertainty_report(self.main, self.cold_pair, bad_cold_owner)
        receipt = self.evaluate(
            cold_owner_report=bad_cold_owner,
            uncertainty_report=uncertainty,
        )
        self.assertFalse(receipt["passed"])
        self.assertIn("cold_owner_brier_regression", receipt["failures"])

    def test_offline_research_bypass_cannot_emit_promotion_decision(self):
        receipt = evaluate_gate(
            self.main,
            "baseline_score",
            "learned_score",
            Thresholds(),
            cold_pair_report=self.cold_pair,
            cold_owner_report=self.cold_owner,
            require_uncertainty=False,
            require_telemetry=False,
        )
        self.assertTrue(receipt["passed"])
        self.assertFalse(receipt["authoritativeEligible"])
        self.assertEqual(receipt["decision"], "research_only")


if __name__ == "__main__":
    unittest.main()
