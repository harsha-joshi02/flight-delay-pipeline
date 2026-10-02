"""Unit tests for the model promotion gate logic."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.evaluation.gate import (
    GateDecision,
    GateReport,
    evaluate_and_gate,
)


def make_gate_report(**overrides) -> GateReport:
    defaults = dict(
        decision=GateDecision.PROMOTED,
        new_run_id="abc123",
        new_auc=0.82,
        production_run_id="def456",
        production_auc=0.80,
        auc_delta=0.02,
        threshold=0.005,
        reason="New model is better.",
    )
    defaults.update(overrides)
    return GateReport(**defaults)


class TestGateReport:
    def test_to_dict_serializes_decision_as_string(self):
        report = make_gate_report()
        d = report.to_dict()
        assert d["decision"] == "promoted"
        assert isinstance(d["decision"], str)

    def test_first_model_has_no_delta(self):
        report = make_gate_report(
            decision=GateDecision.FIRST_MODEL,
            production_run_id=None,
            production_auc=None,
            auc_delta=None,
        )
        d = report.to_dict()
        assert d["production_auc"] is None
        assert d["auc_delta"] is None

    def test_rejected_decision_preserves_info(self):
        report = make_gate_report(
            decision=GateDecision.REJECTED,
            auc_delta=-0.01,
            reason="Worse than production.",
        )
        assert report.decision == GateDecision.REJECTED
        assert report.auc_delta == -0.01


class TestEvaluateAndGate:
    @patch("src.evaluation.gate._get_production_model", return_value=None)
    @patch("src.evaluation.gate._auc_from_run", return_value=0.82)
    @patch("src.evaluation.gate._get_new_model_version", return_value="1")
    @patch("src.evaluation.gate._transition_model")
    def test_first_model_always_promoted(self, mock_transition, mock_version, mock_auc, mock_prod):
        report = evaluate_and_gate("run_abc", save_report=False)
        assert report.decision == GateDecision.FIRST_MODEL
        mock_transition.assert_called_once_with("1", "Production")

    @patch("src.evaluation.gate._get_production_model")
    @patch("src.evaluation.gate._auc_from_run")
    @patch("src.evaluation.gate._get_new_model_version", return_value="2")
    @patch("src.evaluation.gate._transition_model")
    def test_model_promoted_when_above_threshold(self, mock_transition, mock_version, mock_auc, mock_prod):
        prod_mock = MagicMock()
        prod_mock.run_id = "prod_run"
        mock_prod.return_value = prod_mock
        mock_auc.side_effect = [0.82, 0.80]

        report = evaluate_and_gate("new_run", threshold=0.005, save_report=False)
        assert report.decision == GateDecision.PROMOTED
        mock_transition.assert_called_with("2", "Production")

    @patch("src.evaluation.gate._get_production_model")
    @patch("src.evaluation.gate._auc_from_run")
    @patch("src.evaluation.gate._get_new_model_version", return_value="3")
    @patch("src.evaluation.gate._transition_model")
    def test_model_rejected_when_below_threshold(self, mock_transition, mock_version, mock_auc, mock_prod):
        prod_mock = MagicMock()
        prod_mock.run_id = "prod_run"
        mock_prod.return_value = prod_mock
        mock_auc.side_effect = [0.800, 0.800]

        report = evaluate_and_gate("new_run", threshold=0.005, save_report=False)
        assert report.decision == GateDecision.REJECTED
        mock_transition.assert_called_with("3", "Staging")

    @patch("src.evaluation.gate._get_production_model")
    @patch("src.evaluation.gate._auc_from_run")
    @patch("src.evaluation.gate._get_new_model_version", return_value="4")
    @patch("src.evaluation.gate._transition_model")
    def test_model_rejected_when_worse(self, mock_transition, mock_version, mock_auc, mock_prod):
        prod_mock = MagicMock()
        prod_mock.run_id = "prod_run"
        mock_prod.return_value = prod_mock
        mock_auc.side_effect = [0.78, 0.82]

        report = evaluate_and_gate("new_run", threshold=0.005, save_report=False)
        assert report.decision == GateDecision.REJECTED
        assert report.auc_delta == pytest.approx(-0.04, abs=1e-4)

    @patch("src.evaluation.gate._auc_from_run", return_value=None)
    def test_raises_when_auc_unavailable(self, mock_auc):
        with pytest.raises(ValueError, match="AUC"):
            evaluate_and_gate("run_xyz", save_report=False)

    @patch("src.evaluation.gate._auc_from_run", return_value=0.82)
    @patch("src.evaluation.gate._get_new_model_version", return_value=None)
    def test_raises_when_registered_version_is_unavailable(self, mock_version, mock_auc):
        with pytest.raises(ValueError, match="registered model version"):
            evaluate_and_gate("run_xyz", save_report=False)

    @patch("src.evaluation.gate._get_production_model")
    @patch("src.evaluation.gate._auc_from_run")
    @patch("src.evaluation.gate._get_new_model_version", return_value="2")
    def test_raises_when_production_auc_is_unavailable(self, mock_version, mock_auc, mock_prod):
        prod_mock = MagicMock()
        prod_mock.run_id = "prod_run"
        mock_prod.return_value = prod_mock
        mock_auc.side_effect = [0.82, None]

        with pytest.raises(ValueError, match="production AUC"):
            evaluate_and_gate("new_run", save_report=False)

    @patch("src.evaluation.gate._get_production_model")
    @patch("src.evaluation.gate._auc_from_run")
    @patch("src.evaluation.gate._get_new_model_version", return_value="5")
    @patch("src.evaluation.gate._transition_model")
    def test_gate_report_saved_to_disk(self, mock_transition, mock_version, mock_auc, mock_prod, tmp_path):
        import json

        prod_mock = MagicMock()
        prod_mock.run_id = "prod_run"
        mock_prod.return_value = prod_mock
        mock_auc.side_effect = [0.85, 0.80]

        with patch("src.evaluation.gate.REPORTS_DIR", tmp_path):
            evaluate_and_gate("new_run", threshold=0.005, save_report=True)

        report_path = tmp_path / "gate_report.json"
        assert report_path.exists()
        with open(report_path) as f:
            d = json.load(f)
        assert d["decision"] == "promoted"
        assert d["new_auc"] == pytest.approx(0.85)


class TestThresholdEdgeCases:
    @patch("src.evaluation.gate._get_production_model")
    @patch("src.evaluation.gate._auc_from_run")
    @patch("src.evaluation.gate._get_new_model_version", return_value="6")
    @patch("src.evaluation.gate._transition_model")
    def test_exactly_at_threshold_promotes(self, mock_transition, mock_version, mock_auc, mock_prod):
        prod_mock = MagicMock()
        prod_mock.run_id = "prod_run"
        mock_prod.return_value = prod_mock
        mock_auc.side_effect = [0.805, 0.800]

        report = evaluate_and_gate("run_x", threshold=0.005, save_report=False)
        assert report.decision == GateDecision.PROMOTED

    @patch("src.evaluation.gate._get_production_model")
    @patch("src.evaluation.gate._auc_from_run")
    @patch("src.evaluation.gate._get_new_model_version", return_value="7")
    @patch("src.evaluation.gate._transition_model")
    def test_just_below_threshold_rejects(self, mock_transition, mock_version, mock_auc, mock_prod):
        prod_mock = MagicMock()
        prod_mock.run_id = "prod_run"
        mock_prod.return_value = prod_mock
        mock_auc.side_effect = [0.8049, 0.800]

        report = evaluate_and_gate("run_y", threshold=0.005, save_report=False)
        assert report.decision == GateDecision.REJECTED


def _should_retrain_from_report(report: GateReport) -> bool:
    return report.decision == GateDecision.REJECTED


class TestShouldRetrain:
    def test_rejected_model_suggests_retrain(self):
        report = make_gate_report(decision=GateDecision.REJECTED)
        assert _should_retrain_from_report(report) is True

    def test_promoted_model_no_retrain(self):
        report = make_gate_report(decision=GateDecision.PROMOTED)
        assert _should_retrain_from_report(report) is False

    def test_first_model_no_retrain(self):
        report = make_gate_report(
            decision=GateDecision.FIRST_MODEL,
            production_run_id=None,
            production_auc=None,
            auc_delta=None,
        )
        assert _should_retrain_from_report(report) is False
