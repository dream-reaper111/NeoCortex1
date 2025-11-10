import pytest

pytest.importorskip("pandas")

from apps.dashboard.dependencies import get_data_store
from apps.dashboard.services import DashboardService


def test_dashboard_service_summary_contains_expected_sections():
    service = DashboardService(get_data_store())
    summary = service.summary_payloads()
    assert set(summary.pnl_breakdown.keys())
    assert "values" in summary.win_rate_heatmap
    assert len(summary.equity_curve["timestamp"]) == len(summary.equity_curve["value"])
    assert "matrix" in summary.risk_matrix
    assert "sharpe" in summary.metrics
    assert "paths" in summary.monte_carlo


def test_trade_replay_payload_includes_explain_text():
    service = DashboardService(get_data_store())
    payload = service.trade_replay_payload(limit=5)
    assert len(payload) <= 5
    if payload:
        assert "explain_text" in payload[0]
