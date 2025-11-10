"""Dash application that consumes the analytics API."""
from __future__ import annotations

import os
from typing import Any, Dict

import dash
from dash import Dash, dcc, html
import plotly.graph_objects as go
import requests

from .api import router  # noqa: F401  # Ensure router is discoverable when app imported
from .services import DashboardService
from .dependencies import get_data_store

API_BASE_URL = os.getenv("DASHBOARD_API_BASE", "http://localhost:8000/api/dashboard")


class DashboardAPIClient:
    """Simple HTTP client that retrieves dashboard data via the FastAPI endpoints."""

    def __init__(self, base_url: str = API_BASE_URL) -> None:
        self._base_url = base_url.rstrip("/")
        self._service = DashboardService(get_data_store())

    def _get(self, path: str) -> Dict[str, Any]:
        url = f"{self._base_url}/{path.lstrip('/')}"
        try:
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            return response.json()
        except Exception:
            # Fallback to in-process service to keep dashboards functional without HTTP roundtrips.
            if path.startswith("summary"):
                summary = self._service.summary_payloads()
                return {
                    "pnl_breakdown": summary.pnl_breakdown,
                    "win_rate_heatmap": summary.win_rate_heatmap,
                    "equity_curve": summary.equity_curve,
                    "risk_matrix": summary.risk_matrix,
                    "metrics": summary.metrics,
                    "monte_carlo": summary.monte_carlo,
                }
            if path.startswith("pnl-breakdown"):
                return self._service.pnl_breakdown()
            if path.startswith("win-rate-heatmap"):
                return self._service.win_rate_heatmap()
            if path.startswith("equity-curve"):
                return self._service.equity_curve()
            if path.startswith("risk-matrix"):
                return self._service.risk_matrix()
            if path.startswith("portfolio-metrics"):
                return self._service.portfolio_metrics()
            if path.startswith("monte-carlo"):
                return self._service.monte_carlo()
            return {}

    def summary(self) -> Dict[str, Any]:
        return self._get("summary")

    def trade_replay(self, limit: int = 20) -> Dict[str, Any]:
        try:
            response = requests.get(f"{self._base_url}/trade-replay", params={"limit": limit}, timeout=5)
            response.raise_for_status()
            return response.json()
        except Exception:
            return {"events": self._service.trade_replay_payload(limit=limit)}


def build_dashboard_app(server=None) -> Dash:
    external_stylesheets = [
        "https://cdnjs.cloudflare.com/ajax/libs/normalize/8.0.1/normalize.min.css",
        "https://cdnjs.cloudflare.com/ajax/libs/skeleton/2.0.4/skeleton.min.css",
    ]
    dash_app = dash.Dash(
        __name__,
        server=server,
        url_base_pathname="/dashboard/",
        external_stylesheets=external_stylesheets,
    )
    client = DashboardAPIClient()

    dash_app.layout = html.Div(
        className="container",
        children=[
            html.H1("NeoCortex Trading Analytics Dashboard"),
            html.Style(
                """
                .trade-replay { max-height: 320px; overflow-y: auto; padding: 1rem; border: 1px solid #ccc; }
                .trade-event { margin-bottom: 0.75rem; padding-bottom: 0.5rem; border-bottom: 1px dashed #ddd; }
                .trade-explain { display: block; margin-top: 0.3rem; color: #555; font-style: italic; }
                """
            ),
            dcc.Interval(id="refresh-interval", interval=60_000, n_intervals=0),
            dcc.Interval(id="trade-replay-interval", interval=10_000, n_intervals=0),
            dcc.Store(id="dashboard-data"),
            dcc.Store(id="trade-replay-data"),
            html.Div(
                className="row",
                children=[
                    html.Div(className="six columns", children=[dcc.Graph(id="pnl-breakdown")]),
                    html.Div(className="six columns", children=[dcc.Graph(id="equity-curve")]),
                ],
            ),
            html.Div(
                className="row",
                children=[
                    html.Div(className="six columns", children=[dcc.Graph(id="win-rate-heatmap")]),
                    html.Div(className="six columns", children=[dcc.Graph(id="risk-matrix")]),
                ],
            ),
            html.Div(
                className="row",
                children=[
                    html.Div(className="six columns", children=[dcc.Graph(id="monte-carlo")]),
                    html.Div(
                        className="six columns",
                        children=[
                            html.H3("Portfolio Metrics"),
                            html.Ul(id="portfolio-metrics"),
                        ],
                    ),
                ],
            ),
            html.Div(
                className="row",
                children=[
                    html.Div(
                        className="twelve columns",
                        children=[
                            html.H3("Explain Trade Replay"),
                            html.Div(id="trade-replay-log", className="trade-replay"),
                        ],
                    )
                ],
            ),
        ],
    )

    @dash_app.callback(
        dash.Output("dashboard-data", "data"),
        dash.Input("refresh-interval", "n_intervals"),
        prevent_initial_call=False,
    )
    def refresh_data(_: int) -> Dict[str, Any]:
        return client.summary()

    @dash_app.callback(dash.Output("pnl-breakdown", "figure"), dash.Input("dashboard-data", "data"))
    def update_pnl(data: Dict[str, Any]) -> go.Figure:
        pnl = data.get("pnl_breakdown", {}) if data else {}
        figure = go.Figure()
        if pnl:
            symbols = list(pnl.keys())
            values = list(pnl.values())
            figure.add_bar(x=symbols, y=values, marker_color="teal")
        figure.update_layout(title="P/L Breakdown", xaxis_title="Symbol", yaxis_title="Total P/L")
        return figure

    @dash_app.callback(dash.Output("equity-curve", "figure"), dash.Input("dashboard-data", "data"))
    def update_equity(data: Dict[str, Any]) -> go.Figure:
        equity = data.get("equity_curve", {}) if data else {}
        timestamps = equity.get("timestamp", [])
        values = equity.get("value", [])
        figure = go.Figure()
        if timestamps and values:
            figure.add_trace(go.Scatter(x=timestamps, y=values, mode="lines", name="Equity"))
        figure.update_layout(title="Equity Curve", xaxis_title="Date", yaxis_title="Equity")
        return figure

    @dash_app.callback(dash.Output("win-rate-heatmap", "figure"), dash.Input("dashboard-data", "data"))
    def update_heatmap(data: Dict[str, Any]) -> go.Figure:
        heatmap = data.get("win_rate_heatmap", {}) if data else {}
        figure = go.Figure()
        if heatmap:
            figure.add_trace(
                go.Heatmap(
                    z=heatmap.get("values", []),
                    x=heatmap.get("columns", []),
                    y=heatmap.get("index", []),
                    colorscale="Viridis",
                )
            )
        figure.update_layout(title="Win-Rate Heatmap")
        return figure

    @dash_app.callback(dash.Output("risk-matrix", "figure"), dash.Input("dashboard-data", "data"))
    def update_risk_matrix(data: Dict[str, Any]) -> go.Figure:
        matrix = data.get("risk_matrix", {}) if data else {}
        figure = go.Figure()
        if matrix:
            figure.add_trace(
                go.Heatmap(
                    z=matrix.get("matrix", []),
                    x=matrix.get("labels", []),
                    y=matrix.get("labels", []),
                    colorscale="RdBu",
                    zmin=-1,
                    zmax=1,
                )
            )
        figure.update_layout(title="Rolling Correlation Matrix")
        return figure

    @dash_app.callback(dash.Output("monte-carlo", "figure"), dash.Input("dashboard-data", "data"))
    def update_monte_carlo(data: Dict[str, Any]) -> go.Figure:
        payload = data.get("monte_carlo", {}) if data else {}
        figure = go.Figure()
        if payload:
            paths = payload.get("paths", {})
            for _, values in paths.items():
                figure.add_trace(go.Scatter(y=values, mode="lines", line=dict(width=1), opacity=0.3))
        figure.update_layout(title="Monte Carlo Equity Simulations", xaxis_title="Period", yaxis_title="Equity")
        return figure

    @dash_app.callback(dash.Output("portfolio-metrics", "children"), dash.Input("dashboard-data", "data"))
    def update_metrics(data: Dict[str, Any]):
        metrics = data.get("metrics", {}) if data else {}
        return [html.Li(f"{key.title().replace('_', ' ')}: {value:.2f}") for key, value in metrics.items()]

    @dash_app.callback(
        dash.Output("trade-replay-data", "data"),
        dash.Input("trade-replay-interval", "n_intervals"),
        prevent_initial_call=False,
    )
    def refresh_trade_replay(_: int) -> Dict[str, Any]:
        return client.trade_replay(limit=20)

    @dash_app.callback(dash.Output("trade-replay-log", "children"), dash.Input("trade-replay-data", "data"))
    def update_trade_replay(data: Dict[str, Any]):
        events = data.get("events", []) if data else []
        if not events:
            return html.Em("No trades available")
        return [
            html.Div(
                className="trade-event",
                children=[
                    html.Strong(f"{event['timestamp']} - {event['symbol']} {event['side']}"),
                    html.Span(f" Size: {event['quantity']:.0f} @ {event['price']:.2f} | P/L: {event['pnl']:.2f}"),
                    html.Div(event.get("explain_text", ""), className="trade-explain"),
                ],
            )
            for event in events
        ]

    return dash_app


__all__ = ["build_dashboard_app", "DashboardAPIClient"]
