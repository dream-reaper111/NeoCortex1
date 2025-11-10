"""FastAPI router exposing analytics endpoints for the dashboard."""
from __future__ import annotations
from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect

from .dependencies import get_data_store
from .services import DashboardService

router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])


def get_service() -> DashboardService:
    store = get_data_store()
    return DashboardService(store)


@router.get("/pnl-breakdown")
def pnl_breakdown(service: DashboardService = Depends(get_service)) -> dict:
    return service.pnl_breakdown()


@router.get("/win-rate-heatmap")
def win_rate_heatmap(service: DashboardService = Depends(get_service)) -> dict:
    return service.win_rate_heatmap()


@router.get("/equity-curve")
def equity_curve(service: DashboardService = Depends(get_service)) -> dict:
    return service.equity_curve()


@router.get("/risk-matrix")
def risk_matrix(service: DashboardService = Depends(get_service), window: int = 30) -> dict:
    return service.risk_matrix(window=window)


@router.get("/portfolio-metrics")
def portfolio_metrics(service: DashboardService = Depends(get_service)) -> dict:
    return service.portfolio_metrics()


@router.get("/monte-carlo")
def monte_carlo(service: DashboardService = Depends(get_service), simulations: int = 50) -> dict:
    return service.monte_carlo(num_simulations=simulations)


@router.get("/summary")
def summary(service: DashboardService = Depends(get_service)) -> dict:
    payloads = service.summary_payloads()
    return {
        "pnl_breakdown": payloads.pnl_breakdown,
        "win_rate_heatmap": payloads.win_rate_heatmap,
        "equity_curve": payloads.equity_curve,
        "risk_matrix": payloads.risk_matrix,
        "metrics": payloads.metrics,
        "monte_carlo": payloads.monte_carlo,
    }


@router.get("/trade-replay")
def trade_replay_batch(service: DashboardService = Depends(get_service), limit: int = 50) -> dict:
    return {"events": service.trade_replay_payload(limit=limit)}


@router.websocket("/ws/trade-replay")
async def trade_replay(websocket: WebSocket) -> None:
    await websocket.accept()
    service = get_service()
    try:
        for event in service.trade_replay_events():
            await websocket.send_json(
                {
                    "timestamp": event.timestamp.isoformat(),
                    "symbol": event.symbol,
                    "side": event.side,
                    "quantity": event.quantity,
                    "price": event.price,
                    "pnl": event.pnl,
                    "explain_text": event.explain_text,
                }
            )
    except WebSocketDisconnect:
        return


__all__ = ["router"]
