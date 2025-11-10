import math
from datetime import datetime, timedelta

import pytest

from quant import (
    AIExplanationLayer,
    AdaptiveRiskModule,
    APIKeyVault,
    AccessControlList,
    AuditLog,
    AutoEnsembleEngine,
    DataIntegrationLayer,
    DynamicAllocator,
    EquityCurveTarget,
    EquityWaterfall,
    FeatureLab,
    JournalAutomation,
    LatencyTracker,
    ModelRegistry,
    ModelTrainer,
    MonteCarloStressTester,
    MultiBrokerSynchronizer,
    OrderBookHeatmap,
    ProfitLossHeatmap,
    RegimeDetector,
    ReinforcementLearner,
    RiskCorrelationMatrix,
    SignalBroadcaster,
    SmartOrderRouter,
    SystemHealthMonitor,
    TradeReplayer,
)
from quant.community import ChatAndAlerts, Leaderboard, MultiUserDashboard, PortfolioView
from quant.execution import Venue
from quant.portfolio import AccountSnapshot
from quant.research import ExplainableAI
from quant.security import ProcessStatus


def test_auto_ensemble_engine_votes():
    models = {
        "momentum": lambda feats: feats["mom"],
        "mean_reversion": lambda feats: -feats["mom"],
        "volatility": lambda feats: feats["vol"] - 0.5,
    }
    engine = AutoEnsembleEngine(models)
    result = engine.run({"mom": 1.0, "vol": 0.8})
    assert result.consensus == "long"
    assert math.isclose(result.votes["momentum"], 1.0)


def test_adaptive_risk_module_adjusts():
    risk = AdaptiveRiskModule(target_volatility=0.2, base_position=1.0)
    first = risk.update(0.01)
    second = risk.update(0.02)
    assert first.position_size > 0
    assert second.position_size <= risk.base_position * risk.max_leverage


def test_regime_detector_classifies_trend():
    prices = [100 + i for i in range(60)]
    detector = RegimeDetector(lookback=50)
    assert detector.classify(prices) == "trend"


def test_monte_carlo_returns_metrics():
    tester = MonteCarloStressTester(simulations=10)
    stats = tester.simulate([0.01, -0.005, 0.003])
    assert set(stats) == {"worst_drawdown", "average_drawdown", "tail_risk", "expected_equity"}


def test_multi_broker_sync_and_allocation():
    sync = MultiBrokerSynchronizer()
    sync.upsert(
        AccountSnapshot(broker="alpaca", equity=10000, cash=5000, positions={"AAPL": 10, "TSLA": 5})
    )
    sync.upsert(
        AccountSnapshot(broker="ibkr", equity=15000, cash=7000, positions={"AAPL": -2, "MSFT": 7})
    )
    assert pytest.approx(sync.total_equity(), rel=1e-6) == 25000
    positions = sync.consolidated_positions()
    assert positions["AAPL"] == 8

    allocator = DynamicAllocator()
    weights = allocator.allocate({"strat1": {"sharpe": 1.5, "win_rate": 0.6, "trades": 20}})
    assert math.isclose(weights["strat1"], 1.0)


def test_equity_curve_target():
    target = EquityCurveTarget(smoothing=0.5)
    result = target.update(10000)
    assert result.exposure_multiplier == 1.0
    result = target.update(11000)
    assert result.target_equity > 10000


def test_smart_order_router_and_heatmap():
    router = SmartOrderRouter()
    venues = [Venue("A", 5, 1), Venue("B", 10, 2)]
    plan = router.route(12, venues)
    assert sum(order.quantity for order in plan) == pytest.approx(12)

    heatmap = OrderBookHeatmap()
    levels = heatmap.build(bids=[(100, 5), (99.5, 3)], asks=[(100.5, 4), (101, 2)], bin_size=0.5)
    assert "bids" in levels and "asks" in levels


def test_latency_tracker():
    tracker = LatencyTracker()
    tracker.record("api", timestamp=1.0)
    tracker.record("broker", timestamp=1.1)
    tracker.record("webhook", timestamp=1.2)
    stats = tracker.compute()
    assert stats.total_latency_ms == pytest.approx(200.0)
    assert "api->broker" in stats.per_hop_ms


def test_api_key_vault_and_acl():
    vault = APIKeyVault()
    vault.put("alpaca", "secret")
    assert vault.get("alpaca") == "secret"

    acl = AccessControlList()
    acl.assign("alice", "analyst")
    assert acl.check("alice", "read")
    assert not acl.check("alice", "write")


def test_audit_log_anomalies():
    log = AuditLog(anomaly_threshold=1.5)
    now = datetime.utcnow()
    log.record("bot", "order", {"size": 1, "ts": now.isoformat()})
    log.record("bot", "order", {"size": 100, "ts": now.isoformat()})
    anomalies = log.detect_anomalies()
    assert anomalies


def test_system_health_monitor_triggers_remediation():
    monitor = SystemHealthMonitor()
    triggered: list[str] = []

    def remediation(status: ProcessStatus) -> None:
        triggered.append(status.name)

    monitor.register_remediation("uvicorn", remediation)
    monitor.update_status(ProcessStatus(name="uvicorn", healthy=False, latency_ms=500, details={"restart": True}))
    assert triggered == ["uvicorn"]


def test_trade_replayer_and_heatmap():
    replayer = TradeReplayer()
    start = datetime.utcnow()
    prices = [(start + timedelta(minutes=i), 100 + i) for i in range(3)]
    trades = [(start + timedelta(minutes=1), 1.0)]
    frames = replayer.replay(prices, trades)
    assert frames[-1].position == 1.0

    heatmap = ProfitLossHeatmap()
    pnl = heatmap.build([(start, 10.0)])
    assert pnl[start.strftime("%a")][start.hour] == 10.0


def test_risk_correlation_and_waterfall():
    corr = RiskCorrelationMatrix().compute({"A": [0.1, 0.2, 0.3], "B": [0.05, 0.15, 0.2]})
    assert corr["A"]["B"] != 0

    waterfall = EquityWaterfall().build({"strat": [1, 2, 3]})
    assert waterfall[-1]["cumulative"] == 6


def test_ai_explanations():
    layer = AIExplanationLayer()
    explanations = layer.explain({"z_score": 0.7}, {"z_score": ("delta Z > 0.5", 2.0)})
    assert explanations[0]["reason"] == "delta Z > 0.5"


def test_feature_lab_and_registry():
    lab = FeatureLab()
    features = lab.generate([1, 2, 3], [10, 15, 20])
    assert "orderbook_imbalance" in features

    registry = ModelRegistry()
    registry.register("model", {"sharpe": 1.2})
    best = registry.best("model")
    assert best.metrics["sharpe"] == 1.2


def test_reinforcement_learner_and_xai():
    agent = ReinforcementLearner(actions=["buy", "sell"])
    agent.update([0], "buy", reward=1.0, next_state=[0])
    assert agent.policy([0]) == "buy"

    trainer = ModelTrainer()
    data = [([0.0, 1.0], 1.0), ([1.0, 0.0], 0.0)]
    if trainer._torch is None and trainer._tf is None:
        with pytest.raises(RuntimeError):
            trainer.train(data, framework="torch")

    explainer = AIExplanationLayer()
    explanation = explainer.explain({"rsi": 0.6}, {"rsi": ("RSI 60-70", 1.5)})
    assert explanation[0]["reason"] == "RSI 60-70"

    xai_module = ExplainableAI()
    contributions = xai_module.attribute([0.5, 0.6], [1.0, 1.5])
    assert len(contributions) == 2

    xai = ReinforcementLearner(actions=["hold", "exit"])
    xai.update([1], "exit", reward=0.5, next_state=[0])
    assert xai.policy([1]) in {"hold", "exit"}


def test_signal_broadcaster_and_dashboard():
    broadcaster = SignalBroadcaster()
    received: list[dict[str, float]] = []

    broadcaster.subscribe("user", received.append)
    broadcaster.broadcast({"AAPL": 1.0})
    assert received[0]["AAPL"] == 1.0

    dashboard = MultiUserDashboard()
    dashboard.update(PortfolioView(user_id="bob", equity=1000, positions={}))
    assert dashboard.get("bob").equity == 1000

    leaderboard = Leaderboard()
    ranks = leaderboard.rank({"alpha": {"sharpe": 2.0}})
    assert ranks[0][0] == "alpha"


def test_chat_alerts_and_journal():
    alerts = ChatAndAlerts()
    alerts.register_channel("discord", "https://discord")
    dispatched = alerts.broadcast("discord", "Trade executed")
    assert "Trade executed" in dispatched[0]

    journal = JournalAutomation()
    page = journal.create_entry(datetime(2024, 1, 1), "Great session", [{"symbol": "AAPL", "pnl": 10.0}])
    assert "Great session" in page


def test_data_integration_layer():
    layer = DataIntegrationLayer()
    layer.register("sentiment", lambda: {"score": 0.2})
    collected = layer.collect()
    assert collected["sentiment"]["score"] == 0.2
