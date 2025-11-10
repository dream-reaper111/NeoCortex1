from datetime import datetime, timedelta, timezone
import random

import pytest

torch = pytest.importorskip("torch")
from torch.utils.data import DataLoader, TensorDataset

from services.data_pipeline.ingest import IngestionConfig, MarketDataIngestor, OHLCVBar
from services.feature_engineering.indicators import batch_enrich
from services.feature_engineering.regime import batch_label
from services.model_orchestration.ensemble import EnsembleController
from services.models.forecast import (
    ForecastTrainer,
    PriceLSTMForecaster,
    PriceTransformerForecaster,
    SequenceModelConfig,
)
from services.models.reinforcement import (
    DeepQConfig,
    DeepQTradingAgent,
    PPOConfig,
    PPOTradingAgent,
)
from services.risk_management.position_sizing import (
    SizingConfig,
    adaptive_position_size,
    atr_position_size,
    equity_curve_drawdown,
    is_max_drawdown_exceeded,
    vix_volatility_scaler,
)


def make_ingested_store():
    end = datetime(2024, 1, 2, tzinfo=timezone.utc)
    start = end - timedelta(days=2)
    config = IngestionConfig(tickers=["TEST"], start=start, end=end, seed=42)
    store = MarketDataIngestor(config).run()
    return store


def test_data_ingestion_produces_structured_output():
    store = make_ingested_store()
    ohlcv = store.ohlcv["TEST"]
    order_flow = store.order_flow["TEST"]
    sentiment = store.sentiment["TEST"]
    assert isinstance(ohlcv[0], OHLCVBar)
    assert len(ohlcv) == len(order_flow) == len(sentiment) > 0
    assert order_flow[0].imbalance == order_flow[0].imbalance  # finite number


def test_feature_engineering_and_regime_labelling():
    store = make_ingested_store()
    enriched = batch_enrich({"TEST": store.ohlcv["TEST"]}, sentiment={"TEST": store.sentiment["TEST"]})
    labelled = batch_label(enriched)
    rows = labelled["TEST"]
    assert rows[0]["rsi"] >= 0
    assert "regime" in rows[-1]
    assert rows[-1]["regime"] in {"neutral", "trend_up", "trend_down", "range", "squeeze", "news_spike"}


def _build_sequence_dataset(samples: int = 64, seq_len: int = 16, input_dim: int = 5):
    torch.manual_seed(0)
    features = torch.randn(samples, seq_len, input_dim)
    targets = features.mean(dim=1, keepdim=True)
    dataset = TensorDataset(features, targets)
    return DataLoader(dataset, batch_size=16, shuffle=True)


def test_sequence_models_train():
    loader = _build_sequence_dataset()
    config = SequenceModelConfig(input_dim=5, sequence_length=16, hidden_dim=32)

    lstm = PriceLSTMForecaster(config)
    trainer = ForecastTrainer(lstm, lr=5e-3)
    initial = trainer.evaluate(loader)
    for _ in range(3):
        trainer.train_epoch(loader)
    assert trainer.evaluate(loader) < initial

    transformer = PriceTransformerForecaster(config)
    trainer_tf = ForecastTrainer(transformer, lr=5e-3)
    initial_tf = trainer_tf.evaluate(loader)
    for _ in range(3):
        trainer_tf.train_epoch(loader)
    assert trainer_tf.evaluate(loader) < initial_tf


def test_deep_q_agent_update():
    config = DeepQConfig(state_dim=4, action_dim=3, batch_size=16, min_buffer=32, buffer_size=256)
    agent = DeepQTradingAgent(config)
    random.seed(0)
    torch.manual_seed(0)
    for _ in range(64):
        state = torch.randn(config.state_dim)
        next_state = torch.randn(config.state_dim)
        action = random.randrange(config.action_dim)
        reward = random.uniform(-1, 1)
        agent.push_transition(state, action, reward, next_state, done=False)
    loss = agent.update()
    assert loss is not None
    assert agent.epsilon < 1.0


def test_ppo_agent_update():
    config = PPOConfig(state_dim=4, action_dim=2)
    agent = PPOTradingAgent(config)
    torch.manual_seed(0)
    trajectories = []
    state = torch.randn(config.state_dim)
    for _ in range(8):
        action, log_prob, value = agent.act(state)
        reward = random.uniform(-1, 1)
        next_state = torch.randn(config.state_dim)
        trajectories.append((state, action, reward, log_prob.detach(), value.detach()))
        state = next_state
    loss = agent.update(trajectories)
    assert loss > 0


def test_ensemble_blending():
    controller = EnsembleController(window=10)
    controller.record_performance("lstm", [0.01, 0.02, -0.005], 0.6)
    controller.record_performance("transformer", [0.015, 0.01, 0.0], 0.7)
    weights = controller.compute_weights()
    assert abs(sum(weights.values()) - 1.0) < 1e-6
    blended = controller.blend({"lstm": 0.1, "transformer": -0.05})
    assert isinstance(blended, float)


def test_risk_management_utilities():
    config = SizingConfig(account_equity=100_000)
    size = atr_position_size(atr=1.5, config=config)
    assert size > 0
    scaled = adaptive_position_size(atr=1.5, vix_value=30.0, config=config)
    assert scaled <= size
    scaler = vix_volatility_scaler(20.0, config)
    assert 0.1 <= scaler <= 1.0

    equity = [100_000, 101_000, 99_000, 98_500, 102_000]
    dd = equity_curve_drawdown(equity)
    assert min(dd) <= 0
    assert is_max_drawdown_exceeded(equity, limit=0.02) is True
