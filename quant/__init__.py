"""High-level quant and automation features for NeoCortex."""

from __future__ import annotations

from quant.strategy import AutoEnsembleEngine, AdaptiveRiskModule, MonteCarloStressTester, RegimeDetector
from quant.portfolio import DynamicAllocator, EquityCurveTarget, MultiBrokerSynchronizer
from quant.execution import LatencyTracker, OrderBookHeatmap, SmartOrderRouter
from quant.security import APIKeyVault, AccessControlList, AuditLog, ProcessStatus, SystemHealthMonitor
from quant.analytics import AIExplanationLayer, EquityWaterfall, ProfitLossHeatmap, RiskCorrelationMatrix, TradeReplayer
from quant.data_integration import DataIntegrationLayer
from quant.research import ExplainableAI, FeatureLab, ModelRegistry, ModelTrainer, ReinforcementLearner
from quant.community import ChatAndAlerts, JournalAutomation, Leaderboard, MultiUserDashboard, SignalBroadcaster

__all__ = [
    "AdaptiveRiskModule",
    "AIExplanationLayer",
    "APIKeyVault",
    "AccessControlList",
    "AuditLog",
    "AutoEnsembleEngine",
    "ChatAndAlerts",
    "DataIntegrationLayer",
    "DynamicAllocator",
    "EquityCurveTarget",
    "EquityWaterfall",
    "FeatureLab",
    "JournalAutomation",
    "LatencyTracker",
    "Leaderboard",
    "ModelRegistry",
    "ModelTrainer",
    "MonteCarloStressTester",
    "MultiBrokerSynchronizer",
    "MultiUserDashboard",
    "OrderBookHeatmap",
    "ProcessStatus",
    "ProfitLossHeatmap",
    "RegimeDetector",
    "ReinforcementLearner",
    "RiskCorrelationMatrix",
    "SignalBroadcaster",
    "SmartOrderRouter",
    "SystemHealthMonitor",
    "TradeReplayer",
    "ExplainableAI",
]
