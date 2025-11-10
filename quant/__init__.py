"""High-level quant terminal feature suite."""

from .strategy import (
    AutoEnsembleEngine,
    AdaptiveRiskModule,
    RegimeDetector,
    MonteCarloStressTester,
)
from .portfolio import (
    MultiBrokerSynchronizer,
    DynamicAllocator,
    EquityCurveTarget,
)
from .execution import (
    SmartOrderRouter,
    OrderBookHeatmap,
    LatencyTracker,
)
from .security import (
    APIKeyVault,
    AccessControlList,
    AuditLog,
    SystemHealthMonitor,
)
from .analytics import (
    TradeReplayer,
    ProfitLossHeatmap,
    RiskCorrelationMatrix,
    EquityWaterfall,
    AIExplanationLayer,
)
from .research import (
    ModelTrainer,
    FeatureLab,
    ModelRegistry,
    ReinforcementLearner,
    ExplainableAI,
)
from .community import (
    SignalBroadcaster,
    MultiUserDashboard,
    Leaderboard,
    ChatAndAlerts,
    JournalAutomation,
)
from .data_integration import DataIntegrationLayer

__all__ = [
    "AutoEnsembleEngine",
    "AdaptiveRiskModule",
    "RegimeDetector",
    "MonteCarloStressTester",
    "MultiBrokerSynchronizer",
    "DynamicAllocator",
    "EquityCurveTarget",
    "SmartOrderRouter",
    "OrderBookHeatmap",
    "LatencyTracker",
    "APIKeyVault",
    "AccessControlList",
    "AuditLog",
    "SystemHealthMonitor",
    "TradeReplayer",
    "ProfitLossHeatmap",
    "RiskCorrelationMatrix",
    "EquityWaterfall",
    "AIExplanationLayer",
    "ModelTrainer",
    "FeatureLab",
    "ModelRegistry",
    "ReinforcementLearner",
    "ExplainableAI",
    "SignalBroadcaster",
    "MultiUserDashboard",
    "Leaderboard",
    "ChatAndAlerts",
    "JournalAutomation",
    "DataIntegrationLayer",
]
