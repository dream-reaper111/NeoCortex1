"""High-level quant and automation features for NeoCortex."""

from .strategy_intelligence import (
    AutoEnsembleEngine,
    AdaptiveRiskModule,
    RegimeDetector,
    MonteCarloStressTester,
)
from .portfolio_management import (
    BrokerAccount,
    MultiBrokerSync,
    DynamicAllocator,
    EquityCurveTargeting,
)
from .execution_engine import (
    SmartOrderRouter,
    OrderBookHeatmap,
    LatencyTracker,
)
from .system_security import (
    APIKeyVault,
    AccessControl,
    AuditLog,
    HealthMonitor,
    SelfHealingSupervisor,
)
from .analytics import (
    TradeReplay,
    ProfitHeatmap,
    RiskCorrelationMatrix,
    EquityWaterfall,
    AIExplanationLayer,
)
from .ai_research import (
    TrainingEndpoint,
    FeatureLab,
    ModelRegistry,
    ReinforcementLearningAgent,
    ExplainableAI,
)
from .community import (
    SignalStreamer,
    MultiUserDashboard,
    Leaderboard,
    ChatAlerts,
    JournalAutomation,
)
from .data_integration import (
    SentimentFeedIntegrator,
    OptionsAnalytics,
    EconomicCalendarGuard,
    OnChainMetrics,
    TickStorage,
)

__all__ = [
    'AutoEnsembleEngine',
    'AdaptiveRiskModule',
    'RegimeDetector',
    'MonteCarloStressTester',
    'BrokerAccount',
    'MultiBrokerSync',
    'DynamicAllocator',
    'EquityCurveTargeting',
    'SmartOrderRouter',
    'OrderBookHeatmap',
    'LatencyTracker',
    'APIKeyVault',
    'AccessControl',
    'AuditLog',
    'HealthMonitor',
    'SelfHealingSupervisor',
    'TradeReplay',
    'ProfitHeatmap',
    'RiskCorrelationMatrix',
    'EquityWaterfall',
    'AIExplanationLayer',
    'TrainingEndpoint',
    'FeatureLab',
    'ModelRegistry',
    'ReinforcementLearningAgent',
    'ExplainableAI',
    'SignalStreamer',
    'MultiUserDashboard',
    'Leaderboard',
    'ChatAlerts',
    'JournalAutomation',
    'SentimentFeedIntegrator',
    'OptionsAnalytics',
    'EconomicCalendarGuard',
    'OnChainMetrics',
    'TickStorage',
]
