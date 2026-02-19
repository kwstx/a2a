import time
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from src.models.impact import ImpactVector, ImpactCategory

class FundStatus(Enum):
    OPEN = "open"
    LOCKED = "locked" # No more contributions
    DEPLOYED = "deployed"
    COMPLETED = "completed"

@dataclass
class FundContribution:
    agent_id: str
    amount: float
    timestamp: float = field(default_factory=lambda: time.time())

@dataclass
class CooperativeFund:
    """
    A shared repository of credits tied to a specific future impact objective.
    """
    fund_id: str
    target_objective: ImpactVector
    status: FundStatus = FundStatus.OPEN
    contributions: List[FundContribution] = field(default_factory=list)
    deployed_task_cluster_id: Optional[str] = None
    
    @property
    def total_pooled(self) -> float:
        return sum(c.amount for c in self.contributions)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fund_id": self.fund_id,
            "target_objective": self.target_objective.to_dict(),
            "status": self.status.value,
            "total_pooled": self.total_pooled,
            "contributions": [{"agent_id": c.agent_id, "amount": c.amount, "timestamp": c.timestamp} for c in self.contributions],
            "deployed_task_cluster_id": self.deployed_task_cluster_id
        }

@dataclass
class InvestmentEvaluation:
    """
    Outcome of predictive risk modeling for a potential fund deployment.
    """
    fund_id: str
    cluster_id: str
    expected_return: float # Predicted surplus
    investment_cost: float # Credits deployed
    risk_score: float # 0 to 1, derived from uncertainty bounds
    roci: float # Return on Cooperative Investment
    confidence_interval: Tuple[float, float]
    recommendation: str # "ALLOCATE", "REJECT", "NEUTRAL"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fund_id": self.fund_id,
            "cluster_id": self.cluster_id,
            "expected_return": self.expected_return,
            "investment_cost": self.investment_cost,
            "risk_score": self.risk_score,
            "roci": self.roci,
            "confidence_interval": self.confidence_interval,
            "recommendation": self.recommendation,
            "metadata": self.metadata
        }
