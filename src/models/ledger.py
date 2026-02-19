import time
from enum import Enum
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from src.models.impact import ImpactVector, ImpactCategory

class EntryType(Enum):
    CREDIT = "credit"
    DEBIT = "debit"

@dataclass
class CreditProvenance:
    """
    Metadata linking a credit to the cooperative surplus event that generated it.
    Ensures economic traceability.
    """
    surplus_event_id: str  # maps to cluster_id
    contribution_claim_id: Optional[str] = None
    task_ids: List[str] = field(default_factory=list)
    negotiation_round: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "surplus_event_id": self.surplus_event_id,
            "contribution_claim_id": self.contribution_claim_id,
            "task_ids": self.task_ids,
            "negotiation_round": self.negotiation_round
        }

@dataclass
class CreditEntry:
    """
    A single entry in the contextualized ledger.
    Every credit carries provenance information and impact context.
    """
    entry_id: str
    agent_id: str
    amount: float
    entry_type: EntryType
    impact_vector: ImpactVector  # The origin impact vector
    domain_context: ImpactCategory # The domain context (e.g., RESEARCH, TECHNICAL)
    provenance: CreditProvenance
    timestamp: float = field(default_factory=lambda: time.time())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "agent_id": self.agent_id,
            "amount": self.amount,
            "entry_type": self.entry_type.value,
            "impact_vector": self.impact_vector.to_dict(),
            "domain_context": self.domain_context.value,
            "provenance": self.provenance.to_dict(),
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }

@dataclass
class AgentBalance:
    """
    Represents the current state of an agent's credits,
    broken down by domain and impact category.
    """
    agent_id: str
    total_balance: float
    balances_by_category: Dict[str, float]
    recent_entries: List[CreditEntry] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "total_balance": self.total_balance,
            "balances_by_category": self.balances_by_category,
            "recent_entries": [e.to_dict() for e in self.recent_entries]
        }
