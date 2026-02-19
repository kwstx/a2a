import uuid
import time
from typing import Dict, List, Any, Optional, Tuple
from src.models.cooperative_fund import CooperativeFund, FundStatus, FundContribution, InvestmentEvaluation
from src.models.impact import ImpactVector, ImpactProjection, SurplusPool
from src.engine.ledger import ContextualizedLedgerEngine
from src.engine.surplus import CooperativeSurplusEngine
from src.models.ledger import EntryType, CreditProvenance, CreditEntry

class CooperativeInvestingEngine:
    """
    Manages cross-role pooling and cooperative fund structures.
    Applies predictive risk modeling to evaluate ROCI and manages deployment of pooled credits.
    """
    
    def __init__(self, ledger_engine: ContextualizedLedgerEngine, surplus_engine: CooperativeSurplusEngine):
        self.ledger = ledger_engine
        self.surplus_engine = surplus_engine
        self.funds: Dict[str, CooperativeFund] = {}

    def create_fund(self, fund_id: str, target_objective: ImpactVector) -> CooperativeFund:
        """
        Initializes a new cooperative fund tied to a specific future impact objective.
        """
        fund = CooperativeFund(fund_id=fund_id, target_objective=target_objective)
        self.funds[fund_id] = fund
        return fund

    def allocate_credits_to_fund(self, fund_id: str, agent_id: str, amount: float) -> bool:
        """
        Agents allocate credits from their balance into a shared CooperativeFund.
        Ensures economic traceability by tagging the transaction with the fund objective.
        """
        if fund_id not in self.funds:
            return False
            
        fund = self.funds[fund_id]
        if fund.status != FundStatus.OPEN:
            return False

        # 1. Verify agent balance
        balance = self.ledger.get_agent_balance(agent_id)
        if balance.total_balance < amount:
            return False

        # 2. Debit the agent via ledger
        category = fund.target_objective.category
        
        provenance = CreditProvenance(
            surplus_event_id=f"COOP_FUND_INVESTMENT_{fund_id}"
        )

        debit_entry = CreditEntry(
            entry_id=str(uuid.uuid4()),
            agent_id=agent_id,
            amount=-amount,
            entry_type=EntryType.DEBIT,
            impact_vector=fund.target_objective,
            domain_context=category,
            provenance=provenance,
            metadata={"fund_id": fund_id, "contribution_type": "pooling_debit"}
        )
        
        # Credit the fund account (represented as a specifically scoped system account)
        fund_account = f"FUND_{fund_id}"
        credit_entry = CreditEntry(
            entry_id=str(uuid.uuid4()),
            agent_id=fund_account,
            amount=amount,
            entry_type=EntryType.CREDIT,
            impact_vector=fund.target_objective,
            domain_context=category,
            provenance=provenance,
            metadata={"fund_id": fund_id, "contribution_type": "pooling_credit"}
        )

        # Internal method usage to bypass high-level allocation logic
        self.ledger._add_entry(debit_entry)
        self.ledger._add_entry(credit_entry)

        # 3. Update fund record
        contribution = FundContribution(
            agent_id=agent_id, 
            amount=amount,
            timestamp=time.time()
        )
        fund.contributions.append(contribution)
        
        return True

    def evaluate_investment(self, fund_id: str, cluster_id: str, projections: List[ImpactProjection]) -> InvestmentEvaluation:
        """
        Applies predictive risk modeling to evaluate the expected return on cooperative investment (ROCI).
        Considers surplus projections, uncertainty bounds, and structural risks.
        """
        fund = self.funds.get(fund_id)
        if not fund:
            raise ValueError(f"Fund {fund_id} not found")

        # 1. Calculate predicted surplus for the target cluster
        surplus_pool = self.surplus_engine.calculate_cluster_surplus(cluster_id, projections)
        
        expected_return = surplus_pool.total_surplus
        investment_cost = fund.total_pooled
        
        # 2. Calculate ROCI (Return on Cooperative Investment)
        # Measure value generated per unit of pooled capital
        roci = expected_return / investment_cost if investment_cost > 0 else 0.0
        
        # 3. Risk Score calculation
        # Risk factors: Confidence interval width (low confidence) and dependency risk
        mu = surplus_pool.total_surplus
        low, high = surplus_pool.confidence_interval
        ci_width = high - low
        
        # Normalized confidence risk (0 to 1)
        confidence_risk = min(1.0, ci_width / (mu * 2) if mu > 0 else 1.0)
        
        # Dependency risk (from surplus engine metadata)
        # risk_discount = 1.0 / (1.0 + external_deps * factor)
        # So (1.0 - risk_discount) is a measure of risk from external factors
        dependency_risk = 1.0 - surplus_pool.metadata.get("risk_discount", 1.0)
        
        # Combined Risk Score (Weighted average)
        risk_score = round((confidence_risk * 0.6) + (dependency_risk * 0.4), 4)
        
        # 4. Recommendation logic based on ROCI/Risk profiles
        if roci > 2.5 and risk_score < 0.35:
            rec = "ALLOCATE"
        elif roci > 1.5 and risk_score < 0.6:
            rec = "NEUTRAL"
        else:
            rec = "REJECT"

        return InvestmentEvaluation(
            fund_id=fund_id,
            cluster_id=cluster_id,
            expected_return=round(expected_return, 4),
            investment_cost=round(investment_cost, 4),
            risk_score=risk_score,
            roci=round(roci, 4),
            confidence_interval=surplus_pool.confidence_interval,
            recommendation=rec,
            metadata=surplus_pool.metadata
        )

    def deploy_fund(self, fund_id: str, cluster_id: str, evaluation: InvestmentEvaluation) -> bool:
        """
        Deploys pooled credits into a specific task cluster.
        Marks the fund as deployed and links it to the cluster for impact tracking.
        """
        fund = self.funds.get(fund_id)
        if not fund:
            return False
            
        if fund.status != FundStatus.OPEN:
            return False
            
        # We only deploy if the recommendation is not REJECT
        # In a real system, there might be a threshold override
        if evaluation.recommendation == "REJECT":
            return False

        fund.status = FundStatus.DEPLOYED
        fund.deployed_task_cluster_id = cluster_id
        
        return True
