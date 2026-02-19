import unittest
import uuid
from src.models.impact import ImpactVector, ImpactCategory, ImpactProjection
from src.models.ledger import EntryType, CreditProvenance, CreditEntry
from src.models.cooperative_fund import CooperativeFund, FundStatus, InvestmentEvaluation
from src.engine.ledger import ContextualizedLedgerEngine
from src.engine.surplus import CooperativeSurplusEngine
from src.engine.cooperation import CooperativeInvestingEngine

class TestCooperativeInvesting(unittest.TestCase):
    def setUp(self):
        self.ledger = ContextualizedLedgerEngine()
        self.surplus_engine = CooperativeSurplusEngine()
        self.investing_engine = CooperativeInvestingEngine(self.ledger, self.surplus_engine)
        
        # Give some credits to agents
        self.agent_a = "Agent_Research"
        self.agent_b = "Agent_Technical"
        
        # Setup initial balances via system issuance
        cluster_id = "initial_surplus"
        provenance = CreditProvenance(surplus_event_id=cluster_id)
        
        # Agent A: 100 credits
        self.ledger._add_entry(CreditEntry(
            entry_id=str(uuid.uuid4()),
            agent_id=self.agent_a,
            amount=100.0,
            entry_type=EntryType.CREDIT,
            impact_vector=ImpactVector(ImpactCategory.RESEARCH, 100.0, 1.0, (90, 110)),
            domain_context=ImpactCategory.RESEARCH,
            provenance=provenance
        ))
        
        # Agent B: 100 credits
        self.ledger._add_entry(CreditEntry(
            entry_id=str(uuid.uuid4()),
            agent_id=self.agent_b,
            amount=100.0,
            entry_type=EntryType.CREDIT,
            impact_vector=ImpactVector(ImpactCategory.TECHNICAL, 100.0, 1.0, (90, 110)),
            domain_context=ImpactCategory.TECHNICAL,
            provenance=provenance
        ))

    def test_fund_creation_and_pooling(self):
        objective = ImpactVector(
            category=ImpactCategory.ECOSYSTEM,
            magnitude=500.0,
            time_horizon=2.0,
            uncertainty_bounds=(400, 600)
        )
        fund = self.investing_engine.create_fund("eco_boost_2026", objective)
        
        self.assertEqual(fund.fund_id, "eco_boost_2026")
        self.assertEqual(fund.status, FundStatus.OPEN)
        
        # Agent A contributes 40
        success_a = self.investing_engine.allocate_credits_to_fund("eco_boost_2026", self.agent_a, 40.0)
        self.assertTrue(success_a)
        
        # Agent B contributes 60
        success_b = self.investing_engine.allocate_credits_to_fund("eco_boost_2026", self.agent_b, 60.0)
        self.assertTrue(success_b)
        
        # Check balances
        self.assertEqual(self.ledger.get_agent_balance(self.agent_a).total_balance, 60.0)
        self.assertEqual(self.ledger.get_agent_balance(self.agent_b).total_balance, 40.0)
        
        # Check fund balance
        self.assertEqual(fund.total_pooled, 100.0)
        self.assertEqual(self.ledger.get_agent_balance("FUND_eco_boost_2026").total_balance, 100.0)

    def test_insufficient_balance(self):
        objective = ImpactVector(ImpactCategory.RESEARCH, 10.0, 1.0, (9, 11))
        self.investing_engine.create_fund("small_fund", objective)
        
        # Try to contribute more than balance
        success = self.investing_engine.allocate_credits_to_fund("small_fund", self.agent_a, 150.0)
        self.assertFalse(success)
        self.assertEqual(self.ledger.get_agent_balance(self.agent_a).total_balance, 100.0)

    def test_investment_evaluation(self):
        # Create and pool
        objective = ImpactVector(ImpactCategory.TECHNICAL, 1000.0, 1.0, (800, 1200))
        fund = self.investing_engine.create_fund("tech_expansion", objective)
        self.investing_engine.allocate_credits_to_fund("tech_expansion", self.agent_a, 50.0)
        self.investing_engine.allocate_credits_to_fund("tech_expansion", self.agent_b, 50.0)
        
        # Potential cluster to invest in
        cluster_id = "cluster_99"
        projections = [
            ImpactProjection(
                task_id="task_1",
                target_vector=ImpactVector(ImpactCategory.TECHNICAL, 200.0, 1.0, (180, 220)),
                distribution_mean=300.0, # High return
                distribution_std=10.0,   # Low risk
                confidence_interval=(280, 320),
                metadata={"agent_id": "agent_x", "agent_role": "dev"}
            ),
            ImpactProjection(
                task_id="task_2",
                target_vector=ImpactVector(ImpactCategory.TECHNICAL, 200.0, 1.0, (180, 220)),
                distribution_mean=300.0, # High return
                distribution_std=10.0,   # Low risk
                confidence_interval=(280, 320),
                metadata={"agent_id": "agent_y", "agent_role": "qa"}
            )
        ]
        
        evaluation = self.investing_engine.evaluate_investment("tech_expansion", cluster_id, projections)
        
        self.assertEqual(evaluation.fund_id, "tech_expansion")
        self.assertEqual(evaluation.cluster_id, cluster_id)
        self.assertGreater(evaluation.roci, 5.0) # 600+ / 100
        self.assertLess(evaluation.risk_score, 0.3)
        self.assertEqual(evaluation.recommendation, "ALLOCATE")

    def test_fund_deployment(self):
        objective = ImpactVector(ImpactCategory.REVENUE, 100.0, 1.0, (90, 110))
        fund = self.investing_engine.create_fund("revenue_fund", objective)
        self.investing_engine.allocate_credits_to_fund("revenue_fund", self.agent_a, 10.0)
        
        # Mock evaluation
        eval_allocate = InvestmentEvaluation(
            fund_id="revenue_fund",
            cluster_id="cluster_1",
            expected_return=50.0,
            investment_cost=10.0,
            risk_score=0.1,
            roci=5.0,
            confidence_interval=(45, 55),
            recommendation="ALLOCATE"
        )
        
        success = self.investing_engine.deploy_fund("revenue_fund", "cluster_1", eval_allocate)
        self.assertTrue(success)
        self.assertEqual(fund.status, FundStatus.DEPLOYED)
        self.assertEqual(fund.deployed_task_cluster_id, "cluster_1")
        
        # Try to deploy already deployed fund
        success_again = self.investing_engine.deploy_fund("revenue_fund", "cluster_2", eval_allocate)
        self.assertFalse(success_again)

    def test_deployment_rejection(self):
        objective = ImpactVector(ImpactCategory.REVENUE, 100.0, 1.0, (90, 110))
        fund = self.investing_engine.create_fund("risky_fund", objective)
        self.investing_engine.allocate_credits_to_fund("risky_fund", self.agent_a, 10.0)
        
        eval_reject = InvestmentEvaluation(
            fund_id="risky_fund",
            cluster_id="cluster_bad",
            expected_return=5.0,
            investment_cost=10.0,
            risk_score=0.9,
            roci=0.5,
            confidence_interval=(-10, 20),
            recommendation="REJECT"
        )
        
        success = self.investing_engine.deploy_fund("risky_fund", "cluster_bad", eval_reject)
        self.assertFalse(success)
        self.assertEqual(fund.status, FundStatus.OPEN)

if __name__ == "__main__":
    unittest.main()
