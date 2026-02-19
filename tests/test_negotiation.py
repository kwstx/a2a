import unittest
from src.models.impact import SurplusPool, ContributionClaim
from src.engine.negotiation import AutonomousNegotiationEngine

class TestNegotiationEngine(unittest.TestCase):
    def setUp(self):
        self.engine = AutonomousNegotiationEngine(max_iterations=100, equilibrium_tolerance=1e-7, fairness_weight=0.3)

    def test_balanced_negotiation(self):
        # Pool with 1000 surplus
        pool = SurplusPool(
            cluster_id="c1",
            total_surplus=1000.0,
            confidence_interval=(900, 1100),
            aggregated_vectors={},
            task_ids=["t1", "t2"]
        )
        
        # Agent A: High contribution, low leverage, high uncertainty
        claim_a = ContributionClaim(
            agent_id="agent-A",
            cluster_id="c1",
            marginal_impact_estimate=800.0,
            uncertainty_margin=200.0,
            dependency_influence_weight=0.1,
            task_ids=["t1"]
        )
        
        # Agent B: Low contribution, high leverage, low uncertainty
        claim_b = ContributionClaim(
            agent_id="agent-B",
            cluster_id="c1",
            marginal_impact_estimate=200.0,
            uncertainty_margin=10.0,
            dependency_influence_weight=0.9,
            task_ids=["t2"]
        )
        
        result = self.engine.negotiate_splits(pool, [claim_a, claim_b])
        
        self.assertTrue(result["converged"])
        self.assertGreater(result["rounds_to_convergence"], 0)
        
        allocations = result["final_allocations"]
        # With fairness weight 0.3:
        # Agent A target: 0.7 * 800 + 0.3 * 500 = 560 + 150 = 710
        # Agent B target: 0.7 * 200 + 0.3 * 500 = 140 + 150 = 290
        
        self.assertAlmostEqual(allocations["agent-A"], 710.0, delta=1.0)
        self.assertAlmostEqual(allocations["agent-B"], 290.0, delta=1.0)
        self.assertAlmostEqual(sum(allocations.values()), 1000.0, delta=0.1)

    def test_high_fairness_constraint(self):
        self.engine.fairness_weight = 0.9 # Almost pure egalitarian
        
        pool = SurplusPool(cluster_id="c2", total_surplus=100.0, confidence_interval=(90, 110), aggregated_vectors={}, task_ids=["t1", "t2"])
        
        claim_a = ContributionClaim(agent_id="A", cluster_id="c2", marginal_impact_estimate=90.0, uncertainty_margin=5.0, dependency_influence_weight=0.5, task_ids=["t1"])
        claim_b = ContributionClaim(agent_id="B", cluster_id="c2", marginal_impact_estimate=10.0, uncertainty_margin=5.0, dependency_influence_weight=0.5, task_ids=["t2"])
        
        result = self.engine.negotiate_splits(pool, [claim_a, claim_b])
        allocations = result["final_allocations"]
        
        # Targets:
        # A: 0.1 * 90 + 0.9 * 50 = 9 + 45 = 54
        # B: 0.1 * 10 + 0.9 * 50 = 1 + 45 = 46
        
        self.assertAlmostEqual(allocations["A"], 54.0, delta=1.0)
        self.assertAlmostEqual(allocations["B"], 46.0, delta=1.0)

    def test_zero_surplus(self):
        pool = SurplusPool(cluster_id="c0", total_surplus=0.0, confidence_interval=(0, 0), aggregated_vectors={}, task_ids=[])
        result = self.engine.negotiate_splits(pool, [])
        self.assertEqual(result["final_allocations"], {})

if __name__ == "__main__":
    unittest.main()
