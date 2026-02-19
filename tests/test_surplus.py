import unittest
from src.models.impact import ImpactProjection, ImpactVector, ImpactCategory
from src.engine.surplus import CooperativeSurplusEngine

class TestSurplusEngine(unittest.TestCase):
    def setUp(self):
        self.engine = CooperativeSurplusEngine(synergy_multiplier=0.2, dependency_risk_factor=0.1)

    def test_empty_cluster(self):
        pool = self.engine.calculate_cluster_surplus("cluster-0", [])
        self.assertEqual(pool.total_surplus, 0.0)
        self.assertEqual(pool.task_ids, [])

    def test_single_task_no_deps(self):
        v = ImpactVector(category=ImpactCategory.TECHNICAL, magnitude=10.0, time_horizon=30.0, uncertainty_bounds=(9, 11))
        p = ImpactProjection(
            task_id="t1", 
            target_vector=v, 
            distribution_mean=100.0, 
            distribution_std=10.0, 
            confidence_interval=(80.4, 119.6)
        )
        
        pool = self.engine.calculate_cluster_surplus("cluster-1", [p])
        
        # Base surplus should be 100.0, no deps means multiplier 1.0
        self.assertEqual(pool.total_surplus, 100.0)
        self.assertEqual(pool.aggregated_vectors["TECHNICAL"], 10.0)

    def test_synergy_and_risk(self):
        # Task 1: No deps, Role: RESEARCHER
        v1 = ImpactVector(category=ImpactCategory.TECHNICAL, magnitude=10.0, time_horizon=30.0, uncertainty_bounds=(9, 11))
        p1 = ImpactProjection(
            task_id="t1", 
            target_vector=v1, 
            distribution_mean=100.0, 
            distribution_std=10.0, 
            confidence_interval=(80.4, 119.6),
            metadata={"agent_role": "researcher"}
        )
        
        # Task 2: Depends on T1 (internal), Role: CODER
        v2 = ImpactVector(
            category=ImpactCategory.EFFICIENCY, 
            magnitude=5.0, 
            time_horizon=30.0, 
            uncertainty_bounds=(4, 6),
            causal_dependencies=["t1"]
        )
        p2 = ImpactProjection(
            task_id="t2", 
            target_vector=v2, 
            distribution_mean=50.0, 
            distribution_std=5.0, 
            confidence_interval=(40.2, 59.8),
            metadata={"agent_role": "coder"}
        )
        
        # Task 3: Depends on external task (risk), Role: ANALYST
        v3 = ImpactVector(
            category=ImpactCategory.REVENUE, 
            magnitude=20.0, 
            time_horizon=60.0, 
            uncertainty_bounds=(18, 22),
            causal_dependencies=["ext-001"]
        )
        p3 = ImpactProjection(
            task_id="t3", 
            target_vector=v3, 
            distribution_mean=200.0, 
            distribution_std=20.0, 
            confidence_interval=(160.8, 239.2),
            metadata={"agent_role": "analyst"}
        )
        
        projections = [p1, p2, p3]
        pool = self.engine.calculate_cluster_surplus("cluster-multi", projections)
        
        # Manual calculation for verification:
        # Base surplus = 350
        # internal_deps = 1
        # roles = {researcher, coder, analyst} -> len=3
        # diversity = 3/3 = 1.0
        # interdependence = 1/3 = 0.3333
        # density = 1/(3*2) = 0.1667
        # multiplier = 0.2
        
        # base_synergy = (1.0 ** 0.4) * (1.0 + 0.3333 ** 1.2) = 1.0 * (1.0 + 0.2681) = 1.2681
        # exponential_boost = exp(0.1667 * 0.2 * 4) = exp(0.13336) = 1.1427
        # synergy_bonus = 1.2681 * 1.1427 = 1.4491
        
        # external_deps = 1
        # risk_discount = 1.0 / (1.0 + 1 * 0.1) = 0.9091
        # total_surplus = 350 * 1.4491 * 0.9091 = 461.0772
        
        self.assertAlmostEqual(pool.total_surplus, 460.8545, places=3)
        self.assertEqual(pool.metadata["synergy_metrics"]["diversity"], 1.0)
        self.assertEqual(pool.metadata["synergy_metrics"]["density"], 0.1667)
        self.assertIn("TECHNICAL", pool.aggregated_vectors)
        self.assertIn("EFFICIENCY", pool.aggregated_vectors)
        self.assertIn("REVENUE", pool.aggregated_vectors)
        self.assertEqual(pool.aggregated_vectors["TECHNICAL"], 10.0)
        self.assertEqual(pool.aggregated_vectors["EFFICIENCY"], 5.0)
        self.assertEqual(pool.aggregated_vectors["REVENUE"], 20.0)

    def test_superlinear_synergy(self):
        # A highly integrated cluster of 4 tasks with 4 unique roles
        # Density will be higher
        projections = []
        roles = ["researcher", "coder", "analyst", "architect"]
        for i in range(4):
            v = ImpactVector(
                category=ImpactCategory.TECHNICAL, 
                magnitude=10.0, 
                time_horizon=30.0, 
                uncertainty_bounds=(9, 11),
                causal_dependencies=[f"t{j}" for j in range(i)] # Each depends on all previous
            )
            p = ImpactProjection(
                task_id=f"t{i}", 
                target_vector=v, 
                distribution_mean=100.0, 
                distribution_std=10.0, 
                confidence_interval=(80.4, 119.6),
                metadata={"agent_role": roles[i]}
            )
            projections.append(p)
            
        # Total tasks = 4
        # Internal deps = 0+1+2+3 = 6
        # Max deps = 4*3 = 12
        # Density = 0.5
        # Interdependence = 6/4 = 1.5
        # Diversity = 1.0
        
        pool = self.engine.calculate_cluster_surplus("high-synergy", projections)
        
        # Base surplus = 400
        # synergy_bonus should be high due to exponential term
        # base_synergy = (1.0**0.4) * (1.0 + 1.5**1.2) = 1.0 * (1.0 + 1.626) = 2.626
        # exp_boost = exp(0.5 * 0.2 * 4) = exp(0.4) = 1.4918
        # synergy_bonus = 2.626 * 1.4918 = 3.9175
        # total_surplus = 400 * 3.9175 = 1567.0
        
        self.assertGreater(pool.total_surplus, 1000.0)
        self.assertEqual(pool.metadata["synergy_metrics"]["density"], 0.5)
        self.assertEqual(pool.metadata["synergy_metrics"]["diversity"], 1.0)
        print(f"High synergy surplus: {pool.total_surplus}")

if __name__ == "__main__":
    unittest.main()
