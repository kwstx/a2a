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
        # Task 1: No deps
        v1 = ImpactVector(category=ImpactCategory.TECHNICAL, magnitude=10.0, time_horizon=30.0, uncertainty_bounds=(9, 11))
        p1 = ImpactProjection(task_id="t1", target_vector=v1, distribution_mean=100.0, distribution_std=10.0, confidence_interval=(80.4, 119.6))
        
        # Task 2: Depends on T1 (internal)
        v2 = ImpactVector(
            category=ImpactCategory.EFFICIENCY, 
            magnitude=5.0, 
            time_horizon=30.0, 
            uncertainty_bounds=(4, 6),
            causal_dependencies=["t1"]
        )
        p2 = ImpactProjection(task_id="t2", target_vector=v2, distribution_mean=50.0, distribution_std=5.0, confidence_interval=(40.2, 59.8))
        
        # Task 3: Depends on external task (risk)
        v3 = ImpactVector(
            category=ImpactCategory.REVENUE, 
            magnitude=20.0, 
            time_horizon=60.0, 
            uncertainty_bounds=(18, 22),
            causal_dependencies=["ext-001"]
        )
        p3 = ImpactProjection(task_id="t3", target_vector=v3, distribution_mean=200.0, distribution_std=20.0, confidence_interval=(160.8, 239.2))
        
        projections = [p1, p2, p3]
        pool = self.engine.calculate_cluster_surplus("cluster-multi", projections)
        
        # Base surplus = 100 + 50 + 200 = 350
        # Internal deps = 1 (t2 -> t1)
        # Num tasks = 3 -> Max possible internal deps = 3 * 2 = 6
        # Synergy density = 1/6 = 0.1666...
        # Synergy multiplier = 0.2
        # Synergy bonus = 1.0 + (1 * 0.2) * (1 + 0.1666...) = 1.233333
        # External deps = 1 (t3 -> ext-001)
        # Risk discount = 1.0 / (1.0 + 1 * 0.1) = 1.0 / 1.1 = 0.90909
        # Expected surplus = 350 * 1.233333 / 1.1 = 392.4242...
        
        self.assertAlmostEqual(pool.total_surplus, 392.4242, places=4)
        self.assertIn("TECHNICAL", pool.aggregated_vectors)
        self.assertIn("EFFICIENCY", pool.aggregated_vectors)
        self.assertIn("REVENUE", pool.aggregated_vectors)
        self.assertEqual(pool.aggregated_vectors["TECHNICAL"], 10.0)
        self.assertEqual(pool.aggregated_vectors["EFFICIENCY"], 5.0)
        self.assertEqual(pool.aggregated_vectors["REVENUE"], 20.0)

if __name__ == "__main__":
    unittest.main()
