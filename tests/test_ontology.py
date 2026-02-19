import unittest
from src.models.impact import ImpactVector, ImpactCategory
from src.models.registry import ImpactMetricRegistry, revenue_mapper

class TestImpactOntology(unittest.TestCase):
    def test_impact_vector_creation(self):
        vector = ImpactVector(
            category=ImpactCategory.EFFICIENCY,
            magnitude=10.5,
            time_horizon=7.0,
            uncertainty_bounds=(8.0, 12.0),
            domain_weights={"compute": 1.0}
        )
        self.assertEqual(vector.category, ImpactCategory.EFFICIENCY)
        self.assertEqual(vector.magnitude, 10.5)

    def test_registry_translation(self):
        registry = ImpactMetricRegistry()
        registry.register_metric("revenue_deal", revenue_mapper)
        
        raw_data = {
            "expected_value": 5000,
            "contract_length": 120,
            "min": 4000,
            "max": 6000,
            "is_recurring": True
        }
        
        vector = registry.translate("revenue_deal", raw_data)
        self.assertEqual(vector.category, ImpactCategory.REVENUE)
        self.assertEqual(vector.magnitude, 5000)
        self.assertTrue(vector.metrics["recurring"])

if __name__ == "__main__":
    unittest.main()
