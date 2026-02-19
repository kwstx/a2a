import unittest
from src.models.impact import ImpactVector, ImpactCategory, ImpactProjection
from src.models.registry import ImpactMetricRegistry, revenue_mapper, research_mapper
from src.models.task import Task
from src.engine.forecasting import ForecastingLayer

class TestForecastingPipeline(unittest.TestCase):
    def setUp(self):
        self.registry = ImpactMetricRegistry()
        self.registry.register_metric("revenue_deal", revenue_mapper)
        self.registry.register_metric("research_project", research_mapper)
        self.forecaster = ForecastingLayer(self.registry)

    def test_pipeline_transformation(self):
        task = Task(
            id="task-001",
            domain="revenue_deal",
            metrics={
                "expected_value": 1000,
                "contract_length": 60,
                "min": 800,
                "max": 1200,
                "is_recurring": False
            }
        )
        
        projection = self.forecaster.project(task)
        
        self.assertEqual(projection.task_id, "task-001")
        self.assertEqual(projection.target_vector.category, ImpactCategory.REVENUE)
        self.assertEqual(projection.target_vector.magnitude, 1000)
        
        # Verify distribution properties
        self.assertGreater(projection.distribution_mean, 0)
        self.assertGreater(projection.distribution_std, 0)
        self.assertEqual(len(projection.confidence_interval), 2)
        
    def test_serialization_roundtrip(self):
        task = Task(id="t1", domain="revenue_deal", metrics={"expected_value": 100})
        projection = self.forecaster.project(task)
        
        # to_dict -> from_dict
        data = projection.to_dict()
        reconstructed = ImpactProjection.from_dict(data)
        
        self.assertEqual(projection.task_id, reconstructed.task_id)
        self.assertEqual(projection.distribution_mean, reconstructed.distribution_mean)
        self.assertEqual(projection.target_vector.category, reconstructed.target_vector.category)
        self.assertEqual(len(projection.effect_chain), len(reconstructed.effect_chain))

    def test_causal_chain(self):
        # Research often leads to Technical -> Efficiency -> Revenue
        task = Task(
            id="task-002",
            domain="research_project",
            metrics={
                "novelty_score": 5.0,
                "est_citations": 100
            }
        )
        
        projection = self.forecaster.project(task)
        
        # Check if we have multiple orders of effects
        categories_in_chain = [v.category for v in projection.effect_chain]
        self.assertIn(ImpactCategory.RESEARCH, categories_in_chain)
        self.assertTrue(len(projection.effect_chain) >= 1)


if __name__ == "__main__":
    unittest.main()
