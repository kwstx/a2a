import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
from src.models.impact import ImpactProjection, ImpactVector, ImpactCategory, SurplusPool
from src.models.agent import Agent
from src.engine.forecasting import ForecastingLayer
from src.engine.surplus import CooperativeSurplusEngine
from src.engine.recalibration import ImpactRecalibrationEngine
from src.models.registry import ImpactMetricRegistry

class TestRecalibration(unittest.TestCase):

    def setUp(self):
        self.registry = ImpactMetricRegistry()
        self.forecasting = ForecastingLayer(self.registry)
        self.surplus = CooperativeSurplusEngine()
        self.recalibration = ImpactRecalibrationEngine(self.forecasting, self.surplus)

    def test_agent_reliability_update(self):
        agent = Agent.create("test_agent", "coder")
        initial_dev = agent.performance.impact_deviation # Default 0.1
        
        # Predicted 100, Actual 80 -> Deviation 0.2
        target = ImpactVector(ImpactCategory.TECHNICAL, 100.0, 10.0, (90, 110))
        projection = ImpactProjection("task_1", target, 100.0, 5.0, (90, 110))
        
        self.recalibration.recalibrate_agent_performance(agent, projection, 80.0)
        
        # New deviation should be blended: 0.8*0.1 + 0.2*0.2 = 0.08 + 0.04 = 0.12
        # Assuming alpha=0.2 in Agent.update_performance
        self.assertAlmostEqual(agent.performance.impact_deviation, 0.12, places=4)

    def test_synergy_scaling_update(self):
        # Initial multiplier 0.15
        self.assertEqual(self.surplus.synergy_multiplier, 0.15)
        
        # Predicted 100, Realized 120 -> +20% deviation
        # CI (90, 110) -> Significant deviation (120 > 110)
        pool = SurplusPool("cluster_1", 100.0, (90, 110), {}, ["task_1"])
        
        self.recalibration.recalibrate_synergy_model(pool, 120.0)
        
        # Should increase multiplier
        self.assertGreater(self.surplus.synergy_multiplier, 0.15)

    def test_forecasting_weights_update(self):
        # TECH -> EFFICIENCY multiplier is 1.2
        rules = self.forecasting.causal_rules[ImpactCategory.TECHNICAL]
        initial_mult = next(r[2] for r in rules if r[0] == ImpactCategory.EFFICIENCY)
        self.assertEqual(initial_mult, 1.2)
        
        # Chain: TECH (100) -> EFFICIENCY (120 predicted)
        # Ratio Actual/Predicted = 150/120 = 1.25 -> +25% higher magnitude
        
        v1 = ImpactVector(ImpactCategory.TECHNICAL, 100.0, 10.0, (90, 110))
        v2 = ImpactVector(ImpactCategory.EFFICIENCY, 120.0, 15.0, (100, 140))
        
        projection = ImpactProjection("task_1", v1, 120.0, 5.0, (110, 130), effect_chain=[v1, v2])
        actual = ImpactVector(ImpactCategory.EFFICIENCY, 150.0, 15.0, (140, 160))
        
        self.recalibration.recalibrate_forecasting_weights(projection, actual)
        
        # Check new multiplier
        new_rules = self.forecasting.causal_rules[ImpactCategory.TECHNICAL]
        new_mult = next(r[2] for r in new_rules if r[0] == ImpactCategory.EFFICIENCY)
        
        # Calculation:
        # mag_ratio = 1.25
        # mult_delta = (1.25 - 1.0) * 0.02 = 0.005
        # new_mult = 1.2 + 0.005 = 1.205
        self.assertAlmostEqual(new_mult, 1.205, places=4)

if __name__ == '__main__':
    unittest.main()
