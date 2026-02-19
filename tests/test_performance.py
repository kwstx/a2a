import unittest
from src.models.agent import Agent, PerformanceSignature
from src.models.impact import ImpactProjection, ImpactVector, ImpactCategory
from src.engine.valuation import ValuationEngine

class TestPerformanceEmbeddings(unittest.TestCase):
    def setUp(self):
        self.valuation_engine = ValuationEngine()
        
        # High performance agent
        self.star_agent = Agent.create("agent-001", "specialist")
        self.star_agent.performance.prediction_accuracy = 0.95
        self.star_agent.performance.impact_deviation = 0.05
        self.star_agent.performance.domain_reliability = {"technical": 0.9}
        
        # Low performance agent
        self.rookie_agent = Agent.create("agent-002", "generalist")
        self.rookie_agent.performance.prediction_accuracy = 0.6
        self.rookie_agent.performance.impact_deviation = 0.3
        self.rookie_agent.performance.domain_reliability = {"technical": 0.4}

    def test_embedding_generation(self):
        embedding = self.star_agent.performance.generate_embedding()
        self.assertEqual(len(embedding), 5)
        self.assertGreater(embedding[0], 0.9) # accuracy
        self.assertLess(embedding[1], 0.1)    # deviation

    def test_trust_score(self):
        star_score = self.star_agent.performance.get_trust_score("technical")
        rookie_score = self.rookie_agent.performance.get_trust_score("technical")
        
        self.assertGreater(star_score, rookie_score)
        self.assertGreater(star_score, 0.8)
        self.assertLess(rookie_score, 0.6)

    def test_valuation_adjustment(self):
        # Create a mock projection
        vector = ImpactVector(ImpactCategory.TECHNICAL, 100.0, 30.0, (90, 110))
        projection = ImpactProjection(
            task_id="task-test",
            target_vector=vector,
            distribution_mean=100.0,
            distribution_std=10.0,
            confidence_interval=(80.0, 120.0),
            metadata={"domain": "technical"}
        )
        
        star_adjusted = self.valuation_engine.adjust_projection(projection, self.star_agent)
        rookie_adjusted = self.valuation_engine.adjust_projection(projection, self.rookie_agent)
        
        # Star agent should have higher adjusted impact than rookie
        self.assertGreater(star_adjusted.distribution_mean, rookie_adjusted.distribution_mean)
        
        # Star agent should have lower uncertainty (std) than rookie
        self.assertLess(star_adjusted.distribution_std, rookie_adjusted.distribution_std)
        
        print(f"\nStar Agent Adjusted Mean: {star_adjusted.distribution_mean}")
        print(f"Rookie Agent Adjusted Mean: {rookie_adjusted.distribution_mean}")

    def test_agent_ranking(self):
        vector = ImpactVector(ImpactCategory.TECHNICAL, 100.0, 30.0, (90, 110))
        projection = ImpactProjection(
            task_id="task-test",
            target_vector=vector,
            distribution_mean=100.0,
            distribution_std=10.0,
            confidence_interval=(80.0, 120.0),
            metadata={"domain": "technical"}
        )
        
        rankings = self.valuation_engine.rank_agents_for_task(projection, [self.star_agent, self.rookie_agent])
        
        self.assertEqual(rankings[0][0].id, "agent-001")
        self.assertEqual(rankings[1][0].id, "agent-002")

    def test_performance_accumulation(self):
        agent = Agent.create("agent-003", "learner")
        initial_trust = agent.performance.get_trust_score("technical")
        
        # Simulate perfect performance
        agent.performance.update_performance(100.0, 100.0, False, "technical")
        agent.performance.update_performance(100.0, 100.0, False, "technical")
        
        new_trust = agent.performance.get_trust_score("technical")
        self.assertGreater(new_trust, initial_trust)
        
        # Simulate bad performance
        agent.performance.update_performance(100.0, 50.0, False, "technical")
        bad_trust = agent.performance.get_trust_score("technical")
        self.assertLess(bad_trust, new_trust)

if __name__ == "__main__":

    unittest.main()
