from dataclasses import dataclass, field
from typing import Dict, List, Any, Tuple
import math

@dataclass
class PerformanceSignature:
    """
    Accumulated performance metadata for an agent.
    Used to generate embeddings for dynamic valuation weighting.
    """
    agent_id: str
    # Accuracy of the agent's own internal predictions (0.0 to 1.0)
    prediction_accuracy: float = 0.8
    # Average deviation from predicted impact in past tasks (lower is better)
    impact_deviation: float = 0.1
    # Ratio of collaborative tasks to total tasks
    collaboration_density: float = 0.5
    # Domain-specific reliability scores (domain -> score 0.0 to 1.0)
    domain_reliability: Dict[str, float] = field(default_factory=dict)
    # Total tasks completed
    task_count: int = 0

    def get_trust_score(self, domain: str) -> float:
        """
        Calculates an empirical trust score based on the performance signature.
        """
        base_reliability = self.domain_reliability.get(domain, 0.5)
        # Weighted average of metrics
        # Accuracy and domain reliability are positive, deviation is negative
        score = (
            self.prediction_accuracy * 0.3 +
            (1.0 - min(self.impact_deviation, 1.0)) * 0.3 +
            base_reliability * 0.4
        )
        return round(score, 4)

    def update_performance(self, predicted_impact: float, actual_impact: float, is_collaboration: bool, domain: str):
        """
        Updates the signature based on a completed task outcome.
        """
        # Calculate deviation for this task
        if predicted_impact > 0:
            deviation = abs(actual_impact - predicted_impact) / predicted_impact
        else:
            deviation = 0.0
            
        # Moving average for impact deviation (simple smoothing)
        alpha = 0.2
        self.impact_deviation = (1 - alpha) * self.impact_deviation + alpha * deviation
        
        # Update task count and collaboration density
        self.task_count += 1
        collab_val = 1.0 if is_collaboration else 0.0
        self.collaboration_density = (1 - alpha) * self.collaboration_density + alpha * collab_val
        
        # Update domain reliability
        current_rel = self.domain_reliability.get(domain, 0.5)
        # Success is defined as staying within a reasonable deviation
        success = 1.0 if deviation < 0.2 else (1.0 - min(deviation, 1.0))
        self.domain_reliability[domain] = (1 - alpha) * current_rel + alpha * success
        
        # Update prediction accuracy (how well they predicted their own success if applicable)
        # For now, we use a simple heuristic: high success increases accuracy
        self.prediction_accuracy = (1 - alpha) * self.prediction_accuracy + alpha * (1.0 if deviation < 0.1 else 0.5)

    def generate_embedding(self) -> List[float]:

        """
        Generates a vector representation of the agent's performance.
        Includes global metrics and an average of domain reliabilities.
        """
        avg_reliability = sum(self.domain_reliability.values()) / len(self.domain_reliability) if self.domain_reliability else 0.5
        
        return [
            self.prediction_accuracy,
            self.impact_deviation,
            self.collaboration_density,
            avg_reliability,
            math.tanh(self.task_count / 100.0) # Normalized task volume
        ]

@dataclass
class Agent:
    """
    Represents an intelligent agent in the economic engine.
    """
    id: str
    role_label: str # Static label (e.g., "coder", "researcher")
    performance: PerformanceSignature

    @classmethod
    def create(cls, agent_id: str, role: str) -> "Agent":
        return cls(
            id=agent_id,
            role_label=role,
            performance=PerformanceSignature(agent_id=agent_id)
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "role_label": self.role_label,
            "performance": {
                "prediction_accuracy": self.performance.prediction_accuracy,
                "impact_deviation": self.performance.impact_deviation,
                "collaboration_density": self.performance.collaboration_density,
                "domain_reliability": self.performance.domain_reliability,
                "task_count": self.performance.task_count
            }
        }
