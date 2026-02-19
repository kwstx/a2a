from typing import List, Dict, Any
from src.models.impact import ImpactProjection
from src.models.agent import Agent

class ValuationEngine:
    """
    Adjusts predicted impact based on agent performance embeddings and trust scores.
    """
    
    def __init__(self, base_multiplier: float = 1.0):
        self.base_multiplier = base_multiplier

    def adjust_projection(self, projection: ImpactProjection, agent: Agent) -> ImpactProjection:
        """
        Applies performance-based weighting to an impact projection.
        Returns a new projection with adjusted mean and confidence intervals.
        """
        domain = projection.metadata.get("domain", "general")
        trust_score = agent.performance.get_trust_score(domain)
        
        # Calculate weight derived from embedding (simplified as trust score here)
        # In a more complex version, we might use cosine similarity to domain-specific performance patterns
        weight = trust_score * self.base_multiplier
        
        # Adjust distribution parameters
        adjusted_mean = round(projection.distribution_mean * weight, 4)
        
        # Uncertainty might increase if trust is low
        uncertainty_expansion = 1.0 + (1.0 - trust_score)
        adjusted_std = round(projection.distribution_std * uncertainty_expansion, 4)
        
        # Recalculate confidence interval
        ci_low = round(adjusted_mean - 1.96 * adjusted_std, 4)
        ci_high = round(adjusted_mean + 1.96 * adjusted_std, 4)
        
        # Create a copy with updated values
        adjusted_projection = ImpactProjection(
            task_id=projection.task_id,
            target_vector=projection.target_vector,
            distribution_mean=adjusted_mean,
            distribution_std=adjusted_std,
            confidence_interval=(ci_low, ci_high),
            effect_chain=projection.effect_chain,
            metadata={
                **projection.metadata,
                "agent_id": agent.id,
                "trust_score": trust_score,
                "adjustment_weight": weight,
                "original_mean": projection.distribution_mean
            }
        )
        
        return adjusted_projection

    def rank_agents_for_task(self, projection: ImpactProjection, agents: List[Agent]) -> List[tuple[Agent, float]]:
        """
        Ranks agents based on their potential weighted impact for a specific projection.
        """
        rankings = []
        for agent in agents:
            adjusted = self.adjust_projection(projection, agent)
            rankings.append((agent, adjusted.distribution_mean))
        
        # Sort by adjusted mean impact descending
        return sorted(rankings, key=lambda x: x[1], reverse=True)
