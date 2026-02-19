import time
import math
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, field
from src.models.impact import ContributionClaim, SurplusPool

@dataclass
class NegotiationRound:
    """
    Metadata for a single round of the bargaining loop.
    """
    round_number: int
    allocations: Dict[str, float] # agent_id -> allocated_amount
    deviation_from_objective: float
    allocation_variance: float
    timestamp: float = field(default_factory=lambda: time.time())

class AutonomousNegotiationEngine:
    """
    Implements the autonomous cooperative negotiation protocol (Step Seven).
    Agents exchange ContributionClaim objects within a bounded iterative bargaining loop.
    Each round minimizes deviation from an optimization objective that balances 
    marginal contribution alignment with fairness constraints.
    """
    
    def __init__(self, 
                 max_iterations: int = 50, 
                 equilibrium_tolerance: float = 1e-6, 
                 fairness_weight: float = 0.3):
        """
        Initialize the negotiation engine.
        
        :param max_iterations: Maximum number of rounds in the bargaining loop.
        :param equilibrium_tolerance: Convergence threshold for allocation variance.
        :param fairness_weight: Weight assigned to fairness/egalitarian constraints (0.0 to 1.0).
        """
        self.max_iterations = max_iterations
        self.equilibrium_tolerance = equilibrium_tolerance
        self.fairness_weight = fairness_weight

    def _calculate_variance(self, values: List[float]) -> float:
        """Helper to calculate variance without external libraries."""
        if not values:
            return 0.0
        n = len(values)
        mean = sum(values) / n
        return sum((x - mean) ** 2 for x in values) / n

    def negotiate_splits(self, pool: SurplusPool, claims: List[ContributionClaim]) -> Dict[str, Any]:
        """
        Runs the iterative cooperative bargaining loop to reach an allocation consensus.
        
        The objective function J minimizes deviation from a target that balances
        marginal contribution (efficiency) and equal distribution (fairness).
        Convergence is reached when the variance of the allocations stabilizes.
        """
        if not claims or pool.total_surplus <= 0:
            return {
                "cluster_id": pool.cluster_id,
                "total_surplus": pool.total_surplus,
                "final_allocations": {},
                "rounds_to_convergence": 0,
                "converged": True,
                "history": []
            }

        num_agents = len(claims)
        total_surplus = pool.total_surplus
        agent_ids = [c.agent_id for c in claims]
        
        # 1. Define Optimization Objective Component: Marginal Contribution Alignment
        marginal_impacts = [c.marginal_impact_estimate for c in claims]
        sum_marginal = sum(marginal_impacts)
        
        if sum_marginal > 0:
            marginal_targets = [(m / sum_marginal) * total_surplus for m in marginal_impacts]
        else:
            # Fallback to egalitarian if no marginal contribution can be measured
            marginal_targets = [total_surplus / num_agents] * num_agents
            
        # 2. Define Optimization Objective Component: Fairness Constraints
        # Pure egalitarian distribution for fairness baseline
        fairness_target = total_surplus / num_agents
        
        # 3. Compute Composite Optimization Objective Target
        # This is the "Equilibrium Point" the system gravitates towards
        objective_targets = [
            (1 - self.fairness_weight) * mt + self.fairness_weight * fairness_target
            for mt in marginal_targets
        ]
        
        # Initial state: Start from a neutral uniform distribution
        current_allocations = [total_surplus / num_agents] * num_agents
        
        history = []
        prev_variance = self._calculate_variance(current_allocations)
        converged = False
        
        # 4. Iterative Bargaining Loop
        for iteration in range(self.max_iterations):
            new_allocations = []
            
            for i in range(num_agents):
                claim = claims[i]
                target = objective_targets[i]
                current = current_allocations[i]
                
                # Agents adjust their stance based on their ContributionClaim metadata
                
                # Leverage Score: High dependency influence increases resistance to change
                leverage = claim.dependency_influence_weight
                
                # Flex Score: Higher uncertainty makes an agent more willing to yield 
                # to the collective objective
                flexibility = min(1.0, claim.uncertainty_margin / (claim.marginal_impact_estimate + 1e-9))
                
                # Negotiated adjustment velocity
                # (How fast the agent moves towards the collective objective target)
                base_velocity = 0.25
                velocity = base_velocity * (1.0 + flexibility - 0.4 * leverage)
                velocity = max(0.05, min(0.6, velocity))
                
                # Update step
                adjustment = velocity * (target - current)
                new_allocations.append(current + adjustment)
            
            # 5. Global Normalization (Closed System Constraint)
            # Ensures total allocated always equals total pool surplus
            sum_new = sum(new_allocations)
            if sum_new > 0:
                current_allocations = [(val / sum_new) * total_surplus for val in new_allocations]
            else:
                current_allocations = [total_surplus / num_agents] * num_agents
                
            # Compute Round Metrics
            current_variance = self._calculate_variance(current_allocations)
            deviation = sum(abs(current_allocations[i] - objective_targets[i]) for i in range(num_agents))
            
            history.append(NegotiationRound(
                round_number=iteration,
                allocations={agent_ids[i]: round(current_allocations[i], 4) for i in range(num_agents)},
                deviation_from_objective=round(deviation, 4),
                allocation_variance=round(current_variance, 4)
            ))
            
            # 6. Convergence Check (Stabilization of variance or reaching target)
            if iteration > 0 and (abs(current_variance - prev_variance) < self.equilibrium_tolerance or deviation < self.equilibrium_tolerance):
                converged = True
                break
                
            prev_variance = current_variance

        return {
            "cluster_id": pool.cluster_id,
            "total_surplus": total_surplus,
            "final_allocations": {agent_ids[i]: round(current_allocations[i], 4) for i in range(num_agents)},
            "rounds_to_convergence": len(history),
            "converged": converged,
            "final_deviation": round(sum(abs(current_allocations[i] - objective_targets[i]) for i in range(num_agents)), 4),
            "history": history,
            "objective_targets": {agent_ids[i]: round(objective_targets[i], 4) for i in range(num_agents)}
        }
