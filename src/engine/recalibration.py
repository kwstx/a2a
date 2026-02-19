from typing import List, Dict, Any, Optional, Tuple
import math

from src.models.impact import ImpactProjection, ImpactVector, ImpactCategory, SurplusPool
from src.models.agent import Agent
from src.engine.forecasting import ForecastingLayer
from src.engine.surplus import CooperativeSurplusEngine

class ImpactRecalibrationEngine:
    """
    Adaptive post-impact recalibration engine.
    Updates agent reliability coefficients, synergy scaling parameters, and forecasting weights
    based on the deviation between predicted impact distributions and realized outcomes.
    """
    def __init__(self, 
                 forecasting_layer: ForecastingLayer, 
                 surplus_engine: CooperativeSurplusEngine):
        self.forecasting_layer = forecasting_layer
        self.surplus_engine = surplus_engine

    def recalibrate_agent_performance(self, 
                                      agent: Agent, 
                                      projection: ImpactProjection, 
                                      actual_impact: float,
                                      is_collaboration: bool = False) -> Dict[str, float]:
        """
        Updates an agent's reliability coefficients based on prediction deviation.
        Returns the updated metrics for logging.
        """
        # Use the mean of the distribution as the target prediction
        predicted = projection.distribution_mean
        domain = projection.target_vector.category.name
        
        # Helper to calculate deviation before update
        initial_deviation = agent.performance.impact_deviation
        
        # Update the agent's internal performance signature
        agent.performance.update_performance(
            predicted_impact=predicted,
            actual_impact=actual_impact,
            is_collaboration=is_collaboration,
            domain=domain
        )
        
        return {
            "agent_id": agent.id,
            "previous_deviation": initial_deviation,
            "new_deviation": agent.performance.impact_deviation,
            "new_reliability": agent.performance.domain_reliability.get(domain, 0.5)
        }

    def recalibrate_synergy_model(self, 
                                  surplus_pool: SurplusPool, 
                                  realized_surplus: float) -> Dict[str, float]:
        """
        Updates the global synergy scaling parameters if the realized surplus
        deviates significantly from the predicted surplus pool.
        """
        predicted = surplus_pool.total_surplus
        if predicted == 0:
            return {}

        deviation_ratio = (realized_surplus - predicted) / predicted
        
        # Sensitivity factors
        # We want gradual adjustment so we don't oscillate wildly
        learning_rate = 0.05 
        
        updates = {}

        # 1. Update Synergy Multiplier
        # If realized > predicted, we underestimated synergy (or overestimated risk)
        # We'll attribute a portion of the deviation to synergy scaling.
        # We only adjust if the deviation is outside the confidence interval, 
        # or we can just use the raw deviation ratio with a dampener.
        
        ci_low, ci_high = surplus_pool.confidence_interval
        significant_deviation = (realized_surplus < ci_low) or (realized_surplus > ci_high)
        
        if significant_deviation:
            # If we underestimated (realized > high), boost synergy
            # If we overestimated (realized < low), reduce synergy
            
            # Simple gradient step
            synergy_delta = deviation_ratio * learning_rate * 0.5 # Conservative step
            
            # Update the engine's global parameters
            self.surplus_engine.update_synergy_parameters(
                synergy_multiplier_delta=synergy_delta * 0.5, # Reduced global impact
                dependency_risk_delta=0.0 
            )
            updates["synergy_multiplier_delta"] = synergy_delta * 0.5

            # Update specific structural pattern modifier
            pattern_key_list = surplus_pool.metadata.get("pattern_key")
            if pattern_key_list:
                pattern_key = tuple(pattern_key_list)
                # Pattern evolution: faster learning for specific structures
                pattern_delta = deviation_ratio * learning_rate * 2.0
                self.surplus_engine.update_pattern_modifier(pattern_key, pattern_delta)
                updates["pattern_modifier_delta"] = pattern_delta
                updates["pattern_key"] = str(pattern_key)

        # 2. Update Risk Factors (Dependency Risk)
        # If we overestimated surplus (realized < predicted), it might be due to 
        # underestimating the risk of external dependencies.
        internal_deps = surplus_pool.metadata.get("internal_dependencies", 0)
        external_deps = surplus_pool.metadata.get("external_dependencies", 0)
        
        if external_deps > 0 and realized_surplus < predicted:
            # We fell short, and there were external dependencies. 
            # We should increase the penalty (risk factor) for external deps.
            risk_delta = abs(deviation_ratio) * learning_rate * 0.2
            
            self.surplus_engine.update_synergy_parameters(
                synergy_multiplier_delta=0.0,
                dependency_risk_delta=risk_delta
            )
            updates["risk_factor_delta"] = risk_delta
            
        elif external_deps > 0 and realized_surplus > predicted:
            # We overperformed, maybe we were too harsh on external deps.
            risk_delta = -1 * abs(deviation_ratio) * learning_rate * 0.1
             
            self.surplus_engine.update_synergy_parameters(
                synergy_multiplier_delta=0.0,
                dependency_risk_delta=risk_delta
            )
            updates["risk_factor_delta"] = risk_delta

        return updates

    def recalibrate_forecasting_weights(self, 
                                        projection: ImpactProjection, 
                                        actual_outcome: ImpactVector) -> List[str]:
        """
        Updates forecasting causal rules based on the deviation.
        Focuses on the primary causal chain.
        """
        # We need to look at the effect chain to see what transitions were predicted
        chain = projection.effect_chain
        if not chain or len(chain) < 2:
            return []

        updates_made = []
        learning_rate = 0.02
        
        # Heuristic: Compare the magnitude prediction of the chain vs reality
        # If the actual magnitude > predicted magnitude for the SAME category,
        # boost the multiplier for the dominant transition.
        
        # We assume the last element in the chain represents the final state/category
        predicted_final = chain[-1]
        
        if actual_outcome.category == predicted_final.category:
            # Categories match, check magnitude
            mag_ratio = actual_outcome.magnitude / predicted_final.magnitude if predicted_final.magnitude > 0 else 1.0
            
            if abs(mag_ratio - 1.0) > 0.1: # 10% tolerance
                # Adjust multiplier
                mult_delta = (mag_ratio - 1.0) * learning_rate
                
                # Apply to the transition leading to this node
                # We need to find the parent node in the chain
                # The chain is linear here: [0] -> [1] -> [2]
                if len(chain) >= 2:
                    parent = chain[-2]
                    child = chain[-1]
                    
                    self.forecasting_layer.update_causal_rule(
                        source_category=parent.category,
                        target_category=child.category,
                        probability_delta=0.0, # Reliability of occurrence
                        multiplier_delta=mult_delta  # Magnitude scaling
                    )
                    updates_made.append(f"Updated multiplier {parent.category.name}->{child.category.name} by {mult_delta:.4f}")

        else:
            # Category mismatch!
            # The predicted transition [start] -> ... -> [end] did not happen as expected.
            # Instead we got [actual_cat].
            # This suggests the probability of the predicted path was lower than thought,
            # or we missed a transition.
            
            # Penalize the probability of the immediate first transition in the chain
            if len(chain) >= 2:
                source = chain[0].category
                predicted_target = chain[1].category
                
                # Reduce probability slightly
                self.forecasting_layer.update_causal_rule(
                    source_category=source,
                    target_category=predicted_target,
                    probability_delta=-learning_rate,
                    multiplier_delta=0.0
                )
                updates_made.append(f"Reduced probability {source.name}->{predicted_target.name}")

        return updates_made
