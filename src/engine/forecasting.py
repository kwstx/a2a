import random
import math
from typing import List, Dict, Any, Tuple
from src.models.impact import ImpactVector, ImpactProjection, ImpactCategory
from src.models.registry import ImpactMetricRegistry
from src.models.task import Task

class ForecastingLayer:
    """
    Forecasting layer that applies domain-specific predictive models and
    simulates downstream effects using causal graph traversal.
    """
    def __init__(self, registry: ImpactMetricRegistry):
        self.registry = registry
        # Causal transitions: (source_category) -> List[(target_category, probability, multiplier)]
        # This represents how one type of impact tends to cause another.
        self.causal_rules: Dict[ImpactCategory, List[Tuple[ImpactCategory, float, float]]] = {
            ImpactCategory.TECHNICAL: [
                (ImpactCategory.EFFICIENCY, 0.8, 1.2),
                (ImpactCategory.ECOSYSTEM, 0.3, 0.5)
            ],
            ImpactCategory.EFFICIENCY: [
                (ImpactCategory.REVENUE, 0.6, 1.5),
                (ImpactCategory.SOCIAL, 0.2, 0.4)
            ],
            ImpactCategory.RESEARCH: [
                (ImpactCategory.TECHNICAL, 0.7, 2.0),
                (ImpactCategory.ECOSYSTEM, 0.5, 1.1)
            ],
            ImpactCategory.ECOSYSTEM: [
                (ImpactCategory.REVENUE, 0.4, 0.8),
                (ImpactCategory.SOCIAL, 0.5, 1.2)
            ]
        }

    def project(self, task: Task) -> ImpactProjection:
        """
        Transforms a task into an ImpactProjection.
        """
        task_id = task.id
        domain = task.domain
        raw_data = task.metrics


        # 1. First-order output (Direct Impact)
        try:
            first_order_vector = self.registry.translate(domain, raw_data)
        except Exception as e:
            # Fallback to a neutral vector if mapping fails
            first_order_vector = ImpactVector(
                category=ImpactCategory.TECHNICAL,
                magnitude=1.0,
                time_horizon=30.0,
                uncertainty_bounds=(0.5, 1.5)
            )

        # 2. Simulate downstream effects (Causal Graph Traversal)
        # We simulate multiple paths to build a distribution
        simulations = 100
        total_impact_samples = []
        all_effect_chains = []

        for _ in range(simulations):
            # Sample the first-order magnitude from its uncertainty bounds
            low, high = first_order_vector.uncertainty_bounds
            if low < high:
                sampled_mag = random.uniform(low, high)
            else:
                sampled_mag = first_order_vector.magnitude
            
            # Create a localized version of the first-order vector for this simulation
            base_vector = ImpactVector(
                category=first_order_vector.category,
                magnitude=sampled_mag,
                time_horizon=first_order_vector.time_horizon,
                uncertainty_bounds=first_order_vector.uncertainty_bounds,
                causal_dependencies=first_order_vector.causal_dependencies,
                domain_weights=first_order_vector.domain_weights,
                metrics=first_order_vector.metrics
            )

            chain = self._simulate_chain(base_vector, max_orders=3)
            all_effect_chains.append(chain)
            
            # Sum up magnitudes (simplified impact aggregation)
            total_mag = sum(v.magnitude for v in chain)
            total_impact_samples.append(total_mag)


        # 3. Probabilistic Distribution & Confidence Intervals
        mean = sum(total_impact_samples) / simulations
        variance = sum((x - mean) ** 2 for x in total_impact_samples) / simulations
        std = math.sqrt(variance)
        
        # 95% confidence interval (approximate using 1.96 * std)
        ci_low = mean - 1.96 * std
        ci_high = mean + 1.96 * std

        # Representative chain (the first one or an average one)
        # For metadata, we persist the first simulation's chain as an example
        representative_chain = all_effect_chains[0]

        return ImpactProjection(
            task_id=task_id,
            target_vector=first_order_vector,
            distribution_mean=round(mean, 4),
            distribution_std=round(std, 4),
            confidence_interval=(round(ci_low, 4), round(ci_high, 4)),
            effect_chain=representative_chain,
            metadata={
                "simulations": simulations,
                "domain": domain
            }
        )

    def _simulate_chain(self, start_vector: ImpactVector, max_orders: int) -> List[ImpactVector]:
        """
        Traverses the causal graph starting from an initial vector.
        """
        chain = [start_vector]
        current_vectors = [start_vector]

        for order in range(1, max_orders):
            next_order_vectors = []
            for v in current_vectors:
                rules = self.causal_rules.get(v.category, [])
                for target_cat, prob, multiplier in rules:
                    if random.random() < prob:
                        # Create a new downstream vector
                        # Magnitude is derived from parent, with some noise and multiplier
                        noise = random.uniform(0.8, 1.2)
                        new_mag = v.magnitude * multiplier * noise
                        
                        # Time horizon usually expands as we go deeper
                        new_horizon = v.time_horizon * random.uniform(1.5, 3.0)
                        
                        downstream = ImpactVector(
                            category=target_cat,
                            magnitude=new_mag,
                            time_horizon=new_horizon,
                            uncertainty_bounds=(new_mag * 0.7, new_mag * 1.3),
                            causal_dependencies=[f"order_{order-1}_{v.category.value}"]
                        )
                        next_order_vectors.append(downstream)
            
            if not next_order_vectors:
                break
                
            chain.extend(next_order_vectors)
            current_vectors = next_order_vectors

        return chain

    def update_causal_rule(self, source_category: ImpactCategory, target_category: ImpactCategory, 
                           probability_delta: float, multiplier_delta: float):
        """
        Updates the causal probability and multiplier for a specific transition.
        Used by the recalibration engine to adjust forecasting weights based on real-world data.
        """
        if source_category not in self.causal_rules:
            return

        rules = self.causal_rules[source_category]
        updated_rules = []
        found = False

        for target, prob, mult in rules:
            if target == target_category:
                # Apply deltas with clamping
                new_prob = max(0.0, min(1.0, prob + probability_delta))
                new_mult = max(0.1, mult + multiplier_delta) # Multiplier shouldn't be zero/negative
                updated_rules.append((target, new_prob, new_mult))
                found = True
            else:
                updated_rules.append((target, prob, mult))
        
        if not found and probability_delta > 0:
            # If the transition didn't exist but we observed it (implied by positive delta), 
            # we might want to add it. For now, we only update existing rules to stay safe.
            pass

        self.causal_rules[source_category] = updated_rules
