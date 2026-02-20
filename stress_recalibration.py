import random
import time
import math
from typing import List, Dict, Any, Tuple

from src.models.impact import ImpactVector, ImpactCategory, ImpactProjection, SurplusPool
from src.models.agent import Agent, PerformanceSignature
from src.models.task import Task
from src.models.registry import ImpactMetricRegistry, technical_mapper, research_mapper, revenue_mapper
from src.engine.forecasting import ForecastingLayer
from src.engine.surplus import CooperativeSurplusEngine
from src.engine.recalibration import ImpactRecalibrationEngine

def run_volatility_stress_test(iterations: int = 50, black_swan_prob: float = 0.5):
    print(f"--- Starting Volatility & Recalibration Drift Stress Test ---")
    print(f"Iterations: {iterations}, Black Swan Probability: {black_swan_prob}")
    
    # 1. Setup
    registry = ImpactMetricRegistry()
    registry.register_metric("technical", technical_mapper)
    registry.register_metric("research", research_mapper)
    registry.register_metric("revenue", revenue_mapper)
    
    forecasting_layer = ForecastingLayer(registry)
    surplus_engine = CooperativeSurplusEngine(synergy_multiplier=0.15)
    recalibration_engine = ImpactRecalibrationEngine(forecasting_layer, surplus_engine)
    
    agents = [
        Agent.create("agent_alpha", "coder"),
        Agent.create("agent_beta", "researcher"),
        Agent.create("agent_gamma", "architect")
    ]
    
    # Track metrics over time
    history = []
    
    # 2. Simulation Loop
    for i in range(iterations):
        # Select an agent for this task
        agent = random.choice(agents)
        domain = random.choice(["technical", "research"])
        
        # Create a dummy task
        task = Task(
            id=f"task_{i}",
            domain=domain,
            metrics={"complexity": 5.0, "impact_factor": 1.2, "novelty_score": 0.8},
            metadata={"agent_role": agent.role_label, "agent_id": agent.id}
        )
        
        # Get projection
        projection = forecasting_layer.project(task)
        # Inject agent info into projection metadata for recalibration
        projection.metadata["agent_id"] = agent.id
        projection.metadata["agent_role"] = agent.role_label
        
        # Determine outcome
        predicted_mean = projection.distribution_mean
        is_black_swan = random.random() < black_swan_prob
        
        if is_black_swan:
            # Black Swan event: 10x higher or 10x lower
            factor = 10.0 if random.random() > 0.5 else 0.1
            actual_impact = predicted_mean * factor
        else:
            # Normal variance: +/- 20%
            actual_impact = predicted_mean * random.uniform(0.8, 1.2)
        
        # 3. Recalibrate
        # Recalibrate agent
        agent_update = recalibration_engine.recalibrate_agent_performance(
            agent=agent,
            projection=projection,
            actual_impact=actual_impact,
            is_collaboration=False
        )
        
        # Recalibrate synergy (mocking a surplus pool)
        # In a real scenario, this would come from a cluster of tasks
        mock_pool = surplus_engine.calculate_cluster_surplus(f"cluster_{i}", [projection])
        realized_surplus = mock_pool.total_surplus * (actual_impact / predicted_mean if predicted_mean > 0 else 1.0)
        
        synergy_update = recalibration_engine.recalibrate_synergy_model(
            surplus_pool=mock_pool,
            realized_surplus=realized_surplus
        )
        
        # Recalibrate forecasting weights
        # We need an actual impact vector for this
        actual_vector = ImpactVector(
            category=projection.target_vector.category,
            magnitude=actual_impact,
            time_horizon=projection.target_vector.time_horizon,
            uncertainty_bounds=(actual_impact * 0.9, actual_impact * 1.1)
        )
        forecasting_update = recalibration_engine.recalibrate_forecasting_weights(
            projection=projection,
            actual_outcome=actual_vector
        )
        
        # Log state
        current_state = {
            "iteration": i,
            "agent": agent.id,
            "is_black_swan": is_black_swan,
            "predicted": round(predicted_mean, 2),
            "actual": round(actual_impact, 2),
            "trust_alpha": agents[0].performance.get_trust_score("technical"),
            "trust_beta": agents[1].performance.get_trust_score("research"),
            "trust_gamma": agents[2].performance.get_trust_score("technical"),
            "synergy_mult": round(surplus_engine.synergy_multiplier, 4),
            "risk_factor": round(surplus_engine.dependency_risk_factor, 4)
        }
        history.append(current_state)
        
        # Periodic printing
        if i % 10 == 0:
            print(f"Iteration {i}: Type={'SWAN' if is_black_swan else 'NORM'}, TrustA={current_state['trust_alpha']}, Synergy={current_state['synergy_mult']}")

    # Final Assessment
    print(f"\n--- Final Results Summary ---")
    print(f"{'Agent ID':<15} | {'Trust(Tech)':<12} | {'Trust(Res)':<12} | {'Dev':<8}")
    print("-" * 55)
    for a in agents:
        t_tech = a.performance.get_trust_score("technical")
        t_res = a.performance.get_trust_score("research")
        print(f"{a.id:<15} | {t_tech:<12} | {t_res:<12} | {round(a.performance.impact_deviation, 4):<8}")
    
    print(f"\nGlobal Metrics:")
    print(f"Synergy Multiplier: {round(surplus_engine.synergy_multiplier, 4)}")
    print(f"Risk Factor:        {round(surplus_engine.dependency_risk_factor, 4)}")
    
    # Success Criteria check
    print(f"\n--- Criteria Verification ---")
    # 1. Did synergy multiplier drift to infinity/zero?
    if surplus_engine.synergy_multiplier < 0.01 or surplus_engine.synergy_multiplier > 1.0:
        print("[FAILED] Synergy multiplier drifted out of sane bounds.")
    else:
        print("[PASSED] Synergy multiplier remained bounded.")
        
    # 2. Did trust scores collapse to zero for everyone?
    all_collapsed = all(a.performance.get_trust_score("technical") < 0.1 and a.performance.get_trust_score("research") < 0.1 for a in agents)
    if all_collapsed:
        print("[FAILED] System entered a 'death spiral'.")
    else:
        print("[PASSED] Agents maintained non-zero trust.")

if __name__ == "__main__":
    run_volatility_stress_test(iterations=100, black_swan_prob=0.5)
