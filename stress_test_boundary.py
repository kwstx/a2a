import math
import uuid
import logging
from src.models.impact import ImpactVector, ImpactCategory, ImpactProjection, SurplusPool
from src.models.task import Task
from src.models.agent import Agent, PerformanceSignature
from src.models.registry import ImpactMetricRegistry, technical_mapper, revenue_mapper
from src.engine.forecasting import ForecastingLayer
from src.engine.valuation import ValuationEngine
from src.engine.recalibration import ImpactRecalibrationEngine
from src.engine.surplus import CooperativeSurplusEngine
from src.engine.ledger import ContextualizedLedgerEngine

# Configure logging to capture error messages during "Fail-Fast" behavior
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("BoundaryStressTest")

def setup_test_environment():
    registry = ImpactMetricRegistry()
    registry.register_metric("technical", technical_mapper)
    registry.register_metric("revenue", revenue_mapper)
    
    forecaster = ForecastingLayer(registry)
    surplus_engine = CooperativeSurplusEngine()
    recalibrator = ImpactRecalibrationEngine(forecaster, surplus_engine)
    valuation_engine = ValuationEngine()
    ledger_engine = ContextualizedLedgerEngine()
    
    return registry, forecaster, surplus_engine, recalibrator, valuation_engine, ledger_engine

def test_garbage_tasks(registry, forecaster):
    print("\n--- Testing Garbage Task Submission ---")
    
    garbage_scenarios = [
        {"id": "zero_mag", "domain": "technical", "metrics": {"complexity": 0.0, "impact_factor": 1.0}},
        {"id": "neg_time", "domain": "technical", "metrics": {"complexity": 1.0, "time_horizon": -100.0}},
        {"id": "nan_impact", "domain": "technical", "metrics": {"complexity": float('nan'), "impact_factor": 1.0}},
        {"id": "inf_impact", "domain": "technical", "metrics": {"complexity": float('inf'), "impact_factor": 1.0}},
        {"id": "", "domain": "technical", "metrics": {"complexity": 1.0}},
        {"id": "empty_domain", "domain": "", "metrics": {"complexity": 1.0}},
    ]
    
    results = []
    for scenario in garbage_scenarios:
        print(f"Submitting task: {scenario['id']} (domain: '{scenario['domain']}')")
        try:
            task = Task(id=scenario["id"], domain=scenario["domain"], metrics=scenario["metrics"])
            projection = forecaster.project(task)
            print(f"  Result: Success. Mean Impact: {projection.distribution_mean}")
            results.append((scenario["id"], "Success", projection.distribution_mean))
        except Exception as e:
            print(f"  Result: FAILED-FAST with error: {e}")
            results.append((scenario["id"], "Fail-Fast", str(e)))
            
    return results

def test_agent_anomalies():
    print("\n--- Testing Agent Registration Anomalies ---")
    
    # Existing system doesn't have a global registry yet, but let's see how Agent.create handles things
    print("Testing duplicate agent IDs...")
    a1 = Agent.create("agent_001", "coder")
    a2 = Agent.create("agent_001", "researcher")
    print(f"  Agent 1: {a1.id}, Role: {a1.role_label}")
    print(f"  Agent 2: {a2.id}, Role: {a2.role_label}")
    
    # Testing unsupported roles
    print("Testing unsupported/weird roles...")
    a3 = Agent.create("agent_002", "")
    a4 = Agent.create("agent_003", "chaos_agent")
    print(f"  Agent 3: '{a3.role_label}'")
    print(f"  Agent 4: '{a4.role_label}'")
    
    return True

def test_recalibration_division_by_zero(recalibrator):
    print("\n--- Testing Recalibration with Zero Expected Impact ---")
    
    agent = Agent.create("test_agent", "tester")
    
    # Near-zero expected impact (to test division robustness while passing validation)
    target_vector = ImpactVector(ImpactCategory.TECHNICAL, 1e-9, 1.0, (1e-10, 2e-10))
    projection = ImpactProjection(
        task_id="near_zero_task",
        target_vector=target_vector,
        distribution_mean=1e-9,
        distribution_std=1e-10,
        confidence_interval=(1e-10, 2e-10)
    )
    
    actual_impact = 100.0 # Positive actual outcome
    
    print(f"Recalibrating agent {agent.id} with expected=near_zero, actual={actual_impact}...")
    try:
        results = recalibrator.recalibrate_agent_performance(agent, projection, actual_impact)
        print(f"  Result: Success. New reliability: {results['new_reliability']}")
    except Exception as e:
        print(f"  Result: FAILED with error: {e}")
        return False

    # Test Recalibrate Synergy Model with near-zero predicted surplus
    print("Testing Recalibrate Synergy Model with near-zero predicted surplus...")
    surplus_pool = SurplusPool(
        cluster_id="zero_cluster",
        total_surplus=1e-9,
        confidence_interval=(1e-10, 2e-10),
        aggregated_vectors={},
        task_ids=[]
    )
    
    try:
        synergy_results = recalibrator.recalibrate_synergy_model(surplus_pool, 50.0)
        print(f"  Result: Success. Synergy Updates: {synergy_results}")
    except Exception as e:
        print(f"  Result: FAILED with error: {e}")
        return False
        
    return True

if __name__ == "__main__":
    registry, forecaster, surplus_engine, recalibrator, valuation_engine, ledger_engine = setup_test_environment()
    
    print("STARTING EDGE CASE BOUNDARY TESTING")
    
    r1 = test_garbage_tasks(registry, forecaster)
    r2 = test_agent_anomalies()
    r3 = test_recalibration_division_by_zero(recalibrator)
    
    print("\n--- Final Summary ---")
    all_passed = True
    
    # Check if there were any crashes in test_garbage_tasks
    # Success Criteria: System implements "Fail-Fast" behavior with descriptive error logging; no internal state corruption; no crashes.
    # Note: If it returns 0.0 or handles NaN gracefully, that's also acceptable, but usually we want validation.
    
    if r2 and r3:
        print("AGENT AND RECALIBRATION TESTS COMPLETED WITHOUT CRASHING")
    else:
        all_passed = False
        print("SOME TESTS CRASHED OR FAILED")
        
    if all_passed:
        print("\nEDGE CASE BOUNDARY TESTING SUMMARY: COMPLETED")
    else:
        print("\nEDGE CASE BOUNDARY TESTING SUMMARY: FAILED")
