
import time
import random
import sys
import psutil
import os
from src.interface.protocol import EconomyProtocol

def run_graph_stress_test():
    protocol = EconomyProtocol()
    
    print("--- Starting Dependency Complexity & Graph Stress Test ---")
    
    # 1. Deep and Complex Causal Graph
    # Create task clusters with 100+ tasks where each task depends on 5-10 others.
    num_tasks = 150
    
    print(f"Scenario 1: Complex DAG with {num_tasks} tasks and 5-10 dependencies each.")
    
    task_ids = []
    process = psutil.Process(os.getpid())
    start_mem = process.memory_info().rss / 1024 / 1024 # MB
    
    start_time = time.time()
    
    for i in range(num_tasks):
        # Pick random dependencies from already created tasks (ensuring DAG)
        possible_deps = task_ids[:]
        num_deps = min(len(possible_deps), random.randint(5, 10))
        causal_deps = random.sample(possible_deps, num_deps) if possible_deps else []
        
        # Add some external dependencies to test risk factor
        if random.random() < 0.3:
            causal_deps.append(f"external-dep-{i}")

        task_data = {

            "id": f"task-dag-{i}",
            "domain": "technical",
            "metrics": {
                "complexity": 1.5,
                "impact_factor": 2.0,
                "causal_dependencies": causal_deps
            }
        }
        tid = protocol.submit_task(task_data)
        task_ids.append(tid)

    end_submit_time = time.time()
    
    # Calculate Surplus
    print("Calculating surplus for the cluster...")
    t0 = time.time()
    try:
        surplus_result = protocol.compute_cooperative_surplus("cluster-dag-1", task_ids)
        t1 = time.time()
        
        end_mem = process.memory_info().rss / 1024 / 1024 # MB
        
        print(f"DAG Submission Duration: {end_submit_time - start_time:.2f}s")
        print(f"Surplus Calculation Duration: {t1 - t0:.4f}s")
        print(f"Memory Usage Increase: {end_mem - start_mem:.2f} MB")
        print(f"Total Surplus: {surplus_result['total_surplus']}")
        print(f"Internal Dependencies: {surplus_result.get('metadata', {}).get('internal_dependencies', 'N/A')}")
    except Exception as e:
        print(f"FAIL: Surplus calculation failed: {e}")

    # 2. Circular Dependencies
    print("\nScenario 2: Circular Dependencies (A -> B -> C -> A)")
    # A -> B
    protocol.submit_task({
        "id": "task-A",
        "domain": "technical",
        "metrics": {"complexity": 1.0, "causal_dependencies": ["task-B"]}
    })
    # B -> C
    protocol.submit_task({
        "id": "task-B",
        "domain": "technical",
        "metrics": {"complexity": 1.0, "causal_dependencies": ["task-C"]}
    })
    # C -> A
    protocol.submit_task({
        "id": "task-C",
        "domain": "technical",
        "metrics": {"complexity": 1.0, "causal_dependencies": ["task-A"]}
    })
    
    try:
        t0 = time.time()
        loop_surplus = protocol.compute_cooperative_surplus("cluster-loop", ["task-A", "task-B", "task-C"])
        t1 = time.time()
        print(f"Loop Handling Duration: {t1 - t0:.4f}s")
        print(f"Loop Surplus: {loop_surplus['total_surplus']}")
        print("SUCCESS: Loop handled without infinite recursion.")
    except RecursionError:
        print("FAIL: Infinite recursion detected on circular dependencies.")
    except Exception as e:
        print(f"Caught exception: {e}")

    # 3. Vary dependency_risk_factor
    print("\nScenario 3: Varying dependency_risk_factor")
    protocol.surplus_engine.dependency_risk_factor = 0.5 # High risk
    risk_high = protocol.compute_cooperative_surplus("cluster-high-risk", task_ids)
    
    protocol.surplus_engine.dependency_risk_factor = 0.01 # Low risk
    risk_low = protocol.compute_cooperative_surplus("cluster-low-risk", task_ids)
    
    print(f"High Risk (0.5) Surplus: {risk_high['total_surplus']}")
    print(f"Low Risk (0.01) Surplus: {risk_low['total_surplus']}")
    
    if risk_high['total_surplus'] < risk_low['total_surplus']:
        print("SUCCESS: Surplus scales correctly with risk factor (higher risk = lower surplus).")
    else:
        print("FAIL: Surplus does not decrease with higher risk factor.")

if __name__ == "__main__":
    run_graph_stress_test()
