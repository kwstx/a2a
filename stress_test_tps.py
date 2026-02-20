
import time
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.interface.protocol import EconomyProtocol

def run_stress_test():
    protocol = EconomyProtocol()
    
    num_tasks = 10000
    num_agents = 500
    time_window = 60 # seconds
    
    print(f"Starting Stress Test: {num_tasks} tasks, {num_agents} agents, target window {time_window}s")
    
    start_time = time.time()
    
    # 1. Concurrent Agent Registration
    def register_agents():
        agent_start = time.time()
        for i in range(num_agents):
            role = random.choice(["coder", "researcher", "analyst", "manager"])
            protocol.register_agent(f"agent-{i}", role)
        return time.time() - agent_start

    # 2. task Submission and Valuation
    task_latencies = []
    valuation_latencies = []
    errors = 0
    
    def process_task(i):
        nonlocal errors
        try:
            # Task Submission
            task_data = {
                "id": f"task-{i}",
                "domain": random.choice(["revenue", "research", "technical"]),
                "metrics": {"amount": random.uniform(100, 10000)}
            }
            
            t0 = time.time()
            task_id = protocol.submit_task(task_data)
            t1 = time.time()
            task_latencies.append((t1 - t0) * 1000) # ms
            
            # Valuation
            agent_id = f"agent-{random.randint(0, num_agents-1)}"
            t2 = time.time()
            valuation = protocol.get_valuation(task_id, agent_id)
            t3 = time.time()
            valuation_latencies.append((t3 - t2) * 1000) # ms
            
            if not valuation:
                return False
            return True
        except Exception as e:
            errors += 1
            return False

    # Run registrations and task submissions
    with ThreadPoolExecutor(max_workers=10) as executor:
        # Start agent registration
        reg_future = executor.submit(register_agents)
        
        # Start task processing
        task_futures = [executor.submit(process_task, i) for i in range(num_tasks)]
        
        # Wait for all
        for future in as_completed(task_futures):
            future.result()
            
        reg_duration = reg_future.result()

    end_time = time.time()
    total_duration = end_time - start_time
    
    # Statistics
    avg_task_lat = sum(task_latencies) / len(task_latencies) if task_latencies else 0
    avg_val_lat = sum(valuation_latencies) / len(valuation_latencies) if valuation_latencies else 0
    tps = num_tasks / total_duration
    
    print("\n--- Stress Test Results ---")
    print(f"Total Duration: {total_duration:.2f} seconds")
    print(f"Total Tasks Processed: {num_tasks}")
    print(f"Total Agents Registered: {num_agents}")
    print(f"TPS (Transactions Per Second): {tps:.2f}")
    print(f"Errors/Dropped Tasks: {errors}")
    print(f"Avg Task Submission Latency: {avg_task_lat:.2f} ms")
    print(f"Avg Valuation Latency: {avg_val_lat:.2f} ms")
    print(f"Max Task Submission Latency: {max(task_latencies) if task_latencies else 0:.2f} ms")
    print(f"Max Valuation Latency: {max(valuation_latencies) if valuation_latencies else 0:.2f} ms")
    
    # Integrity Check
    final_tasks = len(protocol.tasks)
    final_agents = len(protocol.agents)
    print(f"Final Task Count: {final_tasks}")
    print(f"Final Agent Count: {final_agents}")
    
    # Success Criteria Check
    success = True
    if total_duration > time_window:
        print("FAIL: Total duration exceeded 60 seconds window.")
        success = False
    if errors > 0:
        print(f"FAIL: {errors} tasks dropped or failed.")
        success = False
    if final_tasks != num_tasks:
        print(f"FAIL: Final task count {final_tasks} does not match requested {num_tasks}.")
        success = False
    if final_agents != num_agents:
        print(f"FAIL: Final agent count {final_agents} does not match requested {num_agents}.")
        success = False
    if avg_val_lat >= 100:
        print(f"FAIL: Average valuation latency {avg_val_lat:.2f}ms exceeds 100ms.")
        success = False
        
    if success:
        print("\nSUCCESS: All stress test criteria met.")
    else:
        print("\nFAILURE: One or more stress test criteria not met.")
        
    # Write results to a file
    with open("stress_test_results_tps.txt", "w") as f:
        f.write(f"TPS Stress Test Results\n")
        f.write(f"-----------------------\n")
        f.write(f"Tasks: {num_tasks}\n")
        f.write(f"Agents: {num_agents}\n")
        f.write(f"Total Duration: {total_duration:.2f}s\n")
        f.write(f"TPS: {tps:.2f}\n")
        f.write(f"Avg Task Latency: {avg_task_lat:.2f}ms\n")
        f.write(f"Avg Valuation Latency: {avg_val_lat:.2f}ms\n")
        f.write(f"Final Tasks: {final_tasks}\n")
        f.write(f"Final Agents: {final_agents}\n")
        f.write(f"Errors: {errors}\n")
        f.write(f"Status: {'SUCCESS' if success else 'FAILURE'}\n")

if __name__ == "__main__":
    run_stress_test()
