
import time
from src.interface.protocol import EconomyProtocol

def benchmark():
    protocol = EconomyProtocol()
    
    # Pre-register some agents
    for i in range(10):
        protocol.register_agent(f"agent-{i}", "worker")
    
    start_time = time.time()
    num_tasks = 100
    for i in range(num_tasks):
        task_data = {
            "id": f"task-{i}",
            "domain": "revenue",
            "metrics": {"amount": 1000}
        }
        protocol.submit_task(task_data)
        protocol.get_valuation(f"task-{i}", "agent-0")
    
    end_time = time.time()
    duration = end_time - start_time
    tps = num_tasks / duration
    print(f"Processed {num_tasks} tasks in {duration:.4f} seconds")
    print(f"TPS: {tps:.2f}")
    print(f"Avg latency per task: {duration/num_tasks*1000:.2f}ms")

if __name__ == "__main__":
    benchmark()
