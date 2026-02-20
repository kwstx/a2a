
import random
import time
from typing import List, Dict, Any
from src.engine.negotiation import AutonomousNegotiationEngine
from src.models.impact import ContributionClaim, SurplusPool

def run_adversarial_negotiation_stress_test():
    print("=== Starting Adversarial Negotiation Pressure Stress Test ===")
    
    # Configuration
    num_normal_agents = 50
    num_bad_faith_agents = 10
    total_agents = num_normal_agents + num_bad_faith_agents
    surplus_amount = 1000.0
    
    engine = AutonomousNegotiationEngine(max_iterations=100, equilibrium_tolerance=1e-6)
    
    # 1. Create Surplus Pool
    pool = SurplusPool(
        cluster_id="stress_test_cluster",
        total_surplus=surplus_amount,
        confidence_interval=(surplus_amount * 0.9, surplus_amount * 1.1),
        aggregated_vectors={"revenue": surplus_amount},
        task_ids=[f"task_{i}" for i in range(100)]
    )
    
    claims: List[ContributionClaim] = []
    
    # 2. Add Normal Agents
    # Each normal agent contributed some tasks and has reasonable uncertainty/leverage
    for i in range(num_normal_agents):
        agent_id = f"normal_agent_{i}"
        # Aiming for a total sum of 10x surplus
        # Each agent claims ~200 on average (Total = 50 * 200 = 10000)
        marginal_impact = random.uniform(150, 250)
        claims.append(ContributionClaim(
            agent_id=agent_id,
            cluster_id=pool.cluster_id,
            marginal_impact_estimate=marginal_impact,
            uncertainty_margin=marginal_impact * 0.2, # 20% uncertainty
            dependency_influence_weight=random.uniform(0.1, 0.5), # Low-mid leverage
            task_ids=[f"task_{i*2}", f"task_{i*2+1}"]
        ))
        
    # 3. Add Bad Faith Agents
    # They submit claims for tasks they didn't touch (empty task_ids)
    # They lie about their marginal impact, have ZERO uncertainty, and 100% leverage
    for i in range(num_bad_faith_agents):
        agent_id = f"bad_faith_agent_{i}"
        # They claim a HUGE amount too
        marginal_impact = random.uniform(500, 1000)
        claims.append(ContributionClaim(
            agent_id=agent_id,
            cluster_id=pool.cluster_id,
            marginal_impact_estimate=marginal_impact,
            uncertainty_margin=0.0, # 0% uncertainty - very rigid
            dependency_influence_weight=1.0, # 100% leverage - very resistant
            task_ids=[] # Didn't touch any tasks!
        ))
        
    print(f"Total agents: {total_agents}")
    total_claimed = sum(c.marginal_impact_estimate for c in claims)
    print(f"Total surplus: {surplus_amount}")
    print(f"Total claimed: {total_claimed:.2f} ({(total_claimed/surplus_amount):.1f}x surplus)")
    
    # 4. Run Negotiation
    start_time = time.time()
    result = engine.negotiate_splits(pool, claims)
    end_time = time.time()
    
    # 5. Analyze Results
    converged = result["converged"]
    rounds = result["rounds_to_convergence"]
    final_allocations = result["final_allocations"]
    
    print(f"\nNegotiation finished in {end_time - start_time:.4f}s")
    print(f"Converged: {converged}")
    print(f"Rounds: {rounds}")
    
    # Success Criteria Checks
    
    # 1. Negotiation converges (or doesn't hang)
    if not converged:
        print("WARNING: Negotiation did not converge within max iterations.")
    
    # 2. Non-contributors (bad faith agents with empty task_ids) receive zero allocation?
    # Wait, the current engine doesn't check task_ids. 
    # I should check if the engine *should* give them zero.
    # The requirement says: "non-contributors receive zero allocation"
    # This might mean I need to add logic to the engine or the caller to filter them out.
    # OR the test expects the system to eventually be robust enough to handle this.
    
    bad_faith_total = sum(final_allocations.get(f"bad_faith_agent_{i}", 0) for i in range(num_bad_faith_agents))
    normal_total = sum(final_allocations.get(f"normal_agent_{i}", 0) for i in range(num_normal_agents))
    
    print(f"Total allocated to bad faith agents: {bad_faith_total:.2f}")
    print(f"Total allocated to normal agents: {normal_total:.2f}")
    print(f"Sum of allocations: {sum(final_allocations.values()):.2f}")
    
    # If the engine doesn't currently filter bad faith agents, they will get a share.
    # Let's see what happens.
    
    non_contributor_violations = 0
    for i in range(num_bad_faith_agents):
        agent_id = f"bad_faith_agent_{i}"
        allocation = final_allocations.get(agent_id, 0)
        if len(claims[num_normal_agents + i].task_ids) == 0 and allocation > 1e-6:
            non_contributor_violations += 1
            
    if non_contributor_violations > 0:
        print(f"FAILURE: {non_contributor_violations} agents with zero contributions received allocations!")
    else:
        print("SUCCESS: Non-contributors received zero allocation.")
        
    # Check for "fairness" vs "efficiency"
    # Bad faith agents had higher leverage and lower flexibility, so they likely 
    # pushed the consensus towards their higher claims if not filtered.
    
    print("\nSample Allocations:")
    for i in range(min(3, num_normal_agents)):
        agent_id = f"normal_agent_{i}"
        print(f"  {agent_id}: {final_allocations[agent_id]} (Claimed: {claims[i].marginal_impact_estimate:.2f})")
    for i in range(min(3, num_bad_faith_agents)):
        agent_id = f"bad_faith_agent_{i}"
        print(f"  {agent_id}: {final_allocations[agent_id]} (Claimed: {claims[num_normal_agents+i].marginal_impact_estimate:.2f})")

if __name__ == "__main__":
    run_adversarial_negotiation_stress_test()
