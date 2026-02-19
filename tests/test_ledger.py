import sys
import os
import json

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.impact import ImpactProjection, ImpactVector, ImpactCategory, SurplusPool, ContributionClaim
from src.engine.ledger import ContextualizedLedgerEngine

def test_ledger_flow():
    print("--- Testing Contextualized Credit Ledger ---")
    
    # 1. Setup Mock Surplus and Negotiation Results
    cluster_id = "cluster-888"
    
    # Mock Surplus Pool
    pool = SurplusPool(
        cluster_id=cluster_id,
        total_surplus=150.0,
        confidence_interval=(140.0, 160.0),
        aggregated_vectors={"RESEARCH": 100.0, "TECHNICAL": 50.0},
        task_ids=["task-1", "task-2"],
        metadata={"avg_time_horizon": 30.0}
    )
    
    # Mock Negotiation Results
    negotiation_results = {
        "final_allocations": {
            "agent-alpha": 90.0,
            "agent-beta": 60.0
        },
        "rounds_to_convergence": 12
    }
    
    # Mock Claims (to provide task_ids and context)
    claims = [
        ContributionClaim(
            agent_id="agent-alpha",
            cluster_id=cluster_id,
            marginal_impact_estimate=85.0,
            uncertainty_margin=5.0,
            dependency_influence_weight=0.8,
            task_ids=["task-1"],
            metadata={"primary_category": "RESEARCH"}
        ),
        ContributionClaim(
            agent_id="agent-beta",
            cluster_id=cluster_id,
            marginal_impact_estimate=55.0,
            uncertainty_margin=8.0,
            dependency_influence_weight=0.4,
            task_ids=["task-2"],
            metadata={"primary_category": "TECHNICAL"}
        )
    ]
    
    # 2. Record in Ledger
    engine = ContextualizedLedgerEngine()
    entry_ids = engine.record_negotiated_allocations(pool, negotiation_results, claims)
    
    print(f"Recorded {len(entry_ids)} credit entries.")
    
    # 3. Verify Balances
    balance_alpha = engine.get_agent_balance("agent-alpha")
    print(f"\nAgent Alpha Balance: {balance_alpha.total_balance}")
    print(f"Balances by Category: {balance_alpha.balances_by_category}")
    
    balance_beta = engine.get_agent_balance("agent-beta")
    print(f"Agent Beta Balance: {balance_beta.total_balance}")
    print(f"Balances by Category: {balance_beta.balances_by_category}")
    
    # 4. Verify Double-Entry Integrity
    audit = engine.audit_ledger()
    print(f"\nAudit Result: {audit['integrity_check']}")
    print(f"Net Balance: {audit['net_balance']}")
    print(f"System Issuance: {audit['system_issuance']}")
    
    # 5. Verify Provenance and Tagging
    first_entry_id = entry_ids[0]
    trace = engine.verify_provenance(first_entry_id)
    print(f"\nProvenance Trace for {first_entry_id}:")
    print(json.dumps(trace, indent=2))
    
    # Check tagging on the entry itself
    entry = next(e for e in engine.entries if e.entry_id == first_entry_id)
    print(f"\nEntry Tagging Check:")
    print(f"  Impact Category: {entry.domain_context.name}")
    print(f"  Origin Event: {entry.provenance.surplus_event_id}")
    print(f"  Backing Magnitude: {entry.impact_vector.magnitude}")

    assert balance_alpha.total_balance == 90.0
    assert balance_beta.total_balance == 60.0
    assert audit['integrity_check'] == "passed"
    assert trace['origin_surplus_id'] == cluster_id
    
    print("\n--- Ledger Test Passed ---")

if __name__ == "__main__":
    test_ledger_flow()
