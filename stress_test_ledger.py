import time
import uuid
import random
import threading
import concurrent.futures
from src.engine.ledger import ContextualizedLedgerEngine
from src.engine.cooperation import CooperativeInvestingEngine
from src.engine.surplus import CooperativeSurplusEngine
from src.models.impact import ImpactVector, ImpactCategory
from src.models.ledger import CreditProvenance, CreditEntry, EntryType

def run_ledger_integrity_test():
    print("--- Starting Ledger Integrity Stress Test ---")
    engine = ContextualizedLedgerEngine()
    
    num_events = 50000
    num_entries_per_event = 20 # 10 pairs
    total_expected_entries = num_events * num_entries_per_event
    
    print(f"Generating {total_expected_entries} entries across {num_events} events...")
    
    start_time = time.time()
    
    # Pre-generate some agent IDs
    agent_ids = [f"agent_{i}" for i in range(1000)]
    categories = list(ImpactCategory)
    
    for i in range(num_events):
        cluster_id = f"cluster_{i}"
        provenance = CreditProvenance(
            surplus_event_id=cluster_id,
            task_ids=[f"task_{i}_{j}" for j in range(3)]
        )
        
        for j in range(num_entries_per_event // 2):
            agent_id = random.choice(agent_ids)
            amount = round(random.uniform(10.0, 1000.0), 2)
            category = random.choice(categories)
            
            impact = ImpactVector(
                category=category,
                magnitude=amount,
                time_horizon=1.0,
                uncertainty_bounds=(0.9 * amount, 1.1 * amount)
            )
            
            # Credit Agent
            credit = CreditEntry(
                entry_id=str(uuid.uuid4()),
                agent_id=agent_id,
                amount=amount,
                entry_type=EntryType.CREDIT,
                impact_vector=impact,
                domain_context=category,
                provenance=provenance
            )
            
            # Debit System
            debit = CreditEntry(
                entry_id=str(uuid.uuid4()),
                agent_id=engine.SYSTEM_ACCOUNT,
                amount=-amount,
                entry_type=EntryType.DEBIT,
                impact_vector=impact,
                domain_context=category,
                provenance=provenance
            )
            
            engine._add_entry(credit)
            engine._add_entry(debit)
            
        if (i + 1) % 10000 == 0:
            print(f"  Processed {i + 1} events ({(i + 1) * 20} entries)...")

    duration = time.time() - start_time
    print(f"Generation completed in {duration:.2f} seconds ({total_expected_entries / duration:.2f} entries/sec).")
    
    # Integrity Check
    print("Running audit...")
    audit_results = engine.audit_ledger()
    print(f"Audit Results: {audit_results}")
    
    # Performance Check: Balance Retrieval
    print("Testing balance retrieval performance...")
    test_agents = random.sample(agent_ids, 100)
    perf_start = time.time()
    for aid in test_agents:
        engine.get_agent_balance(aid)
    perf_duration = time.time() - perf_start
    print(f"Avg balance retrieval time: {perf_duration/100:.6f} seconds.")
    
    return audit_results["integrity_check"] == "passed" and audit_results["entry_count"] == total_expected_entries

def run_concurrency_race_condition_test():
    print("\n--- Starting Concurrency & Race Condition Test ---")
    ledger = ContextualizedLedgerEngine()
    surplus = CooperativeSurplusEngine()
    coop = CooperativeInvestingEngine(ledger, surplus)
    
    # 1. Fund setup
    fund_id = "STRESS_FUND_001"
    objective = ImpactVector(ImpactCategory.TECHNICAL, 1000.0, 1.0, (900.0, 1100.0))
    coop.create_fund(fund_id, objective)
    
    # 2. Give some agents initial capital
    agent_id = "rich_agent"
    impact = ImpactVector(ImpactCategory.TECHNICAL, 1000000.0, 1.0, (900000.0, 1100000.0))
    prov = CreditProvenance(surplus_event_id="INITIAL_MINT")
    
    # Mint 1,000,000 credits to rich_agent
    ledger._add_entry(CreditEntry(
        entry_id="mint_credit", agent_id=agent_id, amount=1000000.0, 
        entry_type=EntryType.CREDIT, impact_vector=impact, domain_context=ImpactCategory.TECHNICAL,
        provenance=prov
    ))
    ledger._add_entry(CreditEntry(
        entry_id="mint_debit", agent_id=ledger.SYSTEM_ACCOUNT, amount=-1000000.0, 
        entry_type=EntryType.DEBIT, impact_vector=impact, domain_context=ImpactCategory.TECHNICAL,
        provenance=prov
    ))
    
    print(f"Initial balance for {agent_id}: {ledger.get_agent_balance(agent_id).total_balance}")

    # 3. Concurrent allocations
    num_threads = 20
    allocations_per_thread = 100
    amount_per_alloc = 10.0
    
    def worker():
        success_count = 0
        for _ in range(allocations_per_thread):
            if coop.allocate_credits_to_fund(fund_id, agent_id, amount_per_alloc):
                success_count += 1
        return success_count

    print(f"Starting {num_threads} threads for concurrent allocation...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(worker) for _ in range(num_threads)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]
        
    total_success = sum(results)
    print(f"Total successful allocations: {total_success}")
    
    # 4. Verification
    final_balance = ledger.get_agent_balance(agent_id).total_balance
    fund_account = f"FUND_{fund_id}"
    fund_balance = ledger.get_agent_balance(fund_account).total_balance
    
    expected_debit = total_success * amount_per_alloc
    print(f"Final Agent Balance: {final_balance}")
    print(f"Final Fund Balance: {fund_balance}")
    print(f"Audit: {ledger.audit_ledger()}")
    
    # 5. Over-allocation attempt
    print("Attempting to over-allocate (double spend)...")
    exact_remaining = final_balance
    # This should succeed
    s1 = coop.allocate_credits_to_fund(fund_id, agent_id, exact_remaining)
    # This should fail
    s2 = coop.allocate_credits_to_fund(fund_id, agent_id, 1.0)
    
    print(f"Last valid allocation result: {s1}")
    print(f"Over-allocation result: {s2}")
    
    balance_after_over = ledger.get_agent_balance(agent_id).total_balance
    print(f"Balance after over-allocation attempt: {balance_after_over}")
    
    return s2 == False and balance_after_over >= 0

if __name__ == "__main__":
    import sys
    original_stdout = sys.stdout
    try:
        with open("stress_test_ledger.log", "w") as f:
            sys.stdout = f
            r1 = run_ledger_integrity_test()
            r2 = run_concurrency_race_condition_test()
            
            if r1 and r2:
                print("\nALL LEDGER STRESS TESTS PASSED")
            else:
                print("\nSOME TESTS FAILED")
    finally:
        sys.stdout = original_stdout
    
    # Also print to terminal so I can see it's done
    print("Stress test completed. Results in stress_test_ledger.log")
