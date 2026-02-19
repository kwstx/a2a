import uuid
import time
from typing import Dict, List, Any, Optional
from src.models.ledger import CreditEntry, CreditProvenance, EntryType, AgentBalance
from src.models.impact import ImpactVector, ImpactCategory, ContributionClaim, SurplusPool

class ContextualizedLedgerEngine:
    """
    Implements a double-entry ledger where credits are tagged with origin 
    impact vectors, domain context, and time metadata.
    Ensures economic traceability and prevents detached speculation.
    """
    
    def __init__(self):
        # All ledger entries in order
        self.entries: List[CreditEntry] = []
        # Index for faster balance calculation
        self._agent_balances: Dict[str, Dict[str, float]] = {} # agent_id -> {category_name -> amount}
        # System account for surplus issuance (tracks total credits minted)
        self.SYSTEM_ACCOUNT = "SYSTEM_SURPLUS_ISSUANCE"

    def record_negotiated_allocations(self, 
                                     pool: SurplusPool, 
                                     negotiation_results: Dict[str, Any],
                                     claims: List[ContributionClaim]) -> List[str]:
        """
        Transforms negotiation results into ledger entries.
        Each allocation to an agent is recorded as a credit, balanced by a system debit.
        """
        new_entry_ids = []
        cluster_id = pool.cluster_id
        allocations = negotiation_results.get("final_allocations", {})
        
        # Map claims for easy lookup to get impact vectors
        claim_map = {c.agent_id: c for c in claims}
        
        for agent_id, amount in allocations.items():
            if amount <= 0:
                continue
                
            claim = claim_map.get(agent_id)
            # Find the primary impact category for this agent's contribution
            # If multiple tasks, we could aggregate, but for now we take the dominant one or from metadata
            category = ImpactCategory.TECHNICAL # Default
            if claim and claim.task_ids:
                category_val = claim.metadata.get("primary_category")
                if category_val:
                    if isinstance(category_val, str):
                        try:
                            # Try as value (e.g. "research")
                            category = ImpactCategory(category_val.lower())
                        except ValueError:
                            try:
                                # Try as name (e.g. "RESEARCH")
                                category = ImpactCategory[category_val.upper()]
                            except KeyError:
                                pass
                elif hasattr(pool, 'aggregated_vectors') and pool.aggregated_vectors:
                    # Pick most relevant from pool if available
                    category_name = max(pool.aggregated_vectors, key=pool.aggregated_vectors.get)
                    try:
                        category = ImpactCategory[category_name.upper()]
                    except KeyError:
                        try:
                            category = ImpactCategory(category_name.lower())
                        except ValueError:
                            pass

            # Construct representative ImpactVector for the credit
            # This represents the "backing" of the credit
            credit_impact = ImpactVector(
                category=category,
                magnitude=amount,
                time_horizon=pool.metadata.get("avg_time_horizon", 1.0),
                uncertainty_bounds=pool.confidence_interval,
                causal_dependencies=claim.task_ids if claim else [],
                domain_weights={"surplus_contribution": claim.marginal_impact_estimate if claim else 0.0}
            )

            provenance = CreditProvenance(
                surplus_event_id=cluster_id,
                contribution_claim_id=claim.agent_id if claim else None,
                task_ids=claim.task_ids if claim else [],
                negotiation_round=negotiation_results.get("rounds_to_convergence")
            )

            # 1. Credit the Agent
            agent_entry = CreditEntry(
                entry_id=str(uuid.uuid4()),
                agent_id=agent_id,
                amount=amount,
                entry_type=EntryType.CREDIT,
                impact_vector=credit_impact,
                domain_context=category,
                provenance=provenance,
                timestamp=time.time()
            )
            
            # 2. Debit the System Account (Double Entry)
            system_entry = CreditEntry(
                entry_id=str(uuid.uuid4()),
                agent_id=self.SYSTEM_ACCOUNT,
                amount=-amount,
                entry_type=EntryType.DEBIT,
                impact_vector=credit_impact,
                domain_context=category,
                provenance=provenance,
                timestamp=time.time()
            )

            self._add_entry(agent_entry)
            self._add_entry(system_entry)
            
            new_entry_ids.append(agent_entry.entry_id)
            
        return new_entry_ids

    def _add_entry(self, entry: CreditEntry):
        """Internal helper to append entry and update balance cache."""
        self.entries.append(entry)
        
        agent_id = entry.agent_id
        category = entry.domain_context.name
        
        if agent_id not in self._agent_balances:
            self._agent_balances[agent_id] = {}
            
        current_cat_balance = self._agent_balances[agent_id].get(category, 0.0)
        self._agent_balances[agent_id][category] = round(current_cat_balance + entry.amount, 6)

    def get_agent_balance(self, agent_id: str) -> AgentBalance:
        """
        Returns a detailed balance for an agent, preserving domain context.
        """
        cat_balances = self._agent_balances.get(agent_id, {})
        total = sum(cat_balances.values())
        
        recent = [e for e in self.entries if e.agent_id == agent_id][-10:]
        
        return AgentBalance(
            agent_id=agent_id,
            total_balance=round(total, 6),
            balances_by_category=cat_balances,
            recent_entries=recent
        )

    def verify_provenance(self, entry_id: str) -> Optional[Dict[str, Any]]:
        """
        Trace a specific credit back to its origin surplus event.
        """
        for entry in self.entries:
            if entry.entry_id == entry_id:
                return {
                    "entry_id": entry.entry_id,
                    "agent_id": entry.agent_id,
                    "origin_surplus_id": entry.provenance.surplus_event_id,
                    "impact_magnitude": entry.impact_vector.magnitude,
                    "impact_category": entry.domain_context.value,
                    "timestamp": entry.timestamp,
                    "traceability_status": "verified"
                }
        return None

    def audit_ledger(self) -> Dict[str, Any]:
        """
        Verify the integrity of the double-entry system.
        Total credits - Total debits should be zero.
        """
        total_sum = sum(e.amount for e in self.entries)
        return {
            "entry_count": len(self.entries),
            "net_balance": round(total_sum, 8),
            "integrity_check": "passed" if abs(total_sum) < 1e-6 else "failed",
            "system_issuance": self.get_agent_balance(self.SYSTEM_ACCOUNT).total_balance
        }
