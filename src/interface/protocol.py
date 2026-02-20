from typing import Dict, List, Any, Optional, Tuple
from dataclasses import asdict

from src.models.task import Task
from src.models.agent import Agent
from src.models.impact import ImpactVector, ImpactProjection, ImpactCategory, ContributionClaim, SurplusPool
from src.models.registry import ImpactMetricRegistry, revenue_mapper, research_mapper, technical_mapper
from src.models.cooperative_fund import InvestmentEvaluation

from src.engine.forecasting import ForecastingLayer
from src.engine.valuation import ValuationEngine
from src.engine.surplus import CooperativeSurplusEngine
from src.engine.negotiation import AutonomousNegotiationEngine
from src.engine.ledger import ContextualizedLedgerEngine
from src.engine.cooperation import CooperativeInvestingEngine
from src.engine.recalibration import ImpactRecalibrationEngine

class EconomyProtocol:
    """
    Modular protocol interface exposing the economic engine's capabilities.
    Decouples internal logic from external agent environments.
    """

    def __init__(self):
        # 1. Initialize Registry and Engines
        self.registry = ImpactMetricRegistry()
        self.registry.register_metric("revenue", revenue_mapper)
        self.registry.register_metric("research", research_mapper)
        self.registry.register_metric("technical", technical_mapper)

        
        self.forecasting = ForecastingLayer(self.registry)
        self.valuation = ValuationEngine()
        self.surplus_engine = CooperativeSurplusEngine()
        self.negotiation = AutonomousNegotiationEngine()
        self.ledger = ContextualizedLedgerEngine()
        self.investing = CooperativeInvestingEngine(self.ledger, self.surplus_engine)
        self.recalibration = ImpactRecalibrationEngine(self.forecasting, self.surplus_engine)

        # 2. In-Memory State Storage
        # In a production system, this would be backed by a database
        self.tasks: Dict[str, Task] = {}
        self.projections: Dict[str, ImpactProjection] = {}
        self.agents: Dict[str, Agent] = {}
        self.surplus_pools: Dict[str, SurplusPool] = {}
        self.negotiation_results: Dict[str, Any] = {}

    # --- Task Submission & Forecasting ---

    def submit_task(self, task_data: Dict[str, Any]) -> str:
        """
        Submits a task with structured descriptors.
        Returns the generated Task ID.
        """
        task = Task.from_dict(task_data)
        self.tasks[task.id] = task
        
        # Automatically project impact upon submission
        projection = self.forecasting.project(task)
        self.projections[task.id] = projection
        
        return task.id

    def get_projection(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves the raw impact projection for a task."""
        proj = self.projections.get(task_id)
        if not proj:
            return None
        # Return dict representation roughly
        return {
            "task_id": proj.task_id,
            "mean_impact": proj.distribution_mean,
            "confidence_interval": proj.confidence_interval,
            "category": proj.target_vector.category.name,
            "metadata": proj.metadata
        }

    # --- Valuation & Agent Registration ---

    def register_agent(self, agent_id: str, role: str) -> str:
        """Registers an agent or updates its role."""
        if agent_id not in self.agents:
            self.agents[agent_id] = Agent.create(agent_id, role)
        else:
            self.agents[agent_id].role_label = role
        return agent_id

    def get_valuation(self, task_id: str, agent_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves valuation prediction for a specific agent working on a task.
        Adjusts impact based on agent performance.
        """
        projection = self.projections.get(task_id)
        agent = self.agents.get(agent_id)
        
        if not projection or not agent:
            return None

        adjusted_proj = self.valuation.adjust_projection(projection, agent)
        
        return {
            "task_id": task_id,
            "agent_id": agent_id,
            "adjusted_mean": adjusted_proj.distribution_mean,
            "trust_score": adjusted_proj.metadata.get("trust_score"),
            "confidence_interval": adjusted_proj.confidence_interval
        }

    # --- Surplus & Negotiation ---

    def compute_cooperative_surplus(self, cluster_id: str, task_ids: List[str]) -> Dict[str, Any]:
        """
        Computes the collective surplus for a cluster of tasks.
        """
        projections = [self.projections[tid] for tid in task_ids if tid in self.projections]
        if not projections:
            return {"error": "No valid projections found for tasks"}

        pool = self.surplus_engine.calculate_cluster_surplus(cluster_id, projections)
        self.surplus_pools[cluster_id] = pool
        
        return {
            "cluster_id": pool.cluster_id,
            "total_surplus": pool.total_surplus,
            "confidence_interval": pool.confidence_interval,
            "task_ids": pool.task_ids,
            "metadata": pool.metadata
        }

    def run_negotiation(self, cluster_id: str, claims_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Orchestrates a negotiation session for a surplus pool.
        claims_data should contain: {"agent_id": str, "marginal_contribution": float, ...}
        """
        pool = self.surplus_pools.get(cluster_id)
        if not pool:
            return {"error": "Surplus pool not found"}

        # Convert dicts to ContributionClaim objects
        claims = []
        for c in claims_data:
            claim = ContributionClaim(
                agent_id=c["agent_id"],
                task_ids=c.get("task_ids", []),
                marginal_impact_estimate=c.get("marginal_contribution", 0.0),
                uncertainty_margin=c.get("uncertainty", 0.1),
                dependency_influence_weight=c.get("dependency_weight", 0.0),
                metadata=c.get("metadata", {})
            )
            claims.append(claim)

        result = self.negotiation.negotiate_splits(pool, claims)
        self.negotiation_results[cluster_id] = (result, claims) # Store both for ledger recording
        
        return result

    # --- Ledger Operations ---

    def commit_allocations(self, cluster_id: str) -> List[str]:
        """
        Commits negotiated allocations to the immutable ledger.
        """
        if cluster_id not in self.negotiation_results:
            return []
        
        result, claims = self.negotiation_results[cluster_id]
        pool = self.surplus_pools[cluster_id]
        
        entry_ids = self.ledger.record_negotiated_allocations(pool, result, claims)
        return entry_ids

    def get_agent_balance(self, agent_id: str) -> Dict[str, Any]:
        """Retrieves detailed credit balance for an agent."""
        balance = self.ledger.get_agent_balance(agent_id)
        return {
            "agent_id": balance.agent_id,
            "total_balance": balance.total_balance,
            "breakdown": balance.balances_by_category
        }

    # --- Cooperative Funds ---

    def create_fund(self, fund_id: str, objective_data: Dict[str, Any]) -> str:
        """Creates a new cooperative fund."""
        # Convert dict to ImpactVector
        # Simplified: expecting manual construction or use a default mapper
        # For this API, let's assume raw construction from minimal fields
        target = ImpactVector(
            category=ImpactCategory[objective_data.get("category", "TECHNICAL")],
            magnitude=objective_data.get("magnitude", 1.0),
            time_horizon=objective_data.get("time_horizon", 10.0),
            uncertainty_bounds=objective_data.get("bounds", (0.5, 1.5))
        )
        
        self.investing.create_fund(fund_id, target)
        return fund_id

    def contribute_to_fund(self, fund_id: str, agent_id: str, amount: float) -> bool:
        """Agent contributes credits to a fund."""
        return self.investing.allocate_credits_to_fund(fund_id, agent_id, amount)

    def evaluate_fund_deployment(self, fund_id: str, cluster_id: str, task_ids: List[str]) -> Dict[str, Any]:
        """Evaluates if a fund should be deployed to a task cluster."""
        projections = [self.projections[tid] for tid in task_ids if tid in self.projections]
        eval_result = self.investing.evaluate_investment(fund_id, cluster_id, projections)
        
        # Serialize simply
        return {
            "fund_id": eval_result.fund_id,
            "roci": eval_result.roci,
            "risk_score": eval_result.risk_score,
            "recommendation": eval_result.recommendation
        }

    # --- Recalibration ---

    def report_outcome(self, task_id: str, actual_outcome: float, agent_id: str) -> Dict[str, Any]:
        """
        Reports the actual outcome of a task to trigger recalibration.
        """
        projection = self.projections.get(task_id)
        agent = self.agents.get(agent_id)
        
        if not projection or not agent:
            return {"error": "Task or Agent not found"}

        # 1. Recalibrate Agent Performance
        perf_updates = self.recalibration.recalibrate_agent_performance(
            agent, projection, actual_outcome
        )
        
        # 2. Recalibrate Forecasting Weights (assuming we can construct a realized vector)
        # Simplified: create a realized vector with the actual magnitude
        realized_vector = ImpactVector(
            category=projection.target_vector.category,
            magnitude=actual_outcome,
            time_horizon=projection.target_vector.time_horizon,
            uncertainty_bounds=(actual_outcome, actual_outcome)
        )
        
        self.recalibration.recalibrate_forecasting_weights(projection, realized_vector)
        
        return {
            "status": "recalibrated",
            "agent_performance_updates": perf_updates
        }
