import time
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

class ImpactCategory(Enum):
    REVENUE = "revenue"
    RESEARCH = "research"
    EFFICIENCY = "efficiency"
    SOCIAL = "social"
    ECOSYSTEM = "ecosystem"
    TECHNICAL = "technical"

@dataclass
class ImpactVector:
    """
    A multi-dimensional representation of a task's downstream impact.
    """
    category: ImpactCategory
    # Magnitude projection (normalized or raw metric value)
    magnitude: float
    # Time horizon in arbitrary units (e.g., days or project phases)
    time_horizon: float
    # Uncertainty bounds (e.g., [min_magnitude, max_magnitude])
    uncertainty_bounds: Tuple[float, float]
    # Causal dependencies (IDs of tasks or impact vectors that must precede/synergize)
    causal_dependencies: List[str] = field(default_factory=list)
    # Domain-specific weight parameters for synergy-based scaling
    domain_weights: Dict[str, float] = field(default_factory=dict)
    # Extensible metadata for plugging in domain-specific metrics
    metrics: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """
        Validates the impact vector data to ensure robustness and fail-fast behavior.
        """
        import math
        if math.isnan(self.magnitude) or math.isinf(self.magnitude) or self.magnitude <= 0:
            raise ValueError(f"Invalid magnitude: {self.magnitude}")
            
        if self.time_horizon < 0:
            raise ValueError(f"Negative time horizon: {self.time_horizon}")
            
        if any(math.isnan(b) or math.isinf(b) for b in self.uncertainty_bounds):
            raise ValueError(f"Invalid uncertainty bounds: {self.uncertainty_bounds}")
            
        if self.uncertainty_bounds[0] > self.uncertainty_bounds[1]:
            raise ValueError(f"Uncertainty bounds must be [min, max], got {self.uncertainty_bounds}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category.value,
            "magnitude": self.magnitude,
            "time_horizon": self.time_horizon,
            "uncertainty_bounds": self.uncertainty_bounds,
            "causal_dependencies": self.causal_dependencies,
            "domain_weights": self.domain_weights,
            "metrics": self.metrics
        }

@dataclass
class ImpactProjection:
    """
    The result of a forecasting pipeline, representing a probabilistic distribution
    of expected total impact across multiple orders of effects.
    """
    task_id: str
    target_vector: ImpactVector # First-order impact
    distribution_mean: float
    distribution_std: float
    confidence_interval: Tuple[float, float]
    # Detailed breakdown of effects (first, second, third order)
    effect_chain: List[ImpactVector] = field(default_factory=list)
    # Recursion/Recalibration metadata
    timestamp: float = field(default_factory=lambda: time.time())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "target_vector": self.target_vector.to_dict(),
            "distribution_mean": self.distribution_mean,
            "distribution_std": self.distribution_std,
            "confidence_interval": self.confidence_interval,
            "effect_chain": [v.to_dict() for v in self.effect_chain],
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ImpactProjection":
        target_vector_data = data["target_vector"]
        target_vector = ImpactVector(
            category=ImpactCategory(target_vector_data["category"]),
            magnitude=target_vector_data["magnitude"],
            time_horizon=target_vector_data["time_horizon"],
            uncertainty_bounds=tuple(target_vector_data["uncertainty_bounds"]),
            causal_dependencies=target_vector_data.get("causal_dependencies", []),
            domain_weights=target_vector_data.get("domain_weights", {}),
            metrics=target_vector_data.get("metrics", {})
        )
        
        effect_chain = []
        for v_data in data.get("effect_chain", []):
            effect_chain.append(ImpactVector(
                category=ImpactCategory(v_data["category"]),
                magnitude=v_data["magnitude"],
                time_horizon=v_data["time_horizon"],
                uncertainty_bounds=tuple(v_data["uncertainty_bounds"]),
                causal_dependencies=v_data.get("causal_dependencies", []),
                domain_weights=v_data.get("domain_weights", {}),
                metrics=v_data.get("metrics", {})
            ))

        return cls(
            task_id=data["task_id"],
            target_vector=target_vector,
            distribution_mean=data["distribution_mean"],
            distribution_std=data["distribution_std"],
            confidence_interval=tuple(data["confidence_interval"]),
            effect_chain=effect_chain,
            timestamp=data.get("timestamp", time.time()),
            metadata=data.get("metadata", {})
        )


@dataclass
class SurplusPool:
    """
    A shared pool of value derived from a task cluster's collective impact.
    """
    cluster_id: str
    total_surplus: float
    confidence_interval: Tuple[float, float]
    aggregated_vectors: Dict[str, float] # Category name to total magnitude
    task_ids: List[str]
    timestamp: float = field(default_factory=lambda: time.time())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "total_surplus": self.total_surplus,
            "confidence_interval": self.confidence_interval,
            "aggregated_vectors": self.aggregated_vectors,
            "task_ids": self.task_ids,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }


@dataclass
class ContributionClaim:
    """
    A structured claim representing an agent's marginal contribution to a surplus pool.
    Produced by counterfactual modeling.
    """
    agent_id: str
    cluster_id: str
    # V(All) - V(All - agent)
    marginal_impact_estimate: float
    # Based on the difference in uncertainty bounds/intervals
    uncertainty_margin: float
    # Measure of how much this agent's presence influenced synergy/dependencies
    dependency_influence_weight: float
    # List of task IDs the agent contributed
    task_ids: List[str]
    timestamp: float = field(default_factory=lambda: time.time())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "cluster_id": self.cluster_id,
            "marginal_impact_estimate": self.marginal_impact_estimate,
            "uncertainty_margin": self.uncertainty_margin,
            "dependency_influence_weight": self.dependency_influence_weight,
            "task_ids": self.task_ids,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }
