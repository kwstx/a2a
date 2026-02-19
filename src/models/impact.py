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


