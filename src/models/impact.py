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
