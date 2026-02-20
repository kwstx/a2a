from typing import Dict, Any, Callable, List
from .impact import ImpactVector, ImpactCategory

class ImpactMetricRegistry:
    """
    Registry for domain-specific metrics that map to the common ImpactVector schema.
    This allows plugging in revenue, research, or social metrics without changing logic.
    """
    def __init__(self):
        self._mappers: Dict[str, Callable[[Any], ImpactVector]] = {}

    def register_metric(self, name: str, mapper: Callable[[Any], ImpactVector]):
        """
        Registers a mapper function that converts raw domain data into an ImpactVector.
        """
        self._mappers[name] = mapper

    def translate(self, metric_name: str, data: Any) -> ImpactVector:
        if metric_name not in self._mappers:
            raise ValueError(f"Metric '{metric_name}' not registered.")
        return self._mappers[metric_name](data)

# Example Usage (Domain Agnostic Plug-ins)
def revenue_mapper(data: Dict[str, Any]) -> ImpactVector:
    return ImpactVector(
        category=ImpactCategory.REVENUE,
        magnitude=data.get("expected_value", 0.0) or data.get("amount", 0.0),
        time_horizon=data.get("contract_length", 30.0),
        uncertainty_bounds=(data.get("min", 0), data.get("max", 0)),
        domain_weights={"market_cap": 0.8, "agent_liquidity": 0.2},
        metrics={"currency": "CRED", "recurring": data.get("is_recurring", False)},
        causal_dependencies=data.get("causal_dependencies", [])
    )

def research_mapper(data: Dict[str, Any]) -> ImpactVector:
    return ImpactVector(
        category=ImpactCategory.RESEARCH,
        magnitude=data.get("novelty_score", 0.0),
        time_horizon=365.0, # Long term
        uncertainty_bounds=(0.1, 1.0),
        domain_weights={"academic_impact": 0.9, "patent_potential": 0.1},
        metrics={"citations_estimate": data.get("est_citations", 0)},
        causal_dependencies=data.get("causal_dependencies", [])
    )

def technical_mapper(data: Dict[str, Any]) -> ImpactVector:
    return ImpactVector(
        category=ImpactCategory.TECHNICAL,
        magnitude=data.get("complexity", 1.0) * data.get("impact_factor", 1.0),
        time_horizon=data.get("time_horizon", 30.0),
        uncertainty_bounds=(0.5, 2.0),
        domain_weights={"robustness": 0.5, "extensibility": 0.5},
        metrics={"loc": data.get("loc", 0)},
        causal_dependencies=data.get("causal_dependencies", [])
    )

