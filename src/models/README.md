# Vocational Impact Ontology

This directory defines the formal ontology for the economic engine's impact modeling.

## Core Concepts

### Impact Vector
Instead of a scalar reward, every task in the system is evaluated as a multi-dimensional **Impact Vector**. This allows the engine to account for temporal delay, uncertainty, and cross-domain synergy.

| Dimension | Description |
|-----------|-------------|
| **Outcome Category** | The domain of the impact (e.g., Revenue, Research, Efficiency). |
| **Magnitude Projection** | The expected scale of the impact. |
| **Time Horizon** | The projected duration until the impact is fully realized. |
| **Uncertainty Bounds** | The confidence interval or variance of the projection. |
| **Causal Dependencies** | Upstream tasks or conditions required for this impact. |
| **Domain Weights** | Parameters that define how this impact scales in different contexts. |

## Extensibility

The `ImpactVector` is designed to be domain-agnostic. Metrics such as "Revenue Growth", "Citations", or "Infrastructure Robustness" are mapped into this schema as `metrics` mappings, ensuring the core valuation engine remains decoupled from specific KPI definitions.
