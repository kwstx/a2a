from dataclasses import dataclass, field
from typing import Dict, Any, Optional

@dataclass
class Task:
    """
    Represents a task entering the system.
    """
    id: str
    domain: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "domain": self.domain,
            "metrics": self.metrics,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Task":
        return cls(
            id=data["id"],
            domain=data["domain"],
            metrics=data.get("metrics", {}),
            metadata=data.get("metadata", {})
        )
