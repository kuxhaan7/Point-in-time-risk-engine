from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class TelemetryEvent:
    event_id: str
    revision: int
    shipment_id: str
    device_time: datetime
    received_at: datetime
    kind: str
    value: float | str | None
    source: str
    payload: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class TrainingRow:
    shipment_id: str
    decision_time: datetime
    features: Mapping[str, float | int | str | None]
    label: int
    metadata: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class Prediction:
    shipment_id: str
    as_of: datetime
    probability: float
    model_version: str
    feature_digest: str
    degraded: bool
    reasons: tuple[str, ...]

    def to_wire(self) -> bytes:
        """Canonical serialized form used by replay checks."""
        import json

        value = {
            "as_of": self.as_of.isoformat(),
            "degraded": self.degraded,
            "feature_digest": self.feature_digest,
            "model_version": self.model_version,
            "probability": self.probability,
            "reasons": list(self.reasons),
            "shipment_id": self.shipment_id,
        }
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")

