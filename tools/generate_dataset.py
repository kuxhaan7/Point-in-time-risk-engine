#!/usr/bin/env python3
"""Generate a deterministic dataset containing realistic temporal hazards."""

from __future__ import annotations

import argparse
import json
import math
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path


def emit(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def iso(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def generate(seed: int, shipments: int) -> tuple[list[dict], list[dict], list[dict]]:
    rng = random.Random(seed)
    origin = datetime(2026, 1, 1, tzinfo=timezone.utc)
    events: list[dict] = []
    labels: list[dict] = []
    decisions: list[dict] = []

    for number in range(shipments):
        sid = f"s-{number:05d}"
        start = origin + timedelta(hours=number * 3)
        carrier = ("north", "central", "coast")[number % 3]
        latent = rng.random() + (0.18 if carrier == "coast" else 0.0)
        incident = latent > 0.82
        incident_at = start + timedelta(hours=17, minutes=number % 37)

        for hour in range(16):
            device_time = start + timedelta(hours=hour)
            temp = 3.8 + 0.08 * hour + rng.gauss(0.0, 0.55)
            if incident and hour >= 10:
                temp += (hour - 9) * 0.9
            event_id = f"{sid}-temp-{hour:02d}"
            delay_minutes = rng.choice([0, 1, 2, 5, 35, 180])
            received = device_time + timedelta(minutes=delay_minutes)
            events.append(
                {
                    "event_id": event_id,
                    "revision": 1,
                    "shipment_id": sid,
                    "device_time": iso(device_time),
                    "received_at": iso(received),
                    "kind": "temperature_c",
                    "value": round(temp, 3),
                    "source": f"sensor-{carrier}",
                    "payload": {"firmware": "4.8.1" if number % 5 else "4.7.9"},
                }
            )

            if hour in (4, 11):
                # An exact redelivery.
                events.append(dict(events[-1]))

            if hour == 8 and number % 7 == 0:
                # A later correction must not rewrite decisions made before arrival.
                corrected = dict(events[-1])
                corrected["revision"] = 2
                corrected["received_at"] = iso(received + timedelta(hours=8))
                corrected["value"] = round(temp - 2.25, 3)
                corrected["payload"] = {"firmware": "4.8.1", "correction": "calibrated"}
                events.append(corrected)

        # A post-incident operational event has a misleading old device clock.
        # It is legitimate telemetry, not a special field to strip by name.
        if incident:
            events.append(
                {
                    "event_id": f"{sid}-ticket",
                    "revision": 1,
                    "shipment_id": sid,
                    "device_time": iso(start + timedelta(hours=9)),
                    "received_at": iso(incident_at + timedelta(minutes=20)),
                    "kind": "door_open",
                    "value": 1.0,
                    "source": "ops-console",
                    "payload": {"firmware": None},
                }
            )
            labels.append(
                {
                    "incident_id": f"inc-{number:05d}",
                    "shipment_id": sid,
                    "incident_at": iso(incident_at),
                    "label_available_at": iso(incident_at + timedelta(hours=18 + number % 11)),
                    "severity": 1 + number % 3,
                }
            )

        for hour in (8, 11, 14):
            decisions.append({"shipment_id": sid, "decision_time": iso(start + timedelta(hours=hour))})

    # Delivery order is correlated with received time but not perfectly sorted.
    events.sort(key=lambda row: (row["received_at"], row["event_id"], row["revision"]))
    for idx in range(0, len(events) - 3, 41):
        events[idx], events[idx + 3] = events[idx + 3], events[idx]

    return events, labels, decisions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--shipments", type=int, default=600)
    parser.add_argument("--output", type=Path, default=Path(__file__).parents[1] / "data")
    args = parser.parse_args()
    events, labels, decisions = generate(args.seed, args.shipments)
    emit(args.output / "events.jsonl", events)
    emit(args.output / "labels.jsonl", labels)
    emit(args.output / "decision_times.jsonl", decisions)
    summary = {
        "decision_times": len(decisions),
        "events": len(events),
        "labels": len(labels),
        "seed": args.seed,
        "shipments": args.shipments,
    }
    (args.output / "MANIFEST.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()

