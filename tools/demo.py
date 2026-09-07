"""Run the three entry points end to end against a dataset.

    python3 tools/demo.py                          # committed data/
    python3 tools/demo.py --data /tmp/new-stream   # any generated stream
"""
from __future__ import annotations

import argparse
import json
import tempfile
from datetime import datetime
from pathlib import Path

from dispatch_risk import RiskEngine, TelemetryEvent, build_training_rows, train


def _dt(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(__file__).parents[1] / "data")
    parser.add_argument("--artifact", type=Path, default=None)
    parser.add_argument("--max-shipments", type=int, default=32)
    args = parser.parse_args()

    events = [
        TelemetryEvent(
            r["event_id"], r["revision"], r["shipment_id"], _dt(r["device_time"]),
            _dt(r["received_at"]), r["kind"], r["value"], r["source"], r["payload"],
        )
        for r in _jsonl(args.data / "events.jsonl")
    ]
    labels = _jsonl(args.data / "labels.jsonl")
    decisions = [(r["shipment_id"], _dt(r["decision_time"]))
                 for r in _jsonl(args.data / "decision_times.jsonl")]

    print(f"[1] build_training_rows  <- {len(events)} events, {len(labels)} incidents, "
          f"{len(decisions)} decision times")
    rows = build_training_rows(events, labels, decisions)
    print(f"    {len(rows)} rows, {sum(r.label for r in rows)} positive, "
          f"{len(rows[0].features)} features")

    artifact = args.artifact or Path(tempfile.mkdtemp()) / "artifact"
    report = train(rows, artifact)
    print(f"\n[2] train  -> {artifact}")
    print(f"    split      {report['split_rule']}")
    print(f"    holdout    {report['n_rows_eval']} rows / {report['n_shipments_eval']} shipments, "
          f"prevalence {report['eval_prevalence']:.3f}")
    print(f"    avg prec   {report['eval_average_precision']:.4f}  "
          f"(constant baseline {report['baseline_constant_average_precision']:.4f})")
    print(f"    brier      {report['eval_brier_score']:.5f}")
    for dimension, group in report["slices"].items():
        print(f"    {dimension}:")
        for name, metrics in group.items():
            if metrics.get("n"):
                print(f"        {name:<38} n={metrics['n']:<5} AP={metrics['average_precision']:.3f}")

    engine = RiskEngine(artifact, max_shipments=args.max_shipments)
    print(f"\n[3] RiskEngine  {engine.stats()['model_version']}, "
          f"max_shipments={args.max_shipments}")
    print("    replaying every event in delivery order...")
    for event in sorted(events, key=lambda e: (e.received_at, e.event_id, e.revision)):
        engine.ingest(event)
        engine.score(event.shipment_id, event.received_at).to_wire()

    snapshot = Path(tempfile.mkdtemp()) / "state.json"
    engine.snapshot(snapshot)
    restored = RiskEngine.restore(artifact, snapshot)
    print(f"    {json.dumps(dict(engine.stats()))}")
    print(f"    snapshot {snapshot.stat().st_size:,} bytes -> "
          f"restore matches: {restored.stats() == engine.stats()}")

    # Score a shipment still resident after the replay; an arbitrary one may
    # have been evicted by max_shipments and would score degraded.
    sample = max(events, key=lambda e: e.received_at)
    prediction = engine.score(sample.shipment_id, sample.received_at)
    print(f"\n    sample prediction:\n    {prediction.to_wire().decode()}")


if __name__ == "__main__":
    main()
