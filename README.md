# Take-Home: Point-in-Time Risk Engine

**Role:** Senior Machine Learning Software Engineer  
**Expected effort:** 7 hours maximum  
**Language:** Python 3.11+  

Do not spend time on UI, infrastructure-as-code, or presentation slides. We assess the repository, executable behavior, tests, and technical decisions.

## Scenario

You own an online risk engine for refrigerated shipments. A device stream reports temperature, door, compressor, and location events. The engine must estimate whether a shipment will suffer a temperature excursion in the next six hours.

Events are delivered at least once. They can arrive late, be corrected, or carry a bad device clock. Operations wants a prediction after every accepted event. Labels arrive later from audited incident records.

You are given a deterministic data generator and a small starter interface. Build a thin training-and-serving implementation that is correct under replay and usable under concurrent scoring and model reload.

## Input records

Telemetry is JSON Lines. Every event contains:

```json
{
  "event_id": "evt-123",
  "revision": 2,
  "shipment_id": "s-17",
  "device_time": "2026-01-12T10:03:00-05:00",
  "received_at": "2026-01-12T15:07:13Z",
  "kind": "temperature_c",
  "value": 9.4,
  "source": "sensor-v2",
  "payload": {"firmware": "4.8.1"}
}
```

Labels are also JSON Lines:

```json
{
  "incident_id": "inc-88",
  "shipment_id": "s-17",
  "incident_at": "2026-01-12T20:15:00Z",
  "label_available_at": "2026-01-14T09:00:00Z",
  "severity": 2
}
```

Important semantics:

- `(event_id, revision)` identifies a delivered record.
- A higher revision supersedes lower revisions with the same `event_id`.
- `device_time` is when the device claims the measurement occurred.
- `received_at` is when the platform could first use that revision.
- Input order is delivery order and is not guaranteed to be chronological.
- All public outputs must use UTC-aware timestamps.
- An incident is positive for an `as_of` time if it occurs in `(as_of, as_of + 6 hours]`.

## Required implementation

Implement the functions and class in `src/dispatch_risk/solution.py`. You may change other starter files while keeping the public contract stable.

### 1. Point-in-time training set

Implement:

```python
build_training_rows(events, labels, decision_times)
```

For each `(shipment_id, decision_time)`, produce features, a binary label, and metadata. A feature may use only information that would have been available to the platform at that decision time.

You must make and document a policy for corrected events whose latest revision arrives after a decision time.

### 2. Model training and artifact

Implement:

```python
train(rows, artifact_dir)
```

The artifact must be loadable in a fresh Python process and include everything required to reproduce feature interpretation and scoring.

Report at least:

- the evaluation split rule;
- PR-AUC or average precision;
- one probability-quality metric;
- performance against a constant or simple rule baseline;
- results for at least two operationally meaningful slices.

Do not optimize for a leaderboard. We care more about whether the evaluation estimates future behavior.

### 3. Online engine

Implement `RiskEngine` with the contract in `contracts.py`.

It must:

- accept duplicate, late, out-of-order, and corrected events;
- make repeated delivery of the same record idempotent;
- score a shipment as of an explicit UTC timestamp;
- return the model version and a deterministic feature digest;
- snapshot and restore its state;
- enforce `max_shipments` without unbounded auxiliary structures;
- reload a valid model while scoring requests are active;
- keep serving the previous model if a reload fails.

Two identical replays from an empty state must produce byte-identical serialized predictions and snapshots.

### 4. Tests

Add tests for the failure modes you think are most dangerous. We will run public and private tests, including generated streams not present in this repository.

### 5. Decision record

Complete `DECISIONS.md`. Keep it concise and concrete. Explicitly state anything you chose not to implement within the timebox.

## Customer-provided implementation notes

The following notes came from different customer engineers. Treat them as requirements unless you believe one is unsafe or technically invalid. If you reject or reinterpret one, do so explicitly in `DECISIONS.md` and implement the safer contract.

1. “Use a random 80/20 row split; we need every carrier represented in both sets.”
2. “Always apply the newest revision of an event, even when replaying an old decision.”
3. “Sort everything by device time before replay so the model sees the real sequence.”
4. “Deduplicate on `shipment_id`; downstream only needs one current prediction.”
5. “Kafka is exactly-once, so snapshot consistency does not need application logic.”
6. “If the model cannot load, return probability `0.0` to preserve the API SLO.”
7. “Use the full incident table during feature generation; it is the authoritative source.”
8. “AUC above 0.90 is sufficient for launch.”
9. “Keep every shipment in memory because historical corrections can arrive at any time.”
10. “Model reload may briefly clear in-memory state; deploys occur during low traffic.”

No clarification is available during the exercise. Make reasonable assumptions and record them.

## Constraints

- The private evaluator has no network access.
- Tests may call the engine from several threads.
- The evaluator may set `max_shipments=32` and replay more than 10,000 events.
- Do not key logic to sample IDs, fixed row counts, or dates in the generated data.
- Avoid hosted APIs and external model services.
- Dependencies must be declared in `pyproject.toml`.

## Run instructions

Generate the sample data:

```bash
python tools/generate_dataset.py
```

Install and run:

```bash
python -m pip install -e '.[dev]'
pytest
```

Your submission should include generated data only if its total compressed size is below 5 MB.

## Follow-up

In a 60-minute technical interview, you will:

- run your solution against a new event stream;
- investigate one failed invariant;
- modify one requirement;
- defend the statistical validity of your evaluation;
- make a small code change while preserving replay determinism.

