# Point-in-Time Risk Engine

Online risk engine for refrigerated shipments: estimates the probability of a **temperature excursion in the next six hours**, re-scoring after every accepted event. Events arrive at least once, late, out of order, corrected by a higher revision, and sometimes with a bad device clock.

One rule runs through it: **a score may only use what had arrived by the decision time.** `received_at` gates visibility; `device_time` is a claim, not a permission.

Zero runtime dependencies (stdlib only, numerics included). Deterministic — two replays produce byte-identical predictions and snapshots. Thread-safe under concurrent ingest, score, snapshot and reload.

## Run

```bash
python3 tools/generate_dataset.py      # deterministic; writes data/
python3 -m pip install -e '.[dev]' && python3 -m pytest
python3 tools/demo.py                  # full pipeline, end to end
```

## API

```python
build_training_rows(events, labels, decision_times) -> list[TrainingRow]
train(rows, artifact_dir)                           -> report; writes model.json + metrics.json
RiskEngine(artifact_dir, max_shipments=10_000)      # ingest / score / reload_model / snapshot / restore / stats
```

## Implementation

[solution.py](src/dispatch_risk/solution.py)

- **One kernel, offline and online.** [`_visible_events`](src/dispatch_risk/solution.py#L360) → [`_features`](src/dispatch_risk/solution.py#L378) → [`_predict_proba`](src/dispatch_risk/solution.py#L444) are statics on `RiskEngine`; the offline builder replays through them and `train()` scores its holdout through them. Train/serve skew has no second implementation to drift from.
- **Visibility.** Highest revision per `event_id` with `received_at <= as_of`. Corrections are never applied retroactively. Full revision history is kept, since `score()` must answer for an `as_of` earlier than something already ingested.
- **Features** derive from the `kind`s present, not a fixed schema (9 columns on the sample data): counts, recency, clock skew, and per kind `_last_value` / `_mean` / `_max` / `_slope_per_hr`. No `_min` — the target is an excursion *above* a threshold.
- **Idempotency** keys on `(event_id, revision)`; a seen key is a no-op, any new key is new information.
- **Bounded state** via an `OrderedDict` LRU — the store is the eviction index, so nothing auxiliary can grow.
- **Concurrency**: one `RLock` guards the shipment map and the model reference, held while touching state, released before the arithmetic — a reload can't swap the model mid-prediction.
- **Reload validates before activating** (keys, column/weight agreement, finite coefficients). On failure the old model keeps serving and `reload_model()` returns `False`.
- **Snapshot** builds its payload under the lock, then writes temp → `fsync` → `rename`. Bytes that parse cleanly can still describe an inconsistent state.

## Model

**Ridge logistic regression by Newton's method (IRLS)**, hand-rolled on the stdlib ([`fit_logistic`](src/dispatch_risk/solution.py#L129)). No learning rate, no RNG, no minibatching — identical rows give identical weights and an identical `model_version` (a sha256 content hash). Bias unpenalized so the model can match the base rate. Median imputation and z-scoring fit on **training rows only**. `model.json` carries columns, medians, means, stds, weights, bias and version: enough to reproduce any score in a fresh process.

A linear model is the deliberate choice — the interesting risk here is leakage and replay correctness, not model capacity.

**Evaluation.** Split grouped by shipment and ordered in time: the last 40% of shipments held out whole, so no eval row shares its shipment's telemetry with training. Labels come from incidents alone, features from events alone; positive iff an incident falls in the half-open `(as_of, as_of + 6h]`.

| Holdout 720 rows / 240 shipments | prevalence 0.076 |
|---|---|
| Average precision | **0.977** (constant baseline 0.079) |
| Brier | 0.00497 |
| By source | central 1.000 · coast 0.985 · north 0.949 |
| By richness | sparse 1.000 · rich 0.979 |

Read the headline skeptically: the generator's excursion rule is nearly deterministic given the features. It shows the pipeline is wired correctly, not production skill.

## Customer notes

Ten notes came from customer engineers, to be treated as requirements unless unsafe. Nine rejected, one reinterpreted — each pinned by a test named for its number.

| # | Note | Verdict |
|---|---|---|
| 1 | Random 80/20 row split | **Rejected** — puts one shipment on both sides; measures memorization. |
| 2 | Always apply the newest revision | **Rejected** — rewrites decisions made before the correction arrived. |
| 3 | Sort by `device_time` before replay | **Rejected** — a stale clock would grant early visibility. |
| 4 | Deduplicate on `shipment_id` | **Rejected** — identity is `(event_id, revision)`. |
| 5 | Kafka is exactly-once, snapshots need no app logic | **Rejected** — says nothing about a half-written local file. |
| 6 | Return `0.0` if the model can't load | **Rejected** — a confidently wrong answer with a healthy status code. |
| 7 | Use the full incident table in features | **Rejected** — not knowable at decision time; leakage. |
| 8 | AUC > 0.90 is sufficient | **Reinterpreted** — says nothing about calibration or who the model fails; hence AP, Brier, baseline, slices. |
| 9 | Keep every shipment in memory | **Rejected** — unbounded, and the spec caps it. |
| 10 | Reload may briefly clear state | **Rejected** — loses history late corrections still need. |

Full reasoning in [DECISIONS.md](DECISIONS.md).

## Tests

[tests/test_public_contract.py](tests/test_public_contract.py) — 27 invariants plus the wire-format check, covering leakage (corrections after a decision, early-delivered future events, labels never reaching features), replay determinism, the half-open window, bounded state at evaluator scale (`max_shipments=32`, >10,000 events), reload rejection paths, concurrency, and one test per rejected customer note.

## Limitations

Calibration is measured, not corrected. No time decay on features. Snapshot is a full rewrite. `restore()` trusts its snapshot. Eviction is amnesia — the accepted cost of bounding by shipment count. Not attempted: hyperparameter search, feature selection, per-source models, drift monitoring.
