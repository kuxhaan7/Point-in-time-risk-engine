# Decision Record

One rule runs through all of it: a score may only use what the platform actually knew at the instant it was asked.

`solution.py` exposes three names — `build_training_rows`, `train`, `RiskEngine`. The point-in-time kernel (`_visible_events` → `_features` → `_predict_proba`) lives as private statics on `RiskEngine`, and the offline builder replays through it rather than reimplementing it. Sharing one kernel makes train/serve skew unexpressible rather than merely unlikely.

## Point-in-time policy

- **`received_at` is the only clock that gates visibility.** A revision is usable at `t` only if `received_at <= t`. `device_time` is a claim about when a measurement happened, never about when we could act on it — it feeds trend and recency features only.
- **Corrections are never applied retroactively.** If revision 2 lands after a decision time, that decision keeps revision 1 forever: the wrong value in hindsight, the value we actually had. This is the policy §1 of the README asks for.
- **Full revision history is kept per shipment**, not just the newest — `score(id, as_of)` must answer for an `as_of` earlier than something already ingested.
- **Ties are broken deterministically.** Within a `kind` the latest reading is by `(device_time, received_at, event_id)`; the third key exists only so replays agree.
- **The horizon is half-open, `(t, t+6h]`.** An incident at `t` is already underway; one at the boundary counts.

## Evaluation

- **Split: grouped by shipment, ordered in time.** Shipments sort by earliest decision time; the last 20% are held out whole, so no eval row shares its own shipment's telemetry with training.
- **Imputation and scaling are fit on training rows only** — statistics over all rows would leak the holdout into the model.
- **Labels and features never touch.** Features come from events alone, labels from incident records alone. `label_available_at` is ignored: it governs when you may *train*, not whether an incident happened.
- **Reported: average precision, Brier, a constant-prevalence baseline, and per-source and per-richness slices.** Prevalence is printed alongside, because AP at 8% positives means something very different than at 50%.
- **The holdout is scored through `_predict_proba()`, the serving function** — so the evaluation also proves the persisted artifact reproduces training-time scores.

## Model

Ridge logistic regression by Newton's method: no learning rate, no RNG, no minibatching, so identical rows give an identical `model_version`. Feature names derive from the `kind`s present, not a fixed schema, so an unseen kind adds columns instead of crashing. Every numeric is hand-rolled on the stdlib — the evaluator has no network access, so `pyproject.toml` declares zero runtime dependencies. `model.json` carries columns, medians, means, stds, weights, bias and version: enough to reproduce any score in a fresh process.

## State and concurrency

- **Idempotency key is `(event_id, revision)`.** A seen key is a no-op returning `False`; any new key is new information, including a late lower revision.
- **`max_shipments` is an `OrderedDict` used as an LRU** — the store is also the eviction index, so no auxiliary structure can grow. Eviction is amnesia: an evicted shipment restarts empty, the accepted cost of bounding by shipment count.
- **One `RLock` guards the shipment map and the model reference.** `score()` takes both under it, then does the arithmetic outside, so a reload can't swap the model mid-prediction.
- **Reload validates before it activates** — required keys, `feature_columns`/`weights` agreement, finite coefficients. On failure the previous model keeps serving and `reload_model()` returns `False`. A NaN weight is rejected here rather than at `to_wire()`, far from the deploy that caused it.
- **`snapshot()` builds its payload under the lock, then writes temp → fsync → rename.** Building under the lock is the harder half: bytes that parse cleanly can still describe an inconsistent state. `note_5_...` is mutation-checked — removing the lock fails it 100/100.

## Rejected or reinterpreted customer notes

| # | Note | Verdict |
|---|---|---|
| 1 | Random 80/20 row split | **Rejected.** Mixes past and future shipments and lets one shipment sit on both sides — measures memorization, not future performance. |
| 2 | Always apply newest revision when replaying | **Rejected.** Lets a correction rewrite a decision made before it arrived, training on information serving will never have. |
| 3 | Sort by device time before replay | **Rejected.** A stale device clock would grant a future event early visibility. We order by `received_at`. |
| 4 | Deduplicate on `shipment_id` | **Rejected.** Collapses genuinely distinct events; the record's identity is `(event_id, revision)`. |
| 5 | Kafka is exactly-once, so snapshots need no app logic | **Rejected.** Broker semantics say nothing about a local reader seeing a half-written file. |
| 6 | Return `0.0` if the model can't load | **Rejected.** A confidently wrong answer wearing a healthy status code. We keep serving the old model and report the failure. |
| 7 | Use the full incident table in feature generation | **Rejected.** Authoritative but not *knowable* at decision time — textbook leakage. Incidents build labels only. |
| 8 | AUC > 0.90 is sufficient for launch | **Reinterpreted.** One threshold on one ranking metric says nothing about calibration or about who the model fails; hence AP, Brier, baseline and slices. |
| 9 | Keep every shipment in memory | **Rejected.** Unbounded growth, and the spec caps it. |
| 10 | Reload may briefly clear state | **Rejected.** Clearing loses history late corrections still need, and "low traffic" is an assumption, not a guarantee. |

## Limitations

- **The headline number is too good to trust.** Eval AP of 0.98 reflects a generator whose excursion rule is nearly deterministic given the features — evidence the pipeline is wired correctly, not an estimate of production skill.
- **Calibration is measured, not corrected** — Brier is reported; no reliability curve, no isotonic or Platt step.
- **Slices with no positives report `nan`** — AP is undefined without a positive, and `nan` is honest, but the field needs reading alongside `n` and `positive_rate`.
- **No time-decay on features**; every visible event counts equally regardless of age.
- **Snapshot is a full rewrite**, proportional to records held — fine at `max_shipments=32`, not a design for large state.
- **`restore()` trusts its snapshot**, on the assumption it is our own file.
- **Not attempted in the timebox:** cross-validated hyperparameter selection (`L2_PENALTY` fixed at 1.0), feature selection, per-source models, live drift monitoring.

## Reproduction

```bash
python3 tools/generate_dataset.py      # deterministic
python3 -m pip install -e '.[dev]'
python3 -m pytest
```

On the generated dataset (11,019 events, 133 incidents, 1,800 decision times): 120 shipments / 360 rows held out whole; eval AP 0.983 against a 0.076 constant baseline; Brier 0.0042; prevalence 0.077 train / 0.081 eval. Model bytes and replayed prediction streams are byte-identical across independent runs, which `two_independent_replays_produce_byte_identical_predictions_and_snapshots` asserts directly.
