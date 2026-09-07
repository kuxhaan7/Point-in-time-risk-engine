# Decision Record

Everything here is implemented in `src/dispatch_risk/solution.py` and asserted in
`tests/test_public_contract.py`. One rule runs through all of it: a score may only use what the platform actually knew at the instant it was asked.

**Module shape.** `solution.py` exposes exactly three names — `build_training_rows`, `train`, `RiskEngine`. The point-in-time kernel (`_visible_events` → `_features` → `_predict_proba`) lives as private static methods on `RiskEngine`, and the offline builder replays through it rather than reimplementing it; training-only numerics are nested inside `train()`. Sharing one kernel is what makes train/serve feature skew unexpressible rather than merely unlikely.

## Event-time and knowledge-time policy

- **`received_at` is the only clock that gates visibility.** A revision is usable at time `t` only if `received_at <= t`; `device_time` is a claim about when a measurement happened, never about when we could act on it.
- **Corrections are never applied retroactively.** If revision 2 lands after a decision time, that decision keeps revision 1 forever — the wrong value in hindsight, but the value we actually had.
- **A bad device clock cannot buy early visibility.** It can only shift feature *content* (recency, trend), never *eligibility*, so a sensor reporting a 30-day-old timestamp still can't reach back into an earlier decision.
- **Ordering is deterministic, not chronological.** Input arrives in delivery order, so within a `kind` we pick the latest reading by `(device_time, received_at, event_id)` — the third key only exists so ties resolve the same way on every replay.

## Label construction and leakage controls

- **The horizon is half-open, `(t, t+6h]`.** An incident exactly at `t` is already underway and isn't a prediction; one landing exactly on the 6h boundary counts.
- **Labels and features never touch.** Features are computed from events alone, the label from incident records alone, so the incident table cannot leak into a feature no matter what it contains.
- **`label_available_at` is deliberately ignored.** It describes reporting lag, which governs when you may *train*, not whether a historical incident happened.
- **The split is grouped by shipment and ordered in time.** Shipments sort by their earliest decision time and the last 20% are held out whole, so no eval row can share its own shipment's earlier telemetry with the training set.
- **Imputation and scaling statistics are fit on training rows only.** Medians, means and standard deviations computed over all rows would quietly leak the holdout into the model.

## Features and model

- **Feature names are derived from the `kind`s present, not a fixed schema.** An unseen event kind produces new columns instead of a crash; `train()` unions every row's keys into one persisted schema and imputes what a given row lacks.
- **Per kind: count, age and value of the latest reading, mean, max, slope per hour.** There is no `_min` — the target is an excursion *above* a threshold, so the coldest reading in the window carries no signal for it.
- **Ridge logistic regression, fit by Newton's method (IRLS).** No learning rate to tune, converges in a handful of iterations at this width, and is fully deterministic — no RNG, no minibatching, so identical rows always produce an identical `model_version`.
- **Every numeric is hand-rolled on the standard library.** The evaluator has no network access, so neither numpy nor scikit-learn can be assumed installable; `pyproject.toml` declares zero runtime dependencies and a clean-venv install loads no third-party module. The one exception is `statistics.median`, which is stdlib and bit-identical to the hand-rolled version it replaced.
- **The logistic function saturates instead of overflowing.** A score below −709 returns `0.0` rather than raising `OverflowError` from `math.exp`; in range the arithmetic is unchanged, so this costs nothing and removes a crash a degenerate model could cause.
- **The artifact is self-contained.** `model.json` carries columns, medians, means, standard deviations, weights, bias and version — enough to reproduce any score in a fresh process with no access to the training data.
- **The holdout is scored through `RiskEngine._predict_proba()`, the same function that serves.** That makes the evaluation a proof that the persisted artifact reproduces training-time scores, rather than a separate code path that might not.
- **Reported: average precision, Brier score, a constant-prevalence baseline, and two slices.** Prevalence is printed alongside because it is what makes the rest readable — average precision at 8% positives means something very different than at 50%.

## State, idempotency, and eviction

- **The idempotency key is `(event_id, revision)`** — the identity the spec assigns a delivered record. Re-delivery of a seen key is a no-op returning `False`; any new key is new information, including a lower revision arriving very late.
- **Full revision history is kept per shipment, not just the newest.** `score(id, as_of)` has to answer for an `as_of` earlier than something already ingested, and that is only correct if the not-yet-visible revision hasn't already overwritten the one it supersedes.
- **`max_shipments` is enforced by an `OrderedDict` used as an LRU.** The same dict is both the store and the eviction index, so there is no auxiliary structure left to grow unbounded.
- **Eviction is amnesia, and we accept it.** An evicted shipment restarts empty, so a correction arriving afterwards is treated as a first sighting — the deliberate cost of bounding memory by shipment count.

## Concurrency and model reload

- **One `RLock` guards the shipment map and the model reference, and nothing else.** It is held while reading or mutating state, never while computing features or scoring.
- **`score()` takes its events and model under the lock, then does the arithmetic outside it.** A reload landing mid-score therefore cannot swap the model out from under a half-computed prediction, and the critical section stays proportional to one shipment's records rather than to the model.
- **A reload validates before it activates.** Required keys, `feature_columns`/`weights` agreement, and finiteness of every coefficient are checked on the candidate; if any fails, the previous model keeps serving and `reload_model()` returns `False`.
- **A NaN weight is rejected at load, not at serialization.** It would otherwise load cleanly and only blow up later inside `Prediction.to_wire()`, which is far from the deploy that caused it.
- **`snapshot()` is write-temp → fsync → rename.** A concurrent reader sees either the whole old file or the whole new one, never a partial write.
- **The payload is also *built* under the lock, which is the harder half.** Bytes that parse cleanly can still describe an inconsistent state if the map mutates mid-assembly; `note_5_...` verifies this by asserting `shipment_order` and `shipments` always agree, and is mutation-checked — deleting the lock makes it fail 100/100 with `OrderedDict mutated during iteration`.
- **Reload leaves ingested state untouched.** Only the model reference is swapped.

## Rejected or reinterpreted customer notes

| # | Note | What we did instead, and why |
|---|---|---|
| 1 | Random 80/20 row split | **Rejected.** A random split mixes past and future shipments across both sides and lets a shipment's own history sit on both, which estimates memorization rather than performance on shipments that haven't played out yet. |
| 2 | Always apply the newest revision, even when replaying | **Rejected.** That lets a correction rewrite a decision made before it arrived, so the model would train on information the platform could not have had — and will not have when serving live. |
| 3 | Sort everything by device time before replay | **Rejected.** `device_time` is a device's claim and can be arbitrarily wrong; sorting by it lets a stale clock grant a future event early visibility. We order by `received_at` and use `device_time` only for trend and recency features. |
| 4 | Deduplicate on `shipment_id` | **Rejected.** That collapses genuinely distinct events on the same shipment; the delivered record's identity is `(event_id, revision)`, which is what we key on. |
| 5 | Kafka is exactly-once, so snapshots need no application logic | **Rejected.** Broker delivery semantics say nothing about whether a local reader sees a half-written file, so `snapshot()` does temp-write-and-rename regardless. |
| 6 | Return probability `0.0` if the model can't load | **Rejected.** That is a confidently wrong answer wearing a healthy status code; we keep the previous model serving and report the failure through `reload_model()`'s return value. |
| 7 | Use the full incident table during feature generation | **Rejected.** It is authoritative but not *knowable* at decision time — using it is textbook label leakage. Incidents build labels only. |
| 8 | AUC above 0.90 is sufficient for launch | **Reinterpreted.** A single threshold on a single ranking metric says nothing about calibration or about who the model fails, so we report average precision, Brier, a baseline, and per-source and per-richness slices. |
| 9 | Keep every shipment in memory | **Rejected.** That is unbounded growth, and the spec explicitly caps it; we bound by `max_shipments` and document eviction amnesia above. |
| 10 | Reload may briefly clear in-memory state | **Rejected.** Clearing state loses history that late corrections still need, and "deploys happen at low traffic" is an assumption, not a guarantee. The swap is a single reference assignment under the lock. |

## Known limitations

- **The headline number is too good to trust.** Eval average precision of 0.98 reflects a generator whose excursion rule is nearly deterministic given the features; treat it as evidence the pipeline is wired correctly, not as an estimate of production skill.
- **Calibration is measured but not corrected.** Brier score is reported; there is no reliability curve and no isotonic or Platt recalibration step.
- **One global model, no per-source specialization.** Source appears as an evaluation slice, not as a feature or a routing key.
- **Snapshot is a full rewrite, not incremental.** Cost is proportional to total records held — fine at `max_shipments=32`, not a design for very large state.
- **No time-decay or windowing on features.** Every visible event counts equally regardless of age, so a long-running shipment's early readings never age out.
- **Slices with no positives report `average_precision: nan`.** Average precision is undefined without a positive, and `nan` is the honest answer rather than a fabricated `0.0` — but on small streams several slices will show it, so the field needs reading alongside `n` and `positive_rate`.
- **`restore()` trusts the snapshot it reads,** on the assumption that it is our own file, not attacker-controlled input.
- **Not attempted in the timebox:** cross-validated hyperparameter selection (`L2_PENALTY` is fixed at 1.0), feature selection, and any monitoring of live feature drift.

## Reproduction

```bash
python3 tools/generate_dataset.py      # writes data/*.jsonl, deterministic
python3 -m pip install -e '.[dev]'
python3 -m pytest                      # 2 passed
```

Training on the generated dataset (11,019 events, 133 incidents, 1,800 decision times):

| | |
|---|---|
| Split | 120 shipments / 360 rows held out, whole, last 20% by earliest decision time |
| Eval average precision | 0.983 |
| Constant-prevalence baseline | 0.076 |
| Eval Brier score | 0.0042 |
| Prevalence | 0.077 train / 0.081 eval |

`train()` writes `model.json` and `metrics.json` to `artifact_dir`. Both the model
bytes and a replayed prediction stream are byte-identical across independent runs,
which `two_independent_replays_produce_byte_identical_predictions_and_snapshots`
asserts directly.
