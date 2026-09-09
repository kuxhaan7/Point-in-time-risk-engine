from __future__ import annotations
import hashlib
import json
import math
import os
import statistics
import threading
from collections import OrderedDict, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from .contracts import Prediction, TelemetryEvent, TrainingRow

HORIZON = timedelta(hours=6)
EVAL_FRACTION = 0.2
L2_PENALTY = 1.0
NEWTON_ITERATIONS = 50
NEWTON_TOL = 1e-8

MODEL_FILENAME = "model.json"
METRICS_FILENAME = "metrics.json"

# Every key a model artifact must carry to be loadable in a fresh process.
MODEL_KEYS = ("bias", "feature_columns", "feature_mean", "feature_std",
              "impute_median", "model_version", "weights")

# Snapshot record shape. The two time fields round-trip through isoformat;
# the rest are already JSON-native.
EVENT_FIELDS = ("event_id", "revision", "shipment_id", "kind", "value", "source", "payload")
EVENT_TIMES = ("device_time", "received_at")


def build_training_rows(
    events: Iterable[TelemetryEvent],
    labels: Iterable[Mapping[str, object]],
    decision_times: Iterable[tuple[str, datetime]],
) -> list[TrainingRow]:
    """Build point-in-time-correct rows, one per (shipment_id, decision_time).

    A revision is usable only if `received_at <= decision_time`: corrections
    are never applied retroactively. See DECISIONS.md.
    """
    events_by_shipment: dict[str, list[TelemetryEvent]] = defaultdict(list)
    for event in events:
        events_by_shipment[event.shipment_id].append(event)

    # Only incident_at is read; `label_available_at` describes reporting lag,
    # which never gates an offline label.
    incidents_by_shipment: dict[str, list[datetime]] = defaultdict(list)
    for label in labels:
        incidents_by_shipment[str(label["shipment_id"])].append(
            RiskEngine._to_utc(RiskEngine._as_datetime(label["incident_at"]))
        )

    rows: list[TrainingRow] = []
    for shipment_id, decision_time in decision_times:
        decision_time = RiskEngine._to_utc(decision_time)

        # Replaying through the engine's own kernel is what rules out train().
        visible = RiskEngine._visible_events(
            events_by_shipment.get(shipment_id, ()), decision_time
        )
        features = RiskEngine._features(visible, decision_time)

        # Positive iff an incident falls in the half-open window (t, t + 6h].
        window_end = decision_time + HORIZON
        label = int(
            any(
                decision_time < incident_at <= window_end
                for incident_at in incidents_by_shipment.get(shipment_id, ())
            )
        )

        # `source` is the slicing dimension train() reports on
        visible_sources = sorted({e.source for e in visible})

        rows.append(
            TrainingRow(
                shipment_id=shipment_id,
                decision_time=decision_time,
                features=features,
                label=label,
                metadata={"source": visible_sources[0] if visible_sources else None},
            )
        )

    return rows


def train(rows: Sequence[TrainingRow], artifact_dir: Path) -> Mapping[str, object]:
    """Fit, evaluate and persist a self-contained artifact; return the report."""
    if not rows:
        raise ValueError("train() requires at least one row")

    is_numeric = RiskEngine._is_numeric

    # -- Numerics, nested because nothing outside training needs them, and
    # -- hand-rolled because the evaluator has no network access (README). --

    def mean(values: Sequence[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    def median(values: Sequence[float]) -> float:
        return statistics.median(values) if values else 0.0

    def population_std(values: Sequence[float]) -> float:
        if not values:
            return 0.0
        m = mean(values)
        return math.sqrt(sum((v - m) ** 2 for v in values) / len(values))

    def solve(matrix: Sequence[Sequence[float]], vector: Sequence[float]) -> list[float]:
        """Solve Ax = b by Gauss-Jordan elimination with partial pivoting."""
        n = len(vector)
        aug = [list(row) + [vector[i]] for i, row in enumerate(matrix)]
        for col in range(n):
            pivot_row = max(range(col, n), key=lambda r: abs(aug[r][col]))
            if abs(aug[pivot_row][col]) < 1e-12:
                raise ValueError("singular matrix while fitting logistic regression")
            aug[col], aug[pivot_row] = aug[pivot_row], aug[col]
            pivot = aug[col][col]
            aug[col] = [v / pivot for v in aug[col]]
            for r in range(n):
                if r != col and aug[r][col] != 0.0:
                    factor = aug[r][col]
                    aug[r] = [aug[r][k] - factor * aug[col][k] for k in range(n + 1)]
        return [row[n] for row in aug]

    def fit_logistic(X: Sequence[Sequence[float]], y: Sequence[float]) -> tuple[float, list[float]]:
        """Ridge logistic regression by Newton's method (IRLS), identical rows, identical weights."""
        n = len(X)
        d = len(X[0]) if n else 0
        design = [[1.0] + list(row) for row in X]  # bias column first
        beta = [0.0] * (d + 1)

        for _ in range(NEWTON_ITERATIONS):
            z = [sum(beta[k] * design[i][k] for k in range(d + 1)) for i in range(n)]
            p = [0.0 if zi < -709.0 else 1.0 / (1.0 + math.exp(-zi)) for zi in z]
            w = [max(pi * (1.0 - pi), 1e-6) for pi in p]

            # Gradient of the penalized log-likelihood; the bias (k == 0) is
            # unpenalized so the model can still match the base rate.
            gradient = [
                sum((y[i] - p[i]) * design[i][k] for i in range(n))
                - (0.0 if k == 0 else L2_PENALTY * beta[k])
                for k in range(d + 1)
            ]

            hessian = [[0.0] * (d + 1) for _ in range(d + 1)]
            for a in range(d + 1):
                for b in range(a, d + 1):
                    value = sum(w[i] * design[i][a] * design[i][b] for i in range(n))
                    if a == b and a != 0:
                        value += L2_PENALTY
                    hessian[a][b] = hessian[b][a] = value

            delta = solve(hessian, gradient)
            beta = [beta[k] + delta[k] for k in range(d + 1)]
            if max(abs(dk) for dk in delta) < NEWTON_TOL:
                break

        return beta[0], beta[1:]

    def average_precision(y_true: Sequence[float], scores: Sequence[float]) -> float:
        """Area under the precision-recall curve, by the step-wise sum."""
        total_positives = sum(y_true)
        if total_positives == 0:
            return float("nan")
        order = sorted(range(len(scores)), key=lambda i: -scores[i])
        cum_tp = ap = prev_recall = 0.0
        for rank, i in enumerate(order, start=1):
            cum_tp += y_true[i]
            ap += (cum_tp / rank) * (cum_tp / total_positives - prev_recall)
            prev_recall = cum_tp / total_positives
        return ap

    def brier(y_true: Sequence[float], probabilities: Sequence[float]) -> float:
        """Mean squared error of the probabilities: the calibration metric."""
        if not y_true:
            return float("nan")
        return sum((p - y) ** 2 for p, y in zip(probabilities, y_true)) / len(y_true)

    def vectorize(features: Mapping[str, object], schema: Mapping[str, Any]) -> list[float]:
        """Feature mapping -> model-order vector: impute missing, then standardize."""
        medians, means = schema["impute_median"], schema["feature_mean"]
        stds = schema["feature_std"]
        vector = []
        for column in schema["feature_columns"]:
            value = features.get(column)
            if value is None or not is_numeric(value):
                value = medians[column]
            vector.append((float(value) - means[column]) / (stds[column] or 1.0))
        return vector

    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # -- Split by whole shipment, ordered in time: shipments sort by earliest
    # -- decision_time and the last EVAL_FRACTION are held out entire, never
    # -- split by row. Rejects customer note #1; see DECISIONS.md. --
    first_decision: dict[str, datetime] = {}
    for row in rows:
        current = first_decision.get(row.shipment_id)
        if current is None or row.decision_time < current:
            first_decision[row.shipment_id] = row.decision_time

    ordered = sorted(first_decision, key=lambda sid: (first_decision[sid], sid))
    n_eval_shipments = max(1, round(len(ordered) * EVAL_FRACTION))
    eval_shipments = set(ordered[len(ordered) - n_eval_shipments :])

    is_eval = [row.shipment_id in eval_shipments for row in rows]
    is_train = [not e for e in is_eval]

    # -- Fit imputation and scaling on training rows only; fitting either on
    # -- the holdout would leak it into the model. --
    train_rows = [row for row, t in zip(rows, is_train) if t]
    columns = sorted({key for row in rows for key in row.features})
    medians = {
        c: median([float(r.features[c]) for r in train_rows if is_numeric(r.features.get(c))])
        for c in columns
    }
    imputed = [
        [float(r.features[c]) if is_numeric(r.features.get(c)) else medians[c] for c in columns]
        for r in train_rows
    ]
    columnwise = [[v[j] for v in imputed] for j in range(len(columns))]
    schema: dict[str, Any] = {
        "feature_columns": columns,
        "impute_median": medians,
        "feature_mean": {c: mean(columnwise[j]) for j, c in enumerate(columns)},
        "feature_std": {c: population_std(columnwise[j]) or 1.0 for j, c in enumerate(columns)},
    }

    labels = [float(row.label) for row in rows]
    X_train = [vectorize(row.features, schema) for row, t in zip(rows, is_train) if t]
    y_train = [y for y, t in zip(labels, is_train) if t]
    y_eval = [y for y, e in zip(labels, is_eval) if e]

    bias, weights = fit_logistic(X_train, y_train)
    model: dict[str, object] = {**schema, "bias": bias, "weights": weights}

    # Content hash of the fitted model: identical rows give an identical version.
    version_source = json.dumps(
        {
            "bias": round(bias, 12),
            "feature_columns": columns,
            "n_rows": len(y_train),
            "weights": [round(w, 12) for w in weights],
        },
        sort_keys=True,
    )
    model["model_version"] = "sha256:" + hashlib.sha256(version_source.encode("utf-8")).hexdigest()[:16]

    # Scoring the holdout through the engine's own scorer (not through raw
    # weights) exercises the exact path RiskEngine uses at serving time.
    eval_probabilities = [
        RiskEngine._predict_proba(model, row.features) for row, e in zip(rows, is_eval) if e
    ]

    # -- Report. Baseline is the training prevalence for everyone: what a model
    # -- with no ranking ability scores, so eval AP only means something above it. --
    train_prevalence = mean(y_train)
    eval_rows = [row for row, e in zip(rows, is_eval) if e]
    eval_sources = [row.metadata.get("source") for row in eval_rows]
    event_counts = [
        float(row.features["event_count_total"])
        if is_numeric(row.features.get("event_count_total"))
        else 0.0
        for row in eval_rows
    ]
    median_event_count = median(event_counts)
    sparse_mask = [count < median_event_count for count in event_counts]

    def sliced(mask: Sequence[bool]) -> dict[str, object]:
        n = sum(mask)
        if n == 0:
            return {"n": 0}
        sub_y = [y for y, m in zip(y_eval, mask) if m]
        sub_p = [p for p, m in zip(eval_probabilities, mask) if m]
        return {
            "n": n,
            "positive_rate": mean(sub_y),
            "average_precision": average_precision(sub_y, sub_p),
            "brier_score": brier(sub_y, sub_p),
        }

    report: dict[str, object] = {
        "model_version": model["model_version"],
        "split_rule": (
            "group-by-shipment, time-ordered: shipments sorted by their earliest "
            f"decision_time, last {EVAL_FRACTION:.0%} of shipments held out entirely"
        ),
        "n_rows_eval": len(y_eval),
        "n_shipments_eval": n_eval_shipments,
        # Prevalence is what makes the other numbers readable: average
        # precision at 8% positives means something very different than at 50%.
        "train_prevalence": train_prevalence,
        "eval_prevalence": mean(y_eval) if y_eval else float("nan"),
        "eval_average_precision": average_precision(y_eval, eval_probabilities),
        "eval_brier_score": brier(y_eval, eval_probabilities),
        "baseline_constant_average_precision": average_precision(
            y_eval, [train_prevalence] * len(y_eval)
        ),
        "slices": {
            "by_source": {
                str(source): sliced([s == source for s in eval_sources])
                for source in sorted(set(eval_sources), key=lambda s: (s is None, s))
            },
            "by_telemetry_richness": {
                "sparse_below_median_event_count": sliced(sparse_mask),
                "rich_at_or_above_median_event_count": sliced([not s for s in sparse_mask]),
            },
        },
    }

    for filename, payload in ((MODEL_FILENAME, model), (METRICS_FILENAME, report)):
        with (artifact_dir / filename).open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, indent=2, default=str)

    return report


class RiskEngine:
    """Online stateful scorer. Public methods may be called concurrently.

    State: per shipment, every distinct (event_id, revision) ever ingested, in
    an `OrderedDict` used as an LRU so `max_shipments` bounds it with no second
    index. Full revision history is required, not incidental — `score()` must
    answer for an `as_of` earlier than an already-ingested record.

    Concurrency: one lock guards the shipment map and the active model, held
    while touching state but never while scoring. See DECISIONS.md.
    """

    # -- Point-in-time kernel: visibility -> features -> probability. Static
    # -- because build_training_rows() and train() replay through these too;
    # -- one implementation offline and online is what makes train/serve
    # -- feature skew unexpressible rather than merely unlikely. --

    @staticmethod
    def _to_utc(value: datetime) -> datetime:
        """All public inputs and outputs are UTC-aware; a naive datetime is ambiguous."""
        if value.tzinfo is None:
            raise ValueError(f"naive datetime not allowed: {value!r}")
        return value.astimezone(timezone.utc)

    @staticmethod
    def _as_datetime(value: object) -> datetime:
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        raise TypeError(f"cannot interpret as datetime: {value!r}")

    @staticmethod
    def _is_numeric(value: object) -> bool:
        return isinstance(value, (int, float)) and not isinstance(value, bool)

    @staticmethod
    def _visible_events(
        events: Sequence[TelemetryEvent], as_of: datetime
    ) -> list[TelemetryEvent]:
        """Latest revision per event_id that had arrived by `as_of`.

        `received_at` is the only clock that gates visibility; `device_time` is
        deliberately not consulted. See DECISIONS.md.
        """
        latest_by_id: dict[str, TelemetryEvent] = {}
        for event in events:
            if event.received_at > as_of:
                continue
            current = latest_by_id.get(event.event_id)
            if current is None or event.revision > current.revision:
                latest_by_id[event.event_id] = event
        return list(latest_by_id.values())

    @staticmethod
    def _features(
        visible_events: Sequence[TelemetryEvent], as_of: datetime
    ) -> dict[str, float | int | str | None]:
        """Features for one decision, shared by offline training and online scoring.

        Keys derive from the `kind`s present, not a fixed schema, so an unseen
        kind degrades gracefully; `train()` aligns rows onto one schema.
        """
        is_numeric = RiskEngine._is_numeric
        features: dict[str, float | int | str | None] = {
            "event_count_total": len(visible_events),
        }

        if not visible_events:
            features["time_since_last_received_s"] = None
            features["max_device_clock_skew_s"] = None
            return features

        features["time_since_last_received_s"] = (
            as_of - max(e.received_at for e in visible_events)
        ).total_seconds()
        features["max_device_clock_skew_s"] = max(
            abs((e.received_at - e.device_time).total_seconds()) for e in visible_events
        )

        by_kind: dict[str, list[TelemetryEvent]] = defaultdict(list)
        for event in visible_events:
            by_kind[event.kind].append(event)

        for kind in sorted(by_kind):
            kind_events = by_kind[kind]
            numeric_events = [e for e in kind_events if is_numeric(e.value)]

            features[f"{kind}_count"] = len(kind_events)

            # "Latest" by device_time reflects true physical recency for trend
            # purposes; ties broken by received_at then event_id for determinism.
            latest = max(kind_events, key=lambda e: (e.device_time, e.received_at, e.event_id))
            features[f"{kind}_last_age_s"] = (as_of - latest.received_at).total_seconds()
            features[f"{kind}_last_value"] = latest.value if is_numeric(latest.value) else None

            if numeric_events:
                # No `_min`: the target is an excursion above a threshold, so the
                # coldest reading in the window carries no signal for it.
                values = [float(e.value) for e in numeric_events]  # type: ignore[arg-type]
                features[f"{kind}_mean"] = sum(values) / len(values)
                features[f"{kind}_max"] = max(values)

            if len(numeric_events) >= 2:
                # Least-squares slope per hour, skipped when every reading shares
                # one device_time and there is no span to regress against.
                ordered = sorted(numeric_events, key=lambda e: e.device_time)
                t0 = ordered[0].device_time
                xs = [(e.device_time - t0).total_seconds() / 3600.0 for e in ordered]
                ys = [float(e.value) for e in ordered]  # type: ignore[arg-type]
                mean_x = sum(xs) / len(xs)
                mean_y = sum(ys) / len(ys)
                denominator = sum((x - mean_x) ** 2 for x in xs)
                if denominator > 0:
                    features[f"{kind}_slope_per_hr"] = (
                        sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / denominator
                    )

        return features

    @staticmethod
    def _predict_proba(model: Mapping[str, object], features: Mapping[str, object]) -> float:
        """Score one feature mapping with a loaded artifact.

        Missing or non-numeric values take the training-time median and unknown
        keys are ignored, so a row from an unseen kind scores instead of
        failing. The sigmoid saturates rather than raising OverflowError.
        """
        medians: Mapping[str, float] = model["impute_median"]  # type: ignore[assignment]
        means: Mapping[str, float] = model["feature_mean"]  # type: ignore[assignment]
        stds: Mapping[str, float] = model["feature_std"]  # type: ignore[assignment]

        z = float(model["bias"])  # type: ignore[arg-type]
        for weight, column in zip(model["weights"], model["feature_columns"]):  # type: ignore[call-overload]
            value = features.get(column)
            if value is None or not RiskEngine._is_numeric(value):
                value = medians[column]
            z += weight * ((float(value) - means[column]) / (stds[column] or 1.0))
        return 0.0 if z < -709.0 else 1.0 / (1.0 + math.exp(-z))

    @staticmethod
    def _load_model(artifact_dir: Path) -> dict[str, object]:
        """Read model.json, rejecting anything that could not serve safely.

        Runs on every load, so a bad artifact never becomes the active model.
        """
        with (Path(artifact_dir) / MODEL_FILENAME).open("r", encoding="utf-8") as handle:
            model: dict[str, object] = json.load(handle)

        missing = set(MODEL_KEYS) - model.keys()
        if missing:
            raise ValueError(f"model artifact missing keys: {sorted(missing)}")
        if len(model["feature_columns"]) != len(model["weights"]):  # type: ignore[arg-type]
            raise ValueError("feature_columns/weights length mismatch")

        # A NaN/Inf weight loads fine but yields a NaN probability that would
        # only fail later, inside Prediction.to_wire(). Reject it here instead.
        coefficients = list(model["weights"]) + [model["bias"]]  # type: ignore[arg-type]
        if not all(isinstance(c, (int, float)) and math.isfinite(c) for c in coefficients):
            raise ValueError("model weights/bias must all be finite numbers")

        return model

    # -- Lifecycle --

    def __init__(self, artifact_dir: Path, max_shipments: int = 10_000):
        self._model: dict[str, object] = self._load_model(artifact_dir)
        self._max_shipments = max_shipments
        self._shipments: OrderedDict[str, dict[tuple[str, int], TelemetryEvent]] = OrderedDict()
        self._lock = threading.RLock()
        self._total_ingested = 0
        self._duplicate_ingests = 0

    def ingest(self, event: TelemetryEvent) -> bool:
        """Apply one delivery; return True iff effective state changed.

        Keyed on (event_id, revision), not shipment_id (rejects note #4): any
        new key is new information, including a very late lower revision.
        """
        key = (event.event_id, event.revision)
        with self._lock:
            records = self._shipments.get(event.shipment_id)
            if records is None:
                records = {}
                self._shipments[event.shipment_id] = records
                if len(self._shipments) > self._max_shipments:
                    self._shipments.popitem(last=False)  # evict least-recently-used
            else:
                self._shipments.move_to_end(event.shipment_id)

            if key in records:
                self._duplicate_ingests += 1
                return False

            records[key] = event
            self._total_ingested += 1
            return True

    def score(self, shipment_id: str, as_of: datetime) -> Prediction:
        """Score a shipment as of an explicit UTC timestamp."""
        as_of = self._to_utc(as_of)
        with self._lock:
            records = self._shipments.get(shipment_id)
            events = list(records.values()) if records is not None else []
            model = self._model

        # The same chain build_training_rows() runs offline.
        visible = self._visible_events(events, as_of)
        features = self._features(visible, as_of)
        probability = self._predict_proba(model, features)

        # The digest fingerprints the exact inputs a score used, so two replays
        # compare field by field. "no_telemetry" (a cold-start shipment) is the
        # one reason that matters operationally, and it drives `degraded`.
        canonical = json.dumps(features, sort_keys=True, separators=(",", ":"), default=str)

        return Prediction(
            shipment_id=shipment_id,
            as_of=as_of,
            probability=probability,
            model_version=str(model["model_version"]),
            feature_digest=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
            degraded=not visible,
            reasons=() if visible else ("no_telemetry",),
        )

    def reload_model(self, artifact_dir: Path) -> bool:
        """Activate a valid model atomically; return False, undisrupted, on failure.

        Failure keeps the old model serving and leaves state untouched (notes #6, #10).
        """
        try:
            candidate = self._load_model(artifact_dir)
        except Exception:
            return False
        with self._lock:
            self._model = candidate
        return True

    def snapshot(self, destination: Path) -> None:
        """Atomically write deterministic, recoverable state.

        Records only — `restore()` reloads the model from `artifact_dir`. They
        sort by (event_id, revision) so the bytes are canonical, and
        `shipment_order` preserves LRU order. Temp-then-rename rejects note #5.
        """
        destination = Path(destination)
        with self._lock:
            payload = {
                "max_shipments": self._max_shipments,
                "shipment_order": list(self._shipments.keys()),
                "shipments": {
                    shipment_id: [
                        {f: getattr(e, f) for f in EVENT_FIELDS}
                        | {"payload": dict(e.payload)}
                        | {t: getattr(e, t).isoformat() for t in EVENT_TIMES}
                        for _, e in sorted(records.items())
                    ]
                    for shipment_id, records in self._shipments.items()
                },
                "counters": {
                    "total_ingested": self._total_ingested,
                    "duplicate_ingests": self._duplicate_ingests,
                },
            }

        raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        destination.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = destination.with_name(destination.name + ".tmp")
        with tmp_path.open("wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, destination)

    @classmethod
    def restore(cls, artifact_dir: Path, snapshot: Path) -> "RiskEngine":
        """Rebuild an engine from `artifact_dir` plus a snapshot written above."""
        payload = json.loads(Path(snapshot).read_text(encoding="utf-8"))
        engine = cls(artifact_dir, max_shipments=payload["max_shipments"])
        with engine._lock:
            for shipment_id in payload["shipment_order"]:
                engine._shipments[shipment_id] = {
                    (raw["event_id"], raw["revision"]): TelemetryEvent(
                        **{f: raw[f] for f in EVENT_FIELDS},
                        **{t: cls._as_datetime(raw[t]) for t in EVENT_TIMES},
                    )
                    for raw in payload["shipments"][shipment_id]
                }
            engine._total_ingested = payload["counters"]["total_ingested"]
            engine._duplicate_ingests = payload["counters"]["duplicate_ingests"]
        return engine

    def stats(self) -> Mapping[str, int | float | str]:
        """Bounded-state and operational counters."""
        with self._lock:
            return {
                "shipment_count": len(self._shipments),
                "max_shipments": self._max_shipments,
                "model_version": str(self._model["model_version"]),
                "total_ingested": self._total_ingested,
                "duplicate_ingests": self._duplicate_ingests,
            }
