from __future__ import annotations

import json
import shutil
import tempfile
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

from dispatch_risk.contracts import Prediction, TelemetryEvent, TrainingRow
from dispatch_risk.solution import RiskEngine, build_training_rows, train

BASE = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _event(
    event_id: str,
    revision: int,
    shipment_id: str,
    received_at: datetime,
    value: float = 4.0,
    device_time: datetime | None = None,
) -> TelemetryEvent:
    return TelemetryEvent(
        event_id=event_id,
        revision=revision,
        shipment_id=shipment_id,
        device_time=device_time or received_at,
        received_at=received_at,
        kind="temperature_c",
        value=value,
        source="sensor-test",
        payload={},
    )


def test_prediction_wire_format_is_canonical() -> None:
    prediction = Prediction(
        shipment_id="s-1",
        as_of=datetime(2026, 1, 1, tzinfo=timezone.utc),
        probability=0.25,
        model_version="m-1",
        feature_digest="abc",
        degraded=False,
        reasons=("temperature_high",),
    )
    assert prediction.to_wire() == (
        b'{"as_of":"2026-01-01T00:00:00+00:00","degraded":false,'
        b'"feature_digest":"abc","model_version":"m-1","probability":0.25,'
        b'"reasons":["temperature_high"],"shipment_id":"s-1"}'
    )


def _synthetic_rows() -> list[TrainingRow]:
    """Minimal, independent rows just to give train() something to fit —
    not related to the leakage scenarios below."""
    rows = []
    for i in range(40):
        value = 3.0 if i % 2 == 0 else 7.0
        rows.append(
            TrainingRow(
                shipment_id=f"s-{i:03d}",
                decision_time=BASE + timedelta(hours=i),
                features={"temperature_c_last_value": value, "event_count_total": 1},
                label=1 if value > 5.0 else 0,
                metadata={},
            )
        )
    return rows


def additional_tests() -> None:
    """README customer note #1 says: "Always apply the newest revision of
    an event, even when replaying an old decision." That is unsafe: a
    correction that arrives after a decision was made would silently
    rewrite that decision's inputs. Each inner function below asserts the
    opposite, for both the offline builder and the online engine: only a
    revision that had actually arrived (received_at <= decision_time / as_of)
    may influence a score.
    """
    # Built once and reused (read-only) by every inner function below that
    # needs a RiskEngine, instead of each calling train() for itself.
    artifact_dir = _build_test_artifact()

    def a_correction_arriving_after_decision_time_is_not_used():
        original = _event("e1", 1, "s-1", received_at=BASE, value=4.0)
        # Same event_id, higher revision, but it does not arrive until 8h later.
        correction = _event("e1", 2, "s-1", received_at=BASE + timedelta(hours=8), value=99.0)

        decision_time = BASE + timedelta(hours=1)  # before the correction arrives
        rows = build_training_rows([original, correction], [], [("s-1", decision_time)])

        assert rows[0].features["temperature_c_last_value"] == 4.0

    def a_future_event_delivered_early_in_the_input_list_is_still_excluded():
        # Delivery order is not chronological: this future event is first in
        # the list but must be excluded because its received_at is after
        # the decision time.
        future_event = _event("e2", 1, "s-1", received_at=BASE + timedelta(hours=5), value=50.0)
        past_event = _event("e1", 1, "s-1", received_at=BASE, value=4.0)

        decision_time = BASE + timedelta(hours=1)
        rows = build_training_rows([future_event, past_event], [], [("s-1", decision_time)])

        assert rows[0].features["event_count_total"] == 1
        assert rows[0].features["temperature_c_last_value"] == 4.0

    def the_naive_always_apply_newest_revision_policy_would_have_leaked():
        original = _event("e1", 1, "s-1", received_at=BASE, value=4.0)
        correction = _event("e1", 2, "s-1", received_at=BASE + timedelta(hours=8), value=99.0)
        decision_time = BASE + timedelta(hours=1)

        rows = build_training_rows([original, correction], [], [("s-1", decision_time)])
        correct_value = rows[0].features["temperature_c_last_value"]

        naive_value = max([original, correction], key=lambda e: e.revision).value
        assert correct_value != naive_value

    def a_correction_already_held_in_memory_stays_invisible_before_it_arrives():
        engine = RiskEngine(artifact_dir, max_shipments=10)
        engine.ingest(_event("e1", 1, "s-1", received_at=BASE, value=4.0))
        # Already ingested, but must stay invisible to any as_of before its
        # own received_at.
        engine.ingest(_event("e1", 2, "s-1", received_at=BASE + timedelta(hours=8), value=99.0))

        early = engine.score("s-1", BASE + timedelta(hours=1))
        late = engine.score("s-1", BASE + timedelta(hours=9))

        assert early.feature_digest != late.feature_digest
        assert early.to_wire() != late.to_wire()

    def out_of_order_future_delivery_stays_invisible_before_its_received_at():
        engine = RiskEngine(artifact_dir, max_shipments=10)
        # Ingest the *later* event first — mirrors the spec's "input order
        # is delivery order and is not guaranteed to be chronological."
        engine.ingest(_event("e2", 1, "s-1", received_at=BASE + timedelta(hours=5), value=50.0))
        engine.ingest(_event("e1", 1, "s-1", received_at=BASE, value=4.0))

        prediction = engine.score("s-1", BASE + timedelta(hours=1))
        repeated = engine.score("s-1", BASE + timedelta(hours=1))
        later_prediction = engine.score("s-1", BASE + timedelta(hours=6))

        assert prediction.to_wire() == repeated.to_wire()
        assert prediction.to_wire() != later_prediction.to_wire()

    def incident_window_is_half_open():
        # (as_of, as_of + 6h]: exactly at decision_time is excluded, exactly
        # at the 6h boundary is included.
        decision_time = BASE

        def label_at(offset_hours: float) -> list[dict]:
            return [
                {
                    "incident_id": "inc-1",
                    "shipment_id": "s-1",
                    "incident_at": (decision_time + timedelta(hours=offset_hours)).isoformat(),
                    "label_available_at": (decision_time + timedelta(hours=20)).isoformat(),
                    "severity": 1,
                }
            ]

        assert build_training_rows([], label_at(0), [("s-1", decision_time)])[0].label == 0
        assert build_training_rows([], label_at(0.5), [("s-1", decision_time)])[0].label == 1
        assert build_training_rows([], label_at(6), [("s-1", decision_time)])[0].label == 1
        assert build_training_rows([], label_at(6.5), [("s-1", decision_time)])[0].label == 0

    def duplicate_delivery_of_the_same_revision_is_a_no_op():
        engine = RiskEngine(artifact_dir, max_shipments=10)
        event = _event("e1", 1, "s-1", received_at=BASE, value=4.0)

        assert engine.ingest(event) is True
        before = engine.stats()
        assert engine.ingest(event) is False
        after = engine.stats()

        assert before["total_ingested"] == after["total_ingested"]
        assert after["duplicate_ingests"] == before["duplicate_ingests"] + 1

    def a_lower_revision_arriving_after_a_higher_one_does_not_regress_the_score():
        # Real-world redelivery can bring an old revision back after a newer
        # one is already known. It must not un-apply the correction.
        engine = RiskEngine(artifact_dir, max_shipments=10)
        engine.ingest(_event("e1", 2, "s-1", received_at=BASE, value=99.0))
        as_of = BASE + timedelta(hours=1)
        before = engine.score("s-1", as_of)

        engine.ingest(_event("e1", 1, "s-1", received_at=BASE, value=4.0))
        after = engine.score("s-1", as_of)

        assert before.to_wire() == after.to_wire()

    def revision_resolution_does_not_depend_on_ingest_call_order():
        rev1 = _event("e1", 1, "s-1", received_at=BASE, value=4.0)
        rev2 = _event("e1", 2, "s-1", received_at=BASE + timedelta(hours=2), value=9.0)
        rev3 = _event("e1", 3, "s-1", received_at=BASE + timedelta(hours=4), value=15.0)
        as_of = BASE + timedelta(hours=3)  # rev3 not yet visible, rev2 is

        forward = build_training_rows([rev1, rev2, rev3], [], [("s-1", as_of)])
        scrambled = build_training_rows([rev3, rev1, rev2], [], [("s-1", as_of)])

        assert forward[0].features == scrambled[0].features
        assert forward[0].features["temperature_c_last_value"] == 9.0

    def two_independent_replays_produce_byte_identical_predictions_and_snapshots():
        stream = [
            _event("e1", 1, "s-1", received_at=BASE, value=4.0),
            _event("e2", 1, "s-1", received_at=BASE + timedelta(hours=1), value=4.5),
            _event("e1", 2, "s-1", received_at=BASE + timedelta(hours=6), value=9.0),
        ]

        def replay() -> tuple[list[bytes], bytes]:
            engine = RiskEngine(artifact_dir, max_shipments=10)
            wire = []
            for event in stream:
                engine.ingest(event)
                wire.append(engine.score("s-1", event.received_at).to_wire())
            snap_path = Path(tempfile.mkdtemp()) / "snap.json"
            engine.snapshot(snap_path)
            return wire, snap_path.read_bytes()

        predictions_a, snapshot_a = replay()
        predictions_b, snapshot_b = replay()

        assert predictions_a == predictions_b
        assert snapshot_a == snapshot_b

    def max_shipments_bounds_shipment_count_without_growing():
        limit = 5
        engine = RiskEngine(artifact_dir, max_shipments=limit)
        for i in range(50):
            engine.ingest(_event(f"e-{i}", 1, f"s-{i}", received_at=BASE, value=4.0))
            assert engine.stats()["shipment_count"] <= limit

    def evicted_shipment_reappearing_starts_from_fresh_state():
        engine = RiskEngine(artifact_dir, max_shipments=1)
        original = _event("a1", 1, "s-a", received_at=BASE, value=4.0)
        assert engine.ingest(original) is True

        # This evicts s-a (max_shipments=1).
        engine.ingest(_event("b1", 1, "s-b", received_at=BASE, value=4.0))
        assert engine.stats()["shipment_count"] == 1

        # s-a's history is gone, so re-ingesting the exact same record it had
        # before is "new" again — this is the documented cost of bounding
        # memory by shipment count rather than keeping every shipment
        # forever (which the spec's constraints explicitly rule out).
        assert engine.ingest(original) is True
        prediction = engine.score("s-a", BASE)
        assert 0.0 <= prediction.probability <= 1.0

    def reload_with_a_valid_model_activates_the_new_version():
        engine = RiskEngine(artifact_dir, max_shipments=10)
        original_version = engine.stats()["model_version"]

        other_dir = Path(tempfile.mkdtemp()) / "retrained"
        rows = _synthetic_rows()
        # Perturb the data so the retrained artifact gets a different hash.
        rows = [
            TrainingRow(r.shipment_id, r.decision_time, {**r.features, "extra_feature": 1.0}, r.label, r.metadata)
            for r in rows
        ]
        train(rows, other_dir)

        assert engine.reload_model(other_dir) is True
        assert engine.stats()["model_version"] != original_version

    def reload_with_corrupt_json_is_rejected_and_old_model_keeps_serving():
        engine = RiskEngine(artifact_dir, max_shipments=10)
        engine.ingest(_event("e1", 1, "s-1", received_at=BASE, value=4.0))
        before_prediction = engine.score("s-1", BASE).to_wire()
        before_version = engine.stats()["model_version"]

        bad_dir = Path(tempfile.mkdtemp()) / "bad"
        bad_dir.mkdir()
        (bad_dir / "model.json").write_text("{not valid json")

        assert engine.reload_model(bad_dir) is False
        assert engine.stats()["model_version"] == before_version
        assert engine.score("s-1", BASE).to_wire() == before_prediction

    def reload_with_non_finite_weights_is_rejected():
        # Schema-valid but numerically corrupt: a NaN weight would load fine
        # and only blow up later in Prediction.to_wire(), so it must be
        # rejected before it can become the active model.
        engine = RiskEngine(artifact_dir, max_shipments=10)
        engine.ingest(_event("e1", 1, "s-1", received_at=BASE, value=4.0))
        before_prediction = engine.score("s-1", BASE).to_wire()

        model = json.loads((artifact_dir / "model.json").read_text())
        model["weights"][0] = float("nan")
        nan_dir = Path(tempfile.mkdtemp()) / "nan_model"
        nan_dir.mkdir()
        (nan_dir / "model.json").write_text(json.dumps(model))

        assert engine.reload_model(nan_dir) is False
        assert engine.score("s-1", BASE).to_wire() == before_prediction

    def reload_with_missing_required_keys_is_rejected():
        engine = RiskEngine(artifact_dir, max_shipments=10)

        incomplete_dir = Path(tempfile.mkdtemp()) / "incomplete"
        incomplete_dir.mkdir()
        (incomplete_dir / "model.json").write_text('{"feature_columns": [], "weights": []}')

        assert engine.reload_model(incomplete_dir) is False

    # -- One check per numbered customer note in README.md's "Customer-
    # provided implementation notes" section. Each is named for the note it
    # covers and would fail if that note's unsafe policy had been
    # implemented literally. --

    def note_1_split_is_grouped_by_shipment_not_random_by_row():
        # 10 shipments, 3 decision times each, ordered in time so there is
        # an unambiguous "last 20%". If rows were split randomly (note #1)
        # instead of by whole shipment, the eval row count would not
        # reliably be an exact multiple of the per-shipment row count.
        rows = []
        for i in range(10):
            for j in range(3):
                hour = i * 10 + j
                value = 8.0 if i >= 8 else 3.0  # shipments 8,9 are the positive class
                rows.append(
                    TrainingRow(
                        shipment_id=f"s-{i:02d}",
                        decision_time=BASE + timedelta(hours=hour),
                        features={"temperature_c_last_value": value, "event_count_total": 1},
                        label=1 if value > 5.0 else 0,
                        metadata={},
                    )
                )
        report = train(rows, Path(tempfile.mkdtemp()))

        assert "random" not in report["split_rule"].lower()
        assert report["n_rows_eval"] % 3 == 0
        assert report["n_shipments_eval"] == report["n_rows_eval"] // 3

    def note_3_a_stale_device_clock_does_not_grant_early_visibility():
        # device_time looks 30 days old, but received_at (the platform's
        # actual knowledge time) is after decision_time — must be excluded.
        # Sorting/gating by device_time instead of received_at (note #3)
        # would have wrongly treated this as already-known.
        misleading_event = _event(
            "ticket",
            1,
            "s-1",
            received_at=BASE + timedelta(hours=5),
            value=999.0,
            device_time=BASE - timedelta(days=30),
        )
        decision_time = BASE + timedelta(hours=1)
        rows = build_training_rows([misleading_event], [], [("s-1", decision_time)])
        assert rows[0].features["event_count_total"] == 0

    def note_4_deduplication_is_by_event_id_not_shipment_id():
        e1 = _event("e1", 1, "s-1", received_at=BASE, value=4.0)
        e2 = _event("e2", 1, "s-1", received_at=BASE, value=8.0)  # distinct event, same shipment
        decision_time = BASE + timedelta(minutes=1)

        rows = build_training_rows([e1, e2], [], [("s-1", decision_time)])
        assert rows[0].features["event_count_total"] == 2

        engine = RiskEngine(artifact_dir, max_shipments=10)
        assert engine.ingest(e1) is True
        assert engine.ingest(e2) is True

    def note_5_snapshot_is_atomic_under_concurrent_ingest_without_relying_on_kafka():
        # Bounded ingest workload, not an unbounded loop: snapshot() cost
        # scales with total records held, so racing it against an
        # open-ended ingest loop makes each side feed the other's growth.
        engine = RiskEngine(artifact_dir, max_shipments=50)
        snap_path = Path(tempfile.mkdtemp()) / "snap.json"
        errors: list[Exception] = []

        def ingest_loop():
            for i in range(3000):
                engine.ingest(
                    _event(f"e-{i}", 1, f"s-{i % 20}", received_at=BASE + timedelta(seconds=i), value=float(i % 5))
                )

        ingest_thread = threading.Thread(target=ingest_loop)
        ingest_thread.start()
        try:
            while ingest_thread.is_alive():
                engine.snapshot(snap_path)
                json.loads(snap_path.read_bytes())  # must always be complete, parseable JSON
        except Exception as exc:  # pragma: no cover - failure path
            errors.append(exc)
        finally:
            ingest_thread.join()

        engine.snapshot(snap_path)
        json.loads(snap_path.read_bytes())
        assert not errors

    def note_6_failed_reload_does_not_fall_back_to_a_hardcoded_probability():
        engine = RiskEngine(artifact_dir, max_shipments=10)
        engine.ingest(_event("e1", 1, "s-1", received_at=BASE, value=4.0))
        before = engine.score("s-1", BASE)

        missing_dir = Path(tempfile.mkdtemp()) / "does-not-exist"
        assert engine.reload_model(missing_dir) is False

        after = engine.score("s-1", BASE)
        assert after.to_wire() == before.to_wire()

    def note_7_labels_never_leak_into_features():
        events = [_event("e1", 1, "s-1", received_at=BASE, value=4.0)]
        decision_times = [("s-1", BASE + timedelta(hours=1))]

        rows_without_labels = build_training_rows(events, [], decision_times)
        rows_with_incident = build_training_rows(
            events,
            [
                {
                    "incident_id": "inc-1",
                    "shipment_id": "s-1",
                    "incident_at": (BASE + timedelta(hours=2)).isoformat(),
                    "label_available_at": (BASE + timedelta(hours=20)).isoformat(),
                    "severity": 3,
                }
            ],
            decision_times,
        )

        assert rows_without_labels[0].features == rows_with_incident[0].features
        assert rows_without_labels[0].label != rows_with_incident[0].label

    def note_8_report_does_not_reduce_to_a_single_auc_threshold():
        report = train(_synthetic_rows(), Path(tempfile.mkdtemp()))
        assert "eval_brier_score" in report  # probability-quality/calibration
        assert "baseline_constant_average_precision" in report  # baseline comparison
        slice_count = sum(len(group) for group in report["slices"].values())
        assert slice_count >= 2

    def note_10_reload_does_not_clear_ingested_shipment_state():
        engine = RiskEngine(artifact_dir, max_shipments=10)
        engine.ingest(_event("e1", 1, "s-1", received_at=BASE, value=4.0))
        before_stats = engine.stats()

        assert engine.reload_model(artifact_dir) is True

        after_stats = engine.stats()
        assert after_stats["shipment_count"] == before_stats["shipment_count"]
        assert after_stats["total_ingested"] == before_stats["total_ingested"]

    # Notes #2 and #9 are already covered above by
    # a_correction_already_held_in_memory_stays_invisible_before_it_arrives
    # and max_shipments_bounds_shipment_count_without_growing, respectively.

    # -- README constraints: "Tests may call the engine from several
    # threads" and "The evaluator may set max_shipments=32 and replay more
    # than 10,000 events." --

    def several_threads_can_ingest_and_score_concurrently_on_distinct_shipments():
        engine = RiskEngine(artifact_dir, max_shipments=20)
        errors: list[Exception] = []

        def worker(shipment_id: str, offset: int):
            try:
                for i in range(200):
                    engine.ingest(
                        _event(
                            f"{shipment_id}-{i}",
                            1,
                            shipment_id,
                            received_at=BASE + timedelta(seconds=offset + i),
                            value=float(i % 7),
                        )
                    )
                    engine.score(shipment_id, BASE + timedelta(seconds=offset + i)).to_wire()
            except Exception as exc:  # pragma: no cover - failure path
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(f"s-conc-{t}", t * 1000)) for t in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors

    def concurrent_ingests_on_the_same_shipment_do_not_lose_updates():
        # The harder case: multiple threads racing on one shipment's shared
        # records dict. Distinct event_ids per thread rule out dedup, so a
        # correct lock must land exactly n_threads * n_per_thread ingests.
        engine = RiskEngine(artifact_dir, max_shipments=10)
        n_threads, n_per_thread = 8, 500

        def worker(thread_id: int):
            for i in range(n_per_thread):
                engine.ingest(_event(f"t{thread_id}-{i}", 1, "s-shared", received_at=BASE, value=1.0))

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert engine.stats()["total_ingested"] == n_threads * n_per_thread

    def evaluator_scale_max_shipments_32_with_over_10000_events():
        engine = RiskEngine(artifact_dir, max_shipments=32)
        total_events = 10_500
        for i in range(total_events):
            shipment_id = f"s-{i % 100}"  # 100 distinct shipments cycling forces eviction churn
            engine.ingest(_event(f"e-{i}", 1, shipment_id, received_at=BASE + timedelta(seconds=i), value=float(i % 10)))
            assert engine.stats()["shipment_count"] <= 32

        stats = engine.stats()
        assert stats["total_ingested"] == total_events
        assert stats["shipment_count"] <= 32

        prediction = engine.score(f"s-{(total_events - 1) % 100}", BASE + timedelta(seconds=total_events))
        assert 0.0 <= prediction.probability <= 1.0

    checks = [
        a_correction_arriving_after_decision_time_is_not_used,
        a_future_event_delivered_early_in_the_input_list_is_still_excluded,
        the_naive_always_apply_newest_revision_policy_would_have_leaked,
        a_correction_already_held_in_memory_stays_invisible_before_it_arrives,
        out_of_order_future_delivery_stays_invisible_before_its_received_at,
        incident_window_is_half_open,
        duplicate_delivery_of_the_same_revision_is_a_no_op,
        a_lower_revision_arriving_after_a_higher_one_does_not_regress_the_score,
        revision_resolution_does_not_depend_on_ingest_call_order,
        two_independent_replays_produce_byte_identical_predictions_and_snapshots,
        max_shipments_bounds_shipment_count_without_growing,
        evicted_shipment_reappearing_starts_from_fresh_state,
        reload_with_a_valid_model_activates_the_new_version,
        reload_with_corrupt_json_is_rejected_and_old_model_keeps_serving,
        reload_with_missing_required_keys_is_rejected,
        reload_with_non_finite_weights_is_rejected,
        note_1_split_is_grouped_by_shipment_not_random_by_row,
        note_3_a_stale_device_clock_does_not_grant_early_visibility,
        note_4_deduplication_is_by_event_id_not_shipment_id,
        note_5_snapshot_is_atomic_under_concurrent_ingest_without_relying_on_kafka,
        note_6_failed_reload_does_not_fall_back_to_a_hardcoded_probability,
        note_7_labels_never_leak_into_features,
        note_8_report_does_not_reduce_to_a_single_auc_threshold,
        note_10_reload_does_not_clear_ingested_shipment_state,
        several_threads_can_ingest_and_score_concurrently_on_distinct_shipments,
        concurrent_ingests_on_the_same_shipment_do_not_lose_updates,
        evaluator_scale_max_shipments_32_with_over_10000_events,
    ]

    for check in checks:
        try:
            check()
        except Exception:
            print(f"[FAIL] {check.__name__}")
            raise
        else:
            print(f"[PASS] {check.__name__}")


def _build_test_artifact() -> Path:
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "artifact"
        train(_synthetic_rows(), path)
        # train() only needs artifact_dir to exist for the caller's lifetime;
        # RiskEngine reads model.json eagerly in __init__, so copy it out
        # before the temporary directory is cleaned up.
        permanent = Path(tempfile.mkdtemp()) / "artifact"
        shutil.copytree(path, permanent)
    return permanent


def test_additional_invariants() -> None:
    """pytest entry point — see `additional_tests` for the actual checks."""
    additional_tests()

