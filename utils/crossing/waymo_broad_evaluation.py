"""Evaluate the frozen Waymo speed model on every laterally moving pedestrian.

The calibration trains and validates the speed model only on the tracks the
CROWD crossing detector selects, which is about 3% of the camera-visible
Waymo pedestrians (roughly 170 training and 27 validation tracks). That keeps
training faithful to deployment, but it leaves far too few tracks to tell
whether a change to the reliability gates or features really helps.

This module measures the speed step on a much larger set without touching the
frozen model. Every YOLO plus BoT-SORT pedestrian track that is strictly
associated with a Waymo ground-truth track, and that moves across at least
``min_crossing_x_range`` of the image (the crossing detector's own lateral
threshold), is predicted exactly as the CROWD analysis predicts it: the same
box loading, camera-motion profile, per-video context features and frozen
model. Selection uses only the predicted track's geometry, never the Waymo
crossing label, so it can be applied to CROWD in the same way.

Two roles keep the evaluation honest:

* ``development``: training recordings that contributed no track to the speed
  fit. Decisions such as new gates are made here.
* ``audit``: the untouched validation recordings, used only to confirm a
  decision already made on the development role.

Usage:
    python -m utils.crossing.waymo_broad_evaluation
"""

from __future__ import annotations

import copy
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

import common
import utils.crossing.metrics as crossing_metrics
from utils.crossing.detection import Detection
from utils.crossing.waymo_calibration import (
    _load_sequences,
    _matched_waymo_speed_target,
    _predicted_crossings,
    _write_csv,
    _write_json,
)

CALIBRATION_FOLDER = "calibration_v32"
OUTPUT_FOLDER = "broad_speed_evaluation"
# Same plausible range the calibration applies to its ground-truth targets.
GROUND_TRUTH_SPEED_RANGE_MPS = (0.10, 3.50)


def _ungated_prediction(features: Any, model: Mapping[str, Any]) -> Dict[str, Any]:
    """Return the model's prediction with every reliability gate switched off.

    Only the basic validity checks stay on, so a track whose features cannot
    be computed is still rejected. This is what lets a candidate gate be
    evaluated on tracks that the production gates currently reject.
    """
    open_model = copy.deepcopy(dict(model))
    open_model.pop("feature_lower_bound", None)
    open_model.pop("feature_upper_bound", None)
    open_model.pop("maximum_absolute_uncertainty_mps", None)
    open_model["gates"] = {
        "minimum_predicted_speed_mps": -math.inf,
        "maximum_predicted_speed_mps": math.inf,
    }
    original = crossing_metrics.reliability_rejection_reason
    crossing_metrics.reliability_rejection_reason = lambda _features: ""
    try:
        return crossing_metrics._predict_metric_speed(features, open_model)
    finally:
        crossing_metrics.reliability_rejection_reason = original


def _metrics(rows: Sequence[Mapping[str, Any]], field: str) -> Dict[str, Any]:
    pairs = [
        (float(row["ground_truth_speed_mps"]), float(row[field]))
        for row in rows
        if row.get(field) not in (None, "")
    ]
    if not pairs:
        return {"count": 0}
    truth = np.asarray([pair[0] for pair in pairs])
    estimate = np.asarray([pair[1] for pair in pairs])
    error = estimate - truth
    absolute = np.abs(error)
    return {
        "count": int(truth.size),
        "sources": len({row["source_id"] for row in rows if row.get(field) not in (None, "")}),
        "mae_mps": float(absolute.mean()),
        "median_absolute_error_mps": float(np.median(absolute)),
        "rmse_mps": float(np.sqrt(np.mean(error ** 2))),
        "bias_mps": float(error.mean()),
        "pearson_correlation": float(np.corrcoef(truth, estimate)[0, 1]) if truth.size > 2 else None,
        "prediction_to_reference_sd_ratio": (
            float(estimate.std(ddof=1) / truth.std(ddof=1)) if truth.size > 2 and truth.std(ddof=1) > 0 else None
        ),
        "within_0_25_mps": float(np.mean(absolute <= 0.25)),
        "within_0_50_mps": float(np.mean(absolute <= 0.50)),
        "reference_mean_mps": float(truth.mean()),
    }


def _evaluate_split(
    index_csv: Path,
    role: str,
    excluded_sources: set[str],
    model: Mapping[str, Any],
    crossing_parameters: Mapping[str, Any],
    minimum_confidence: float,
    harness: Any,
) -> List[Dict[str, Any]]:
    minimum_x_range = float(crossing_parameters["min_crossing_x_range"])
    detector = Detection()
    rows: List[Dict[str, Any]] = []
    sequences = _load_sequences(str(index_csv), harness, minimum_confidence)
    print(f"{role}: {len(sequences)} recordings loaded from {index_csv}")
    for number, sequence in enumerate(sequences, start=1):
        if sequence.source_id in excluded_sources:
            continue
        # Exactly the production path of Metrics.calculate_speed_of_crossing.
        boxes = [
            row
            for row in crossing_metrics.load_bbox_csv(sequence.prediction_path)
            if row.confidence >= minimum_confidence
        ]
        person_tracks = crossing_metrics.group_tracks(
            row for row in boxes if row.class_id == crossing_metrics.PERSON_CLASS_ID
        )
        scene_profile = crossing_metrics.build_scene_motion_profile(boxes, sequence.fps)
        features_by_track = crossing_metrics.contextual_track_features(
            person_tracks,
            sequence.fps,
            sequence.source_id,
            sequence.aspect_ratio,
            scene_profile,
        )
        crossings = _predicted_crossings(sequence, detector, crossing_parameters)

        for prediction_id, association in sorted(sequence.associations.items()):
            track_id = crossing_metrics.normalise_id(prediction_id)
            ground_truth_id = str(association["ground_truth_id"])
            speed, speed_samples = _matched_waymo_speed_target(sequence, ground_truth_id, association)
            features = features_by_track.get(track_id)
            lateral = features is not None and features.horizontal_range >= minimum_x_range
            in_range = (
                speed is not None
                and GROUND_TRUTH_SPEED_RANGE_MPS[0] <= speed <= GROUND_TRUTH_SPEED_RANGE_MPS[1]
            )
            row: Dict[str, Any] = {
                "role": role,
                "source_id": sequence.source_id,
                "prediction_track_id": track_id,
                "ground_truth_track_id": ground_truth_id,
                "ground_truth_speed_mps": speed if speed is not None else "",
                "ground_truth_speed_sample_count": speed_samples,
                "matched_frames": association["match_frames"],
                "mean_iou": association["mean_iou"],
                "algorithm_selected_crossing": int(track_id in crossings),
                "waymo_crossing_label": int(ground_truth_id in sequence.crossing_tracks),
                "lateral_motion": int(lateral),
                "eligible": int(bool(lateral and in_range)),
                "production_status": "",
                "production_reject_reason": "",
                "production_speed_mps": "",
                "ungated_speed_mps": "",
                "ungated_uncertainty_mps": "",
            }
            if features is None:
                row["production_status"] = "no_features"
            else:
                production = crossing_metrics._predict_metric_speed(features, model)
                ungated = _ungated_prediction(features, model)
                row["production_status"] = production.get("speed_status", "")
                row["production_reject_reason"] = production.get("reject_reason") or ""
                if production.get("speed_status") == "valid":
                    row["production_speed_mps"] = float(production["estimated_speed_mps"])
                if ungated.get("speed_status") == "valid":
                    row["ungated_speed_mps"] = float(ungated["estimated_speed_mps"])
                    row["ungated_uncertainty_mps"] = ungated.get("speed_uncertainty_mps", "")
                row.update({f"feature_{key}": value for key, value in asdict(features).items()})
            rows.append(row)
        if number % 100 == 0:
            print(f"{role}: {number}/{len(sequences)} recordings, {len(rows)} associated tracks")
    return rows


def build_broad_speed_evaluation(processed_root: Optional[Path] = None) -> Dict[str, Any]:
    """Write per-track results and a summary for both evaluation roles."""
    import speed_estimation_harness as harness

    if processed_root is None:
        processed_root = Path(str(common.get_configs("waymo_dataset_path"))).expanduser() / "waymo_processed"
    calibration_root = processed_root / CALIBRATION_FOLDER
    output_root = calibration_root / OUTPUT_FOLDER
    output_root.mkdir(parents=True, exist_ok=True)

    crossing_metrics.load_tuned_pipeline_model(calibration_root / "crowd_waymo_pipeline_model.json")
    model = crossing_metrics._SPEED_MODEL
    if not model:
        raise RuntimeError(f"No qualified frozen speed model in {calibration_root}")
    crossing_parameters = crossing_metrics.tuned_crossing_parameters()
    fit_report = json.loads((calibration_root / "speed_fit" / "fit_report.json").read_text(encoding="utf-8"))
    development_sources = set(fit_report["cross_validation"]["development_source_ids"])
    minimum_confidence = float(common.get_configs("min_confidence"))

    rows = _evaluate_split(
        processed_root / "training" / "waymo_sequence_index.csv",
        "development",
        development_sources,
        model,
        crossing_parameters,
        minimum_confidence,
        harness,
    ) + _evaluate_split(
        processed_root / "validation" / "waymo_sequence_index.csv",
        "audit",
        set(),
        model,
        crossing_parameters,
        minimum_confidence,
        harness,
    )

    tracks_path = output_root / "broad_speed_evaluation_tracks.csv"
    _write_csv(tracks_path, rows)

    summary: Dict[str, Any] = {
        "schema": "crowd_waymo_broad_speed_evaluation_v1",
        "selection": (
            "strictly associated YOLO plus BoT-SORT pedestrian tracks whose horizontal range is at least "
            f"min_crossing_x_range={crossing_parameters['min_crossing_x_range']} and whose Waymo speed lies in "
            f"{GROUND_TRUTH_SPEED_RANGE_MPS[0]}-{GROUND_TRUTH_SPEED_RANGE_MPS[1]} m/s; the Waymo crossing label "
            "is never used for selection"
        ),
        "excluded_development_sources": len(development_sources),
        "minimum_confidence": minimum_confidence,
        "tracks_csv": str(tracks_path),
        "roles": {},
    }
    for role in ("development", "audit"):
        role_rows = [row for row in rows if row["role"] == role]
        eligible = [row for row in role_rows if row["eligible"]]
        selected = [row for row in eligible if row["algorithm_selected_crossing"]]
        summary["roles"][role] = {
            "associated_tracks": len(role_rows),
            "eligible_lateral_tracks": len(eligible),
            "production_valid_tracks": sum(1 for row in eligible if row["production_speed_mps"] != ""),
            "eligible_ungated": _metrics(eligible, "ungated_speed_mps"),
            "eligible_production_gates": _metrics(eligible, "production_speed_mps"),
            "algorithm_selected_production_gates": _metrics(selected, "production_speed_mps"),
        }
    _write_json(output_root / "broad_speed_evaluation_summary.json", summary)
    return summary


if __name__ == "__main__":
    result = build_broad_speed_evaluation()
    print(json.dumps(result["roles"], indent=2))
