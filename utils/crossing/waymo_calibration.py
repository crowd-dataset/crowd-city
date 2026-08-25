"""Tune the bbox speed model for CROWD algorithm selected crossing tracks.

The original CROWD crossing detector decides which YOLO plus BoT SORT tracks
are crossings.  Waymo crossing labels never include or exclude speed training
examples.  A strict 2D track association supplies Waymo planar pedestrian
speed for the algorithm selected crossings.  Waymo training selects the speed
model and Waymo validation remains untouched until the model is frozen.
"""

from __future__ import annotations

import csv
import json
import math
import pickle
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import polars as pl

from utils.crossing.detection import Detection


CALIBRATION_BUILD_ID = "crowd_algorithm_selected_waymo_speed_v32_20260825"
PIPELINE_MODEL_SCHEMA = "crowd_waymo_pipeline_model_v32"
DIAGNOSTIC_FIGURE_BUILD_ID = (
    "waymo_train_test_validation_speed_error_v2_20260825"
)
PERSON_CLASS_ID = 0


@dataclass
class SequenceData:
    source_id: str
    fps: float
    aspect_ratio: float
    prediction_path: Path
    ground_truth_path: Path
    prediction_rows: List[Any]
    prediction_dataframe: pl.DataFrame
    ground_truth_rows: List[Any]
    ground_truth_tracks: Dict[str, List[Any]]
    crossing_tracks: set[str]
    crossing_speeds: Dict[str, float]
    ground_truth_speed_samples: Dict[str, Dict[int, float]]
    associations: Dict[str, Dict[str, Any]]
    ground_truth_to_prediction: Dict[str, str]


def _normalise_id(value: Any) -> str:
    text = str(value if value is not None else "").strip()
    if not text:
        return ""
    try:
        numeric = float(text)
    except (TypeError, ValueError):
        return text
    if math.isfinite(numeric) and abs(numeric - round(numeric)) < 1e-9:
        return str(int(round(numeric)))
    return text


def _truthy(value: Any) -> bool:
    return str(value if value is not None else "").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }


def _safe_float(value: Any) -> Optional[float]:
    try:
        output = float(value)
    except (TypeError, ValueError):
        return None
    return output if math.isfinite(output) else None


def _safe_int(value: Any) -> Optional[int]:
    output = _safe_float(value)
    return int(round(output)) if output is not None else None


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    fieldnames: Optional[Sequence[str]] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(fieldnames or [])
    if not fields:
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _descriptive_speed_statistics(values: Sequence[float]) -> Dict[str, Any]:
    """Return descriptive speed statistics using sample standard deviation."""
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {
            "count": 0,
            "mean_mps": None,
            "sample_sd_mps": None,
            "median_mps": None,
            "q1_mps": None,
            "q3_mps": None,
            "iqr_mps": None,
            "minimum_mps": None,
            "maximum_mps": None,
        }
    q1, median, q3 = np.percentile(array, [25, 50, 75])
    return {
        "count": int(array.size),
        "mean_mps": float(np.mean(array)),
        "sample_sd_mps": (
            float(np.std(array, ddof=1)) if array.size > 1 else None
        ),
        "median_mps": float(median),
        "q1_mps": float(q1),
        "q3_mps": float(q3),
        "iqr_mps": float(q3 - q1),
        "minimum_mps": float(np.min(array)),
        "maximum_mps": float(np.max(array)),
    }


def _resolve_index_path(index_path: Path, value: Any) -> Path:
    """Resolve both current and pre-move Waymo index entries.

    Earlier exports stored repository-relative paths such as
    ``data/speed_calibration/waymo/<source>/<file>``.  The user later moved the
    complete processed dataset under ``<waymo_dataset>/waymo_processed``.  The
    source directory and file name are stable, so prefer that location beside
    the current index before falling back to the historical path.
    """
    text = str(value or "").strip()
    if not text:
        return Path("__missing_waymo_index_path__")
    candidate = Path(text).expanduser()
    if candidate.is_absolute() and candidate.exists():
        return candidate

    candidate_parts = candidate.parts
    split_name = index_path.parent.name
    if split_name in candidate_parts:
        split_position = len(candidate_parts) - 1 - list(
            reversed(candidate_parts)
        ).index(split_name)
        relocated = index_path.parent.joinpath(
            *candidate_parts[split_position + 1 :]
        )
        if relocated.exists() or relocated.parent.exists():
            return relocated

    if len(candidate_parts) >= 2:
        relocated = index_path.parent / candidate_parts[-2] / candidate_parts[-1]
        if relocated.exists() or relocated.parent.exists():
            return relocated

    if candidate.is_absolute():
        try:
            relative = candidate.relative_to("/workspace")
        except ValueError:
            return candidate
        return Path.cwd() / relative
    project_candidate = Path.cwd() / candidate
    if project_candidate.exists():
        return project_candidate
    index_candidate = index_path.parent / candidate
    return index_candidate if index_candidate.exists() else project_candidate


def _empty_prediction_dataframe() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "yolo-id": pl.Int64,
            "x-center": pl.Float64,
            "y-center": pl.Float64,
            "width": pl.Float64,
            "height": pl.Float64,
            "unique-id": pl.Utf8,
            "confidence": pl.Float64,
            "frame-count": pl.Int64,
        }
    )


def _prediction_dataframe(path: Path, minimum_confidence: float) -> pl.DataFrame:
    if not path.is_file() or path.stat().st_size == 0:
        return _empty_prediction_dataframe()
    try:
        dataframe = pl.read_csv(path)
    except (OSError, pl.exceptions.PolarsError):
        return _empty_prediction_dataframe()
    required = set(_empty_prediction_dataframe().columns)
    if not required.issubset(dataframe.columns) or dataframe.height == 0:
        return _empty_prediction_dataframe()
    return dataframe.with_columns(
        pl.col("yolo-id").cast(pl.Int64, strict=False),
        pl.col("x-center").cast(pl.Float64, strict=False),
        pl.col("y-center").cast(pl.Float64, strict=False),
        pl.col("width").cast(pl.Float64, strict=False),
        pl.col("height").cast(pl.Float64, strict=False),
        pl.col("unique-id").cast(pl.Utf8, strict=False),
        pl.col("confidence").cast(pl.Float64, strict=False),
        pl.col("frame-count").cast(pl.Int64, strict=False),
    ).drop_nulls(list(required)).filter(
        pl.col("confidence") >= float(minimum_confidence)
    )


def _load_all_ground_truth(path: Path, harness: Any) -> Tuple[
    List[Any],
    Dict[str, List[Any]],
    set[str],
    Dict[str, float],
    Dict[str, Dict[int, float]],
]:
    rows = _read_csv(path)
    output: List[Any] = []
    grouped: Dict[str, List[Any]] = defaultdict(list)
    crossing_tracks: set[str] = set()
    speed_values: Dict[str, List[float]] = defaultdict(list)
    total_speed_samples: Dict[str, Dict[int, float]] = defaultdict(dict)
    for raw in rows:
        track_id = _normalise_id(raw.get("ground_truth_track_id"))
        frame = _safe_int(raw.get("frame-count"))
        x_value = _safe_float(raw.get("x-center"))
        y_value = _safe_float(raw.get("y-center"))
        width = _safe_float(raw.get("width"))
        height = _safe_float(raw.get("height"))
        if not track_id or None in (frame, x_value, y_value, width, height):
            continue
        if width <= 0.0 or height <= 0.0:
            continue
        bbox = harness.BBoxRow(
            PERSON_CLASS_ID,
            float(x_value),
            float(y_value),
            float(width),
            float(height),
            track_id,
            1.0,
            int(frame),
        )
        output.append(bbox)
        grouped[track_id].append(bbox)
        total_speed = None
        for field in (
            "instantaneous_total_speed_mps",
            "metadata_total_speed_mps",
            "ground_truth_total_speed_mps",
            "instantaneous_speed_mps",
        ):
            total_speed = _safe_float(raw.get(field))
            if total_speed is not None:
                break
        if total_speed is not None and total_speed >= 0.0:
            total_speed_samples[track_id][int(frame)] = float(total_speed)
        if _truthy(raw.get("crossing_track")):
            crossing_tracks.add(track_id)
            speed = _safe_float(
                raw.get("ground_truth_crosswalk_axis_speed_mps")
                or raw.get("ground_truth_speed_mps")
            )
            if speed is not None and speed >= 0.0:
                speed_values[track_id].append(float(speed))
    speeds = {
        track_id: float(np.median(values))
        for track_id, values in speed_values.items()
        if values
    }
    return (
        output,
        dict(grouped),
        crossing_tracks,
        speeds,
        {track_id: dict(values) for track_id, values in total_speed_samples.items()},
    )


def _strict_associations(
    prediction_rows: Sequence[Any],
    ground_truth_rows: Sequence[Any],
    harness: Any,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, str], int]:
    person_predictions = [row for row in prediction_rows if row.class_id == PERSON_CLASS_ID]
    prediction_tracks = harness.group_tracks(person_predictions)
    ground_truth_tracks = harness.group_tracks(ground_truth_rows)
    offset_candidates = []
    for offset in (-1, 0, 1):
        statistics, score = harness.matching_statistics(
            person_predictions,
            ground_truth_rows,
            offset,
        )
        offset_candidates.append((score, -abs(offset), offset, statistics))
    _, _, frame_offset, statistics = max(
        offset_candidates,
        key=lambda item: (item[0], item[1]),
    )
    candidates: List[Dict[str, Any]] = []
    for (prediction_id, ground_truth_id), item in statistics.items():
        match_frames = len(item["frames"])
        mean_iou = float(np.mean(item["ious"]))
        prediction_coverage = match_frames / max(
            1,
            len(prediction_tracks.get(prediction_id, [])),
        )
        ground_truth_coverage = match_frames / max(
            1,
            len(ground_truth_tracks.get(ground_truth_id, [])),
        )
        score = match_frames * mean_iou * math.sqrt(
            max(prediction_coverage * ground_truth_coverage, 1e-9)
        )
        candidates.append(
            {
                "prediction_id": prediction_id,
                "ground_truth_id": ground_truth_id,
                "match_frames": match_frames,
                "mean_iou": mean_iou,
                "prediction_coverage": prediction_coverage,
                "ground_truth_coverage": ground_truth_coverage,
                "score": score,
                "frame_offset": frame_offset,
                "matched_frame_pairs": sorted(
                    (int(prediction_frame), int(ground_truth_frame))
                    for prediction_frame, ground_truth_frame in item["frames"]
                ),
            }
        )
    candidates.sort(key=lambda item: item["score"], reverse=True)
    used_predictions: set[str] = set()
    used_ground_truth: set[str] = set()
    associations: Dict[str, Dict[str, Any]] = {}
    reverse: Dict[str, str] = {}
    for candidate in candidates:
        prediction_id = str(candidate["prediction_id"])
        ground_truth_id = str(candidate["ground_truth_id"])
        if prediction_id in used_predictions or ground_truth_id in used_ground_truth:
            continue
        if harness.calibration_match_rejection_reason(candidate):
            continue
        used_predictions.add(prediction_id)
        used_ground_truth.add(ground_truth_id)
        associations[prediction_id] = candidate
        reverse[ground_truth_id] = prediction_id
    return associations, reverse, frame_offset


def _load_sequences(
    index_csv: str,
    harness: Any,
    minimum_confidence: float = 0.70,
) -> List[SequenceData]:
    index_path = Path(index_csv).expanduser().resolve()
    index_rows = _read_csv(index_path)
    sequences: List[SequenceData] = []
    skipped_missing_identity_or_fps = 0
    skipped_missing_ground_truth = 0
    missing_ground_truth_examples: List[str] = []
    for raw in index_rows:
        source_id = str(raw.get("source_id", "")).strip()
        fps = _safe_float(raw.get("fps"))
        aspect_ratio = _safe_float(raw.get("aspect_ratio")) or harness.DEFAULT_ASPECT_RATIO
        prediction_path = _resolve_index_path(index_path, raw.get("prediction_bbox_csv"))
        ground_truth_path = _resolve_index_path(index_path, raw.get("ground_truth_bbox_csv"))
        sequence_directory = index_path.parent / source_id
        if not prediction_path.is_file():
            local_prediction = sequence_directory / "crowd_yolo_botsort_bbox.csv"
            if local_prediction.is_file():
                prediction_path = local_prediction
        if not ground_truth_path.is_file():
            local_ground_truth = sequence_directory / "waymo_ground_truth_bbox.csv"
            if local_ground_truth.is_file():
                ground_truth_path = local_ground_truth
        if not source_id or fps is None or fps <= 0.0:
            skipped_missing_identity_or_fps += 1
            continue
        if not ground_truth_path.is_file():
            skipped_missing_ground_truth += 1
            if len(missing_ground_truth_examples) < 5:
                missing_ground_truth_examples.append(str(ground_truth_path))
            continue
        dataframe = _prediction_dataframe(prediction_path, minimum_confidence)
        prediction_rows = (
            [
                row
                for row in harness.load_bbox_csv(
                    str(prediction_path),
                    person_only=False,
                )
                if row.confidence >= float(minimum_confidence)
            ]
            if prediction_path.is_file() and prediction_path.stat().st_size > 0
            else []
        )
        gt_rows, gt_tracks, crossing_tracks, speeds, total_speed_samples = _load_all_ground_truth(
            ground_truth_path,
            harness,
        )
        associations, reverse, _ = _strict_associations(
            prediction_rows,
            gt_rows,
            harness,
        )
        sequences.append(
            SequenceData(
                source_id=source_id,
                fps=float(fps),
                aspect_ratio=float(aspect_ratio),
                prediction_path=prediction_path,
                ground_truth_path=ground_truth_path,
                prediction_rows=prediction_rows,
                prediction_dataframe=dataframe,
                ground_truth_rows=gt_rows,
                ground_truth_tracks=gt_tracks,
                crossing_tracks=crossing_tracks,
                crossing_speeds=speeds,
                ground_truth_speed_samples=total_speed_samples,
                associations=associations,
                ground_truth_to_prediction=reverse,
            )
        )
    if not sequences and index_rows:
        print(
            "Waymo index resolution summary: "
            f"rows={len(index_rows)}, "
            f"invalid source or FPS={skipped_missing_identity_or_fps}, "
            f"missing ground truth={skipped_missing_ground_truth}"
        )
        for example in missing_ground_truth_examples:
            print(f"Missing Waymo ground truth example: {example}")
    return sequences


def _predicted_crossings(
    sequence: SequenceData,
    detector: Detection,
    parameters: Mapping[str, Any],
) -> set[str]:
    boundary_left = float(parameters["boundary_left"])
    boundary_right = float(parameters["boundary_right"])
    keyword_parameters = {
        key: value
        for key, value in parameters.items()
        if key not in {"boundary_left", "boundary_right"}
    }
    ids, _ = detector.pedestrian_crossing(
        sequence.prediction_dataframe,
        sequence.source_id,
        pl.DataFrame(),
        boundary_left,
        boundary_right,
        person_id=PERSON_CLASS_ID,
        fps=sequence.fps,
        **keyword_parameters,
    )
    return {_normalise_id(value) for value in ids if _normalise_id(value)}


def _matched_waymo_speed_target(
    sequence: SequenceData,
    ground_truth_id: str,
    association: Mapping[str, Any],
) -> Tuple[Optional[float], int]:
    """Return the robust Waymo planar speed over strictly associated frames."""
    samples = sequence.ground_truth_speed_samples.get(ground_truth_id, {})
    values = [
        float(samples[ground_truth_frame])
        for _, ground_truth_frame in association.get("matched_frame_pairs", [])
        if ground_truth_frame in samples
    ]
    values = [value for value in values if math.isfinite(value) and value >= 0.0]
    if not values:
        return None, 0
    array = np.asarray(values, dtype=float)
    median = float(np.median(array))
    absolute_deviation = np.abs(array - median)
    mad = float(np.median(absolute_deviation))
    if mad > 1e-9:
        robust = array[absolute_deviation <= 4.0 * 1.4826 * mad]
        if robust.size:
            array = robust
    return float(np.median(array)), int(array.size)


def _speed_manifest_rows(
    sequences: Sequence[SequenceData],
    detected_by_source: Mapping[str, set[str]],
    split: str,
    harness: Any,
) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for sequence in sequences:
        prediction_tracks = harness.group_tracks(
            [row for row in sequence.prediction_rows if row.class_id == PERSON_CLASS_ID]
        )
        scene_motion = harness.build_scene_motion_profile(
            sequence.prediction_rows,
            sequence.fps,
        )
        for prediction_id in sorted(detected_by_source.get(sequence.source_id, set())):
            association = sequence.associations.get(prediction_id)
            if association is None:
                continue
            ground_truth_id = str(association["ground_truth_id"])
            speed, speed_sample_count = _matched_waymo_speed_target(
                sequence,
                ground_truth_id,
                association,
            )
            track = prediction_tracks.get(prediction_id, [])
            features = harness.track_features(
                track,
                sequence.fps,
                sequence.source_id,
                sequence.aspect_ratio,
                scene_motion,
            )
            reason = "feature_extraction_failed" if features is None else harness.base_rejection_reason(features)
            include = features is not None and not reason and speed is not None and 0.10 <= speed <= 3.50
            if speed is None and not reason:
                reason = "ground_truth_speed_missing"
            elif speed is not None and not 0.10 <= speed <= 3.50 and not reason:
                reason = "ground_truth_speed_out_of_range"
            output.append(
                {
                    "source_id": sequence.source_id,
                    "bbox_csv": str(sequence.prediction_path.resolve()),
                    "fps": sequence.fps,
                    "aspect_ratio": sequence.aspect_ratio,
                    "prediction_track_id": prediction_id,
                    "ground_truth_track_id": ground_truth_id,
                    "ground_truth_speed_mps": speed if speed is not None else "",
                    "calibration_target": harness.WAYMO_CALIBRATION_TARGET,
                    "selection_rule": "original_CROWD_algorithm_predicted_crossing",
                    "algorithm_predicted_crossing": 1,
                    "waymo_derived_crossing_label": (
                        1 if ground_truth_id in sequence.crossing_tracks else 0
                    ),
                    "ground_truth_speed_sample_count": speed_sample_count,
                    "speed_target_source": (
                        "Waymo_instantaneous_total_speed_on_strictly_matched_frames"
                    ),
                    "split": split,
                    "include": 1 if include else 0,
                    "exclusion_reason": reason,
                    "matched_frames": association["match_frames"],
                    "mean_iou": association["mean_iou"],
                    "prediction_match_coverage": association["prediction_coverage"],
                    "ground_truth_match_coverage": association["ground_truth_coverage"],
                    "frame_offset": association["frame_offset"],
                }
            )
    return output


def _algorithm_detected_crossings(
    sequences: Sequence[SequenceData],
    detector: Detection,
    parameters: Mapping[str, Any],
) -> Dict[str, set[str]]:
    return {
        sequence.source_id: _predicted_crossings(sequence, detector, parameters)
        for sequence in sequences
    }


def _speed_selection_metrics(
    sequences: Sequence[SequenceData],
    detected_by_source: Mapping[str, set[str]],
    manifest_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    selected = sum(
        len(detected_by_source.get(sequence.source_id, set()))
        for sequence in sequences
    )
    matched = sum(
        1
        for sequence in sequences
        for prediction_id in detected_by_source.get(sequence.source_id, set())
        if prediction_id in sequence.associations
    )
    eligible = sum(1 for row in manifest_rows if _truthy(row.get("include")))
    return {
        "sequences": len(sequences),
        "waymo_camera_pedestrian_tracks": sum(
            len(sequence.ground_truth_tracks) for sequence in sequences
        ),
        "algorithm_selected_crossing_tracks": selected,
        "strictly_matched_waymo_tracks": matched,
        "unmatched_algorithm_crossing_tracks": selected - matched,
        "speed_eligible_tracks": eligible,
        "feature_or_speed_rejected_tracks": len(manifest_rows) - eligible,
        "strict_match_rate": matched / selected if selected else 0.0,
        "speed_eligible_rate": eligible / selected if selected else 0.0,
        "selection_authority": "original_CROWD_crossing_algorithm",
        "waymo_crossing_label_used_for_selection": False,
    }


def _frozen_speed_validation(
    manifest_path: Path,
    candidate_model_path: Path,
    output_dir: Path,
    harness: Any,
) -> Dict[str, Any]:
    model = harness.load_model(str(candidate_model_path), require_qualified=False)
    rows = [
        row
        for row in harness.load_manifest_feature_rows(str(manifest_path))
        if row["split"] == "test"
    ]
    evaluated = harness.evaluate_feature_rows(rows, model) if rows else []
    metrics = harness.metric_summary(evaluated)
    qualification = harness.external_test_qualification(rows, metrics) if rows else {
        "passed": False,
        "reasons": ["no_usable_validation_tracks"],
        "test_tracks": 0,
        "test_sources": 0,
        "test_source_ids": [],
        "settings": dict(harness.EXTERNAL_TEST_SETTINGS),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    harness.write_dict_csv(str(output_dir / "evaluation_tracks.csv"), evaluated)
    information_diagnostic = (
        harness.write_speed_information_diagnostics(
            evaluated,
            output_dir,
            "untouched_waymo_validation_audit_only",
        )
        if evaluated
        else None
    )
    report = {
        "release_id": harness.RELEASE_ID,
        "selection_rule": "frozen model; Waymo validation was not used for tuning",
        "candidate_calibration_qualified": bool(model.get("calibration_qualified")),
        "metrics": metrics,
        "speed_information_diagnostic": information_diagnostic,
        "external_test_qualification": qualification,
        "production_model_path": None,
    }
    if bool(model.get("calibration_qualified")) and qualification["passed"]:
        production_model = dict(model)
        production_model["external_test_passed"] = True
        production_model["external_test_reasons"] = []
        production_model["external_test_metrics"] = metrics
        production_model["external_test_source_ids"] = qualification["test_source_ids"]
        production_path = output_dir / "production_model.json"
        harness.write_json(str(production_path), production_model)
        report["production_model_path"] = str(production_path.resolve())
    harness.write_json(str(output_dir / "evaluation_report.json"), report)
    return report


def _write_waymo_figures(
    output_root: Path,
    training_selection: Mapping[str, Any],
    validation_selection: Mapping[str, Any],
) -> List[str]:
    """Write reproducible Waymo figures; PNG and EPS require Kaleido."""
    figure_root = output_root / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    written: List[str] = []
    metric_rows = []
    for split, metrics in (
        ("training", training_selection),
        ("validation", validation_selection),
    ):
        metric_rows.append({"split": split, **metrics})
    _write_csv(figure_root / "waymo_speed_selection_metrics.csv", metric_rows)
    written.append(str((figure_root / "waymo_speed_selection_metrics.csv").resolve()))

    fit_path = output_root / "speed_fit" / "fit_predictions.csv"
    cross_validation_path = (
        output_root / "speed_fit" / "cross_validation_predictions.csv"
    )
    evaluation_path = output_root / "speed_validation" / "evaluation_tracks.csv"

    def read_speed_records(
        source_path: Path,
        estimate_fields: Sequence[str],
    ) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        if not source_path.is_file():
            return records
        for row in _read_csv(source_path):
            ground_truth = _safe_float(row.get("ground_truth_speed_mps"))
            estimated = None
            for field in estimate_fields:
                estimated = _safe_float(row.get(field))
                if estimated is not None:
                    break
            if ground_truth is None or estimated is None:
                continue
            records.append(
                {
                    "source_id": row.get("source_id", ""),
                    "prediction_track_id": row.get("prediction_track_id", ""),
                    "ground_truth_track_id": row.get("ground_truth_track_id", ""),
                    "waymo_reference_speed_mps": ground_truth,
                    "algorithm_estimated_speed_mps": estimated,
                    "signed_error_mps": estimated - ground_truth,
                    "absolute_error_mps": abs(estimated - ground_truth),
                }
            )
        return records

    training_fit_records = read_speed_records(
        fit_path,
        (
            "model_speed_before_reliability_gate_mps",
            "estimated_speed_mps",
        ),
    )
    training_cross_validation_records = read_speed_records(
        cross_validation_path,
        ("predicted_speed_mps",),
    )
    untouched_validation_records = read_speed_records(
        evaluation_path,
        (
            "model_speed_before_reliability_gate_mps",
            "estimated_speed_mps",
        ),
    )
    distribution_roles: Dict[str, Dict[str, Any]] = {
        "training_fit": {
            "label": "Training fit",
            "source_path": str(fit_path.resolve()),
            "records": training_fit_records,
        },
        "training_cross_validation": {
            "label": "Test (source held out training CV)",
            "source_path": str(cross_validation_path.resolve()),
            "records": training_cross_validation_records,
        },
        "untouched_validation_test": {
            "label": "Validation (untouched)",
            "source_path": str(evaluation_path.resolve()),
            "records": untouched_validation_records,
        },
    }
    distribution_statistics: Dict[str, Dict[str, Any]] = {}
    for role, payload in distribution_roles.items():
        records = payload["records"]
        reference_values = [
            row["waymo_reference_speed_mps"] for row in records
        ]
        estimate_values = [
            row["algorithm_estimated_speed_mps"] for row in records
        ]
        error_values = [row["signed_error_mps"] for row in records]
        absolute_error_values = [abs(value) for value in error_values]
        error_metrics = {
            "count": len(error_values),
            "mae_mps": (
                float(np.mean(absolute_error_values))
                if absolute_error_values
                else None
            ),
            "rmse_mps": (
                float(
                    math.sqrt(
                        np.mean([value * value for value in error_values])
                    )
                )
                if error_values
                else None
            ),
            "bias_mps": (
                float(np.mean(error_values)) if error_values else None
            ),
            "median_absolute_error_mps": (
                float(np.median(absolute_error_values))
                if absolute_error_values
                else None
            ),
            "within_0_25_mps": (
                float(
                    np.mean(
                        [value <= 0.25 for value in absolute_error_values]
                    )
                )
                if absolute_error_values
                else None
            ),
            "within_0_50_mps": (
                float(
                    np.mean(
                        [value <= 0.50 for value in absolute_error_values]
                    )
                )
                if absolute_error_values
                else None
            ),
        }
        distribution_statistics[role] = {
            "label": payload["label"],
            "source_path": payload["source_path"],
            "waymo_reference": _descriptive_speed_statistics(reference_values),
            "algorithm_estimate": _descriptive_speed_statistics(estimate_values),
            "signed_error": _descriptive_speed_statistics(error_values),
            "absolute_error": _descriptive_speed_statistics(
                absolute_error_values
            ),
            "error_metrics": error_metrics,
        }

    distribution_summary_path = (
        figure_root / "waymo_speed_distribution_statistics.json"
    )
    diagnostics_pickle_path = figure_root / "waymo_diagnostics.pickle"
    for legacy_pickle_name in (
        "waymo_speed_distributions.pickle",
        "waymo_validation_speed_comparison.pickle",
    ):
        legacy_pickle_path = figure_root / legacy_pickle_name
        if legacy_pickle_path.is_file():
            legacy_pickle_path.unlink()
    role_groups: List[Dict[str, Any]] = []
    train_validation_groups: List[Dict[str, Any]] = []
    if any(payload["records"] for payload in distribution_roles.values()):
        distribution_summary = {
            "schema": "waymo_speed_distribution_statistics_v1",
            "calibration_build_id": CALIBRATION_BUILD_ID,
            "standard_deviation": "sample standard deviation with ddof=1",
            "role_definition": {
                "training_fit": "in sample fit on Waymo training",
                "training_cross_validation": (
                    "source held out cross validation within Waymo training"
                ),
                "untouched_validation_test": (
                    "frozen model evaluated on untouched Waymo validation"
                ),
            },
            "roles": distribution_statistics,
        }
        _write_json(distribution_summary_path, distribution_summary)
        written.append(str(distribution_summary_path.resolve()))

        role_groups = [
            {
                "label": distribution_roles[role]["label"],
                "values": [
                    row["algorithm_estimated_speed_mps"]
                    for row in distribution_roles[role]["records"]
                ],
                "colour": colour,
            }
            for role, colour in (
                ("training_fit", "#009E73"),
                ("training_cross_validation", "#E69F00"),
                ("untouched_validation_test", "#D55E00"),
            )
            if distribution_roles[role]["records"]
        ]
        if training_cross_validation_records:
            train_validation_groups.extend(
                [
                    {
                        "label": "Training Waymo",
                        "values": [
                            row["waymo_reference_speed_mps"]
                            for row in training_cross_validation_records
                        ],
                        "colour": "#0072B2",
                    },
                    {
                        "label": "Training algorithm",
                        "values": [
                            row["algorithm_estimated_speed_mps"]
                            for row in training_cross_validation_records
                        ],
                        "colour": "#56B4E9",
                    },
                ]
            )
        if untouched_validation_records:
            train_validation_groups.extend(
                [
                    {
                        "label": "Validation Waymo",
                        "values": [
                            row["waymo_reference_speed_mps"]
                            for row in untouched_validation_records
                        ],
                        "colour": "#D55E00",
                    },
                    {
                        "label": "Validation algorithm",
                        "values": [
                            row["algorithm_estimated_speed_mps"]
                            for row in untouched_validation_records
                        ],
                        "colour": "#E69F00",
                    },
                ]
            )
    comparison_rows: List[Dict[str, Any]] = []
    scatter_pairs: List[Tuple[float, float]] = []
    speed_metrics: Dict[str, Any] = {}
    errors: List[float] = []
    absolute_errors: List[float] = []
    if evaluation_path.is_file():
        for row in _read_csv(evaluation_path):
            ground_truth = _safe_float(row.get("ground_truth_speed_mps"))
            estimated = _safe_float(
                row.get("model_speed_before_reliability_gate_mps")
                or row.get("estimated_speed_mps")
            )
            if ground_truth is not None and estimated is not None:
                scatter_pairs.append((ground_truth, estimated))
                error = estimated - ground_truth
                comparison_rows.append(
                    {
                        "source_id": row.get("source_id", ""),
                        "prediction_track_id": row.get("prediction_track_id", ""),
                        "ground_truth_track_id": row.get("ground_truth_track_id", ""),
                        "waymo_ground_truth_speed_mps": ground_truth,
                        "estimated_speed_mps": estimated,
                        "signed_error_mps": error,
                        "absolute_error_mps": abs(error),
                        "prediction_status": (
                            row.get("speed_status")
                            or row.get("prediction_status", "")
                        ),
                        "rejection_reason": (
                            row.get("reject_reason")
                            or row.get("rejection_reason", "")
                        ),
                    }
                )
    if scatter_pairs:
        errors = [estimated - truth for truth, estimated in scatter_pairs]
        absolute_errors = [abs(value) for value in errors]
        mae = float(np.mean(absolute_errors))
        rmse = float(math.sqrt(np.mean([value * value for value in errors])))
        bias = float(np.mean(errors))
        median_absolute_error = float(np.median(absolute_errors))
        speed_metrics = {
            "count": len(scatter_pairs),
            "mae_mps": mae,
            "rmse_mps": rmse,
            "bias_mps": bias,
            "median_absolute_error_mps": median_absolute_error,
            "within_0_25_mps": float(
                np.mean([value <= 0.25 for value in absolute_errors])
            ),
            "within_0_50_mps": float(
                np.mean([value <= 0.50 for value in absolute_errors])
            ),
            "descriptive_statistics": {
                "waymo_reference": _descriptive_speed_statistics(
                    [truth for truth, _ in scatter_pairs]
                ),
                "algorithm_estimate": _descriptive_speed_statistics(
                    [estimate for _, estimate in scatter_pairs]
                ),
                "signed_error": _descriptive_speed_statistics(errors),
                "absolute_error": _descriptive_speed_statistics(
                    absolute_errors
                ),
            },
        }
        if distribution_summary_path.is_file():
            try:
                distribution_summary = json.loads(
                    distribution_summary_path.read_text(encoding="utf-8")
                )
                distribution_summary["validation_error_metrics"] = speed_metrics
                _write_json(distribution_summary_path, distribution_summary)
            except (OSError, ValueError):
                pass
        comparison_path = figure_root / "waymo_validation_speed_comparison.csv"
        _write_csv(comparison_path, comparison_rows)
        written.append(str(comparison_path.resolve()))
        metrics_path = figure_root / "waymo_validation_speed_metrics.json"
        _write_json(metrics_path, speed_metrics)
        written.append(str(metrics_path.resolve()))

    if any(payload["records"] for payload in distribution_roles.values()) or comparison_rows:
        with diagnostics_pickle_path.open("wb") as handle:
            pickle.dump(
                {
                    "schema": "waymo_diagnostics_v1",
                    "calibration_build_id": CALIBRATION_BUILD_ID,
                    "standard_deviation": "sample standard deviation with ddof=1",
                    "roles": {
                        role: {
                            **distribution_statistics[role],
                            "records": payload["records"],
                        }
                        for role, payload in distribution_roles.items()
                    },
                    "validation_error_metrics": speed_metrics,
                    "validation_comparison": {
                        "count": len(scatter_pairs),
                        "records": comparison_rows,
                        "waymo_ground_truth_speed_mps": [
                            truth for truth, _ in scatter_pairs
                        ],
                        "algorithm_estimated_speed_mps": [
                            estimate for _, estimate in scatter_pairs
                        ],
                        "signed_error_mps": errors,
                        "absolute_error_mps": absolute_errors,
                    },
                },
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        written.append(str(diagnostics_pickle_path.resolve()))

    try:
        import plotly.graph_objects as go

        def write_plotly_static_formats(
            figure: Any,
            file_stem: str,
            width: int,
            height: int,
        ) -> None:
            for suffix, scale in (("png", 2), ("eps", 1)):
                try:
                    image_path = figure_root / f"{file_stem}.{suffix}"
                    figure.write_image(
                        str(image_path),
                        width=width,
                        height=height,
                        scale=scale,
                    )
                    written.append(str(image_path.resolve()))
                except Exception:
                    pass

        def write_plotly_violin(
            groups: Sequence[Mapping[str, Any]],
            title: str,
            file_stem: str,
        ) -> None:
            usable_groups = [
                group for group in groups if len(group.get("values", [])) > 0
            ]
            if not usable_groups:
                return
            violin_figure = go.Figure()
            tick_values: List[str] = []
            tick_labels: List[str] = []
            for group in usable_groups:
                values = [float(value) for value in group["values"]]
                statistics = _descriptive_speed_statistics(values)
                sd_value = statistics["sample_sd_mps"]
                sd_text = f"{sd_value:.2f}" if sd_value is not None else "NA"
                group_label = str(group["label"])
                tick_values.append(group_label)
                tick_labels.append(
                    f"<b>{group_label}</b><br>"
                    f"n={statistics['count']}<br>"
                    f"mean={statistics['mean_mps']:.2f} m/s<br>"
                    f"SD={sd_text} m/s"
                )
                violin_figure.add_trace(
                    go.Violin(
                        x=[group_label] * len(values),
                        y=values,
                        name=group_label,
                        box_visible=True,
                        meanline_visible=True,
                        points="all",
                        jitter=0.08,
                        pointpos=0.0,
                        marker={"size": 4, "opacity": 0.55},
                        line_color=str(group["colour"]),
                        fillcolor=str(group["colour"]),
                        opacity=0.45,
                        width=0.65,
                        scalegroup="equal_width_speed_distributions",
                        scalemode="width",
                    )
                )
            violin_figure.update_layout(
                title=title,
                xaxis={
                    "title": {
                        "text": "Evaluation group",
                        "standoff": 55,
                    },
                    "tickmode": "array",
                    "tickvals": tick_values,
                    "ticktext": tick_labels,
                    "categoryorder": "array",
                    "categoryarray": tick_values,
                },
                yaxis_title="Pedestrian crossing speed (m/s)",
                violinmode="overlay",
                template="plotly_white",
                showlegend=False,
                margin={"b": 175, "t": 90},
            )
            html_path = figure_root / f"{file_stem}.html"
            violin_figure.write_html(
                str(html_path),
                include_plotlyjs="cdn",
                auto_open=True,
            )
            written.append(str(html_path.resolve()))
            write_plotly_static_formats(
                violin_figure,
                file_stem,
                max(1000, 250 * len(usable_groups) + 350),
                800,
            )

        figure = go.Figure()
        for key, label in (
            ("algorithm_selected_crossing_tracks", "Algorithm selected"),
            ("strictly_matched_waymo_tracks", "Strict Waymo match"),
            ("speed_eligible_tracks", "Speed eligible"),
        ):
            figure.add_bar(
                name=label,
                x=["Waymo training", "Waymo validation"],
                y=[training_selection.get(key, 0), validation_selection.get(key, 0)],
            )
        figure.update_layout(
            barmode="group",
            title="Algorithm selected Waymo speed samples",
            xaxis_title="Dataset split",
            yaxis_title="Pedestrian tracks",
            template="plotly_white",
        )
        html_path = figure_root / "waymo_speed_selection_summary.html"
        figure.write_html(
            str(html_path),
            include_plotlyjs="cdn",
            auto_open=True,
        )
        written.append(str(html_path.resolve()))
        write_plotly_static_formats(
            figure,
            "waymo_speed_selection_summary",
            1200,
            720,
        )

        if evaluation_path.is_file():
            evaluation = _read_csv(evaluation_path)
            truth: List[float] = []
            prediction: List[float] = []
            for row in evaluation:
                ground_truth = _safe_float(row.get("ground_truth_speed_mps"))
                estimated = _safe_float(
                    row.get("model_speed_before_reliability_gate_mps")
                    or row.get("estimated_speed_mps")
                )
                if ground_truth is not None and estimated is not None:
                    truth.append(ground_truth)
                    prediction.append(estimated)
            if truth:
                write_plotly_violin(
                    [
                        {
                            "label": "Waymo reference",
                            "values": truth,
                            "colour": "#0072B2",
                        },
                        {
                            "label": "Algorithm estimate",
                            "values": prediction,
                            "colour": "#D55E00",
                        },
                    ],
                    "Waymo reference and algorithm estimated speeds",
                    "waymo_validation_speed_violin",
                )
                write_plotly_violin(
                    role_groups,
                    "Algorithm speed by evaluation role",
                    "waymo_algorithm_speed_train_cv_test_violin",
                )
                write_plotly_violin(
                    train_validation_groups,
                    "Waymo reference and algorithm speeds by dataset split",
                    "waymo_reference_algorithm_train_validation_violin",
                )

                # Show the same reference versus estimate relationship for all
                # evaluation roles in one common coordinate system.  Here,
                # "Test" means source held out cross validation within Waymo
                # training; "Validation" remains the frozen Waymo validation
                # split and is never used for model selection.
                combined_speed_groups = [
                    {
                        "label": "Train (fit)",
                        "records": training_fit_records,
                        "colour": "#009E73",
                        "symbol": "circle",
                    },
                    {
                        "label": "Test (source held out CV)",
                        "records": training_cross_validation_records,
                        "colour": "#E69F00",
                        "symbol": "diamond",
                    },
                    {
                        "label": "Validation (untouched)",
                        "records": untouched_validation_records,
                        "colour": "#0072B2",
                        "symbol": "x",
                    },
                ]
                combined_scatter = go.Figure()
                combined_values: List[float] = []
                combined_counts: List[str] = []
                for group in combined_speed_groups:
                    records = list(group["records"])
                    if not records:
                        continue
                    reference_values = [
                        float(row["waymo_reference_speed_mps"])
                        for row in records
                    ]
                    estimate_values = [
                        float(row["algorithm_estimated_speed_mps"])
                        for row in records
                    ]
                    combined_values.extend(reference_values)
                    combined_values.extend(estimate_values)
                    combined_counts.append(
                        f"{group['label']}: n={len(records)}"
                    )
                    combined_scatter.add_trace(
                        go.Scatter(
                            x=reference_values,
                            y=estimate_values,
                            mode="markers",
                            name=str(group["label"]),
                            customdata=[
                                [
                                    row.get("source_id", ""),
                                    row.get("prediction_track_id", ""),
                                ]
                                for row in records
                            ],
                            marker={
                                "color": str(group["colour"]),
                                "symbol": str(group["symbol"]),
                                "size": 8,
                                "opacity": 0.62,
                            },
                            hovertemplate=(
                                "Role=%{fullData.name}<br>"
                                "Waymo=%{x:.2f} m/s<br>"
                                "Algorithm=%{y:.2f} m/s<br>"
                                "Source=%{customdata[0]}<br>"
                                "Track=%{customdata[1]}<extra></extra>"
                            ),
                        )
                    )
                if combined_values:
                    combined_low = min(combined_values)
                    combined_high = max(combined_values)
                    combined_padding = max(
                        0.05,
                        0.04 * (combined_high - combined_low),
                    )
                    combined_low -= combined_padding
                    combined_high += combined_padding
                    combined_scatter.add_shape(
                        type="line",
                        x0=combined_low,
                        y0=combined_low,
                        x1=combined_high,
                        y1=combined_high,
                        line={"dash": "dash", "color": "grey"},
                    )
                    combined_scatter.update_layout(
                        title=(
                            "Waymo reference versus algorithm speed by "
                            "evaluation role"
                            f"<br><sup>{' | '.join(combined_counts)}</sup>"
                        ),
                        xaxis={
                            "title": "Waymo crossing speed (m/s)",
                            "range": [combined_low, combined_high],
                            "constrain": "domain",
                        },
                        yaxis={
                            "title": "Estimated crossing speed (m/s)",
                            "range": [combined_low, combined_high],
                            "scaleanchor": "x",
                            "scaleratio": 1,
                        },
                        legend={
                            "orientation": "h",
                            "yanchor": "bottom",
                            "y": 1.02,
                            "xanchor": "left",
                            "x": 0.0,
                        },
                        template="plotly_white",
                        margin={"t": 130},
                    )
                    combined_scatter_path = (
                        figure_root
                        / "waymo_train_test_validation_speed_error.html"
                    )
                    combined_scatter.write_html(
                        str(combined_scatter_path),
                        include_plotlyjs="cdn",
                        auto_open=True,
                    )
                    written.append(str(combined_scatter_path.resolve()))
                    write_plotly_static_formats(
                        combined_scatter,
                        "waymo_train_test_validation_speed_error",
                        1100,
                        850,
                    )

                scatter = go.Figure(
                    data=[
                        go.Scatter(
                            x=truth,
                            y=prediction,
                            mode="markers",
                            marker={"opacity": 0.70},
                            name="Validation tracks",
                        )
                    ]
                )
                low = min(truth + prediction)
                high = max(truth + prediction)
                scatter.add_shape(
                    type="line",
                    x0=low,
                    y0=low,
                    x1=high,
                    y1=high,
                    line={"dash": "dash", "color": "grey"},
                )
                scatter.update_layout(
                    title=(
                        "Frozen bbox speed model on untouched Waymo validation"
                        f"<br><sup>MAE {mae:.2f} m/s, RMSE {rmse:.2f} m/s, "
                        f"bias {bias:.2f} m/s, n={len(truth)}</sup>"
                    ),
                    xaxis_title="Waymo crossing speed (m/s)",
                    yaxis_title="Estimated crossing speed (m/s)",
                    template="plotly_white",
                )
                scatter_path = figure_root / "waymo_validation_speed_error.html"
                scatter.write_html(
                    str(scatter_path),
                    include_plotlyjs="cdn",
                    auto_open=True,
                )
                written.append(str(scatter_path.resolve()))
                write_plotly_static_formats(
                    scatter,
                    "waymo_validation_speed_error",
                    900,
                    800,
                )

                residual_figure = go.Figure()
                residual_figure.add_scatter(
                    x=truth,
                    y=[estimate - actual for actual, estimate in zip(truth, prediction)],
                    mode="markers",
                    marker={"opacity": 0.70, "color": "#D55E00"},
                    name="Validation tracks",
                )
                residual_figure.add_hline(
                    y=0.0,
                    line_dash="dash",
                    line_color="grey",
                )
                residual_figure.update_layout(
                    title="Waymo validation speed residuals",
                    xaxis_title="Waymo ground truth speed (m/s)",
                    yaxis_title="Estimated minus ground truth (m/s)",
                    template="plotly_white",
                )
                residual_html_path = figure_root / "waymo_validation_speed_residuals.html"
                residual_figure.write_html(
                    str(residual_html_path),
                    include_plotlyjs="cdn",
                    auto_open=True,
                )
                written.append(str(residual_html_path.resolve()))
                write_plotly_static_formats(
                    residual_figure,
                    "waymo_validation_speed_residuals",
                    900,
                    800,
                )

                histogram_figure = go.Figure()
                histogram_figure.add_histogram(
                    x=[abs(estimate - actual) for actual, estimate in zip(truth, prediction)],
                    xbins={"start": 0.0, "size": 0.10},
                    marker_color="#0072B2",
                    name="Validation tracks",
                )
                histogram_figure.update_layout(
                    title="Waymo validation absolute speed error",
                    xaxis_title="Absolute error (m/s)",
                    yaxis_title="Pedestrian tracks",
                    template="plotly_white",
                )
                histogram_html_path = (
                    figure_root / "waymo_validation_absolute_error_histogram.html"
                )
                histogram_figure.write_html(
                    str(histogram_html_path),
                    include_plotlyjs="cdn",
                    auto_open=True,
                )
                written.append(str(histogram_html_path.resolve()))
                write_plotly_static_formats(
                    histogram_figure,
                    "waymo_validation_absolute_error_histogram",
                    1000,
                    720,
                )
    except ImportError:
        pass
    return written


def _mirror_waymo_diagnostic_outputs(paths: Sequence[str]) -> List[str]:
    """Mirror diagnostics into the standard repository figure locations."""
    repository_root = Path(__file__).resolve().parents[2]
    final_figure_root = repository_root / "figures"
    analysis_output_root = repository_root / "_output"
    final_figure_root.mkdir(parents=True, exist_ok=True)
    analysis_output_root.mkdir(parents=True, exist_ok=True)
    for legacy_pickle_name in (
        "waymo_speed_distributions.pickle",
        "waymo_validation_speed_comparison.pickle",
    ):
        for legacy_root in (final_figure_root, analysis_output_root):
            legacy_path = legacy_root / legacy_pickle_name
            if legacy_path.is_file():
                legacy_path.unlink()
    figure_suffixes = {".html", ".png", ".eps"}
    data_suffixes = {".csv", ".json", ".pickle", ".pkl"}
    mirrored: List[str] = []
    for path_text in paths:
        source = Path(path_text).expanduser().resolve()
        if not source.is_file():
            continue
        destinations: List[Path] = []
        if source.suffix.lower() in figure_suffixes:
            destinations.extend(
                [
                    final_figure_root / source.name,
                    analysis_output_root / source.name,
                ]
            )
        elif source.suffix.lower() in data_suffixes:
            destinations.append(analysis_output_root / source.name)
        for destination in destinations:
            shutil.copy2(source, destination)
            mirrored.append(str(destination.resolve()))
    return mirrored


def refresh_waymo_diagnostic_figures(
    calibration_directory: str | Path,
    mirror_repository_outputs: bool = True,
) -> List[str]:
    """Regenerate Waymo figures from an existing frozen calibration.

    This function never fits or selects a model.  It only reads the already
    frozen crossing report and validation predictions, making diagnostics
    available when analysis.py reuses completed Waymo processing.
    """
    output_root = Path(calibration_directory).expanduser().resolve()
    report_path = output_root / "crossing_calibration_report.json"
    if not report_path.is_file():
        return []
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    training_metrics = report.get("training_speed_selection", {})
    validation_metrics = report.get("validation_speed_selection", {})
    if not isinstance(training_metrics, Mapping):
        training_metrics = {}
    if not isinstance(validation_metrics, Mapping):
        validation_metrics = {}
    figures = _write_waymo_figures(
        output_root,
        training_metrics,
        validation_metrics,
    )
    if mirror_repository_outputs:
        figures.extend(_mirror_waymo_diagnostic_outputs(figures))
    report["figures"] = figures
    report["diagnostic_figure_build_id"] = DIAGNOSTIC_FIGURE_BUILD_ID
    _write_json(report_path, report)
    print(
        "Waymo diagnostic figure build: "
        f"{DIAGNOSTIC_FIGURE_BUILD_ID}"
    )
    return figures


def calibrate_waymo_pipeline(
    training_index_csv: str,
    validation_index_csv: str,
    output_directory: str,
    harness: Any,
) -> Dict[str, Any]:
    """Fit speed on algorithm selected training crossings and test validation."""
    output_root = Path(output_directory).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    minimum_confidence = 0.70
    try:
        import common

        configured_confidence = _safe_float(common.get_configs("min_confidence"))
        if configured_confidence is not None:
            minimum_confidence = configured_confidence
    except Exception:
        pass
    print("Loading Waymo training sequences and strict BoT SORT associations")
    training = _load_sequences(training_index_csv, harness, minimum_confidence)
    if not training:
        raise SystemExit("Waymo training index contains no usable sequences")

    detector = Detection()
    parameters: Dict[str, Any] = {
        "boundary_left": 0.45,
        "boundary_right": 0.55,
        **detector.crossing_parameter_defaults(),
    }
    print("Selecting training crossings with the fixed original CROWD algorithm")
    training_detected = _algorithm_detected_crossings(
        training,
        detector,
        parameters,
    )
    print("Loading untouched Waymo validation after freezing the speed protocol")
    validation = _load_sequences(validation_index_csv, harness, minimum_confidence)
    if not validation:
        raise SystemExit("Waymo validation index contains no usable sequences")
    training_sources = {sequence.source_id for sequence in training}
    validation_sources = {sequence.source_id for sequence in validation}
    overlap = sorted(training_sources & validation_sources)
    if overlap:
        raise SystemExit(
            "Waymo validation source overlap with training: " + ", ".join(overlap[:10])
        )
    validation_detected = _algorithm_detected_crossings(
        validation,
        detector,
        parameters,
    )

    training_manifest_rows = _speed_manifest_rows(
        training,
        training_detected,
        "train",
        harness,
    )
    validation_manifest_rows = _speed_manifest_rows(
        validation,
        validation_detected,
        "test",
        harness,
    )
    manifest_rows = training_manifest_rows + validation_manifest_rows
    training_selection = _speed_selection_metrics(
        training,
        training_detected,
        training_manifest_rows,
    )
    validation_selection = _speed_selection_metrics(
        validation,
        validation_detected,
        validation_manifest_rows,
    )
    manifest_path = output_root / "waymo_crossing_speed_manifest.csv"
    _write_csv(manifest_path, manifest_rows, harness.MANIFEST_FIELDS)

    candidate_model_path = output_root / "crowd_bbox_speed_candidate_v32.json"
    fit_directory = output_root / "speed_fit"
    speed_fit_error = ""
    speed_validation_report: Dict[str, Any] = {
        "metrics": {},
        "production_model_path": None,
    }
    if any(_truthy(row.get("include")) and row.get("split") == "train" for row in manifest_rows):
        try:
            harness.mode_fit(
                str(manifest_path),
                str(candidate_model_path),
                str(fit_directory),
            )
            speed_validation_report = _frozen_speed_validation(
                manifest_path,
                candidate_model_path,
                output_root / "speed_validation",
                harness,
            )
        except (SystemExit, ValueError, OSError) as error:
            speed_fit_error = str(error)
    else:
        speed_fit_error = "no_eligible_training_speed_tracks"

    figure_paths = _write_waymo_figures(
        output_root,
        training_selection,
        validation_selection,
    )
    figure_paths.extend(_mirror_waymo_diagnostic_outputs(figure_paths))

    report = {
        "calibration_build_id": CALIBRATION_BUILD_ID,
        "diagnostic_figure_build_id": DIAGNOSTIC_FIGURE_BUILD_ID,
        "pipeline_model_schema": PIPELINE_MODEL_SCHEMA,
        "protocol": {
            "crossing_selection": "fixed original CROWD algorithm",
            "waymo_crossing_label_used_for_speed_selection": False,
            "speed_model_selection": "Waymo training only",
            "final_evaluation": "Waymo validation only after the speed model was frozen",
            "production_input": "CROWD YOLO plus BoT SORT bounding box CSV and FPS",
            "camera_pose_used_in_production": False,
            "minimum_yolo_confidence": minimum_confidence,
        },
        "training_sequences": len(training),
        "validation_sequences": len(validation),
        "crossing_parameters": parameters,
        "training_speed_selection": training_selection,
        "validation_speed_selection": validation_selection,
        "speed_manifest_rows": len(manifest_rows),
        "speed_fit_error": speed_fit_error,
        "speed_candidate_model_path": (
            str(candidate_model_path.resolve()) if candidate_model_path.is_file() else None
        ),
        "speed_production_model_path": speed_validation_report.get("production_model_path"),
        "speed_validation_metrics": speed_validation_report.get("metrics", {}),
        "metric_speed_qualified_for_crowd": bool(
            speed_validation_report.get("production_model_path")
        ),
        "figures": figure_paths,
    }
    _write_json(output_root / "crossing_calibration_report.json", report)
    production_model_value = report["speed_production_model_path"]
    portable_production_model = production_model_value
    if production_model_value:
        try:
            portable_production_model = str(
                Path(production_model_value).resolve().relative_to(output_root)
            )
        except ValueError:
            pass
    pipeline_model = {
        "schema": PIPELINE_MODEL_SCHEMA,
        "calibration_build_id": CALIBRATION_BUILD_ID,
        "crossing_parameters": parameters,
        "crossing_parameters_source": "fixed_original_CROWD_algorithm",
        "waymo_crossing_label_used_for_speed_selection": False,
        "training_speed_selection": training_selection,
        "validation_speed_selection": validation_selection,
        "speed_candidate_model_path": report["speed_candidate_model_path"],
        "speed_production_model_path": portable_production_model,
        "metric_speed_qualified_for_crowd": report["metric_speed_qualified_for_crowd"],
        "validation_was_not_used_for_speed_model_selection": True,
    }
    _write_json(output_root / "crowd_waymo_pipeline_model.json", pipeline_model)
    print(f"Crossing calibration report: {output_root / 'crossing_calibration_report.json'}")
    print(f"Frozen pipeline model: {output_root / 'crowd_waymo_pipeline_model.json'}")
    return report
