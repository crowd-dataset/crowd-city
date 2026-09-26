"""Test road-surface segmentation against Waymo ground truth.

The CROWD segmentation pass (SegFormer road/footpath under each pedestrian's
feet, then the on-road interval) has only ever run on CROWD footage, where
there is no ground truth. This runs exactly that code on the exported Waymo
camera videos and scores two uses of it against Waymo:

* Crossing detection. Each YOLO pedestrian track is judged by three rules and
  compared with the Waymo crosswalk-crossing label:
  ``detector`` (the CROWD detector's middle-strip rule, as used today),
  ``road_lateral`` (the feet are on the road and the track moves across at
  least min_crossing_x_range of the image while there), ``road_edge``
  (``road_lateral`` plus footpath seen before road entry or after road exit,
  i.e. the pedestrian was seen stepping on or off) and ``road_full``
  (footpath both before and after, i.e. a complete kerb-to-kerb crossing).
* Crossing speed. The on-road speed from the segmentation path
  (``road_restricted_speed``) and the whole-track bounding-box speed are both
  compared with Waymo's speed, over the on-road frames and over the whole
  matched track respectively.

Results go to ``<waymo_processed>/segmentation_evaluation``. The speed model
is the one in ``calibration_v32``; if that model did not qualify, it is still
used here, for evaluation only, and the summary says so.

Usage:
    python -m utils.crossing.waymo_segmentation_evaluation [split ...]
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import polars as pl

import common
import utils.crossing.metrics as crossing_metrics
from utils.crossing.detection import Detection
from utils.crossing.road_metrics import road_intervals_for_tracks, road_restricted_speed
from utils.crossing.waymo_calibration import (
    _load_sequences,
    _matched_waymo_speed_target,
    _predicted_crossings,
    _write_csv,
    _write_json,
)
from utils.crossing.road_crossing import (
    road_crossing_flags,
    waymo_person_tracks,
    waymo_segmentation_pipeline,
    waymo_surface_timelines,
)
from utils.segmentation.constants import SURFACE_FOOTPATH
from utils.segmentation.pipeline import SegmentationPipeline

CALIBRATION_FOLDER = "calibration_v32"
OUTPUT_FOLDER = "segmentation_evaluation"
RULES = ("detector", "detector_on_road", "road_lateral", "road_crossing", "road_edge", "road_full")


def _load_speed_model(calibration_root: Path) -> bool:
    """Load the calibration's speed model; return whether it qualified.

    load_tuned_pipeline_model only exposes a qualified model. An unqualified
    one is loaded directly so that it can still be evaluated here.
    """
    crossing_metrics.load_tuned_pipeline_model(calibration_root / "crowd_waymo_pipeline_model.json")
    if crossing_metrics._SPEED_MODEL:
        return True
    production = calibration_root / "speed_validation" / "production_model.json"
    crossing_metrics._SPEED_MODEL = json.loads(production.read_text(encoding="utf-8"))
    return False


def _evaluate_sequence(
    sequence: Any,
    split: str,
    pipeline: SegmentationPipeline,
    detector: Detection,
    crossing_parameters: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    fps = float(sequence.fps)
    detections = sequence.prediction_dataframe
    tracks = waymo_person_tracks(sequence)
    if not tracks:
        return []
    timelines = waymo_surface_timelines(sequence, split, pipeline, tracks)
    intervals = road_intervals_for_tracks(timelines)
    road_speeds, _ = road_restricted_speed(
        pl.DataFrame(), detections, sequence.source_id, fps, intervals, float(sequence.aspect_ratio),
    )

    # Whole-track bounding-box speed: the production path of calculate_speed_of_crossing.
    boxes = crossing_metrics.bbox_rows_from_polars(detections)
    features_by_track = crossing_metrics.contextual_track_features(
        crossing_metrics.group_tracks(row for row in boxes if row.class_id == crossing_metrics.PERSON_CLASS_ID),
        fps,
        sequence.source_id,
        float(sequence.aspect_ratio),
        crossing_metrics.build_scene_motion_profile(boxes, fps),
    )
    picks = _predicted_crossings(sequence, detector, crossing_parameters)
    minimum_x_range = float(crossing_parameters["min_crossing_x_range"])
    associations = {crossing_metrics.normalise_id(key): value for key, value in sequence.associations.items()}

    rows: List[Dict[str, Any]] = []
    for track_id, track in tracks.items():
        interval = intervals.get(track_id)
        samples = timelines.get(track_id, [])
        road_lateral = road_edge = road_full = False
        footpath_before = footpath_after = False
        on_road_x_range = None
        if interval is not None:
            on_road = track.filter(pl.col("frame-count").is_between(interval.entry_frame, interval.exit_frame))
            if on_road.height:
                on_road_x_range = float(on_road["x-center"].max() - on_road["x-center"].min())
            road_lateral = on_road_x_range is not None and on_road_x_range >= minimum_x_range
            footpath_before = any(
                s.surface == SURFACE_FOOTPATH and s.frame < interval.entry_frame for s in samples
            )
            footpath_after = any(
                s.surface == SURFACE_FOOTPATH and s.frame > interval.exit_frame for s in samples
            )
            road_edge = road_lateral and (footpath_before or footpath_after)
            road_full = road_lateral and footpath_before and footpath_after

        association = associations.get(track_id)
        ground_truth_id = str(association["ground_truth_id"]) if association else ""
        whole_truth = on_road_truth = None
        if association is not None:
            whole_truth, _ = _matched_waymo_speed_target(sequence, ground_truth_id, association)
            if interval is not None:
                on_road_pairs = [
                    pair for pair in association.get("matched_frame_pairs", [])
                    if interval.entry_frame <= pair[0] <= interval.exit_frame
                ]
                on_road_truth, _ = _matched_waymo_speed_target(
                    sequence, ground_truth_id, {"matched_frame_pairs": on_road_pairs},
                )

        bbox_speed = None
        features = features_by_track.get(track_id)
        if features is not None:
            prediction = crossing_metrics._predict_metric_speed(features, crossing_metrics._SPEED_MODEL)
            if prediction.get("speed_status") == "valid":
                bbox_speed = float(prediction["estimated_speed_mps"])

        flags = road_crossing_flags(track, interval, features_by_track.get(track_id), minimum_x_range)
        rows.append(
            {
                "split": split,
                "source_id": sequence.source_id,
                "prediction_track_id": track_id,
                "track_rows": track.height,
                "ground_truth_track_id": ground_truth_id,
                "matched": int(association is not None),
                "real_crossing": int(ground_truth_id in sequence.crossing_tracks),
                "detector": int(track_id in picks),
                "detector_on_road": int(track_id in picks and interval is not None),
                "road_crossing": flags["road_crossing"],
                "box_size_change_rate": (
                    flags["box_size_change_rate"] if flags["box_size_change_rate"] is not None else ""
                ),
                "road_lateral": int(road_lateral),
                "road_edge": int(road_edge),
                "road_full": int(road_full),
                "footpath_before_entry": int(footpath_before),
                "footpath_after_exit": int(footpath_after),
                "road_interval": int(interval is not None),
                "road_entry_frame": interval.entry_frame if interval else "",
                "road_exit_frame": interval.exit_frame if interval else "",
                "on_road_x_range": on_road_x_range if on_road_x_range is not None else "",
                "surface_samples": len(samples),
                "waymo_speed_whole_mps": whole_truth if whole_truth is not None else "",
                "waymo_speed_on_road_mps": on_road_truth if on_road_truth is not None else "",
                "bbox_speed_mps": bbox_speed if bbox_speed is not None else "",
                "segmentation_speed_mps": road_speeds.get(track_id, ""),
                **({f"feature_{k}": v for k, v in asdict(features).items()} if features is not None else {}),
            }
        )
    return rows


def _speed_summary(rows: Sequence[Mapping[str, Any]], estimate: str, truth: str) -> Dict[str, Any]:
    pairs = [
        (float(row[truth]), float(row[estimate]))
        for row in rows
        if row.get(estimate) not in (None, "") and row.get(truth) not in (None, "")
        and 0.10 <= float(row[truth]) <= 3.50
    ]
    if len(pairs) < 3:
        return {"count": len(pairs)}
    reference = np.asarray([pair[0] for pair in pairs])
    predicted = np.asarray([pair[1] for pair in pairs])
    error = predicted - reference
    return {
        "count": len(pairs),
        "mae_mps": float(np.abs(error).mean()),
        "median_absolute_error_mps": float(np.median(np.abs(error))),
        "bias_mps": float(error.mean()),
        "pearson_correlation": float(np.corrcoef(reference, predicted)[0, 1]),
        "within_0_25_mps": float(np.mean(np.abs(error) <= 0.25)),
    }


def _summary(rows: Sequence[Mapping[str, Any]], real_crossers: int) -> Dict[str, Any]:
    output: Dict[str, Any] = {"real_crossers": real_crossers, "tracks": len(rows), "rules": {}}
    for rule in RULES:
        positive = [row for row in rows if row[rule]]
        correct = [row for row in positive if row["real_crossing"]]
        found = len({(row["source_id"], row["ground_truth_track_id"]) for row in correct})
        output["rules"][rule] = {
            "picks": len(positive),
            "correct": len(correct),
            "precision": len(correct) / len(positive) if positive else None,
            "real_crossers_found": found,
            "recall": found / real_crossers if real_crossers else None,
            "segmentation_speed_vs_waymo_on_road": _speed_summary(
                correct, "segmentation_speed_mps", "waymo_speed_on_road_mps",
            ),
            "bbox_speed_vs_waymo_whole_track": _speed_summary(correct, "bbox_speed_mps", "waymo_speed_whole_mps"),
        }
    both = [
        row for row in rows
        if row["real_crossing"] and row["segmentation_speed_mps"] != "" and row["bbox_speed_mps"] != ""
    ]
    output["same_real_crossers_both_speeds"] = {
        "segmentation_vs_waymo_on_road": _speed_summary(both, "segmentation_speed_mps", "waymo_speed_on_road_mps"),
        "bbox_vs_waymo_whole_track": _speed_summary(both, "bbox_speed_mps", "waymo_speed_whole_mps"),
        "bbox_vs_waymo_on_road": _speed_summary(both, "bbox_speed_mps", "waymo_speed_on_road_mps"),
    }
    return output


def evaluate(splits: Sequence[str] = ("training", "validation")) -> Dict[str, Any]:
    import speed_estimation_harness as harness

    processed_root = Path(str(common.get_configs("waymo_dataset_path"))).expanduser() / "waymo_processed"
    calibration_root = processed_root / CALIBRATION_FOLDER
    output_root = processed_root / OUTPUT_FOLDER
    output_root.mkdir(parents=True, exist_ok=True)

    qualified = _load_speed_model(calibration_root)
    crossing_parameters = crossing_metrics.tuned_crossing_parameters()
    pipeline = waymo_segmentation_pipeline(processed_root, crossing_parameters)
    detector = Detection()
    minimum_confidence = float(common.get_configs("min_confidence"))

    summary: Dict[str, Any] = {
        "schema": "crowd_waymo_segmentation_evaluation_v1",
        "speed_model_qualified": qualified,
        "speed_model_note": "" if qualified else "unqualified candidate model, used for evaluation only",
        "tracking": crossing_metrics._current_tracking_settings(),
        "segmentation_model": pipeline.segmenter.model_identifier,
        "splits": {},
    }
    try:
        for split in splits:
            sequences = _load_sequences(
                str(processed_root / split / "waymo_sequence_index.csv"), harness, minimum_confidence,
            )
            rows: List[Dict[str, Any]] = []
            real_crossers = 0
            for number, sequence in enumerate(sequences, start=1):
                real_crossers += len(sequence.crossing_tracks)
                try:
                    rows.extend(_evaluate_sequence(sequence, split, pipeline, detector, crossing_parameters))
                except Exception as error:  # one bad recording must not stop the evaluation
                    print(f"{split} {sequence.source_id}: failed: {error}", flush=True)
                if number % 25 == 0 or number == len(sequences):
                    print(f"{split}: {number}/{len(sequences)} recordings, {len(rows)} tracks", flush=True)
            _write_csv(output_root / f"segmentation_evaluation_{split}.csv", rows)
            summary["splits"][split] = _summary(rows, real_crossers)
    finally:
        pipeline.close()
    summary["pipeline_statistics"] = dict(pipeline.statistics)
    _write_json(output_root / "segmentation_evaluation_summary.json", summary)
    return summary


def main(argv: Optional[Sequence[str]] = None) -> None:
    splits = list(argv if argv is not None else sys.argv[1:]) or ["training", "validation"]
    print(json.dumps(evaluate(splits)["splits"], indent=2))


if __name__ == "__main__":
    main()
