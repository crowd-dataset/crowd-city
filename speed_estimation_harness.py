from __future__ import annotations

"""Pedestrian crossing speed calibration and evaluation harness.

The CROWD detector and tracker CSV remains the only per-pedestrian input:

    yolo-id,x-center,y-center,width,height,unique-id,confidence,frame-count

Version 32 keeps the original CROWD crossing decision fixed and calibrates
only its speed estimate. It adds a compact Extra Trees regressor to represent
nonlinear interactions between twelve quantities derived from the YOLO plus
BoT SORT CSV. A bounded, source-cross-validated correction then restores
individual variation without allowing unrestricted rescaling to amplify
extreme errors. Source-grouped cross-validation selects every parameter
without Waymo validation:

* ``calibrate_waymo_pipeline`` applies ``Detection.pedestrian_crossing()`` to
  the YOLO plus BoT SORT CSV, strictly associates each selected track with one
  Waymo pedestrian, and uses Waymo planar velocity on the matched frames as
  the speed target.  The separately derived Waymo crossing label is retained
  for audit only and never selects calibration tracks.

* ``predict_metric`` maps the bottom centre of every pedestrian box through a
  per-video ground-plane calibration.  Static cameras use one homography.
  Moving dashcams require a per-frame image-to-world homography.  This is the
  production path for a speed claim in metres per second.
* ``predict_relative`` consumes only the CSV and FPS and reports a within-video
  bbox motion score.  It deliberately does not label that score as metric speed.

The CSV calibrated ``predict`` command refuses to run unless its model passed
both source grouped validation and untouched external testing.

The command line deliberately uses ``sys.argv`` rather than a parser so this
file remains compatible with the original project style.
"""

import csv
import hashlib
import importlib.metadata
import json
import math
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


RELEASE_ID = "crowd_algorithm_selected_waymo_speed_v32_20260825"
MODEL_SCHEMA = "crowd_bbox_csv_algorithm_selected_speed_model_v14"
LEGACY_MODEL_SCHEMAS = {
    "crowd_bbox_csv_algorithm_selected_speed_model_v13",
    "crowd_bbox_csv_algorithm_selected_speed_model_v12",
    "crowd_bbox_csv_algorithm_selected_speed_model_v11",
    "crowd_bbox_csv_algorithm_selected_speed_model_v10",
    "crowd_bbox_csv_algorithm_selected_speed_model_v9",
}
GROUND_CALIBRATION_SCHEMA = "crowd_ground_plane_calibration_v1"
WAYMO_CROSSING_SCHEMA = "waymo_confirmed_crosswalk_axis_speed_v4"
WAYMO_CALIBRATION_CAMERA = "FRONT"
WAYMO_CALIBRATION_TARGET = (
    "median Waymo instantaneous planar pedestrian speed over the strictly "
    "matched YOLO plus BoT SORT crossing-track frames"
)
PERSON_CLASS_ID = 0
DEFAULT_ASPECT_RATIO = 16.0 / 9.0
DEFAULT_PERSON_HEIGHT_M = 1.70
DEFAULT_RELATIVE_UNCERTAINTY_LIMIT: Optional[float] = None
DEFAULT_RANDOM_SEED = 13
DEFAULT_WAYMO_FPS = 10.0
SCENE_MOTION_SETTINGS: Dict[str, float] = {
    "maximum_pair_gap_frames": 2,
    "window_radius_frames": 3,
    "minimum_reference_tracks": 3,
    "maximum_absolute_x_rate_per_second": 3.0,
    "outlier_mad_multiplier": 4.0,
}
SOURCE_CONTEXT_SETTINGS: Dict[str, float] = {
    "minimum_reference_tracks": 3,
    "minimum_proxy_mps": 0.001,
    "maximum_absolute_log_ratio": 2.00,
}
RELIABILITY_GATES: Dict[str, float] = {
    "minimum_duration_seconds": 2.00,
    "minimum_x_fit_r2": 0.90,
}
METRIC_SPEED_GATES: Dict[str, float] = {
    "minimum_rows": 8,
    "minimum_duration_seconds": 2.00,
    "minimum_mapping_coverage": 0.80,
    "minimum_ground_displacement_m": 0.50,
    "minimum_motion_fit_r2": 0.80,
    "minimum_speed_mps": 0.10,
    "maximum_speed_mps": 3.50,
    "maximum_calibration_rmse_m": 0.25,
    "maximum_absolute_uncertainty_mps": 0.40,
    "maximum_pair_gap_seconds": 0.50,
}
CALIBRATION_MATCH_SETTINGS: Dict[str, float] = {
    "minimum_matched_frames": 8,
    "minimum_mean_iou": 0.50,
    "minimum_prediction_coverage": 0.50,
    "minimum_ground_truth_coverage": 0.20,
}
GROUP_SPLIT_SETTINGS: Dict[str, float] = {
    "validation_fraction": 0.15,
    "test_fraction": 0.25,
    "minimum_training_sources": 3,
    "minimum_validation_sources": 2,
    "minimum_test_sources": 5,
    "minimum_test_tracks": 20,
}

# Model selection is intentionally restricted to a short list of physically
# interpretable bbox motion features.  The previous thirteen feature fit used
# only thirty three training tracks and learned scene specific quality and
# scale patterns.  Source grouped cross validation now decides between these
# small candidates without looking at the held out test sources.
MODEL_CANDIDATES: Dict[str, List[str]] = {
    "raw": ["raw_speed_proxy_mps"],
    "q": ["q_speed_proxy_mps"],
    "robust": ["robust_speed_proxy_mps"],
    "raw_q": ["raw_speed_proxy_mps", "q_speed_proxy_mps"],
    "raw_robust": ["raw_speed_proxy_mps", "robust_speed_proxy_mps"],
    "proxies": [
        "raw_speed_proxy_mps",
        "q_speed_proxy_mps",
        "robust_speed_proxy_mps",
    ],
    "compensated_raw": ["compensated_raw_speed_proxy_mps"],
    "compensated_q": ["compensated_q_speed_proxy_mps"],
    "compensated_robust": ["compensated_robust_speed_proxy_mps"],
    "raw_and_compensated": [
        "raw_speed_proxy_mps",
        "compensated_raw_speed_proxy_mps",
        "scene_motion_equivalent_speed_mps",
    ],
    "compensated_proxies": [
        "compensated_raw_speed_proxy_mps",
        "compensated_q_speed_proxy_mps",
        "compensated_robust_speed_proxy_mps",
    ],
    "compensated_robust_compact_context": [
        "compensated_robust_speed_proxy_mps",
        "source_relative_log_q_proxy",
        "source_relative_log_robust_proxy",
        "source_compensated_robust_percentile",
        "source_context_available",
    ],
    "compensated_robust_full_context": [
        "compensated_robust_speed_proxy_mps",
        "source_relative_log_raw_proxy",
        "source_relative_log_q_proxy",
        "source_relative_log_robust_proxy",
        "source_relative_log_compensated_robust_proxy",
        "source_compensated_robust_percentile",
        "source_context_available",
        "q_residual_mad",
        "q_fit_r2",
        "log_duration",
    ],
}
DIRECT_PROXY_CANDIDATES: Dict[str, List[str]] = {
    "direct_raw": ["raw_speed_proxy_mps"],
    "direct_q": ["q_speed_proxy_mps"],
    "direct_robust": ["robust_speed_proxy_mps"],
    "direct_median": [
        "raw_speed_proxy_mps",
        "q_speed_proxy_mps",
        "robust_speed_proxy_mps",
    ],
    "direct_compensated_raw": ["compensated_raw_speed_proxy_mps"],
    "direct_compensated_q": ["compensated_q_speed_proxy_mps"],
    "direct_compensated_robust": ["compensated_robust_speed_proxy_mps"],
    "direct_compensated_median": [
        "compensated_raw_speed_proxy_mps",
        "compensated_q_speed_proxy_mps",
        "compensated_robust_speed_proxy_mps",
    ],
}
RIDGE_CANDIDATES = [0.00, 0.03, 0.10, 0.30, 1.00, 3.00, 10.00, 30.00]

# The v27 ridge fit retained only about forty per cent of the Waymo speed
# standard deviation.  V28 therefore adds a low complexity shape constrained
# additive model.  Its first component is a monotonic piecewise linear spline
# of the physical bbox speed proxy.  A compact challenger adds regularised
# track quality terms that can correct systematic measurement error without
# using pixels, camera pose, optical flow, or Waymo information at deployment.
MONOTONIC_SPLINE_CANDIDATES: Dict[str, Dict[str, Any]] = {
    "monotonic_spline": {
        "primary_feature": "compensated_robust_speed_proxy_mps",
        "quality_features": [],
    },
    "monotonic_spline_quality": {
        "primary_feature": "compensated_robust_speed_proxy_mps",
        "quality_features": [
            "q_residual_mad",
            "one_minus_x_r2",
            "log_duration",
            "log_median_height",
            "scene_motion_fraction",
            "log_scene_motion_support",
        ],
    },
    "monotonic_spline_source_context": {
        "primary_feature": "compensated_robust_speed_proxy_mps",
        "quality_features": [
            "source_relative_log_raw_proxy",
            "source_relative_log_q_proxy",
            "source_relative_log_robust_proxy",
            "source_relative_log_compensated_robust_proxy",
            "source_compensated_robust_percentile",
            "source_context_available",
            "q_residual_mad",
            "q_fit_r2",
            "log_duration",
        ],
    },
}
MONOTONIC_SPLINE_KNOT_CANDIDATES = [4, 6]
MONOTONIC_SPLINE_SMOOTHING_CANDIDATES = [0.30, 3.00]
MONOTONIC_SPLINE_QUALITY_RIDGE = 10.00
# The monotonic model estimates the conditional mean and consequently shrinks
# slow and fast pedestrians towards the centre of the training distribution.
# V31 tests a deliberately small, predeclared contribution from the direct
# physical bbox proxy.  Candidate selection remains leave-one-source-out on
# Waymo training sources; validation labels do not choose this weight.
MONOTONIC_SPLINE_PHYSICS_BLEND_WEIGHTS = [0.05, 0.10, 0.15]
MONOTONIC_SPLINE_PHYSICS_BLEND_PROXY = (
    "compensated_robust_speed_proxy_mps"
)
# V32 calibrates the variance of the complete V31 prediction.  Each factor is
# paired with a strict maximum correction, so even the largest factor cannot
# move one prediction by more than the declared cap.  Candidate selection uses
# only source held out Waymo development predictions.
BOUNDED_VARIANCE_FACTORS = [1.25, 1.50, 1.75, 2.00]
BOUNDED_VARIANCE_CAPS_MPS = [0.04, 0.06]
# V32 nonlinear challenger. These twelve features are all available from the
# same YOLO plus BoT SORT CSV at deployment. The forest is deliberately small,
# shallow through its leaf-size constraint, deterministic, and stored as plain
# JSON so CROWD inference does not import scikit-learn.
EXTRA_TREES_FEATURES = [
    "compensated_robust_speed_proxy_mps",
    "raw_speed_proxy_mps",
    "q_speed_proxy_mps",
    "robust_speed_proxy_mps",
    "source_relative_log_raw_proxy",
    "source_relative_log_q_proxy",
    "source_relative_log_robust_proxy",
    "source_relative_log_compensated_robust_proxy",
    "source_compensated_robust_percentile",
    "q_fit_r2",
    "duration_seconds",
    "height_ratio",
]
EXTRA_TREES_N_ESTIMATORS = 80
EXTRA_TREES_MIN_SAMPLES_LEAF = 3
EXTRA_TREES_VARIANCE_FACTORS = [1.05, 1.10, 1.15, 1.25]
EXTRA_TREES_VARIANCE_CAPS_MPS = [0.04, 0.06]
CROSS_VALIDATION_SETTINGS: Dict[str, float] = {
    "minimum_development_tracks": 30,
    "minimum_development_sources": 5,
    "simplicity_margin_mps": 0.02,
    "minimum_baseline_improvement_fraction": 0.10,
    "maximum_source_balanced_mae_mps": 0.35,
    "maximum_worst_source_mae_mps": 0.60,
    "conformal_coverage": 0.90,
    # Source grouped Waymo cross validation places the bbox only 90 percent
    # absolute error bound at about 0.39 m/s.  A 0.40 m/s limit records that
    # empirical resolution without pretending that the CSV input supports the
    # stricter 0.35 m/s precision used by earlier experimental releases.
    "maximum_conformal_absolute_uncertainty_mps": 0.40,
    "minimum_reliable_coverage": 0.50,
    "minimum_reliable_sources": 5,
    # Agreement gates prevent a low MAE model from qualifying by collapsing
    # every estimate towards the mean pedestrian speed.
    "minimum_prediction_reference_sd_ratio": 0.65,
    "maximum_prediction_reference_sd_ratio": 1.35,
    # Slope equals correlation multiplied by the prediction/reference SD
    # ratio.  Requiring 0.50 in addition to the independent correlation,
    # spread, and concordance gates forced over-dispersed estimates.  A 0.30
    # floor still rejects the V30 mean-collapsed model (0.23) while allowing a
    # calibrated estimate that retains useful individual variation.
    "minimum_calibration_slope": 0.30,
    "maximum_calibration_slope": 1.50,
    "minimum_pearson_correlation": 0.35,
    "minimum_spearman_correlation": 0.35,
    "minimum_lins_concordance": 0.30,
}
EXTERNAL_TEST_SETTINGS: Dict[str, float] = {
    "minimum_test_tracks": 20,
    "minimum_test_sources": 5,
    "maximum_unfiltered_mae_mps": 0.35,
    "maximum_unfiltered_rmse_mps": 0.50,
    "minimum_within_0_50_mps": 0.75,
    "minimum_reliable_coverage": 0.50,
    "maximum_reliable_mae_mps": 0.30,
    "minimum_prediction_reference_sd_ratio": 0.65,
    "maximum_prediction_reference_sd_ratio": 1.35,
    "minimum_calibration_slope": 0.50,
    "maximum_calibration_slope": 1.50,
    "minimum_pearson_correlation": 0.35,
    "minimum_spearman_correlation": 0.35,
    "minimum_lins_concordance": 0.30,
}

# High precision Waymo calibration labels.  A positive label requires a
# map-aligned 3D pedestrian trajectory to traverse the centre of a mapped
# crosswalk.  Pedestrians near a crosswalk, walking along a kerb, or visible
# only on one side of the crosswalk remain negative and are not used to fit the
# production bbox-only estimator.
WAYMO_CROSSING_SETTINGS: Dict[str, float] = {
    "crosswalk_boundary_tolerance_m": 0.75,
    "maximum_gap_frames": 2,
    "minimum_observations": 5,
    "minimum_duration_seconds": 0.50,
    "minimum_displacement_m": 2.00,
    "minimum_crosswalk_progress_fraction": 0.30,
    "minimum_axis_direction_fraction": 0.55,
    "minimum_combined_fit_r2": 0.30,
    "minimum_speed_mps": 0.10,
    "maximum_speed_mps": 3.50,
    "minimum_crosswalk_axis_speed_mps": 0.10,
    "maximum_crosswalk_axis_speed_mps": 3.50,
    "maximum_metadata_speed_error_mps": 0.75,
    "centre_margin_fraction": 0.05,
}

# Exact settings used by the CROWD paper/code.  track_buffer is filled from FPS.
CROWD_YOLO_MODEL = "yolo11x.pt"
CROWD_YOLO_CONFIDENCE = 0.0
CROWD_TRACK_BUFFER_SECONDS = 2.0
CROWD_BOTSORT_SETTINGS: Dict[str, Any] = {
    "tracker_type": "botsort",
    "track_high_thresh": 0.70,
    "track_low_thresh": 0.30,
    "new_track_thresh": 0.70,
    "track_buffer": None,
    "match_thresh": 0.60,
    "fuse_score": True,
    "gmc_method": "sparseOptFlow",
    "proximity_thresh": 0.50,
    "appearance_thresh": 0.25,
    "with_reid": True,
    "model": "auto",
}

MODEL_FEATURES = [
    "duration_seconds",
    "height_ratio",
    "raw_speed_proxy_mps",
    "q_speed_proxy_mps",
    "robust_speed_proxy_mps",
    "log_height_rate_abs",
    "bottom_rate_abs",
    "q_residual_mad",
    "q_fit_r2",
    "one_minus_x_r2",
    "log_height_ratio",
    "gap_ratio",
    "edge_fraction",
    "reversal_fraction",
    "log_duration",
    "log_median_height",
    "compensated_raw_speed_proxy_mps",
    "compensated_q_speed_proxy_mps",
    "compensated_robust_speed_proxy_mps",
    "scene_motion_equivalent_speed_mps",
    "scene_motion_fraction",
    "log_scene_motion_support",
    "compensated_proxy_disagreement_mps",
    "source_context_tracks",
    "source_context_available",
    "source_relative_log_raw_proxy",
    "source_relative_log_q_proxy",
    "source_relative_log_robust_proxy",
    "source_relative_log_compensated_robust_proxy",
    "source_compensated_robust_percentile",
]

# These are the quantities that can be derived from the production YOLO plus
# BoT SORT CSV.  The diagnostic command measures both pooled and within source
# association with Waymo speed.  Within source association is essential here:
# a feature that correlates only because two videos have different camera
# geometry is not evidence that it can rank pedestrians in a new CROWD video.
SPEED_INFORMATION_FEATURES = [
    "duration_seconds",
    "coverage",
    "gap_ratio",
    "median_confidence",
    "median_height",
    "median_width",
    "height_ratio",
    "horizontal_range",
    "vertical_range",
    "x_slope_per_second",
    "x_fit_r2",
    "q_slope_per_second",
    "q_fit_r2",
    "q_residual_mad",
    "log_height_rate_abs",
    "bottom_rate_abs",
    "edge_fraction",
    "truncation_fraction",
    "reversal_fraction",
    "raw_speed_proxy_mps",
    "q_speed_proxy_mps",
    "robust_speed_proxy_mps",
    "compensated_raw_speed_proxy_mps",
    "compensated_q_speed_proxy_mps",
    "compensated_robust_speed_proxy_mps",
    "scene_motion_rate_abs",
    "scene_motion_equivalent_speed_mps",
    "scene_motion_fraction",
    "scene_motion_support",
    "compensated_proxy_disagreement_mps",
    "source_context_tracks",
    "source_context_available",
    "source_relative_log_raw_proxy",
    "source_relative_log_q_proxy",
    "source_relative_log_robust_proxy",
    "source_relative_log_compensated_robust_proxy",
    "source_compensated_robust_percentile",
]

BASE_GATES: Dict[str, float] = {
    "minimum_rows": 8,
    "minimum_duration_seconds": 0.50,
    "minimum_coverage": 0.50,
    "minimum_median_height": 0.01,
    "minimum_horizontal_range": 0.08,
    "minimum_x_fit_r2": 0.15,
    "maximum_height_ratio": 3.50,
    "maximum_edge_fraction": 0.60,
    "maximum_reversal_fraction": 0.50,
    "minimum_scene_motion_support": 3,
    "minimum_predicted_speed_mps": 0.10,
    "maximum_predicted_speed_mps": 3.50,
}

MANIFEST_FIELDS = [
    "source_id",
    "bbox_csv",
    "fps",
    "aspect_ratio",
    "prediction_track_id",
    "ground_truth_track_id",
    "ground_truth_speed_mps",
    "calibration_target",
    "selection_rule",
    "algorithm_predicted_crossing",
    "waymo_derived_crossing_label",
    "ground_truth_speed_sample_count",
    "speed_target_source",
    "split",
    "include",
    "exclusion_reason",
    "matched_frames",
    "mean_iou",
    "prediction_match_coverage",
    "ground_truth_match_coverage",
    "frame_offset",
]

GROUND_TRUTH_BBOX_FIELDS = [
    "source_id",
    "frame-count",
    "timestamp_micros",
    "ground_truth_track_id",
    "camera_track_id",
    "x-center",
    "y-center",
    "width",
    "height",
    "ground_truth_speed_mps",
    "ground_truth_crosswalk_axis_speed_mps",
    "ground_truth_lateral_speed_mps",
    "ground_truth_total_speed_mps",
    "instantaneous_speed_mps",
    "instantaneous_total_speed_mps",
    "instantaneous_lateral_speed_mps",
    "instantaneous_lateral_velocity_mps",
    "map_x",
    "map_y",
    "crossing_track",
    "crossing_label",
    "crosswalk_id",
    "crossing_start_frame",
    "crossing_end_frame",
    "crossing_observations",
    "crossing_duration_seconds",
    "crossing_displacement_m",
    "crosswalk_progress_fraction",
    "axis_direction_fraction",
    "trajectory_fit_r2",
    "metadata_speed_mps",
    "metadata_total_speed_mps",
    "metadata_lateral_speed_mps",
    "lateral_speed_fraction",
    "metadata_speed_error_mps",
]

WAYMO_CROSSING_TRACK_FIELDS = [
    "source_id",
    "ground_truth_track_id",
    "camera_rows",
    "trajectory_rows",
    "crossing_track",
    "crossing_label",
    "rejection_reason",
    "crosswalk_id",
    "crossing_start_frame",
    "crossing_end_frame",
    "crossing_observations",
    "crossing_duration_seconds",
    "crossing_displacement_m",
    "crosswalk_progress_fraction",
    "axis_direction_fraction",
    "trajectory_fit_r2",
    "ground_truth_speed_mps",
    "ground_truth_crosswalk_axis_speed_mps",
    "ground_truth_lateral_speed_mps",
    "ground_truth_total_speed_mps",
    "metadata_speed_mps",
    "metadata_total_speed_mps",
    "metadata_lateral_speed_mps",
    "lateral_speed_fraction",
    "metadata_speed_error_mps",
]

WAYMO_SEQUENCE_INDEX_FIELDS = [
    "source_id",
    "video_path",
    "ground_truth_bbox_csv",
    "crossing_tracks_csv",
    "fps",
    "aspect_ratio",
    "frames",
    "ground_truth_rows",
    "ground_truth_tracks",
    "all_pedestrian_rows",
    "all_pedestrian_tracks",
    "crosswalk_features",
    "ground_truth_target",
    "prediction_bbox_csv",
    "manifest_csv",
]


@dataclass
class BBoxRow:
    class_id: int
    x: float
    y: float
    width: float
    height: float
    track_id: str
    confidence: float
    frame: int


@dataclass
class GroundCalibration:
    calibration_id: str
    mode: str
    image_coordinates: str
    frame_width: Optional[int]
    frame_height: Optional[int]
    camera_is_static: bool
    static_homography: Optional[np.ndarray]
    homographies_by_frame: Dict[int, np.ndarray]
    maximum_frame_gap: int
    crossing_axis_world: Optional[np.ndarray]
    calibration_rmse_m: float

    def homography_for_frame(self, frame: int) -> Optional[np.ndarray]:
        if self.static_homography is not None:
            return self.static_homography
        exact = self.homographies_by_frame.get(int(frame))
        if exact is not None:
            return exact
        if self.maximum_frame_gap <= 0 or not self.homographies_by_frame:
            return None
        nearest_frame = min(
            self.homographies_by_frame,
            key=lambda candidate: abs(candidate - int(frame)),
        )
        if abs(nearest_frame - int(frame)) > self.maximum_frame_gap:
            return None
        return self.homographies_by_frame[nearest_frame]


@dataclass
class SceneMotionProfile:
    samples_by_frame: Dict[int, List[Tuple[float, str]]]

    def rate_at(self, frame: float) -> Tuple[float, int]:
        radius = int(SCENE_MOTION_SETTINGS["window_radius_frames"])
        centre_frame = int(round(frame))
        samples: List[Tuple[float, str]] = []
        for candidate_frame in range(centre_frame - radius, centre_frame + radius + 1):
            samples.extend(self.samples_by_frame.get(candidate_frame, []))
        reference_tracks = {track_id for _, track_id in samples}
        support = len(reference_tracks)
        if support < int(SCENE_MOTION_SETTINGS["minimum_reference_tracks"]):
            return 0.0, support
        rates = np.asarray([rate for rate, _ in samples], dtype=float)
        centre = float(np.median(rates))
        scale = robust_scale(rates)
        limit = float(SCENE_MOTION_SETTINGS["outlier_mad_multiplier"]) * scale
        retained = rates[np.abs(rates - centre) <= limit]
        if len(retained):
            centre = float(np.median(retained))
        return centre, support


@dataclass
class TrackFeatures:
    source_id: str
    prediction_track_id: str
    input_rows: int
    clean_rows: int
    removed_rows: int
    first_frame: int
    last_frame: int
    duration_seconds: float
    coverage: float
    gap_ratio: float
    median_confidence: float
    median_height: float
    median_width: float
    height_ratio: float
    horizontal_range: float
    vertical_range: float
    crosses_image_midline: bool
    overlaps_central_band: bool
    direction: str
    x_slope_per_second: float
    x_fit_r2: float
    q_slope_per_second: float
    q_fit_r2: float
    q_residual_mad: float
    log_height_rate_abs: float
    bottom_rate_abs: float
    edge_fraction: float
    truncation_fraction: float
    reversal_fraction: float
    raw_speed_proxy_mps: float
    q_speed_proxy_mps: float
    robust_speed_proxy_mps: float
    compensated_raw_speed_proxy_mps: float
    compensated_q_speed_proxy_mps: float
    compensated_robust_speed_proxy_mps: float
    scene_motion_rate_abs: float
    scene_motion_equivalent_speed_mps: float
    scene_motion_fraction: float
    scene_motion_support: float
    log_scene_motion_support: float
    compensated_proxy_disagreement_mps: float
    one_minus_x_r2: float
    log_height_ratio: float
    log_duration: float
    log_median_height: float
    source_context_tracks: float = 0.0
    source_context_available: float = 0.0
    source_relative_log_raw_proxy: float = 0.0
    source_relative_log_q_proxy: float = 0.0
    source_relative_log_robust_proxy: float = 0.0
    source_relative_log_compensated_robust_proxy: float = 0.0
    source_compensated_robust_percentile: float = 0.5

    def model_vector(self, feature_names: Optional[Sequence[str]] = None) -> np.ndarray:
        selected = list(feature_names) if feature_names is not None else MODEL_FEATURES
        values = [float(getattr(self, name)) for name in selected]
        return np.asarray(values, dtype=float)


def log(message: str) -> None:
    print(message, flush=True)


def fail(message: str, exit_code: int = 2) -> None:
    raise SystemExit(f"ERROR: {message}")


def safe_float(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def safe_int(value: Any) -> Optional[int]:
    try:
        return int(round(float(value)))
    except (TypeError, ValueError, OverflowError):
        return None


def normalise_id(value: Any) -> str:
    text = str(value).strip()
    try:
        number = float(text)
        if math.isfinite(number) and number.is_integer():
            return str(int(number))
    except ValueError:
        pass
    return text


def truthy(value: Any, default: bool = True) -> bool:
    if value is None or str(value).strip() == "":
        return default
    return str(value).strip().lower() not in {"0", "false", "no", "off", "exclude"}


def canonical_header(value: str) -> str:
    return "".join(ch for ch in str(value).lower() if ch.isalnum())


def first_column(fieldnames: Sequence[str], aliases: Sequence[str], required: bool = True) -> Optional[str]:
    lookup = {canonical_header(name): name for name in fieldnames}
    for alias in aliases:
        actual = lookup.get(canonical_header(alias))
        if actual is not None:
            return actual
    if required:
        fail(f"CSV is missing a required column. Accepted names: {', '.join(aliases)}")
    return None


def read_dict_csv(path: str) -> List[Dict[str, str]]:
    input_path = Path(path).expanduser().resolve()
    if not input_path.is_file():
        fail(f"CSV does not exist: {input_path}")
    with input_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            fail(f"CSV has no header: {input_path}")
        return [dict(row) for row in reader]


def write_dict_csv(path: str, rows: Sequence[Dict[str, Any]], fieldnames: Optional[Sequence[str]] = None) -> None:
    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        ordered: List[str] = []
        seen = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    ordered.append(key)
        fieldnames = ordered
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: str, value: Any) -> None:
    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def package_version(name: str) -> Optional[str]:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def build_info() -> Dict[str, Any]:
    tracker = dict(CROWD_BOTSORT_SETTINGS)
    tracker["track_buffer"] = "round(2.0 * video_fps)"
    return {
        "release_id": RELEASE_ID,
        "model_schema": MODEL_SCHEMA,
        "script_path": str(Path(__file__).resolve()),
        "recommended_production_mode": "calibrate_waymo_pipeline then analysis.py",
        "production_input": (
            "CROWD YOLO plus BoT SORT bounding box CSV and video FPS"
        ),
        "production_modes": {
            "predict_metric": {
                "output": "absolute ground-plane pedestrian speed in metres per second",
                "inputs": [
                    "person bbox CSV",
                    "BoT SORT track ID",
                    "frame number",
                    "video FPS",
                    "per-video ground-plane calibration JSON",
                ],
                "moving_camera_requirement": (
                    "a per-frame image-to-world homography in a common metric frame"
                ),
            },
            "predict_relative": {
                "output": "within-video bbox motion score, not metric speed",
                "inputs": ["all class bbox CSV", "video FPS"],
            },
            "predict": {
                "output": "qualified calibrated bbox speed model",
                "release_gate": (
                    "requires source grouped validation and untouched external test"
                ),
            },
        },
        "production_uses_pixels": False,
        "production_uses_camera_pose": False,
        "production_uses_optical_flow": False,
        "speed_information_diagnostic": {
            "automatic_after_fit_and_evaluation": True,
            "speed_bins_mps": ["below 1.00", "1.00 to 1.80", "above 1.80"],
            "feature_signal_test": (
                "pooled and within source correlation with Waymo reference speed"
            ),
            "validation_use": "audit only; never used for model selection",
        },
        "speed_interpretation": (
            "Waymo calibrated planar pedestrian speed for tracks selected as "
            "crossing by the original CROWD algorithm"
        ),
        "crossing_parameter_protocol": (
            "keep the original Detection.pedestrian_crossing thresholds fixed; "
            "Waymo crossing labels do not select speed-calibration tracks"
        ),
        "ground_calibration_schema": GROUND_CALIBRATION_SCHEMA,
        "metric_speed_gates": dict(METRIC_SPEED_GATES),
        "calibration_dataset": "Waymo Open Perception moving-camera data",
        "calibration_camera": WAYMO_CALIBRATION_CAMERA,
        "calibration_target": WAYMO_CALIBRATION_TARGET,
        "calibration_target_note": (
            "The CROWD algorithm first selects crossing tracks from YOLO plus "
            "BoT SORT CSV data. Strictly matched Waymo pedestrian velocity then "
            "supplies the metric target, regardless of the derived Waymo "
            "crosswalk label."
        ),
        "transfer_dataset": "nuScenes moving-camera data",
        "deployment_dataset": "CROWD",
        "waymo_crossing_schema": WAYMO_CROSSING_SCHEMA,
        "waymo_crossing_definition": (
            "map-aligned 3D pedestrian trajectory traverses the centre of a mapped crosswalk"
        ),
        "waymo_crossing_scope": (
            "diagnostic only; not used to include or exclude speed-calibration tracks"
        ),
        "waymo_crossing_settings": dict(WAYMO_CROSSING_SETTINGS),
        "calibration_match_settings": dict(CALIBRATION_MATCH_SETTINGS),
        "scene_motion_settings": dict(SCENE_MOTION_SETTINGS),
        "source_context_settings": dict(SOURCE_CONTEXT_SETTINGS),
        "source_context_protocol": (
            "label free log ratios and percentile ranks are computed from all "
            "base eligible pedestrian tracks in each input bbox CSV; Waymo "
            "labels are never used to construct source context"
        ),
        "reliability_gates": dict(RELIABILITY_GATES),
        "compensated_proxy_disagreement_role": "diagnostic_only",
        "compensated_proxy_disagreement_used_as_rejection_gate": False,
        "group_split_settings": dict(GROUP_SPLIT_SETTINGS),
        "model_candidates": dict(MODEL_CANDIDATES),
        "direct_proxy_candidates": dict(DIRECT_PROXY_CANDIDATES),
        "monotonic_spline_candidates": dict(MONOTONIC_SPLINE_CANDIDATES),
        "monotonic_spline_knot_candidates": list(
            MONOTONIC_SPLINE_KNOT_CANDIDATES
        ),
        "monotonic_spline_smoothing_candidates": list(
            MONOTONIC_SPLINE_SMOOTHING_CANDIDATES
        ),
        "monotonic_spline_quality_ridge": MONOTONIC_SPLINE_QUALITY_RIDGE,
        "monotonic_spline_physics_blend": {
            "proxy_feature": MONOTONIC_SPLINE_PHYSICS_BLEND_PROXY,
            "candidate_weights": list(
                MONOTONIC_SPLINE_PHYSICS_BLEND_WEIGHTS
            ),
            "selection_data": "development source grouped cross validation only",
        },
        "bounded_variance_calibration": {
            "candidate_expansion_factors": list(
                BOUNDED_VARIANCE_FACTORS
            ),
            "candidate_correction_caps_mps": list(
                BOUNDED_VARIANCE_CAPS_MPS
            ),
            "centre": "mean of per-source means",
            "selection_data": (
                "source-held-out development predictions only"
            ),
        },
        "extra_trees_nonlinear_candidate": {
            "feature_names": list(EXTRA_TREES_FEATURES),
            "n_estimators": EXTRA_TREES_N_ESTIMATORS,
            "minimum_samples_per_leaf": EXTRA_TREES_MIN_SAMPLES_LEAF,
            "random_seed": DEFAULT_RANDOM_SEED,
            "variance_expansion_factors": list(
                EXTRA_TREES_VARIANCE_FACTORS
            ),
            "variance_correction_caps_mps": list(
                EXTRA_TREES_VARIANCE_CAPS_MPS
            ),
            "selection_data": (
                "development source grouped cross validation only"
            ),
            "production_serialisation": "plain JSON tree arrays",
        },
        "ridge_candidates": list(RIDGE_CANDIDATES),
        "cross_validation_settings": dict(CROSS_VALIDATION_SETTINGS),
        "external_test_settings": dict(EXTERNAL_TEST_SETTINGS),
        "model_selection": (
            "leave one source out on development sources; a compact Extra Trees "
            "regressor, monotonic spline GAM, conservative physics blends, "
            "calibrated ridge regressions and deterministic physical bbox proxies "
            "compete without test labels; candidates must "
            "pass MAE, uncertainty, spread, calibration, rank and concordance "
            "gates, then concordance breaks ties inside the MAE margin"
        ),
        "development_protocol": (
            "every already examined source is development data and is eligible "
            "only for source grouped cross validation"
        ),
        "test_use": "Waymo validation only; never used for speed-model selection",
        "external_test_preflight": (
            "requires a qualified candidate, minimum sample size, and zero "
            "development source overlap before test metrics are calculated"
        ),
        "yolo_model": CROWD_YOLO_MODEL,
        "yolo_confidence": CROWD_YOLO_CONFIDENCE,
        "track_buffer_seconds": CROWD_TRACK_BUFFER_SECONDS,
        "botsort": tracker,
        "relative_uncertainty_limit": DEFAULT_RELATIVE_UNCERTAINTY_LIMIT,
        "relative_uncertainty_used_as_rejection_gate": False,
        "maximum_absolute_uncertainty_mps": CROSS_VALIDATION_SETTINGS[
            "maximum_conformal_absolute_uncertainty_mps"
        ],
        "numpy_version": package_version("numpy"),
        "ultralytics_version": package_version("ultralytics"),
        "waymo_open_dataset_version": package_version("waymo-open-dataset-tf-2-12-0")
        or package_version("waymo-open-dataset"),
    }


def load_bbox_csv(path: str, person_only: bool = True) -> List[BBoxRow]:
    input_path = Path(path).expanduser().resolve()
    if not input_path.is_file():
        fail(f"Bounding box CSV does not exist: {input_path}")
    with input_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            fail(f"Bounding box CSV has no header: {input_path}")
        names = reader.fieldnames
        class_col = first_column(names, ["yolo-id", "yolo_id", "class-id", "class_id", "class"])
        x_col = first_column(names, ["x-center", "x_center", "xcentre", "xcenter"])
        y_col = first_column(names, ["y-center", "y_center", "ycentre", "ycenter"])
        width_col = first_column(names, ["width", "bbox_width", "w"])
        height_col = first_column(names, ["height", "bbox_height", "h"])
        track_col = first_column(names, ["unique-id", "unique_id", "track-id", "track_id", "prediction_track_id"])
        conf_col = first_column(names, ["confidence", "conf", "score"], required=False)
        frame_col = first_column(names, ["frame-count", "frame_count", "frame-id", "frame_id", "frame"])

        output: List[BBoxRow] = []
        for raw in reader:
            class_id = safe_int(raw.get(class_col))
            x = safe_float(raw.get(x_col))
            y = safe_float(raw.get(y_col))
            width = safe_float(raw.get(width_col))
            height = safe_float(raw.get(height_col))
            frame = safe_int(raw.get(frame_col))
            track_id = normalise_id(raw.get(track_col, ""))
            confidence = safe_float(raw.get(conf_col)) if conf_col else 1.0
            if None in (class_id, x, y, width, height, frame) or not track_id:
                continue
            if person_only and class_id != PERSON_CLASS_ID:
                continue
            if width <= 0.0 or height <= 0.0 or confidence is None:
                continue
            output.append(
                BBoxRow(
                    class_id=int(class_id),
                    x=float(x),
                    y=float(y),
                    width=float(width),
                    height=float(height),
                    track_id=track_id,
                    confidence=float(confidence),
                    frame=int(frame),
                )
            )
    return output


def group_tracks(rows: Sequence[BBoxRow]) -> Dict[str, List[BBoxRow]]:
    grouped: Dict[str, List[BBoxRow]] = defaultdict(list)
    for row in rows:
        grouped[row.track_id].append(row)
    return dict(grouped)


def rolling_median(values: np.ndarray, window: int = 3) -> np.ndarray:
    if len(values) < 3 or window <= 1:
        return values.astype(float, copy=True)
    radius = window // 2
    output = np.empty(len(values), dtype=float)
    for index in range(len(values)):
        left = max(0, index - radius)
        right = min(len(values), index + radius + 1)
        output[index] = float(np.median(values[left:right]))
    return output


def robust_scale(values: np.ndarray) -> float:
    if len(values) == 0:
        return 0.0
    centre = float(np.median(values))
    mad = float(np.median(np.abs(values - centre)))
    return max(1.4826 * mad, 1e-9)


def robust_line(time_values: np.ndarray, values: np.ndarray) -> Tuple[float, float, float, float]:
    if len(time_values) < 2 or float(np.ptp(time_values)) <= 0.0:
        return 0.0, float(values[0]) if len(values) else 0.0, 0.0, 0.0
    design = np.column_stack([np.ones(len(time_values)), time_values])
    beta = np.linalg.lstsq(design, values, rcond=None)[0]
    weights = np.ones(len(values), dtype=float)
    for _ in range(20):
        residual = values - design @ beta
        scale = robust_scale(residual)
        threshold = 1.345 * scale
        weights = np.ones(len(values), dtype=float)
        large = np.abs(residual) > threshold
        weights[large] = threshold / np.maximum(np.abs(residual[large]), 1e-12)
        weighted_design = design * np.sqrt(weights)[:, None]
        weighted_values = values * np.sqrt(weights)
        new_beta = np.linalg.lstsq(weighted_design, weighted_values, rcond=None)[0]
        if float(np.max(np.abs(new_beta - beta))) < 1e-9:
            beta = new_beta
            break
        beta = new_beta
    fitted = design @ beta
    residual = values - fitted
    weighted_mean = float(np.average(values, weights=weights))
    ss_res = float(np.sum(weights * residual * residual))
    ss_tot = float(np.sum(weights * (values - weighted_mean) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    return float(beta[1]), float(beta[0]), float(max(-1.0, min(1.0, r2))), robust_scale(residual)


def homography_matrix(value: Any, label: str) -> np.ndarray:
    try:
        matrix = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        fail(f"{label} must contain nine finite numbers")
    if matrix.shape == (9,):
        matrix = matrix.reshape(3, 3)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        fail(f"{label} must be a finite 3 by 3 matrix")
    scale = float(matrix[2, 2])
    if abs(scale) > 1e-12:
        matrix = matrix / scale
    if np.linalg.matrix_rank(matrix) < 3:
        fail(f"{label} is singular")
    return matrix


def apply_homography(matrix: np.ndarray, point: Sequence[float]) -> Optional[np.ndarray]:
    if len(point) != 2:
        return None
    value = np.asarray([float(point[0]), float(point[1]), 1.0], dtype=float)
    projected = matrix @ value
    if not np.all(np.isfinite(projected)) or abs(float(projected[2])) <= 1e-12:
        return None
    output = projected[:2] / projected[2]
    return output if np.all(np.isfinite(output)) else None


def fit_homography_from_points(
    image_points: Any,
    world_points: Any,
    label: str,
) -> Tuple[np.ndarray, float]:
    try:
        image = np.asarray(image_points, dtype=float)
        world = np.asarray(world_points, dtype=float)
    except (TypeError, ValueError):
        fail(f"{label} point correspondences must be numeric")
    if image.ndim != 2 or image.shape[1:] != (2,):
        fail(f"{label} image_points must have shape N by 2")
    if world.shape != image.shape:
        fail(f"{label} world_points_m must match image_points")
    if len(image) < 4:
        fail(f"{label} requires at least four point correspondences")
    if not np.all(np.isfinite(image)) or not np.all(np.isfinite(world)):
        fail(f"{label} point correspondences must be finite")

    design_rows: List[List[float]] = []
    targets: List[float] = []
    for (u, v), (x_value, y_value) in zip(image, world):
        design_rows.append(
            [u, v, 1.0, 0.0, 0.0, 0.0, -x_value * u, -x_value * v]
        )
        targets.append(float(x_value))
        design_rows.append(
            [0.0, 0.0, 0.0, u, v, 1.0, -y_value * u, -y_value * v]
        )
        targets.append(float(y_value))
    design = np.asarray(design_rows, dtype=float)
    target = np.asarray(targets, dtype=float)
    if np.linalg.matrix_rank(design) < 8:
        fail(f"{label} point layout is degenerate")
    coefficients = np.linalg.lstsq(design, target, rcond=None)[0]
    matrix = np.asarray(
        [
            [coefficients[0], coefficients[1], coefficients[2]],
            [coefficients[3], coefficients[4], coefficients[5]],
            [coefficients[6], coefficients[7], 1.0],
        ],
        dtype=float,
    )
    errors: List[float] = []
    for source, expected in zip(image, world):
        projected = apply_homography(matrix, source)
        if projected is None:
            fail(f"{label} produced an invalid projection")
        errors.append(float(np.linalg.norm(projected - expected)))
    rmse = float(math.sqrt(np.mean(np.square(errors))))
    return matrix, rmse


def calibration_entry_homography(
    entry: Dict[str, Any],
    label: str,
) -> Tuple[np.ndarray, float]:
    direct = entry.get("image_to_world_homography")
    if direct is None:
        direct = entry.get("pixel_to_ground_homography")
    stated_rmse = safe_float(entry.get("calibration_rmse_m"))
    if direct is not None:
        if stated_rmse is None or stated_rmse < 0.0:
            fail(
                f"{label} must provide calibration_rmse_m when a direct "
                "homography is supplied"
            )
        return homography_matrix(direct, label), float(stated_rmse)

    image_points = entry.get("image_points")
    world_points = entry.get("world_points_m")
    if world_points is None:
        world_points = entry.get("ground_points_m")
    if stated_rmse is None or stated_rmse < 0.0:
        fail(
            f"{label} must provide a non-negative calibration_rmse_m "
            "measured on independent road points"
        )
    matrix, measured_rmse = fit_homography_from_points(
        image_points,
        world_points,
        label,
    )
    measured_rmse = max(measured_rmse, float(stated_rmse))
    return matrix, measured_rmse


def load_ground_calibration(path: str) -> GroundCalibration:
    calibration_path = Path(path).expanduser().resolve()
    if not calibration_path.is_file():
        fail(f"Ground calibration JSON does not exist: {calibration_path}")
    try:
        with calibration_path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        fail(f"Cannot read ground calibration JSON: {error}")
    if not isinstance(value, dict):
        fail("Ground calibration JSON must contain an object")
    if value.get("schema") != GROUND_CALIBRATION_SCHEMA:
        fail(
            f"Unsupported ground calibration schema {value.get('schema')!r}; "
            f"expected {GROUND_CALIBRATION_SCHEMA!r}"
        )

    mode = str(value.get("mode", "")).strip().lower()
    if mode not in {"static_pixel_to_ground", "per_frame_pixel_to_world"}:
        fail(
            "Ground calibration mode must be static_pixel_to_ground or "
            "per_frame_pixel_to_world"
        )
    coordinate_text = str(value.get("image_coordinates", "normalised")).strip().lower()
    if coordinate_text == "normalized":
        coordinate_text = "normalised"
    if coordinate_text not in {"normalised", "pixels"}:
        fail("image_coordinates must be normalised or pixels")
    frame_width = safe_int(value.get("frame_width"))
    frame_height = safe_int(value.get("frame_height"))
    if coordinate_text == "pixels" and (
        frame_width is None
        or frame_height is None
        or frame_width <= 0
        or frame_height <= 0
    ):
        fail("Pixel calibration requires positive frame_width and frame_height")

    axis_value = value.get("crossing_axis_world")
    crossing_axis: Optional[np.ndarray] = None
    if axis_value is not None:
        try:
            axis = np.asarray(axis_value, dtype=float)
        except (TypeError, ValueError):
            fail("crossing_axis_world must contain two finite numbers")
        if axis.shape != (2,) or not np.all(np.isfinite(axis)):
            fail("crossing_axis_world must contain two finite numbers")
        norm = float(np.linalg.norm(axis))
        if norm <= 1e-12:
            fail("crossing_axis_world cannot be zero")
        crossing_axis = axis / norm

    static_homography: Optional[np.ndarray] = None
    homographies_by_frame: Dict[int, np.ndarray] = {}
    rmse_values: List[float] = []
    camera_is_static = truthy(value.get("camera_is_static"), False)
    if mode == "static_pixel_to_ground":
        if not camera_is_static:
            fail(
                "static_pixel_to_ground is valid only when camera_is_static is true. "
                "A moving dashcam requires per_frame_pixel_to_world."
            )
        static_homography, rmse = calibration_entry_homography(
            value,
            "static ground calibration",
        )
        rmse_values.append(rmse)
    else:
        raw_entries = value.get("frame_calibrations")
        if not isinstance(raw_entries, list) or not raw_entries:
            fail("per_frame_pixel_to_world requires non-empty frame_calibrations")
        for index, raw_entry in enumerate(raw_entries):
            if not isinstance(raw_entry, dict):
                fail(f"frame_calibrations item {index} must be an object")
            frame = safe_int(raw_entry.get("frame"))
            if frame is None or frame < 0:
                fail(f"frame_calibrations item {index} has an invalid frame")
            if frame in homographies_by_frame:
                fail(f"Duplicate frame calibration for frame {frame}")
            matrix, rmse = calibration_entry_homography(
                raw_entry,
                f"frame {frame} calibration",
            )
            homographies_by_frame[frame] = matrix
            rmse_values.append(rmse)

    maximum_frame_gap = safe_int(value.get("maximum_frame_gap"))
    if maximum_frame_gap is None:
        maximum_frame_gap = 0
    if maximum_frame_gap < 0:
        fail("maximum_frame_gap cannot be negative")
    calibration_rmse = max(rmse_values) if rmse_values else math.inf
    return GroundCalibration(
        calibration_id=str(value.get("calibration_id", calibration_path.stem)),
        mode=mode,
        image_coordinates=coordinate_text,
        frame_width=frame_width,
        frame_height=frame_height,
        camera_is_static=camera_is_static,
        static_homography=static_homography,
        homographies_by_frame=homographies_by_frame,
        maximum_frame_gap=maximum_frame_gap,
        crossing_axis_world=crossing_axis,
        calibration_rmse_m=float(calibration_rmse),
    )


def bbox_footpoint_in_calibration_coordinates(
    row: BBoxRow,
    calibration: GroundCalibration,
) -> np.ndarray:
    point = np.asarray([row.x, row.y + 0.50 * row.height], dtype=float)
    if calibration.image_coordinates == "pixels":
        point[0] *= float(calibration.frame_width)
        point[1] *= float(calibration.frame_height)
    return point


def mode_make_ground_calibration_template(
    camera_mode: str,
    frame_width: int,
    frame_height: int,
    output_json: str,
) -> None:
    if frame_width <= 0 or frame_height <= 0:
        fail("Frame width and height must be positive")
    selected = camera_mode.strip().lower()
    if selected not in {"static", "moving"}:
        fail("Camera mode must be static or moving")
    value: Dict[str, Any] = {
        "schema": GROUND_CALIBRATION_SCHEMA,
        "calibration_id": Path(output_json).stem,
        "mode": (
            "static_pixel_to_ground"
            if selected == "static"
            else "per_frame_pixel_to_world"
        ),
        "image_coordinates": "pixels",
        "frame_width": int(frame_width),
        "frame_height": int(frame_height),
        "camera_is_static": selected == "static",
        "crossing_axis_world": None,
        "maximum_frame_gap": 0,
        "instructions": [
            "Use at least four non-collinear road-plane reference points.",
            "image_points are [pixel_x, pixel_y].",
            "world_points_m are the matching [x_m, y_m] points in one metric frame.",
            "calibration_rmse_m must be measured on independent road points.",
            "Set crossing_axis_world to a unit direction only when axis speed is required.",
            "A moving camera needs a frame_calibrations entry for every evaluated frame.",
        ],
    }
    if selected == "static":
        value["image_points"] = []
        value["world_points_m"] = []
        value["calibration_rmse_m"] = None
    else:
        value["frame_calibrations"] = []
    write_json(output_json, value)
    log(f"Calibration template: {Path(output_json).expanduser().resolve()}")


def mode_validate_ground_calibration(calibration_json: str) -> None:
    calibration = load_ground_calibration(calibration_json)
    summary = {
        "schema": GROUND_CALIBRATION_SCHEMA,
        "calibration_id": calibration.calibration_id,
        "mode": calibration.mode,
        "image_coordinates": calibration.image_coordinates,
        "camera_is_static": calibration.camera_is_static,
        "calibrated_frames": len(calibration.homographies_by_frame),
        "maximum_frame_gap": calibration.maximum_frame_gap,
        "crossing_axis_configured": calibration.crossing_axis_world is not None,
        "calibration_rmse_m": calibration.calibration_rmse_m,
        "passes_rmse_gate": (
            calibration.calibration_rmse_m
            <= METRIC_SPEED_GATES["maximum_calibration_rmse_m"]
        ),
    }
    log(json.dumps(summary, indent=2, sort_keys=True))


def waymo_map_aligned_point(frame: Any, box: Any) -> Optional[Tuple[float, float, float]]:
    """Transform a Waymo vehicle-frame label centre into the map frame."""
    values = list(frame.pose.transform) if frame.HasField("pose") else []
    if len(values) != 16:
        return None
    transform = np.asarray(values, dtype=float).reshape(4, 4)
    local = np.asarray(
        [float(box.center_x), float(box.center_y), float(box.center_z), 1.0],
        dtype=float,
    )
    global_point = transform @ local
    if frame.HasField("map_pose_offset"):
        global_point[0] += float(frame.map_pose_offset.x)
        global_point[1] += float(frame.map_pose_offset.y)
        global_point[2] += float(frame.map_pose_offset.z)
    if not np.all(np.isfinite(global_point[:3])):
        return None
    return float(global_point[0]), float(global_point[1]), float(global_point[2])


def waymo_crosswalk_polygons(frame: Any) -> List[Dict[str, Any]]:
    crosswalks: List[Dict[str, Any]] = []
    for feature in frame.map_features:
        if feature.WhichOneof("feature_data") != "crosswalk":
            continue
        polygon = [
            (float(point.x), float(point.y))
            for point in feature.crosswalk.polygon
            if math.isfinite(float(point.x)) and math.isfinite(float(point.y))
        ]
        if len(polygon) >= 3:
            crosswalks.append({"crosswalk_id": str(feature.id), "polygon": polygon})
    return crosswalks


def point_in_polygon(point: Tuple[float, float], polygon: Sequence[Tuple[float, float]]) -> bool:
    x, y = point
    inside = False
    previous_x, previous_y = polygon[-1]
    for current_x, current_y in polygon:
        if (current_y > y) != (previous_y > y):
            denominator = previous_y - current_y
            intersection_x = (
                (previous_x - current_x) * (y - current_y)
                / denominator
                + current_x
            )
            if x < intersection_x:
                inside = not inside
        previous_x, previous_y = current_x, current_y
    return inside


def point_segment_distance(
    point: Tuple[float, float],
    start: Tuple[float, float],
    end: Tuple[float, float],
) -> float:
    point_array = np.asarray(point, dtype=float)
    start_array = np.asarray(start, dtype=float)
    segment = np.asarray(end, dtype=float) - start_array
    denominator = float(segment @ segment)
    if denominator <= 1e-12:
        return float(np.linalg.norm(point_array - start_array))
    position = float(np.clip(((point_array - start_array) @ segment) / denominator, 0.0, 1.0))
    return float(np.linalg.norm(point_array - (start_array + position * segment)))


def point_polygon_distance(point: Tuple[float, float], polygon: Sequence[Tuple[float, float]]) -> float:
    if point_in_polygon(point, polygon):
        return 0.0
    return min(
        point_segment_distance(point, polygon[index - 1], polygon[index])
        for index in range(len(polygon))
    )


def crosswalk_axis(
    polygon: Sequence[Tuple[float, float]],
) -> Optional[Tuple[np.ndarray, np.ndarray, float, float]]:
    points = np.asarray(polygon, dtype=float)
    if points.shape[0] < 3:
        return None
    centre = np.mean(points, axis=0)
    centred = points - centre
    covariance = centred.T @ centred / max(1, len(points))
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    major_axis = eigenvectors[:, int(np.argmax(eigenvalues))]
    minor_axis = np.asarray([-major_axis[1], major_axis[0]], dtype=float)
    major_length = float(np.ptp(centred @ major_axis))
    minor_length = float(np.ptp(centred @ minor_axis))
    if major_length <= 1e-6 or minor_length <= 1e-6:
        return None
    return centre, major_axis, major_length, minor_length


def consecutive_waymo_groups(
    samples: Sequence[Dict[str, Any]],
    selected_indices: Sequence[int],
    maximum_gap_frames: int,
) -> List[List[Dict[str, Any]]]:
    groups: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []
    previous_frame: Optional[int] = None
    for sample_index in selected_indices:
        sample = samples[sample_index]
        frame = int(sample["frame-count"])
        if previous_frame is not None and frame - previous_frame > maximum_gap_frames + 1:
            if current:
                groups.append(current)
            current = []
        current.append(sample)
        previous_frame = frame
    if current:
        groups.append(current)
    return groups


def evaluate_waymo_crosswalk_group(
    group: Sequence[Dict[str, Any]],
    crosswalk: Dict[str, Any],
) -> Tuple[Optional[Dict[str, Any]], str]:
    settings = WAYMO_CROSSING_SETTINGS
    if len(group) < int(settings["minimum_observations"]):
        return None, "too_few_crosswalk_observations"
    axis = crosswalk_axis(crosswalk["polygon"])
    if axis is None:
        return None, "invalid_crosswalk_polygon"
    centre, major_axis, major_length, _ = axis
    times = np.asarray([float(sample["timestamp_seconds"]) for sample in group], dtype=float)
    times -= times[0]
    duration = float(times[-1])
    if duration < settings["minimum_duration_seconds"]:
        return None, "crosswalk_observation_too_short"
    # Waymo map positions are already metric labels. A centred rolling median
    # biases the first and last observations towards the interior and can
    # substantially attenuate short crossing speeds. The Huber line fit below
    # supplies the required outlier resistance without changing endpoints.
    x_values = np.asarray([float(sample["map_x"]) for sample in group], dtype=float)
    y_values = np.asarray([float(sample["map_y"]) for sample in group], dtype=float)
    positions = np.column_stack([x_values, y_values])
    along = (positions - centre) @ major_axis
    x_slope, _, _, _ = robust_line(times, x_values)
    y_slope, _, _, _ = robust_line(times, y_values)
    along_slope, _, along_r2, _ = robust_line(times, along)
    fitted_speed = float(math.hypot(x_slope, y_slope))
    crosswalk_axis_speed = float(abs(along_slope))
    displacement = float(np.linalg.norm(positions[-1] - positions[0]))
    progress = float(np.ptp(along) / max(major_length, 1e-9))
    direction_fraction = float(abs(along_slope) / max(fitted_speed, 1e-9))
    centre_margin = float(settings["centre_margin_fraction"] * major_length)
    crosses_centre = float(np.min(along)) <= -centre_margin and float(np.max(along)) >= centre_margin
    total_metadata_values = [
        safe_float(
            sample.get(
                "instantaneous_total_speed_mps",
                sample.get("instantaneous_speed_mps"),
            )
        )
        for sample in group
    ]
    lateral_metadata_values = [
        safe_float(sample.get("instantaneous_lateral_speed_mps"))
        for sample in group
    ]
    if any(value is None for value in total_metadata_values):
        return None, "missing_total_velocity_metadata"
    if any(value is None for value in lateral_metadata_values):
        return None, "missing_camera_lateral_velocity_metadata"
    metadata_total_speed = float(
        np.median(np.asarray(total_metadata_values, dtype=float))
    )
    metadata_lateral_speed = float(
        np.median(np.asarray(lateral_metadata_values, dtype=float))
    )
    metadata_error = float(abs(metadata_total_speed - fitted_speed))
    lateral_speed_fraction = float(
        metadata_lateral_speed / max(metadata_total_speed, 1e-9)
    )
    checks = [
        (displacement >= settings["minimum_displacement_m"], "insufficient_crossing_displacement"),
        (
            progress >= settings["minimum_crosswalk_progress_fraction"],
            "insufficient_crosswalk_progress",
        ),
        (crosses_centre, "trajectory_does_not_traverse_crosswalk_centre"),
        (
            direction_fraction >= settings["minimum_axis_direction_fraction"],
            "motion_not_aligned_with_crosswalk_axis",
        ),
        (along_r2 >= settings["minimum_combined_fit_r2"], "unstable_crossing_trajectory"),
        (
            settings["minimum_speed_mps"] <= fitted_speed <= settings["maximum_speed_mps"],
            "crossing_speed_outside_pedestrian_range",
        ),
        (
            settings["minimum_crosswalk_axis_speed_mps"]
            <= crosswalk_axis_speed
            <= settings["maximum_crosswalk_axis_speed_mps"],
            "crosswalk_axis_speed_outside_pedestrian_range",
        ),
        (
            metadata_error <= settings["maximum_metadata_speed_error_mps"],
            "trajectory_and_metadata_speed_disagree",
        ),
    ]
    for accepted, reason in checks:
        if not accepted:
            return None, reason
    result = {
        "crossing_track": 1,
        "crossing_label": "confirmed_crosswalk_traversal",
        "rejection_reason": "",
        "crosswalk_id": str(crosswalk["crosswalk_id"]),
        "crossing_start_frame": int(group[0]["frame-count"]),
        "crossing_end_frame": int(group[-1]["frame-count"]),
        "crossing_observations": len(group),
        "crossing_duration_seconds": duration,
        "crossing_displacement_m": displacement,
        "crosswalk_progress_fraction": progress,
        "axis_direction_fraction": direction_fraction,
        "trajectory_fit_r2": along_r2,
        # The scientific endpoint is walking speed along the crossing path.
        # It is therefore the robust map trajectory rate projected onto the
        # mapped crosswalk axis. Vehicle-frame lateral and total velocities
        # remain independent audit fields and are never used as the target.
        "ground_truth_speed_mps": crosswalk_axis_speed,
        "ground_truth_crosswalk_axis_speed_mps": crosswalk_axis_speed,
        "ground_truth_lateral_speed_mps": metadata_lateral_speed,
        "ground_truth_total_speed_mps": fitted_speed,
        "metadata_speed_mps": metadata_total_speed,
        "metadata_total_speed_mps": metadata_total_speed,
        "metadata_lateral_speed_mps": metadata_lateral_speed,
        "lateral_speed_fraction": lateral_speed_fraction,
        "metadata_speed_error_mps": metadata_error,
    }
    return result, ""


def classify_waymo_crossing_track(
    source_id: str,
    track_id: str,
    samples: Sequence[Dict[str, Any]],
    crosswalks: Sequence[Dict[str, Any]],
    camera_rows: int,
) -> Dict[str, Any]:
    base: Dict[str, Any] = {
        "source_id": source_id,
        "ground_truth_track_id": track_id,
        "camera_rows": camera_rows,
        "trajectory_rows": len(samples),
        "crossing_track": 0,
        "crossing_label": "not_confirmed",
        "rejection_reason": "",
        "crosswalk_id": "",
        "crossing_start_frame": "",
        "crossing_end_frame": "",
        "crossing_observations": 0,
        "crossing_duration_seconds": 0.0,
        "crossing_displacement_m": 0.0,
        "crosswalk_progress_fraction": 0.0,
        "axis_direction_fraction": 0.0,
        "trajectory_fit_r2": 0.0,
        "ground_truth_speed_mps": 0.0,
        "ground_truth_crosswalk_axis_speed_mps": 0.0,
        "ground_truth_lateral_speed_mps": 0.0,
        "ground_truth_total_speed_mps": 0.0,
        "metadata_speed_mps": 0.0,
        "metadata_total_speed_mps": 0.0,
        "metadata_lateral_speed_mps": 0.0,
        "lateral_speed_fraction": 0.0,
        "metadata_speed_error_mps": 0.0,
    }
    if camera_rows <= 0:
        base["rejection_reason"] = "not_visible_in_selected_camera"
        return base
    if len(samples) < int(WAYMO_CROSSING_SETTINGS["minimum_observations"]):
        base["rejection_reason"] = "insufficient_trajectory_rows"
        return base
    if not crosswalks:
        base["rejection_reason"] = "no_mapped_crosswalk_in_segment"
        return base
    accepted: List[Dict[str, Any]] = []
    rejected: List[Tuple[int, str]] = []
    tolerance = WAYMO_CROSSING_SETTINGS["crosswalk_boundary_tolerance_m"]
    for crosswalk in crosswalks:
        selected = [
            index
            for index, sample in enumerate(samples)
            if point_polygon_distance(
                (float(sample["map_x"]), float(sample["map_y"])),
                crosswalk["polygon"],
            )
            <= tolerance
        ]
        if not selected:
            continue
        groups = consecutive_waymo_groups(
            samples,
            selected,
            int(WAYMO_CROSSING_SETTINGS["maximum_gap_frames"]),
        )
        for group in groups:
            result, reason = evaluate_waymo_crosswalk_group(group, crosswalk)
            if result is not None:
                accepted.append(result)
            else:
                rejected.append((len(group), reason))
    if accepted:
        best = max(
            accepted,
            key=lambda row: (
                float(row["crosswalk_progress_fraction"]),
                float(row["crossing_duration_seconds"]),
                int(row["crossing_observations"]),
            ),
        )
        base.update(best)
        return base
    base["rejection_reason"] = (
        max(rejected, key=lambda item: item[0])[1]
        if rejected
        else "trajectory_does_not_intersect_mapped_crosswalk"
    )
    return base


def clean_track(rows: Sequence[BBoxRow]) -> List[BBoxRow]:
    best_by_frame: Dict[int, BBoxRow] = {}
    for row in rows:
        if not (-0.25 <= row.x <= 1.25 and -0.25 <= row.y <= 1.25):
            continue
        if not (0.001 <= row.width <= 1.50 and 0.001 <= row.height <= 1.50):
            continue
        current = best_by_frame.get(row.frame)
        if current is None or row.confidence > current.confidence:
            best_by_frame[row.frame] = row
    ordered = [best_by_frame[key] for key in sorted(best_by_frame)]
    if len(ordered) < 7:
        return ordered

    values = np.asarray([[row.x, row.y, row.width, row.height] for row in ordered], dtype=float)
    baseline = np.column_stack([rolling_median(values[:, col], 5) for col in range(4)])
    residual = values - baseline
    keep = np.ones(len(ordered), dtype=bool)
    for col in range(4):
        scale = robust_scale(residual[:, col])
        if scale > 1e-8:
            keep &= np.abs(residual[:, col]) <= 6.0 * scale
    # Never discard both endpoints merely because the track starts or ends abruptly.
    keep[0] = True
    keep[-1] = True
    return [row for index, row in enumerate(ordered) if bool(keep[index])]


def build_scene_motion_profile(
    rows: Sequence[BBoxRow],
    fps: float,
) -> SceneMotionProfile:
    if fps <= 0.0 or not math.isfinite(fps):
        fail(f"FPS must be positive, received {fps}")
    grouped: Dict[Tuple[int, str], List[BBoxRow]] = defaultdict(list)
    for row in rows:
        if row.class_id == PERSON_CLASS_ID:
            continue
        grouped[(row.class_id, row.track_id)].append(row)
    maximum_gap = int(SCENE_MOTION_SETTINGS["maximum_pair_gap_frames"])
    maximum_rate = float(
        SCENE_MOTION_SETTINGS["maximum_absolute_x_rate_per_second"]
    )
    samples_by_frame: Dict[int, List[Tuple[float, str]]] = defaultdict(list)
    for (class_id, track_id), track_rows in grouped.items():
        cleaned = clean_track(track_rows)
        reference_id = f"{class_id}:{track_id}"
        for first, second in zip(cleaned, cleaned[1:]):
            frame_gap = second.frame - first.frame
            if frame_gap < 1 or frame_gap > maximum_gap:
                continue
            delta_seconds = frame_gap / fps
            rate = (second.x - first.x) / delta_seconds
            if not math.isfinite(rate) or abs(rate) > maximum_rate:
                continue
            midpoint = int(round((first.frame + second.frame) / 2.0))
            samples_by_frame[midpoint].append((float(rate), reference_id))
    return SceneMotionProfile(samples_by_frame=dict(samples_by_frame))


def reversal_fraction(x_values: np.ndarray) -> float:
    if len(x_values) < 4:
        return 0.0
    delta = np.diff(x_values)
    noise = max(robust_scale(delta) * 0.25, 0.0005)
    signs = np.sign(delta[np.abs(delta) > noise])
    if len(signs) < 2:
        return 0.0
    return float(np.mean(signs[1:] != signs[:-1]))


def track_features(
    rows: Sequence[BBoxRow],
    fps: float,
    source_id: str,
    aspect_ratio: float = DEFAULT_ASPECT_RATIO,
    scene_motion_profile: Optional[SceneMotionProfile] = None,
) -> Optional[TrackFeatures]:
    if fps <= 0.0 or not math.isfinite(fps):
        fail(f"FPS must be positive, received {fps}")
    if aspect_ratio <= 0.0 or not math.isfinite(aspect_ratio):
        fail(f"Aspect ratio must be positive, received {aspect_ratio}")
    if not rows:
        return None
    cleaned = clean_track(rows)
    if len(cleaned) < 2:
        return None
    frames = np.asarray([row.frame for row in cleaned], dtype=float)
    x_raw = np.asarray([row.x for row in cleaned], dtype=float)
    y_raw = np.asarray([row.y for row in cleaned], dtype=float)
    width_raw = np.asarray([row.width for row in cleaned], dtype=float)
    height_raw = np.asarray([row.height for row in cleaned], dtype=float)
    confidence = np.asarray([row.confidence for row in cleaned], dtype=float)
    x = rolling_median(x_raw, 3)
    y = rolling_median(y_raw, 3)
    width = rolling_median(width_raw, 3)
    height = np.maximum(rolling_median(height_raw, 3), 0.001)
    time_values = (frames - frames[0]) / fps
    duration = float(time_values[-1])
    if duration <= 0.0:
        return None

    median_height = float(np.median(height))
    median_width = float(np.median(width))
    bottom = y + 0.50 * height
    q = aspect_ratio * (x - 0.50) / height
    log_height = np.log(height)
    x_slope, _, x_r2, _ = robust_line(time_values, x)
    q_slope, _, q_r2, q_mad = robust_line(time_values, q)
    log_height_slope, _, _, _ = robust_line(time_values, log_height)
    bottom_slope, _, _, _ = robust_line(time_values, bottom)

    delta_time = np.diff(time_values)
    delta_x = np.diff(x)
    valid_delta = delta_time > 0.0
    local_rates = aspect_ratio * delta_x[valid_delta] / delta_time[valid_delta] / max(median_height, 0.001)
    robust_rate = abs(float(np.median(local_rates))) if len(local_rates) else 0.0
    body_height_rate = aspect_ratio * abs(x_slope) / max(median_height, 0.001)
    q_rate = abs(q_slope)
    raw_proxy = DEFAULT_PERSON_HEIGHT_M * body_height_rate
    q_proxy = DEFAULT_PERSON_HEIGHT_M * q_rate
    robust_proxy = DEFAULT_PERSON_HEIGHT_M * robust_rate

    scene_rates: List[float] = []
    scene_support: List[int] = []
    for first_frame_value, second_frame_value in zip(frames, frames[1:]):
        midpoint = (first_frame_value + second_frame_value) / 2.0
        if scene_motion_profile is None:
            rate, support = 0.0, 0
        else:
            rate, support = scene_motion_profile.rate_at(midpoint)
        scene_rates.append(float(rate))
        scene_support.append(int(support))
    scene_rate_array = np.asarray(scene_rates, dtype=float)
    corrected_delta_x = delta_x - scene_rate_array * delta_time
    corrected_x = np.concatenate(
        [[x[0]], x[0] + np.cumsum(corrected_delta_x)]
    )
    corrected_x_slope, _, _, _ = robust_line(time_values, corrected_x)
    corrected_q = aspect_ratio * (corrected_x - 0.50) / height
    corrected_q_slope, _, _, _ = robust_line(time_values, corrected_q)
    corrected_local_rates = (
        aspect_ratio
        * corrected_delta_x[valid_delta]
        / delta_time[valid_delta]
        / max(median_height, 0.001)
    )
    corrected_robust_rate = (
        abs(float(np.median(corrected_local_rates)))
        if len(corrected_local_rates)
        else 0.0
    )
    compensated_raw_proxy = (
        DEFAULT_PERSON_HEIGHT_M
        * aspect_ratio
        * abs(float(corrected_x_slope))
        / max(median_height, 0.001)
    )
    compensated_q_proxy = DEFAULT_PERSON_HEIGHT_M * abs(float(corrected_q_slope))
    compensated_robust_proxy = DEFAULT_PERSON_HEIGHT_M * corrected_robust_rate
    supported_rates = [
        abs(rate)
        for rate, support in zip(scene_rates, scene_support)
        if support >= int(SCENE_MOTION_SETTINGS["minimum_reference_tracks"])
    ]
    scene_motion_rate_abs = (
        float(np.median(supported_rates)) if supported_rates else 0.0
    )
    median_scene_support = (
        float(np.median(scene_support)) if scene_support else 0.0
    )
    scene_motion_equivalent_speed = (
        DEFAULT_PERSON_HEIGHT_M
        * aspect_ratio
        * scene_motion_rate_abs
        / max(median_height, 0.001)
    )
    scene_motion_fraction = scene_motion_equivalent_speed / max(
        raw_proxy + scene_motion_equivalent_speed,
        1e-9,
    )
    compensated_proxy_disagreement = float(
        max(
            compensated_raw_proxy,
            compensated_q_proxy,
            compensated_robust_proxy,
        )
        - min(
            compensated_raw_proxy,
            compensated_q_proxy,
            compensated_robust_proxy,
        )
    )

    first_frame = int(frames[0])
    last_frame = int(frames[-1])
    expected_rows = max(1, last_frame - first_frame + 1)
    coverage = min(1.0, len(cleaned) / expected_rows)
    left = x - 0.50 * width
    right = x + 0.50 * width
    top = y - 0.50 * height
    lower = y + 0.50 * height
    edge = (left <= 0.01) | (right >= 0.99)
    truncated = edge | (top <= 0.01) | (lower >= 0.99)
    height_low = max(float(np.quantile(height, 0.10)), 0.001)
    height_high = float(np.quantile(height, 0.90))
    horizontal_range = float(np.max(x) - np.min(x))
    direction = "left_to_right" if x_slope > 0.0 else "right_to_left" if x_slope < 0.0 else "stationary"

    return TrackFeatures(
        source_id=str(source_id),
        prediction_track_id=cleaned[0].track_id,
        input_rows=len(rows),
        clean_rows=len(cleaned),
        removed_rows=len(rows) - len(cleaned),
        first_frame=first_frame,
        last_frame=last_frame,
        duration_seconds=duration,
        coverage=float(coverage),
        gap_ratio=float(1.0 - coverage),
        median_confidence=float(np.median(confidence)),
        median_height=median_height,
        median_width=median_width,
        height_ratio=float(height_high / height_low),
        horizontal_range=horizontal_range,
        vertical_range=float(np.max(y) - np.min(y)),
        crosses_image_midline=bool(float(np.min(x)) <= 0.50 <= float(np.max(x))),
        overlaps_central_band=bool(float(np.min(x)) <= 0.60 and float(np.max(x)) >= 0.40),
        direction=direction,
        x_slope_per_second=float(x_slope),
        x_fit_r2=float(x_r2),
        q_slope_per_second=float(q_slope),
        q_fit_r2=float(q_r2),
        q_residual_mad=float(q_mad),
        log_height_rate_abs=abs(float(log_height_slope)),
        bottom_rate_abs=abs(float(bottom_slope)),
        edge_fraction=float(np.mean(edge)),
        truncation_fraction=float(np.mean(truncated)),
        reversal_fraction=reversal_fraction(x),
        raw_speed_proxy_mps=float(raw_proxy),
        q_speed_proxy_mps=float(q_proxy),
        robust_speed_proxy_mps=float(robust_proxy),
        compensated_raw_speed_proxy_mps=float(compensated_raw_proxy),
        compensated_q_speed_proxy_mps=float(compensated_q_proxy),
        compensated_robust_speed_proxy_mps=float(compensated_robust_proxy),
        scene_motion_rate_abs=float(scene_motion_rate_abs),
        scene_motion_equivalent_speed_mps=float(scene_motion_equivalent_speed),
        scene_motion_fraction=float(scene_motion_fraction),
        scene_motion_support=float(median_scene_support),
        log_scene_motion_support=float(math.log1p(median_scene_support)),
        compensated_proxy_disagreement_mps=compensated_proxy_disagreement,
        one_minus_x_r2=float(max(0.0, 1.0 - x_r2)),
        log_height_ratio=float(math.log(max(height_high / height_low, 1.0))),
        log_duration=float(math.log1p(duration)),
        log_median_height=float(math.log(max(median_height, 1e-6))),
    )


def base_rejection_reason(features: TrackFeatures) -> str:
    gates = BASE_GATES
    checks = [
        (features.clean_rows < int(gates["minimum_rows"]), "too_few_rows"),
        (features.duration_seconds < gates["minimum_duration_seconds"], "duration_too_short"),
        (features.coverage < gates["minimum_coverage"], "track_too_fragmented"),
        (features.median_height < gates["minimum_median_height"], "box_too_small"),
        (features.horizontal_range < gates["minimum_horizontal_range"], "insufficient_lateral_motion"),
        (features.x_fit_r2 < gates["minimum_x_fit_r2"], "nonlinear_or_unstable_motion"),
        (features.height_ratio > gates["maximum_height_ratio"], "excessive_scale_change"),
        (features.edge_fraction > gates["maximum_edge_fraction"], "track_truncated_at_image_edge"),
        (features.reversal_fraction > gates["maximum_reversal_fraction"], "too_many_direction_reversals"),
        (
            features.scene_motion_support
            < gates["minimum_scene_motion_support"],
            "insufficient_scene_motion_references",
        ),
    ]
    for rejected, reason in checks:
        if rejected:
            return reason
    return ""


def reliability_gate_reason(features: TrackFeatures) -> str:
    checks = [
        (
            features.duration_seconds
            < RELIABILITY_GATES["minimum_duration_seconds"],
            "duration_below_reliable_minimum",
        ),
        (
            features.x_fit_r2 < RELIABILITY_GATES["minimum_x_fit_r2"],
            "horizontal_fit_below_reliable_minimum",
        ),
    ]
    for rejected, reason in checks:
        if rejected:
            return reason
    return ""


def reliability_gate_reason_from_row(row: Dict[str, Any]) -> str:
    duration = safe_float(row.get("duration_seconds"))
    x_fit_r2 = safe_float(row.get("x_fit_r2"))
    if duration is None or duration < RELIABILITY_GATES["minimum_duration_seconds"]:
        return "duration_below_reliable_minimum"
    if x_fit_r2 is None or x_fit_r2 < RELIABILITY_GATES["minimum_x_fit_r2"]:
        return "horizontal_fit_below_reliable_minimum"
    return ""


def apply_source_context(
    features_by_track: Dict[str, TrackFeatures],
) -> Dict[str, TrackFeatures]:
    """Add label free within video context to every pedestrian track.

    The reference population is formed only from tracks that pass the locked
    base motion checks.  It deliberately uses no crossing label, Waymo track,
    or Waymo speed.  Therefore calibration and CROWD prediction construct the
    context from the same input: all pedestrian tracks in one bbox CSV.
    """
    reference = [
        features
        for features in features_by_track.values()
        if base_rejection_reason(features) == ""
    ]
    reference_count = len(reference)
    minimum_count = int(SOURCE_CONTEXT_SETTINGS["minimum_reference_tracks"])
    context_available = reference_count >= minimum_count

    for features in features_by_track.values():
        features.source_context_tracks = float(reference_count)
        features.source_context_available = 1.0 if context_available else 0.0
        features.source_relative_log_raw_proxy = 0.0
        features.source_relative_log_q_proxy = 0.0
        features.source_relative_log_robust_proxy = 0.0
        features.source_relative_log_compensated_robust_proxy = 0.0
        features.source_compensated_robust_percentile = 0.5

    if not context_available:
        return features_by_track

    minimum_proxy = float(SOURCE_CONTEXT_SETTINGS["minimum_proxy_mps"])
    maximum_log_ratio = float(
        SOURCE_CONTEXT_SETTINGS["maximum_absolute_log_ratio"]
    )
    proxy_fields = {
        "source_relative_log_raw_proxy": "raw_speed_proxy_mps",
        "source_relative_log_q_proxy": "q_speed_proxy_mps",
        "source_relative_log_robust_proxy": "robust_speed_proxy_mps",
        "source_relative_log_compensated_robust_proxy": (
            "compensated_robust_speed_proxy_mps"
        ),
    }
    medians = {
        output_field: float(
            np.median(
                [
                    max(float(getattr(features, input_field)), minimum_proxy)
                    for features in reference
                ]
            )
        )
        for output_field, input_field in proxy_fields.items()
    }
    rank_values = np.asarray(
        [
            max(
                float(features.compensated_robust_speed_proxy_mps),
                minimum_proxy,
            )
            for features in reference
        ],
        dtype=float,
    )

    for features in features_by_track.values():
        for output_field, input_field in proxy_fields.items():
            value = max(float(getattr(features, input_field)), minimum_proxy)
            log_ratio = math.log(value / max(medians[output_field], minimum_proxy))
            setattr(
                features,
                output_field,
                float(np.clip(log_ratio, -maximum_log_ratio, maximum_log_ratio)),
            )
        rank_value = max(
            float(features.compensated_robust_speed_proxy_mps),
            minimum_proxy,
        )
        less = float(np.sum(rank_values < rank_value))
        equal = float(np.sum(rank_values == rank_value))
        features.source_compensated_robust_percentile = float(
            (less + 0.5 * equal) / max(len(rank_values), 1)
        )
    return features_by_track


def contextual_track_features(
    tracks: Dict[str, List[BBoxRow]],
    fps: float,
    source_id: str,
    aspect_ratio: float,
    scene_motion_profile: SceneMotionProfile,
) -> Dict[str, TrackFeatures]:
    features_by_track: Dict[str, TrackFeatures] = {}
    for track_id in sorted(tracks, key=str):
        features = track_features(
            tracks[track_id],
            fps,
            source_id,
            aspect_ratio,
            scene_motion_profile,
        )
        if features is not None:
            features_by_track[track_id] = features
    return apply_source_context(features_by_track)


def features_to_row(features: TrackFeatures) -> Dict[str, Any]:
    row = asdict(features)
    row["base_status"] = "rejected" if base_rejection_reason(features) else "eligible"
    row["base_reject_reason"] = base_rejection_reason(features)
    row["reliability_status"] = (
        "rejected" if reliability_gate_reason(features) else "eligible"
    )
    row["reliability_reject_reason"] = reliability_gate_reason(features)
    return row


def mode_features(bbox_csv: str, fps: float, output_csv: str, source_id: str, aspect_ratio: float) -> None:
    all_rows = load_bbox_csv(bbox_csv, person_only=False)
    person_rows = [row for row in all_rows if row.class_id == PERSON_CLASS_ID]
    tracks = group_tracks(person_rows)
    scene_motion_profile = build_scene_motion_profile(all_rows, fps)
    features_by_track = contextual_track_features(
        tracks,
        fps,
        source_id,
        aspect_ratio,
        scene_motion_profile,
    )
    feature_rows = [
        features_to_row(features_by_track[track_id])
        for track_id in sorted(features_by_track, key=str)
    ]
    write_dict_csv(output_csv, feature_rows)
    log(f"Person tracks: {len(tracks)}")
    log(f"Feature rows written: {len(feature_rows)}")
    log(f"Output: {Path(output_csv).expanduser().resolve()}")


def bbox_xyxy(row: BBoxRow) -> Tuple[float, float, float, float]:
    return (
        row.x - row.width / 2.0,
        row.y - row.height / 2.0,
        row.x + row.width / 2.0,
        row.y + row.height / 2.0,
    )


def iou(a: BBoxRow, b: BBoxRow) -> float:
    ax1, ay1, ax2, ay2 = bbox_xyxy(a)
    bx1, by1, bx2, by2 = bbox_xyxy(b)
    intersection_width = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    intersection_height = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = intersection_width * intersection_height
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0.0 else 0.0


def load_ground_truth_bbox_csv(
    path: str,
) -> Tuple[List[BBoxRow], Dict[Tuple[int, str], float], str]:
    input_path = Path(path).expanduser().resolve()
    if not input_path.is_file():
        fail(f"Ground truth bounding box CSV does not exist: {input_path}")
    with input_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            fail(f"Ground truth CSV has no header: {input_path}")
        names = reader.fieldnames
        track_col = first_column(names, ["ground_truth_track_id", "track_id", "unique-id", "id"])
        frame_col = first_column(names, ["frame-count", "frame_count", "frame", "frame_id"])
        x_col = first_column(names, ["x-center", "x_center", "xcenter"])
        y_col = first_column(names, ["y-center", "y_center", "ycenter"])
        width_col = first_column(names, ["width", "bbox_width"])
        height_col = first_column(names, ["height", "bbox_height"])
        crosswalk_axis_speed_col = first_column(
            names,
            ["ground_truth_crosswalk_axis_speed_mps"],
            required=False,
        )
        speed_col = crosswalk_axis_speed_col or first_column(
            names,
            ["ground_truth_speed_mps", "gt_speed_mps", "speed_mps", "speed"],
        )
        calibration_target = (
            WAYMO_CALIBRATION_TARGET
            if crosswalk_axis_speed_col is not None
            else "legacy_or_non_crosswalk_axis_ground_truth_speed_mps"
        )
        crossing_col = first_column(
            names,
            ["crossing_track", "confirmed_crossing", "crossing_include"],
            required=False,
        )
        raw_rows = list(reader)
        crossing_labels_present = crossing_col is not None and any(
            str(raw.get(crossing_col, "")).strip() for raw in raw_rows
        )
        output: List[BBoxRow] = []
        speeds: Dict[Tuple[int, str], float] = {}
        for raw in raw_rows:
            if crossing_labels_present and not truthy(raw.get(crossing_col), default=False):
                continue
            track_id = normalise_id(raw.get(track_col, ""))
            frame = safe_int(raw.get(frame_col))
            x = safe_float(raw.get(x_col))
            y = safe_float(raw.get(y_col))
            width = safe_float(raw.get(width_col))
            height = safe_float(raw.get(height_col))
            speed = safe_float(raw.get(speed_col))
            if not track_id or None in (frame, x, y, width, height, speed):
                continue
            if width <= 0.0 or height <= 0.0 or speed < 0.0:
                continue
            row = BBoxRow(PERSON_CLASS_ID, x, y, width, height, track_id, 1.0, frame)
            output.append(row)
            speeds[(frame, track_id)] = speed
    return output, speeds, calibration_target


def matching_statistics(
    predictions: Sequence[BBoxRow],
    ground_truth: Sequence[BBoxRow],
    frame_offset: int,
) -> Tuple[Dict[Tuple[str, str], Dict[str, Any]], float]:
    gt_by_frame: Dict[int, List[BBoxRow]] = defaultdict(list)
    for row in ground_truth:
        gt_by_frame[row.frame].append(row)
    stats: Dict[Tuple[str, str], Dict[str, Any]] = {}
    total_best_iou = 0.0
    for pred in predictions:
        candidates = gt_by_frame.get(pred.frame + frame_offset, [])
        for gt in candidates:
            overlap = iou(pred, gt)
            if overlap < 0.10:
                continue
            key = (pred.track_id, gt.track_id)
            item = stats.setdefault(key, {"frames": set(), "ious": []})
            item["frames"].add((pred.frame, gt.frame))
            item["ious"].append(overlap)
            total_best_iou += overlap
    return stats, total_best_iou


def calibration_match_rejection_reason(candidate: Dict[str, Any]) -> str:
    settings = CALIBRATION_MATCH_SETTINGS
    checks = [
        (
            int(candidate["match_frames"]) < int(settings["minimum_matched_frames"]),
            "association_too_few_matched_frames",
        ),
        (
            float(candidate["mean_iou"]) < settings["minimum_mean_iou"],
            "association_mean_iou_too_low",
        ),
        (
            float(candidate["prediction_coverage"]) < settings["minimum_prediction_coverage"],
            "association_prediction_coverage_too_low",
        ),
        (
            float(candidate["ground_truth_coverage"]) < settings["minimum_ground_truth_coverage"],
            "association_ground_truth_coverage_too_low",
        ),
    ]
    for rejected, reason in checks:
        if rejected:
            return reason
    return ""


def mode_match(
    source_id: str,
    prediction_bbox_csv: str,
    ground_truth_bbox_csv: str,
    fps: float,
    output_manifest_csv: str,
    split: str,
    aspect_ratio: float,
) -> None:
    all_prediction_rows = load_bbox_csv(prediction_bbox_csv, person_only=False)
    predictions = [
        row for row in all_prediction_rows if row.class_id == PERSON_CLASS_ID
    ]
    scene_motion_profile = build_scene_motion_profile(all_prediction_rows, fps)
    ground_truth, gt_speeds, calibration_target = load_ground_truth_bbox_csv(
        ground_truth_bbox_csv
    )
    if not predictions:
        fail("Prediction CSV contains no tracked person rows (class 0)")
    if not ground_truth:
        fail("Ground truth CSV contains no usable pedestrian rows")

    offset_candidates = []
    for offset in (-1, 0, 1):
        stats, score = matching_statistics(predictions, ground_truth, offset)
        offset_candidates.append((score, -abs(offset), offset, stats))
    _, _, frame_offset, stats = max(offset_candidates, key=lambda item: (item[0], item[1]))
    prediction_tracks = group_tracks(predictions)
    ground_truth_tracks = group_tracks(ground_truth)
    candidates: List[Dict[str, Any]] = []
    for (prediction_id, ground_truth_id), item in stats.items():
        match_frames = len(item["frames"])
        mean_iou = float(np.mean(item["ious"]))
        prediction_coverage = match_frames / max(1, len(prediction_tracks.get(prediction_id, [])))
        ground_truth_coverage = match_frames / max(1, len(ground_truth_tracks.get(ground_truth_id, [])))
        score = match_frames * mean_iou * math.sqrt(max(prediction_coverage * ground_truth_coverage, 1e-9))
        candidates.append(
            {
                "prediction_id": prediction_id,
                "ground_truth_id": ground_truth_id,
                "match_frames": match_frames,
                "mean_iou": mean_iou,
                "prediction_coverage": prediction_coverage,
                "ground_truth_coverage": ground_truth_coverage,
                "score": score,
                "frame_pairs": item["frames"],
            }
        )
    candidates.sort(key=lambda item: item["score"], reverse=True)
    used_predictions = set()
    used_ground_truth = set()
    manifest_rows: List[Dict[str, Any]] = []
    for candidate in candidates:
        prediction_id = candidate["prediction_id"]
        ground_truth_id = candidate["ground_truth_id"]
        if prediction_id in used_predictions or ground_truth_id in used_ground_truth:
            continue
        if candidate["match_frames"] < 3 or candidate["mean_iou"] < 0.20 or candidate["prediction_coverage"] < 0.20:
            continue
        used_predictions.add(prediction_id)
        used_ground_truth.add(ground_truth_id)
        matched_speeds = [
            gt_speeds[(gt_frame, ground_truth_id)]
            for _, gt_frame in candidate["frame_pairs"]
            if (gt_frame, ground_truth_id) in gt_speeds
        ]
        if not matched_speeds:
            continue
        gt_speed = float(np.median(matched_speeds))
        features = track_features(
            prediction_tracks[prediction_id],
            fps,
            source_id,
            aspect_ratio,
            scene_motion_profile,
        )
        reason = "feature_extraction_failed" if features is None else base_rejection_reason(features)
        if not reason:
            reason = calibration_match_rejection_reason(candidate)
        include = features is not None and reason == "" and 0.10 <= gt_speed <= 3.50
        if not (0.10 <= gt_speed <= 3.50) and not reason:
            reason = "ground_truth_speed_outside_pedestrian_range"
        manifest_rows.append(
            {
                "source_id": source_id,
                "bbox_csv": str(Path(prediction_bbox_csv).expanduser().resolve()),
                "fps": fps,
                "aspect_ratio": aspect_ratio,
                "prediction_track_id": prediction_id,
                "ground_truth_track_id": ground_truth_id,
                "ground_truth_speed_mps": gt_speed,
                "calibration_target": calibration_target,
                "split": split,
                "include": 1 if include else 0,
                "exclusion_reason": reason,
                "matched_frames": candidate["match_frames"],
                "mean_iou": candidate["mean_iou"],
                "prediction_match_coverage": candidate["prediction_coverage"],
                "ground_truth_match_coverage": candidate["ground_truth_coverage"],
                "frame_offset": frame_offset,
            }
        )
    manifest_rows.sort(key=lambda row: str(row["prediction_track_id"]))
    write_dict_csv(output_manifest_csv, manifest_rows, MANIFEST_FIELDS)
    log(f"Automatic frame offset: {frame_offset}")
    log(f"Matched one-to-one tracks: {len(manifest_rows)}")
    log(f"Eligible calibration tracks: {sum(int(row['include']) for row in manifest_rows)}")
    log(f"Manifest: {Path(output_manifest_csv).expanduser().resolve()}")


def mode_merge(output_csv: str, input_csvs: Sequence[str]) -> None:
    if not input_csvs:
        fail("merge requires at least one input manifest")
    rows: List[Dict[str, Any]] = []
    for input_csv in input_csvs:
        rows.extend(read_dict_csv(input_csv))
    write_dict_csv(output_csv, rows)
    log(f"Merged rows: {len(rows)}")
    log(f"Output: {Path(output_csv).expanduser().resolve()}")


def stable_hash(value: str) -> int:
    digest = hashlib.sha256(f"{DEFAULT_RANDOM_SEED}:{value}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def source_subset_near_target(
    candidates: Sequence[str],
    weights: Dict[str, int],
    target: float,
    minimum_sources: int,
    maximum_sources: int,
    minimum_tracks: int = 0,
) -> List[str]:
    """Choose a deterministic source subset whose eligible count is near target."""
    if maximum_sources < 1 or not candidates:
        return []
    states: Dict[Tuple[int, int], Tuple[str, ...]] = {(0, 0): ()}
    for source in candidates:
        weight = int(weights[source])
        additions: Dict[Tuple[int, int], Tuple[str, ...]] = {}
        for (total, count), subset in list(states.items()):
            if len(subset) >= maximum_sources:
                continue
            new_total = total + weight
            new_subset = subset + (source,)
            new_key = (new_total, count + 1)
            current = states.get(new_key) or additions.get(new_key)
            if current is None or tuple(stable_hash(item) for item in new_subset) < tuple(
                stable_hash(item) for item in current
            ):
                additions[new_key] = new_subset
        states.update(additions)
    choices = [
        (total, subset)
        for (total, count), subset in states.items()
        if count >= minimum_sources
        and count <= maximum_sources
        and total >= minimum_tracks
    ]
    if not choices:
        return []
    _, selected = min(
        choices,
        key=lambda item: (
            abs(float(item[0]) - target),
            len(item[1]),
            tuple(stable_hash(source) for source in item[1]),
        ),
    )
    return list(selected)


def assign_group_splits(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    output = [dict(row) for row in rows]
    sources = sorted(
        {
            str(row.get("source_id", "")).strip()
            for row in output
            if str(row.get("source_id", "")).strip()
        },
        key=stable_hash,
    )
    eligible_counts: Dict[str, int] = defaultdict(int)
    for row in output:
        source = str(row.get("source_id", "")).strip()
        if source and truthy(row.get("include"), True):
            eligible_counts[source] += 1
    eligible_sources = [source for source in sources if eligible_counts[source] > 0]
    minimum_training_sources = int(GROUP_SPLIT_SETTINGS["minimum_training_sources"])
    minimum_validation_sources = int(GROUP_SPLIT_SETTINGS["minimum_validation_sources"])
    minimum_test_sources = int(GROUP_SPLIT_SETTINGS["minimum_test_sources"])
    minimum_eligible_sources = (
        minimum_training_sources + minimum_validation_sources + minimum_test_sources
    )
    if len(eligible_sources) < minimum_eligible_sources:
        fail(
            f"At least {minimum_eligible_sources} independent source_id values with eligible "
            "tracks are required for three training sources plus validation and test"
        )
    total_eligible = sum(eligible_counts.values())
    ranked = sorted(
        eligible_sources,
        key=lambda source: (-eligible_counts[source], stable_hash(source)),
    )
    # Keep the largest source in training.  Otherwise one dense scene could
    # consume an entire held-out split and leave too few tracks to fit.
    training_anchor = ranked[0]
    available = ranked[1:]
    validation_target = total_eligible * GROUP_SPLIT_SETTINGS["validation_fraction"]
    validation_sources = source_subset_near_target(
        available,
        eligible_counts,
        validation_target,
        minimum_validation_sources,
        len(available) - (minimum_training_sources - 1) - minimum_test_sources,
    )
    remaining = [source for source in available if source not in set(validation_sources)]
    minimum_test_tracks = int(GROUP_SPLIT_SETTINGS["minimum_test_tracks"])
    test_target = max(
        total_eligible * GROUP_SPLIT_SETTINGS["test_fraction"],
        float(minimum_test_tracks),
    )
    test_sources = source_subset_near_target(
        remaining,
        eligible_counts,
        test_target,
        minimum_test_sources,
        len(remaining) - (minimum_training_sources - 1),
        minimum_test_tracks,
    )
    if not validation_sources or not test_sources:
        fail("Could not construct non-empty source-grouped validation and test splits")
    source_split: Dict[str, str] = {source: "train" for source in sources}
    source_split[training_anchor] = "train"
    for source in validation_sources:
        source_split[source] = "validation"
    for source in test_sources:
        source_split[source] = "test"
    for row in output:
        source = str(row.get("source_id", "")).strip()
        row["split"] = source_split.get(source, "")
    return output


def mode_assign_splits(input_manifest: str, output_manifest: str) -> None:
    rows = assign_group_splits(read_dict_csv(input_manifest))
    write_dict_csv(output_manifest, rows)
    source_counts: Dict[str, set] = defaultdict(set)
    eligible_source_counts: Dict[str, set] = defaultdict(set)
    eligible_counts: Dict[str, int] = defaultdict(int)
    for row in rows:
        split = str(row.get("split", ""))
        source_counts[split].add(str(row.get("source_id", "")))
        if truthy(row.get("include"), True):
            eligible_counts[split] += 1
            eligible_source_counts[split].add(str(row.get("source_id", "")))
    log(
        "Source split counts: "
        + ", ".join(
            f"{key}={len(value)}" for key, value in sorted(source_counts.items())
        )
    )
    log(
        "Eligible track split counts: "
        + ", ".join(f"{key}={value}" for key, value in sorted(eligible_counts.items()))
    )
    log(
        "Eligible source split counts: "
        + ", ".join(
            f"{key}={len(value)}"
            for key, value in sorted(eligible_source_counts.items())
        )
    )
    minimum_test_tracks = int(GROUP_SPLIT_SETTINGS["minimum_test_tracks"])
    minimum_test_sources = int(GROUP_SPLIT_SETTINGS["minimum_test_sources"])
    if eligible_counts.get("test", 0) < minimum_test_tracks:
        fail(
            f"The source-grouped test split requires at least {minimum_test_tracks} "
            f"eligible tracks; found {eligible_counts.get('test', 0)}"
        )
    if len(eligible_source_counts.get("test", set())) < minimum_test_sources:
        fail(
            f"The source-grouped test split requires at least {minimum_test_sources} "
            "independent eligible sources"
        )
    log(f"Output: {Path(output_manifest).expanduser().resolve()}")


def mode_make_development_manifest(
    input_manifest: str,
    output_manifest: str,
) -> None:
    rows = read_dict_csv(input_manifest)
    if not rows:
        fail("Development manifest input is empty")
    sources: set = set()
    eligible_sources: set = set()
    eligible_tracks = 0
    for row in rows:
        row["split"] = "train"
        source = str(row.get("source_id", "")).strip()
        if source:
            sources.add(source)
        if source and truthy(row.get("include"), True):
            eligible_sources.add(source)
            eligible_tracks += 1
    write_dict_csv(output_manifest, rows)
    log(
        f"Development manifest: {eligible_tracks} eligible tracks from "
        f"{len(eligible_sources)} eligible sources; {len(sources)} total sources"
    )
    log(
        "All supplied sources are now development sources and must not be used "
        "again as an untouched external test"
    )
    log(f"Output: {Path(output_manifest).expanduser().resolve()}")


def mode_make_external_test_manifest(
    full_manifest: str,
    development_manifest: str,
    output_manifest: str,
) -> None:
    full_rows = read_dict_csv(full_manifest)
    development_rows = read_dict_csv(development_manifest)
    known_sources = {
        str(row.get("source_id", "")).strip()
        for row in development_rows
        if str(row.get("source_id", "")).strip()
    }
    if not known_sources:
        fail("Development manifest contains no source_id values")
    output: List[Dict[str, Any]] = []
    eligible_sources: set = set()
    eligible_tracks = 0
    for raw in full_rows:
        source = str(raw.get("source_id", "")).strip()
        if not source or source in known_sources:
            continue
        row = dict(raw)
        row["split"] = "test"
        output.append(row)
        if truthy(row.get("include"), True):
            eligible_sources.add(source)
            eligible_tracks += 1
    if not output:
        fail("No sources remain after excluding every development source")
    write_dict_csv(output_manifest, output)
    log(
        f"Untouched external test manifest: {eligible_tracks} eligible tracks "
        f"from {len(eligible_sources)} eligible sources"
    )
    minimum_tracks = int(EXTERNAL_TEST_SETTINGS["minimum_test_tracks"])
    minimum_sources = int(EXTERNAL_TEST_SETTINGS["minimum_test_sources"])
    if eligible_tracks < minimum_tracks or len(eligible_sources) < minimum_sources:
        fail(
            f"External test requires at least {minimum_tracks} eligible tracks "
            f"from {minimum_sources} independent sources"
        )
    log(f"Output: {Path(output_manifest).expanduser().resolve()}")


def resolve_manifest_path(manifest_path: str, value: str) -> str:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = Path(manifest_path).expanduser().resolve().parent / candidate
    return str(candidate.resolve())


def portable_index_path(path: Path, index_root: Optional[Path] = None) -> str:
    """Write a relocatable index path whenever a common index root is known."""
    resolved = path.expanduser().resolve()
    if index_root is not None:
        try:
            return str(resolved.relative_to(index_root.expanduser().resolve()))
        except ValueError:
            pass
    try:
        return str(resolved.relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(resolved)


def resolve_index_path(index_csv: str, value: str) -> str:
    """Resolve current and relocated Waymo index paths."""
    text = str(value).strip()
    if not text:
        return ""
    candidate = Path(text).expanduser()
    index_directory = Path(index_csv).expanduser().resolve().parent

    # Older Waymo indices may still mention the former repository output path
    # or Docker's /workspace mount.  Preserve the path suffix below the split
    # directory so moving the whole waymo_processed folder remains valid.
    split_name = index_directory.name
    candidate_parts = candidate.parts
    if split_name in candidate_parts:
        split_position = len(candidate_parts) - 1 - list(
            reversed(candidate_parts)
        ).index(split_name)
        relocated = index_directory.joinpath(
            *candidate_parts[split_position + 1 :]
        )
        if relocated.exists() or relocated.parent.exists():
            return str(relocated.resolve())

    # Pre-move indices used paths such as
    # data/speed_calibration/waymo/<source>/<file>.  Recover the stable source
    # directory and file name under the directory containing the current
    # training or validation index.
    if len(candidate_parts) >= 2:
        relocated = index_directory / candidate_parts[-2] / candidate_parts[-1]
        if relocated.exists() or relocated.parent.exists():
            return str(relocated.resolve())

    if candidate.is_absolute():
        try:
            relative = candidate.relative_to("/workspace")
        except ValueError:
            return str(candidate)
        return str((Path.cwd().resolve() / relative).resolve())

    working_directory_candidate = (Path.cwd().resolve() / candidate).resolve()
    if working_directory_candidate.exists():
        return str(working_directory_candidate)
    index_directory_candidate = (index_directory / candidate).resolve()
    if index_directory_candidate.exists():
        return str(index_directory_candidate)
    return str(working_directory_candidate)


def load_manifest_feature_rows(manifest_path: str) -> List[Dict[str, Any]]:
    manifest_rows = read_dict_csv(manifest_path)
    if not manifest_rows:
        fail("Calibration manifest is empty")
    required = [
        "source_id",
        "bbox_csv",
        "fps",
        "prediction_track_id",
        "ground_truth_speed_mps",
        "calibration_target",
    ]
    missing = [name for name in required if canonical_header(name) not in {canonical_header(key) for key in manifest_rows[0]}]
    if missing:
        fail("Manifest is missing columns: " + ", ".join(missing))
    incompatible_targets = sorted(
        {
            str(row.get("calibration_target", "")).strip() or "<missing>"
            for row in manifest_rows
            if truthy(row.get("include"), True)
            and str(row.get("calibration_target", "")).strip()
            != WAYMO_CALIBRATION_TARGET
        }
    )
    if incompatible_targets:
        fail(
            "Manifest contains an incompatible legacy calibration target. "
            "Regenerate it with calibrate_waymo_pipeline. Found: "
            + ", ".join(incompatible_targets)
        )
    feature_cache: Dict[str, Dict[str, TrackFeatures]] = {}
    output: List[Dict[str, Any]] = []
    for raw in manifest_rows:
        if not truthy(raw.get("include"), True):
            continue
        source_id = str(raw.get("source_id", "")).strip()
        bbox_path = resolve_manifest_path(manifest_path, str(raw.get("bbox_csv", "")))
        fps = safe_float(raw.get("fps"))
        aspect_ratio = safe_float(raw.get("aspect_ratio")) or DEFAULT_ASPECT_RATIO
        track_id = normalise_id(raw.get("prediction_track_id", ""))
        gt_speed = safe_float(raw.get("ground_truth_speed_mps"))
        split = str(raw.get("split", "")).strip().lower()
        if not source_id or fps is None or fps <= 0.0 or not track_id or gt_speed is None:
            continue
        if bbox_path not in feature_cache:
            all_rows = load_bbox_csv(bbox_path, person_only=False)
            person_rows = [
                row for row in all_rows if row.class_id == PERSON_CLASS_ID
            ]
            tracks = group_tracks(person_rows)
            scene_motion_profile = build_scene_motion_profile(all_rows, fps)
            feature_cache[bbox_path] = contextual_track_features(
                tracks,
                fps,
                source_id,
                aspect_ratio,
                scene_motion_profile,
            )
        features = feature_cache[bbox_path].get(track_id)
        if features is None:
            log(f"WARNING: track {track_id} not found in {bbox_path}; skipping")
            continue
        output.append(
            {
                "source_id": source_id,
                "bbox_csv": bbox_path,
                "fps": fps,
                "aspect_ratio": aspect_ratio,
                "prediction_track_id": track_id,
                "ground_truth_track_id": str(raw.get("ground_truth_track_id", "")),
                "ground_truth_speed_mps": gt_speed,
                "calibration_target": str(raw.get("calibration_target", "")),
                "split": split,
                "features": features,
            }
        )
    if not output:
        fail("No usable included manifest rows were found")
    if not all(row["split"] in {"train", "validation", "test"} for row in output):
        source_to_split = {row["source_id"]: row["split"] for row in assign_group_splits(output)}
        for row in output:
            if row["split"] not in {"train", "validation", "test"}:
                row["split"] = source_to_split[row["source_id"]]
    return output


def robust_standardisation(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    centre = np.median(matrix, axis=0)
    scale = np.asarray([robust_scale(matrix[:, index]) for index in range(matrix.shape[1])], dtype=float)
    conventional = np.std(matrix, axis=0)
    scale = np.where(scale > 1e-8, scale, np.where(conventional > 1e-8, conventional, 1.0))
    return centre, scale


def constrained_huber_ridge(
    design: np.ndarray,
    target: np.ndarray,
    ridge: float = 1.0,
    base_weights: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    beta = np.linalg.lstsq(design, target, rcond=None)[0]
    source_weights = (
        np.asarray(base_weights, dtype=float)
        if base_weights is not None
        else np.ones(len(target), dtype=float)
    )
    source_weights = source_weights / max(float(np.mean(source_weights)), 1e-12)
    weights = source_weights.copy()
    penalty = np.eye(design.shape[1], dtype=float) * ridge
    penalty[0, 0] = 0.0
    for _ in range(50):
        residual = target - design @ beta
        scale = max(robust_scale(residual), 1e-6)
        threshold = 1.345 * scale
        huber_weights = np.ones(len(target), dtype=float)
        large = np.abs(residual) > threshold
        huber_weights[large] = threshold / np.maximum(np.abs(residual[large]), 1e-12)
        weights = source_weights * huber_weights
        weighted = design * np.sqrt(weights)[:, None]
        normal = weighted.T @ weighted + penalty
        rhs = design.T @ (weights * target)
        try:
            new_beta = np.linalg.solve(normal, rhs)
        except np.linalg.LinAlgError:
            new_beta = np.linalg.lstsq(normal, rhs, rcond=None)[0]
        # The first physical feature is lateral body-height motion.  A negative
        # coefficient would contradict the intended speed interpretation.
        if new_beta[1] < 0.0:
            new_beta[1] = 0.0
            remaining = [0] + list(range(2, design.shape[1]))
            residual_target = target
            sub_design = design[:, remaining]
            sub_penalty = penalty[np.ix_(remaining, remaining)]
            sub_normal = sub_design.T @ (weights[:, None] * sub_design) + sub_penalty
            sub_rhs = sub_design.T @ (weights * residual_target)
            try:
                solved = np.linalg.solve(sub_normal, sub_rhs)
            except np.linalg.LinAlgError:
                solved = np.linalg.lstsq(sub_normal, sub_rhs, rcond=None)[0]
            new_beta[remaining] = solved
        if float(np.max(np.abs(new_beta - beta))) < 1e-8:
            beta = new_beta
            break
        beta = new_beta
    weighted = design * np.sqrt(weights)[:, None]
    covariance_basis = np.linalg.pinv(weighted.T @ weighted + penalty)
    return beta, covariance_basis


def expanded_feature_bounds(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    low = np.quantile(matrix, 0.01, axis=0)
    high = np.quantile(matrix, 0.99, axis=0)
    q25 = np.quantile(matrix, 0.25, axis=0)
    q75 = np.quantile(matrix, 0.75, axis=0)
    margin = 1.50 * np.maximum(q75 - q25, 1e-6)
    return low - margin, high + margin


def source_balancing_weights(rows: Sequence[Dict[str, Any]]) -> np.ndarray:
    counts: Dict[str, int] = defaultdict(int)
    for row in rows:
        counts[str(row["source_id"])] += 1
    weights = np.asarray(
        [1.0 / counts[str(row["source_id"])] for row in rows],
        dtype=float,
    )
    return weights / max(float(np.mean(weights)), 1e-12)


def fit_model_components(
    rows: Sequence[Dict[str, Any]],
    feature_names: Sequence[str],
    ridge: float,
) -> Dict[str, Any]:
    if not rows:
        fail("Cannot fit a model without calibration rows")
    matrix = np.vstack(
        [row["features"].model_vector(feature_names) for row in rows]
    )
    target = np.asarray(
        [float(row["ground_truth_speed_mps"]) for row in rows],
        dtype=float,
    )
    centre, scale = robust_standardisation(matrix)
    standard = (matrix - centre) / scale
    design = np.column_stack([np.ones(len(rows)), standard])
    coefficients, covariance = constrained_huber_ridge(
        design,
        target,
        ridge=ridge,
        base_weights=source_balancing_weights(rows),
    )
    fitted = design @ coefficients
    residual_sigma = robust_scale(target - fitted)
    if residual_sigma <= 1e-6:
        residual_sigma = max(float(np.std(target - fitted)), 0.01)
    lower, upper = expanded_feature_bounds(matrix)
    return {
        "prediction_mode": "calibrated_regression",
        "feature_names": list(feature_names),
        "feature_centre": centre,
        "feature_scale": scale,
        "coefficients": coefficients,
        "covariance_basis": covariance,
        "feature_lower_bound": lower,
        "feature_upper_bound": upper,
        "residual_sigma_mps": float(residual_sigma),
        "ridge": float(ridge),
    }


def linear_spline_basis(values: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """Return linear interpolation weights for ordered spline knots."""
    x_value = np.asarray(values, dtype=float).reshape(-1)
    knot_value = np.asarray(knots, dtype=float).reshape(-1)
    if knot_value.size < 2 or np.any(np.diff(knot_value) <= 0.0):
        fail("A monotonic spline requires at least two distinct ordered knots")
    basis = np.zeros((x_value.size, knot_value.size), dtype=float)
    for row_index, value in enumerate(x_value):
        if value <= knot_value[0]:
            basis[row_index, 0] = 1.0
            continue
        if value >= knot_value[-1]:
            basis[row_index, -1] = 1.0
            continue
        right = int(np.searchsorted(knot_value, value, side="right"))
        left = right - 1
        fraction = (value - knot_value[left]) / (
            knot_value[right] - knot_value[left]
        )
        basis[row_index, left] = 1.0 - fraction
        basis[row_index, right] = fraction
    return basis


def monotonic_spline_design(
    primary_values: np.ndarray,
    knots: np.ndarray,
    quality_matrix: np.ndarray,
    quality_centre: np.ndarray,
    quality_scale: np.ndarray,
) -> np.ndarray:
    """Create a design whose nonnegative increments imply monotonic speed."""
    basis = linear_spline_basis(primary_values, knots)
    knot_count = basis.shape[1]
    cumulative = (
        np.arange(knot_count)[:, None]
        > np.arange(max(knot_count - 1, 0))[None, :]
    ).astype(float)
    primary_design = np.column_stack(
        [np.ones(basis.shape[0], dtype=float), basis @ cumulative]
    )
    if quality_matrix.size == 0:
        return primary_design
    standard_quality = (quality_matrix - quality_centre) / quality_scale
    return np.column_stack([primary_design, standard_quality])


def fit_monotonic_spline_components(
    rows: Sequence[Dict[str, Any]],
    primary_feature: str,
    quality_features: Sequence[str],
    knot_count: int,
    smoothing: float,
    quality_ridge: float,
) -> Dict[str, Any]:
    """Fit a source balanced robust monotonic additive bbox speed model.

    Knot values are represented by one unrestricted base value followed by
    nonnegative increments.  This converts the monotonicity requirement into
    simple coefficient bounds and keeps every optimisation step convex.
    """
    if not rows:
        fail("Cannot fit a monotonic spline without calibration rows")
    try:
        from scipy.optimize import lsq_linear
    except ImportError as error:
        fail(f"SciPy is required for the monotonic spline model: {error}")

    primary = np.asarray(
        [float(getattr(row["features"], primary_feature)) for row in rows],
        dtype=float,
    )
    target = np.asarray(
        [float(row["ground_truth_speed_mps"]) for row in rows],
        dtype=float,
    )
    requested_quantiles = np.linspace(0.0, 1.0, max(int(knot_count), 2))
    knots = np.unique(np.quantile(primary, requested_quantiles))
    if knots.size < 2:
        fail("The primary bbox proxy has no variation for monotonic fitting")

    if quality_features:
        quality_matrix = np.vstack(
            [row["features"].model_vector(quality_features) for row in rows]
        )
        quality_centre, quality_scale = robust_standardisation(quality_matrix)
    else:
        quality_matrix = np.empty((len(rows), 0), dtype=float)
        quality_centre = np.empty(0, dtype=float)
        quality_scale = np.empty(0, dtype=float)

    design = monotonic_spline_design(
        primary,
        knots,
        quality_matrix,
        quality_centre,
        quality_scale,
    )
    parameter_count = design.shape[1]
    increment_count = knots.size - 1
    penalty_rows: List[np.ndarray] = []
    if increment_count > 1 and smoothing > 0.0:
        for index in range(increment_count - 1):
            penalty = np.zeros(parameter_count, dtype=float)
            penalty[1 + index] = -math.sqrt(float(smoothing))
            penalty[1 + index + 1] = math.sqrt(float(smoothing))
            penalty_rows.append(penalty)
    if quality_features and quality_ridge > 0.0:
        quality_start = 1 + increment_count
        for index in range(len(quality_features)):
            penalty = np.zeros(parameter_count, dtype=float)
            penalty[quality_start + index] = math.sqrt(float(quality_ridge))
            penalty_rows.append(penalty)
    penalty_matrix = (
        np.vstack(penalty_rows)
        if penalty_rows
        else np.empty((0, parameter_count), dtype=float)
    )
    penalty_target = np.zeros(penalty_matrix.shape[0], dtype=float)

    source_weights = source_balancing_weights(rows)
    robust_weights = np.ones(len(rows), dtype=float)
    lower = np.full(parameter_count, -np.inf, dtype=float)
    upper = np.full(parameter_count, np.inf, dtype=float)
    lower[1 : 1 + increment_count] = 0.0
    coefficients = np.zeros(parameter_count, dtype=float)
    coefficients[0] = source_balanced_constant(rows)
    for _ in range(20):
        weights = source_weights * robust_weights
        weighted_design = design * np.sqrt(weights)[:, None]
        weighted_target = target * np.sqrt(weights)
        augmented_design = np.vstack([weighted_design, penalty_matrix])
        augmented_target = np.concatenate([weighted_target, penalty_target])
        solution = lsq_linear(
            augmented_design,
            augmented_target,
            bounds=(lower, upper),
            method="trf",
            tol=1e-10,
            lsmr_tol="auto",
            max_iter=500,
        )
        new_coefficients = np.asarray(solution.x, dtype=float)
        residual = target - design @ new_coefficients
        residual_scale = max(robust_scale(residual), 1e-6)
        huber_threshold = 1.345 * residual_scale
        new_robust_weights = np.ones(len(rows), dtype=float)
        large = np.abs(residual) > huber_threshold
        new_robust_weights[large] = huber_threshold / np.maximum(
            np.abs(residual[large]),
            1e-12,
        )
        converged = float(
            np.max(np.abs(new_coefficients - coefficients))
        ) < 1e-8
        coefficients = new_coefficients
        robust_weights = new_robust_weights
        if converged:
            break

    fitted = design @ coefficients
    residual_sigma = robust_scale(target - fitted)
    if residual_sigma <= 1e-6:
        residual_sigma = max(float(np.std(target - fitted)), 0.01)
    final_weights = source_weights * robust_weights
    normal = design.T @ (final_weights[:, None] * design)
    if penalty_matrix.size:
        normal = normal + penalty_matrix.T @ penalty_matrix
    covariance = np.linalg.pinv(normal)
    all_feature_names = [primary_feature, *quality_features]
    all_matrix = np.vstack(
        [row["features"].model_vector(all_feature_names) for row in rows]
    )
    feature_lower, feature_upper = expanded_feature_bounds(all_matrix)
    return {
        "prediction_mode": "monotonic_spline_gam",
        "feature_names": all_feature_names,
        "primary_feature": primary_feature,
        "quality_features": list(quality_features),
        "spline_knots": knots,
        "spline_knot_count": int(knots.size),
        "spline_smoothing": float(smoothing),
        "spline_quality_ridge": float(quality_ridge),
        "quality_centre": quality_centre,
        "quality_scale": quality_scale,
        "feature_centre": np.zeros(len(all_feature_names), dtype=float),
        "feature_scale": np.ones(len(all_feature_names), dtype=float),
        "coefficients": coefficients,
        "covariance_basis": covariance,
        "feature_lower_bound": feature_lower,
        "feature_upper_bound": feature_upper,
        "residual_sigma_mps": float(residual_sigma),
        "ridge": 0.0,
    }


def monotonic_spline_prediction_design(
    features: TrackFeatures,
    components: Dict[str, Any],
) -> np.ndarray:
    primary_feature = str(components["primary_feature"])
    quality_features = list(components.get("quality_features", []))
    primary = np.asarray([float(getattr(features, primary_feature))], dtype=float)
    if quality_features:
        quality = features.model_vector(quality_features).reshape(1, -1)
    else:
        quality = np.empty((1, 0), dtype=float)
    return monotonic_spline_design(
        primary,
        np.asarray(components["spline_knots"], dtype=float),
        quality,
        np.asarray(components.get("quality_centre", []), dtype=float),
        np.asarray(components.get("quality_scale", []), dtype=float),
    )[0]


def direct_proxy_value(features: TrackFeatures, candidate_name: str) -> float:
    if candidate_name == "direct_raw":
        return float(features.raw_speed_proxy_mps)
    if candidate_name == "direct_q":
        return float(features.q_speed_proxy_mps)
    if candidate_name == "direct_robust":
        return float(features.robust_speed_proxy_mps)
    if candidate_name == "direct_median":
        return float(
            np.median(
                [
                    features.raw_speed_proxy_mps,
                    features.q_speed_proxy_mps,
                    features.robust_speed_proxy_mps,
                ]
            )
        )
    if candidate_name == "direct_compensated_raw":
        return float(features.compensated_raw_speed_proxy_mps)
    if candidate_name == "direct_compensated_q":
        return float(features.compensated_q_speed_proxy_mps)
    if candidate_name == "direct_compensated_robust":
        return float(features.compensated_robust_speed_proxy_mps)
    if candidate_name == "direct_compensated_median":
        return float(
            np.median(
                [
                    features.compensated_raw_speed_proxy_mps,
                    features.compensated_q_speed_proxy_mps,
                    features.compensated_robust_speed_proxy_mps,
                ]
            )
        )
    fail(f"Unsupported direct proxy candidate: {candidate_name}")
    return 0.0


def fit_direct_proxy_components(
    rows: Sequence[Dict[str, Any]],
    candidate_name: str,
    feature_names: Sequence[str],
) -> Dict[str, Any]:
    if not rows:
        fail("Cannot calibrate a direct proxy without development rows")
    matrix = np.vstack(
        [row["features"].model_vector(feature_names) for row in rows]
    )
    target = np.asarray(
        [float(row["ground_truth_speed_mps"]) for row in rows],
        dtype=float,
    )
    fitted = np.asarray(
        [direct_proxy_value(row["features"], candidate_name) for row in rows],
        dtype=float,
    )
    residual_sigma = robust_scale(target - fitted)
    if residual_sigma <= 1e-6:
        residual_sigma = max(float(np.std(target - fitted)), 0.01)
    lower, upper = expanded_feature_bounds(matrix)
    dimension = len(feature_names) + 1
    return {
        "prediction_mode": "direct_proxy",
        "direct_proxy_name": candidate_name,
        "feature_names": list(feature_names),
        "feature_centre": np.zeros(len(feature_names), dtype=float),
        "feature_scale": np.ones(len(feature_names), dtype=float),
        "coefficients": np.zeros(dimension, dtype=float),
        "covariance_basis": np.zeros((dimension, dimension), dtype=float),
        "feature_lower_bound": lower,
        "feature_upper_bound": upper,
        "residual_sigma_mps": float(residual_sigma),
        "ridge": 0.0,
    }


def serialise_extra_trees_regressor(regressor: Any) -> List[Dict[str, Any]]:
    """Store only the arrays needed for deterministic forest inference."""
    trees: List[Dict[str, Any]] = []
    for estimator in regressor.estimators_:
        tree = estimator.tree_
        trees.append(
            {
                "children_left": tree.children_left.astype(int).tolist(),
                "children_right": tree.children_right.astype(int).tolist(),
                "feature": tree.feature.astype(int).tolist(),
                "threshold": tree.threshold.astype(float).tolist(),
                "value": tree.value[:, 0, 0].astype(float).tolist(),
            }
        )
    return trees


def extra_trees_tree_prediction(
    raw_vector: np.ndarray,
    tree: Dict[str, Any],
) -> float:
    """Evaluate one JSON-serialised regression tree."""
    left = tree["children_left"]
    right = tree["children_right"]
    feature = tree["feature"]
    threshold = tree["threshold"]
    value = tree["value"]
    node = 0
    while int(left[node]) >= 0:
        feature_index = int(feature[node])
        node = (
            int(left[node])
            if float(raw_vector[feature_index]) <= float(threshold[node])
            else int(right[node])
        )
    return float(value[node])


def extra_trees_prediction(
    features: TrackFeatures,
    components: Dict[str, Any],
) -> Tuple[float, float]:
    """Return the forest mean and between-tree sample standard deviation."""
    raw_vector = features.model_vector(components["feature_names"])
    predictions = np.asarray(
        [
            extra_trees_tree_prediction(raw_vector, tree)
            for tree in components.get("extra_trees", [])
        ],
        dtype=float,
    )
    if predictions.size == 0:
        fail("Extra Trees model contains no trees")
    dispersion = (
        float(np.std(predictions, ddof=1))
        if predictions.size > 1
        else 0.0
    )
    return float(np.mean(predictions)), dispersion


def fit_extra_trees_components(
    rows: Sequence[Dict[str, Any]],
    feature_names: Sequence[str],
    n_estimators: int,
    minimum_samples_per_leaf: int,
) -> Dict[str, Any]:
    """Fit the bounded-complexity nonlinear candidate on development rows."""
    if not rows:
        fail("Cannot fit Extra Trees without calibration rows")
    try:
        from sklearn.ensemble import ExtraTreesRegressor
    except ImportError as error:
        fail(
            "scikit-learn is required to calibrate the Extra Trees speed "
            f"candidate: {error}"
        )
    matrix = np.vstack(
        [row["features"].model_vector(feature_names) for row in rows]
    )
    target = np.asarray(
        [float(row["ground_truth_speed_mps"]) for row in rows],
        dtype=float,
    )
    regressor = ExtraTreesRegressor(
        n_estimators=int(n_estimators),
        min_samples_leaf=int(minimum_samples_per_leaf),
        max_features=1.0,
        bootstrap=False,
        random_state=DEFAULT_RANDOM_SEED,
        n_jobs=-1,
    )
    # Every labelled crossing is one observation. Source grouping is enforced
    # by the outer validation folds; no held-out source labels enter a tree.
    regressor.fit(matrix, target)
    fitted = np.asarray(regressor.predict(matrix), dtype=float)
    residual_sigma = robust_scale(target - fitted)
    if residual_sigma <= 1e-6:
        residual_sigma = max(float(np.std(target - fitted)), 0.01)
    lower, upper = expanded_feature_bounds(matrix)
    return {
        "prediction_mode": "extra_trees_regression",
        "feature_names": list(feature_names),
        "feature_centre": np.zeros(len(feature_names), dtype=float),
        "feature_scale": np.ones(len(feature_names), dtype=float),
        "coefficients": np.empty(0, dtype=float),
        "covariance_basis": np.empty((0, 0), dtype=float),
        "feature_lower_bound": lower,
        "feature_upper_bound": upper,
        "residual_sigma_mps": float(residual_sigma),
        "ridge": 0.0,
        "extra_trees_n_estimators": int(n_estimators),
        "extra_trees_min_samples_leaf": int(minimum_samples_per_leaf),
        "extra_trees_random_seed": DEFAULT_RANDOM_SEED,
        "extra_trees": serialise_extra_trees_regressor(regressor),
    }


def bounded_variance_prediction(
    base_prediction: float,
    prediction_centre_mps: float,
    target_centre_mps: float,
    expansion_factor: float,
    correction_cap_mps: float,
) -> Tuple[float, float]:
    """Expand variation around the mean while strictly capping each change."""
    delta = float(base_prediction) - float(prediction_centre_mps)
    unbounded_correction = (float(expansion_factor) - 1.0) * delta
    cap = max(float(correction_cap_mps), 0.0)
    correction = float(np.clip(unbounded_correction, -cap, cap))
    derivative = (
        float(expansion_factor)
        if abs(unbounded_correction) < cap
        else 1.0
    )
    return float(target_centre_mps) + delta + correction, derivative


def component_prediction(features: TrackFeatures, components: Dict[str, Any]) -> float:
    if components.get("prediction_mode") == "direct_proxy":
        return direct_proxy_value(features, str(components["direct_proxy_name"]))
    if components.get("prediction_mode") in {
        "extra_trees_regression",
        "extra_trees_bounded_variance",
    }:
        base_prediction, _ = extra_trees_prediction(features, components)
        if components.get("prediction_mode") == "extra_trees_bounded_variance":
            prediction, _ = bounded_variance_prediction(
                base_prediction,
                float(components["variance_prediction_centre_mps"]),
                float(components["variance_target_centre_mps"]),
                float(components["variance_expansion_factor"]),
                float(components["variance_correction_cap_mps"]),
            )
            return prediction
        return base_prediction
    if components.get("prediction_mode") in {
        "monotonic_spline_gam",
        "monotonic_spline_physics_blend",
        "bounded_variance_calibration",
    }:
        design = monotonic_spline_prediction_design(features, components)
        spline_prediction = float(
            design @ np.asarray(components["coefficients"], dtype=float)
        )
        if components.get("prediction_mode") in {
            "monotonic_spline_physics_blend",
            "bounded_variance_calibration",
        }:
            weight = float(components.get("blend_weight", 0.0))
            proxy_feature = str(components.get("blend_proxy_feature", ""))
            proxy_prediction = float(getattr(features, proxy_feature))
            base_prediction = float(
                (1.0 - weight) * spline_prediction
                + weight * proxy_prediction
            )
            if components.get("prediction_mode") == (
                "bounded_variance_calibration"
            ):
                prediction, _ = bounded_variance_prediction(
                    base_prediction,
                    float(components["variance_prediction_centre_mps"]),
                    float(components["variance_target_centre_mps"]),
                    float(components["variance_expansion_factor"]),
                    float(components["variance_correction_cap_mps"]),
                )
                return prediction
            return base_prediction
        return spline_prediction
    vector = features.model_vector(components["feature_names"])
    centre = np.asarray(components["feature_centre"], dtype=float)
    scale = np.asarray(components["feature_scale"], dtype=float)
    standard = (vector - centre) / scale
    design = np.concatenate([[1.0], standard])
    return float(design @ np.asarray(components["coefficients"], dtype=float))


def source_balanced_constant(rows: Sequence[Dict[str, Any]]) -> float:
    source_values: Dict[str, List[float]] = defaultdict(list)
    for row in rows:
        source_values[str(row["source_id"])].append(
            float(row["ground_truth_speed_mps"])
        )
    source_medians = [float(np.median(values)) for values in source_values.values()]
    return float(np.median(source_medians))


def source_balanced_mean(
    rows: Sequence[Dict[str, Any]],
    value_getter: Any,
) -> float:
    """Return the mean of per-source means so each video has equal weight."""
    if not rows:
        fail("Source balanced mean requires at least one row")
    source_values: Dict[str, List[float]] = defaultdict(list)
    for row in rows:
        source_values[str(row["source_id"])].append(
            float(value_getter(row))
        )
    return float(
        np.mean(
            [float(np.mean(values)) for values in source_values.values()]
        )
    )


def average_ranks(values: np.ndarray) -> np.ndarray:
    """Return one based average ranks without adding another dependency."""
    array = np.asarray(values, dtype=float)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(array.size, dtype=float)
    start = 0
    while start < array.size:
        end = start + 1
        while end < array.size and array[order[end]] == array[order[start]]:
            end += 1
        average = 0.5 * ((start + 1) + end)
        ranks[order[start:end]] = average
        start = end
    return ranks


def safe_correlation(left: np.ndarray, right: np.ndarray) -> Optional[float]:
    if left.size < 2 or right.size != left.size:
        return None
    if float(np.std(left)) <= 1e-12 or float(np.std(right)) <= 1e-12:
        return None
    value = float(np.corrcoef(left, right)[0, 1])
    return value if math.isfinite(value) else None


def agreement_metrics(truth: np.ndarray, predicted: np.ndarray) -> Dict[str, Any]:
    """Distribution, calibration, rank, and concordance diagnostics."""
    truth_value = np.asarray(truth, dtype=float)
    prediction_value = np.asarray(predicted, dtype=float)
    if truth_value.size == 0 or prediction_value.size != truth_value.size:
        return {
            "reference_mean_mps": None,
            "prediction_mean_mps": None,
            "reference_sample_sd_mps": None,
            "prediction_sample_sd_mps": None,
            "prediction_reference_sd_ratio": None,
            "prediction_on_reference_calibration_intercept_mps": None,
            "prediction_on_reference_calibration_slope": None,
            "pearson_correlation": None,
            "spearman_correlation": None,
            "lins_concordance_correlation": None,
            "speed_bin_metrics": {},
        }
    reference_mean = float(np.mean(truth_value))
    prediction_mean = float(np.mean(prediction_value))
    reference_sd = (
        float(np.std(truth_value, ddof=1)) if truth_value.size > 1 else 0.0
    )
    prediction_sd = (
        float(np.std(prediction_value, ddof=1))
        if prediction_value.size > 1
        else 0.0
    )
    sd_ratio = (
        prediction_sd / reference_sd if reference_sd > 1e-12 else None
    )
    reference_variance = float(np.var(truth_value))
    if reference_variance > 1e-12:
        covariance = float(
            np.mean(
                (truth_value - reference_mean)
                * (prediction_value - prediction_mean)
            )
        )
        calibration_slope = covariance / reference_variance
        calibration_intercept = (
            prediction_mean - calibration_slope * reference_mean
        )
    else:
        covariance = 0.0
        calibration_slope = None
        calibration_intercept = None
    prediction_variance = float(np.var(prediction_value))
    concordance_denominator = (
        reference_variance
        + prediction_variance
        + (reference_mean - prediction_mean) ** 2
    )
    concordance = (
        2.0 * covariance / concordance_denominator
        if concordance_denominator > 1e-12
        else None
    )
    pearson = safe_correlation(truth_value, prediction_value)
    spearman = safe_correlation(
        average_ranks(truth_value),
        average_ranks(prediction_value),
    )
    bins = {
        "slow_below_1_00_mps": truth_value < 1.00,
        "typical_1_00_to_1_80_mps": (
            (truth_value >= 1.00) & (truth_value <= 1.80)
        ),
        "fast_above_1_80_mps": truth_value > 1.80,
    }
    speed_bins: Dict[str, Dict[str, Any]] = {}
    for name, mask in bins.items():
        count = int(np.sum(mask))
        if count == 0:
            speed_bins[name] = {"count": 0, "mae_mps": None, "bias_mps": None}
            continue
        errors = prediction_value[mask] - truth_value[mask]
        speed_bins[name] = {
            "count": count,
            "mae_mps": float(np.mean(np.abs(errors))),
            "bias_mps": float(np.mean(errors)),
        }
    return {
        "reference_mean_mps": reference_mean,
        "prediction_mean_mps": prediction_mean,
        "reference_sample_sd_mps": reference_sd,
        "prediction_sample_sd_mps": prediction_sd,
        "prediction_reference_sd_ratio": sd_ratio,
        "prediction_on_reference_calibration_intercept_mps": calibration_intercept,
        "prediction_on_reference_calibration_slope": calibration_slope,
        "pearson_correlation": pearson,
        "spearman_correlation": spearman,
        "lins_concordance_correlation": concordance,
        "speed_bin_metrics": speed_bins,
    }


def cross_validation_metrics(predictions: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    truth = np.asarray(
        [float(row["ground_truth_speed_mps"]) for row in predictions],
        dtype=float,
    )
    predicted = np.asarray(
        [float(row["predicted_speed_mps"]) for row in predictions],
        dtype=float,
    )
    error = predicted - truth
    absolute = np.abs(error)
    source_errors: Dict[str, List[float]] = defaultdict(list)
    for row, value in zip(predictions, absolute):
        source_errors[str(row["source_id"])].append(float(value))
    per_source = {
        source: float(np.mean(values))
        for source, values in sorted(source_errors.items())
    }
    return {
        "tracks": len(predictions),
        "sources": len(per_source),
        "track_weighted_mae_mps": float(np.mean(absolute)),
        "source_balanced_mae_mps": float(np.mean(list(per_source.values()))),
        "worst_source_mae_mps": float(max(per_source.values())),
        "rmse_mps": float(math.sqrt(float(np.mean(error * error)))),
        "bias_mps": float(np.mean(error)),
        "within_0_25_mps": float(np.mean(absolute <= 0.25)),
        "within_0_50_mps": float(np.mean(absolute <= 0.50)),
        "per_source_mae_mps": per_source,
        **agreement_metrics(truth, predicted),
    }


def diagnostic_prediction_value(row: Dict[str, Any]) -> Optional[float]:
    """Read a prediction from cross validation, fit, or test output."""
    for field in (
        "predicted_speed_mps",
        "model_speed_before_reliability_gate_mps",
        "estimated_speed_mps",
    ):
        value = safe_float(row.get(field))
        if value is not None:
            return value
    return None


def normalise_speed_diagnostic_rows(
    rows: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Return finite labelled rows with one standard prediction field."""
    output: List[Dict[str, Any]] = []
    for row in rows:
        reference = safe_float(row.get("ground_truth_speed_mps"))
        prediction = diagnostic_prediction_value(row)
        source_id = str(row.get("source_id", "")).strip()
        if reference is None or prediction is None or not source_id:
            continue
        normalised = dict(row)
        normalised["source_id"] = source_id
        normalised["ground_truth_speed_mps"] = float(reference)
        normalised["predicted_speed_mps"] = float(prediction)
        normalised["error_mps"] = float(prediction - reference)
        normalised["absolute_error_mps"] = float(abs(prediction - reference))
        output.append(normalised)
    return output


def sample_standard_deviation(values: Sequence[float]) -> Optional[float]:
    array = np.asarray(values, dtype=float)
    if array.size < 2:
        return None
    return float(np.std(array, ddof=1))


def within_source_correlation(
    records: Sequence[Dict[str, Any]],
    value_field: str,
    target_field: str,
) -> Optional[float]:
    """Correlation after removing each video's mean from both variables."""
    grouped: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    for row in records:
        value = safe_float(row.get(value_field))
        target = safe_float(row.get(target_field))
        if value is None or target is None:
            continue
        grouped[str(row["source_id"])].append((value, target))
    value_residuals: List[float] = []
    target_residuals: List[float] = []
    for pairs in grouped.values():
        if len(pairs) < 2:
            continue
        values = np.asarray([pair[0] for pair in pairs], dtype=float)
        targets = np.asarray([pair[1] for pair in pairs], dtype=float)
        value_residuals.extend((values - float(np.mean(values))).tolist())
        target_residuals.extend((targets - float(np.mean(targets))).tolist())
    return safe_correlation(
        np.asarray(value_residuals, dtype=float),
        np.asarray(target_residuals, dtype=float),
    )


def speed_bin_name(reference_speed_mps: float) -> str:
    if reference_speed_mps < 1.00:
        return "slow_below_1_00_mps"
    if reference_speed_mps <= 1.80:
        return "typical_1_00_to_1_80_mps"
    return "fast_above_1_80_mps"


def speed_bin_diagnostic_rows(
    records: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[speed_bin_name(float(row["ground_truth_speed_mps"]))].append(row)
    output: List[Dict[str, Any]] = []
    for bin_name in (
        "slow_below_1_00_mps",
        "typical_1_00_to_1_80_mps",
        "fast_above_1_80_mps",
    ):
        rows = grouped.get(bin_name, [])
        references = np.asarray(
            [float(row["ground_truth_speed_mps"]) for row in rows],
            dtype=float,
        )
        predictions = np.asarray(
            [float(row["predicted_speed_mps"]) for row in rows],
            dtype=float,
        )
        errors = predictions - references
        reference_sd = sample_standard_deviation(references.tolist())
        prediction_sd = sample_standard_deviation(predictions.tolist())
        output.append(
            {
                "speed_bin": bin_name,
                "tracks": len(rows),
                "sources": len({str(row["source_id"]) for row in rows}),
                "reference_mean_mps": (
                    float(np.mean(references)) if references.size else None
                ),
                "prediction_mean_mps": (
                    float(np.mean(predictions)) if predictions.size else None
                ),
                "reference_sample_sd_mps": reference_sd,
                "prediction_sample_sd_mps": prediction_sd,
                "prediction_reference_sd_ratio": (
                    prediction_sd / reference_sd
                    if reference_sd is not None
                    and prediction_sd is not None
                    and reference_sd > 1e-12
                    else None
                ),
                "mae_mps": (
                    float(np.mean(np.abs(errors))) if errors.size else None
                ),
                "rmse_mps": (
                    float(math.sqrt(float(np.mean(errors * errors))))
                    if errors.size
                    else None
                ),
                "bias_mps": float(np.mean(errors)) if errors.size else None,
                "underestimated_tracks": int(np.sum(errors < 0.0)),
                "overestimated_tracks": int(np.sum(errors > 0.0)),
            }
        )
    return output


def quantile_distribution_summary(values: Sequence[float]) -> Dict[str, Any]:
    array = np.asarray(values, dtype=float)
    if not array.size:
        return {
            "minimum": None,
            "q10": None,
            "q25": None,
            "median": None,
            "q75": None,
            "q90": None,
            "maximum": None,
            "interquartile_range": None,
            "interdecile_range": None,
        }
    quantiles = np.quantile(array, [0.10, 0.25, 0.50, 0.75, 0.90])
    return {
        "minimum": float(np.min(array)),
        "q10": float(quantiles[0]),
        "q25": float(quantiles[1]),
        "median": float(quantiles[2]),
        "q75": float(quantiles[3]),
        "q90": float(quantiles[4]),
        "maximum": float(np.max(array)),
        "interquartile_range": float(quantiles[3] - quantiles[1]),
        "interdecile_range": float(quantiles[4] - quantiles[0]),
    }


def speed_feature_signal_rows(
    records: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    references_all = np.asarray(
        [float(row["ground_truth_speed_mps"]) for row in records],
        dtype=float,
    )
    errors_all = np.asarray(
        [float(row["absolute_error_mps"]) for row in records],
        dtype=float,
    )
    for feature in SPEED_INFORMATION_FEATURES:
        retained = [
            (index, value)
            for index, row in enumerate(records)
            for value in [safe_float(row.get(feature))]
            if value is not None
        ]
        if len(retained) < 3:
            continue
        indices = np.asarray([index for index, _ in retained], dtype=int)
        values = np.asarray([value for _, value in retained], dtype=float)
        references = references_all[indices]
        absolute_errors = errors_all[indices]
        pearson = safe_correlation(values, references)
        spearman = safe_correlation(
            average_ranks(values),
            average_ranks(references),
        )
        within_source = within_source_correlation(
            [records[index] for index in indices.tolist()],
            feature,
            "ground_truth_speed_mps",
        )
        error_correlation = safe_correlation(values, absolute_errors)
        bin_medians: Dict[str, Optional[float]] = {}
        for bin_name in (
            "slow_below_1_00_mps",
            "typical_1_00_to_1_80_mps",
            "fast_above_1_80_mps",
        ):
            selected_values = [
                value
                for value, reference in zip(values, references)
                if speed_bin_name(float(reference)) == bin_name
            ]
            bin_medians[bin_name] = (
                float(np.median(selected_values)) if selected_values else None
            )
        signal_value = within_source if within_source is not None else pearson
        absolute_signal = abs(signal_value) if signal_value is not None else 0.0
        if absolute_signal >= 0.35:
            strength = "strong"
        elif absolute_signal >= 0.20:
            strength = "moderate"
        else:
            strength = "weak"
        output.append(
            {
                "feature": feature,
                "tracks": len(values),
                "sources": len(
                    {str(records[index]["source_id"]) for index in indices}
                ),
                "pooled_pearson_with_reference_speed": pearson,
                "pooled_spearman_with_reference_speed": spearman,
                "within_source_pearson_with_reference_speed": within_source,
                "pooled_pearson_with_absolute_error": error_correlation,
                "slow_bin_median": bin_medians["slow_below_1_00_mps"],
                "typical_bin_median": bin_medians[
                    "typical_1_00_to_1_80_mps"
                ],
                "fast_bin_median": bin_medians["fast_above_1_80_mps"],
                "signal_strength": strength,
            }
        )
    output.sort(
        key=lambda row: (
            abs(
                safe_float(
                    row.get("within_source_pearson_with_reference_speed")
                )
                or 0.0
            ),
            abs(
                safe_float(row.get("pooled_pearson_with_reference_speed"))
                or 0.0
            ),
        ),
        reverse=True,
    )
    for rank, row in enumerate(output, start=1):
        row["within_source_signal_rank"] = rank
    return output


def speed_information_diagnostic_report(
    rows: Sequence[Dict[str, Any]],
    role: str,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    records = normalise_speed_diagnostic_rows(rows)
    if not records:
        fail("No finite ground truth and predicted speed pairs are available")
    truth = np.asarray(
        [float(row["ground_truth_speed_mps"]) for row in records],
        dtype=float,
    )
    prediction = np.asarray(
        [float(row["predicted_speed_mps"]) for row in records],
        dtype=float,
    )
    metrics = cross_validation_metrics(records)
    reference_range = quantile_distribution_summary(truth.tolist())
    prediction_range = quantile_distribution_summary(prediction.tolist())
    reference_iqr = safe_float(reference_range["interquartile_range"])
    prediction_iqr = safe_float(prediction_range["interquartile_range"])
    reference_idr = safe_float(reference_range["interdecile_range"])
    prediction_idr = safe_float(prediction_range["interdecile_range"])
    range_compression = {
        "reference_speed_mps": reference_range,
        "predicted_speed_mps": prediction_range,
        "prediction_reference_iqr_ratio": (
            prediction_iqr / reference_iqr
            if reference_iqr is not None
            and prediction_iqr is not None
            and reference_iqr > 1e-12
            else None
        ),
        "prediction_reference_interdecile_ratio": (
            prediction_idr / reference_idr
            if reference_idr is not None
            and prediction_idr is not None
            and reference_idr > 1e-12
            else None
        ),
    }
    bin_rows = speed_bin_diagnostic_rows(records)
    feature_rows = speed_feature_signal_rows(records)
    worst_rows = sorted(
        records,
        key=lambda row: float(row["absolute_error_mps"]),
        reverse=True,
    )[:25]
    worst_output = [
        {
            "error_rank": rank,
            "source_id": row["source_id"],
            "prediction_track_id": row.get("prediction_track_id", ""),
            "ground_truth_track_id": row.get("ground_truth_track_id", ""),
            "reference_speed_bin": speed_bin_name(
                float(row["ground_truth_speed_mps"])
            ),
            "ground_truth_speed_mps": row["ground_truth_speed_mps"],
            "predicted_speed_mps": row["predicted_speed_mps"],
            "error_mps": row["error_mps"],
            "absolute_error_mps": row["absolute_error_mps"],
            "error_direction": (
                "underestimate" if float(row["error_mps"]) < 0.0 else "overestimate"
            ),
            **{
                feature: row.get(feature, "")
                for feature in SPEED_INFORMATION_FEATURES
            },
        }
        for rank, row in enumerate(worst_rows, start=1)
    ]
    strong_features = [
        str(row["feature"])
        for row in feature_rows
        if row["signal_strength"] == "strong"
    ]
    moderate_features = [
        str(row["feature"])
        for row in feature_rows
        if row["signal_strength"] == "moderate"
    ]
    sd_ratio = safe_float(metrics.get("prediction_reference_sd_ratio"))
    slope = safe_float(
        metrics.get("prediction_on_reference_calibration_slope")
    )
    spread_is_adequate = bool(
        sd_ratio is not None
        and slope is not None
        and sd_ratio
        >= float(CROSS_VALIDATION_SETTINGS["minimum_prediction_reference_sd_ratio"])
        and slope >= float(CROSS_VALIDATION_SETTINGS["minimum_calibration_slope"])
    )
    report = {
        "schema": "crowd_bbox_speed_information_diagnostic_v1",
        "release_id": RELEASE_ID,
        "evaluation_role": role,
        "tracks": len(records),
        "sources": len({str(row["source_id"]) for row in records}),
        "selection_note": (
            "Development diagnostics may guide a predeclared next model. "
            "Untouched validation diagnostics are audit only and must not be "
            "used to choose features, thresholds, or hyperparameters."
        ),
        "overall_metrics": metrics,
        "range_compression": range_compression,
        "speed_bins": bin_rows,
        "feature_signal_summary": {
            "strong_within_source_features": strong_features,
            "moderate_within_source_features": moderate_features,
            "ranking_method": (
                "absolute Pearson correlation after subtracting each source mean"
            ),
        },
        "diagnostic_conclusion": {
            "prediction_spread_adequate": spread_is_adequate,
            "status": (
                "speed_variation_preserved"
                if spread_is_adequate
                else "speed_variation_compressed"
            ),
            "recommended_next_step": (
                "Test only compact source grouped models using the strongest "
                "development features, then evaluate once on untouched data."
                if strong_features
                else "The current CSV features show no strong source independent "
                "speed signal. Add metric geometry or another deployment input "
                "before claiming individual speed in metres per second."
            ),
        },
    }
    return report, bin_rows, feature_rows, worst_output


def write_speed_information_diagnostics(
    rows: Sequence[Dict[str, Any]],
    output_dir: Path,
    role: str,
) -> Dict[str, Any]:
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    report, bin_rows, feature_rows, worst_rows = (
        speed_information_diagnostic_report(rows, role)
    )
    write_json(str(output_path / "speed_information_diagnostic.json"), report)
    write_dict_csv(str(output_path / "speed_bin_diagnostics.csv"), bin_rows)
    write_dict_csv(str(output_path / "speed_feature_signal.csv"), feature_rows)
    write_dict_csv(str(output_path / "worst_speed_errors.csv"), worst_rows)
    log(
        "Speed information diagnostic: "
        f"role={role}, tracks={report['tracks']}, sources={report['sources']}, "
        f"status={report['diagnostic_conclusion']['status']}"
    )
    log(
        "Strong within source bbox signals: "
        + (
            ", ".join(
                report["feature_signal_summary"]["strong_within_source_features"]
            )
            or "none"
        )
    )
    log(f"Diagnostic report: {output_path / 'speed_information_diagnostic.json'}")
    return report


def mode_diagnose_speed_information(
    prediction_csv: str,
    output_dir: str,
    role: str,
) -> None:
    rows = read_dict_csv(prediction_csv)
    write_speed_information_diagnostics(rows, Path(output_dir), role)


def leave_one_source_out_predictions(
    rows: Sequence[Dict[str, Any]],
    feature_names: Sequence[str],
    ridge: float,
    candidate_name: str,
) -> List[Dict[str, Any]]:
    sources = sorted({str(row["source_id"]) for row in rows})
    output: List[Dict[str, Any]] = []
    for held_out_source in sources:
        training = [
            row for row in rows if str(row["source_id"]) != held_out_source
        ]
        held_out = [
            row for row in rows if str(row["source_id"]) == held_out_source
        ]
        components = fit_model_components(training, feature_names, ridge)
        baseline = source_balanced_constant(training)
        for row in held_out:
            predicted = component_prediction(row["features"], components)
            truth = float(row["ground_truth_speed_mps"])
            output.append(
                {
                    "source_id": row["source_id"],
                    "prediction_track_id": row["prediction_track_id"],
                    "ground_truth_track_id": row.get("ground_truth_track_id", ""),
                    "ground_truth_speed_mps": truth,
                    "predicted_speed_mps": predicted,
                    "error_mps": predicted - truth,
                    "absolute_error_mps": abs(predicted - truth),
                    "baseline_speed_mps": baseline,
                    "baseline_absolute_error_mps": abs(baseline - truth),
                    "candidate": candidate_name,
                    "prediction_mode": "calibrated_regression",
                    "ridge": float(ridge),
                    "held_out_source": held_out_source,
                    **features_to_row(row["features"]),
                }
            )
    return output


def monotonic_spline_cross_validation_predictions(
    rows: Sequence[Dict[str, Any]],
    candidate_name: str,
    primary_feature: str,
    quality_features: Sequence[str],
    knot_count: int,
    smoothing: float,
    quality_ridge: float,
) -> List[Dict[str, Any]]:
    sources = sorted({str(row["source_id"]) for row in rows})
    output: List[Dict[str, Any]] = []
    for held_out_source in sources:
        training = [
            row for row in rows if str(row["source_id"]) != held_out_source
        ]
        held_out = [
            row for row in rows if str(row["source_id"]) == held_out_source
        ]
        components = fit_monotonic_spline_components(
            training,
            primary_feature,
            quality_features,
            knot_count,
            smoothing,
            quality_ridge,
        )
        baseline = source_balanced_constant(training)
        for row in held_out:
            predicted = component_prediction(row["features"], components)
            truth = float(row["ground_truth_speed_mps"])
            output.append(
                {
                    "source_id": row["source_id"],
                    "prediction_track_id": row["prediction_track_id"],
                    "ground_truth_track_id": row.get("ground_truth_track_id", ""),
                    "ground_truth_speed_mps": truth,
                    "predicted_speed_mps": predicted,
                    "error_mps": predicted - truth,
                    "absolute_error_mps": abs(predicted - truth),
                    "baseline_speed_mps": baseline,
                    "baseline_absolute_error_mps": abs(baseline - truth),
                    "candidate": candidate_name,
                    "prediction_mode": "monotonic_spline_gam",
                    "ridge": 0.0,
                    "spline_knot_count": int(knot_count),
                    "spline_smoothing": float(smoothing),
                    "spline_quality_ridge": float(quality_ridge),
                    "held_out_source": held_out_source,
                    **features_to_row(row["features"]),
                }
            )
    return output


def monotonic_spline_physics_blend_cross_validation_predictions(
    rows: Sequence[Dict[str, Any]],
    candidate_name: str,
    primary_feature: str,
    quality_features: Sequence[str],
    knot_count: int,
    smoothing: float,
    quality_ridge: float,
    blend_proxy_feature: str,
    blend_weight: float,
) -> List[Dict[str, Any]]:
    """Cross validate a calibrated spline blended with a physical proxy.

    The spline provides the low-error conditional estimate.  A small direct
    proxy contribution restores individual speed variation that conditional
    regression otherwise shrinks towards the mean.  Both terms use only the
    bbox CSV and FPS at prediction time.
    """
    sources = sorted({str(row["source_id"]) for row in rows})
    weight = float(blend_weight)
    output: List[Dict[str, Any]] = []
    for held_out_source in sources:
        training = [
            row for row in rows if str(row["source_id"]) != held_out_source
        ]
        held_out = [
            row for row in rows if str(row["source_id"]) == held_out_source
        ]
        components = fit_monotonic_spline_components(
            training,
            primary_feature,
            quality_features,
            knot_count,
            smoothing,
            quality_ridge,
        )
        baseline = source_balanced_constant(training)
        for row in held_out:
            spline_prediction = component_prediction(
                row["features"],
                components,
            )
            proxy_prediction = float(
                getattr(row["features"], blend_proxy_feature)
            )
            predicted = (
                (1.0 - weight) * spline_prediction
                + weight * proxy_prediction
            )
            truth = float(row["ground_truth_speed_mps"])
            output.append(
                {
                    "source_id": row["source_id"],
                    "prediction_track_id": row["prediction_track_id"],
                    "ground_truth_track_id": row.get(
                        "ground_truth_track_id",
                        "",
                    ),
                    "ground_truth_speed_mps": truth,
                    "predicted_speed_mps": predicted,
                    "error_mps": predicted - truth,
                    "absolute_error_mps": abs(predicted - truth),
                    "baseline_speed_mps": baseline,
                    "baseline_absolute_error_mps": abs(baseline - truth),
                    "candidate": candidate_name,
                    "prediction_mode": "monotonic_spline_physics_blend",
                    "ridge": 0.0,
                    "spline_knot_count": int(knot_count),
                    "spline_smoothing": float(smoothing),
                    "spline_quality_ridge": float(quality_ridge),
                    "blend_proxy_feature": blend_proxy_feature,
                    "blend_weight": weight,
                    "held_out_source": held_out_source,
                    **features_to_row(row["features"]),
                }
            )
    return output


def extra_trees_cross_validation_predictions(
    rows: Sequence[Dict[str, Any]],
    candidate_name: str,
    feature_names: Sequence[str],
    n_estimators: int,
    minimum_samples_per_leaf: int,
) -> List[Dict[str, Any]]:
    """Leave one complete video source out of nonlinear model fitting."""
    sources = sorted({str(row["source_id"]) for row in rows})
    output: List[Dict[str, Any]] = []
    for held_out_source in sources:
        training = [
            row for row in rows if str(row["source_id"]) != held_out_source
        ]
        testing = [
            row for row in rows if str(row["source_id"]) == held_out_source
        ]
        components = fit_extra_trees_components(
            training,
            feature_names,
            n_estimators,
            minimum_samples_per_leaf,
        )
        baseline = source_balanced_constant(training)
        for row in testing:
            predicted = component_prediction(row["features"], components)
            truth = float(row["ground_truth_speed_mps"])
            output.append(
                {
                    "source_id": row["source_id"],
                    "prediction_track_id": row["prediction_track_id"],
                    "ground_truth_track_id": row.get(
                        "ground_truth_track_id",
                        "",
                    ),
                    "ground_truth_speed_mps": truth,
                    "predicted_speed_mps": predicted,
                    "error_mps": predicted - truth,
                    "absolute_error_mps": abs(predicted - truth),
                    "baseline_speed_mps": baseline,
                    "baseline_absolute_error_mps": abs(baseline - truth),
                    "candidate": candidate_name,
                    "prediction_mode": "extra_trees_regression",
                    "extra_trees_n_estimators": int(n_estimators),
                    "extra_trees_min_samples_leaf": int(
                        minimum_samples_per_leaf
                    ),
                    "held_out_source": held_out_source,
                    **features_to_row(row["features"]),
                }
            )
    return output


def bounded_variance_cross_validation_predictions(
    base_predictions: Sequence[Dict[str, Any]],
    expansion_factor: float,
    correction_cap_mps: float,
    candidate_name: str = "bounded_variance_calibration",
    prediction_mode: str = "bounded_variance_calibration",
) -> List[Dict[str, Any]]:
    """Cross fit a bounded variance correction without validation labels.

    Each held out source uses target and prediction centres calculated from
    the other development sources.  The base prediction is already source
    held out.  This prevents a video's own labels from determining its centre
    or its corrected prediction.
    """
    output: List[Dict[str, Any]] = []
    sources = sorted(
        {str(row["source_id"]) for row in base_predictions}
    )
    for held_out_source in sources:
        calibration_rows = [
            row
            for row in base_predictions
            if str(row["source_id"]) != held_out_source
        ]
        target_centre = source_balanced_mean(
            calibration_rows,
            lambda row: row["ground_truth_speed_mps"],
        )
        prediction_centre = source_balanced_mean(
            calibration_rows,
            lambda row: row["predicted_speed_mps"],
        )
        for row in base_predictions:
            if str(row["source_id"]) != held_out_source:
                continue
            predicted, _ = bounded_variance_prediction(
                float(row["predicted_speed_mps"]),
                prediction_centre,
                target_centre,
                expansion_factor,
                correction_cap_mps,
            )
            truth = float(row["ground_truth_speed_mps"])
            output.append(
                {
                    **row,
                    "predicted_speed_mps": predicted,
                    "error_mps": predicted - truth,
                    "absolute_error_mps": abs(predicted - truth),
                    "candidate": candidate_name,
                    "prediction_mode": prediction_mode,
                    "variance_expansion_factor": float(expansion_factor),
                    "variance_correction_cap_mps": float(
                        correction_cap_mps
                    ),
                    "variance_prediction_centre_mps": prediction_centre,
                    "variance_target_centre_mps": target_centre,
                    "held_out_source": held_out_source,
                }
            )
    return output


def direct_proxy_cross_validation_predictions(
    rows: Sequence[Dict[str, Any]],
    candidate_name: str,
) -> List[Dict[str, Any]]:
    sources = sorted({str(row["source_id"]) for row in rows})
    output: List[Dict[str, Any]] = []
    for held_out_source in sources:
        training = [
            row for row in rows if str(row["source_id"]) != held_out_source
        ]
        baseline = source_balanced_constant(training)
        for row in rows:
            if str(row["source_id"]) != held_out_source:
                continue
            predicted = direct_proxy_value(row["features"], candidate_name)
            truth = float(row["ground_truth_speed_mps"])
            output.append(
                {
                    "source_id": row["source_id"],
                    "prediction_track_id": row["prediction_track_id"],
                    "ground_truth_track_id": row.get("ground_truth_track_id", ""),
                    "ground_truth_speed_mps": truth,
                    "predicted_speed_mps": predicted,
                    "error_mps": predicted - truth,
                    "absolute_error_mps": abs(predicted - truth),
                    "baseline_speed_mps": baseline,
                    "baseline_absolute_error_mps": abs(baseline - truth),
                    "candidate": candidate_name,
                    "prediction_mode": "direct_proxy",
                    "ridge": 0.0,
                    "held_out_source": held_out_source,
                    **features_to_row(row["features"]),
                }
            )
    return output


def baseline_cross_validation_predictions(
    rows: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    sources = sorted({str(row["source_id"]) for row in rows})
    output: List[Dict[str, Any]] = []
    for held_out_source in sources:
        training = [
            row for row in rows if str(row["source_id"]) != held_out_source
        ]
        baseline = source_balanced_constant(training)
        for row in rows:
            if str(row["source_id"]) != held_out_source:
                continue
            truth = float(row["ground_truth_speed_mps"])
            output.append(
                {
                    "source_id": row["source_id"],
                    "prediction_track_id": row["prediction_track_id"],
                    "ground_truth_speed_mps": truth,
                    "predicted_speed_mps": baseline,
                    **features_to_row(row["features"]),
                }
            )
    return output


def select_cross_validated_model(
    development: Sequence[Dict[str, Any]],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    sources = {str(row["source_id"]) for row in development}
    minimum_tracks = int(CROSS_VALIDATION_SETTINGS["minimum_development_tracks"])
    minimum_sources = int(CROSS_VALIDATION_SETTINGS["minimum_development_sources"])
    if len(development) < minimum_tracks:
        fail(
            f"At least {minimum_tracks} eligible development tracks are required; "
            f"found {len(development)}"
        )
    if len(sources) < minimum_sources:
        fail(
            f"At least {minimum_sources} independent development sources are required; "
            f"found {len(sources)}"
        )

    baseline_predictions = baseline_cross_validation_predictions(development)
    reliable_baseline_predictions = [
        row
        for row in baseline_predictions
        if reliability_gate_reason_from_row(row) == ""
    ]
    if not reliable_baseline_predictions:
        fail("No development tracks pass the locked reliability gates")
    baseline_metrics = cross_validation_metrics(reliable_baseline_predictions)
    unfiltered_baseline_metrics = cross_validation_metrics(baseline_predictions)
    baseline_mae = float(baseline_metrics["source_balanced_mae_mps"])
    candidate_results: List[Dict[str, Any]] = []
    prediction_cache: Dict[
        Tuple[str, float, int, float, float, float, float, int, int],
        List[Dict[str, Any]],
    ] = {}

    def register_candidate(
        candidate_name: str,
        prediction_mode: str,
        feature_names: Sequence[str],
        ridge: float,
        predictions: List[Dict[str, Any]],
        spline_knot_count: int = 0,
        spline_smoothing: float = 0.0,
        spline_quality_ridge: float = 0.0,
        blend_proxy_feature: str = "",
        blend_weight: float = 0.0,
        variance_expansion_factor: float = 1.0,
        variance_correction_cap_mps: float = 0.0,
        extra_trees_n_estimators: int = 0,
        extra_trees_min_samples_leaf: int = 0,
    ) -> None:
        reliable_predictions = [
            row
            for row in predictions
            if reliability_gate_reason_from_row(row) == ""
        ]
        metrics = cross_validation_metrics(reliable_predictions)
        unfiltered_metrics = cross_validation_metrics(predictions)
        reliable_coverage = len(reliable_predictions) / max(len(predictions), 1)
        conformal_uncertainty = conformal_absolute_error(
            reliable_predictions,
            float(CROSS_VALIDATION_SETTINGS["conformal_coverage"]),
        )
        improvement = (
            baseline_mae - float(metrics["source_balanced_mae_mps"])
        ) / max(baseline_mae, 1e-12)
        reasons: List[str] = []
        if improvement < float(
            CROSS_VALIDATION_SETTINGS["minimum_baseline_improvement_fraction"]
        ):
            reasons.append("insufficient_improvement_over_constant_baseline")
        if float(metrics["source_balanced_mae_mps"]) > float(
            CROSS_VALIDATION_SETTINGS["maximum_source_balanced_mae_mps"]
        ):
            reasons.append("source_balanced_mae_too_high")
        if float(metrics["worst_source_mae_mps"]) > float(
            CROSS_VALIDATION_SETTINGS["maximum_worst_source_mae_mps"]
        ):
            reasons.append("worst_source_mae_too_high")
        if conformal_uncertainty > float(
            CROSS_VALIDATION_SETTINGS[
                "maximum_conformal_absolute_uncertainty_mps"
            ]
        ):
            reasons.append("cross_validated_uncertainty_too_high")
        if reliable_coverage < float(
            CROSS_VALIDATION_SETTINGS["minimum_reliable_coverage"]
        ):
            reasons.append("reliable_development_coverage_too_low")
        if int(metrics["sources"]) < int(
            CROSS_VALIDATION_SETTINGS["minimum_reliable_sources"]
        ):
            reasons.append("too_few_reliable_development_sources")
        sd_ratio = safe_float(metrics.get("prediction_reference_sd_ratio"))
        if (
            sd_ratio is None
            or sd_ratio
            < float(
                CROSS_VALIDATION_SETTINGS[
                    "minimum_prediction_reference_sd_ratio"
                ]
            )
        ):
            reasons.append("prediction_spread_too_narrow")
        elif sd_ratio > float(
            CROSS_VALIDATION_SETTINGS["maximum_prediction_reference_sd_ratio"]
        ):
            reasons.append("prediction_spread_too_wide")
        calibration_slope = safe_float(
            metrics.get("prediction_on_reference_calibration_slope")
        )
        if (
            calibration_slope is None
            or calibration_slope
            < float(CROSS_VALIDATION_SETTINGS["minimum_calibration_slope"])
        ):
            reasons.append("calibration_slope_too_low")
        elif calibration_slope > float(
            CROSS_VALIDATION_SETTINGS["maximum_calibration_slope"]
        ):
            reasons.append("calibration_slope_too_high")
        pearson = safe_float(metrics.get("pearson_correlation"))
        if pearson is None or pearson < float(
            CROSS_VALIDATION_SETTINGS["minimum_pearson_correlation"]
        ):
            reasons.append("pearson_correlation_too_low")
        spearman = safe_float(metrics.get("spearman_correlation"))
        if spearman is None or spearman < float(
            CROSS_VALIDATION_SETTINGS["minimum_spearman_correlation"]
        ):
            reasons.append("spearman_correlation_too_low")
        concordance = safe_float(metrics.get("lins_concordance_correlation"))
        if concordance is None or concordance < float(
            CROSS_VALIDATION_SETTINGS["minimum_lins_concordance"]
        ):
            reasons.append("lins_concordance_too_low")
        for row in predictions:
            row["reliability_status"] = (
                "eligible"
                if reliability_gate_reason_from_row(row) == ""
                else "rejected"
            )
            row["reliability_reject_reason"] = reliability_gate_reason_from_row(
                row
            )
        prediction_cache[
            (
                candidate_name,
                float(ridge),
                int(spline_knot_count),
                float(spline_smoothing),
                float(blend_weight),
                float(variance_expansion_factor),
                float(variance_correction_cap_mps),
                int(extra_trees_n_estimators),
                int(extra_trees_min_samples_leaf),
            )
        ] = predictions
        candidate_results.append(
            {
                "candidate": candidate_name,
                "prediction_mode": prediction_mode,
                "feature_names": list(feature_names),
                "feature_count": len(feature_names),
                "ridge": float(ridge),
                "spline_knot_count": int(spline_knot_count),
                "spline_smoothing": float(spline_smoothing),
                "spline_quality_ridge": float(spline_quality_ridge),
                "blend_proxy_feature": str(blend_proxy_feature),
                "blend_weight": float(blend_weight),
                "variance_expansion_factor": float(
                    variance_expansion_factor
                ),
                "variance_correction_cap_mps": float(
                    variance_correction_cap_mps
                ),
                "extra_trees_n_estimators": int(
                    extra_trees_n_estimators
                ),
                "extra_trees_min_samples_leaf": int(
                    extra_trees_min_samples_leaf
                ),
                **metrics,
                "unfiltered_metrics": unfiltered_metrics,
                "reliable_tracks": len(reliable_predictions),
                "reliable_sources": int(metrics["sources"]),
                "reliable_coverage": float(reliable_coverage),
                "baseline_improvement_fraction": float(improvement),
                "conformal_absolute_uncertainty_mps": float(
                    conformal_uncertainty
                ),
                "calibration_qualified": not reasons,
                "qualification_reasons": reasons,
            }
        )

    for candidate_name, feature_names in MODEL_CANDIDATES.items():
        for ridge in RIDGE_CANDIDATES:
            predictions = leave_one_source_out_predictions(
                development,
                feature_names,
                ridge,
                candidate_name,
            )
            register_candidate(
                candidate_name,
                "calibrated_regression",
                feature_names,
                float(ridge),
                predictions,
            )
    for candidate_name, feature_names in DIRECT_PROXY_CANDIDATES.items():
        predictions = direct_proxy_cross_validation_predictions(
            development,
            candidate_name,
        )
        register_candidate(
            candidate_name,
            "direct_proxy",
            feature_names,
            0.0,
            predictions,
        )
    for candidate_name, settings in MONOTONIC_SPLINE_CANDIDATES.items():
        primary_feature = str(settings["primary_feature"])
        quality_features = list(settings.get("quality_features", []))
        feature_names = [primary_feature, *quality_features]
        for knot_count in MONOTONIC_SPLINE_KNOT_CANDIDATES:
            for smoothing in MONOTONIC_SPLINE_SMOOTHING_CANDIDATES:
                predictions = monotonic_spline_cross_validation_predictions(
                    development,
                    candidate_name,
                    primary_feature,
                    quality_features,
                    knot_count,
                    smoothing,
                    MONOTONIC_SPLINE_QUALITY_RIDGE,
                )
                register_candidate(
                    candidate_name,
                    "monotonic_spline_gam",
                    feature_names,
                    0.0,
                    predictions,
                    spline_knot_count=int(knot_count),
                    spline_smoothing=float(smoothing),
                    spline_quality_ridge=float(MONOTONIC_SPLINE_QUALITY_RIDGE),
                )

    # V31 challenger: retain the low-error monotonic estimate while restoring
    # a small amount of the physical proxy's individual speed variation.
    # Every weight is evaluated with the same source-held-out protocol.
    blend_candidate = "monotonic_spline_physics_blend"
    blend_primary = "compensated_robust_speed_proxy_mps"
    blend_features = [blend_primary]
    for knot_count in MONOTONIC_SPLINE_KNOT_CANDIDATES:
        for smoothing in MONOTONIC_SPLINE_SMOOTHING_CANDIDATES:
            for blend_weight in MONOTONIC_SPLINE_PHYSICS_BLEND_WEIGHTS:
                predictions = (
                    monotonic_spline_physics_blend_cross_validation_predictions(
                        development,
                        blend_candidate,
                        blend_primary,
                        [],
                        knot_count,
                        smoothing,
                        MONOTONIC_SPLINE_QUALITY_RIDGE,
                        MONOTONIC_SPLINE_PHYSICS_BLEND_PROXY,
                        blend_weight,
                    )
                )
                register_candidate(
                    blend_candidate,
                    "monotonic_spline_physics_blend",
                    blend_features,
                    0.0,
                    predictions,
                    spline_knot_count=int(knot_count),
                    spline_smoothing=float(smoothing),
                    spline_quality_ridge=float(
                        MONOTONIC_SPLINE_QUALITY_RIDGE
                    ),
                    blend_proxy_feature=(
                        MONOTONIC_SPLINE_PHYSICS_BLEND_PROXY
                    ),
                    blend_weight=float(blend_weight),
                )
                for expansion_factor in BOUNDED_VARIANCE_FACTORS:
                    for correction_cap in BOUNDED_VARIANCE_CAPS_MPS:
                        variance_predictions = (
                            bounded_variance_cross_validation_predictions(
                                predictions,
                                expansion_factor,
                                correction_cap,
                            )
                        )
                        register_candidate(
                            "bounded_variance_calibration",
                            "bounded_variance_calibration",
                            blend_features,
                            0.0,
                            variance_predictions,
                            spline_knot_count=int(knot_count),
                            spline_smoothing=float(smoothing),
                            spline_quality_ridge=float(
                                MONOTONIC_SPLINE_QUALITY_RIDGE
                            ),
                            blend_proxy_feature=(
                                MONOTONIC_SPLINE_PHYSICS_BLEND_PROXY
                            ),
                            blend_weight=float(blend_weight),
                            variance_expansion_factor=float(
                                expansion_factor
                            ),
                            variance_correction_cap_mps=float(
                                correction_cap
                            ),
                        )

    # V32 nonlinear challenger. Hyperparameters and the bounded expansion grid
    # are fixed before Waymo validation is opened. Each prediction comes from
    # a forest that never saw the held-out video's labels.
    tree_candidate = "extra_trees_compact"
    tree_predictions = extra_trees_cross_validation_predictions(
        development,
        tree_candidate,
        EXTRA_TREES_FEATURES,
        EXTRA_TREES_N_ESTIMATORS,
        EXTRA_TREES_MIN_SAMPLES_LEAF,
    )
    register_candidate(
        tree_candidate,
        "extra_trees_regression",
        EXTRA_TREES_FEATURES,
        0.0,
        tree_predictions,
        extra_trees_n_estimators=EXTRA_TREES_N_ESTIMATORS,
        extra_trees_min_samples_leaf=EXTRA_TREES_MIN_SAMPLES_LEAF,
    )
    for expansion_factor in EXTRA_TREES_VARIANCE_FACTORS:
        for correction_cap in EXTRA_TREES_VARIANCE_CAPS_MPS:
            variance_predictions = bounded_variance_cross_validation_predictions(
                tree_predictions,
                expansion_factor,
                correction_cap,
                candidate_name="extra_trees_bounded_variance",
                prediction_mode="extra_trees_bounded_variance",
            )
            register_candidate(
                "extra_trees_bounded_variance",
                "extra_trees_bounded_variance",
                EXTRA_TREES_FEATURES,
                0.0,
                variance_predictions,
                variance_expansion_factor=float(expansion_factor),
                variance_correction_cap_mps=float(correction_cap),
                extra_trees_n_estimators=EXTRA_TREES_N_ESTIMATORS,
                extra_trees_min_samples_leaf=EXTRA_TREES_MIN_SAMPLES_LEAF,
            )

    qualified_candidates = [
        row for row in candidate_results if row["calibration_qualified"]
    ]
    if qualified_candidates:
        selection_pool = qualified_candidates
    else:
        minimum_failure_count = min(
            len(row["qualification_reasons"]) for row in candidate_results
        )
        selection_pool = [
            row
            for row in candidate_results
            if len(row["qualification_reasons"]) == minimum_failure_count
        ]
    best_mae = min(float(row["source_balanced_mae_mps"]) for row in selection_pool)
    margin = float(CROSS_VALIDATION_SETTINGS["simplicity_margin_mps"])
    shortlist = [
        row
        for row in selection_pool
        if float(row["source_balanced_mae_mps"]) <= best_mae + margin
    ]
    selected = min(
        shortlist,
        key=lambda row: (
            -float(row["lins_concordance_correlation"]),
            int(row["feature_count"]),
            float(row["source_balanced_mae_mps"]),
            float(row["rmse_mps"]),
            float(row["ridge"]),
            int(row["spline_knot_count"]),
            float(row["spline_smoothing"]),
            float(row["blend_weight"]),
            float(row["variance_correction_cap_mps"]),
            float(row["variance_expansion_factor"]),
            int(row["extra_trees_min_samples_leaf"]),
            int(row["extra_trees_n_estimators"]),
        ),
    )
    reasons = list(selected["qualification_reasons"])

    report = {
        "development_tracks": len(development),
        "development_sources": len(sources),
        "development_source_ids": sorted(sources),
        "held_out_test_tracks_used_for_selection": 0,
        "held_out_test_sources_used_for_selection": 0,
        "baseline_metrics": baseline_metrics,
        "unfiltered_baseline_metrics": unfiltered_baseline_metrics,
        "reliability_gates": dict(RELIABILITY_GATES),
        "qualified_candidate_count": len(qualified_candidates),
        "candidate_results": sorted(
            candidate_results,
            key=lambda row: (
                float(row["source_balanced_mae_mps"]),
                int(row["feature_count"]),
                float(row["ridge"]),
            ),
        ),
        "selected_candidate": selected["candidate"],
        "selected_prediction_mode": selected["prediction_mode"],
        "selected_feature_names": selected["feature_names"],
        "selected_ridge": selected["ridge"],
        "selected_spline_knot_count": selected["spline_knot_count"],
        "selected_spline_smoothing": selected["spline_smoothing"],
        "selected_spline_quality_ridge": selected["spline_quality_ridge"],
        "selected_blend_proxy_feature": selected["blend_proxy_feature"],
        "selected_blend_weight": selected["blend_weight"],
        "selected_variance_expansion_factor": selected[
            "variance_expansion_factor"
        ],
        "selected_variance_correction_cap_mps": selected[
            "variance_correction_cap_mps"
        ],
        "selected_extra_trees_n_estimators": selected[
            "extra_trees_n_estimators"
        ],
        "selected_extra_trees_min_samples_leaf": selected[
            "extra_trees_min_samples_leaf"
        ],
        "selected_metrics": {
            key: value
            for key, value in selected.items()
            if key
            not in {
                "candidate",
                "prediction_mode",
                "feature_names",
                "feature_count",
                "ridge",
                "spline_knot_count",
                "spline_smoothing",
                "spline_quality_ridge",
                "blend_proxy_feature",
                "blend_weight",
                "variance_expansion_factor",
                "variance_correction_cap_mps",
                "extra_trees_n_estimators",
                "extra_trees_min_samples_leaf",
                "qualification_reasons",
                "calibration_qualified",
            }
        },
        "baseline_improvement_fraction": float(
            selected["baseline_improvement_fraction"]
        ),
        "conformal_coverage": float(
            CROSS_VALIDATION_SETTINGS["conformal_coverage"]
        ),
        "conformal_absolute_uncertainty_mps": float(
            selected["conformal_absolute_uncertainty_mps"]
        ),
        "calibration_qualified": not reasons,
        "qualification_reasons": reasons,
        "settings": dict(CROSS_VALIDATION_SETTINGS),
    }
    selected_predictions = prediction_cache[
        (
            str(selected["candidate"]),
            float(selected["ridge"]),
            int(selected["spline_knot_count"]),
            float(selected["spline_smoothing"]),
            float(selected["blend_weight"]),
            float(selected["variance_expansion_factor"]),
            float(selected["variance_correction_cap_mps"]),
            int(selected["extra_trees_n_estimators"]),
            int(selected["extra_trees_min_samples_leaf"]),
        )
    ]
    return report, selected_predictions


def conformal_absolute_error(
    predictions: Sequence[Dict[str, Any]],
    coverage: float,
) -> float:
    values = sorted(float(row["absolute_error_mps"]) for row in predictions)
    if not values:
        return 0.0
    rank = int(math.ceil((len(values) + 1) * coverage)) - 1
    return float(values[min(max(rank, 0), len(values) - 1)])


def prediction_from_model(features: TrackFeatures, model: Dict[str, Any]) -> Dict[str, Any]:
    reason = base_rejection_reason(features)
    if not reason:
        reason = reliability_gate_reason(features)
    feature_names = list(model["feature_names"])
    raw_vector = features.model_vector(feature_names)
    centre = np.asarray(model["feature_centre"], dtype=float)
    scale = np.asarray(model["feature_scale"], dtype=float)
    standard = (raw_vector - centre) / scale
    design = np.concatenate([[1.0], standard])
    coefficients = np.asarray(model["coefficients"], dtype=float)
    variance_derivative = 1.0
    tree_prediction_dispersion = 0.0
    if model.get("prediction_mode") == "direct_proxy":
        prediction = direct_proxy_value(features, str(model["direct_proxy_name"]))
    elif model.get("prediction_mode") in {
        "extra_trees_regression",
        "extra_trees_bounded_variance",
    }:
        prediction, tree_prediction_dispersion = extra_trees_prediction(
            features,
            model,
        )
        design = np.empty(0, dtype=float)
        if model.get("prediction_mode") == "extra_trees_bounded_variance":
            prediction, variance_derivative = bounded_variance_prediction(
                prediction,
                float(model["variance_prediction_centre_mps"]),
                float(model["variance_target_centre_mps"]),
                float(model["variance_expansion_factor"]),
                float(model["variance_correction_cap_mps"]),
            )
    elif model.get("prediction_mode") in {
        "monotonic_spline_gam",
        "monotonic_spline_physics_blend",
        "bounded_variance_calibration",
    }:
        design = monotonic_spline_prediction_design(features, model)
        spline_prediction = float(design @ coefficients)
        if model.get("prediction_mode") in {
            "monotonic_spline_physics_blend",
            "bounded_variance_calibration",
        }:
            blend_weight = float(model.get("blend_weight", 0.0))
            blend_proxy_feature = str(model.get("blend_proxy_feature", ""))
            proxy_prediction = float(getattr(features, blend_proxy_feature))
            prediction = float(
                (1.0 - blend_weight) * spline_prediction
                + blend_weight * proxy_prediction
            )
            if model.get("prediction_mode") == (
                "bounded_variance_calibration"
            ):
                prediction, variance_derivative = (
                    bounded_variance_prediction(
                        prediction,
                        float(model["variance_prediction_centre_mps"]),
                        float(model["variance_target_centre_mps"]),
                        float(model["variance_expansion_factor"]),
                        float(model["variance_correction_cap_mps"]),
                    )
                )
        else:
            prediction = spline_prediction
    else:
        prediction = float(design @ coefficients)
    lower = np.asarray(model["feature_lower_bound"], dtype=float)
    upper = np.asarray(model["feature_upper_bound"], dtype=float)
    outside = (raw_vector < lower) | (raw_vector > upper)
    if not reason and not truthy(model.get("calibration_qualified"), False):
        reason = "calibration_not_qualified"
    if not reason and bool(np.any(outside)):
        reason = "out_of_calibration_distribution"
    covariance = np.asarray(model["covariance_basis"], dtype=float)
    residual_sigma = float(model["residual_sigma_mps"])
    leverage = (
        max(0.0, float(design @ covariance @ design))
        if covariance.shape == (len(design), len(design))
        else 0.0
    )
    if model.get("prediction_mode") in {
        "monotonic_spline_physics_blend",
        "bounded_variance_calibration",
    }:
        spline_weight = 1.0 - float(model.get("blend_weight", 0.0))
        leverage *= spline_weight * spline_weight
    leverage *= variance_derivative * variance_derivative
    parametric_uncertainty = max(
        residual_sigma * math.sqrt(1.0 + leverage),
        tree_prediction_dispersion / math.sqrt(
            max(float(model.get("extra_trees_n_estimators", 1)), 1.0)
        ),
    )
    uncertainty = max(
        parametric_uncertainty,
        float(model.get("conformal_absolute_uncertainty_mps", 0.0)),
    )
    relative = uncertainty / max(abs(prediction), 1e-9)
    gates = model.get("gates", BASE_GATES)
    maximum_absolute_uncertainty = safe_float(
        model.get("maximum_absolute_uncertainty_mps")
    )
    if (
        not reason
        and maximum_absolute_uncertainty is not None
        and uncertainty > maximum_absolute_uncertainty
    ):
        reason = "absolute_uncertainty_too_high"
    if not reason and prediction < float(gates["minimum_predicted_speed_mps"]):
        reason = "predicted_speed_too_low"
    if not reason and prediction > float(gates["maximum_predicted_speed_mps"]):
        reason = "predicted_speed_too_high"
    relative_limit = safe_float(model.get("relative_uncertainty_limit"))
    if not reason and relative_limit is not None and relative > relative_limit:
        reason = "relative_uncertainty_too_high"
    status = "rejected" if reason else "valid"
    return {
        "estimated_speed_mps": prediction if status == "valid" else None,
        "model_speed_before_reliability_gate_mps": prediction,
        "speed_uncertainty_mps": uncertainty,
        "relative_uncertainty": relative,
        "speed_status": status,
        "reject_reason": reason,
    }


def metric_summary(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(rows)
    all_numeric = [row for row in rows if safe_float(row.get("model_speed_before_reliability_gate_mps")) is not None]
    valid = [row for row in rows if row.get("speed_status") == "valid" and safe_float(row.get("estimated_speed_mps")) is not None]

    def calculate(selected: Sequence[Dict[str, Any]], prediction_key: str) -> Dict[str, Any]:
        if not selected:
            return {
                "count": 0,
                "mae_mps": None,
                "rmse_mps": None,
                "bias_mps": None,
                "median_absolute_error_mps": None,
                "within_0_25_mps": None,
                "within_0_50_mps": None,
                **agreement_metrics(np.asarray([]), np.asarray([])),
            }
        truth = np.asarray([float(row["ground_truth_speed_mps"]) for row in selected], dtype=float)
        predicted = np.asarray([float(row[prediction_key]) for row in selected], dtype=float)
        error = predicted - truth
        absolute = np.abs(error)
        return {
            "count": len(selected),
            "mae_mps": float(np.mean(absolute)),
            "rmse_mps": float(math.sqrt(float(np.mean(error * error)))),
            "bias_mps": float(np.mean(error)),
            "median_absolute_error_mps": float(np.median(absolute)),
            "within_0_25_mps": float(np.mean(absolute <= 0.25)),
            "within_0_50_mps": float(np.mean(absolute <= 0.50)),
            **agreement_metrics(truth, predicted),
        }

    rejection_counts: Dict[str, int] = defaultdict(int)
    for row in rows:
        if row.get("speed_status") != "valid":
            rejection_counts[str(row.get("reject_reason") or "unknown")] += 1
    return {
        "total_labelled_tracks": total,
        "valid_reliable_tracks": len(valid),
        "coverage": len(valid) / total if total else 0.0,
        "all_model_eligible_metrics": calculate(all_numeric, "model_speed_before_reliability_gate_mps"),
        "reliable_metrics": calculate(valid, "estimated_speed_mps"),
        "rejection_counts": dict(sorted(rejection_counts.items())),
    }


def evaluate_feature_rows(rows: Sequence[Dict[str, Any]], model: Dict[str, Any]) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for row in rows:
        features: TrackFeatures = row["features"]
        prediction = prediction_from_model(features, model)
        result = {
            "source_id": row["source_id"],
            "split": row["split"],
            "prediction_track_id": row["prediction_track_id"],
            "ground_truth_track_id": row.get("ground_truth_track_id", ""),
            "ground_truth_speed_mps": row["ground_truth_speed_mps"],
            **prediction,
            **features_to_row(features),
        }
        model_value = safe_float(prediction.get("model_speed_before_reliability_gate_mps"))
        if model_value is not None:
            result["error_before_reliability_gate_mps"] = model_value - float(row["ground_truth_speed_mps"])
            result["absolute_error_before_reliability_gate_mps"] = abs(result["error_before_reliability_gate_mps"])
        if prediction["speed_status"] == "valid":
            result["error_mps"] = float(prediction["estimated_speed_mps"]) - float(row["ground_truth_speed_mps"])
            result["absolute_error_mps"] = abs(result["error_mps"])
        output.append(result)
    return output


def mode_fit(manifest_csv: str, model_json: str, output_dir: str) -> None:
    rows = load_manifest_feature_rows(manifest_csv)
    development = [
        row
        for row in rows
        if row["split"] in {"train", "validation"}
        and base_rejection_reason(row["features"]) == ""
    ]
    held_out_test = [
        row
        for row in rows
        if row["split"] == "test" and base_rejection_reason(row["features"]) == ""
    ]
    cross_validation, cross_validation_predictions = select_cross_validated_model(
        development
    )
    prediction_mode = str(cross_validation["selected_prediction_mode"])
    selected_candidate = str(cross_validation["selected_candidate"])
    feature_names = list(cross_validation["selected_feature_names"])
    ridge = float(cross_validation["selected_ridge"])
    spline_knot_count = int(
        cross_validation.get("selected_spline_knot_count", 0)
    )
    spline_smoothing = float(
        cross_validation.get("selected_spline_smoothing", 0.0)
    )
    spline_quality_ridge = float(
        cross_validation.get("selected_spline_quality_ridge", 0.0)
    )
    blend_proxy_feature = str(
        cross_validation.get("selected_blend_proxy_feature", "")
    )
    blend_weight = float(
        cross_validation.get("selected_blend_weight", 0.0)
    )
    variance_expansion_factor = float(
        cross_validation.get("selected_variance_expansion_factor", 1.0)
    )
    variance_correction_cap_mps = float(
        cross_validation.get(
            "selected_variance_correction_cap_mps",
            0.0,
        )
    )
    extra_trees_n_estimators = int(
        cross_validation.get("selected_extra_trees_n_estimators", 0)
    )
    extra_trees_min_samples_leaf = int(
        cross_validation.get("selected_extra_trees_min_samples_leaf", 0)
    )
    if prediction_mode == "direct_proxy":
        components = fit_direct_proxy_components(
            development,
            selected_candidate,
            feature_names,
        )
    elif prediction_mode in {
        "monotonic_spline_gam",
        "monotonic_spline_physics_blend",
        "bounded_variance_calibration",
    }:
        settings = MONOTONIC_SPLINE_CANDIDATES.get(
            selected_candidate,
            {
                "primary_feature": "compensated_robust_speed_proxy_mps",
                "quality_features": [],
            },
        )
        components = fit_monotonic_spline_components(
            development,
            str(settings["primary_feature"]),
            list(settings.get("quality_features", [])),
            spline_knot_count,
            spline_smoothing,
            spline_quality_ridge,
        )
        if prediction_mode in {
            "monotonic_spline_physics_blend",
            "bounded_variance_calibration",
        }:
            components["prediction_mode"] = (
                "monotonic_spline_physics_blend"
            )
            components["blend_proxy_feature"] = blend_proxy_feature
            components["blend_weight"] = blend_weight
            base_fitted = np.asarray(
                [
                    component_prediction(row["features"], components)
                    for row in development
                ],
                dtype=float,
            )
            if prediction_mode == "bounded_variance_calibration":
                prediction_by_row_id = {
                    id(row): float(prediction)
                    for row, prediction in zip(development, base_fitted)
                }
                components["variance_prediction_centre_mps"] = (
                    source_balanced_mean(
                        development,
                        lambda row: prediction_by_row_id[id(row)],
                    )
                )
                components["variance_target_centre_mps"] = (
                    source_balanced_mean(
                        development,
                        lambda row: row["ground_truth_speed_mps"],
                    )
                )
                components["variance_expansion_factor"] = (
                    variance_expansion_factor
                )
                components["variance_correction_cap_mps"] = (
                    variance_correction_cap_mps
                )
                components["prediction_mode"] = prediction_mode
            fitted = np.asarray(
                [
                    component_prediction(row["features"], components)
                    for row in development
                ],
                dtype=float,
            )
            development_target = np.asarray(
                [
                    float(row["ground_truth_speed_mps"])
                    for row in development
                ],
                dtype=float,
            )
            blended_residual_sigma = robust_scale(
                development_target - fitted
            )
            if blended_residual_sigma <= 1e-6:
                blended_residual_sigma = max(
                    float(np.std(development_target - fitted)),
                    0.01,
                )
            components["residual_sigma_mps"] = float(
                blended_residual_sigma
            )
    elif prediction_mode in {
        "extra_trees_regression",
        "extra_trees_bounded_variance",
    }:
        components = fit_extra_trees_components(
            development,
            feature_names,
            extra_trees_n_estimators,
            extra_trees_min_samples_leaf,
        )
        base_fitted = np.asarray(
            [
                component_prediction(row["features"], components)
                for row in development
            ],
            dtype=float,
        )
        if prediction_mode == "extra_trees_bounded_variance":
            prediction_by_row_id = {
                id(row): float(prediction)
                for row, prediction in zip(development, base_fitted)
            }
            components["variance_prediction_centre_mps"] = (
                source_balanced_mean(
                    development,
                    lambda row: prediction_by_row_id[id(row)],
                )
            )
            components["variance_target_centre_mps"] = source_balanced_mean(
                development,
                lambda row: row["ground_truth_speed_mps"],
            )
            components["variance_expansion_factor"] = (
                variance_expansion_factor
            )
            components["variance_correction_cap_mps"] = (
                variance_correction_cap_mps
            )
            components["prediction_mode"] = prediction_mode
        fitted = np.asarray(
            [
                component_prediction(row["features"], components)
                for row in development
            ],
            dtype=float,
        )
        target = np.asarray(
            [float(row["ground_truth_speed_mps"]) for row in development],
            dtype=float,
        )
        fitted_residual_sigma = robust_scale(target - fitted)
        components["residual_sigma_mps"] = float(
            fitted_residual_sigma
            if fitted_residual_sigma > 1e-6
            else max(float(np.std(target - fitted)), 0.01)
        )
    else:
        components = fit_model_components(development, feature_names, ridge)
    coverage = float(CROSS_VALIDATION_SETTINGS["conformal_coverage"])
    reliable_cross_validation_predictions = [
        row
        for row in cross_validation_predictions
        if reliability_gate_reason_from_row(row) == ""
    ]
    conformal_uncertainty = conformal_absolute_error(
        reliable_cross_validation_predictions,
        coverage,
    )
    if conformal_uncertainty > float(
        CROSS_VALIDATION_SETTINGS["maximum_conformal_absolute_uncertainty_mps"]
    ):
        cross_validation["calibration_qualified"] = False
        if "cross_validated_uncertainty_too_high" not in cross_validation[
            "qualification_reasons"
        ]:
            cross_validation["qualification_reasons"].append(
                "cross_validated_uncertainty_too_high"
            )
    cross_validation["conformal_coverage"] = coverage
    cross_validation["conformal_absolute_uncertainty_mps"] = conformal_uncertainty
    development_sources = {str(row["source_id"]) for row in development}
    held_out_test_sources = {str(row["source_id"]) for row in held_out_test}
    method_by_mode = {
        "direct_proxy": "source_grouped_selected_direct_physics_proxy",
        "extra_trees_regression": (
            "source_grouped_selected_extra_trees_regression"
        ),
        "extra_trees_bounded_variance": (
            "source_grouped_selected_extra_trees_bounded_variance"
        ),
        "bounded_variance_calibration": (
            "source_grouped_selected_bounded_variance_calibration"
        ),
        "monotonic_spline_physics_blend": (
            "source_grouped_selected_monotonic_spline_physics_blend"
        ),
        "monotonic_spline_gam": (
            "source_grouped_selected_monotonic_spline_gam"
        ),
    }
    model = {
        "schema": MODEL_SCHEMA,
        "release_id": RELEASE_ID,
        "method": method_by_mode.get(
            prediction_mode,
            "source_grouped_selected_source_balanced_huber_ridge",
        ),
        "speed_interpretation": (
            "scene motion compensated bbox derived planar pedestrian speed for "
            "CROWD algorithm selected crossing tracks in metres per second"
        ),
        "calibration_camera": WAYMO_CALIBRATION_CAMERA,
        "calibration_target": WAYMO_CALIBRATION_TARGET,
        "production_inputs": [
            "all class bbox CSV",
            "BoT SORT track ID",
            "frame number",
            "video FPS",
            "aspect ratio",
        ],
        "camera_motion_modelled_explicitly": False,
        "camera_motion_estimated_from_bbox_consensus": True,
        "scene_motion_settings": dict(SCENE_MOTION_SETTINGS),
        "source_context_settings": dict(SOURCE_CONTEXT_SETTINGS),
        "source_context_is_label_free": True,
        "source_context_population": (
            "all base eligible pedestrian tracks in the same bbox CSV"
        ),
        "reliability_gates": dict(RELIABILITY_GATES),
        "compensated_proxy_disagreement_role": "diagnostic_only",
        "compensated_proxy_disagreement_used_as_rejection_gate": False,
        "selected_candidate": selected_candidate,
        "prediction_mode": prediction_mode,
        "direct_proxy_name": components.get("direct_proxy_name"),
        "selected_ridge": ridge,
        "selected_spline_knot_count": spline_knot_count,
        "selected_spline_smoothing": spline_smoothing,
        "selected_spline_quality_ridge": spline_quality_ridge,
        "blend_proxy_feature": components.get("blend_proxy_feature", ""),
        "blend_weight": float(components.get("blend_weight", 0.0)),
        "variance_expansion_factor": float(
            components.get("variance_expansion_factor", 1.0)
        ),
        "variance_correction_cap_mps": float(
            components.get("variance_correction_cap_mps", 0.0)
        ),
        "variance_prediction_centre_mps": safe_float(
            components.get("variance_prediction_centre_mps")
        ),
        "variance_target_centre_mps": safe_float(
            components.get("variance_target_centre_mps")
        ),
        "extra_trees_n_estimators": int(
            components.get("extra_trees_n_estimators", 0)
        ),
        "extra_trees_min_samples_leaf": int(
            components.get("extra_trees_min_samples_leaf", 0)
        ),
        "extra_trees_random_seed": int(
            components.get("extra_trees_random_seed", DEFAULT_RANDOM_SEED)
        ),
        "extra_trees": list(components.get("extra_trees", [])),
        "feature_names": feature_names,
        "feature_centre": components["feature_centre"].tolist(),
        "feature_scale": components["feature_scale"].tolist(),
        "coefficients": components["coefficients"].tolist(),
        "covariance_basis": components["covariance_basis"].tolist(),
        "feature_lower_bound": components["feature_lower_bound"].tolist(),
        "feature_upper_bound": components["feature_upper_bound"].tolist(),
        "residual_sigma_mps": float(components["residual_sigma_mps"]),
        "primary_feature": components.get("primary_feature"),
        "quality_features": components.get("quality_features", []),
        "spline_knots": (
            components["spline_knots"].tolist()
            if components.get("spline_knots") is not None
            else []
        ),
        "quality_centre": (
            components["quality_centre"].tolist()
            if components.get("quality_centre") is not None
            else []
        ),
        "quality_scale": (
            components["quality_scale"].tolist()
            if components.get("quality_scale") is not None
            else []
        ),
        "conformal_coverage": coverage,
        "conformal_absolute_uncertainty_mps": conformal_uncertainty,
        "relative_uncertainty_limit": DEFAULT_RELATIVE_UNCERTAINTY_LIMIT,
        "relative_uncertainty_used_as_rejection_gate": False,
        "maximum_absolute_uncertainty_mps": float(
            CROSS_VALIDATION_SETTINGS[
                "maximum_conformal_absolute_uncertainty_mps"
            ]
        ),
        "gates": BASE_GATES,
        "calibration_qualified": bool(cross_validation["calibration_qualified"]),
        "qualification_reasons": list(cross_validation["qualification_reasons"]),
        "external_test_passed": False,
        "external_test_reasons": ["held_out_test_not_evaluated"],
        "development_tracks": len(development),
        "development_sources": len(development_sources),
        "development_source_ids": sorted(development_sources),
        "held_out_test_tracks": len(held_out_test),
        "held_out_test_sources": len(held_out_test_sources),
        "held_out_test_source_ids": sorted(held_out_test_sources),
        "cross_validation_summary": {
            "baseline_metrics": cross_validation["baseline_metrics"],
            "selected_metrics": cross_validation["selected_metrics"],
            "baseline_improvement_fraction": cross_validation[
                "baseline_improvement_fraction"
            ],
        },
        "manifest_path": str(Path(manifest_csv).expanduser().resolve()),
    }
    write_json(model_json, model)
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    evaluated = evaluate_feature_rows(development, model)
    write_dict_csv(str(output_path / "fit_predictions.csv"), evaluated)
    write_dict_csv(
        str(output_path / "cross_validation_predictions.csv"),
        cross_validation_predictions,
    )
    write_json(
        str(output_path / "cross_validation_report.json"),
        cross_validation,
    )
    information_diagnostic = write_speed_information_diagnostics(
        cross_validation_predictions,
        output_path,
        "development_source_grouped_cross_validation",
    )
    report = {
        "release_id": RELEASE_ID,
        "model_path": str(Path(model_json).expanduser().resolve()),
        "calibration_qualified": model["calibration_qualified"],
        "qualification_reasons": model["qualification_reasons"],
        "selected_candidate": model["selected_candidate"],
        "selected_prediction_mode": model["prediction_mode"],
        "selected_feature_names": model["feature_names"],
        "selected_ridge": model["selected_ridge"],
        "selected_spline_knot_count": model["selected_spline_knot_count"],
        "selected_spline_smoothing": model["selected_spline_smoothing"],
        "selected_spline_quality_ridge": model[
            "selected_spline_quality_ridge"
        ],
        "selected_blend_proxy_feature": model.get(
            "blend_proxy_feature",
            "",
        ),
        "selected_blend_weight": float(model.get("blend_weight", 0.0)),
        "selected_variance_expansion_factor": float(
            model.get("variance_expansion_factor", 1.0)
        ),
        "selected_variance_correction_cap_mps": float(
            model.get("variance_correction_cap_mps", 0.0)
        ),
        "selected_extra_trees_n_estimators": int(
            model.get("extra_trees_n_estimators", 0)
        ),
        "selected_extra_trees_min_samples_leaf": int(
            model.get("extra_trees_min_samples_leaf", 0)
        ),
        "conformal_absolute_uncertainty_mps": conformal_uncertainty,
        "cross_validation": cross_validation,
        "development_fit_metrics": metric_summary(evaluated),
        "speed_information_diagnostic": information_diagnostic,
        "held_out_test_tracks_not_used": len(held_out_test),
        "held_out_test_sources_not_used": len(held_out_test_sources),
    }
    write_json(str(output_path / "fit_report.json"), report)
    log(f"Model written: {Path(model_json).expanduser().resolve()}")
    log(
        f"Development tracks: {len(development)} from "
        f"{len(development_sources)} sources"
    )
    log(
        f"Selected model: {model['selected_candidate']}, "
        f"mode={model['prediction_mode']}, ridge={ridge:g}, "
        f"knots={spline_knot_count}, smoothing={spline_smoothing:g}, "
        f"blend={blend_weight:g}, "
        f"variance_factor={variance_expansion_factor:g}, "
        f"variance_cap={variance_correction_cap_mps:g} m/s, "
        f"trees={extra_trees_n_estimators}, "
        f"minimum_leaf={extra_trees_min_samples_leaf}, "
        f"features={','.join(feature_names)}"
    )
    log(
        "Calibration qualified: "
        + str(model["calibration_qualified"])
        + (
            ""
            if model["calibration_qualified"]
            else "; " + ", ".join(model["qualification_reasons"])
        )
    )
    selected_metrics = cross_validation["selected_metrics"]

    def format_selected_metric(name: str) -> str:
        value = safe_float(selected_metrics.get(name))
        return f"{value:.3f}" if value is not None else "NA"

    log(
        "Selected source grouped agreement: "
        f"MAE={format_selected_metric('source_balanced_mae_mps')} m/s, "
        f"RMSE={format_selected_metric('rmse_mps')} m/s, "
        f"SD ratio={format_selected_metric('prediction_reference_sd_ratio')}, "
        "calibration slope="
        f"{format_selected_metric('prediction_on_reference_calibration_slope')}, "
        f"Pearson={format_selected_metric('pearson_correlation')}, "
        f"Spearman={format_selected_metric('spearman_correlation')}, "
        "Lin concordance="
        f"{format_selected_metric('lins_concordance_correlation')}"
    )
    log(
        f"Held out test rows left untouched: {len(held_out_test)} from "
        f"{len(held_out_test_sources)} sources"
    )
    log(f"Cross validation report: {output_path / 'cross_validation_report.json'}")
    log(f"Report: {output_path / 'fit_report.json'}")


def load_model(path: str, require_qualified: bool = False) -> Dict[str, Any]:
    model_path = Path(path).expanduser().resolve()
    if not model_path.is_file():
        fail(f"Model JSON does not exist: {model_path}")
    with model_path.open("r", encoding="utf-8") as handle:
        model = json.load(handle)
    if model.get("schema") not in {MODEL_SCHEMA, *LEGACY_MODEL_SCHEMAS}:
        fail(
            f"Unsupported model schema {model.get('schema')!r}; expected "
            f"{MODEL_SCHEMA!r} or a supported legacy schema"
        )
    feature_names = model.get("feature_names")
    if (
        not isinstance(feature_names, list)
        or not feature_names
        or any(name not in MODEL_FEATURES for name in feature_names)
    ):
        fail("Model contains an unsupported feature list")
    if model.get("prediction_mode") in {
        "monotonic_spline_gam",
        "monotonic_spline_physics_blend",
        "bounded_variance_calibration",
    }:
        knots = model.get("spline_knots")
        primary_feature = model.get("primary_feature")
        if (
            not isinstance(knots, list)
            or len(knots) < 2
            or primary_feature not in MODEL_FEATURES
        ):
            fail("Monotonic spline model metadata is incomplete")
    if model.get("prediction_mode") in {
        "bounded_variance_calibration",
        "extra_trees_bounded_variance",
    }:
        variance_values = [
            safe_float(model.get("variance_prediction_centre_mps")),
            safe_float(model.get("variance_target_centre_mps")),
            safe_float(model.get("variance_expansion_factor")),
            safe_float(model.get("variance_correction_cap_mps")),
        ]
        if any(value is None for value in variance_values):
            fail("Bounded variance calibration metadata is incomplete")
    if model.get("prediction_mode") in {
        "extra_trees_regression",
        "extra_trees_bounded_variance",
    }:
        trees = model.get("extra_trees")
        if not isinstance(trees, list) or not trees:
            fail("Extra Trees model metadata is incomplete")
        required_tree_keys = {
            "children_left",
            "children_right",
            "feature",
            "threshold",
            "value",
        }
        if any(
            not isinstance(tree, dict)
            or not required_tree_keys.issubset(tree)
            for tree in trees
        ):
            fail("Extra Trees model contains an invalid tree")
    if require_qualified:
        if not truthy(model.get("calibration_qualified"), False):
            reasons = ", ".join(model.get("qualification_reasons") or ["unknown"])
            fail(
                "The calibration model did not pass source grouped validation and "
                f"cannot be used for CROWD prediction: {reasons}"
            )
        if not truthy(model.get("external_test_passed"), False):
            reasons = ", ".join(model.get("external_test_reasons") or ["unknown"])
            fail(
                "The calibration model has not passed the untouched source grouped "
                f"test and cannot be used for CROWD prediction: {reasons}"
            )
    return model


def external_test_qualification(
    rows: Sequence[Dict[str, Any]],
    metrics: Dict[str, Any],
) -> Dict[str, Any]:
    sources = sorted({str(row["source_id"]) for row in rows})
    all_metrics = metrics["all_model_eligible_metrics"]
    reliable_metrics = metrics["reliable_metrics"]
    reasons: List[str] = []
    if len(rows) < int(EXTERNAL_TEST_SETTINGS["minimum_test_tracks"]):
        reasons.append("too_few_untouched_test_tracks")
    if len(sources) < int(EXTERNAL_TEST_SETTINGS["minimum_test_sources"]):
        reasons.append("too_few_untouched_test_sources")
    if (
        safe_float(all_metrics.get("mae_mps")) is None
        or float(all_metrics["mae_mps"])
        > float(EXTERNAL_TEST_SETTINGS["maximum_unfiltered_mae_mps"])
    ):
        reasons.append("unfiltered_test_mae_too_high")
    if (
        safe_float(all_metrics.get("rmse_mps")) is None
        or float(all_metrics["rmse_mps"])
        > float(EXTERNAL_TEST_SETTINGS["maximum_unfiltered_rmse_mps"])
    ):
        reasons.append("unfiltered_test_rmse_too_high")
    if (
        safe_float(all_metrics.get("within_0_50_mps")) is None
        or float(all_metrics["within_0_50_mps"])
        < float(EXTERNAL_TEST_SETTINGS["minimum_within_0_50_mps"])
    ):
        reasons.append("too_few_test_predictions_within_0_50_mps")
    if float(metrics.get("coverage", 0.0)) < float(
        EXTERNAL_TEST_SETTINGS["minimum_reliable_coverage"]
    ):
        reasons.append("reliable_test_coverage_too_low")
    if (
        safe_float(reliable_metrics.get("mae_mps")) is None
        or float(reliable_metrics["mae_mps"])
        > float(EXTERNAL_TEST_SETTINGS["maximum_reliable_mae_mps"])
    ):
        reasons.append("reliable_test_mae_too_high")
    sd_ratio = safe_float(
        all_metrics.get("prediction_reference_sd_ratio")
    )
    if (
        sd_ratio is None
        or sd_ratio
        < float(
            EXTERNAL_TEST_SETTINGS["minimum_prediction_reference_sd_ratio"]
        )
    ):
        reasons.append("test_prediction_spread_too_narrow")
    elif sd_ratio > float(
        EXTERNAL_TEST_SETTINGS["maximum_prediction_reference_sd_ratio"]
    ):
        reasons.append("test_prediction_spread_too_wide")
    calibration_slope = safe_float(
        all_metrics.get("prediction_on_reference_calibration_slope")
    )
    if (
        calibration_slope is None
        or calibration_slope
        < float(EXTERNAL_TEST_SETTINGS["minimum_calibration_slope"])
    ):
        reasons.append("test_calibration_slope_too_low")
    elif calibration_slope > float(
        EXTERNAL_TEST_SETTINGS["maximum_calibration_slope"]
    ):
        reasons.append("test_calibration_slope_too_high")
    pearson = safe_float(all_metrics.get("pearson_correlation"))
    if pearson is None or pearson < float(
        EXTERNAL_TEST_SETTINGS["minimum_pearson_correlation"]
    ):
        reasons.append("test_pearson_correlation_too_low")
    spearman = safe_float(all_metrics.get("spearman_correlation"))
    if spearman is None or spearman < float(
        EXTERNAL_TEST_SETTINGS["minimum_spearman_correlation"]
    ):
        reasons.append("test_spearman_correlation_too_low")
    concordance = safe_float(
        all_metrics.get("lins_concordance_correlation")
    )
    if concordance is None or concordance < float(
        EXTERNAL_TEST_SETTINGS["minimum_lins_concordance"]
    ):
        reasons.append("test_lins_concordance_too_low")
    return {
        "passed": not reasons,
        "reasons": reasons,
        "test_tracks": len(rows),
        "test_sources": len(sources),
        "test_source_ids": sources,
        "settings": dict(EXTERNAL_TEST_SETTINGS),
    }


def external_test_preflight(
    rows: Sequence[Dict[str, Any]],
    model: Dict[str, Any],
    stage: str,
) -> Dict[str, Any]:
    """Protect untouched labels before any external test metric is calculated."""
    test_sources = {
        str(row.get("source_id", "")).strip()
        for row in rows
        if str(row.get("source_id", "")).strip()
    }
    development_values = model.get("development_source_ids")
    if not isinstance(development_values, list) or not development_values:
        fail(
            "External test preflight failed before metrics were calculated: "
            "candidate model has no development_source_ids"
        )
    development_sources = {
        str(value).strip() for value in development_values if str(value).strip()
    }
    overlap = sorted(test_sources & development_sources)
    reasons: List[str] = []
    if not truthy(model.get("calibration_qualified"), False):
        qualification_reasons = ", ".join(
            str(value)
            for value in model.get("qualification_reasons", [])
            if str(value).strip()
        ) or "unknown"
        reasons.append(f"candidate_not_cross_validated ({qualification_reasons})")
    minimum_tracks = int(EXTERNAL_TEST_SETTINGS["minimum_test_tracks"])
    minimum_sources = int(EXTERNAL_TEST_SETTINGS["minimum_test_sources"])
    if len(rows) < minimum_tracks:
        reasons.append(
            f"too_few_untouched_test_tracks ({len(rows)} < {minimum_tracks})"
        )
    if len(test_sources) < minimum_sources:
        reasons.append(
            "too_few_untouched_test_sources "
            f"({len(test_sources)} < {minimum_sources})"
        )
    if overlap:
        preview = ", ".join(overlap[:10])
        if len(overlap) > 10:
            preview += f", and {len(overlap) - 10} more"
        reasons.append(f"development_source_overlap ({preview})")
    if reasons:
        fail(
            "External test preflight failed before metrics were calculated "
            f"at {stage}: " + "; ".join(reasons)
        )
    return {
        "stage": stage,
        "test_tracks": len(rows),
        "test_sources": len(test_sources),
        "test_source_ids": sorted(test_sources),
        "development_source_overlap": [],
        "calibration_qualified": True,
    }


def mode_evaluate(manifest_csv: str, model_json: str, output_dir: str, selected_split: str) -> None:
    model = load_model(model_json, require_qualified=False)
    raw_preflight: Optional[Dict[str, Any]] = None
    if selected_split == "test":
        raw_test_rows = [
            row
            for row in read_dict_csv(manifest_csv)
            if truthy(row.get("include"), True)
            and str(row.get("split", "")).strip().lower() == "test"
            and str(row.get("source_id", "")).strip()
        ]
        raw_preflight = external_test_preflight(
            raw_test_rows,
            model,
            "manifest rows",
        )
    rows = load_manifest_feature_rows(manifest_csv)
    if selected_split != "all":
        rows = [row for row in rows if row["split"] == selected_split]
    if not rows:
        fail(f"No included rows are available for split {selected_split!r}")
    usable_preflight: Optional[Dict[str, Any]] = None
    if selected_split == "test":
        usable_preflight = external_test_preflight(
            rows,
            model,
            "usable feature rows",
        )
    evaluated = evaluate_feature_rows(rows, model)
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    write_dict_csv(str(output_path / "evaluation_tracks.csv"), evaluated)
    metrics = metric_summary(evaluated)
    information_diagnostic = write_speed_information_diagnostics(
        evaluated,
        output_path,
        (
            "untouched_external_test_audit_only"
            if selected_split == "test"
            else f"{selected_split}_evaluation"
        ),
    )
    report: Dict[str, Any] = {
        "release_id": RELEASE_ID,
        "selected_split": selected_split,
        "model_path": str(Path(model_json).expanduser().resolve()),
        "metrics": metrics,
        "speed_information_diagnostic": information_diagnostic,
    }
    if selected_split == "test":
        report["external_test_preflight"] = {
            "manifest": raw_preflight,
            "usable": usable_preflight,
        }
        qualification = external_test_qualification(rows, metrics)
        report["external_test_qualification"] = qualification
        if truthy(model.get("calibration_qualified"), False) and qualification["passed"]:
            production_model = dict(model)
            production_model["external_test_passed"] = True
            production_model["external_test_reasons"] = []
            production_model["external_test_metrics"] = metrics
            production_model["external_test_source_ids"] = qualification[
                "test_source_ids"
            ]
            production_model_path = output_path / "production_model.json"
            write_json(str(production_model_path), production_model)
            report["production_model_path"] = str(production_model_path)
        else:
            report["production_model_path"] = None
    write_json(str(output_path / "evaluation_report.json"), report)
    log(json.dumps(report["metrics"], indent=2))
    if selected_split == "test":
        qualification = report["external_test_qualification"]
        log(
            "External test passed: "
            + str(qualification["passed"])
            + (
                ""
                if qualification["passed"]
                else "; " + ", ".join(qualification["reasons"])
            )
        )
        if report.get("production_model_path"):
            log(f"Production model: {report['production_model_path']}")
    log(f"Report: {output_path / 'evaluation_report.json'}")


def mode_predict(
    bbox_csv: str,
    fps: float,
    model_json: str,
    output_csv: str,
    source_id: str,
    aspect_ratio: float,
) -> None:
    model = load_model(model_json, require_qualified=True)
    all_rows = load_bbox_csv(bbox_csv, person_only=False)
    person_rows = [row for row in all_rows if row.class_id == PERSON_CLASS_ID]
    tracks = group_tracks(person_rows)
    scene_motion_profile = build_scene_motion_profile(all_rows, fps)
    features_by_track = contextual_track_features(
        tracks,
        fps,
        source_id,
        aspect_ratio,
        scene_motion_profile,
    )
    output: List[Dict[str, Any]] = []
    for track_id in sorted(features_by_track, key=str):
        features = features_by_track[track_id]
        prediction = prediction_from_model(features, model)
        output.append(
            {
                "source_id": source_id,
                "prediction_track_id": track_id,
                "estimated_crossing_speed_mps": prediction["estimated_speed_mps"],
                "estimated_crossing_speed_kph": (
                    float(prediction["estimated_speed_mps"]) * 3.6
                    if prediction["estimated_speed_mps"] is not None
                    else None
                ),
                # Legacy aliases retained so downstream CROWD analysis does
                # not break. Here they mean speed along the road crossing
                # direction, as recorded by speed_interpretation.
                "estimated_speed_mps": prediction["estimated_speed_mps"],
                "estimated_speed_kph": (
                    float(prediction["estimated_speed_mps"]) * 3.6
                    if prediction["estimated_speed_mps"] is not None
                    else None
                ),
                "speed_uncertainty_mps": prediction["speed_uncertainty_mps"],
                "relative_uncertainty": prediction["relative_uncertainty"],
                "speed_status": prediction["speed_status"],
                "reject_reason": prediction["reject_reason"],
                "speed_interpretation": model["speed_interpretation"],
                **features_to_row(features),
            }
        )
    write_dict_csv(output_csv, output)
    valid = sum(row["speed_status"] == "valid" for row in output)
    log(f"Person tracks: {len(tracks)}")
    log(f"Reliable speed estimates: {valid}")
    log(f"Rejected tracks: {len(output) - valid}")
    log(f"Output: {Path(output_csv).expanduser().resolve()}")


def mode_predict_index(index_csv: str, model_json: str, output_dir: str) -> None:
    rows = read_dict_csv(index_csv)
    output_root = Path(output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    combined: List[Dict[str, Any]] = []
    processed = 0
    skipped = 0
    for index, row in enumerate(rows, start=1):
        source_id = str(row.get("source_id", "")).strip() or f"source_{index:05d}"
        bbox_csv = resolve_index_path(
            index_csv,
            str(row.get("bbox_csv") or row.get("prediction_bbox_csv") or ""),
        )
        fps = safe_float(row.get("fps"))
        aspect_ratio = safe_float(row.get("aspect_ratio")) or DEFAULT_ASPECT_RATIO
        if not bbox_csv or fps is None or fps <= 0.0 or not Path(bbox_csv).expanduser().is_file():
            log(f"WARNING: CROWD index row {index} is missing bbox_csv or FPS; skipping")
            skipped += 1
            continue
        safe_source = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in source_id)
        output_csv = output_root / f"{safe_source}_speed.csv"
        mode_predict(bbox_csv, fps, model_json, str(output_csv), source_id, aspect_ratio)
        combined.extend(read_dict_csv(str(output_csv)))
        processed += 1
    if not combined:
        fail("No CROWD index row produced a prediction table")
    combined_path = output_root / "crowd_speed_predictions.csv"
    write_dict_csv(str(combined_path), combined)
    log(f"CROWD batch complete: processed={processed}, skipped={skipped}, tracks={len(combined)}")
    log(f"Combined predictions: {combined_path}")


def metric_track_estimate(
    rows: Sequence[BBoxRow],
    fps: float,
    calibration: GroundCalibration,
) -> Dict[str, Any]:
    if fps <= 0.0 or not math.isfinite(fps):
        fail(f"FPS must be positive, received {fps}")
    cleaned = clean_track(rows)
    base: Dict[str, Any] = {
        "input_rows": len(rows),
        "clean_rows": len(cleaned),
        "mapped_rows": 0,
        "mapping_coverage": 0.0,
        "first_frame": cleaned[0].frame if cleaned else None,
        "last_frame": cleaned[-1].frame if cleaned else None,
        "duration_seconds": 0.0,
        "ground_displacement_m": 0.0,
        "motion_fit_r2": 0.0,
        "coordinate_residual_mad_m": None,
        "pair_speed_median_mps": None,
        "pair_speed_mad_mps": None,
        "split_half_disagreement_mps": None,
        "calibration_rmse_m": calibration.calibration_rmse_m,
        "candidate_speed_mps": None,
        "candidate_speed_kph": None,
        "speed_uncertainty_mps": None,
        "estimated_crossing_speed_mps": None,
        "estimated_crossing_speed_kph": None,
        "estimated_speed_mps": None,
        "estimated_speed_kph": None,
        "speed_status": "rejected",
        "reject_reason": "insufficient_ground_mapped_rows",
        "speed_interpretation": (
            "ground speed projected onto configured crossing axis"
            if calibration.crossing_axis_world is not None
            else "planar ground speed"
        ),
        "calibration_id": calibration.calibration_id,
        "calibration_mode": calibration.mode,
    }
    if len(cleaned) < 2:
        return base

    mapped_frames: List[int] = []
    mapped_points: List[np.ndarray] = []
    for row in cleaned:
        matrix = calibration.homography_for_frame(row.frame)
        if matrix is None:
            continue
        image_point = bbox_footpoint_in_calibration_coordinates(row, calibration)
        world_point = apply_homography(matrix, image_point)
        if world_point is None:
            continue
        mapped_frames.append(row.frame)
        mapped_points.append(world_point)
    base["mapped_rows"] = len(mapped_points)
    base["mapping_coverage"] = len(mapped_points) / max(len(cleaned), 1)
    if len(mapped_points) < 2:
        return base

    frames = np.asarray(mapped_frames, dtype=float)
    time_values = (frames - frames[0]) / fps
    points = np.asarray(mapped_points, dtype=float)
    points[:, 0] = rolling_median(points[:, 0], 3)
    points[:, 1] = rolling_median(points[:, 1], 3)
    duration = float(time_values[-1])
    base["first_frame"] = int(frames[0])
    base["last_frame"] = int(frames[-1])
    base["duration_seconds"] = duration
    if duration <= 0.0:
        base["reject_reason"] = "non_positive_duration"
        return base

    x_slope, _, _, _ = robust_line(time_values, points[:, 0])
    y_slope, _, _, _ = robust_line(time_values, points[:, 1])
    velocity = np.asarray([x_slope, y_slope], dtype=float)
    if calibration.crossing_axis_world is not None:
        direction = calibration.crossing_axis_world
    else:
        velocity_norm = float(np.linalg.norm(velocity))
        if velocity_norm <= 1e-12:
            displacement_vector = points[-1] - points[0]
            displacement_norm = float(np.linalg.norm(displacement_vector))
            direction = (
                displacement_vector / displacement_norm
                if displacement_norm > 1e-12
                else np.asarray([1.0, 0.0], dtype=float)
            )
        else:
            direction = velocity / velocity_norm

    motion_coordinate = points @ direction
    motion_slope, _, motion_r2, coordinate_mad = robust_line(
        time_values,
        motion_coordinate,
    )
    speed = abs(float(motion_slope))
    displacement = abs(float(motion_coordinate[-1] - motion_coordinate[0]))

    delta_time = np.diff(time_values)
    delta_coordinate = np.diff(motion_coordinate)
    pair_mask = (
        (delta_time > 0.0)
        & (delta_time <= METRIC_SPEED_GATES["maximum_pair_gap_seconds"])
    )
    pair_speeds = np.abs(delta_coordinate[pair_mask] / delta_time[pair_mask])
    pair_median = float(np.median(pair_speeds)) if len(pair_speeds) else speed
    pair_mad = robust_scale(pair_speeds) if len(pair_speeds) >= 2 else 0.0

    split_disagreement = 0.0
    if len(time_values) >= 8:
        middle = len(time_values) // 2
        first_slope, _, _, _ = robust_line(
            time_values[:middle],
            motion_coordinate[:middle],
        )
        second_slope, _, _, _ = robust_line(
            time_values[middle:],
            motion_coordinate[middle:],
        )
        split_disagreement = abs(abs(first_slope) - abs(second_slope))

    fit_uncertainty = 2.0 * coordinate_mad / max(duration, 1e-9)
    pair_uncertainty = pair_mad / math.sqrt(max(len(pair_speeds), 1))
    split_uncertainty = 0.50 * split_disagreement
    calibration_uncertainty = (
        2.0 * calibration.calibration_rmse_m / max(duration, 1e-9)
    )
    uncertainty = float(
        math.sqrt(
            fit_uncertainty * fit_uncertainty
            + pair_uncertainty * pair_uncertainty
            + split_uncertainty * split_uncertainty
            + calibration_uncertainty * calibration_uncertainty
        )
    )

    base.update(
        {
            "ground_displacement_m": displacement,
            "motion_fit_r2": motion_r2,
            "coordinate_residual_mad_m": coordinate_mad,
            "pair_speed_median_mps": pair_median,
            "pair_speed_mad_mps": pair_mad,
            "split_half_disagreement_mps": split_disagreement,
            "candidate_speed_mps": speed,
            "candidate_speed_kph": speed * 3.6,
            "speed_uncertainty_mps": uncertainty,
            "motion_axis_world_x": float(direction[0]),
            "motion_axis_world_y": float(direction[1]),
        }
    )
    gates = METRIC_SPEED_GATES
    checks = [
        (len(cleaned) < int(gates["minimum_rows"]), "too_few_rows"),
        (len(mapped_points) < int(gates["minimum_rows"]), "too_few_ground_mapped_rows"),
        (duration < gates["minimum_duration_seconds"], "duration_below_reliable_minimum"),
        (
            base["mapping_coverage"] < gates["minimum_mapping_coverage"],
            "ground_mapping_coverage_too_low",
        ),
        (
            calibration.calibration_rmse_m
            > gates["maximum_calibration_rmse_m"],
            "ground_calibration_error_too_high",
        ),
        (
            displacement < gates["minimum_ground_displacement_m"],
            "insufficient_ground_displacement",
        ),
        (motion_r2 < gates["minimum_motion_fit_r2"], "ground_motion_fit_too_low"),
        (speed < gates["minimum_speed_mps"], "predicted_speed_too_low"),
        (speed > gates["maximum_speed_mps"], "predicted_speed_too_high"),
        (
            uncertainty > gates["maximum_absolute_uncertainty_mps"],
            "speed_uncertainty_too_high",
        ),
    ]
    reason = next((name for rejected, name in checks if rejected), "")
    if not reason:
        base["speed_status"] = "valid"
        base["reject_reason"] = ""
        base["estimated_crossing_speed_mps"] = speed
        base["estimated_crossing_speed_kph"] = speed * 3.6
        base["estimated_speed_mps"] = speed
        base["estimated_speed_kph"] = speed * 3.6
    else:
        base["reject_reason"] = reason
    return base


def mode_predict_metric(
    bbox_csv: str,
    fps: float,
    calibration_json: str,
    output_csv: str,
    source_id: str,
) -> None:
    calibration = load_ground_calibration(calibration_json)
    person_rows = load_bbox_csv(bbox_csv, person_only=True)
    tracks = group_tracks(person_rows)
    output: List[Dict[str, Any]] = []
    for track_id in sorted(tracks, key=str):
        estimate = metric_track_estimate(tracks[track_id], fps, calibration)
        output.append(
            {
                "source_id": source_id,
                "prediction_track_id": track_id,
                **estimate,
            }
        )
    write_dict_csv(output_csv, output)
    valid = sum(row["speed_status"] == "valid" for row in output)
    log(f"Person tracks: {len(tracks)}")
    log(f"Reliable metric speed estimates: {valid}")
    log(f"Rejected tracks: {len(output) - valid}")
    log(f"Output: {Path(output_csv).expanduser().resolve()}")


def mode_predict_metric_index(index_csv: str, output_dir: str) -> None:
    rows = read_dict_csv(index_csv)
    output_root = Path(output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    combined: List[Dict[str, Any]] = []
    processed = 0
    skipped = 0
    for index, row in enumerate(rows, start=1):
        source_id = str(row.get("source_id", "")).strip() or f"source_{index:05d}"
        bbox_csv = resolve_index_path(
            index_csv,
            str(row.get("bbox_csv") or row.get("prediction_bbox_csv") or ""),
        )
        calibration_json = resolve_index_path(
            index_csv,
            str(row.get("calibration_json") or row.get("ground_calibration_json") or ""),
        )
        fps = safe_float(row.get("fps"))
        if (
            not bbox_csv
            or not calibration_json
            or fps is None
            or fps <= 0.0
            or not Path(bbox_csv).expanduser().is_file()
            or not Path(calibration_json).expanduser().is_file()
        ):
            log(
                f"WARNING: metric index row {index} is missing bbox_csv, "
                "calibration_json, or FPS; skipping"
            )
            skipped += 1
            continue
        safe_source = "".join(
            ch if ch.isalnum() or ch in "_-" else "_" for ch in source_id
        )
        output_csv = output_root / f"{safe_source}_metric_speed.csv"
        mode_predict_metric(
            bbox_csv,
            float(fps),
            calibration_json,
            str(output_csv),
            source_id,
        )
        combined.extend(read_dict_csv(str(output_csv)))
        processed += 1
    if not combined:
        fail("No metric index row produced a prediction table")
    combined_path = output_root / "crowd_metric_speed_predictions.csv"
    write_dict_csv(str(combined_path), combined)
    log(
        f"Metric batch complete: processed={processed}, skipped={skipped}, "
        f"tracks={len(combined)}"
    )
    log(f"Combined predictions: {combined_path}")


def mode_predict_relative(
    bbox_csv: str,
    fps: float,
    output_csv: str,
    source_id: str,
    aspect_ratio: float,
) -> None:
    all_rows = load_bbox_csv(bbox_csv, person_only=False)
    person_rows = [row for row in all_rows if row.class_id == PERSON_CLASS_ID]
    tracks = group_tracks(person_rows)
    scene_motion_profile = build_scene_motion_profile(all_rows, fps)
    features_by_track = contextual_track_features(
        tracks,
        fps,
        source_id,
        aspect_ratio,
        scene_motion_profile,
    )
    prepared: List[Tuple[TrackFeatures, float, str]] = []
    for track_id in sorted(features_by_track, key=str):
        features = features_by_track[track_id]
        if features.scene_motion_support >= int(
            SCENE_MOTION_SETTINGS["minimum_reference_tracks"]
        ):
            proxy_values = [
                features.compensated_raw_speed_proxy_mps,
                features.compensated_q_speed_proxy_mps,
                features.compensated_robust_speed_proxy_mps,
            ]
            proxy_role = "scene_motion_compensated_bbox_proxy"
        else:
            proxy_values = [
                features.raw_speed_proxy_mps,
                features.q_speed_proxy_mps,
                features.robust_speed_proxy_mps,
            ]
            proxy_role = "uncompensated_bbox_proxy"
        prepared.append((features, float(np.median(proxy_values)), proxy_role))

    eligible_values = [
        proxy
        for features, proxy, _ in prepared
        if base_rejection_reason(features) == "" and proxy > 0.0
    ]
    reference = float(np.median(eligible_values)) if eligible_values else None
    log_values = np.log(np.maximum(np.asarray(eligible_values, dtype=float), 1e-9))
    log_centre = float(np.median(log_values)) if len(log_values) else None
    log_scale = robust_scale(log_values) if len(log_values) >= 3 else None
    output: List[Dict[str, Any]] = []
    for features, proxy, proxy_role in prepared:
        reason = base_rejection_reason(features)
        if not reason and len(eligible_values) < 3:
            reason = "too_few_comparable_tracks"
        score = proxy / reference if reference is not None and reference > 0.0 else None
        z_score = (
            (math.log(max(proxy, 1e-9)) - float(log_centre)) / float(log_scale)
            if log_centre is not None and log_scale is not None and log_scale > 1e-8
            else None
        )
        category: Optional[str] = None
        if not reason and z_score is not None:
            if z_score <= -0.75:
                category = "slower_within_video"
            elif z_score >= 0.75:
                category = "faster_within_video"
            else:
                category = "typical_within_video"
        output.append(
            {
                "source_id": source_id,
                "prediction_track_id": features.prediction_track_id,
                "bbox_motion_proxy": proxy,
                "relative_motion_index": score,
                "relative_motion_robust_z": z_score,
                "relative_motion_category": category,
                "relative_motion_status": "valid" if not reason else "rejected",
                "reject_reason": reason,
                "proxy_role": proxy_role,
                "speed_interpretation": (
                    "within-video bbox motion only; not metres per second"
                ),
                "comparable_tracks": len(eligible_values),
                **features_to_row(features),
            }
        )
    write_dict_csv(output_csv, output)
    valid = sum(row["relative_motion_status"] == "valid" for row in output)
    log(f"Person tracks: {len(tracks)}")
    log(f"Valid within-video motion scores: {valid}")
    log(f"Output: {Path(output_csv).expanduser().resolve()}")


def mode_self_test_metric() -> None:
    image_points = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    world_points = [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]
    matrix, rmse = fit_homography_from_points(
        image_points,
        world_points,
        "internal test",
    )
    calibration = GroundCalibration(
        calibration_id="internal_test",
        mode="static_pixel_to_ground",
        image_coordinates="normalised",
        frame_width=None,
        frame_height=None,
        camera_is_static=True,
        static_homography=matrix,
        homographies_by_frame={},
        maximum_frame_gap=0,
        crossing_axis_world=np.asarray([1.0, 0.0], dtype=float),
        calibration_rmse_m=max(rmse, 0.01),
    )
    rows = [
        BBoxRow(
            class_id=PERSON_CLASS_ID,
            x=0.20 + 0.01 * frame,
            y=0.60,
            width=0.08,
            height=0.20,
            track_id="test_person",
            confidence=0.95,
            frame=frame,
        )
        for frame in range(30)
    ]
    estimate = metric_track_estimate(rows, 10.0, calibration)
    candidate = safe_float(estimate.get("candidate_speed_mps"))
    if (
        estimate.get("speed_status") != "valid"
        or candidate is None
        or abs(candidate - 1.0) > 0.02
    ):
        fail(f"Internal metric test failed: {estimate}")
    log(
        json.dumps(
            {
                "status": "passed",
                "expected_speed_mps": 1.0,
                "estimated_speed_mps": candidate,
                "speed_uncertainty_mps": estimate["speed_uncertainty_mps"],
            },
            indent=2,
            sort_keys=True,
        )
    )


def write_tracker_yaml(path: str, fps: float) -> Dict[str, Any]:
    settings = dict(CROWD_BOTSORT_SETTINGS)
    settings["track_buffer"] = int(round(CROWD_TRACK_BUFFER_SECONDS * fps))
    yaml_path = Path(path).expanduser().resolve()
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for key, value in settings.items():
        if isinstance(value, bool):
            rendered = "true" if value else "false"
        else:
            rendered = str(value)
        lines.append(f"{key}: {rendered}")
    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return settings


def mode_track_video(video_path: str, output_bbox_csv: str, device: str, tracker_yaml: str) -> None:
    try:
        import cv2
        from ultralytics import YOLO
    except ImportError as exc:
        fail(f"track_video requires opencv-python and ultralytics: {exc}")
    input_path = Path(video_path).expanduser().resolve()
    if not input_path.is_file():
        fail(f"Video does not exist: {input_path}")
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        fail(f"OpenCV could not open video: {input_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if not math.isfinite(fps) or fps <= 0.0:
        capture.release()
        fail("Video does not report a valid FPS")
    output_path = Path(output_bbox_csv).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tracker_path = (
        Path(tracker_yaml).expanduser().resolve()
        if tracker_yaml and tracker_yaml != "-"
        else output_path.with_suffix(".botsort.yaml")
    )
    settings = write_tracker_yaml(str(tracker_path), fps)
    log(f"Video FPS: {fps:.2f}")
    log(f"track_buffer: {settings['track_buffer']} frames ({CROWD_TRACK_BUFFER_SECONDS:.2f} seconds)")
    log(f"YOLO model: {CROWD_YOLO_MODEL}")
    log(f"Tracker: {tracker_path}")
    model = YOLO(CROWD_YOLO_MODEL)
    output_rows: List[Dict[str, Any]] = []
    frame_index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        kwargs: Dict[str, Any] = {
            "source": frame,
            "persist": True,
            "conf": CROWD_YOLO_CONFIDENCE,
            "tracker": str(tracker_path),
            "verbose": False,
        }
        if device and device.lower() not in {"", "auto", "default"}:
            kwargs["device"] = device
        results = model.track(**kwargs)
        if results and results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes
            xywhn = boxes.xywhn.detach().cpu().numpy()
            class_ids = boxes.cls.detach().cpu().numpy()
            confidence = boxes.conf.detach().cpu().numpy()
            track_ids = boxes.id.detach().cpu().numpy()
            for box, class_id, score, track_id in zip(xywhn, class_ids, confidence, track_ids):
                output_rows.append(
                    {
                        "yolo-id": int(round(float(class_id))),
                        "x-center": float(box[0]),
                        "y-center": float(box[1]),
                        "width": float(box[2]),
                        "height": float(box[3]),
                        "unique-id": int(round(float(track_id))),
                        "confidence": float(score),
                        "frame-count": frame_index,
                    }
                )
        frame_index += 1
        if frame_index % 100 == 0:
            log(f"Tracked {frame_index}/{frame_total if frame_total > 0 else '?'} frames")
    capture.release()
    fieldnames = ["yolo-id", "x-center", "y-center", "width", "height", "unique-id", "confidence", "frame-count"]
    write_dict_csv(str(output_path), output_rows, fieldnames)
    write_json(
        str(output_path) + ".audit.json",
        {
            "release_id": RELEASE_ID,
            "video": str(input_path),
            "fps": fps,
            "frames_processed": frame_index,
            "tracked_rows": len(output_rows),
            "model": CROWD_YOLO_MODEL,
            "confidence": CROWD_YOLO_CONFIDENCE,
            "tracker_settings": settings,
            "ultralytics_version": package_version("ultralytics"),
        },
    )
    log(f"Tracked rows: {len(output_rows)}")
    log(f"Output: {output_path}")


def mode_track_index(
    index_csv: str,
    device: str,
    maximum_sequences: int,
    overwrite: bool,
    include_all_sequences: bool = False,
) -> None:
    rows = read_dict_csv(index_csv)
    if maximum_sequences > 0:
        rows = rows[:maximum_sequences]
    completed = 0
    skipped = 0
    for index, row in enumerate(rows, start=1):
        ground_truth_rows = safe_int(row.get("ground_truth_rows"))
        if (
            not include_all_sequences
            and ground_truth_rows is not None
            and ground_truth_rows <= 0
        ):
            log(
                f"Skipping sequence {index}/{len(rows)} with no pedestrian "
                f"ground truth: {row.get('source_id', '')}"
            )
            skipped += 1
            continue
        video_path = resolve_index_path(index_csv, str(row.get("video_path", "")))
        bbox_path = resolve_index_path(
            index_csv,
            str(row.get("prediction_bbox_csv", "")),
        )
        if not video_path or not bbox_path:
            log(f"WARNING: index row {index} has no video_path or prediction_bbox_csv; skipping")
            skipped += 1
            continue
        if Path(bbox_path).expanduser().is_file() and not overwrite:
            log(f"Skipping existing prediction {index}/{len(rows)}: {bbox_path}")
            skipped += 1
            continue
        log(f"Tracking sequence {index}/{len(rows)}: {row.get('source_id', '')}")
        try:
            mode_track_video(video_path, bbox_path, device, "-")
        except SystemExit as exc:
            log(f"WARNING: tracking failed for {row.get('source_id', index)}: {exc}")
            skipped += 1
            continue
        completed += 1
    log(f"Index tracking complete: processed={completed}, skipped={skipped}")


def mode_match_index(index_csv: str, output_manifest_csv: str, maximum_sequences: int) -> None:
    rows = read_dict_csv(index_csv)
    if maximum_sequences > 0:
        rows = rows[:maximum_sequences]
    combined: List[Dict[str, Any]] = []
    processed = 0
    skipped = 0
    for index, row in enumerate(rows, start=1):
        source_id = str(row.get("source_id", "")).strip()
        ground_truth_rows = safe_int(row.get("ground_truth_rows"))
        if ground_truth_rows is not None and ground_truth_rows <= 0:
            log(f"Skipping sequence with no pedestrian ground truth: {source_id or index}")
            skipped += 1
            continue
        bbox_path = resolve_index_path(
            index_csv,
            str(row.get("prediction_bbox_csv", "")),
        )
        gt_path = resolve_index_path(
            index_csv,
            str(row.get("ground_truth_bbox_csv", "")),
        )
        manifest_path = resolve_index_path(
            index_csv,
            str(row.get("manifest_csv", "")),
        )
        fps = safe_float(row.get("fps"))
        aspect_ratio = safe_float(row.get("aspect_ratio")) or DEFAULT_ASPECT_RATIO
        if not manifest_path and bbox_path:
            manifest_path = str(Path(bbox_path).with_name("matched_manifest.csv"))
        if (
            not source_id
            or fps is None
            or fps <= 0.0
            or not Path(bbox_path).expanduser().is_file()
            or not Path(gt_path).expanduser().is_file()
        ):
            log(f"WARNING: sequence {source_id or index} is missing prediction or GT input; skipping")
            skipped += 1
            continue
        log(f"Matching sequence {index}/{len(rows)}: {source_id}")
        try:
            mode_match(source_id, bbox_path, gt_path, fps, manifest_path, "", aspect_ratio)
        except SystemExit as exc:
            log(f"WARNING: matching failed for {source_id}: {exc}")
            skipped += 1
            continue
        combined.extend(read_dict_csv(manifest_path))
        processed += 1
    if not combined:
        fail("No sequence produced matched manifest rows")
    write_dict_csv(output_manifest_csv, combined)
    log(f"Index matching complete: processed={processed}, skipped={skipped}, rows={len(combined)}")
    log(f"Combined manifest: {Path(output_manifest_csv).expanduser().resolve()}")


def waymo_camera_number(name: str, dataset_pb2: Any) -> int:
    key = str(name).strip().upper()
    try:
        return int(getattr(dataset_pb2.CameraName, key))
    except AttributeError:
        accepted = ["FRONT", "FRONT_LEFT", "FRONT_RIGHT", "SIDE_LEFT", "SIDE_RIGHT"]
        fail(f"Unknown Waymo camera {name!r}. Use one of: {', '.join(accepted)}")
    return 0


def require_waymo_calibration_camera(name: str) -> str:
    camera_name = str(name).strip().upper()
    if camera_name != WAYMO_CALIBRATION_CAMERA:
        fail(
            "The legacy bbox feature calibration is defined only for "
            f"Waymo {WAYMO_CALIBRATION_CAMERA}. The map-frame target itself "
            f"is camera independent. Received {name!r}."
        )
    return camera_name


def tfrecord_files(input_path: str) -> List[Path]:
    path = Path(input_path).expanduser().resolve()
    if path.is_file():
        if path.name.startswith("._"):
            fail(
                f"Refusing macOS AppleDouble metadata file: {path}. "
                "Select the corresponding file without the ._ prefix."
            )
        return [path]
    if not path.is_dir():
        fail(f"Waymo TFRecord input does not exist: {path}")
    files: List[Path] = []
    for item in path.iterdir():
        # Check the name before stat/is_file.  Docker Desktop on macOS can
        # expose AppleDouble entries but deny stat access to them.
        if item.name.startswith("."):
            continue
        if not (item.name.endswith(".tfrecord") or ".tfrecord-" in item.name):
            continue
        try:
            if item.is_file():
                files.append(item)
        except (OSError, PermissionError) as exc:
            log(f"WARNING: cannot inspect TFRecord candidate {item}: {exc}")
    files.sort()
    if not files:
        fail(f"No TFRecord files found under {path}")
    return files


def parse_waymo_frame(raw_record: Any, dataset_pb2: Any) -> Any:
    """Decode one TensorFlow record with protobuf versions that require bytes."""
    serialised = raw_record.numpy()
    if not isinstance(serialised, bytes):
        serialised = bytes(serialised)
    frame = dataset_pb2.Frame()
    frame.ParseFromString(serialised)
    return frame


def waymo_source_id_from_filename(path: Path) -> str:
    name = path.name
    marker = "segment-"
    if marker in name:
        name = name[name.index(marker) + len(marker):]
    if "_with_camera_labels" in name:
        name = name.split("_with_camera_labels", 1)[0]
    elif ".tfrecord" in name:
        name = name.split(".tfrecord", 1)[0]
    return "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in name)


def reusable_waymo_index_row(sequence_dir: Path) -> Optional[Dict[str, Any]]:
    audit_path = sequence_dir / "waymo_export_audit.json"
    if not audit_path.is_file():
        return None
    try:
        with audit_path.open("r", encoding="utf-8") as handle:
            audit = json.load(handle)
    except (OSError, ValueError):
        return None
    if audit.get("waymo_crossing_schema") != WAYMO_CROSSING_SCHEMA:
        return None
    index_row = audit.get("index_row")
    if not isinstance(index_row, dict):
        return None
    index_hint = str(sequence_dir.parent / "waymo_sequence_index.csv")
    required_paths = [
        resolve_index_path(index_hint, str(index_row.get("video_path", ""))),
        resolve_index_path(index_hint, str(index_row.get("ground_truth_bbox_csv", ""))),
        resolve_index_path(index_hint, str(index_row.get("crossing_tracks_csv", ""))),
    ]
    if any(not path or not Path(path).is_file() for path in required_paths):
        return None
    return dict(index_row)


def annotate_waymo_ground_truth_rows(
    gt_rows: List[Dict[str, Any]],
    crossing_rows: Sequence[Dict[str, Any]],
    samples_by_frame_and_track: Dict[Tuple[int, str], Dict[str, Any]],
) -> Tuple[int, int]:
    """Apply the crosswalk-axis target and retain velocity audit values."""
    crossing_by_id = {
        str(row["ground_truth_track_id"]): row for row in crossing_rows
    }
    crossing_keys = set(WAYMO_CROSSING_TRACK_FIELDS) - {
        "source_id",
        "ground_truth_track_id",
        "camera_rows",
        "trajectory_rows",
        "rejection_reason",
        "ground_truth_speed_mps",
        "ground_truth_crosswalk_axis_speed_mps",
        "ground_truth_lateral_speed_mps",
        "ground_truth_total_speed_mps",
    }
    for row in gt_rows:
        track_id = str(row.get("ground_truth_track_id", ""))
        frame = safe_int(row.get("frame-count"))
        sample = (
            samples_by_frame_and_track.get((frame, track_id))
            if frame is not None
            else None
        )
        if sample is not None:
            total_speed = sample["instantaneous_total_speed_mps"]
            lateral_speed = sample["instantaneous_lateral_speed_mps"]
            row["instantaneous_speed_mps"] = total_speed
            row["instantaneous_total_speed_mps"] = total_speed
            row["instantaneous_lateral_speed_mps"] = lateral_speed
            row["instantaneous_lateral_velocity_mps"] = sample[
                "instantaneous_lateral_velocity_mps"
            ]
            row["ground_truth_speed_mps"] = 0.0
            row["ground_truth_crosswalk_axis_speed_mps"] = 0.0
            row["ground_truth_lateral_speed_mps"] = lateral_speed
            row["ground_truth_total_speed_mps"] = total_speed
            row["map_x"] = sample["map_x"]
            row["map_y"] = sample["map_y"]
        crossing = crossing_by_id.get(track_id, {})
        for key in crossing_keys:
            row[key] = crossing.get(key, "")
        if truthy(crossing.get("crossing_track"), default=False):
            row["ground_truth_speed_mps"] = crossing["ground_truth_speed_mps"]
            row["ground_truth_crosswalk_axis_speed_mps"] = crossing[
                "ground_truth_crosswalk_axis_speed_mps"
            ]
            row["ground_truth_lateral_speed_mps"] = crossing[
                "ground_truth_lateral_speed_mps"
            ]
            row["ground_truth_total_speed_mps"] = crossing[
                "ground_truth_total_speed_mps"
            ]
        else:
            row["crossing_track"] = 0
            row["crossing_label"] = "not_confirmed"
    crossing_track_count = sum(
        int(truthy(row.get("crossing_track"), default=False))
        for row in crossing_rows
    )
    crossing_gt_row_count = sum(
        int(truthy(row.get("crossing_track"), default=False)) for row in gt_rows
    )
    return crossing_track_count, crossing_gt_row_count


def mode_waymo_relabel_crosswalk_axis(
    input_path: str,
    output_dir: str,
    camera_name: str,
    maximum_segments: int,
    overwrite: bool,
) -> None:
    """Relabel existing Waymo exports without decoding video or rerunning YOLO."""
    camera_name = require_waymo_calibration_camera(camera_name)
    try:
        import tensorflow as tf
        from waymo_open_dataset import dataset_pb2
        from waymo_open_dataset import label_pb2
    except ImportError as exc:
        fail(
            "waymo_relabel_crosswalk_axis requires compatible TensorFlow and Waymo "
            f"Open Dataset installation. Import error: {exc}"
        )
    files = tfrecord_files(input_path)
    output_root = Path(output_dir).expanduser().resolve()
    index_path = output_root / "waymo_sequence_index.csv"
    if not index_path.is_file():
        fail(f"Existing Waymo sequence index does not exist: {index_path}")
    index_rows = read_dict_csv(str(index_path))
    index_by_source = {
        str(row.get("source_id", "")): row
        for row in index_rows
        if str(row.get("source_id", ""))
    }
    camera_number = waymo_camera_number(camera_name, dataset_pb2)
    processed = 0
    skipped_not_exported = 0
    skipped_current = 0
    failures: List[Dict[str, str]] = []
    selected_existing = 0
    for file_index, record_path in enumerate(files, start=1):
        predicted_source_id = waymo_source_id_from_filename(record_path)
        index_row = index_by_source.get(predicted_source_id)
        if index_row is None:
            skipped_not_exported += 1
            continue
        gt_path_text = resolve_index_path(
            str(index_path), str(index_row.get("ground_truth_bbox_csv", ""))
        )
        if not gt_path_text or not Path(gt_path_text).is_file():
            skipped_not_exported += 1
            continue
        if maximum_segments > 0 and selected_existing >= maximum_segments:
            break
        selected_existing += 1
        sequence_dir = Path(gt_path_text).resolve().parent
        audit_path = sequence_dir / "waymo_export_audit.json"
        if not overwrite and audit_path.is_file():
            try:
                with audit_path.open("r", encoding="utf-8") as handle:
                    audit = json.load(handle)
            except (OSError, ValueError):
                audit = {}
            if audit.get("waymo_crossing_schema") == WAYMO_CROSSING_SCHEMA:
                audit_index_row = audit.get("index_row")
                if isinstance(audit_index_row, dict):
                    index_row.update(audit_index_row)
                skipped_current += 1
                log(
                    f"Skipping current crosswalk-axis labels {selected_existing}: "
                    f"{predicted_source_id}"
                )
                continue
        log(
            f"Relabelling Waymo export {selected_existing}: "
            f"{predicted_source_id} ({file_index}/{len(files)} raw files)"
        )
        try:
            gt_rows = read_dict_csv(gt_path_text)
            camera_row_counts: Dict[str, int] = defaultdict(int)
            for row in gt_rows:
                camera_row_counts[str(row.get("ground_truth_track_id", ""))] += 1
            trajectory_samples: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
            samples_by_frame_and_track: Dict[
                Tuple[int, str], Dict[str, Any]
            ] = {}
            crosswalks: List[Dict[str, Any]] = []
            frame_index = 0
            source_id = predicted_source_id
            dataset = tf.data.TFRecordDataset(
                str(record_path), compression_type=""
            )
            for raw_record in dataset:
                frame = parse_waymo_frame(raw_record, dataset_pb2)
                if frame_index == 0:
                    source_id = str(frame.context.name or predicted_source_id)
                if frame.map_features and not crosswalks:
                    crosswalks = waymo_crosswalk_polygons(frame)
                image_proto = next(
                    (
                        image
                        for image in frame.images
                        if int(image.name) == camera_number
                    ),
                    None,
                )
                if image_proto is None:
                    continue
                for laser_label in frame.laser_labels:
                    if int(laser_label.type) != int(
                        label_pb2.Label.TYPE_PEDESTRIAN
                    ):
                        continue
                    if not laser_label.HasField("metadata"):
                        continue
                    map_point = waymo_map_aligned_point(frame, laser_label.box)
                    if map_point is None:
                        continue
                    track_id = str(laser_label.id)
                    lateral_velocity = float(laser_label.metadata.speed_y)
                    total_speed = math.hypot(
                        float(laser_label.metadata.speed_x), lateral_velocity
                    )
                    sample = {
                        "frame-count": frame_index,
                        "timestamp_seconds": (
                            float(frame.timestamp_micros) / 1_000_000.0
                        ),
                        "map_x": map_point[0],
                        "map_y": map_point[1],
                        "map_z": map_point[2],
                        "instantaneous_speed_mps": total_speed,
                        "instantaneous_total_speed_mps": total_speed,
                        "instantaneous_lateral_speed_mps": abs(lateral_velocity),
                        "instantaneous_lateral_velocity_mps": lateral_velocity,
                    }
                    trajectory_samples[track_id].append(sample)
                    samples_by_frame_and_track[(frame_index, track_id)] = sample
                frame_index += 1
            crossing_rows: List[Dict[str, Any]] = []
            for track_id in sorted(trajectory_samples):
                samples = sorted(
                    trajectory_samples[track_id],
                    key=lambda sample: int(sample["frame-count"]),
                )
                crossing_rows.append(
                    classify_waymo_crossing_track(
                        source_id,
                        track_id,
                        samples,
                        crosswalks,
                        camera_row_counts.get(track_id, 0),
                    )
                )
            crossing_track_count, crossing_gt_row_count = (
                annotate_waymo_ground_truth_rows(
                    gt_rows, crossing_rows, samples_by_frame_and_track
                )
            )
            crossing_path = sequence_dir / "waymo_crossing_tracks.csv"
            write_dict_csv(gt_path_text, gt_rows, GROUND_TRUTH_BBOX_FIELDS)
            write_dict_csv(
                str(crossing_path), crossing_rows, WAYMO_CROSSING_TRACK_FIELDS
            )
            index_row["source_id"] = source_id
            index_row["crossing_tracks_csv"] = portable_index_path(crossing_path)
            index_row["frames"] = frame_index
            index_row["ground_truth_rows"] = crossing_gt_row_count
            index_row["ground_truth_tracks"] = crossing_track_count
            index_row["crosswalk_features"] = len(crosswalks)
            index_row["ground_truth_target"] = WAYMO_CALIBRATION_TARGET
            rejection_counts: Dict[str, int] = defaultdict(int)
            for crossing in crossing_rows:
                if not truthy(crossing.get("crossing_track"), default=False):
                    rejection_counts[
                        str(crossing.get("rejection_reason", "unknown"))
                    ] += 1
            write_json(
                audit_path,
                {
                    "release_id": RELEASE_ID,
                    "waymo_crossing_schema": WAYMO_CROSSING_SCHEMA,
                    "input_file": str(record_path),
                    "camera": camera_name,
                    "ground_truth_target": WAYMO_CALIBRATION_TARGET,
                    "relabelled_without_video_or_tracking": True,
                    "settings": dict(WAYMO_CROSSING_SETTINGS),
                    "index_row": index_row,
                    "rejection_counts": dict(sorted(rejection_counts.items())),
                },
            )
            write_dict_csv(
                str(index_path), index_rows, WAYMO_SEQUENCE_INDEX_FIELDS
            )
            processed += 1
            log(
                f"Relabelled {source_id}: confirmed crossings="
                f"{crossing_track_count}, bbox rows={crossing_gt_row_count}"
            )
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            failures.append({"input_file": str(record_path), "error": message})
            log(
                "WARNING: crosswalk-axis relabelling failed for "
                f"{record_path.name}: {message}"
            )
    write_dict_csv(str(index_path), index_rows, WAYMO_SEQUENCE_INDEX_FIELDS)
    summary_path = output_root / "waymo_crosswalk_axis_relabel_summary.json"
    write_json(
        summary_path,
        {
            "release_id": RELEASE_ID,
            "waymo_crossing_schema": WAYMO_CROSSING_SCHEMA,
            "ground_truth_target": WAYMO_CALIBRATION_TARGET,
            "input_path": str(Path(input_path).expanduser().resolve()),
            "existing_exports_selected": selected_existing,
            "exports_relabelled": processed,
            "exports_already_current": skipped_current,
            "raw_files_without_existing_export": skipped_not_exported,
            "failures": failures,
            "settings": dict(WAYMO_CROSSING_SETTINGS),
        },
    )
    log(
        "Waymo crosswalk-axis relabelling complete: "
        f"processed={processed}, current={skipped_current}, "
        f"not_exported={skipped_not_exported}, failures={len(failures)}"
    )
    log(f"Summary: {summary_path}")


def mode_waymo_export(
    input_path: str,
    output_dir: str,
    camera_name: str,
    fps: float,
    maximum_segments: int,
    overwrite: bool,
) -> None:
    camera_name = require_waymo_calibration_camera(camera_name)
    try:
        import cv2
        import tensorflow as tf
        from waymo_open_dataset import dataset_pb2
        from waymo_open_dataset import label_pb2
    except ImportError as exc:
        fail(
            "waymo_export requires a compatible TensorFlow and Waymo Open Dataset installation. "
            f"Run this mode on Linux or Colab if wheels are unavailable on macOS. Import error: {exc}"
        )
    files = tfrecord_files(input_path)
    if maximum_segments > 0:
        files = files[:maximum_segments]
    log(f"Waymo TFRecord files selected: {len(files)}")
    output_root = Path(output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    camera_number = waymo_camera_number(camera_name, dataset_pb2)
    index_rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, str]] = []
    index_path = output_root / "waymo_sequence_index.csv"
    for file_index, record_path in enumerate(files, start=1):
        predicted_source_id = waymo_source_id_from_filename(record_path)
        predicted_sequence_dir = output_root / predicted_source_id
        if not overwrite:
            reusable = reusable_waymo_index_row(predicted_sequence_dir)
            if reusable is not None:
                index_rows.append(reusable)
                write_dict_csv(str(index_path), index_rows, WAYMO_SEQUENCE_INDEX_FIELDS)
                log(
                    f"Skipping completed Waymo export {file_index}/{len(files)}: "
                    f"{predicted_source_id}"
                )
                continue
        log(f"Reading Waymo TFRecord {file_index}/{len(files)}: {record_path.name}")
        writer = None
        source_id = ""
        sequence_dir: Optional[Path] = None
        video_path: Optional[Path] = None
        partial_video_path: Optional[Path] = None
        gt_rows: List[Dict[str, Any]] = []
        trajectory_samples: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        camera_row_counts: Dict[str, int] = defaultdict(int)
        crosswalks: List[Dict[str, Any]] = []
        frame_index = 0
        width = 0
        height = 0
        processing_error: Optional[Exception] = None
        try:
            dataset = tf.data.TFRecordDataset(str(record_path), compression_type="")
            for raw_record in dataset:
                frame = parse_waymo_frame(raw_record, dataset_pb2)
                if not source_id:
                    source_id = str(frame.context.name or predicted_source_id or record_path.stem)
                    safe_source = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in source_id)
                    sequence_dir = output_root / safe_source
                    sequence_dir.mkdir(parents=True, exist_ok=True)
                    video_path = sequence_dir / f"waymo_{camera_name.lower()}.mp4"
                    partial_video_path = sequence_dir / f"waymo_{camera_name.lower()}.partial.mp4"
                    if partial_video_path.is_file():
                        partial_video_path.unlink()
                if frame.map_features and not crosswalks:
                    crosswalks = waymo_crosswalk_polygons(frame)
                image_proto = next(
                    (image for image in frame.images if int(image.name) == camera_number),
                    None,
                )
                labels_proto = next(
                    (labels for labels in frame.camera_labels if int(labels.name) == camera_number),
                    None,
                )
                if image_proto is None:
                    continue
                image_array = cv2.imdecode(
                    np.frombuffer(image_proto.image, dtype=np.uint8),
                    cv2.IMREAD_COLOR,
                )
                if image_array is None:
                    continue
                if writer is None:
                    height, width = image_array.shape[:2]
                    writer = cv2.VideoWriter(
                        str(partial_video_path),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        fps,
                        (width, height),
                    )
                    if not writer.isOpened():
                        fail(f"Could not create Waymo video: {partial_video_path}")
                writer.write(image_array)
                laser_by_id = {str(label.id): label for label in frame.laser_labels}
                current_samples: Dict[str, Dict[str, Any]] = {}
                for laser_id, laser_label in laser_by_id.items():
                    if int(laser_label.type) != int(label_pb2.Label.TYPE_PEDESTRIAN):
                        continue
                    if not laser_label.HasField("metadata"):
                        continue
                    map_point = waymo_map_aligned_point(frame, laser_label.box)
                    if map_point is None:
                        continue
                    lateral_velocity = float(laser_label.metadata.speed_y)
                    instantaneous_total_speed = math.hypot(
                        float(laser_label.metadata.speed_x),
                        lateral_velocity,
                    )
                    instantaneous_lateral_speed = abs(lateral_velocity)
                    sample = {
                        "frame-count": frame_index,
                        "timestamp_seconds": float(frame.timestamp_micros) / 1_000_000.0,
                        "map_x": map_point[0],
                        "map_y": map_point[1],
                        "map_z": map_point[2],
                        # Keep both metadata components for audit. The legacy
                        # target is assigned after map crosswalk classification.
                        "instantaneous_speed_mps": instantaneous_total_speed,
                        "instantaneous_total_speed_mps": instantaneous_total_speed,
                        "instantaneous_lateral_speed_mps": instantaneous_lateral_speed,
                        "instantaneous_lateral_velocity_mps": lateral_velocity,
                    }
                    trajectory_samples[laser_id].append(sample)
                    current_samples[laser_id] = sample
                if labels_proto is not None:
                    for label in labels_proto.labels:
                        if int(label.type) != int(label_pb2.Label.TYPE_PEDESTRIAN):
                            continue
                        laser_id = (
                            str(label.association.laser_object_id)
                            if label.HasField("association")
                            else ""
                        )
                        sample = current_samples.get(laser_id)
                        if sample is None:
                            continue
                        camera_row_counts[laser_id] += 1
                        box = label.box
                        gt_rows.append(
                            {
                                "source_id": source_id,
                                "frame-count": frame_index,
                                "timestamp_micros": int(frame.timestamp_micros),
                                "ground_truth_track_id": laser_id,
                                "camera_track_id": str(label.id),
                                "x-center": float(box.center_x) / width,
                                "y-center": float(box.center_y) / height,
                                "width": float(box.length) / width,
                                "height": float(box.width) / height,
                                "ground_truth_speed_mps": 0.0,
                                "ground_truth_crosswalk_axis_speed_mps": 0.0,
                                "ground_truth_lateral_speed_mps": sample[
                                    "instantaneous_lateral_speed_mps"
                                ],
                                "ground_truth_total_speed_mps": sample[
                                    "instantaneous_total_speed_mps"
                                ],
                                "instantaneous_speed_mps": sample["instantaneous_speed_mps"],
                                "instantaneous_total_speed_mps": sample[
                                    "instantaneous_total_speed_mps"
                                ],
                                "instantaneous_lateral_speed_mps": sample[
                                    "instantaneous_lateral_speed_mps"
                                ],
                                "instantaneous_lateral_velocity_mps": sample[
                                    "instantaneous_lateral_velocity_mps"
                                ],
                                "map_x": sample["map_x"],
                                "map_y": sample["map_y"],
                            }
                        )
                frame_index += 1
        except Exception as exc:
            processing_error = exc
        finally:
            if writer is not None:
                writer.release()
        if processing_error is not None:
            if partial_video_path is not None and partial_video_path.is_file():
                partial_video_path.unlink()
            message = f"{type(processing_error).__name__}: {processing_error}"
            failures.append({"input_file": str(record_path), "error": message})
            log(f"WARNING: failed Waymo export for {record_path.name}: {message}")
            continue
        if not source_id or sequence_dir is None or video_path is None:
            log(f"WARNING: no usable {camera_name} frames in {record_path}")
            continue
        if partial_video_path is None or not partial_video_path.is_file():
            log(f"WARNING: no video was created for {record_path}")
            continue
        crossing_rows: List[Dict[str, Any]] = []
        for track_id in sorted(trajectory_samples):
            samples = sorted(
                trajectory_samples[track_id],
                key=lambda sample: int(sample["frame-count"]),
            )
            crossing_rows.append(
                classify_waymo_crossing_track(
                    source_id,
                    track_id,
                    samples,
                    crosswalks,
                    camera_row_counts.get(track_id, 0),
                )
            )
        crossing_by_id = {
            str(row["ground_truth_track_id"]): row
            for row in crossing_rows
        }
        crossing_keys = set(WAYMO_CROSSING_TRACK_FIELDS) - {
            "source_id",
            "ground_truth_track_id",
            "camera_rows",
            "trajectory_rows",
            "rejection_reason",
            "ground_truth_speed_mps",
            "ground_truth_crosswalk_axis_speed_mps",
            "ground_truth_lateral_speed_mps",
            "ground_truth_total_speed_mps",
        }
        for row in gt_rows:
            crossing = crossing_by_id.get(str(row["ground_truth_track_id"]), {})
            for key in crossing_keys:
                row[key] = crossing.get(key, "")
            if truthy(crossing.get("crossing_track"), default=False):
                row["ground_truth_speed_mps"] = crossing["ground_truth_speed_mps"]
                row["ground_truth_crosswalk_axis_speed_mps"] = crossing[
                    "ground_truth_crosswalk_axis_speed_mps"
                ]
                row["ground_truth_lateral_speed_mps"] = crossing[
                    "ground_truth_lateral_speed_mps"
                ]
                row["ground_truth_total_speed_mps"] = crossing[
                    "ground_truth_total_speed_mps"
                ]
            else:
                row["crossing_track"] = 0
                row["crossing_label"] = "not_confirmed"
        crossing_track_count = sum(int(row["crossing_track"]) for row in crossing_rows)
        camera_visible_track_count = sum(
            1 for row in crossing_rows if int(row.get("camera_rows", 0)) > 0
        )
        crossing_gt_row_count = sum(
            1 for row in gt_rows if truthy(row.get("crossing_track"), default=False)
        )
        partial_video_path.replace(video_path)
        gt_path = sequence_dir / "waymo_ground_truth_bbox.csv"
        write_dict_csv(str(gt_path), gt_rows, GROUND_TRUTH_BBOX_FIELDS)
        crossing_path = sequence_dir / "waymo_crossing_tracks.csv"
        write_dict_csv(str(crossing_path), crossing_rows, WAYMO_CROSSING_TRACK_FIELDS)
        index_row = {
            "source_id": source_id,
            "video_path": portable_index_path(video_path, output_root),
            "ground_truth_bbox_csv": portable_index_path(gt_path, output_root),
            "crossing_tracks_csv": portable_index_path(crossing_path, output_root),
            "fps": fps,
            "aspect_ratio": width / height if height else DEFAULT_ASPECT_RATIO,
            "frames": frame_index,
            "ground_truth_rows": crossing_gt_row_count,
            "ground_truth_tracks": crossing_track_count,
            "all_pedestrian_rows": len(gt_rows),
            "all_pedestrian_tracks": camera_visible_track_count,
            "crosswalk_features": len(crosswalks),
            "ground_truth_target": WAYMO_CALIBRATION_TARGET,
            "prediction_bbox_csv": portable_index_path(
                sequence_dir / "crowd_yolo_botsort_bbox.csv",
                output_root,
            ),
            "manifest_csv": portable_index_path(
                sequence_dir / "matched_manifest.csv",
                output_root,
            ),
        }
        rejection_counts: Dict[str, int] = defaultdict(int)
        for crossing in crossing_rows:
            if not truthy(crossing.get("crossing_track"), default=False):
                rejection_counts[str(crossing.get("rejection_reason", "unknown"))] += 1
        write_json(
            sequence_dir / "waymo_export_audit.json",
            {
                "release_id": RELEASE_ID,
                "waymo_crossing_schema": WAYMO_CROSSING_SCHEMA,
                "input_file": str(record_path),
                "camera": camera_name,
                "ground_truth_target": WAYMO_CALIBRATION_TARGET,
                "settings": dict(WAYMO_CROSSING_SETTINGS),
                "index_row": index_row,
                "rejection_counts": dict(sorted(rejection_counts.items())),
            },
        )
        index_rows.append(index_row)
        write_dict_csv(str(index_path), index_rows, WAYMO_SEQUENCE_INDEX_FIELDS)
        log(
            f"Exported {file_index}/{len(files)}: {source_id}, "
            f"frames={frame_index}, pedestrians={len(crossing_rows)}, "
            f"confirmed crossings={crossing_track_count}"
        )
    write_dict_csv(str(index_path), index_rows, WAYMO_SEQUENCE_INDEX_FIELDS)
    write_json(
        output_root / "waymo_export_summary.json",
        {
            "release_id": RELEASE_ID,
            "waymo_crossing_schema": WAYMO_CROSSING_SCHEMA,
            "ground_truth_target": WAYMO_CALIBRATION_TARGET,
            "input_path": str(Path(input_path).expanduser().resolve()),
            "files_selected": len(files),
            "files_completed": len(index_rows),
            "files_failed": len(failures),
            "failures": failures,
            "confirmed_crossing_tracks": sum(
                int(row.get("ground_truth_tracks", 0)) for row in index_rows
            ),
            "confirmed_crossing_bbox_rows": sum(
                int(row.get("ground_truth_rows", 0)) for row in index_rows
            ),
            "all_pedestrian_tracks": sum(
                int(row.get("all_pedestrian_tracks", 0)) for row in index_rows
            ),
            "settings": dict(WAYMO_CROSSING_SETTINGS),
        },
    )
    log(f"Waymo index: {index_path}")
    log(f"Waymo export failures: {len(failures)}")


def selected_nuscenes_scenes(nusc: Any, scene_argument: str, maximum_scenes: int) -> List[Dict[str, Any]]:
    requested = str(scene_argument).strip()
    if requested.lower() in {"", "all", "*"}:
        scenes = list(nusc.scene)
    else:
        names = {name.strip() for name in requested.split(",") if name.strip()}
        scenes = [scene for scene in nusc.scene if scene.get("name") in names or scene.get("token") in names]
        missing = names - {str(scene.get("name")) for scene in scenes} - {str(scene.get("token")) for scene in scenes}
        if missing:
            fail("nuScenes scenes not found: " + ", ".join(sorted(missing)))
    if maximum_scenes > 0:
        scenes = scenes[:maximum_scenes]
    return scenes


def mode_nuscenes_export(
    dataroot: str,
    version: str,
    output_dir: str,
    scene_argument: str,
    maximum_scenes: int,
) -> None:
    try:
        import cv2
        from nuscenes.nuscenes import NuScenes
        from nuscenes.utils.geometry_utils import BoxVisibility, view_points
    except ImportError as exc:
        fail(f"nuscenes_export requires nuscenes-devkit and opencv-python: {exc}")
    root = Path(dataroot).expanduser().resolve()
    if not root.is_dir():
        fail(f"nuScenes dataroot does not exist: {root}")
    nusc = NuScenes(version=version, dataroot=str(root), verbose=False)
    scenes = selected_nuscenes_scenes(nusc, scene_argument, maximum_scenes)
    if not scenes:
        fail("No nuScenes scenes were selected")
    output_root = Path(output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    index_rows: List[Dict[str, Any]] = []
    for scene_index, scene in enumerate(scenes, start=1):
        source_id = str(scene["name"])
        sequence_dir = output_root / source_id
        sequence_dir.mkdir(parents=True, exist_ok=True)
        video_path = sequence_dir / "nuscenes_cam_front.mp4"
        gt_path = sequence_dir / "nuscenes_ground_truth_bbox.csv"
        writer = None
        gt_rows: List[Dict[str, Any]] = []
        frame_index = 0
        sample_token = str(scene["first_sample_token"])
        width = 0
        height = 0
        while sample_token:
            sample = nusc.get("sample", sample_token)
            sample_data_token = sample["data"].get("CAM_FRONT")
            if not sample_data_token:
                sample_token = str(sample.get("next", ""))
                continue
            sample_data = nusc.get("sample_data", sample_data_token)
            image_path, boxes, intrinsic = nusc.get_sample_data(
                sample_data_token,
                box_vis_level=BoxVisibility.ANY,
            )
            image_array = cv2.imread(str(image_path))
            if image_array is None:
                sample_token = str(sample.get("next", ""))
                continue
            if writer is None:
                height, width = image_array.shape[:2]
                writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 2.0, (width, height))
                if not writer.isOpened():
                    fail(f"Could not create nuScenes video: {video_path}")
            writer.write(image_array)
            for box in boxes:
                if not str(box.name).startswith("human.pedestrian"):
                    continue
                corners_3d = box.corners()
                visible = corners_3d[2, :] > 0.10
                if not bool(np.any(visible)):
                    continue
                projected = view_points(corners_3d[:, visible], np.asarray(intrinsic), normalize=True)
                x1 = max(0.0, float(np.min(projected[0, :])))
                y1 = max(0.0, float(np.min(projected[1, :])))
                x2 = min(float(width), float(np.max(projected[0, :])))
                y2 = min(float(height), float(np.max(projected[1, :])))
                if x2 <= x1 or y2 <= y1:
                    continue
                annotation = nusc.get("sample_annotation", box.token)
                velocity = np.asarray(nusc.box_velocity(box.token), dtype=float)
                finite_velocity = velocity[:2][np.isfinite(velocity[:2])]
                if len(finite_velocity) != 2:
                    continue
                speed = float(np.linalg.norm(velocity[:2]))
                gt_rows.append(
                    {
                        "source_id": source_id,
                        "frame-count": frame_index,
                        "timestamp_micros": int(sample_data["timestamp"]),
                        "ground_truth_track_id": str(annotation["instance_token"]),
                        "camera_track_id": str(box.token),
                        "x-center": ((x1 + x2) / 2.0) / width,
                        "y-center": ((y1 + y2) / 2.0) / height,
                        "width": (x2 - x1) / width,
                        "height": (y2 - y1) / height,
                        "ground_truth_speed_mps": speed,
                    }
                )
            frame_index += 1
            sample_token = str(sample.get("next", ""))
        if writer is not None:
            writer.release()
        write_dict_csv(str(gt_path), gt_rows, GROUND_TRUTH_BBOX_FIELDS)
        gt_track_count = len(
            {
                str(row["ground_truth_track_id"])
                for row in gt_rows
                if str(row.get("ground_truth_track_id", ""))
            }
        )
        index_rows.append(
            {
                "source_id": source_id,
                "video_path": portable_index_path(video_path),
                "ground_truth_bbox_csv": portable_index_path(gt_path),
                "fps": 2.0,
                "aspect_ratio": width / height if height else DEFAULT_ASPECT_RATIO,
                "frames": frame_index,
                "ground_truth_rows": len(gt_rows),
                "ground_truth_tracks": gt_track_count,
                "prediction_bbox_csv": portable_index_path(
                    sequence_dir / "crowd_yolo_botsort_bbox.csv"
                ),
                "manifest_csv": portable_index_path(
                    sequence_dir / "matched_manifest.csv"
                ),
            }
        )
        log(
            f"Exported nuScenes {scene_index}/{len(scenes)}: {source_id}, "
            f"frames={frame_index}, GT rows={len(gt_rows)}, GT tracks={gt_track_count}"
        )
    index_path = output_root / "nuscenes_sequence_index.csv"
    write_dict_csv(str(index_path), index_rows)
    log(f"nuScenes index: {index_path}")


def print_usage() -> None:
    print(
        f"""{RELEASE_ID}

Required production CSV columns:
  yolo-id,x-center,y-center,width,height,unique-id,confidence,frame-count

predict_metric can use a person-only CSV. predict_relative and predict need
all object classes for bbox scene-motion and within-video context estimation.

Commands:
  python3 speed_estimation_harness.py build_info

  python3 speed_estimation_harness.py self_test_metric

  python3 speed_estimation_harness.py make_ground_calibration_template \\
    <static|moving> <frame-width> <frame-height> <output-calibration.json>

  python3 speed_estimation_harness.py validate_ground_calibration \\
    <calibration.json>

  python3 speed_estimation_harness.py predict_metric \\
    <CROWD-bbox.csv> <fps> <calibration.json> <output-speed.csv> \\
    [source-id=CSV-stem]

  For a moving dashcam, calibration.json must provide a per-frame
  image-to-world homography in one common metric coordinate frame.

  python3 speed_estimation_harness.py predict_metric_index \\
    <CROWD-index.csv> <output-directory>

  The metric index requires source_id,bbox_csv,fps,calibration_json.

  python3 speed_estimation_harness.py predict_relative \\
    <CROWD-bbox.csv> <fps> <output-score.csv> \\
    [source-id=CSV-stem] [aspect-ratio={DEFAULT_ASPECT_RATIO:.6f}]

  predict_relative is CSV only. Its output is a within-video motion score,
  never a speed in metres per second.

  python3 speed_estimation_harness.py waymo_export \\
    <Waymo TFRecord file-or-directory> <output-directory> \\
    [camera=FRONT] [fps=10] [max-segments=0] [overwrite=false]

  python3 speed_estimation_harness.py waymo_relabel_crosswalk_axis \\
    <Waymo TFRecord file-or-directory> <existing-export-directory> \\
    [camera=FRONT] [max-existing-segments=0] [overwrite=false]

  This relabel command reuses existing Waymo videos and YOLO plus BoT SORT
  CSV files. It reads TFRecord trajectories and maps only and does not rerun
  video export or tracking.

  python3 speed_estimation_harness.py nuscenes_export \\
    <nuScenes-dataroot> <version> <output-directory> [scenes=all] [max-scenes=0]

  python3 speed_estimation_harness.py track_video \\
    <video.mp4> <output-bbox.csv> [device=auto] [tracker-yaml-or-=auto]

  python3 speed_estimation_harness.py track_index \\
    <Waymo-or-nuScenes-index.csv> [device=auto] [max-sequences=0] \
    [overwrite=false] [include-all-sequences=false]

  python3 speed_estimation_harness.py calibrate_waymo_pipeline \
    <training-index.csv> <validation-index.csv> <output-directory>

  python3 speed_estimation_harness.py match \\
    <source-id> <prediction-bbox.csv> <Waymo-GT-bbox.csv> <fps> <output-manifest.csv> \\
    [split=blank] [aspect-ratio={DEFAULT_ASPECT_RATIO:.6f}]

  python3 speed_estimation_harness.py match_index \\
    <Waymo-or-nuScenes-index.csv> <combined-manifest.csv> [max-sequences=0]

  python3 speed_estimation_harness.py merge \\
    <output-manifest.csv> <manifest-1.csv> [manifest-2.csv ...]

  python3 speed_estimation_harness.py assign_splits \\
    <input-manifest.csv> <output-manifest.csv>

  python3 speed_estimation_harness.py make_development_manifest \\
    <examined-manifest.csv> <development-manifest.csv>

  python3 speed_estimation_harness.py make_external_test_manifest \\
    <full-manifest-with-new-sources.csv> <development-manifest.csv> \\
    <external-test-manifest.csv>

  python3 speed_estimation_harness.py fit \\
    <split-manifest.csv> <candidate-model.json> <output-directory>

  python3 speed_estimation_harness.py evaluate \\
    <manifest.csv> <candidate-model.json> <output-directory> [split=test|validation|train|all]

  python3 speed_estimation_harness.py diagnose_speed_information \\
    <cross-validation-or-evaluation.csv> <output-directory> \\
    [role=development_diagnostic]

  This command measures error by reference speed bin, prediction range
  compression, pooled and within-video feature correlations, and the worst
  individual errors. It never changes a fitted model.

  A test evaluation writes <output-directory>/production_model.json only when
  source grouped calibration and the untouched external test both pass.

  python3 speed_estimation_harness.py predict \\
    <CROWD-bbox.csv> <fps> <production-model.json> <output-speed.csv> \\
    [source-id=CSV-stem] [aspect-ratio={DEFAULT_ASPECT_RATIO:.6f}]

  python3 speed_estimation_harness.py predict_index \\
    <CROWD-index.csv> <production-model.json> <output-directory>

  python3 speed_estimation_harness.py features \\
    <bbox.csv> <fps> <output-features.csv> [source-id=CSV-stem] \\
    [aspect-ratio={DEFAULT_ASPECT_RATIO:.6f}]
"""
    )


def require_args(command: str, minimum: int) -> None:
    if len(sys.argv) < minimum:
        print_usage()
        fail(f"{command} received too few arguments")


def main() -> None:
    command = sys.argv[1].strip().lower() if len(sys.argv) > 1 else "help"
    if command in {"help", "-h", "--help"}:
        print_usage()
        return
    if command in {"build_info", "version", "verify_install"}:
        print(json.dumps(build_info(), indent=2, sort_keys=True))
        return
    if command == "self_test_metric":
        mode_self_test_metric()
        return
    if command == "make_ground_calibration_template":
        require_args(command, 6)
        mode_make_ground_calibration_template(
            sys.argv[2],
            int(sys.argv[3]),
            int(sys.argv[4]),
            sys.argv[5],
        )
        return
    if command == "validate_ground_calibration":
        require_args(command, 3)
        mode_validate_ground_calibration(sys.argv[2])
        return
    if command == "predict_metric":
        require_args(command, 6)
        source_id = sys.argv[6] if len(sys.argv) > 6 else Path(sys.argv[2]).stem
        mode_predict_metric(
            sys.argv[2],
            float(sys.argv[3]),
            sys.argv[4],
            sys.argv[5],
            source_id,
        )
        return
    if command == "predict_metric_index":
        require_args(command, 4)
        mode_predict_metric_index(sys.argv[2], sys.argv[3])
        return
    if command == "predict_relative":
        require_args(command, 5)
        source_id = sys.argv[5] if len(sys.argv) > 5 else Path(sys.argv[2]).stem
        aspect_ratio = float(sys.argv[6]) if len(sys.argv) > 6 else DEFAULT_ASPECT_RATIO
        mode_predict_relative(
            sys.argv[2],
            float(sys.argv[3]),
            sys.argv[4],
            source_id,
            aspect_ratio,
        )
        return
    if command == "features":
        require_args(command, 5)
        source_id = sys.argv[5] if len(sys.argv) > 5 else Path(sys.argv[2]).stem
        aspect_ratio = float(sys.argv[6]) if len(sys.argv) > 6 else DEFAULT_ASPECT_RATIO
        mode_features(sys.argv[2], float(sys.argv[3]), sys.argv[4], source_id, aspect_ratio)
        return
    if command == "waymo_export":
        require_args(command, 4)
        camera = sys.argv[4] if len(sys.argv) > 4 else "FRONT"
        fps = float(sys.argv[5]) if len(sys.argv) > 5 else DEFAULT_WAYMO_FPS
        maximum = int(sys.argv[6]) if len(sys.argv) > 6 else 0
        overwrite = truthy(sys.argv[7], False) if len(sys.argv) > 7 else False
        mode_waymo_export(sys.argv[2], sys.argv[3], camera, fps, maximum, overwrite)
        return
    if command == "waymo_relabel_crosswalk_axis":
        require_args(command, 4)
        camera = sys.argv[4] if len(sys.argv) > 4 else WAYMO_CALIBRATION_CAMERA
        maximum = int(sys.argv[5]) if len(sys.argv) > 5 else 0
        overwrite = truthy(sys.argv[6], False) if len(sys.argv) > 6 else False
        mode_waymo_relabel_crosswalk_axis(
            sys.argv[2], sys.argv[3], camera, maximum, overwrite
        )
        return
    if command == "nuscenes_export":
        require_args(command, 5)
        scenes = sys.argv[5] if len(sys.argv) > 5 else "all"
        maximum = int(sys.argv[6]) if len(sys.argv) > 6 else 0
        mode_nuscenes_export(sys.argv[2], sys.argv[3], sys.argv[4], scenes, maximum)
        return
    if command == "track_video":
        require_args(command, 4)
        device = sys.argv[4] if len(sys.argv) > 4 else "auto"
        tracker_yaml = sys.argv[5] if len(sys.argv) > 5 else "-"
        mode_track_video(sys.argv[2], sys.argv[3], device, tracker_yaml)
        return
    if command == "track_index":
        require_args(command, 3)
        device = sys.argv[3] if len(sys.argv) > 3 else "auto"
        maximum = int(sys.argv[4]) if len(sys.argv) > 4 else 0
        overwrite = truthy(sys.argv[5], False) if len(sys.argv) > 5 else False
        include_all = truthy(sys.argv[6], False) if len(sys.argv) > 6 else False
        mode_track_index(sys.argv[2], device, maximum, overwrite, include_all)
        return
    if command == "calibrate_waymo_pipeline":
        require_args(command, 5)
        from utils.crossing.waymo_calibration import calibrate_waymo_pipeline

        calibrate_waymo_pipeline(
            sys.argv[2],
            sys.argv[3],
            sys.argv[4],
            sys.modules[__name__],
        )
        return
    if command == "match":
        require_args(command, 7)
        split = sys.argv[7].strip().lower() if len(sys.argv) > 7 else ""
        aspect_ratio = float(sys.argv[8]) if len(sys.argv) > 8 else DEFAULT_ASPECT_RATIO
        mode_match(sys.argv[2], sys.argv[3], sys.argv[4], float(sys.argv[5]), sys.argv[6], split, aspect_ratio)
        return
    if command == "match_index":
        require_args(command, 4)
        maximum = int(sys.argv[4]) if len(sys.argv) > 4 else 0
        mode_match_index(sys.argv[2], sys.argv[3], maximum)
        return
    if command == "merge":
        require_args(command, 4)
        mode_merge(sys.argv[2], sys.argv[3:])
        return
    if command == "assign_splits":
        require_args(command, 4)
        mode_assign_splits(sys.argv[2], sys.argv[3])
        return
    if command == "make_development_manifest":
        require_args(command, 4)
        mode_make_development_manifest(sys.argv[2], sys.argv[3])
        return
    if command == "make_external_test_manifest":
        require_args(command, 5)
        mode_make_external_test_manifest(sys.argv[2], sys.argv[3], sys.argv[4])
        return
    if command == "fit":
        require_args(command, 5)
        mode_fit(sys.argv[2], sys.argv[3], sys.argv[4])
        return
    if command == "evaluate":
        require_args(command, 5)
        selected_split = sys.argv[5].strip().lower() if len(sys.argv) > 5 else "test"
        if selected_split not in {"train", "validation", "test", "all"}:
            fail("evaluate split must be train, validation, test, or all")
        mode_evaluate(sys.argv[2], sys.argv[3], sys.argv[4], selected_split)
        return
    if command == "diagnose_speed_information":
        require_args(command, 4)
        role = (
            sys.argv[4].strip()
            if len(sys.argv) > 4
            else "development_diagnostic"
        )
        mode_diagnose_speed_information(sys.argv[2], sys.argv[3], role)
        return
    if command == "predict":
        require_args(command, 6)
        source_id = sys.argv[6] if len(sys.argv) > 6 else Path(sys.argv[2]).stem
        aspect_ratio = float(sys.argv[7]) if len(sys.argv) > 7 else DEFAULT_ASPECT_RATIO
        mode_predict(sys.argv[2], float(sys.argv[3]), sys.argv[4], sys.argv[5], source_id, aspect_ratio)
        return
    if command == "predict_index":
        require_args(command, 5)
        mode_predict_index(sys.argv[2], sys.argv[3], sys.argv[4])
        return
    print_usage()
    fail(f"Unknown command: {command}")


if __name__ == "__main__":
    main()
