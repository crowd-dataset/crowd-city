"""Crossing duration, relative motion, and optional Waymo evaluation.

The production input is a YOLO plus BoT SORT bounding box CSV. CROWD has no
metric road plane or camera pose, so this module reports a within video relative
apparent motion index for CROWD. Waymo metric labels are used only to evaluate
how informative that bounding box measure is.
"""

import csv
import json
import math
import os
import re
import shlex
import subprocess
import sys
import zipfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import polars as pl

import common
from utils.core.grouping import Grouping
from utils.core.metadata import MetaData


METRICS_BUILD_ID = "crowd_bbox_relative_motion_v24_20260822"
metadata_class = MetaData()
grouping_class = Grouping()


class Metrics:
    """Public crossing metrics used by analysis.py."""

    def __init__(self) -> None:
        pass

    @staticmethod
    def _get_config(key: str, default=None):
        try:
            value = common.get_configs(key)
        except Exception:
            return default
        return default if value is None else value

    @staticmethod
    def _as_float(value, default=None):
        try:
            output = float(value)
        except (TypeError, ValueError):
            return default
        return output if math.isfinite(output) else default

    @staticmethod
    def _resolve_fps(df_mapping: pl.DataFrame, video_id: str) -> float | None:
        result = metadata_class.find_values_with_video_id(df_mapping, video_id)
        if result is None:
            return None
        fps = Metrics._as_float(result[17], None)
        return fps if fps is not None and fps > 0 else None

    @staticmethod
    def _track(df: pl.DataFrame, track_id: Any) -> pl.DataFrame:
        track = df.filter(pl.col("unique-id") == track_id).sort("frame-count")
        if track.height:
            return track
        return (
            df.filter(
                pl.col("unique-id").cast(pl.Utf8, strict=False) == str(track_id)
            )
            .sort("frame-count")
        )

    def time_to_cross(
        self,
        dataframe: pl.DataFrame,
        ids: list,
        video_id: str,
        df_mapping: pl.DataFrame,
    ) -> dict:
        """Return observed crossing track duration in seconds."""
        required = {"frame-count", "unique-id"}
        if not ids or not required.issubset(set(dataframe.columns)):
            return {}
        fps = self._resolve_fps(df_mapping, video_id)
        if fps is None:
            return {}

        output: dict = {}
        for track_id in ids:
            track = self._track(dataframe, track_id)
            if track.height < 2:
                continue
            frames = [
                self._as_float(value, None)
                for value in track.get_column("frame-count").to_list()
            ]
            frames = [value for value in frames if value is not None]
            if len(frames) < 2:
                continue
            duration = (max(frames) - min(frames)) / fps
            if duration > 0:
                output[track_id] = float(duration)
        return output

    def calculate_speed_of_crossing(
        self,
        df_mapping: pl.DataFrame,
        df: pl.DataFrame,
        data: dict,
    ):
        """Return relative crossing motion using the legacy result structure.

        The method name is retained because analysis.py and saved result files use
        it. Values are dimensionless within video indices, never metres per
        second.
        """
        if not data or not any(data.values()):
            return None
        source_id = next(iter(data))
        selected_ids = {
            normalise_id(value) for value in data[source_id] if normalise_id(value)
        }
        if not selected_ids:
            return None
        try:
            _video_id, _start, fps_text = str(source_id).rsplit("_", 2)
            fps = float(fps_text)
        except (TypeError, ValueError):
            fps = self._resolve_fps(df_mapping, source_id)
        if fps is None or fps <= 0:
            return None

        rows = bbox_rows_from_polars(df)
        aspect_ratio = safe_float(os.environ.get("CROWD_BBOX_ASPECT_RATIO"))
        if aspect_ratio is None or aspect_ratio <= 0.0:
            aspect_ratio = DEFAULT_ASPECT_RATIO
        relative_rows, _ = predict_relative_bbox_rows(
            rows,
            float(fps),
            str(source_id),
            aspect_ratio,
        )
        person_track_count = len(
            {row.track_id for row in rows if row.class_id == PERSON_CLASS_ID}
        )
        _RELATIVE_MEMORY_CACHE[str(source_id)] = (
            relative_rows,
            person_track_count,
        )
        values = {
            row["prediction_track_id"]: float(row["relative_motion_index"])
            for row in relative_rows
            if normalise_id(row.get("prediction_track_id")) in selected_ids
            and row.get("relative_motion_status") == "valid"
            and safe_float(row.get("relative_motion_index")) is not None
        }
        if not values:
            return None
        return grouping_class.locality_country_wrapper(
            input_dict={str(source_id): values},
            mapping=df_mapping,
        )

    def avg_speed_of_crossing_locality(
        self,
        df_mapping: pl.DataFrame,
        all_speed: dict,
    ):
        """Average the dimensionless relative motion index by locality."""
        del df_mapping
        averages: dict = {}
        complete: dict = {}
        minimum = self._as_float(self._get_config("min_speed_limit", 0), 0)
        maximum = self._as_float(self._get_config("max_speed_limit", 1e20), 1e20)
        for locality, videos in (all_speed or {}).items():
            values = [
                float(value)
                for tracks in videos.values()
                for value in tracks.values()
                if minimum <= float(value) <= maximum
            ]
            if values:
                complete[locality] = values
                averages[locality] = sum(values) / len(values)
        return averages, complete

    def avg_speed_of_crossing_country(
        self,
        df_mapping: pl.DataFrame,
        all_speed: dict,
    ):
        """Average the dimensionless relative motion index by country."""
        grouped: dict[str, list[float]] = {}
        minimum = self._as_float(self._get_config("min_speed_limit", 0), 0)
        maximum = self._as_float(self._get_config("max_speed_limit", 1e20), 1e20)
        for videos in (all_speed or {}).values():
            for video_id, tracks in videos.items():
                result = metadata_class.find_values_with_video_id(df_mapping, video_id)
                if result is None:
                    continue
                key = f"{result[8]}_{result[3]}"
                for value in tracks.values():
                    numeric = self._as_float(value, None)
                    if numeric is not None and minimum <= numeric <= maximum:
                        grouped.setdefault(key, []).append(float(numeric))
        averages = {
            key: sum(values) / len(values)
            for key, values in grouped.items()
            if values
        }
        return averages, grouped

    def time_to_start_cross(
        self,
        df_mapping: pl.DataFrame,
        df: pl.DataFrame,
        data: dict,
        person_id: int = 0,
    ):
        """Estimate the stationary interval immediately before motion starts."""
        del person_id
        if not data or not any(data.values()):
            return None
        required = {"unique-id", "frame-count", "x-center", "height"}
        if not required.issubset(set(df.columns)):
            return None
        source_id = next(iter(data))
        fps = self._resolve_fps(df_mapping, source_id)
        if fps is None:
            return None

        checks_per_second = self._as_float(
            self._get_config("check_per_sec_time", 3),
            3,
        )
        if checks_per_second is None or checks_per_second <= 0:
            return None
        step = max(1, int(round(fps / checks_per_second)))
        durations: dict = {}

        for track_id in data[source_id]:
            track = self._track(df, track_id)
            if track.height <= step:
                continue
            x_values = track.get_column("x-center").cast(
                pl.Float64,
                strict=False,
            ).to_numpy()
            heights = track.get_column("height").cast(
                pl.Float64,
                strict=False,
            ).drop_nulls()
            if not len(x_values) or heights.len() == 0:
                continue
            margin = 0.1 * float(heights.median())
            stable_samples = 0
            for index in range(0, len(x_values) - step, step):
                delta = abs(float(x_values[index + step]) - float(x_values[index]))
                if delta <= margin:
                    stable_samples += 1
                elif stable_samples >= 3:
                    break
                else:
                    stable_samples = 0
            if stable_samples >= 3:
                durations[track_id] = stable_samples

        if not durations:
            return None
        return grouping_class.locality_country_wrapper(
            input_dict={source_id: durations},
            mapping=df_mapping,
        )

    def avg_time_to_start_cross_locality(
        self,
        df_mapping: pl.DataFrame,
        all_time: dict,
    ):
        """Average crossing initiation time by locality."""
        del df_mapping
        averages: dict = {}
        complete: dict = {}
        checks = self._as_float(self._get_config("check_per_sec_time", 3), 3)
        minimum = self._as_float(self._get_config("min_waiting_time", 0), 0)
        maximum = self._as_float(self._get_config("max_waiting_time", 1e20), 1e20)
        for locality, videos in (all_time or {}).items():
            values = [
                float(value) / checks
                for tracks in videos.values()
                if tracks
                for value in tracks.values()
                if minimum <= float(value) / checks <= maximum
            ]
            if values:
                complete[locality] = values
                averages[locality] = sum(values) / len(values)
        return averages, complete

    def avg_time_to_start_cross_country(
        self,
        df_mapping: pl.DataFrame,
        all_time: dict,
    ):
        """Average crossing initiation time by country."""
        grouped: dict[str, list[float]] = {}
        checks = self._as_float(self._get_config("check_per_sec_time", 3), 3)
        minimum = self._as_float(self._get_config("min_waiting_time", 0), 0)
        maximum = self._as_float(self._get_config("max_waiting_time", 1e20), 1e20)
        for videos in (all_time or {}).values():
            for video_id, times in videos.items():
                if not times:
                    continue
                result = metadata_class.find_values_with_video_id(df_mapping, video_id)
                if result is None:
                    continue
                key = f"{result[8]}_{result[3]}"
                for value in times.values():
                    seconds = float(value) / checks
                    if minimum <= seconds <= maximum:
                        grouped.setdefault(key, []).append(seconds)
        averages = {
            key: sum(values) / len(values)
            for key, values in grouped.items()
            if values
        }
        return averages, grouped

REPORT_BUILD_ID = "crowd_waymo_crossing_report_v24_20260822"
PERSON_CLASS_ID = 0
DEFAULT_ASPECT_RATIO = 16.0 / 9.0
DEFAULT_PERSON_HEIGHT_M = 1.70

SCENE_MOTION_SETTINGS: Dict[str, float] = {
    "maximum_pair_gap_frames": 2,
    "window_radius_frames": 3,
    "minimum_reference_tracks": 3,
    "maximum_absolute_x_rate_per_second": 3.0,
    "outlier_mad_multiplier": 4.0,
}

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
}

RELIABILITY_GATES: Dict[str, float] = {
    "minimum_duration_seconds": 2.00,
    "minimum_x_fit_r2": 0.90,
}

ASSOCIATION_REJECTION_REASONS = {
    "too_few_matched_frames",
    "mean_iou_below_minimum",
    "prediction_coverage_below_minimum",
    "ground_truth_coverage_below_minimum",
    "association_too_few_matched_frames",
    "association_mean_iou_too_low",
    "association_prediction_coverage_too_low",
    "association_ground_truth_coverage_too_low",
}

# analysis.py calculates its legacy aggregate fields before producing the new
# report. Reuse those extracted features during the same process instead of
# reading and fitting every bbox track a second time.
_RELATIVE_MEMORY_CACHE: Dict[str, Tuple[List[Dict[str, Any]], int]] = {}


@dataclass(frozen=True)
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
class SceneMotionProfile:
    samples_by_frame: Dict[int, List[Tuple[float, str]]]

    def rate_at(self, frame: float) -> Tuple[float, int]:
        radius = int(SCENE_MOTION_SETTINGS["window_radius_frames"])
        centre_frame = int(round(frame))
        samples: List[Tuple[float, str]] = []
        for candidate in range(centre_frame - radius, centre_frame + radius + 1):
            samples.extend(self.samples_by_frame.get(candidate, []))
        support = len({track_id for _, track_id in samples})
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


@dataclass(frozen=True)
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


def _default_log(message: str) -> None:
    print(message)


def safe_float(value: Any) -> Optional[float]:
    try:
        output = float(value)
    except (TypeError, ValueError):
        return None
    return output if math.isfinite(output) else None


def safe_int(value: Any) -> Optional[int]:
    output = safe_float(value)
    return int(round(output)) if output is not None else None


def truthy(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off", ""}:
        return False
    return default


def normalise_id(value: Any) -> str:
    text = str(value if value is not None else "").strip()
    if not text:
        return ""
    numeric = safe_float(text)
    if numeric is not None and abs(numeric - round(numeric)) < 1e-9:
        return str(int(round(numeric)))
    return text


def safe_name(value: Any) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    return text or "source"


def first_column(
    fieldnames: Sequence[str],
    aliases: Sequence[str],
    required: bool = True,
) -> Optional[str]:
    lowered = {name.strip().lower(): name for name in fieldnames}
    for alias in aliases:
        if alias.lower() in lowered:
            return lowered[alias.lower()]
    if required:
        raise ValueError(
            "CSV is missing a required column. Expected one of: " + ", ".join(aliases)
        )
    return None


def read_dict_csv(path: os.PathLike[str] | str) -> List[Dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_dict_csv(path: os.PathLike[str] | str, rows: Sequence[Mapping[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: os.PathLike[str] | str, value: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def load_bbox_csv(path: os.PathLike[str] | str) -> List[BBoxRow]:
    input_path = Path(path)
    with input_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Bounding box CSV has no header: {input_path}")
        names = reader.fieldnames
        class_col = first_column(names, ["yolo-id", "yolo_id", "class-id", "class_id", "class"])
        x_col = first_column(names, ["x-center", "x_center", "xcentre", "xcenter"])
        y_col = first_column(names, ["y-center", "y_center", "ycentre", "ycenter"])
        width_col = first_column(names, ["width", "bbox_width", "w"])
        height_col = first_column(names, ["height", "bbox_height", "h"])
        track_col = first_column(
            names,
            ["unique-id", "unique_id", "track-id", "track_id", "prediction_track_id"],
        )
        confidence_col = first_column(names, ["confidence", "conf", "score"], required=False)
        frame_col = first_column(
            names,
            ["frame-count", "frame_count", "frame-id", "frame_id", "frame"],
        )
        output: List[BBoxRow] = []
        for raw in reader:
            class_id = safe_int(raw.get(class_col))
            x = safe_float(raw.get(x_col))
            y = safe_float(raw.get(y_col))
            width = safe_float(raw.get(width_col))
            height = safe_float(raw.get(height_col))
            frame = safe_int(raw.get(frame_col))
            track_id = normalise_id(raw.get(track_col))
            confidence = safe_float(raw.get(confidence_col)) if confidence_col else 1.0
            if None in (class_id, x, y, width, height, frame, confidence) or not track_id:
                continue
            if float(width) <= 0.0 or float(height) <= 0.0:
                continue
            output.append(
                BBoxRow(
                    int(class_id),
                    float(x),
                    float(y),
                    float(width),
                    float(height),
                    track_id,
                    float(confidence),
                    int(frame),
                )
            )
    return output


def bbox_rows_from_polars(dataframe: pl.DataFrame) -> List[BBoxRow]:
    """Convert the existing analysis DataFrame without creating a second CSV."""
    names = dataframe.columns
    class_col = first_column(names, ["yolo-id", "yolo_id", "class-id", "class_id", "class"])
    x_col = first_column(names, ["x-center", "x_center", "xcentre", "xcenter"])
    y_col = first_column(names, ["y-center", "y_center", "ycentre", "ycenter"])
    width_col = first_column(names, ["width", "bbox-width", "bbox_width", "w"])
    height_col = first_column(names, ["height", "bbox-height", "bbox_height", "h"])
    track_col = first_column(names, ["unique-id", "unique_id", "track-id", "track_id", "id"])
    frame_col = first_column(names, ["frame-count", "frame_count", "frame", "frame-id", "frame_id"])
    confidence_col = first_column(
        names,
        ["confidence", "conf", "score"],
        required=False,
    )
    output: List[BBoxRow] = []
    for raw in dataframe.to_dicts():
        class_id = safe_int(raw.get(class_col))
        x_value = safe_float(raw.get(x_col))
        y_value = safe_float(raw.get(y_col))
        width = safe_float(raw.get(width_col))
        height = safe_float(raw.get(height_col))
        frame = safe_int(raw.get(frame_col))
        track_id = normalise_id(raw.get(track_col))
        confidence = safe_float(raw.get(confidence_col)) if confidence_col else 1.0
        if None in {class_id, x_value, y_value, width, height, frame} or not track_id:
            continue
        if width <= 0.0 or height <= 0.0:
            continue
        output.append(
            BBoxRow(
                class_id=int(class_id),
                x=float(x_value),
                y=float(y_value),
                width=float(width),
                height=float(height),
                track_id=track_id,
                confidence=float(confidence if confidence is not None else 1.0),
                frame=int(frame),
            )
        )
    return output


def group_tracks(rows: Iterable[BBoxRow]) -> Dict[str, List[BBoxRow]]:
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
        first = float(values[0]) if len(values) else 0.0
        return 0.0, first, 0.0, 0.0
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
    return (
        float(beta[1]),
        float(beta[0]),
        float(max(-1.0, min(1.0, r2))),
        robust_scale(residual),
    )


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
    values = np.asarray(
        [[row.x, row.y, row.width, row.height] for row in ordered], dtype=float
    )
    baseline = np.column_stack(
        [rolling_median(values[:, column], 5) for column in range(4)]
    )
    residual = values - baseline
    keep = np.ones(len(ordered), dtype=bool)
    for column in range(4):
        scale = robust_scale(residual[:, column])
        if scale > 1e-8:
            keep &= np.abs(residual[:, column]) <= 6.0 * scale
    keep[0] = True
    keep[-1] = True
    return [row for index, row in enumerate(ordered) if bool(keep[index])]


def build_scene_motion_profile(rows: Sequence[BBoxRow], fps: float) -> SceneMotionProfile:
    samples_by_frame: Dict[int, List[Tuple[float, str]]] = defaultdict(list)
    grouped: Dict[Tuple[int, str], List[BBoxRow]] = defaultdict(list)
    for row in rows:
        if row.class_id != PERSON_CLASS_ID:
            grouped[(row.class_id, row.track_id)].append(row)
    maximum_gap = int(SCENE_MOTION_SETTINGS["maximum_pair_gap_frames"])
    maximum_rate = float(SCENE_MOTION_SETTINGS["maximum_absolute_x_rate_per_second"])
    for (class_id, track_id), track_rows in grouped.items():
        reference_id = f"{class_id}:{track_id}"
        cleaned = clean_track(track_rows)
        for first, second in zip(cleaned, cleaned[1:]):
            frame_gap = second.frame - first.frame
            if frame_gap < 1 or frame_gap > maximum_gap:
                continue
            rate = (second.x - first.x) / (frame_gap / fps)
            if not math.isfinite(rate) or abs(rate) > maximum_rate:
                continue
            midpoint = int(round((first.frame + second.frame) / 2.0))
            samples_by_frame[midpoint].append((float(rate), reference_id))
    return SceneMotionProfile(dict(samples_by_frame))


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
    aspect_ratio: float,
    scene_motion_profile: SceneMotionProfile,
) -> Optional[TrackFeatures]:
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
    times = (frames - frames[0]) / fps
    duration = float(times[-1])
    if duration <= 0.0:
        return None
    median_height = float(np.median(height))
    median_width = float(np.median(width))
    bottom = y + 0.50 * height
    q_value = aspect_ratio * (x - 0.50) / height
    log_height = np.log(height)
    x_slope, _, x_r2, _ = robust_line(times, x)
    q_slope, _, q_r2, q_mad = robust_line(times, q_value)
    log_height_slope, _, _, _ = robust_line(times, log_height)
    bottom_slope, _, _, _ = robust_line(times, bottom)

    delta_time = np.diff(times)
    delta_x = np.diff(x)
    valid_delta = delta_time > 0.0
    local_rates = (
        aspect_ratio
        * delta_x[valid_delta]
        / delta_time[valid_delta]
        / max(median_height, 0.001)
    )
    raw_proxy = (
        DEFAULT_PERSON_HEIGHT_M
        * aspect_ratio
        * abs(x_slope)
        / max(median_height, 0.001)
    )
    q_proxy = DEFAULT_PERSON_HEIGHT_M * abs(q_slope)
    robust_proxy = (
        DEFAULT_PERSON_HEIGHT_M * abs(float(np.median(local_rates)))
        if len(local_rates)
        else 0.0
    )

    scene_rates: List[float] = []
    scene_support: List[int] = []
    for first_frame, second_frame in zip(frames, frames[1:]):
        rate, support = scene_motion_profile.rate_at((first_frame + second_frame) / 2.0)
        scene_rates.append(float(rate))
        scene_support.append(int(support))
    scene_array = np.asarray(scene_rates, dtype=float)
    corrected_delta_x = delta_x - scene_array * delta_time
    corrected_x = np.concatenate([[x[0]], x[0] + np.cumsum(corrected_delta_x)])
    corrected_x_slope, _, _, _ = robust_line(times, corrected_x)
    corrected_q = aspect_ratio * (corrected_x - 0.50) / height
    corrected_q_slope, _, _, _ = robust_line(times, corrected_q)
    corrected_local_rates = (
        aspect_ratio
        * corrected_delta_x[valid_delta]
        / delta_time[valid_delta]
        / max(median_height, 0.001)
    )
    compensated_raw = (
        DEFAULT_PERSON_HEIGHT_M
        * aspect_ratio
        * abs(corrected_x_slope)
        / max(median_height, 0.001)
    )
    compensated_q = DEFAULT_PERSON_HEIGHT_M * abs(corrected_q_slope)
    compensated_robust = (
        DEFAULT_PERSON_HEIGHT_M * abs(float(np.median(corrected_local_rates)))
        if len(corrected_local_rates)
        else 0.0
    )
    supported_rates = [
        abs(rate)
        for rate, support in zip(scene_rates, scene_support)
        if support >= int(SCENE_MOTION_SETTINGS["minimum_reference_tracks"])
    ]
    scene_rate_abs = float(np.median(supported_rates)) if supported_rates else 0.0
    median_support = float(np.median(scene_support)) if scene_support else 0.0
    scene_equivalent = (
        DEFAULT_PERSON_HEIGHT_M
        * aspect_ratio
        * scene_rate_abs
        / max(median_height, 0.001)
    )
    scene_fraction = scene_equivalent / max(raw_proxy + scene_equivalent, 1e-9)

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
    direction = (
        "left_to_right"
        if x_slope > 0.0
        else "right_to_left"
        if x_slope < 0.0
        else "stationary"
    )
    disagreement = float(
        max(compensated_raw, compensated_q, compensated_robust)
        - min(compensated_raw, compensated_q, compensated_robust)
    )
    return TrackFeatures(
        source_id=source_id,
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
        horizontal_range=float(np.max(x) - np.min(x)),
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
        compensated_raw_speed_proxy_mps=float(compensated_raw),
        compensated_q_speed_proxy_mps=float(compensated_q),
        compensated_robust_speed_proxy_mps=float(compensated_robust),
        scene_motion_rate_abs=scene_rate_abs,
        scene_motion_equivalent_speed_mps=float(scene_equivalent),
        scene_motion_fraction=float(scene_fraction),
        scene_motion_support=median_support,
        log_scene_motion_support=float(math.log1p(median_support)),
        compensated_proxy_disagreement_mps=disagreement,
        one_minus_x_r2=float(max(0.0, 1.0 - x_r2)),
        log_height_ratio=float(math.log(max(height_high / height_low, 1.0))),
        log_duration=float(math.log1p(duration)),
        log_median_height=float(math.log(max(median_height, 1e-6))),
    )


def base_rejection_reason(features: TrackFeatures) -> str:
    checks = [
        (features.clean_rows < int(BASE_GATES["minimum_rows"]), "too_few_rows"),
        (
            features.duration_seconds < BASE_GATES["minimum_duration_seconds"],
            "duration_too_short",
        ),
        (features.coverage < BASE_GATES["minimum_coverage"], "track_too_fragmented"),
        (
            features.median_height < BASE_GATES["minimum_median_height"],
            "box_too_small",
        ),
        (
            features.horizontal_range < BASE_GATES["minimum_horizontal_range"],
            "insufficient_lateral_motion",
        ),
        (
            features.x_fit_r2 < BASE_GATES["minimum_x_fit_r2"],
            "nonlinear_or_unstable_motion",
        ),
        (
            features.height_ratio > BASE_GATES["maximum_height_ratio"],
            "excessive_scale_change",
        ),
        (
            features.edge_fraction > BASE_GATES["maximum_edge_fraction"],
            "track_truncated_at_image_edge",
        ),
        (
            features.reversal_fraction > BASE_GATES["maximum_reversal_fraction"],
            "too_many_direction_reversals",
        ),
        (
            features.scene_motion_support < BASE_GATES["minimum_scene_motion_support"],
            "insufficient_scene_motion_references",
        ),
    ]
    for rejected, reason in checks:
        if rejected:
            return reason
    return ""


def reliability_rejection_reason(features: TrackFeatures) -> str:
    if features.duration_seconds < RELIABILITY_GATES["minimum_duration_seconds"]:
        return "duration_below_reliable_minimum"
    if features.x_fit_r2 < RELIABILITY_GATES["minimum_x_fit_r2"]:
        return "horizontal_fit_below_reliable_minimum"
    return ""


def predict_relative_rows(
    bbox_csv: os.PathLike[str] | str,
    fps: float,
    source_id: str,
    aspect_ratio: float = DEFAULT_ASPECT_RATIO,
) -> Tuple[List[Dict[str, Any]], int]:
    """Return the v22 style within video motion table and person track count."""
    return predict_relative_bbox_rows(
        load_bbox_csv(bbox_csv),
        fps,
        source_id,
        aspect_ratio,
    )


def predict_relative_bbox_rows(
    rows: Sequence[BBoxRow],
    fps: float,
    source_id: str,
    aspect_ratio: float = DEFAULT_ASPECT_RATIO,
) -> Tuple[List[Dict[str, Any]], int]:
    """Calculate relative motion from already loaded bounding box rows."""
    if fps <= 0.0 or not math.isfinite(fps):
        raise ValueError(f"FPS must be positive, received {fps}")
    person_tracks = group_tracks(row for row in rows if row.class_id == PERSON_CLASS_ID)
    scene_profile = build_scene_motion_profile(rows, fps)
    prepared: List[Tuple[TrackFeatures, float, str]] = []
    for track_id in sorted(person_tracks, key=str):
        features = track_features(
            person_tracks[track_id], fps, source_id, aspect_ratio, scene_profile
        )
        if features is None:
            continue
        if features.scene_motion_support >= int(
            SCENE_MOTION_SETTINGS["minimum_reference_tracks"]
        ):
            proxies = [
                features.compensated_raw_speed_proxy_mps,
                features.compensated_q_speed_proxy_mps,
                features.compensated_robust_speed_proxy_mps,
            ]
            role = "scene_motion_compensated_bbox_proxy"
        else:
            proxies = [
                features.raw_speed_proxy_mps,
                features.q_speed_proxy_mps,
                features.robust_speed_proxy_mps,
            ]
            role = "uncompensated_bbox_proxy"
        prepared.append((features, float(np.median(proxies)), role))

    eligible = [
        proxy
        for features, proxy, _ in prepared
        if not base_rejection_reason(features) and proxy > 0.0
    ]
    reference = float(np.median(eligible)) if eligible else None
    logs = np.log(np.maximum(np.asarray(eligible, dtype=float), 1e-9))
    log_centre = float(np.median(logs)) if len(logs) else None
    log_scale = robust_scale(logs) if len(logs) >= 3 else None
    output: List[Dict[str, Any]] = []
    for features, proxy, role in prepared:
        reason = base_rejection_reason(features)
        if not reason and len(eligible) < 3:
            reason = "too_few_comparable_tracks"
        relative_index = (
            proxy / reference if reference is not None and reference > 0.0 else None
        )
        robust_z = (
            (math.log(max(proxy, 1e-9)) - float(log_centre)) / float(log_scale)
            if log_centre is not None and log_scale is not None and log_scale > 1e-8
            else None
        )
        category = ""
        if not reason:
            if robust_z is None:
                category = "typical_within_video"
            elif robust_z <= -0.75:
                category = "slower_within_video"
            elif robust_z >= 0.75:
                category = "faster_within_video"
            else:
                category = "typical_within_video"
        row: Dict[str, Any] = {
            "source_id": source_id,
            "prediction_track_id": features.prediction_track_id,
            "bbox_motion_proxy": proxy,
            "relative_motion_index": relative_index,
            "relative_motion_robust_z": robust_z,
            "relative_motion_category": category,
            "relative_motion_status": "valid" if not reason else "rejected",
            "reject_reason": reason,
            "proxy_role": role,
            "speed_interpretation": "within-video bbox motion only; not metres per second",
            "comparable_tracks": len(eligible),
        }
        row.update(asdict(features))
        reliability_reason = reliability_rejection_reason(features)
        row["base_status"] = "rejected" if base_rejection_reason(features) else "eligible"
        row["base_reject_reason"] = base_rejection_reason(features)
        row["reliability_status"] = "rejected" if reliability_reason else "eligible"
        row["reliability_reject_reason"] = reliability_reason
        output.append(row)
    return output, len(person_tracks)


def _numeric_values(rows: Sequence[Mapping[str, Any]], field: str) -> List[float]:
    output: List[float] = []
    for row in rows:
        value = safe_float(row.get(field))
        if value is not None:
            output.append(value)
    return output


def _quantile(values: Sequence[float], fraction: float) -> Optional[float]:
    return float(np.quantile(np.asarray(values, dtype=float), fraction)) if values else None


def _source_cache_is_current(cache_path: Path, input_path: Path) -> bool:
    try:
        return cache_path.is_file() and cache_path.stat().st_mtime >= input_path.stat().st_mtime
    except OSError:
        return False


def analyse_crowd_sources(
    sources: Sequence[Mapping[str, Any]],
    output_root: Path,
    force: bool,
    log: Callable[[str], None],
) -> Dict[str, Any]:
    cache_root = output_root / "crowd_relative_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    track_rows: List[Dict[str, Any]] = []
    source_rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, str]] = []
    for index, source in enumerate(sources, start=1):
        source_id = str(source.get("source_id", "")).strip()
        bbox_path = Path(str(source.get("bbox_csv", ""))).expanduser()
        fps = safe_float(source.get("fps"))
        aspect_ratio = safe_float(source.get("aspect_ratio")) or DEFAULT_ASPECT_RATIO
        crossing_ids = {
            normalise_id(value)
            for value in source.get("crossing_track_ids", [])
            if normalise_id(value)
        }
        base_source = {
            "source_id": source_id,
            "locality": str(source.get("locality", "Unknown")),
            "state": str(source.get("state", "")),
            "country": str(source.get("country", "Unknown")),
            "fps": fps,
            "bbox_csv": str(bbox_path),
        }
        if not source_id or not bbox_path.is_file() or fps is None or fps <= 0.0:
            failure = {
                "source_id": source_id or f"row_{index}",
                "reason": "missing_bbox_csv_or_fps",
            }
            failures.append(failure)
            log(f"CROWD speed report skipped {failure['source_id']}: {failure['reason']}")
            continue
        cache_path = cache_root / f"{safe_name(source_id)}_relative_motion.csv"
        try:
            memory_result = None if force else _RELATIVE_MEMORY_CACHE.get(source_id)
            if memory_result is not None:
                relative_rows, person_track_count = memory_result
                write_dict_csv(cache_path, relative_rows)
            elif force or not _source_cache_is_current(cache_path, bbox_path):
                relative_rows, person_track_count = predict_relative_rows(
                    bbox_path, fps, source_id, aspect_ratio
                )
                write_dict_csv(cache_path, relative_rows)
            else:
                relative_rows = read_dict_csv(cache_path)
                person_track_count = len(
                    {
                        normalise_id(row.track_id)
                        for row in load_bbox_csv(bbox_path)
                        if row.class_id == PERSON_CLASS_ID
                    }
                )
        except Exception as error:
            failures.append({"source_id": source_id, "reason": str(error)})
            log(f"CROWD speed report failed for {source_id}: {error}")
            continue
        relative_by_track = {
            normalise_id(row.get("prediction_track_id")): dict(row)
            for row in relative_rows
            if normalise_id(row.get("prediction_track_id"))
        }
        valid_count = 0
        for track_id in sorted(crossing_ids, key=str):
            row = relative_by_track.get(track_id)
            if row is None:
                row = {
                    "source_id": source_id,
                    "prediction_track_id": track_id,
                    "relative_motion_status": "rejected",
                    "reject_reason": "feature_extraction_failed",
                    "speed_interpretation": "within-video bbox motion only; not metres per second",
                }
            output_row = {**base_source, **row, "detected_crossing": 1}
            output_row["prediction_track_id"] = track_id
            track_rows.append(output_row)
            if str(output_row.get("relative_motion_status", "")) == "valid":
                valid_count += 1
        source_rows.append(
            {
                **base_source,
                "person_tracks": person_track_count,
                "detected_crossing_tracks": len(crossing_ids),
                "analysed_crossing_tracks": valid_count,
                "rejected_crossing_tracks": len(crossing_ids) - valid_count,
            }
        )
        if index % 50 == 0 or index == len(sources):
            log(f"CROWD crossing motion analysed {index}/{len(sources)} bbox CSV files")

    by_city_sources: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    by_city_tracks: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in source_rows:
        key = (str(row["locality"]), str(row["state"]), str(row["country"]))
        by_city_sources[key].append(row)
    for row in track_rows:
        key = (str(row["locality"]), str(row["state"]), str(row["country"]))
        by_city_tracks[key].append(row)

    city_rows: List[Dict[str, Any]] = []
    for key in sorted(by_city_sources, key=lambda item: (item[2], item[0], item[1])):
        locality, state, country = key
        city_sources = by_city_sources[key]
        city_tracks = by_city_tracks.get(key, [])
        valid = [
            row
            for row in city_tracks
            if str(row.get("relative_motion_status", "")) == "valid"
        ]
        values = _numeric_values(valid, "relative_motion_index")
        proxies = _numeric_values(valid, "bbox_motion_proxy")
        categories = Counter(str(row.get("relative_motion_category", "")) for row in valid)
        rejection_counts = Counter(
            str(row.get("reject_reason", "unknown")) or "unknown"
            for row in city_tracks
            if str(row.get("relative_motion_status", "")) != "valid"
        )
        crossing_count = sum(int(row["detected_crossing_tracks"]) for row in city_sources)
        analysed_count = len(valid)
        city_rows.append(
            {
                "locality": locality,
                "state": state,
                "country": country,
                "source_videos": len(city_sources),
                "person_tracks": sum(int(row["person_tracks"]) for row in city_sources),
                "detected_crossing_tracks": crossing_count,
                "analysed_crossing_tracks": analysed_count,
                "rejected_crossing_tracks": crossing_count - analysed_count,
                "analysis_coverage": analysed_count / crossing_count if crossing_count else None,
                "median_relative_motion_index": float(np.median(values)) if values else None,
                "relative_motion_index_q25": _quantile(values, 0.25),
                "relative_motion_index_q75": _quantile(values, 0.75),
                "median_bbox_motion_proxy": float(np.median(proxies)) if proxies else None,
                "slower_within_video": categories.get("slower_within_video", 0),
                "typical_within_video": categories.get("typical_within_video", 0),
                "faster_within_video": categories.get("faster_within_video", 0),
                "top_rejection_reason": rejection_counts.most_common(1)[0][0]
                if rejection_counts
                else "",
                "motion_unit": "relative index",
                "interpretation": "within-video bbox motion only; not metres per second",
            }
        )

    write_dict_csv(output_root / "crowd_crossing_relative_motion_tracks.csv", track_rows)
    write_dict_csv(output_root / "crowd_crossing_relative_motion_sources.csv", source_rows)
    write_dict_csv(output_root / "crowd_crossing_relative_motion_by_city.csv", city_rows)
    valid_tracks = [
        row for row in track_rows if str(row.get("relative_motion_status", "")) == "valid"
    ]
    rejections = Counter(
        str(row.get("reject_reason", "unknown")) or "unknown"
        for row in track_rows
        if str(row.get("relative_motion_status", "")) != "valid"
    )
    summary = {
        "status": "complete" if source_rows else "no_crowd_sources",
        "source_files_requested": len(sources),
        "source_files_analysed": len(source_rows),
        "cities_analysed": len(city_rows),
        "person_tracks": sum(int(row["person_tracks"]) for row in source_rows),
        "detected_crossing_tracks": len(track_rows),
        "valid_relative_motion_tracks": len(valid_tracks),
        "coverage": len(valid_tracks) / len(track_rows) if track_rows else 0.0,
        "relative_motion_categories": dict(
            Counter(str(row.get("relative_motion_category", "")) for row in valid_tracks)
        ),
        "rejection_counts": dict(rejections),
        "failures": failures,
        "speed_unit": None,
        "interpretation": "CROWD bbox CSV supports relative apparent crossing motion, not absolute metres per second",
    }
    write_json(output_root / "crowd_crossing_relative_motion_summary.json", summary)
    return {"summary": summary, "city_rows": city_rows, "track_rows": track_rows}


def _read_csv_from_zip(path: Path) -> Iterable[List[Dict[str, str]]]:
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            if not name.lower().endswith(".csv") or "/._" in name or name.startswith("._"):
                continue
            with archive.open(name) as raw:
                text = (line.decode("utf-8-sig") for line in raw)
                yield list(csv.DictReader(text))


def _candidate_role(path: Path) -> str:
    return "validation" if "validation" in str(path).lower() else "training"


def _manifest_score(path: Path) -> Tuple[int, int]:
    name = path.name.lower()
    preferred = int("crosswalk_axis" in name or "full_training" in name)
    try:
        rows = len(read_dict_csv(path))
    except Exception:
        rows = -1
    return preferred, rows


def _index_role_and_score(path: Path) -> Tuple[str, Tuple[int, int, int]]:
    """Infer the split from index content and prefer the current crosswalk target."""
    try:
        rows = read_dict_csv(path)
    except Exception:
        return _candidate_role(path), (0, 0, -1)
    sample = rows[0] if rows else {}
    searchable = " ".join(
        [
            str(path),
            str(sample.get("video_path", "")),
            str(sample.get("ground_truth_bbox_csv", "")),
        ]
    ).lower()
    role = "validation" if "validation" in searchable else "training"
    target = str(sample.get("ground_truth_target", "")).lower()
    target_score = int("crosswalk traversal axis" in target)
    confirmed = sum(int(safe_float(row.get("ground_truth_tracks")) or 0) for row in rows)
    return role, (target_score, confirmed, len(rows))


def _resolve_waymo_output_path(
    repository_root: Path,
    index_path: Path,
    value: Any,
) -> Optional[Path]:
    text = str(value or "").strip()
    if not text:
        return None
    candidate = Path(text).expanduser()
    if candidate.is_absolute():
        try:
            relative = candidate.relative_to("/workspace")
        except ValueError:
            return candidate
        return repository_root / relative
    repository_candidate = repository_root / candidate
    if repository_candidate.exists():
        return repository_candidate
    return index_path.parent / candidate


def _raw_waymo_files(split_root: Path) -> List[Path]:
    if not split_root.is_dir():
        return []
    return sorted(
        path
        for path in split_root.iterdir()
        if path.is_file()
        and not path.name.startswith(".")
        and (path.name.endswith(".tfrecord") or ".tfrecord-" in path.name)
    )


def _waymo_split_is_ready(
    repository_root: Path,
    raw_split_root: Path,
    processed_split_root: Path,
    split_name: str,
) -> bool:
    index_path = processed_split_root / "waymo_sequence_index.csv"
    manifest_path = processed_split_root / f"waymo_{split_name}_manifest.csv"
    raw_count = len(_raw_waymo_files(raw_split_root))
    if not index_path.is_file() or not manifest_path.is_file():
        return False

    summary_path = processed_split_root / "waymo_export_summary.json"
    try:
        export_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        index_rows = read_dict_csv(index_path)
    except (OSError, ValueError):
        return False
    selected = safe_int(export_summary.get("files_selected"))
    completed = safe_int(export_summary.get("files_completed"))
    failed = safe_int(export_summary.get("files_failed"))
    if raw_count and selected != raw_count:
        return False
    if selected is None or completed != selected or (failed or 0) != 0:
        return False

    for row in index_rows:
        if (safe_int(row.get("ground_truth_rows")) or 0) <= 0:
            continue
        prediction = _resolve_waymo_output_path(
            repository_root,
            index_path,
            row.get("prediction_bbox_csv"),
        )
        sequence_manifest = _resolve_waymo_output_path(
            repository_root,
            index_path,
            row.get("manifest_csv"),
        )
        if (
            prediction is None
            or not prediction.is_file()
            or sequence_manifest is None
            or not sequence_manifest.is_file()
        ):
            return False
    return bool(index_rows)


def _run_checked(command: Sequence[str], cwd: Path) -> None:
    subprocess.run(list(command), cwd=str(cwd), check=True)


def _docker_waymo_export(
    repository_root: Path,
    raw_dataset_root: Path,
    processed_split_root: Path,
    split_name: str,
    log: Callable[[str], None],
) -> None:
    raw_mount_root = raw_dataset_root.parent
    raw_container_path = Path("/waymo_source") / raw_dataset_root.name / split_name
    output_relative = processed_split_root.relative_to(repository_root)
    export_command = " ".join(
        [
            "python3 speed_estimation_harness.py waymo_export",
            shlex.quote(str(raw_container_path)),
            shlex.quote(str(output_relative)),
            "FRONT 10 0 false",
        ]
    )
    container_command = (
        "pip install --no-cache-dir --no-deps "
        "waymo-open-dataset-tf-2-12-0==1.6.4 && "
        "pip install --no-cache-dir --no-deps "
        "opencv-python-headless==4.8.1.78 && "
        + export_command
    )
    command = [
        "docker",
        "run",
        "--rm",
        "--platform",
        "linux/amd64",
        "-v",
        f"{raw_mount_root}:/waymo_source:ro",
        "-v",
        f"{repository_root}:/workspace",
        "-w",
        "/workspace",
        "tensorflow/tensorflow:2.12.0",
        "bash",
        "-lc",
        container_command,
    ]
    log(f"Exporting raw Waymo {split_name} TFRecords with Docker")
    _run_checked(command, repository_root)


def ensure_waymo_processed(
    raw_dataset_path: os.PathLike[str] | str,
    repository_root: os.PathLike[str] | str,
    output_root: os.PathLike[str] | str,
    process_if_missing: bool,
    log: Optional[Callable[[str], None]] = None,
) -> List[Path]:
    """Prepare Waymo reports from configured raw TFRecords when requested.

    Raw TFRecord decoding runs in the same TensorFlow Docker image used during
    calibration. YOLO plus BoT SORT tracking and matching reuse the existing
    speed_estimation_harness.py from the repository. No parser is introduced.
    """
    logger = log or _default_log
    repository = Path(repository_root).expanduser().resolve()
    raw_path_text = os.path.expandvars(str(raw_dataset_path or "")).strip()
    raw_root = (
        Path(raw_path_text).expanduser().resolve()
        if raw_path_text
        else repository / "__waymo_dataset_not_configured__"
    )
    processed_root = Path(output_root).expanduser().resolve() / "waymo_processed"
    split_roots = {
        split: processed_root / split for split in ("training", "validation")
    }
    combined_manifests = [
        split_root / f"waymo_{split}_manifest.csv"
        for split, split_root in split_roots.items()
    ]
    existing_manifests = [path for path in combined_manifests if path.is_file()]
    ready = {
        split: _waymo_split_is_ready(
            repository,
            raw_root / split,
            processed_split,
            split,
        )
        for split, processed_split in split_roots.items()
    }
    if all(ready.values()):
        logger(f"Reusing processed Waymo data: {processed_root}")
        return [processed_root]
    if not process_if_missing:
        if existing_manifests:
            logger(f"Using available partial Waymo processed data: {processed_root}")
            return [processed_root]
        logger("Waymo raw processing is disabled in config")
        return []
    if not raw_path_text:
        logger("Waymo raw dataset path is not configured")
        return []
    if not raw_root.is_dir():
        logger(f"Waymo raw dataset path does not exist: {raw_root}")
        return []

    harness_path = repository / "speed_estimation_harness.py"
    if not harness_path.is_file():
        logger(
            "Waymo preprocessing cannot start because speed_estimation_harness.py "
            f"is not present in {repository}"
        )
        return []

    processed_root.mkdir(parents=True, exist_ok=True)
    processing_summary: Dict[str, Any] = {
        "metrics_build_id": METRICS_BUILD_ID,
        "raw_dataset_path": str(raw_root),
        "processed_root": str(processed_root),
        "process_if_missing": True,
        "splits": {},
    }
    for split_name, processed_split in split_roots.items():
        raw_split = raw_root / split_name
        if ready[split_name]:
            processing_summary["splits"][split_name] = {"status": "reused"}
            continue
        if not _raw_waymo_files(raw_split):
            logger(f"No raw Waymo TFRecords found for {split_name}: {raw_split}")
            processing_summary["splits"][split_name] = {
                "status": "skipped",
                "reason": "no_tfrecord_files",
            }
            continue
        processed_split.mkdir(parents=True, exist_ok=True)
        index_path = processed_split / "waymo_sequence_index.csv"
        manifest_path = processed_split / f"waymo_{split_name}_manifest.csv"
        try:
            _docker_waymo_export(
                repository,
                raw_root,
                processed_split,
                split_name,
                logger,
            )
            _run_checked(
                [
                    sys.executable,
                    str(harness_path),
                    "track_index",
                    str(index_path),
                    "auto",
                    "0",
                    "false",
                ],
                repository,
            )
            _run_checked(
                [
                    sys.executable,
                    str(harness_path),
                    "match_index",
                    str(index_path),
                    str(manifest_path),
                    "0",
                ],
                repository,
            )
            if not _waymo_split_is_ready(
                repository,
                raw_split,
                processed_split,
                split_name,
            ):
                raise RuntimeError(
                    f"Waymo {split_name} processing finished but completeness checks failed"
                )
            split_summary = {
                "metrics_build_id": METRICS_BUILD_ID,
                "status": "complete",
                "raw_tfrecords": len(_raw_waymo_files(raw_split)),
                "index": str(index_path),
                "manifest": str(manifest_path),
            }
            write_json(
                processed_split / "waymo_processing_complete.json",
                split_summary,
            )
            processing_summary["splits"][split_name] = split_summary
        except (OSError, subprocess.CalledProcessError, RuntimeError) as error:
            logger(f"Waymo {split_name} preprocessing failed: {error}")
            processing_summary["splits"][split_name] = {
                "status": "failed",
                "reason": str(error),
            }
    write_json(processed_root / "waymo_processing_summary.json", processing_summary)
    if any(path.is_file() for path in combined_manifests):
        return [processed_root]
    return []


def discover_waymo_files(
    repository_root: Path,
    explicit_roots: Optional[Sequence[os.PathLike[str] | str]],
) -> Dict[str, Any]:
    roots: List[Path] = []
    if explicit_roots:
        roots.extend(Path(value).expanduser() for value in explicit_roots)
    else:
        roots.extend(
            [
                repository_root / "data" / "speed_calibration" / "waymo",
                repository_root / "data" / "speed_calibration" / "waymo_validation_v18",
            ]
        )
    roots = [root.resolve() for root in roots if root.exists()]
    manifests: Dict[str, List[Path]] = defaultdict(list)
    indices: Dict[str, List[Path]] = defaultdict(list)
    relative_csvs: List[Path] = []
    relative_zips: List[Path] = []
    for root in roots:
        for candidate in root.rglob("*.csv"):
            lower = candidate.name.lower()
            if lower.startswith("waymo_sequence_index"):
                role, _ = _index_role_and_score(candidate)
                indices[role].append(candidate)
            elif "manifest" in lower:
                try:
                    with candidate.open("r", newline="", encoding="utf-8-sig") as handle:
                        fields = csv.DictReader(handle).fieldnames or []
                    if {"source_id", "prediction_track_id", "ground_truth_speed_mps"}.issubset(fields):
                        manifests[_candidate_role(candidate)].append(candidate)
                except OSError:
                    pass
            if lower == "relative_motion.csv" or lower.endswith("_relative_motion.csv"):
                relative_csvs.append(candidate)
        relative_zips.extend(root.rglob("*relative_motion*.zip"))
    selected_manifests = {
        role: max(paths, key=_manifest_score)
        for role, paths in manifests.items()
        if paths
    }
    selected_indices = {
        role: max(paths, key=lambda path: _index_role_and_score(path)[1])
        for role, paths in indices.items()
        if paths
    }
    return {
        "roots": roots,
        "manifests": selected_manifests,
        "indices": selected_indices,
        "relative_csvs": relative_csvs,
        "relative_zips": relative_zips,
    }


def _relative_row_map(files: Mapping[str, Any]) -> Dict[Tuple[str, str], Dict[str, str]]:
    output: Dict[Tuple[str, str], Dict[str, str]] = {}
    for path in files.get("relative_csvs", []):
        try:
            batches = [read_dict_csv(path)]
        except Exception:
            continue
        for rows in batches:
            for row in rows:
                key = (
                    str(row.get("source_id", "")).strip(),
                    normalise_id(row.get("prediction_track_id")),
                )
                if all(key):
                    output[key] = row
    for path in files.get("relative_zips", []):
        try:
            batches = _read_csv_from_zip(path)
            for rows in batches:
                for row in rows:
                    key = (
                        str(row.get("source_id", "")).strip(),
                        normalise_id(row.get("prediction_track_id")),
                    )
                    if all(key):
                        output[key] = row
        except Exception:
            continue
    return output


def _resolve_manifest_bbox(
    raw_path: str,
    source_id: str,
    roots: Sequence[Path],
) -> Optional[Path]:
    candidate = Path(raw_path).expanduser()
    if candidate.is_file():
        return candidate
    for root in roots:
        for name in ("crowd_yolo_botsort_bbox.csv", "imptc_yolo_botsort_bbox.csv"):
            candidate = root / source_id / name
            if candidate.is_file():
                return candidate
    return None


def _ensure_waymo_relative_rows(
    manifests: Mapping[str, Path],
    files: Mapping[str, Any],
    relative_map: Dict[Tuple[str, str], Dict[str, str]],
    output_root: Path,
    force: bool,
    log: Callable[[str], None],
) -> None:
    cache_root = output_root / "waymo_relative_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    for role, manifest_path in manifests.items():
        rows = read_dict_csv(manifest_path)
        by_source: Dict[str, Dict[str, str]] = {}
        for row in rows:
            source_id = str(row.get("source_id", "")).strip()
            if source_id:
                by_source.setdefault(source_id, row)
        for source_id, row in by_source.items():
            required_tracks = {
                normalise_id(item.get("prediction_track_id"))
                for item in rows
                if str(item.get("source_id", "")).strip() == source_id
            }
            if required_tracks and all((source_id, track) in relative_map for track in required_tracks):
                continue
            bbox_path = _resolve_manifest_bbox(
                str(row.get("bbox_csv", "")), source_id, files.get("roots", [])
            )
            fps = safe_float(row.get("fps"))
            aspect = safe_float(row.get("aspect_ratio")) or DEFAULT_ASPECT_RATIO
            if bbox_path is None or fps is None or fps <= 0.0:
                continue
            cache_path = cache_root / f"{safe_name(source_id)}_relative_motion.csv"
            try:
                if force or not _source_cache_is_current(cache_path, bbox_path):
                    relative_rows, _ = predict_relative_rows(bbox_path, fps, source_id, aspect)
                    write_dict_csv(cache_path, relative_rows)
                else:
                    relative_rows = read_dict_csv(cache_path)
                for relative in relative_rows:
                    track_id = normalise_id(relative.get("prediction_track_id"))
                    if track_id:
                        relative_map[(source_id, track_id)] = {
                            key: str(value) if value is not None else ""
                            for key, value in relative.items()
                        }
            except Exception as error:
                log(f"Waymo relative motion generation failed for {source_id}: {error}")
        log(f"Waymo {role} relative motion rows prepared for {len(by_source)} sources")


def regression_fit(rows: Sequence[Mapping[str, Any]]) -> Optional[np.ndarray]:
    pairs = [
        (safe_float(row.get("relative_motion_index")), safe_float(row.get("ground_truth_speed_mps")))
        for row in rows
    ]
    pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
    if len(pairs) < 2:
        return None
    x_value = np.asarray([pair[0] for pair in pairs], dtype=float)
    y_value = np.asarray([pair[1] for pair in pairs], dtype=float)
    design = np.column_stack([np.ones(len(x_value)), x_value])
    return np.linalg.lstsq(design, y_value, rcond=None)[0]


def regression_predict(beta: Optional[np.ndarray], value: Any) -> Optional[float]:
    x_value = safe_float(value)
    if beta is None or x_value is None:
        return None
    return float(np.clip(beta[0] + beta[1] * x_value, 0.10, 3.50))


def error_metrics(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    pairs = [
        (safe_float(row.get("ground_truth_speed_mps")), safe_float(row.get("predicted_speed_mps")))
        for row in rows
    ]
    pairs = [(truth, pred) for truth, pred in pairs if truth is not None and pred is not None]
    if not pairs:
        return {
            "count": 0,
            "mae_mps": None,
            "rmse_mps": None,
            "bias_mps": None,
            "median_absolute_error_mps": None,
            "within_0_25_mps": None,
            "within_0_50_mps": None,
            "r2": None,
        }
    truth = np.asarray([pair[0] for pair in pairs], dtype=float)
    pred = np.asarray([pair[1] for pair in pairs], dtype=float)
    error = pred - truth
    absolute = np.abs(error)
    ss_res = float(np.sum(np.square(error)))
    ss_tot = float(np.sum(np.square(truth - np.mean(truth))))
    return {
        "count": len(pairs),
        "mae_mps": float(np.mean(absolute)),
        "rmse_mps": float(math.sqrt(np.mean(np.square(error)))),
        "bias_mps": float(np.mean(error)),
        "median_absolute_error_mps": float(np.median(absolute)),
        "within_0_25_mps": float(np.mean(absolute <= 0.25)),
        "within_0_50_mps": float(np.mean(absolute <= 0.50)),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else None,
    }


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    position = 0
    while position < len(values):
        end = position + 1
        while end < len(values) and values[order[end]] == values[order[position]]:
            end += 1
        ranks[order[position:end]] = (position + end - 1) / 2.0
        position = end
    return ranks


def correlation_metrics(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    pairs = [
        (safe_float(row.get("relative_motion_index")), safe_float(row.get("ground_truth_speed_mps")))
        for row in rows
    ]
    pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
    if len(pairs) < 3:
        return {"pearson": None, "spearman": None, "within_source_order_accuracy": None}
    x_value = np.asarray([pair[0] for pair in pairs], dtype=float)
    y_value = np.asarray([pair[1] for pair in pairs], dtype=float)
    pearson_value = float(np.corrcoef(x_value, y_value)[0, 1])
    spearman_value = float(np.corrcoef(_rank(x_value), _rank(y_value))[0, 1])
    pearson = pearson_value if math.isfinite(pearson_value) else None
    spearman = spearman_value if math.isfinite(spearman_value) else None
    by_source: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_source[str(row.get("source_id", ""))].append(row)
    correct = 0
    total = 0
    for source_rows in by_source.values():
        for first_index in range(len(source_rows)):
            for second_index in range(first_index + 1, len(source_rows)):
                first = source_rows[first_index]
                second = source_rows[second_index]
                first_x = safe_float(first.get("relative_motion_index"))
                second_x = safe_float(second.get("relative_motion_index"))
                first_y = safe_float(first.get("ground_truth_speed_mps"))
                second_y = safe_float(second.get("ground_truth_speed_mps"))
                if None in (first_x, second_x, first_y, second_y):
                    continue
                if abs(first_x - second_x) < 1e-9 or abs(first_y - second_y) < 1e-9:
                    continue
                total += 1
                correct += int((first_x - second_x) * (first_y - second_y) > 0.0)
    return {
        "pearson": pearson,
        "spearman": spearman,
        "within_source_order_accuracy": correct / total if total else None,
        "within_source_pairs": total,
    }


def _confirmed_crossings(index_path: Optional[Path]) -> Optional[int]:
    if index_path is None or not index_path.is_file():
        return None
    total = 0
    for row in read_dict_csv(index_path):
        value = safe_int(row.get("ground_truth_tracks"))
        if value is not None:
            total += value
    return total


def analyse_waymo(
    repository_root: Path,
    output_root: Path,
    explicit_roots: Optional[Sequence[os.PathLike[str] | str]],
    force: bool,
    log: Callable[[str], None],
) -> Dict[str, Any]:
    files = discover_waymo_files(repository_root, explicit_roots)
    manifests: Dict[str, Path] = files["manifests"]
    if not manifests:
        summary = {
            "status": "skipped",
            "reason": "processed Waymo manifests were not found",
            "searched_roots": [str(path) for path in files["roots"]],
        }
        write_json(output_root / "waymo_crossing_speed_summary.json", summary)
        log("Waymo analysis skipped: processed manifests were not found")
        return {"summary": summary, "tracks": []}

    relative_map = _relative_row_map(files)
    _ensure_waymo_relative_rows(
        manifests, files, relative_map, output_root, force, log
    )
    role_tracks: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    role_summary: Dict[str, Dict[str, Any]] = {}
    for role, manifest_path in manifests.items():
        manifest_rows = read_dict_csv(manifest_path)
        eligible_rows = [row for row in manifest_rows if truthy(row.get("include"))]
        joined: List[Dict[str, Any]] = []
        for row in eligible_rows:
            key = (
                str(row.get("source_id", "")).strip(),
                normalise_id(row.get("prediction_track_id")),
            )
            relative = relative_map.get(key)
            if relative is None or str(relative.get("relative_motion_status", "")) != "valid":
                continue
            joined.append({**row, **relative, "dataset_role": role})
        role_tracks[role] = joined
        excluded = [row for row in manifest_rows if not truthy(row.get("include"))]
        reason_counts = Counter(str(row.get("exclusion_reason", "")) or "unknown" for row in excluded)
        association_failed = sum(
            count for reason, count in reason_counts.items() if reason in ASSOCIATION_REJECTION_REASONS
        )
        confirmed_crossings = _confirmed_crossings(files["indices"].get(role))
        matched_count = len(manifest_rows)
        role_summary[role] = {
            "manifest": str(manifest_path),
            "confirmed_crossing_pedestrians": confirmed_crossings,
            "matched_with_botsort": matched_count,
            "eligible_after_matching_and_motion_checks": len(eligible_rows),
            "analysed_with_valid_relative_motion": len(joined),
            "eligible_but_unanalysable": len(eligible_rows) - len(joined),
            "failed_checks": len(excluded),
            "failed_association_checks": association_failed,
            "failed_motion_quality_checks": len(excluded) - association_failed,
            "not_detected_or_not_matched": (
                max(0, confirmed_crossings - matched_count)
                if confirmed_crossings is not None
                else None
            ),
            "exclusion_reasons": dict(reason_counts),
        }

    training = role_tracks.get("training", [])
    validation = role_tracks.get("validation", [])
    training_predictions: List[Dict[str, Any]] = []
    for source_id in sorted({str(row.get("source_id", "")) for row in training}):
        train_fold = [row for row in training if str(row.get("source_id", "")) != source_id]
        beta = regression_fit(train_fold)
        for row in training:
            if str(row.get("source_id", "")) != source_id:
                continue
            prediction = regression_predict(beta, row.get("relative_motion_index"))
            training_predictions.append({**row, "predicted_speed_mps": prediction})
    beta = regression_fit(training)
    validation_predictions = [
        {
            **row,
            "predicted_speed_mps": regression_predict(beta, row.get("relative_motion_index")),
        }
        for row in validation
    ]
    training_truth = _numeric_values(training, "ground_truth_speed_mps")
    baseline_value = float(np.median(training_truth)) if training_truth else None
    validation_baseline = [
        {**row, "predicted_speed_mps": baseline_value} for row in validation
    ]
    for row in training_predictions + validation_predictions:
        truth = safe_float(row.get("ground_truth_speed_mps"))
        prediction = safe_float(row.get("predicted_speed_mps"))
        row["error_mps"] = prediction - truth if None not in (truth, prediction) else None
        row["absolute_error_mps"] = abs(prediction - truth) if None not in (truth, prediction) else None

    all_tracks = training_predictions + validation_predictions
    write_dict_csv(output_root / "waymo_crossing_speed_evaluation_tracks.csv", all_tracks)
    validation_metrics = error_metrics(validation_predictions)
    baseline_metrics = error_metrics(validation_baseline)
    validation_count = int(validation_metrics.get("count", 0))
    validation_mae = safe_float(validation_metrics.get("mae_mps"))
    baseline_mae = safe_float(baseline_metrics.get("mae_mps"))
    validation_r2 = safe_float(validation_metrics.get("r2"))
    qualification_reasons: List[str] = []
    if validation_count < 20:
        qualification_reasons.append("too_few_independent_validation_tracks")
    if (
        validation_mae is None
        or baseline_mae is None
        or validation_mae >= 0.90 * baseline_mae
    ):
        qualification_reasons.append("no_material_improvement_over_constant_baseline")
    if validation_r2 is None or validation_r2 <= 0.0:
        qualification_reasons.append("non_positive_validation_r2")

    summary = {
        "status": "complete",
        "roles": role_summary,
        "check_definitions": {
            "not_detected_or_not_matched": (
                "a confirmed Waymo crossing without a one to one BoT SORT association"
            ),
            "failed_association_checks": (
                "a matched pair rejected for frame count, IoU, prediction coverage or ground truth coverage"
            ),
            "failed_motion_quality_checks": (
                "a matched pair rejected for short, fragmented, small, unstable or insufficiently lateral motion"
            ),
            "eligible_but_unanalysable": (
                "a manifest eligible pair for which the current relative motion feature extraction was not valid"
            ),
        },
        "training_source_grouped_cross_validation": error_metrics(training_predictions),
        "independent_validation": validation_metrics,
        "independent_validation_constant_baseline": baseline_metrics,
        "independent_validation_correlations": correlation_metrics(validation),
        "absolute_speed_model_qualified": not qualification_reasons,
        "qualification_reasons": qualification_reasons,
        "model": {
            "form": "clipped linear regression of ground truth speed on within video relative motion index",
            "intercept": float(beta[0]) if beta is not None else None,
            "relative_motion_coefficient": float(beta[1]) if beta is not None else None,
            "clip_mps": [0.10, 3.50],
            "deployment_warning": (
                "This is an evaluation conversion only. It must not be applied to CROWD as a validated "
                "absolute speed model unless it improves materially over the constant baseline on an "
                "untouched external dataset."
            ),
        },
    }
    write_json(output_root / "waymo_crossing_speed_summary.json", summary)
    return {
        "summary": summary,
        "tracks": all_tracks,
        "validation_tracks": validation_predictions,
    }


def _write_svg_bar(
    path: Path,
    labels: Sequence[str],
    series: Mapping[str, Sequence[float]],
    title: str,
    y_label: str,
) -> None:
    width = max(900, 38 * len(labels) + 220)
    height = 620
    left, right, top, bottom = 90, 35, 80, 180
    plot_width = width - left - right
    plot_height = height - top - bottom
    maximum = max([float(value) for values in series.values() for value in values] + [1.0])
    colours = ["#0072B2", "#E69F00", "#009E73", "#CC79A7"]
    group_width = plot_width / max(len(labels), 1)
    bar_width = group_width * 0.78 / max(len(series), 1)
    parts = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}">'
        ),
        '<rect width="100%" height="100%" fill="white"/>',
        (
            f'<text x="{width / 2}" y="35" text-anchor="middle" '
            f'font-family="Arial" font-size="22">{_xml(title)}</text>'
        ),
        (
            f'<line x1="{left}" y1="{top + plot_height}" '
            f'x2="{left + plot_width}" y2="{top + plot_height}" stroke="#333"/>'
        ),
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#333"/>',
        (
            f'<text transform="translate(24 {top + plot_height / 2}) rotate(-90)" '
            f'text-anchor="middle" font-family="Arial" font-size="15">{_xml(y_label)}</text>'
        ),
    ]
    for tick in range(6):
        value = maximum * tick / 5.0
        y_value = top + plot_height - plot_height * tick / 5.0
        parts.append(
            f'<line x1="{left}" y1="{y_value:.2f}" x2="{left + plot_width}" '
            f'y2="{y_value:.2f}" stroke="#ddd"/>'
        )
        parts.append(
            f'<text x="{left - 10}" y="{y_value + 5:.2f}" text-anchor="end" '
            f'font-family="Arial" font-size="12">{value:.0f}</text>'
        )
    for label_index, label in enumerate(labels):
        x_centre = left + group_width * (label_index + 0.5)
        for series_index, (name, values) in enumerate(series.items()):
            value = float(values[label_index])
            bar_height = plot_height * value / maximum
            x_value = x_centre - (len(series) * bar_width) / 2.0 + series_index * bar_width
            y_value = top + plot_height - bar_height
            parts.append(
                f'<rect x="{x_value:.2f}" y="{y_value:.2f}" '
                f'width="{bar_width * 0.90:.2f}" height="{bar_height:.2f}" '
                f'fill="{colours[series_index % len(colours)]}"/>'
            )
        parts.append(
            f'<text transform="translate({x_centre:.2f} {top + plot_height + 12}) rotate(55)" '
            f'text-anchor="start" font-family="Arial" font-size="11">{_xml(label)}</text>'
        )
    legend_x = left
    for index, name in enumerate(series):
        parts.append(
            f'<rect x="{legend_x}" y="{height - 32}" width="14" height="14" '
            f'fill="{colours[index % len(colours)]}"/>'
        )
        parts.append(
            f'<text x="{legend_x + 20}" y="{height - 20}" font-family="Arial" '
            f'font-size="13">{_xml(name)}</text>'
        )
        legend_x += 150
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def _write_svg_scatter(
    path: Path,
    truth: Sequence[float],
    prediction: Sequence[float],
    title: str,
) -> None:
    width, height = 760, 660
    left, right, top, bottom = 85, 35, 75, 75
    plot_width, plot_height = width - left - right, height - top - bottom
    values = list(truth) + list(prediction)
    minimum = min(values + [0.0])
    maximum = max(values + [1.0])
    padding = max((maximum - minimum) * 0.08, 0.1)
    minimum -= padding
    maximum += padding
    x_pos = lambda value: left + (value - minimum) / (maximum - minimum) * plot_width
    y_pos = lambda value: top + plot_height - (value - minimum) / (maximum - minimum) * plot_height
    parts = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}">'
        ),
        '<rect width="100%" height="100%" fill="white"/>',
        (
            f'<text x="{width / 2}" y="35" text-anchor="middle" '
            f'font-family="Arial" font-size="22">{_xml(title)}</text>'
        ),
        (
            f'<line x1="{left}" y1="{top + plot_height}" '
            f'x2="{left + plot_width}" y2="{top + plot_height}" stroke="#333"/>'
        ),
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#333"/>',
        (
            f'<line x1="{x_pos(minimum):.2f}" y1="{y_pos(minimum):.2f}" '
            f'x2="{x_pos(maximum):.2f}" y2="{y_pos(maximum):.2f}" '
            'stroke="#777" stroke-dasharray="6 4"/>'
        ),
        (
            f'<text x="{left + plot_width / 2}" y="{height - 20}" '
            'text-anchor="middle" font-family="Arial" font-size="15">'
            'Waymo ground truth speed (m/s)</text>'
        ),
        (
            f'<text transform="translate(22 {top + plot_height / 2}) rotate(-90)" '
            'text-anchor="middle" font-family="Arial" font-size="15">'
            'Predicted speed (m/s)</text>'
        ),
    ]
    for x_value, y_value in zip(truth, prediction):
        parts.append(
            f'<circle cx="{x_pos(x_value):.2f}" cy="{y_pos(y_value):.2f}" r="4.5" fill="#0072B2" fill-opacity="0.72"/>'
        )
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def _xml(value: Any) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def write_figures(
    output_root: Path,
    crowd_result: Mapping[str, Any],
    waymo_result: Mapping[str, Any],
    log: Callable[[str], None],
) -> List[str]:
    figure_root = output_root / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    written: List[str] = []
    city_rows = sorted(
        crowd_result.get("city_rows", []),
        key=lambda row: int(row.get("detected_crossing_tracks", 0)),
        reverse=True,
    )[:30]
    if city_rows:
        path = figure_root / "crowd_city_crossing_coverage.svg"
        _write_svg_bar(
            path,
            [str(row.get("locality", "Unknown")) for row in city_rows],
            {
                "Detected crossings": [float(row.get("detected_crossing_tracks", 0)) for row in city_rows],
                "Valid relative motion": [float(row.get("analysed_crossing_tracks", 0)) for row in city_rows],
            },
            "CROWD crossing tracks by city (top 30)",
            "Tracks",
        )
        written.append(str(path))

    roles = waymo_result.get("summary", {}).get("roles", {})
    if roles:
        labels = [role.title() for role in sorted(roles)]
        role_rows = [roles[role] for role in sorted(roles)]
        path = figure_root / "waymo_crossing_pipeline_funnel.svg"
        _write_svg_bar(
            path,
            labels,
            {
                "Confirmed GT": [float(row.get("confirmed_crossing_pedestrians") or 0) for row in role_rows],
                "Matched": [float(row.get("matched_with_botsort") or 0) for row in role_rows],
                "Eligible": [float(row.get("eligible_after_matching_and_motion_checks") or 0) for row in role_rows],
                "Analysed": [float(row.get("analysed_with_valid_relative_motion") or 0) for row in role_rows],
            },
            "Waymo crossing analysis funnel",
            "Pedestrian tracks",
        )
        written.append(str(path))
    validation = waymo_result.get("validation_tracks", [])
    validation_pairs = [
        (float(truth_value), float(prediction_value))
        for row in validation
        if (truth_value := safe_float(row.get("ground_truth_speed_mps"))) is not None
        if (prediction_value := safe_float(row.get("predicted_speed_mps"))) is not None
    ]
    truth = [pair[0] for pair in validation_pairs]
    prediction = [pair[1] for pair in validation_pairs]
    if validation_pairs:
        path = figure_root / "waymo_validation_observed_vs_predicted.svg"
        _write_svg_scatter(path, truth, prediction, "Waymo independent validation")
        written.append(str(path))

    try:
        import plotly.graph_objects as go

        if city_rows:
            fig = go.Figure()
            fig.add_bar(
                name="Detected crossings",
                x=[str(row.get("locality", "Unknown")) for row in city_rows],
                y=[int(row.get("detected_crossing_tracks", 0)) for row in city_rows],
            )
            fig.add_bar(
                name="Valid relative motion",
                x=[str(row.get("locality", "Unknown")) for row in city_rows],
                y=[int(row.get("analysed_crossing_tracks", 0)) for row in city_rows],
            )
            fig.update_layout(
                barmode="group",
                template="plotly_white",
                title="CROWD crossing tracks by city (top 30)",
                xaxis_title="City",
                yaxis_title="Tracks",
            )
            html_path = figure_root / "crowd_city_crossing_coverage.html"
            fig.write_html(str(html_path), include_plotlyjs="cdn")
            written.append(str(html_path))
            try:
                png_path = figure_root / "crowd_city_crossing_coverage.png"
                fig.write_image(str(png_path), width=1800, height=900, scale=2)
                written.append(str(png_path))
            except Exception as error:
                log(f"PNG export skipped for CROWD figure: {error}")
        if validation_pairs:
            fig = go.Figure()
            fig.add_scatter(x=truth, y=prediction, mode="markers", name="Track")
            lower = min(truth + prediction)
            upper = max(truth + prediction)
            fig.add_scatter(
                x=[lower, upper],
                y=[lower, upper],
                mode="lines",
                line={"dash": "dash", "color": "grey"},
                name="Ideal",
            )
            fig.update_layout(
                template="plotly_white",
                title="Waymo independent validation",
                xaxis_title="Ground truth speed (m/s)",
                yaxis_title="Predicted speed (m/s)",
            )
            html_path = figure_root / "waymo_validation_observed_vs_predicted.html"
            fig.write_html(str(html_path), include_plotlyjs="cdn")
            written.append(str(html_path))
            try:
                png_path = figure_root / "waymo_validation_observed_vs_predicted.png"
                fig.write_image(str(png_path), width=1000, height=900, scale=2)
                written.append(str(png_path))
            except Exception as error:
                log(f"PNG export skipped for Waymo figure: {error}")
    except ImportError:
        log("Plotly is unavailable; SVG figures were written instead")
    return written


def run_integrated_speed_report(
    crowd_sources: Sequence[Mapping[str, Any]],
    repository_root: os.PathLike[str] | str,
    output_root: os.PathLike[str] | str,
    waymo_roots: Optional[Sequence[os.PathLike[str] | str]] = None,
    force: bool = False,
    log: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """Run CROWD reports and, when present, processed Waymo evaluation.

    ``crowd_sources`` must contain source_id, bbox_csv, fps, locality, state,
    country, and crossing_track_ids.  No command line parser is used; the
    calling analysis script supplies these values directly.
    """
    logger = log or _default_log
    root = Path(repository_root).expanduser().resolve()
    output = Path(output_root).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    logger(f"Integrated crossing speed report: {output}")
    crowd_result = analyse_crowd_sources(crowd_sources, output, force, logger)
    waymo_result = analyse_waymo(root, output, waymo_roots, force, logger)
    figures = write_figures(output, crowd_result, waymo_result, logger)
    summary = {
        "report_build_id": REPORT_BUILD_ID,
        "scientific_scope": {
            "crowd": "relative apparent crossing motion from YOLO plus BoT SORT bbox CSV",
            "waymo": "metric ground truth evaluation of the bbox relative motion proxy",
            "absolute_crowd_speed_mps": False,
            "reason": "CROWD bbox CSV has no per video metric ground plane calibration or camera pose",
        },
        "crowd": crowd_result["summary"],
        "waymo": waymo_result["summary"],
        "figures": figures,
    }
    write_json(output / "integrated_crossing_speed_summary.json", summary)
    logger("Integrated crossing speed report complete")
    return summary
