"""
Parallel CSV worker for crowd-city analysis.

Each worker owns its own Polars DataFrame and returns only compact Python
objects to the parent process. Shared aggregation remains in analysis.py.
"""

from __future__ import annotations

import math
import os
from typing import Any, Dict, Optional

import polars as pl

from utils.crossing.detection import Detection
from utils.crossing.metrics import Metrics
import utils.crossing.metrics as crossing_metrics_module


_WORKER_MAPPING: Optional[pl.DataFrame] = None
_WORKER_CROSSING_PARAMETERS: Dict[str, Any] = {}
_WORKER_MIN_CONFIDENCE: float = 0.0
_WORKER_BOUNDARY_LEFT: float = 0.45
_WORKER_BOUNDARY_RIGHT: float = 0.55

_DETECTION = Detection()
_METRICS = Metrics()

YOLO_PERSON = 0
YOLO_BICYCLE = 1
YOLO_CAR = 2
YOLO_MOTORCYCLE = 3
YOLO_BUS = 5
YOLO_TRUCK = 7
YOLO_TRAFFIC_SIGN_IDS = (9, 11)
YOLO_CELLPHONE = 67

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic_light", "fire_hydrant", "stop_sign", "parking_meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports_ball", "kite", "baseball_bat", "baseball_glove",
    "skateboard", "surfboard", "tennis_racket", "bottle", "wine_glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot_dog", "pizza", "donut", "cake", "chair", "couch",
    "potted_plant", "bed", "dining_table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cellphone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy_bear",
    "hair_drier", "toothbrush",
]


def initialise_csv_worker(
    mapping: pl.DataFrame,
    crossing_parameters: Dict[str, Any],
    min_confidence: float,
    boundary_left: float,
    boundary_right: float,
    pipeline_model: Dict[str, Any],
    speed_model: Dict[str, Any],
) -> None:
    """Initialise process local state once per worker."""
    global _WORKER_MAPPING
    global _WORKER_CROSSING_PARAMETERS
    global _WORKER_MIN_CONFIDENCE
    global _WORKER_BOUNDARY_LEFT
    global _WORKER_BOUNDARY_RIGHT

    _WORKER_MAPPING = mapping
    _WORKER_CROSSING_PARAMETERS = dict(crossing_parameters or {})
    _WORKER_MIN_CONFIDENCE = float(min_confidence)
    _WORKER_BOUNDARY_LEFT = float(boundary_left)
    _WORKER_BOUNDARY_RIGHT = float(boundary_right)

    # The Waymo speed model is loaded in the parent before the worker pool
    # starts. Copy the already loaded model state into each spawned process so
    # speed calculations remain identical to the single process path.
    crossing_metrics_module._PIPELINE_MODEL = dict(pipeline_model or {})
    crossing_metrics_module._SPEED_MODEL = dict(speed_model or {})
    os.environ["CROWD_CROSSING_SPEED_UNIT"] = (
        "m/s" if crossing_metrics_module._SPEED_MODEL else "relative"
    )


def _empty_metric_counts() -> Dict[str, int]:
    return {
        "persons": 0,
        "cellphones": 0,
        "traffic_signs": 0,
        "vehicles": 0,
        "bicycles": 0,
        "cars": 0,
        "motorcycles": 0,
        "buses": 0,
        "trucks": 0,
    }


def _metric_counts_from_confidence_filtered(df: pl.DataFrame) -> Dict[str, int]:
    """
    Match MetricsCache's first pass count semantics.

    This intentionally runs before removing null or -1 tracker IDs because the
    existing MetricsCache only applies the confidence threshold.
    """
    required = {"yolo-id", "unique-id"}
    if df.height == 0 or not required.issubset(set(df.columns)):
        return _empty_metric_counts()

    working = df.with_columns(
        pl.col("yolo-id").cast(pl.Int64, strict=False).alias("_metric_yolo_id")
    )
    uid = pl.col("unique-id")
    yolo = pl.col("_metric_yolo_id")

    result = working.select([
        uid.filter(yolo == YOLO_PERSON).n_unique().alias("persons"),
        uid.filter(yolo == YOLO_CELLPHONE).n_unique().alias("cellphones"),
        uid.filter(yolo.is_in(list(YOLO_TRAFFIC_SIGN_IDS))).n_unique().alias("traffic_signs"),
        uid.filter(yolo.is_in([YOLO_CAR, YOLO_MOTORCYCLE, YOLO_BUS, YOLO_TRUCK])).n_unique().alias("vehicles"),
        uid.filter(yolo == YOLO_BICYCLE).n_unique().alias("bicycles"),
        uid.filter(yolo == YOLO_CAR).n_unique().alias("cars"),
        uid.filter(yolo == YOLO_MOTORCYCLE).n_unique().alias("motorcycles"),
        uid.filter(yolo == YOLO_BUS).n_unique().alias("buses"),
        uid.filter(yolo == YOLO_TRUCK).n_unique().alias("trucks"),
    ])

    return {
        name: int(result.item(0, name) or 0)
        for name in _empty_metric_counts()
    }


TrackIndex = Dict[Any, pl.DataFrame]


def _build_track_index(df: pl.DataFrame) -> TrackIndex:
    """Build reusable tracks only for IDs that contain a person detection.

    Timing historically selected rows by ``unique-id`` alone, so if a tracker
    ID also contains another class those rows are intentionally retained.
    IDs that never represent a person are excluded because crossing code never
    queries them.
    """
    required = {"yolo-id", "unique-id", "frame-count"}
    if df.height == 0 or not required.issubset(set(df.columns)):
        return {}

    person_ids = (
        df
        .filter(pl.col("yolo-id") == YOLO_PERSON)
        .select("unique-id")
        .unique()
        .get_column("unique-id")
    )
    if len(person_ids) == 0:
        return {}

    ordered = (
        df
        .filter(pl.col("unique-id").is_in(person_ids))
        .sort(["unique-id", "frame-count"])
    )
    track_index: TrackIndex = {}

    for track in ordered.partition_by(
        "unique-id",
        maintain_order=True,
    ):
        if track.height == 0:
            continue
        unique_id = track.get_column("unique-id")[0]
        track_index[unique_id] = track

    return track_index


def _track(
    track_index: TrackIndex,
    track_id: Any,
) -> Optional[pl.DataFrame]:
    """Return the cached rows for one tracker ID."""
    return track_index.get(track_id)


def _time_to_cross_from_track_index(
    track_index: TrackIndex,
    ids: list,
    fps: float,
) -> Dict[Any, float]:
    """Match Metrics.time_to_cross without filtering and sorting per ID."""
    if not ids or not math.isfinite(float(fps)) or float(fps) <= 0:
        return {}

    output: Dict[Any, float] = {}
    for track_id in ids:
        track = _track(track_index, track_id)
        if track is None or track.height < 2 or "frame-count" not in track.columns:
            continue

        frames = []
        for value in track.get_column("frame-count").to_list():
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(numeric):
                frames.append(numeric)

        if len(frames) < 2:
            continue

        duration = (max(frames) - min(frames)) / float(fps)
        if duration > 0:
            output[track_id] = float(duration)

    return output


def _time_to_start_from_track_index(
    mapping: pl.DataFrame,
    track_index: TrackIndex,
    track_ids: list,
    source_id: str,
    fps: float,
):
    """Match waiting time calculation while reusing already sorted tracks."""
    if not track_ids or not math.isfinite(float(fps)) or float(fps) <= 0:
        return None

    checks_per_second = _METRICS._as_float(
        _METRICS._get_config("check_per_sec_time", 3),
        3,
    )
    if checks_per_second is None or checks_per_second <= 0:
        return None

    step = max(1, int(round(float(fps) / checks_per_second)))
    durations: Dict[Any, int] = {}

    for track_id in track_ids:
        track = _track(track_index, track_id)
        if track is None or track.height <= step:
            continue
        if not {"x-center", "height"}.issubset(set(track.columns)):
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
            delta = abs(
                float(x_values[index + step])
                - float(x_values[index])
            )
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

    return crossing_metrics_module.grouping_class.locality_country_wrapper(
        input_dict={source_id: durations},
        mapping=mapping,
    )


def _object_counts(df: pl.DataFrame) -> Dict[str, int]:
    """Match the legacy per CSV COCO object counting path."""
    if df.height == 0:
        return {}

    required = {"yolo-id", "unique-id"}
    if not required.issubset(set(df.columns)):
        return {}

    counts_df = (
        df.unique(subset=["yolo-id", "unique-id"])
        .group_by("yolo-id")
        .len()
        .rename({"len": "count"})
    )

    id_to_count = {
        int(row["yolo-id"]): int(row["count"])
        for row in counts_df.to_dicts()
        if row.get("yolo-id") is not None
    }
    return {
        class_name: int(id_to_count.get(index, 0))
        for index, class_name in enumerate(COCO_CLASSES)
        if int(id_to_count.get(index, 0)) != 0
    }


def process_csv_task(task: Dict[str, Any]) -> Dict[str, Any]:
    """Read and analyse one CSV, returning only compact mergeable results."""
    mapping = _WORKER_MAPPING
    if mapping is None:
        return {
            "status": "error",
            "file_name": str(task.get("file_name", "")),
            "message": "CSV worker was not initialised.",
        }

    file_path = os.fspath(task["file_path"])
    file_name = str(task["file_name"])
    filename_no_ext = str(task["filename_no_ext"])
    fps = float(task["fps"])
    locality_id = int(task["video_locality_id"])
    is_bbox_stream = bool(task["is_bbox_stream"])

    try:
        raw_df = pl.read_csv(file_path)

        if "confidence" not in raw_df.columns:
            return {
                "status": "error",
                "file_name": file_name,
                "message": f"{file_name}: confidence column is missing.",
            }

        confidence_filtered = raw_df.filter(
            pl.col("confidence").cast(pl.Float64, strict=False)
            >= float(_WORKER_MIN_CONFIDENCE)
        )

        metric_counts = _metric_counts_from_confidence_filtered(
            confidence_filtered
        )

        required = {"unique-id", "yolo-id"}
        if not required.issubset(set(confidence_filtered.columns)):
            return {
                "status": "error",
                "file_name": file_name,
                "message": f"{file_name}: required detection columns are missing.",
            }

        df = confidence_filtered.filter(
            pl.col("unique-id").is_not_null()
            & (pl.col("unique-id") != -1)
        )

        track_index = _build_track_index(df) if is_bbox_stream else {}
        object_counts = _object_counts(df)

        ids = []
        all_ids = []
        temp_data: Dict[Any, Any] = {}
        speed_value = None
        time_value = None

        if is_bbox_stream:
            crossing_parameters = dict(_WORKER_CROSSING_PARAMETERS)
            boundary_left = float(
                crossing_parameters.pop(
                    "boundary_left",
                    _WORKER_BOUNDARY_LEFT,
                )
            )
            boundary_right = float(
                crossing_parameters.pop(
                    "boundary_right",
                    _WORKER_BOUNDARY_RIGHT,
                )
            )

            ids, all_ids = _DETECTION.pedestrian_crossing(
                df,
                filename_no_ext,
                mapping,
                boundary_left,
                boundary_right,
                person_id=0,
                fps=fps,
                track_index=track_index,
                **crossing_parameters,
            )

            temp_data = _time_to_cross_from_track_index(
                track_index,
                ids,
                fps,
            )

            speed_value = _METRICS.calculate_speed_of_crossing(
                mapping,
                df,
                {filename_no_ext: temp_data},
            )

            time_value = _time_to_start_from_track_index(
                mapping,
                track_index,
                list(temp_data.keys()),
                filename_no_ext,
                fps,
            )

        time_video = float(task.get("time_video", 0.0) or 0.0)

        return {
            "status": "ok",
            "file_name": file_name,
            "filename_no_ext": filename_no_ext,
            "video_locality_id": locality_id,
            "is_bbox_stream": is_bbox_stream,
            "ids": ids,
            "all_ids": all_ids,
            "temp_data": temp_data,
            "speed_value": speed_value,
            "time_value": time_value,
            "object_counts": object_counts,
            "metric_counts": metric_counts,
            "time_video": float(time_video or 0),
        }

    except Exception as exc:
        return {
            "status": "error",
            "file_name": file_name,
            "filename_no_ext": filename_no_ext,
            "message": f"{file_name}: {exc}",
        }
