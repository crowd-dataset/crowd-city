"""
Parallel CSV worker for crowd-city analysis.

Each worker owns its own Polars DataFrame and returns only compact Python
objects to the parent process. Shared aggregation remains in analysis.py.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import polars as pl

from utils.analytics.durations import Duration
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
_DURATION = Duration()

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
    video_id = str(task["video_id"])
    start_index = int(task["start_index"])
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
                **crossing_parameters,
            )

            temp_data = _METRICS.time_to_cross(
                df,
                ids,
                filename_no_ext,
                mapping,
            )

            speed_value = _METRICS.calculate_speed_of_crossing(
                mapping,
                df,
                {filename_no_ext: temp_data},
            )

            time_value = _METRICS.time_to_start_cross(
                mapping,
                df,
                {filename_no_ext: temp_data},
            )

        time_video = _DURATION.get_duration(
            mapping,
            video_id,
            start_index,
        )

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
