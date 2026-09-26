"""Road-surface crossing rules, and the segmentation plumbing they need on Waymo.

The CROWD detector decides a crossing from bounding-box geometry alone: the
track must pass through a narrow vertical strip in the middle of the image and
survive a series of motion filters. Tested against Waymo ground truth, that
finds about one real crosser in six, and its mistakes are mostly pedestrians
on the footpath. Knowing which surface the feet are on fixes both:

* ``detector_on_road`` (rule B): a detector pick whose feet are on the road at
  some point. It removes footpath mistakes without losing any real crossing,
  so it is the clean set used to train the speed model.
* ``road_crossing`` (rule D): the feet are on the road, the track moves across
  at least ``min_crossing_x_range`` of the image while there, and its box size
  changes slowly (people walking along the road towards or away from the
  camera grow or shrink quickly). It is not tied to the middle strip, so it
  finds about twice as many real crossings; it is the rule used to count them.

Both were chosen on the Waymo training split and checked on the untouched
validation split (see utils/crossing/waymo_segmentation_evaluation.py).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import polars as pl

import common
import utils.crossing.metrics as crossing_metrics
from utils.crossing.road_metrics import road_intervals_for_tracks
from utils.segmentation.pipeline import SegmentationPipeline, SegmentRequest
from utils.segmentation.segformer import SurfaceSegmenter
from utils.segmentation.store import SegmentationSettings, SurfaceStore, crossing_fingerprint
from utils.segmentation.surface import (
    DEFAULT_FOOTPOINT_BAND_FRACTION,
    DEFAULT_FOOTPOINT_WIDTH_FRACTION,
    RoadInterval,
)

# Largest absolute rate of change of log box height, per second, for rule D.
# Chosen on the Waymo training split: it removes along-the-road walkers while
# keeping 95% of the real crossings the rule would otherwise find.
MAXIMUM_BOX_SIZE_CHANGE_RATE = 0.25
# Shortest pedestrian track worth segmenting: half a second at Waymo's 10 fps.
MINIMUM_TRACK_ROWS = 5
WAYMO_STORE_FOLDER = "segmentation_store"
WAYMO_VIDEO_NAME = "waymo_front.mp4"


def on_road_x_range(track: pl.DataFrame, interval: Optional[RoadInterval]) -> Optional[float]:
    """Return how far across the image the track moves while on the road."""
    if interval is None:
        return None
    on_road = track.filter(pl.col("frame-count").is_between(interval.entry_frame, interval.exit_frame))
    if on_road.height == 0:
        return None
    return float(on_road["x-center"].max() - on_road["x-center"].min())


def road_crossing_flags(
    track: pl.DataFrame,
    interval: Optional[RoadInterval],
    features: Any,
    minimum_x_range: float,
) -> Dict[str, Any]:
    """Return rule D and its parts for one pedestrian track."""
    x_range = on_road_x_range(track, interval)
    road_lateral = x_range is not None and x_range >= float(minimum_x_range)
    size_rate = None if features is None else float(features.log_height_rate_abs)
    slow_size_change = size_rate is not None and size_rate <= MAXIMUM_BOX_SIZE_CHANGE_RATE
    return {
        "on_road": int(interval is not None),
        "on_road_x_range": x_range,
        "road_lateral": int(road_lateral),
        "box_size_change_rate": size_rate,
        "road_crossing": int(road_lateral and slow_size_change),
    }


# ---------------------------------------------------------------------
# Waymo segmentation
# ---------------------------------------------------------------------

def waymo_segmentation_pipeline(
    processed_root: Path,
    crossing_parameters: Mapping[str, Any],
) -> SegmentationPipeline:
    """Build the segmentation pipeline from config, as the CROWD pass does.

    Surface labels are cached in ``<processed_root>/segmentation_store``, keyed
    by everything that decides which tracks were segmented, so the calibration
    and the evaluation share one cache.
    """
    coarse_hz = float(common.get_configs("segmentation_coarse_hz"))
    refine_hz = float(common.get_configs("segmentation_refine_hz"))
    minimum_confidence = float(common.get_configs("segmentation_min_confidence"))
    segmenter = SurfaceSegmenter(
        model_name=str(common.get_configs("segmentation_model")),
        device=str(common.get_configs("segmentation_device")),
        batch_size=int(common.get_configs("segmentation_batch_size")),
        input_width=int(common.get_configs("segmentation_input_width")),
        input_height=int(common.get_configs("segmentation_input_height")),
    )
    settings = SegmentationSettings(
        model_identifier=segmenter.model_identifier,
        coarse_hz=coarse_hz,
        refine_hz=refine_hz,
        footpoint_band_fraction=DEFAULT_FOOTPOINT_BAND_FRACTION,
        footpoint_width_fraction=DEFAULT_FOOTPOINT_WIDTH_FRACTION,
        minimum_confidence=minimum_confidence,
        crossing_fingerprint=crossing_fingerprint(
            {
                "waymo_tracks": "all_person_tracks",
                "minimum_track_rows": MINIMUM_TRACK_ROWS,
                "min_confidence": common.get_configs("min_confidence"),
                "tracking": crossing_metrics._current_tracking_settings(),
                "crossing_parameters": dict(crossing_parameters),
            }
        ),
    )
    store = SurfaceStore(str(Path(processed_root) / WAYMO_STORE_FOLDER))
    store.ensure_directories()
    return SegmentationPipeline(
        store=store,
        segmenter=segmenter,
        settings=settings,
        credentials=None,
        coarse_hz=coarse_hz,
        refine_hz=refine_hz,
        minimum_confidence=minimum_confidence,
    )


def waymo_person_tracks(sequence: Any) -> Dict[str, pl.DataFrame]:
    """Return every pedestrian track of a Waymo sequence worth segmenting."""
    persons = sequence.prediction_dataframe.filter(pl.col("yolo-id") == crossing_metrics.PERSON_CLASS_ID)
    tracks: Dict[str, pl.DataFrame] = {}
    for track in persons.partition_by("unique-id", maintain_order=True):
        track_id = crossing_metrics.normalise_id(track.get_column("unique-id")[0])
        if track_id and track.height >= MINIMUM_TRACK_ROWS:
            tracks[track_id] = track.sort("frame-count")
    return tracks


def waymo_surface_timelines(
    sequence: Any,
    split: str,
    pipeline: SegmentationPipeline,
    tracks: Mapping[str, pl.DataFrame],
) -> Dict[str, List[Any]]:
    """Return per-track surface samples for one Waymo sequence."""
    if not tracks:
        return {}
    return pipeline.timelines(
        SegmentRequest(
            stem=f"waymo_{split}_{sequence.source_id}",
            video_id=sequence.source_id,
            start_seconds=0.0,
            detection_fps=float(sequence.fps),
            tracks=dict(tracks),
            source_fps=float(sequence.fps),
            video_path=str(sequence.prediction_path.parent / WAYMO_VIDEO_NAME),
        )
    ) or {}


def waymo_road_intervals(timelines: Mapping[str, List[Any]]) -> Dict[str, RoadInterval]:
    return road_intervals_for_tracks(timelines)
