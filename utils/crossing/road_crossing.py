"""Road-surface crossing rules, and the segmentation plumbing they need on Waymo.

The CROWD detector decides a crossing from bounding-box geometry alone: the
track must pass through a narrow vertical strip in the middle of the image and
survive a series of motion filters. Tested against Waymo ground truth, that
finds about one real crosser in six, and its mistakes are mostly pedestrians
on the footpath. Knowing which surface the feet are on fixes both:

* ``detector_on_road`` (rule B): a detector pick whose feet are on the road at
  some point. It removes footpath mistakes without losing any real crossing.
  It is reported, but the speed model still trains on every detector pick:
  training on rule B narrowed the predicted spread on untouched validation
  below its limit.
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
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
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
MINIMUM_TRACK_SECONDS = 0.5
# A gap longer than this inside one tracker id is treated as a reused id.
TRACK_GAP_SECONDS = 2.0
# Rule D's candidate step reuses two checks of the CROWD detector that the box
# geometry alone cannot replace. A pedestrian riding a bicycle or motorcycle is
# not crossing on foot (Detection.is_rider_id). And a pedestrian whose
# sideways movement in the image is mostly the camera turning barely moves
# relative to the static objects around them (traffic lights, signs,
# hydrants): with such a reference visible for at least
# STATIC_REFERENCE_MIN_FRAMES (at 30 fps, scaled to the video's rate), a
# candidate that moves less than MINIMUM_RELATIVE_X_RANGE of the image width
# relative to it is rejected.
#
# Validation (Waymo rule-D picks, real = matched to a labelled crosswalk
# crosser): the rider check removed 2 of 232 real training crossers and 9
# other picks, 0 of 51 and 6 on validation. For the camera check, 0.08 was
# chosen from a sweep: it removed 9 real and 22 other training picks (1 and 2
# on validation), against 18 and 27 at 0.12; on a hand-reviewed busy CROWD
# segment it removed 4 fake crossings and none of 22 confirmed ones.
MINIMUM_RELATIVE_X_RANGE = 0.08
# Recorded in the results.pickle fingerprint; bump when rule D changes, so
# cached results from an earlier version of the rule are not reused.
ROAD_CROSSING_RULE_VERSION = "rule_d_rider_camera_passes_camera_tol003_walking_v4"
# How far past each edge of the middle strip a track must go to count as
# having passed in front of the camera (passes_camera). A pedestrian standing
# far ahead can be carried just across the strip when the car turns; with
# 0.03 such a case on a reviewed CROWD segment (0.43 -> 0.58) is rejected,
# while 151 of the 153 real Waymo training crossers that pass the strip and
# all 24 validation ones are kept. The detector's own tolerance (the frozen
# "tol") is left unchanged.
PASS_TOLERANCE = 0.03
STATIC_REFERENCE_MIN_FRAMES = 8
DETECTOR_BASE_FPS = 30.0
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


# A crossing pedestrian walks: a cyclist or a figure swept across the image by
# a turning camera moves too fast, too unevenly, or hardly at all. The walking
# speed is estimated from the box alone, as sideways image speed over box
# height times a typical stature, over WALKING_SPEED_WINDOW_SECONDS steps; the
# track must walk at MINIMUM_WALKING_SPEED to MAXIMUM_WALKING_SPEED (median
# over the track), must not speed up more than MAXIMUM_SPEED_UP times from its
# first to its last third, and must have its feet on the road in at least
# MINIMUM_ROAD_SHARE of its segmented samples.
#
# Chosen on the Waymo training picks you reviewed: all 11 fakes (a cyclist the
# detector saw no bicycle under, people swept across by a turning camera,
# runners) are removed and 205 of 218 real crossings kept. Held out: Waymo
# validation 0 of 4 fakes left, 39 of 41 real kept; the reviewed Paris
# segment keeps all 22 real crossings.
WALKING_SPEED_WINDOW_SECONDS = 0.5
MINIMUM_WALKING_SPEED = 0.2
MAXIMUM_WALKING_SPEED = 3.5
MAXIMUM_SPEED_UP = 5.0
MINIMUM_ROAD_SHARE = 0.2
STATURE_METRES = 1.7


def walking_motion(track: pl.DataFrame, fps: float, aspect_ratio: float) -> Optional[Tuple[float, float]]:
    """Return (median walking speed in m/s, speed-up ratio) of a track, or None."""
    ordered = track.sort("frame-count")
    frames = ordered.get_column("frame-count").cast(pl.Float64, strict=False).to_numpy()
    x = ordered.get_column("x-center").cast(pl.Float64, strict=False).to_numpy()
    height = ordered.get_column("height").cast(pl.Float64, strict=False).to_numpy()
    if len(frames) < 4 or fps <= 0:
        return None
    times = frames / float(fps)
    step = max(1, int(round(WALKING_SPEED_WINDOW_SECONDS * float(fps))))
    speeds = []
    for index in range(len(times) - step):
        elapsed = times[index + step] - times[index]
        box_height = float(np.median(height[index:index + step + 1]))
        if elapsed > 0 and box_height > 0:
            image_speed = abs(x[index + step] - x[index]) / elapsed
            speeds.append(image_speed * float(aspect_ratio) / box_height * STATURE_METRES)
    if len(speeds) < 2:
        return None
    values = np.asarray(speeds)
    third = max(1, len(values) // 3)
    speed_up = (float(np.median(values[-third:])) + 0.05) / (float(np.median(values[:third])) + 0.05)
    return float(np.median(values)), speed_up


def road_crossing_flags(
    track: pl.DataFrame,
    interval: Optional[RoadInterval],
    box_size_change_rate: Optional[float],
    minimum_x_range: float,
    surfaces: Optional[Sequence[str]] = None,
    fps: Optional[float] = None,
    aspect_ratio: Optional[float] = None,
) -> Dict[str, Any]:
    """Return rule D and its parts for one pedestrian track.

    ``box_size_change_rate`` is the track's ``log_height_rate_abs`` feature.
    When ``surfaces`` (the track's segmented surface labels), ``fps`` and
    ``aspect_ratio`` are given, the walking checks (walking_motion and the
    road share) are applied too; callers without them get the earlier rule.
    """
    x_range = on_road_x_range(track, interval)
    road_lateral = x_range is not None and x_range >= float(minimum_x_range)
    size_rate = None if box_size_change_rate is None else float(box_size_change_rate)
    slow_size_change = size_rate is not None and size_rate <= MAXIMUM_BOX_SIZE_CHANGE_RATE
    flags: Dict[str, Any] = {
        "on_road": int(interval is not None),
        "on_road_x_range": x_range,
        "road_lateral": int(road_lateral),
        "box_size_change_rate": size_rate,
        "road_crossing": int(road_lateral and slow_size_change),
    }
    if surfaces is not None and fps and aspect_ratio:
        labelled = [surface for surface in surfaces if surface in ("road", "footpath")]
        road_share = (sum(1 for surface in labelled if surface == "road") / len(labelled)) if labelled else 0.0
        motion = walking_motion(track, float(fps), float(aspect_ratio))
        walking = (
            motion is not None
            and MINIMUM_WALKING_SPEED <= motion[0] <= MAXIMUM_WALKING_SPEED
            and motion[1] <= MAXIMUM_SPEED_UP
        )
        flags.update(
            road_share=road_share,
            walking_speed_mps=None if motion is None else motion[0],
            speed_up=None if motion is None else motion[1],
            walking=int(walking),
        )
        flags["road_crossing"] = int(flags["road_crossing"] and walking and road_share >= MINIMUM_ROAD_SHARE)
    return flags


def longest_track_run(frames: Sequence[int], fps: float) -> Optional[Tuple[int, int]]:
    """Return the longest stretch of ``frames`` without a gap over two seconds.

    A tracker id can be reused for an unrelated object much later in a long
    CROWD segment. Treating each id as one track would merge them, so a
    candidate is limited to its longest continuous appearance.
    """
    values = sorted({int(value) for value in frames})
    if not values:
        return None
    from utils.crossing import track_joining

    # Joined tracks deliberately bridge gaps up to the joining limit (ids were
    # already split where the tracker reused them), so a candidate must not be
    # cut back into its pieces here.
    gap_seconds = TRACK_GAP_SECONDS
    if track_joining.TRACK_JOINING_ENABLED:
        gap_seconds = max(gap_seconds, track_joining.MAXIMUM_JOIN_GAP_SECONDS)
    maximum_gap = max(1, int(round(gap_seconds * float(fps))))
    best = (values[0], values[0])
    start = previous = values[0]
    for value in values[1:]:
        if value - previous > maximum_gap:
            start = value
        previous = value
        if previous - start > best[1] - best[0]:
            best = (start, previous)
    return best


def passes_camera(x_values: Sequence[float], left: float, right: float, tolerance: float = 0.0) -> bool:
    """Return whether a track crosses the camera's path, from one side to the other.

    Having the feet on the road is not enough to be crossing: the pedestrian
    has to pass in front of the camera. As in the CROWD detector, the box
    centre must go from left of the middle strip ``[left, right]`` through it
    to the right of it, or the other way round, in time order. ``tolerance``
    widens the strip edges into a band where the side does not change, so
    jitter at an edge is not a pass.
    """
    left_hard, left_soft = float(left) - float(tolerance), float(left) + float(tolerance)
    right_soft, right_hard = float(right) - float(tolerance), float(right) + float(tolerance)
    state, seen_left, seen_right = None, False, False
    for value in x_values:
        x = float(value)
        if state is None:
            state = 0 if x < left else (2 if x > right else 1)
        elif x <= left_hard:
            state = 0
        elif x >= right_hard:
            state = 2
        elif left_soft <= x <= right_soft:
            state = 1
        if state == 0:
            if seen_right:
                return True
            seen_left = True
        elif state == 2:
            if seen_left:
                return True
            seen_right = True
    return False


def independent_motion_rejection(
    window: pl.DataFrame,
    track_id: Any,
    fps: float,
) -> Optional[str]:
    """Return why a candidate is not a pedestrian crossing on foot, or None.

    ``window`` holds every detection (all classes) in the candidate's frames.
    Returns ``"rider"`` for someone on a bicycle or motorcycle and
    ``"camera_motion"`` when the track hardly moves relative to the static
    objects around it, using the CROWD detector's own checks and its
    frame-count thresholds scaled to ``fps``.
    """
    from utils.crossing.detection import Detection

    def scaled(frames: int, minimum: int) -> int:
        return Detection._scale_frames(frames, float(fps), DETECTOR_BASE_FPS, minimum=minimum)

    if Detection.is_rider_id(
        window,
        track_id,
        None,
        min_shared_frames=scaled(4, 1),
        min_continuous_shared_frames=scaled(12, 1),
        shared_run_gap_allow=scaled(2, 0),
        min_motion_steps=scaled(3, 1),
        short_shared_frames=scaled(8, 1),
    ):
        return "rider"
    reference = Detection.static_reference_motion_stats(
        window, track_id, MIN_SHARED_FRAMES=scaled(STATIC_REFERENCE_MIN_FRAMES, 1),
    )
    if (
        int(reference.get("shared_frames", 0) or 0) >= scaled(STATIC_REFERENCE_MIN_FRAMES, 1)
        and float(reference.get("relative_x_range", 0.0) or 0.0) < MINIMUM_RELATIVE_X_RANGE
    ):
        return "camera_motion"
    return None


def box_only_candidates(
    track_index: Mapping[Any, pl.DataFrame],
    features_by_track: Mapping[str, Any],
    fps: float,
    minimum_x_range: float,
    camera_strip: Optional[Tuple[float, float, float]] = None,
) -> Tuple[List[Any], Dict[Any, Tuple[int, int]], Dict[Any, float]]:
    """Return the tracks that could satisfy rule D, before any segmentation.

    Rule D needs the feet on the road, which only segmentation can tell. Its
    other two parts need nothing but the boxes, and a track has to satisfy
    them over its whole run to satisfy them while on the road, so they are
    checked first: on Waymo this keeps 560 of 574 rule-D tracks while cutting
    the tracks to segment to 4.5 per detector pick.

    ``features_by_track`` holds speed-model features computed on each track
    restricted to its candidate bounds, keyed by normalised id. Returns the
    candidate ids, their ``(start_frame, end_frame)`` bounds and their box
    size change rates. ``camera_strip`` is ``(left, right, tolerance)``; when
    given, a candidate must also pass in front of the camera (passes_camera).
    """
    candidates: List[Any] = []
    bounds: Dict[Any, Tuple[int, int]] = {}
    size_rates: Dict[Any, float] = {}
    minimum_rows = max(MINIMUM_TRACK_ROWS, int(round(MINIMUM_TRACK_SECONDS * float(fps))))
    for track_id, track in track_index.items():
        persons = track.filter(pl.col("yolo-id") == crossing_metrics.PERSON_CLASS_ID)
        run = longest_track_run(persons.get_column("frame-count").to_list(), fps) if persons.height else None
        if run is None:
            continue
        rows = persons.filter(pl.col("frame-count").is_between(*run))
        if rows.height < minimum_rows:
            continue
        x_range = float(rows["x-center"].max() - rows["x-center"].min())
        if camera_strip is not None:
            if not passes_camera(rows.sort("frame-count")["x-center"].to_list(), *camera_strip):
                continue
        features = features_by_track.get(crossing_metrics.normalise_id(track_id))
        if x_range < float(minimum_x_range) or features is None:
            continue
        rate = float(features.log_height_rate_abs)
        if rate > MAXIMUM_BOX_SIZE_CHANGE_RATE:
            continue
        candidates.append(track_id)
        bounds[track_id] = run
        size_rates[track_id] = rate
    return candidates, bounds, size_rates


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
