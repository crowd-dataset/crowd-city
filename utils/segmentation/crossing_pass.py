"""Run road-surface segmentation over the detected crossings of one analysis.

This runs as a single sequential pass in the parent process, after the parallel
detection pass has decided which tracks are crossings. Segmentation is a GPU
workload with a large model, so loading it once here is far cheaper than
loading it inside every detection worker, and the pass only ever touches
segments that actually contain a crossing.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import polars as pl
from tqdm import tqdm

import common
from custom_logger import CustomLogger
import utils.analytics.csv_parallel as csv_parallel
import utils.crossing.metrics as crossing_metrics
from utils.analytics.csv_parallel import (
    _limit_detection_duration,
    _read_confidence_filtered,
    _resample_detection_fps,
    initialise_csv_worker,
)
from utils.core.grouping import Grouping
from utils.crossing.road_metrics import (
    hesitation_seconds,
    road_intervals_for_tracks,
    road_restricted_speed,
)
from utils.segmentation.frames import RemoteCredentials
from utils.segmentation.pipeline import SegmentationPipeline, SegmentRequest
from utils.segmentation.segformer import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_INPUT_HEIGHT,
    DEFAULT_INPUT_WIDTH,
    DEFAULT_MODEL_NAME,
    SurfaceSegmenter,
    segmentation_is_available,
)
from utils.segmentation.store import (
    SegmentationSettings,
    SurfaceStore,
    configured_segmentation_root,
    crossing_fingerprint,
)
from utils.segmentation.surface import (
    DEFAULT_FOOTPOINT_BAND_FRACTION,
    DEFAULT_FOOTPOINT_WIDTH_FRACTION,
    DEFAULT_MINIMUM_CONFIDENCE,
)


logger = CustomLogger(__name__)
grouping = Grouping()


def _config(name: str, default: Any) -> Any:
    try:
        value = common.get_configs(name)
    except Exception:
        return default
    return default if value is None else value


def _secret(name: str) -> Optional[str]:
    try:
        value = common.get_secrets(name)
    except Exception:
        return None
    text = str(value or "").strip()
    return text or None


def _crossing_tracks(
    df: pl.DataFrame,
    track_ids: Sequence[Any],
    id_bounds: Optional[Mapping[Any, Tuple[int, int]]] = None,
) -> Dict[str, pl.DataFrame]:
    """Return the detection rows of each crossing track, keyed by track id.

    A tracker id can be reused for an unrelated object elsewhere in the same
    video, so ``id_bounds`` (the accepted segment's own frame range, from
    ``Detection.pedestrian_crossing``) restricts each track to the frames it
    actually crossed on when known, instead of every row sharing that id.
    """
    if df.height == 0 or not track_ids:
        return {}
    wanted = {str(value) for value in track_ids}
    bounds = {str(key): value for key, value in (id_bounds or {}).items()}
    tracks: Dict[str, pl.DataFrame] = {}
    for track in df.partition_by("unique-id", maintain_order=True):
        if track.height == 0:
            continue
        track_id = str(track.get_column("unique-id")[0])
        if track_id not in wanted:
            continue
        track = track.sort("frame-count")
        track_bounds = bounds.get(track_id)
        if track_bounds is not None and "frame-count" in track.columns:
            start_frame, end_frame = track_bounds
            restricted = track.filter(
                pl.col("frame-count").cast(pl.Int64, strict=False).is_between(
                    start_frame, end_frame,
                )
            )
            if restricted.height > 0:
                track = restricted
        tracks[track_id] = track
    return tracks


def run_segmentation_pass(
    df_mapping: pl.DataFrame,
    detection_tasks: Sequence[Mapping[str, Any]],
    crossing_ids: Mapping[str, Mapping[str, Any]],
    crossing_parameters: Mapping[str, Any],
) -> Dict[str, Any]:
    """Return road-restricted speed and hesitation time, wrapped by locality.

    Both outputs use the same nested ``{locality_condition: {video: {track:
    value}}}`` shape as the baseline metrics, so the existing aggregation and
    plotting paths can consume them unchanged. The hesitation values are in
    seconds rather than sample counts.
    """
    empty: Dict[str, Any] = {
        "seg_speed": {},
        "seg_time": {},
        "diagnostics": Counter(),
        "enabled": False,
    }

    if not bool(_config("use_segmentation", False)):
        logger.info("Segmentation-based crossing metrics are disabled in the configuration.")
        return empty

    if not segmentation_is_available():
        logger.error(
            "use_segmentation is enabled but torch/transformers are unavailable; "
            "skipping the segmentation pass."
        )
        return empty

    root = configured_segmentation_root()
    if not root:
        logger.error(
            "use_segmentation is enabled but 'seg_data' is not configured; "
            "skipping the segmentation pass."
        )
        return empty

    store = SurfaceStore(root)
    if not store.available:
        logger.error(f"Segmentation store {root} could not be opened; skipping the pass.")
        return empty

    coarse_hz = float(_config("segmentation_coarse_hz", 1.0))
    refine_hz = float(_config("segmentation_refine_hz", 4.0))
    segmenter = SurfaceSegmenter(
        model_name=str(_config("segmentation_model", DEFAULT_MODEL_NAME)),
        device=str(_config("segmentation_device", "auto")),
        batch_size=int(_config("segmentation_batch_size", DEFAULT_BATCH_SIZE)),
        input_width=int(_config("segmentation_input_width", DEFAULT_INPUT_WIDTH)),
        input_height=int(_config("segmentation_input_height", DEFAULT_INPUT_HEIGHT)),
    )

    minimum_confidence = float(
        _config("segmentation_min_confidence", DEFAULT_MINIMUM_CONFIDENCE)
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
                "crossing_parameters": dict(crossing_parameters or {}),
                "min_confidence": _config("min_confidence", 0.7),
                "boundary_left": _config("boundary_left", 0.45),
                "boundary_right": _config("boundary_right", 0.55),
                "processing_fps": _config("processing_fps", None),
            }
        ),
    )

    # The detection-file readers below are the ones the parallel detection
    # workers use, and they read process-local state that only
    # initialise_csv_worker sets. Without this the parent would read every
    # detection file at confidence 0.0 while the detection pass used the
    # configured threshold, so the rows behind the crossing tracks would not
    # be the rows the crossings were decided from.
    initialise_csv_worker(
        df_mapping,
        dict(crossing_parameters or {}),
        float(_config("min_confidence", 0.7)),
        float(_config("boundary_left", 0.45)),
        float(_config("boundary_right", 0.55)),
        dict(getattr(crossing_metrics, "_PIPELINE_MODEL", {}) or {}),
        dict(getattr(crossing_metrics, "_SPEED_MODEL", {}) or {}),
    )

    credentials = RemoteCredentials(
        base_url=str(_config("ftp_base_url", "")),
        username=_secret("ftp_username"),
        password=_secret("ftp_password"),
        token=_secret("ftp_token"),
    )

    pipeline = SegmentationPipeline(
        store=store,
        segmenter=segmenter,
        settings=settings,
        credentials=credentials,
        coarse_hz=coarse_hz,
        refine_hz=refine_hz,
        minimum_confidence=minimum_confidence,
    )

    tasks_by_stem = {
        str(task["filename_no_ext"]): task
        for task in detection_tasks
    }
    pending = [
        stem
        for stem, payload in (crossing_ids or {}).items()
        if (payload or {}).get("ids") and stem in tasks_by_stem
    ]
    if not pending:
        logger.warning("No detected crossings available for the segmentation pass.")
        pipeline.close()
        return empty

    logger.info(
        f"Segmenting {len(pending)} detection segment(s) with crossings: "
        f"a {coarse_hz:g} Hz pass to find each road entry and exit, then "
        f"{refine_hz:g} Hz only inside those transitions. "
        f"Model {segmenter.model_identifier} on {segmenter.device}."
    )

    checks_per_second = float(_config("check_per_sec_time", 3))
    raw_speed: Dict[str, Dict[str, float]] = {}
    raw_time: Dict[str, Dict[str, float]] = {}
    diagnostics: Counter = Counter()

    try:
        for stem in tqdm(sorted(pending), desc="Segmenting crossing windows"):
            task = tasks_by_stem[stem]
            try:
                result = _process_segment(
                    df_mapping=df_mapping,
                    pipeline=pipeline,
                    task=task,
                    stem=stem,
                    track_ids=list(crossing_ids[stem]["ids"]),
                    checks_per_second=checks_per_second,
                    id_bounds=crossing_ids[stem].get("id_bounds") or {},
                )
            except Exception as error:
                logger.warning(f"Segmentation failed for {stem}: {error}")
                diagnostics["segment_exception"] += 1
                continue

            if result is None:
                continue

            speed_values, time_values, segment_diagnostics = result
            diagnostics.update(segment_diagnostics)
            if speed_values:
                raw_speed[stem] = speed_values
            if time_values:
                raw_time[stem] = time_values
    finally:
        pipeline.close()

    diagnostics.update(pipeline.statistics)

    seg_speed = grouping.locality_country_wrapper(raw_speed, df_mapping) if raw_speed else {}
    seg_time = grouping.locality_country_wrapper(raw_time, df_mapping) if raw_time else {}

    logger.info(
        "Segmentation pass complete: "
        f"{pipeline.statistics['segments_reused']} segment(s) reused from the store, "
        f"{pipeline.statistics['segments_segmented']} segmented, "
        f"{pipeline.statistics['segments_failed']} failed, "
        f"{pipeline.statistics['frames_segmented']} frame(s) through the model "
        f"({pipeline.statistics['frames_coarse']} locating transitions, "
        f"{pipeline.statistics['frames_refine']} refining them)."
    )
    logger.info(
        f"Road-restricted speed produced for {sum(len(v) for v in raw_speed.values())} track(s); "
        f"hesitation time for {sum(len(v) for v in raw_time.values())} track(s)."
    )
    for reason, count in sorted(diagnostics.items(), key=lambda item: -item[1]):
        logger.info(f"Segmentation diagnostic: {reason}={count}.")

    return {
        "seg_speed": seg_speed,
        "seg_time": seg_time,
        "diagnostics": diagnostics,
        "enabled": True,
    }


def _process_segment(
    df_mapping: pl.DataFrame,
    pipeline: SegmentationPipeline,
    task: Mapping[str, Any],
    stem: str,
    track_ids: List[Any],
    checks_per_second: float,
    id_bounds: Optional[Mapping[Any, Tuple[int, int]]] = None,
) -> Optional[Tuple[Dict[str, float], Dict[str, float], Counter]]:
    """Segment one detection segment and derive both metrics from it."""
    diagnostics: Counter = Counter()

    # The detection rows must be prepared exactly as the detection worker
    # prepared them, otherwise the frame numbers the crossing ids refer to
    # would not line up with the video timestamps derived here.
    detections = _read_confidence_filtered(str(task["file_path"]))
    detections = _limit_detection_duration(
        detections,
        float(task.get("time_video", 0.0) or 0.0),
        float(task["fps"]),
    )
    detections, effective_fps = _resample_detection_fps(
        detections,
        float(task["fps"]),
        csv_parallel._WORKER_PROCESSING_FPS,
    )
    if detections.height == 0:
        diagnostics["empty_detection_file"] += 1
        return None

    tracks = _crossing_tracks(detections, track_ids, id_bounds)
    if not tracks:
        diagnostics["crossing_tracks_missing"] += 1
        return None

    timelines = pipeline.timelines(
        SegmentRequest(
            stem=stem,
            video_id=str(task["video_id"]),
            start_seconds=float(task["start_index"]),
            detection_fps=float(effective_fps),
            tracks=tracks,
        )
    )
    if not timelines:
        diagnostics["no_surface_timeline"] += 1
        return None

    intervals = road_intervals_for_tracks(timelines)
    diagnostics["tracks_without_road_interval"] += len(tracks) - len(intervals)
    if not intervals:
        return None

    time_values: Dict[str, float] = {}
    for track_id, interval in intervals.items():
        wait = hesitation_seconds(
            tracks[track_id],
            interval.entry_frame,
            float(effective_fps),
            checks_per_second,
        )
        if wait is None:
            # The track began with the pedestrian already on the road, so the
            # wait was never in view. Counted, so the share of crossings that
            # cannot contribute a hesitation time stays visible in the log.
            diagnostics["hesitation_unobservable"] += 1
            continue
        time_values[track_id] = float(wait)

    speed_values, speed_diagnostics = road_restricted_speed(
        df_mapping,
        detections,
        stem,
        float(effective_fps),
        intervals,
    )
    diagnostics.update(speed_diagnostics)

    return speed_values, time_values, diagnostics


# ---------------------------------------------------------------------
# Aggregation
#
# The baseline hesitation metric stores sample counts and divides by
# check_per_sec_time when averaging. The segmentation metric stores seconds
# directly, so it needs its own aggregation rather than the baseline helpers.
# ---------------------------------------------------------------------

def average_by_locality(
    nested: Mapping[str, Mapping[str, Mapping[str, float]]],
    minimum: float = 0.0,
    maximum: float = float("inf"),
) -> Tuple[Dict[str, float], Dict[str, List[float]]]:
    """Average per-track values by locality and condition."""
    averages: Dict[str, float] = {}
    complete: Dict[str, List[float]] = {}
    for locality, videos in (nested or {}).items():
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


def average_by_country(
    df_mapping: pl.DataFrame,
    nested: Mapping[str, Mapping[str, Mapping[str, float]]],
    minimum: float = 0.0,
    maximum: float = float("inf"),
) -> Tuple[Dict[str, float], Dict[str, List[float]]]:
    """Average per-track values by country and condition."""
    from utils.core.metadata import MetaData

    metadata = MetaData()
    grouped: Dict[str, List[float]] = {}
    for videos in (nested or {}).values():
        for video_id, tracks in videos.items():
            result = metadata.find_values_with_video_id(df_mapping, video_id)
            if result is None:
                continue
            key = f"{result[8]}_{result[3]}"
            for value in tracks.values():
                try:
                    numeric = float(value)
                except (TypeError, ValueError):
                    continue
                if minimum <= numeric <= maximum:
                    grouped.setdefault(key, []).append(numeric)
    averages = {
        key: sum(values) / len(values)
        for key, values in grouped.items()
        if values
    }
    return averages, grouped
