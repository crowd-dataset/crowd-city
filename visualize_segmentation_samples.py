"""Render sample crossing clips with the road-surface overlay and bounding boxes.

The persistent segmentation store (``seg_data``) keeps only a sparse per-track
index of derived surface labels, never the pixel masks that produced them, so
there is nothing in the store itself to look at. This script picks a handful
of already-processed crossings that the store reports a genuine on-road
interval for, re-decodes just their video window, re-runs the segmentation
model to get full-frame masks, and renders each one as an annotated video:
the road/footpath overlay plus the pedestrian bounding box(es), so the
segmentation quality can be checked by eye instead of trusted blindly.

Usage:
    python visualize_segmentation_samples.py --samples 5
    python visualize_segmentation_samples.py --stems VIDEOID_40_29 --output-dir out/
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import imageio.v2 as imageio
import numpy as np
import polars as pl

import common
from custom_logger import CustomLogger
from logmod import logs
from utils.analytics.parquet_store import configured_parquet_roots
from utils.crossing.metrics import ensure_waymo_processed, tuned_crossing_parameters
from utils.crossing.road_metrics import road_intervals_for_tracks
from utils.segmentation.constants import SURFACE_FOOTPATH, SURFACE_ROAD
from utils.segmentation.frames import (
    FrameClock,
    FrameWindow,
    RemoteCredentials,
    extract_window_frames,
    probe_video_fps,
    resolve_video_url,
)
from utils.segmentation.segformer import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_INPUT_HEIGHT,
    DEFAULT_INPUT_WIDTH,
    DEFAULT_MODEL_NAME,
    SurfaceSegmenter,
    segmentation_is_available,
)
from utils.segmentation.store import (
    INDEX_SCHEMA,
    SurfaceStore,
    configured_segmentation_root,
    crossing_fingerprint,
)
from utils.segmentation.surface import RoadInterval, SurfaceSample

logger = CustomLogger(__name__)

# BGR is irrelevant here: frames are RGB throughout, so these are plain RGB.
SURFACE_COLORS: Dict[str, Tuple[int, int, int]] = {
    SURFACE_ROAD: (220, 60, 60),      # red overlay = carriageway
    SURFACE_FOOTPATH: (60, 200, 90),  # green overlay = sidewalk / verge
}
OVERLAY_ALPHA = 0.40
BOX_COLORS: List[Tuple[int, int, int]] = [
    (255, 210, 0), (0, 200, 255), (255, 0, 200), (150, 255, 0),
]
WINDOW_PADDING_SECONDS = 2.5
MAXIMUM_CLIP_SECONDS = 30.0


@dataclass
class Sample:
    stem: str
    video_id: str
    start_seconds: float
    # Frame rate of the frame-count clock the store's intervals are on. It is
    # the source rate unless processing_fps downsampled the detections.
    detection_fps: float
    # Source frame rate as written in the stem, rounded to an integer.
    source_fps: float
    # Real frame rate the pipeline probed from the video, if recorded.
    video_fps: Optional[float]
    # Source frame the processing_fps resampling grid is anchored to.
    first_source_frame: int
    intervals: Dict[str, RoadInterval]


def _load_credentials() -> RemoteCredentials:
    def secret(name: str) -> Optional[str]:
        try:
            value = common.get_secrets(name)
        except Exception:
            return None
        text = str(value or "").strip()
        return text or None

    return RemoteCredentials(
        base_url=str(common.get_configs("ftp_base_url") or ""),
        username=secret("ftp_username"),
        password=secret("ftp_password"),
        token=secret("ftp_token"),
    )


def _current_crossing_fingerprint() -> str:
    """Return the crossing fingerprint for the configuration active right now.

    ``Detection.pedestrian_crossing`` already rejects riders and geometry
    false-positives before a track ever reaches ``pedestrian_ids``, and only
    those filtered ids are ever written into the segmentation store (see
    ``crossing_pass.run_segmentation_pass`` -> ``_crossing_tracks``), so every
    sample this script can find already passed those filters. The one gap is
    staleness: the store can hold manifests from a since-changed crossing
    configuration (parameters, confidence, boundaries), fingerprinted exactly
    so a stale index is not mistaken for a current one. This mirrors that
    fingerprint so discovery skips anything not produced by today's filters.
    """
    try:
        ensure_waymo_processed(
            raw_dataset_path=common.get_configs("waymo_dataset_path"),
            repository_root=common.root_dir,
            output_root=common.output_dir,
            process_if_missing=bool(common.get_configs("process_waymo_if_missing")),
            log=lambda message: logger.debug(message),
        )
    except Exception as error:
        logger.warning(f"Could not resolve the frozen Waymo crossing parameters: {error}")

    return crossing_fingerprint(
        {
            "crossing_parameters": dict(tuned_crossing_parameters() or {}),
            "min_confidence": common.get_configs("min_confidence") or 0.7,
            "boundary_left": common.get_configs("boundary_left") or 0.45,
            "boundary_right": common.get_configs("boundary_right") or 0.55,
            "processing_fps": common.get_configs("processing_fps"),
        }
    )


def _discover_samples(store: SurfaceStore, limit: Optional[int]) -> List[Sample]:
    """Return manifests that the store reports a genuine on-road interval for.

    A manifest alone does not say whether any track actually reached the
    carriageway, only which tracks were asked about. Deriving the interval
    here, the same way the metrics themselves do, filters out clips where the
    pedestrian never stepped onto the road before any video is decoded.

    Candidates are scanned from a shuffled, bounded pool and then sorted by
    how many tracks share the clip, so a handful of clean one- or two-person
    crossings are preferred over a busy intersection with dozens of
    simultaneous tracks that would be hard to read as a "sample". Manifests
    whose crossing fingerprint does not match the current configuration are
    skipped, since they were filtered by parameters no longer in effect.
    """
    current_fingerprint = _current_crossing_fingerprint()
    manifest_paths = sorted(store.manifest_dir.glob("*.json"))
    random.shuffle(manifest_paths)
    scan_pool = manifest_paths if not limit else manifest_paths[: max(limit * 20, 200)]

    samples: List[Sample] = []
    for path in scan_pool:
        stem = path.stem
        manifest = store.read_manifest(stem)
        if not manifest:
            continue
        # Older schemas mapped frames to time with the rounded file-name rate,
        # so their intervals were derived from misplaced boxes.
        if manifest.get("schema") != INDEX_SCHEMA:
            continue
        stored_fingerprint = (manifest.get("settings") or {}).get("crossing_fingerprint")
        if stored_fingerprint != current_fingerprint:
            continue
        index = store.read_index(stem)
        if index is None or index.height == 0:
            continue

        timelines: Dict[str, List[SurfaceSample]] = {}
        for row in index.sort("frame-count").iter_rows(named=True):
            timelines.setdefault(str(row["unique-id"]), []).append(
                SurfaceSample(
                    track_id=str(row["unique-id"]),
                    frame=int(row["frame-count"]),
                    video_time_s=float(row["video_time_s"]),
                    surface=str(row["surface"]),
                    confidence=float(row["confidence"]),
                )
            )
        intervals = road_intervals_for_tracks(timelines)
        if not intervals:
            continue

        try:
            detection_fps = float(manifest["detection_fps"])
            video_fps = manifest.get("video_fps")
            samples.append(
                Sample(
                    stem=stem,
                    video_id=str(manifest["video_id"]),
                    start_seconds=float(manifest["start_seconds"]),
                    detection_fps=detection_fps,
                    source_fps=float(manifest.get("source_fps") or _stem_fps(stem, detection_fps)),
                    video_fps=float(video_fps) if video_fps else None,
                    first_source_frame=int(manifest.get("first_source_frame") or 0),
                    intervals=intervals,
                )
            )
        except (KeyError, TypeError, ValueError) as error:
            logger.warning(f"Skipping manifest {stem}: {error}")
            continue

    if limit:
        samples.sort(key=lambda sample: len(sample.intervals))
        return samples[:limit]
    return samples


def _stem_fps(stem: str, fallback: float) -> float:
    # Stems are VIDEOID_START_FPS; the video id itself may contain underscores.
    try:
        fps = float(stem.rsplit("_", 2)[2])
    except (IndexError, ValueError):
        return fallback
    return fps if fps > 0 else fallback


def _find_detection_file(stem: str) -> Optional[Path]:
    for root in configured_parquet_roots():
        candidate = Path(root) / "bbox" / f"{stem}.parquet"
        if candidate.is_file():
            return candidate
    return None


def _load_track_boxes(
    detection_path: Path,
    track_ids: Sequence[str],
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Return {track_id: (frames, xywh)} for the wanted tracks."""
    wanted = {str(value) for value in track_ids}
    lazy = pl.scan_parquet(detection_path)
    columns = set(lazy.collect_schema().names())
    required = {"unique-id", "frame-count", "x-center", "y-center", "width", "height"}
    if not required.issubset(columns):
        return {}

    df = (
        lazy
        .filter(pl.col("unique-id").cast(pl.Utf8).is_in(list(wanted)))
        .select(sorted(required))
        .collect()
    )
    boxes: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for track in df.partition_by("unique-id", maintain_order=True):
        if track.height == 0:
            continue
        track_id = str(track.get_column("unique-id")[0])
        ordered = track.sort("frame-count")
        frames = ordered.get_column("frame-count").cast(pl.Int64, strict=False).to_numpy()
        geometry = ordered.select(
            ["x-center", "y-center", "width", "height"]
        ).cast(pl.Float64, strict=False).to_numpy()
        boxes[track_id] = (frames, geometry)
    return boxes


def _nearest_row(frames: np.ndarray, target: float, tolerance: float) -> Optional[int]:
    if frames.size == 0:
        return None
    position = int(np.searchsorted(frames, target))
    candidates = [index for index in (position - 1, position) if 0 <= index < frames.size]
    if not candidates:
        return None
    best = min(candidates, key=lambda index: abs(float(frames[index]) - target))
    return best if abs(float(frames[best]) - target) <= tolerance else None


def _draw_overlay(
    frame: np.ndarray,
    labels_lowres: np.ndarray,
    surface_for_train_id,
) -> np.ndarray:
    """Blend the road/footpath mask onto ``frame`` (both already RGB uint8)."""
    height, width = frame.shape[:2]
    labels = cv2.resize(
        labels_lowres.astype(np.int32), (width, height), interpolation=cv2.INTER_NEAREST,
    )
    out = frame.copy()
    for surface, color in SURFACE_COLORS.items():
        mask = np.zeros(labels.shape, dtype=bool)
        for train_id in range(int(labels.max()) + 1 if labels.size else 0):
            if surface_for_train_id(train_id) == surface:
                mask |= labels == train_id
        if not mask.any():
            continue
        tint = np.array(color, dtype=np.float32)
        out[mask] = (
            (1 - OVERLAY_ALPHA) * out[mask].astype(np.float32) + OVERLAY_ALPHA * tint
        ).astype(np.uint8)
    return out


def _draw_boxes(
    frame: np.ndarray,
    boxes_here: List[Tuple[str, float, float, float, float]],
) -> np.ndarray:
    height, width = frame.shape[:2]
    for index, (track_id, x_center, y_center, box_width, box_height) in enumerate(boxes_here):
        color = BOX_COLORS[index % len(BOX_COLORS)]
        x1 = int((x_center - box_width / 2) * width)
        y1 = int((y_center - box_height / 2) * height)
        x2 = int((x_center + box_width / 2) * width)
        y2 = int((y_center + box_height / 2) * height)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame, f"id {track_id}", (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
        )
    return frame


def _draw_legend(frame: np.ndarray) -> np.ndarray:
    entries = [("road", SURFACE_COLORS[SURFACE_ROAD]), ("footpath", SURFACE_COLORS[SURFACE_FOOTPATH])]
    x, y = 10, 10
    for label, color in entries:
        cv2.rectangle(frame, (x, y), (x + 18, y + 18), color, -1)
        cv2.putText(
            frame, label, (x + 24, y + 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
        )
        y += 24
    return frame


def render_sample(
    sample: Sample,
    segmenter: SurfaceSegmenter,
    credentials: RemoteCredentials,
    output_dir: Path,
    cadence_hz: float,
) -> Optional[Path]:
    from utils.segmentation.constants import surface_for_train_id

    source = resolve_video_url(sample.video_id, credentials)
    if source is None:
        logger.warning(f"{sample.stem}: could not resolve {sample.video_id} on the file server.")
        return None

    # Use the rate the pipeline probed when it wrote these labels, so boxes are
    # placed on the same clock the surface was sampled on. The integer rate in
    # the stem is only a last resort (29.97 is stored there as 30).
    video_fps = sample.video_fps or probe_video_fps(source, credentials)
    if video_fps is None:
        video_fps = sample.source_fps
        logger.warning(
            f"{sample.stem}: could not probe the video frame rate; "
            f"falling back to {video_fps:g} fps from the file name."
        )
    elif abs(video_fps - sample.source_fps) > 1e-3:
        logger.info(
            f"{sample.stem}: video runs at {video_fps:.3f} fps, "
            f"file name says {sample.source_fps:g}; using the video rate."
        )

    # The store's intervals are on the analysis clock, which differs from the
    # raw detection frame-count only when processing_fps downsampled it.
    analysis_clock = FrameClock.for_segment(
        start_seconds=sample.start_seconds,
        video_fps=video_fps,
        source_fps=sample.source_fps,
        detection_fps=sample.detection_fps,
        first_source_frame=sample.first_source_frame,
    )
    # The boxes are read straight from the detection file, on raw frames.
    raw_clock = FrameClock(start_seconds=sample.start_seconds, video_fps=video_fps)

    entry = min(interval.entry_frame for interval in sample.intervals.values())
    exit_ = max(interval.exit_frame for interval in sample.intervals.values())
    window_start = analysis_clock.seconds(entry) - WINDOW_PADDING_SECONDS
    window_end = analysis_clock.seconds(exit_) + WINDOW_PADDING_SECONDS
    duration = min(max(window_end - window_start, 1.0), MAXIMUM_CLIP_SECONDS)
    window = FrameWindow(start_seconds=max(0.0, window_start), duration_seconds=duration)

    frames = extract_window_frames(
        source, window, cadence_hz, segmenter.input_width, segmenter.input_height,
        credentials=credentials,
    )
    if len(frames) == 0:
        logger.warning(f"{sample.stem}: no frames decoded for the crossing window.")
        return None

    labels, _confidence = segmenter.segment(frames)

    detection_path = _find_detection_file(sample.stem)
    track_boxes: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    if detection_path is not None:
        track_boxes = _load_track_boxes(detection_path, list(sample.intervals.keys()))
    else:
        logger.warning(f"{sample.stem}: detection file not found; rendering without boxes.")

    tolerance = video_fps / (2.0 * cadence_hz)
    output_path = output_dir / f"{sample.stem}.mp4"
    with imageio.get_writer(output_path, fps=cadence_hz, codec="libx264", quality=8) as writer:
        for offset in range(len(frames)):
            video_time = window.start_seconds + offset / cadence_hz
            detection_frame = raw_clock.frame(video_time)

            rendered = _draw_overlay(frames[offset], labels[offset], surface_for_train_id)
            boxes_here = []
            for track_id, (track_frames, geometry) in track_boxes.items():
                matched = _nearest_row(track_frames, detection_frame, tolerance)
                if matched is None:
                    continue
                x_center, y_center, box_width, box_height = geometry[matched]
                boxes_here.append((track_id, x_center, y_center, box_width, box_height))
            rendered = _draw_boxes(rendered, boxes_here)
            rendered = _draw_legend(rendered)
            writer.append_data(rendered)

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=5, help="Number of clips to render.")
    parser.add_argument(
        "--stems", nargs="*", default=None,
        help="Render exactly these detection-segment stems instead of sampling randomly.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path(common.output_dir) / "segmentation_samples",
        help="Directory the rendered .mp4 clips are written to.",
    )
    parser.add_argument("--cadence-hz", type=float, default=5.0, help="Playback frame rate of the rendered clips.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for sample selection.")
    args = parser.parse_args()

    logs(show_level=common.get_configs("logger_level"), show_color=True)

    if not segmentation_is_available():
        logger.error("torch/transformers are unavailable; cannot run the segmentation model.")
        return

    root = configured_segmentation_root()
    if not root:
        logger.error("'seg_data' is not configured; nothing to visualise.")
        return
    store = SurfaceStore(root)
    if not store.available:
        logger.error(f"Segmentation store {root} could not be opened.")
        return

    if args.seed is not None:
        random.seed(args.seed)

    if args.stems:
        samples: List[Sample] = []
        for stem in args.stems:
            found = _discover_samples(store, limit=None)
            samples.extend(sample for sample in found if sample.stem == stem)
    else:
        samples = _discover_samples(store, limit=args.samples)

    if not samples:
        logger.error(
            "No processed crossings with a genuine on-road interval, matching the "
            f"current crossing configuration, were found in the segmentation store at "
            f"{root}. Either run analysis.py with use_segmentation enabled first, or "
            "the store only holds entries from a since-changed crossing configuration."
        )
        return

    logger.info(f"Rendering {len(samples)} sample clip(s).")

    segmenter = SurfaceSegmenter(
        model_name=str(common.get_configs("segmentation_model") or DEFAULT_MODEL_NAME),
        device=str(common.get_configs("segmentation_device") or "auto"),
        batch_size=int(common.get_configs("segmentation_batch_size") or DEFAULT_BATCH_SIZE),
        input_width=int(common.get_configs("segmentation_input_width") or DEFAULT_INPUT_WIDTH),
        input_height=int(common.get_configs("segmentation_input_height") or DEFAULT_INPUT_HEIGHT),
    )
    credentials = _load_credentials()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    written: List[Path] = []
    for sample in samples:
        try:
            output_path = render_sample(sample, segmenter, credentials, args.output_dir, args.cadence_hz)
        except Exception as error:
            logger.warning(f"{sample.stem}: rendering failed: {error}")
            continue
        if output_path is not None:
            logger.info(f"Wrote {output_path}")
            written.append(output_path)

    logger.info(f"Done: {len(written)}/{len(samples)} clip(s) written to {args.output_dir}.")


if __name__ == "__main__":
    main()
