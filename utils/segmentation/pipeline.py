"""Produce, cache and reuse per-track road-surface timelines for one segment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import polars as pl
import requests

from custom_logger import CustomLogger
from utils.segmentation.frames import (
    FrameWindow,
    RemoteCredentials,
    extract_window_frames,
    resolve_video_url,
)
from utils.segmentation.segformer import SurfaceSegmenter
from utils.segmentation.store import SegmentationSettings, SurfaceStore
from utils.segmentation.surface import (
    DEFAULT_FOOTPOINT_BAND_FRACTION,
    DEFAULT_FOOTPOINT_WIDTH_FRACTION,
    DEFAULT_MINIMUM_CONFIDENCE,
    SurfaceSample,
    sample_surface,
    surface_timeline,
    transition_brackets,
)


logger = CustomLogger(__name__)

# Windows closer together than this are decoded as one ffmpeg call, because a
# seek costs more than a second of extra decoding.
WINDOW_MERGE_GAP_SECONDS = 2.0
# Extra footage kept before and after a track so the approach to the kerb and
# the arrival at the far side are both visible.
WINDOW_PADDING_SECONDS = 2.0
# A tracker id can be reused for an unrelated object much later in the same
# video. A gap this large within one track's own frames is treated as such a
# reuse rather than a continuous presence, so its span is not stretched across
# the gap; splitting on it keeps a single stale id from producing an
# hours-long decode window.
TRACK_SPAN_GAP_SECONDS = 2.0
# Hard ceiling on one ffmpeg window, applied after splitting and merging.
# No real pedestrian crossing (plus its padded approach and departure) lasts
# this long; a window still this large is a sign the id-reuse split above
# missed a case, and decoding it would risk an ffmpeg timeout or, for a
# window that does complete, materialising tens of gigabytes of raw frames in
# memory at once.
MAXIMUM_WINDOW_SECONDS = 120.0
# Widen each refinement bracket slightly so the transition cannot sit exactly
# on its edge and be missed by rounding.
BRACKET_PADDING_SECONDS = 0.25


@dataclass
class SegmentRequest:
    """One detection segment whose crossing tracks need surface labels."""

    stem: str
    video_id: str
    start_seconds: float
    detection_fps: float
    tracks: Dict[str, pl.DataFrame]


class SegmentationPipeline:
    """Fill the surface store on demand and serve timelines from it."""

    def __init__(
        self,
        store: SurfaceStore,
        segmenter: SurfaceSegmenter,
        settings: SegmentationSettings,
        credentials: Optional[RemoteCredentials] = None,
        coarse_hz: float = 1.0,
        refine_hz: float = 4.0,
        band_fraction: float = DEFAULT_FOOTPOINT_BAND_FRACTION,
        width_fraction: float = DEFAULT_FOOTPOINT_WIDTH_FRACTION,
        minimum_confidence: float = DEFAULT_MINIMUM_CONFIDENCE,
    ) -> None:
        self.store = store
        self.segmenter = segmenter
        self.settings = settings
        self.credentials = credentials
        self.coarse_hz = float(coarse_hz)
        self.refine_hz = float(refine_hz)
        self.band_fraction = float(band_fraction)
        self.width_fraction = float(width_fraction)
        self.minimum_confidence = float(minimum_confidence)
        self._session: Optional[requests.Session] = None
        self._last_frame_count = 0
        self._unresolved: set[str] = set()
        self.statistics: Dict[str, int] = {
            "segments_reused": 0,
            "segments_segmented": 0,
            "segments_failed": 0,
            "frames_coarse": 0,
            "frames_refine": 0,
            "frames_segmented": 0,
            "videos_unresolved": 0,
            "windows_too_long_dropped": 0,
        }

    # ------------------------------------------------------------------
    # Session handling
    # ------------------------------------------------------------------
    def _http_session(self) -> requests.Session:
        if self._session is None:
            session = requests.Session()
            if self.credentials and self.credentials.username and self.credentials.password:
                session.auth = (self.credentials.username, self.credentials.password)
            session.headers.update({"User-Agent": "crowd-city-segmentation/1.0"})
            self._session = session
        return self._session

    def close(self) -> None:
        if self._session is not None:
            self._session.close()
            self._session = None

    def _video_source(self, video_id: str) -> Optional[str]:
        cached = self.store.cached_url(video_id)
        if cached:
            return cached
        if video_id in self._unresolved or self.credentials is None:
            return None
        url = resolve_video_url(
            video_id,
            self.credentials,
            session=self._http_session(),
        )
        if url is None:
            logger.warning(f"Could not locate video {video_id} on the file server.")
            self._unresolved.add(video_id)
            self.statistics["videos_unresolved"] += 1
            return None
        self.store.store_url(video_id, url)
        return url

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def timelines(self, request: SegmentRequest) -> Optional[Dict[str, List[SurfaceSample]]]:
        """Return surface samples per track, segmenting only when necessary."""
        track_ids = [str(key) for key in request.tracks]
        if not track_ids:
            return {}

        if self.store.is_current(request.stem, self.settings, track_ids):
            stored = self.store.read_index(request.stem)
            if stored is not None:
                self.statistics["segments_reused"] += 1
                return self._timelines_from_index(stored, track_ids)

        samples = self._segment_request(request)
        if samples is None:
            self.statistics["segments_failed"] += 1
            return None

        self.store.write_index(request.stem, [sample.as_row() for sample in samples])
        self.store.write_manifest(
            request.stem,
            self.settings,
            track_ids,
            extra={
                "video_id": request.video_id,
                "start_seconds": float(request.start_seconds),
                "detection_fps": float(request.detection_fps),
                "sample_count": len(samples),
            },
        )
        self.statistics["segments_segmented"] += 1
        return surface_timeline(samples)

    @staticmethod
    def _timelines_from_index(
        index: pl.DataFrame,
        track_ids: Sequence[str],
    ) -> Dict[str, List[SurfaceSample]]:
        wanted = {str(value) for value in track_ids}
        samples: List[SurfaceSample] = []
        for row in index.iter_rows(named=True):
            track_id = str(row["unique-id"])
            if track_id not in wanted:
                continue
            samples.append(
                SurfaceSample(
                    track_id=track_id,
                    frame=int(row["frame-count"]),
                    video_time_s=float(row["video_time_s"]),
                    surface=str(row["surface"]),
                    confidence=float(row["confidence"]),
                )
            )
        return surface_timeline(samples)

    # ------------------------------------------------------------------
    # Segmentation
    # ------------------------------------------------------------------
    def _segment_request(self, request: SegmentRequest) -> Optional[List[SurfaceSample]]:
        """Locate each track's road entry and exit, segmenting as little as possible.

        Nothing downstream needs a surface label for every frame. The crossing
        speed needs the frames between road entry and road exit, and the
        hesitation time needs the moment of road entry and nothing after it.
        So a coarse pass establishes the shape of each track, and a second
        pass looks only inside the brackets where a transition must lie.
        """
        source = self._video_source(request.video_id)
        if source is None:
            return None

        boxes = self._boxes_by_track(request)
        if not boxes:
            return None

        coarse_windows = self._merged_windows(request, boxes)
        if not coarse_windows:
            return None

        samples = self._sample_windows(
            source, coarse_windows, self.coarse_hz, boxes, request,
        )
        self.statistics["frames_coarse"] += self._last_frame_count

        refine_windows = self._refinement_windows(samples, request)
        if refine_windows:
            samples.extend(
                self._sample_windows(
                    source, refine_windows, self.refine_hz, boxes, request,
                )
            )
            self.statistics["frames_refine"] += self._last_frame_count

        return samples

    def _refinement_windows(
        self,
        samples: List[SurfaceSample],
        request: SegmentRequest,
    ) -> List[FrameWindow]:
        """Return the short spans that must be looked at more closely.

        One span per transition per track, merged across tracks so that
        pedestrians crossing at the same moment share the same decoded frames.
        """
        fps = float(request.detection_fps)
        if fps <= 0:
            return []

        spans: List[Tuple[float, float]] = []
        for track_samples in surface_timeline(samples).values():
            entry, exit_bracket = transition_brackets(
                [sample.frame for sample in track_samples],
                [sample.surface for sample in track_samples],
            )
            for bracket in (entry, exit_bracket):
                if bracket is None:
                    continue
                low, high = bracket
                start = request.start_seconds + float(low) / fps
                end = request.start_seconds + float(high) / fps
                spans.append(
                    (
                        max(0.0, start - BRACKET_PADDING_SECONDS),
                        end + BRACKET_PADDING_SECONDS,
                    )
                )

        return self._merge_spans(spans)

    def _sample_windows(
        self,
        source: str,
        windows: Sequence[FrameWindow],
        cadence_hz: float,
        boxes: Dict[str, Tuple[np.ndarray, Dict[int, Tuple[float, float, float, float]]]],
        request: SegmentRequest,
    ) -> List[SurfaceSample]:
        """Decode, segment and read the footpoint surface over ``windows``."""
        samples: List[SurfaceSample] = []
        frame_total = 0

        for window in windows:
            frames = extract_window_frames(
                source,
                window,
                cadence_hz,
                self.segmenter.input_width,
                self.segmenter.input_height,
                credentials=self.credentials,
            )
            if len(frames) == 0:
                continue

            labels, confidence = self.segmenter.segment(frames)
            frame_total += int(len(frames))
            self.statistics["frames_segmented"] += int(len(frames))

            for offset in range(len(frames)):
                video_time = window.start_seconds + offset / cadence_hz
                detection_frame = (
                    (video_time - request.start_seconds) * request.detection_fps
                )
                for track_id, (track_frames, geometry) in boxes.items():
                    matched = self._nearest_frame(
                        track_frames,
                        detection_frame,
                        tolerance=request.detection_fps / (2.0 * cadence_hz),
                    )
                    if matched is None:
                        continue
                    x_center, y_center, width, height = geometry[matched]
                    surface, score = sample_surface(
                        labels[offset],
                        confidence[offset],
                        x_center,
                        y_center,
                        width,
                        height,
                        band_fraction=self.band_fraction,
                        width_fraction=self.width_fraction,
                        minimum_confidence=self.minimum_confidence,
                    )
                    samples.append(
                        SurfaceSample(
                            track_id=track_id,
                            frame=int(track_frames[matched]),
                            video_time_s=float(video_time),
                            surface=surface,
                            confidence=float(score),
                        )
                    )

        self._last_frame_count = frame_total
        return samples

    @staticmethod
    def _boxes_by_track(
        request: SegmentRequest,
    ) -> Dict[str, Tuple[np.ndarray, Dict[int, Tuple[float, float, float, float]]]]:
        """Index each track's bounding boxes by frame."""
        required = {"frame-count", "x-center", "y-center", "width", "height"}
        boxes: Dict[str, Tuple[np.ndarray, Dict[int, Tuple[float, float, float, float]]]] = {}

        for track_id, track in request.tracks.items():
            if track is None or track.height == 0:
                continue
            if not required.issubset(set(track.columns)):
                continue
            ordered = track.sort("frame-count")
            frames = (
                ordered.get_column("frame-count")
                .cast(pl.Int64, strict=False)
                .to_numpy()
            )
            geometry: Dict[int, Tuple[float, float, float, float]] = {}
            for index, row in enumerate(
                ordered.select(
                    ["x-center", "y-center", "width", "height"]
                ).iter_rows()
            ):
                try:
                    geometry[index] = (
                        float(row[0]),
                        float(row[1]),
                        float(row[2]),
                        float(row[3]),
                    )
                except (TypeError, ValueError):
                    continue
            if geometry:
                boxes[str(track_id)] = (frames, geometry)

        return boxes

    def _merged_windows(
        self,
        request: SegmentRequest,
        boxes: Dict[str, Tuple[np.ndarray, Dict[int, Tuple[float, float, float, float]]]],
    ) -> List[FrameWindow]:
        """Collapse the tracks' time spans into as few ffmpeg calls as possible."""
        fps = float(request.detection_fps)
        if fps <= 0:
            return []

        gap_frames = max(1, int(round(TRACK_SPAN_GAP_SECONDS * fps)))
        spans: List[Tuple[float, float]] = []
        for frames, _ in boxes.values():
            if frames.size == 0:
                continue
            ordered = np.sort(frames)
            run_start = ordered[0]
            run_end = ordered[0]
            for value in ordered[1:]:
                if value - run_end > gap_frames:
                    # A gap this large means the id was reused for something
                    # else in between, so the run ends here rather than
                    # stretching the span across the gap.
                    start = request.start_seconds + float(run_start) / fps
                    end = request.start_seconds + float(run_end) / fps
                    spans.append(
                        (
                            max(0.0, start - WINDOW_PADDING_SECONDS),
                            end + WINDOW_PADDING_SECONDS,
                        )
                    )
                    run_start = value
                run_end = value
            start = request.start_seconds + float(run_start) / fps
            end = request.start_seconds + float(run_end) / fps
            spans.append(
                (
                    max(0.0, start - WINDOW_PADDING_SECONDS),
                    end + WINDOW_PADDING_SECONDS,
                )
            )

        windows = self._merge_spans(spans)
        kept: List[FrameWindow] = []
        for window in windows:
            if window.duration_seconds > MAXIMUM_WINDOW_SECONDS:
                logger.warning(
                    f"Dropping a {window.duration_seconds:.1f}s window for "
                    f"{request.stem} at {window.start_seconds:.1f}s; longer "
                    f"than the {MAXIMUM_WINDOW_SECONDS:.0f}s ceiling for a "
                    "single crossing, likely a reused tracker id."
                )
                self.statistics["windows_too_long_dropped"] += 1
                continue
            kept.append(window)
        return kept

    @staticmethod
    def _merge_spans(spans: List[Tuple[float, float]]) -> List[FrameWindow]:
        """Collapse overlapping or nearly adjacent time spans into windows."""
        if not spans:
            return []

        spans = sorted(spans)
        merged: List[List[float]] = [list(spans[0])]
        for start, end in spans[1:]:
            if start - merged[-1][1] <= WINDOW_MERGE_GAP_SECONDS:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])

        return [
            FrameWindow(start_seconds=start, duration_seconds=end - start)
            for start, end in merged
            if end > start
        ]

    @staticmethod
    def _nearest_frame(
        frames: np.ndarray,
        target: float,
        tolerance: float,
    ) -> Optional[int]:
        """Return the index of the track row closest to ``target``, if close enough."""
        if frames.size == 0:
            return None
        position = int(np.searchsorted(frames, target))
        candidates = [index for index in (position - 1, position) if 0 <= index < frames.size]
        if not candidates:
            return None
        best = min(candidates, key=lambda index: abs(float(frames[index]) - target))
        if abs(float(frames[best]) - target) > max(float(tolerance), 1.0):
            return None
        return best
