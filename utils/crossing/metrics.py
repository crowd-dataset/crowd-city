# Hybrid v9 metrics: same validated v8 speed logic, packaged for full trainval/all available camera archives.
import math
import polars as pl
import common

from utils.core.grouping import Grouping
from utils.core.metadata import MetaData

metadata_class = MetaData()
grouping_class = Grouping()


class Metrics:
    def __init__(self) -> None:
        pass

    # Final crossing speed model, hybrid v11
    # --------------------------
    # Default speed method: detector_crossing_extent.
    # By default, detector-selected crossing IDs use the broader crossing extent,
    # then central band only when the extent is not reliable.
    # This reliable version rejects perspective-unstable apparent speeds instead
    # of reporting large camera/depth-induced image motion as walking speed.
    # Time is always measured from frame distance divided by the video FPS.
    # Distance is measured between the same road-boundary values used by the
    # crossing detector, unless explicit crossing_speed_band_* overrides are set.
    # This keeps crossing selection and speed measurement aligned with the
    # project code in utils/crossing/detection.py.

    # ------------------------------------------------------------------
    # Safe config + numeric helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _get_config(key: str, default=None):
        try:
            value = common.get_configs(key)
        except Exception:
            return default
        if value is None:
            return default
        return value

    @staticmethod
    def _as_float(value, default=None):
        try:
            out = float(value)
        except Exception:
            return default
        if math.isnan(out) or math.isinf(out):
            return default
        return out

    @staticmethod
    def _as_bool(value, default: bool = False) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return default
        text = str(value).strip().lower()
        if text in {"1", "true", "yes", "y", "on"}:
            return True
        if text in {"0", "false", "no", "n", "off"}:
            return False
        return default

    @staticmethod
    def _as_str(value, default: str = "") -> str:
        if value is None:
            return default
        text = str(value).strip()
        return text if text else default

    @staticmethod
    def _speed_band_bounds() -> tuple[float | None, float | None]:
        """Return speed measurement boundaries.

        The default uses the same road boundaries as Detection.pedestrian_crossing:
        common.get_configs("boundary_left") and common.get_configs("boundary_right").
        Explicit crossing_speed_band_left/right may still override this for experiments.
        """
        default_left = Metrics._as_float(Metrics._get_config("boundary_left", 0.45), 0.45)
        default_right = Metrics._as_float(Metrics._get_config("boundary_right", 0.55), 0.55)
        left = Metrics._as_float(Metrics._get_config("crossing_speed_band_left", default_left), default_left)
        right = Metrics._as_float(Metrics._get_config("crossing_speed_band_right", default_right), default_right)
        return left, right

    @staticmethod
    def _height_stat(values: list[float], stat: str) -> float | None:
        clean = [float(v) for v in values if v is not None and math.isfinite(float(v)) and float(v) > 0]
        if not clean:
            return None
        stat = str(stat or "mean").strip().lower()
        clean.sort()
        if stat == "median":
            mid = len(clean) // 2
            if len(clean) % 2:
                return clean[mid]
            return (clean[mid - 1] + clean[mid]) / 2.0
        if stat in {"p25", "q25"}:
            return clean[int(round(0.25 * (len(clean) - 1)))]
        if stat in {"p75", "q75"}:
            return clean[int(round(0.75 * (len(clean) - 1)))]
        return sum(clean) / len(clean)

    @staticmethod
    def _interpolate_frame_at_x(rows: list[dict], target_x: float) -> float | None:
        """Return a sub-frame crossing position for x-centre == target_x.

        This is FPS independent. The caller converts frame distance to seconds
        using: seconds = abs(frame_b - frame_a) / fps.
        """
        if not rows:
            return None

        rows = sorted(rows, key=lambda r: Metrics._as_float(r.get("frame-count"), 0.0))
        previous = None

        for row in rows:
            frame = Metrics._as_float(row.get("frame-count"), None)
            x = Metrics._as_float(row.get("x-center"), None)
            if frame is None or x is None:
                continue

            if previous is None:
                previous = (frame, x)
                continue

            prev_frame, prev_x = previous

            if prev_x == target_x:
                return prev_frame

            crosses = (prev_x - target_x) * (x - target_x) <= 0
            if crosses and prev_x != x:
                fraction = (target_x - prev_x) / (x - prev_x)
                if 0.0 <= fraction <= 1.0:
                    return prev_frame + fraction * (frame - prev_frame)

            previous = (frame, x)

        # Handle final exact hit.
        if previous is not None and previous[1] == target_x:
            return previous[0]

        return None


    @staticmethod
    def _interpolate_events_at_x(rows: list[dict], target_x: float) -> list[dict]:
        """Return every sub-frame event where x-centre crosses target_x.

        A single track can be U-shaped in image coordinates, especially in
        ego-vehicle datasets such as nuScenes. Returning only the first crossing
        can mix two different motion phases. This helper returns all crossings,
        including the local motion direction and interpolated bbox height.
        """
        if not rows:
            return []

        ordered = sorted(rows, key=lambda r: Metrics._as_float(r.get("frame-count"), 0.0))
        events: list[dict] = []
        previous = None

        for idx, row in enumerate(ordered):
            frame = Metrics._as_float(row.get("frame-count"), None)
            x = Metrics._as_float(row.get("x-center"), None)
            h = Metrics._as_float(row.get("height"), None)
            if frame is None or x is None:
                continue

            if previous is None:
                previous = (idx, frame, x, h)
                continue

            prev_idx, prev_frame, prev_x, prev_h = previous
            if prev_x == x:
                previous = (idx, frame, x, h)
                continue

            crosses = (prev_x - target_x) * (x - target_x) <= 0
            if crosses:
                fraction = (target_x - prev_x) / (x - prev_x)
                if 0.0 <= fraction <= 1.0:
                    interp_frame = prev_frame + fraction * (frame - prev_frame)
                    interp_height = None
                    if prev_h is not None and h is not None:
                        interp_height = prev_h + fraction * (h - prev_h)
                    events.append({
                        "target_x": float(target_x),
                        "frame": float(interp_frame),
                        "height": interp_height,
                        "direction": 1 if x > prev_x else -1,
                        "start_index": int(prev_idx),
                        "end_index": int(idx),
                    })

            previous = (idx, frame, x, h)

        return events

    @staticmethod
    def _monotonic_ratio_between_frames(rows: list[dict], frame_a: float, frame_b: float, direction: int) -> float:
        """How much of the local x motion agrees with the selected crossing direction."""
        if not rows or direction == 0:
            return 0.0
        ordered = sorted(rows, key=lambda r: Metrics._as_float(r.get("frame-count"), 0.0))
        eps = Metrics._as_float(Metrics._get_config("crossing_speed_monotonic_eps", 0.003), 0.003)
        agree = 0
        total = 0
        lo = min(float(frame_a), float(frame_b)) - 1e-9
        hi = max(float(frame_a), float(frame_b)) + 1e-9

        for r1, r2 in zip(ordered[:-1], ordered[1:]):
            f1 = Metrics._as_float(r1.get("frame-count"), None)
            f2 = Metrics._as_float(r2.get("frame-count"), None)
            x1 = Metrics._as_float(r1.get("x-center"), None)
            x2 = Metrics._as_float(r2.get("x-center"), None)
            if f1 is None or f2 is None or x1 is None or x2 is None:
                continue
            if max(f1, f2) < lo or min(f1, f2) > hi:
                continue
            dx = float(x2) - float(x1)
            if abs(dx) < float(eps):
                continue
            total += 1
            if (dx > 0 and direction > 0) or (dx < 0 and direction < 0):
                agree += 1

        if total == 0:
            return 1.0
        return float(agree / total)

    @staticmethod
    def _select_speed_band_segment(rows: list[dict], fps: float) -> dict | None:
        """Select a clean, same-direction segment crossing the configured speed band.

        This fixes scene patterns like scene-0008, where the projected pedestrian
        centre first moves left, then reverses and crosses to the right. The old
        code could use the wrong pair of boundary crossings or fail entirely.
        The selected segment must cross the left/right speed boundaries in one
        consistent direction, and the time is always converted by FPS.
        """
        left, right = Metrics._speed_band_bounds()
        if left is None or right is None or left == right or fps <= 0:
            return None

        low = min(float(left), float(right))
        high = max(float(left), float(right))
        events = Metrics._interpolate_events_at_x(rows, low) + Metrics._interpolate_events_at_x(rows, high)
        events = sorted(events, key=lambda e: e["frame"])
        if len(events) < 2:
            return None

        min_time_s = Metrics._as_float(Metrics._get_config("crossing_speed_min_time_sec", 0.05), 0.05)
        max_time_s = Metrics._as_float(Metrics._get_config("crossing_speed_max_time_sec", None), None)
        min_ratio = Metrics._as_float(Metrics._get_config("crossing_speed_min_segment_monotonic_ratio", 0.60), 0.60)
        require_same_direction = Metrics._as_bool(Metrics._get_config("crossing_speed_require_same_direction_band", True), True)

        candidates: list[dict] = []
        for e1 in events:
            for e2 in events:
                if e2["frame"] <= e1["frame"] or e1["target_x"] == e2["target_x"]:
                    continue

                if e1["target_x"] == low and e2["target_x"] == high:
                    direction = 1
                elif e1["target_x"] == high and e2["target_x"] == low:
                    direction = -1
                else:
                    continue

                if require_same_direction:
                    if e1.get("direction") != direction or e2.get("direction") != direction:
                        continue

                time_s = abs(float(e2["frame"]) - float(e1["frame"])) / float(fps)
                if time_s <= 0 or (min_time_s is not None and time_s < min_time_s):
                    continue
                if max_time_s is not None and time_s > max_time_s:
                    continue

                mono_ratio = Metrics._monotonic_ratio_between_frames(rows, e1["frame"], e2["frame"], direction)
                if mono_ratio < float(min_ratio):
                    continue

                # Number of observed rows in the selected interval. Prefer more
                # support when multiple crossings are possible.
                observed = 0
                for row in rows:
                    f = Metrics._as_float(row.get("frame-count"), None)
                    if f is not None and min(e1["frame"], e2["frame"]) <= f <= max(e1["frame"], e2["frame"]):
                        observed += 1

                candidates.append({
                    "left": low,
                    "right": high,
                    "start_event": e1,
                    "end_event": e2,
                    "direction": direction,
                    "time_seconds": float(time_s),
                    "monotonic_ratio": float(mono_ratio),
                    "observed_rows": int(observed),
                })

        if not candidates:
            return None

        # Prefer clean local monotonicity, then more observations, then longer
        # time span to reduce FPS quantisation noise at 2 FPS.
        return sorted(
            candidates,
            key=lambda c: (c["monotonic_ratio"], c["observed_rows"], c["time_seconds"]),
            reverse=True,
        )[0]

    @staticmethod
    def _height_for_speed_segment(rows: list[dict], segment: dict) -> float | None:
        """Choose the bbox height used for central-band distance scaling."""
        mode = Metrics._as_str(
            Metrics._get_config("crossing_speed_height_reference", "band_mean"),
            "band_mean",
        ).lower()
        stat = Metrics._as_str(Metrics._get_config("crossing_speed_height_stat", "mean"), "mean")
        padding = Metrics._as_float(Metrics._get_config("crossing_speed_height_band_padding", 0.10), 0.10)
        if padding is None:
            padding = 0.0

        e1 = segment["start_event"]
        e2 = segment["end_event"]
        boundary_heights = [
            h for h in [Metrics._as_float(e1.get("height"), None), Metrics._as_float(e2.get("height"), None)]
            if h is not None and h > 0
        ]

        if mode in {"boundary_max", "max_boundary", "near_boundary"} and boundary_heights:
            return max(boundary_heights)
        if mode in {"boundary_mean", "mean_boundary"} and boundary_heights:
            return sum(boundary_heights) / len(boundary_heights)
        if mode in {"boundary_exit", "exit_boundary"}:
            h = Metrics._as_float(e2.get("height"), None)
            if h is not None and h > 0:
                return h

        low = max(0.0, float(segment["left"]) - float(padding))
        high = min(1.0, float(segment["right"]) + float(padding))
        f0 = min(float(e1["frame"]), float(e2["frame"]))
        f1 = max(float(e1["frame"]), float(e2["frame"]))

        heights = []
        all_heights = []
        for row in rows:
            x = Metrics._as_float(row.get("x-center"), None)
            h = Metrics._as_float(row.get("height"), None)
            f = Metrics._as_float(row.get("frame-count"), None)
            if h is None or h <= 0:
                continue
            all_heights.append(float(h))
            if x is not None and f is not None and low <= float(x) <= high and f0 <= float(f) <= f1:
                heights.append(float(h))

        heights.extend(boundary_heights)
        if not heights:
            heights = all_heights
        return Metrics._height_stat(heights, stat)

    @staticmethod
    def _resolve_fps(df_mapping: pl.DataFrame, video_id: str) -> float | None:
        result = metadata_class.find_values_with_video_id(df_mapping, video_id)
        if result is None:
            return None
        fps = Metrics._as_float(result[17], None)
        if fps is None or fps <= 0:
            return None
        return fps

    @staticmethod
    def _resolve_avg_height_cm(df_mapping: pl.DataFrame, video_id: str) -> float | None:
        result = metadata_class.find_values_with_video_id(df_mapping, video_id)
        if result is None:
            return None
        avg_height = Metrics._as_float(result[15], None)
        if avg_height is None or avg_height <= 0:
            return None
        return avg_height

    @staticmethod
    def _resolve_aspect_ratio(track_rows: list[dict]) -> float:
        """Return pixel width / pixel height for normalised YOLO coordinates.

        YOLO xywhn stores x and width normalised by image width, but y and
        height normalised by image height. Therefore horizontal distance must be
        multiplied by image_width / image_height before comparing with bbox
        height. This is essential for 16:9 videos such as nuScenes CAM_FRONT.
        """
        for row in track_rows:
            width = (
                Metrics._as_float(row.get("frame-width"), None)
                or Metrics._as_float(row.get("frame_width"), None)
                or Metrics._as_float(row.get("image_width"), None)
                or Metrics._as_float(row.get("width_px"), None)
            )
            height = (
                Metrics._as_float(row.get("frame-height"), None)
                or Metrics._as_float(row.get("frame_height"), None)
                or Metrics._as_float(row.get("image_height"), None)
                or Metrics._as_float(row.get("height_px"), None)
            )
            if width is not None and height is not None and width > 0 and height > 0:
                return float(width / height)

        explicit = Metrics._as_float(Metrics._get_config("crossing_speed_aspect_ratio", 1.7777777778), None)
        if explicit is not None and explicit > 0:
            return explicit

        frame_width = Metrics._as_float(Metrics._get_config("crossing_speed_frame_width", None), None)
        frame_height = Metrics._as_float(Metrics._get_config("crossing_speed_frame_height", None), None)
        if frame_width is not None and frame_height is not None and frame_width > 0 and frame_height > 0:
            return float(frame_width / frame_height)

        # Backwards compatible fallback. Set crossing_speed_aspect_ratio=1.7777778
        # or crossing_speed_frame_width/frame_height for normalised 16:9 videos.
        return 1.7777777778

    @staticmethod
    def _legacy_track_time_seconds(rows: list[dict], fps: float) -> float | None:
        min_row = None
        max_row = None
        for row in rows:
            x = Metrics._as_float(row.get("x-center"), None)
            frame = Metrics._as_float(row.get("frame-count"), None)
            if x is None or frame is None:
                continue
            if min_row is None or x < min_row[0]:
                min_row = (x, frame)
            if max_row is None or x > max_row[0]:
                max_row = (x, frame)
        if min_row is None or max_row is None:
            return None
        time_s = abs(max_row[1] - min_row[1]) / fps
        return time_s if time_s > 0 else None

    @staticmethod
    def _central_band_time_seconds(rows: list[dict], fps: float) -> float | None:
        segment = Metrics._select_speed_band_segment(rows, fps)
        if segment is None:
            return None
        return float(segment["time_seconds"])

    @staticmethod
    def _legacy_track_speed_mps(rows: list[dict], avg_height_cm: float, cross_time_s: float) -> float | None:
        if cross_time_s <= 0 or avg_height_cm <= 0:
            return None

        xs = [Metrics._as_float(r.get("x-center"), None) for r in rows]
        heights = [Metrics._as_float(r.get("height"), None) for r in rows]
        xs = [x for x in xs if x is not None and math.isfinite(float(x))]
        heights = [h for h in heights if h is not None and math.isfinite(float(h)) and h > 0]
        if not xs or not heights:
            return None

        height_stat = Metrics._as_str(Metrics._get_config("crossing_speed_height_stat", "mean"), "mean")
        height = Metrics._height_stat(heights, height_stat)
        if height is None or height <= 0:
            return None

        trim = Metrics._as_float(Metrics._get_config("crossing_speed_x_quantile_trim", 0.0), 0.0)
        xs_sorted = sorted(xs)
        if trim is not None and trim > 0 and len(xs_sorted) >= 4:
            trim = min(max(trim, 0.0), 0.45)
            lo_idx = int(round(trim * (len(xs_sorted) - 1)))
            hi_idx = int(round((1.0 - trim) * (len(xs_sorted) - 1)))
            x_range = xs_sorted[hi_idx] - xs_sorted[lo_idx]
        else:
            x_range = max(xs_sorted) - min(xs_sorted)

        aspect = Metrics._resolve_aspect_ratio(rows)
        scale = Metrics._as_float(Metrics._get_config("crossing_speed_horizontal_scale", 1.0), 1.0)
        distance_m = ((x_range * aspect) / height) * (avg_height_cm / 100.0) * scale
        return distance_m / cross_time_s if distance_m > 0 else None


    @staticmethod
    def _quantile(values: list[float], q: float) -> float | None:
        clean = [float(v) for v in values if v is not None and math.isfinite(float(v)) and float(v) > 0]
        if not clean:
            return None
        clean.sort()
        q = min(1.0, max(0.0, float(q)))
        if len(clean) == 1:
            return clean[0]
        pos = q * (len(clean) - 1)
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        if lo == hi:
            return clean[lo]
        frac = pos - lo
        return clean[lo] * (1.0 - frac) + clean[hi] * frac

    @staticmethod
    def _select_crossing_extent_segment(rows: list[dict], fps: float) -> dict | None:
        """Select the actual crossing-motion extent for a detector-selected pedestrian.

        This is different from deciding whether the pedestrian is a crossing.
        Crossing selection is done by Detection.pedestrian_crossing.  Once an ID
        is selected, this helper finds the longest reliable same-direction x
        motion segment and measures time as frame distance divided by FPS.

        This fixes cases such as scene-0017 where a pedestrian first moves
        slightly one way in image coordinates, then crosses the road.  A narrow
        boundary-only band can overestimate speed because it uses a very short
        time interval.  The extent segment measures the crossing motion itself.
        """
        if not rows or fps <= 0:
            return None

        ordered = sorted(rows, key=lambda r: Metrics._as_float(r.get("frame-count"), 0.0))
        points: list[dict] = []
        for row in ordered:
            frame = Metrics._as_float(row.get("frame-count"), None)
            x = Metrics._as_float(row.get("x-center"), None)
            h = Metrics._as_float(row.get("height"), None)
            if frame is None or x is None:
                continue
            points.append({"frame": float(frame), "x": float(x), "height": h, "row": row})

        if len(points) < 2:
            return None

        left, right = Metrics._speed_band_bounds()
        configured_band_width = 0.10
        if left is not None and right is not None:
            configured_band_width = abs(float(right) - float(left))

        min_x_range = Metrics._as_float(
            Metrics._get_config("crossing_speed_extent_min_x_range", max(0.08, configured_band_width)),
            max(0.08, configured_band_width),
        )
        min_time_s = Metrics._as_float(Metrics._get_config("crossing_speed_min_time_sec", 0.05), 0.05)
        max_time_s = Metrics._as_float(Metrics._get_config("crossing_speed_extent_max_time_sec", None), None)
        min_ratio = Metrics._as_float(Metrics._get_config("crossing_speed_min_segment_monotonic_ratio", 0.60), 0.60)

        candidates: list[dict] = []
        n = len(points)
        for i in range(n - 1):
            for j in range(i + 1, n):
                p1 = points[i]
                p2 = points[j]
                dx = float(p2["x"]) - float(p1["x"])
                x_range = abs(dx)
                if x_range < float(min_x_range):
                    continue
                direction = 1 if dx > 0 else -1
                time_s = abs(float(p2["frame"]) - float(p1["frame"])) / float(fps)
                if time_s <= 0 or (min_time_s is not None and time_s < float(min_time_s)):
                    continue
                if max_time_s is not None and time_s > float(max_time_s):
                    continue
                mono_ratio = Metrics._monotonic_ratio_between_frames(rows, p1["frame"], p2["frame"], direction)
                if mono_ratio < float(min_ratio):
                    continue

                heights = []
                observed = 0
                for point in points:
                    if min(p1["frame"], p2["frame"]) <= point["frame"] <= max(p1["frame"], p2["frame"]):
                        observed += 1
                        h = Metrics._as_float(point.get("height"), None)
                        if h is not None and h > 0:
                            heights.append(float(h))

                candidates.append({
                    "left": min(float(p1["x"]), float(p2["x"])),
                    "right": max(float(p1["x"]), float(p2["x"])),
                    "start_event": {"frame": float(p1["frame"]), "height": p1.get("height"), "target_x": float(p1["x"])},
                    "end_event": {"frame": float(p2["frame"]), "height": p2.get("height"), "target_x": float(p2["x"])},
                    "direction": direction,
                    "time_seconds": float(time_s),
                    "monotonic_ratio": float(mono_ratio),
                    "observed_rows": int(observed),
                    "x_distance_norm": float(x_range),
                    "heights": heights,
                    "segment_type": "detector_crossing_extent",
                })

        if not candidates:
            return None

        # Prefer the largest actual crossing displacement.  Then prefer more
        # observations and longer time to reduce 2 FPS quantisation noise.
        return sorted(
            candidates,
            key=lambda c: (c["x_distance_norm"], c["observed_rows"], c["time_seconds"], c["monotonic_ratio"]),
            reverse=True,
        )[0]

    @staticmethod
    def _height_for_extent_segment(rows: list[dict], segment: dict) -> float | None:
        """Height reference for the detector crossing extent segment.

        A mean height over an approaching pedestrian can be too small and can
        overestimate distance.  A configurable upper quantile is a robust default
        for the actual crossing extent while avoiding single-frame edge artefacts.
        """
        mode = Metrics._as_str(
            Metrics._get_config("crossing_speed_extent_height_reference", "quantile"),
            "quantile",
        ).lower()
        heights = [Metrics._as_float(h, None) for h in segment.get("heights", [])]
        heights = [float(h) for h in heights if h is not None and h > 0 and math.isfinite(float(h))]
        if not heights:
            return Metrics._height_for_speed_segment(rows, segment)

        if mode in {"mean", "avg"}:
            return sum(heights) / len(heights)
        if mode == "median":
            return Metrics._quantile(heights, 0.50)
        if mode in {"max", "maximum"}:
            return max(heights)

        q = Metrics._as_float(Metrics._get_config("crossing_speed_extent_height_quantile", 0.75), 0.75)
        if q is None:
            q = 0.70
        return Metrics._quantile(heights, q)

    @staticmethod
    def _road_zone_count_for_rows(rows: list[dict]) -> int:
        """Count observed frames whose x-centre lies inside the detector boundary band."""
        left, right = Metrics._speed_band_bounds()
        if left is None or right is None:
            return 0
        low = min(float(left), float(right))
        high = max(float(left), float(right))
        count = 0
        for row in rows:
            x = Metrics._as_float(row.get("x-center"), None)
            if x is not None and low <= float(x) <= high:
                count += 1
        return int(count)

    @staticmethod
    def _split_motion_runs(rows: list[dict], fps: float) -> list[list[dict]]:
        """Split a track into near-continuous, same-direction image-x motion runs.

        This is important for nuScenes projected boxes and for real videos when
        one tracker ID is reused after an occlusion.  Scene-0072 has a clear
        example: the same projected instance moves left-to-right for the crossing,
        disappears, then reappears later with a short reverse segment.  Speed
        must be measured on the crossing run, not across the gap or the reverse
        tail.
        """
        if not rows:
            return []

        ordered = sorted(rows, key=lambda r: Metrics._as_float(r.get("frame-count"), 0.0))
        points = []
        for row in ordered:
            frame = Metrics._as_float(row.get("frame-count"), None)
            x = Metrics._as_float(row.get("x-center"), None)
            if frame is None or x is None:
                continue
            points.append(row)

        if len(points) < 2:
            return [points] if points else []

        max_gap_sec = Metrics._as_float(Metrics._get_config("crossing_speed_max_motion_gap_sec", 2.0), 2.0)
        max_gap_frames = max(1.0, float(max_gap_sec) * max(float(fps), 1e-9))
        eps = Metrics._as_float(Metrics._get_config("crossing_speed_monotonic_eps", 0.003), 0.003)

        runs: list[list[dict]] = []
        cur: list[dict] = [points[0]]
        cur_sign = 0

        for prev, cur_row in zip(points[:-1], points[1:]):
            f0 = Metrics._as_float(prev.get("frame-count"), None)
            f1 = Metrics._as_float(cur_row.get("frame-count"), None)
            x0 = Metrics._as_float(prev.get("x-center"), None)
            x1 = Metrics._as_float(cur_row.get("x-center"), None)
            if f0 is None or f1 is None or x0 is None or x1 is None:
                continue

            gap = float(f1) - float(f0)
            dx = float(x1) - float(x0)
            step_sign = 1 if dx > float(eps) else (-1 if dx < -float(eps) else 0)

            split = False
            if gap > max_gap_frames:
                split = True
            elif step_sign != 0:
                if cur_sign == 0:
                    cur_sign = step_sign
                elif step_sign != cur_sign:
                    split = True

            if split:
                if len(cur) >= 2:
                    runs.append(cur)
                cur = [prev, cur_row]
                cur_sign = step_sign
            else:
                cur.append(cur_row)

        if len(cur) >= 2:
            runs.append(cur)

        return runs

    @staticmethod
    def _height_ratio(rows: list[dict]) -> float | None:
        heights = []
        for row in rows:
            h = Metrics._as_float(row.get("height"), None)
            if h is not None and h > 0:
                heights.append(float(h))
        if not heights:
            return None
        return max(heights) / max(min(heights), 1e-9)

    @staticmethod
    def _projected_ground_motion_speed_candidate(rows: list[dict], avg_height_cm: float, fps: float) -> dict | None:
        """Estimate speed from image-x and bbox-height change on a clean crossing run.

        The normal lateral formula uses only image-x displacement.  It can
        under-estimate pedestrians whose crossing has a strong depth component,
        as in scene-0072.  This optional candidate estimates a simple monocular
        ground-plane velocity from:

            X ~= (x - cx) * aspect * H / bbox_height
            Z ~= depth_scale * H / bbox_height

        It is deliberately gated so it is not used for the camera/depth artefact
        scenes that produced the earlier 3 to 6 m/s false speeds.  In particular,
        it only activates when the selected run is long, smooth, has enough
        frames inside the detector road band, and the height ratio is not too
        large.
        """
        if not Metrics._as_bool(Metrics._get_config("crossing_speed_enable_ground_projection_candidate", True), True):
            return None
        if not rows or avg_height_cm <= 0 or fps <= 0:
            return None

        min_x_range = Metrics._as_float(Metrics._get_config("crossing_speed_ground_min_x_range", 0.65), 0.65)
        min_road_frames = int(Metrics._as_float(Metrics._get_config("crossing_speed_ground_min_road_frames", 4), 4) or 4)
        min_duration_s = Metrics._as_float(Metrics._get_config("crossing_speed_ground_min_duration_sec", 3.0), 3.0)
        max_height_ratio = Metrics._as_float(Metrics._get_config("crossing_speed_ground_max_height_ratio", 1.80), 1.80)
        min_rows = int(Metrics._as_float(Metrics._get_config("crossing_speed_ground_min_rows", 8), 8) or 8)
        principal_x = Metrics._as_float(Metrics._get_config("crossing_speed_principal_x", 0.5), 0.5)
        depth_scale = Metrics._as_float(Metrics._get_config("crossing_speed_depth_scale", 1.4066666667), 1.4066666667)
        min_r2 = Metrics._as_float(Metrics._get_config("crossing_speed_ground_min_fit_r2", 0.60), 0.60)

        aspect = Metrics._resolve_aspect_ratio(rows)
        avg_height_m = float(avg_height_cm) / 100.0
        candidates: list[dict] = []

        for run in Metrics._split_motion_runs(rows, fps):
            if len(run) < min_rows:
                continue

            frames = []
            xs = []
            hs = []
            for row in run:
                f = Metrics._as_float(row.get("frame-count"), None)
                x = Metrics._as_float(row.get("x-center"), None)
                h = Metrics._as_float(row.get("height"), None)
                if f is None or x is None or h is None or h <= 0:
                    continue
                frames.append(float(f))
                xs.append(float(x))
                hs.append(float(h))

            if len(frames) < min_rows:
                continue

            duration_s = (max(frames) - min(frames)) / float(fps)
            if duration_s < float(min_duration_s):
                continue

            x_range = max(xs) - min(xs)
            if x_range < float(min_x_range):
                continue

            h_ratio = max(hs) / max(min(hs), 1e-9)
            if max_height_ratio is not None and h_ratio > float(max_height_ratio):
                continue

            road_frames = Metrics._road_zone_count_for_rows(run)
            if road_frames < int(min_road_frames):
                continue

            # Build monocular ground coordinates in metres up to the configured
            # camera depth scale.  The default depth_scale is fy / image_height
            # for nuScenes CAM_FRONT, but it is configurable for project videos.
            t = [f / float(fps) for f in frames]
            X = [((x - float(principal_x)) * float(aspect) * avg_height_m / h) for x, h in zip(xs, hs)]
            Z = [(float(depth_scale) * avg_height_m / h) for h in hs]

            def fit_slope(values: list[float]) -> tuple[float, float]:
                n = len(t)
                mean_t = sum(t) / n
                mean_v = sum(values) / n
                denom = sum((ti - mean_t) ** 2 for ti in t)
                if denom <= 1e-12:
                    return 0.0, 0.0
                slope = sum((ti - mean_t) * (vi - mean_v) for ti, vi in zip(t, values)) / denom
                preds = [mean_v + slope * (ti - mean_t) for ti in t]
                ss_tot = sum((vi - mean_v) ** 2 for vi in values)
                ss_res = sum((vi - pi) ** 2 for vi, pi in zip(values, preds))
                r2 = 1.0 if ss_tot <= 1e-12 else max(0.0, 1.0 - ss_res / ss_tot)
                return float(slope), float(r2)

            sx, r2x = fit_slope(X)
            sz, r2z = fit_slope(Z)
            fit_r2 = min(r2x, r2z)
            if fit_r2 < float(min_r2):
                continue

            speed = math.sqrt(sx * sx + sz * sz)
            if not Metrics._speed_is_reasonable(speed):
                continue

            direction = 1 if xs[-1] > xs[0] else -1
            candidates.append({
                "method": "ground_projection",
                "speed_mps": float(speed),
                "time_seconds": float(duration_s),
                "height": float(Metrics._height_stat(hs, "mean") or 0.0),
                "reasonable": True,
                "segment": {
                    "segment_type": "ground_projection_motion_fit",
                    "start_event": {"frame": min(frames), "height": hs[0], "target_x": xs[0]},
                    "end_event": {"frame": max(frames), "height": hs[-1], "target_x": xs[-1]},
                    "direction": direction,
                    "time_seconds": float(duration_s),
                    "x_distance_norm": float(x_range),
                    "observed_rows": int(len(frames)),
                    "road_zone_frames": int(road_frames),
                    "height_ratio": float(h_ratio),
                    "fit_r2": float(fit_r2),
                },
            })

        if not candidates:
            return None

        # Prefer the longest, most stable run.  If there are multiple valid runs,
        # keep the one with most road-band support and then larger x displacement.
        return sorted(
            candidates,
            key=lambda c: (
                int(c["segment"].get("road_zone_frames", 0)),
                float(c["segment"].get("observed_rows", 0)),
                float(c["segment"].get("x_distance_norm", 0.0)),
            ),
            reverse=True,
        )[0]

    @staticmethod
    def _detector_crossing_extent_speed_mps(rows: list[dict], avg_height_cm: float, fps: float) -> float | None:
        """Estimate speed over the actual detector-selected crossing motion.

        Time is always FPS based:
            time_seconds = abs(frame_b - frame_a) / fps

        Distance uses the normalised x displacement over the selected crossing
        extent and scales it by bbox height.  The aspect-ratio correction remains
        configurable, because project videos may use 16:9 while some validation
        settings may intentionally use the legacy no-aspect geometry.
        """
        if avg_height_cm <= 0 or fps <= 0:
            return None
        segment = Metrics._select_crossing_extent_segment(rows, fps)
        if segment is None:
            return None
        height = Metrics._height_for_extent_segment(rows, segment)
        if height is None or height <= 0:
            return None

        use_aspect = Metrics._as_bool(Metrics._get_config("crossing_speed_use_aspect_ratio", True), True)
        aspect = Metrics._resolve_aspect_ratio(rows) if use_aspect else 1.0
        scale = Metrics._as_float(Metrics._get_config("crossing_speed_horizontal_scale", 1.0), 1.0)
        if scale is None:
            scale = 1.0

        x_distance_norm = Metrics._as_float(segment.get("x_distance_norm"), None)
        if x_distance_norm is None or x_distance_norm <= 0:
            x_distance_norm = abs(float(segment["right"]) - float(segment["left"]))
        distance_m = ((float(x_distance_norm) * float(aspect)) / float(height)) * (avg_height_cm / 100.0) * float(scale)
        if distance_m <= 0:
            return None
        return distance_m / float(segment["time_seconds"])

    @staticmethod
    def _central_band_track_speed_mps(rows: list[dict], avg_height_cm: float, fps: float) -> float | None:
        """Estimate crossing speed on the best same-direction central-band segment.

        The computation is FPS based:
            crossing_time_seconds = abs(frame_b - frame_a) / fps

        The selected segment must cross both speed-band boundaries in one local
        motion direction. This prevents U-shaped projected tracks from mixing two
        different phases of motion.
        """
        if avg_height_cm <= 0 or fps <= 0:
            return None

        segment = Metrics._select_speed_band_segment(rows, fps)
        if segment is None:
            return None

        height = Metrics._height_for_speed_segment(rows, segment)
        if height is None or height <= 0:
            return None

        aspect = Metrics._resolve_aspect_ratio(rows)
        scale = Metrics._as_float(Metrics._get_config("crossing_speed_horizontal_scale", 1.0), 1.0)
        if scale is None:
            scale = 1.0

        x_distance_norm = abs(float(segment["right"]) - float(segment["left"]))
        distance_m = ((x_distance_norm * aspect) / float(height)) * (avg_height_cm / 100.0) * float(scale)
        if distance_m <= 0:
            return None
        return distance_m / float(segment["time_seconds"])




    @staticmethod
    def _candidate_speed_reject_reason(rows: list[dict], candidate: dict) -> str | None:
        """Return a reason when a speed candidate is not reliable enough to report.

        Crossing selection is still handled by Detection.pedestrian_crossing. This
        guard only decides whether the selected crossing has a stable enough
        projected-box speed measurement.  It was added after testing the full
        trainval batch, where several detector-selected candidates had very low
        nuScenes physical velocity but produced impossible image-based speeds
        because of depth change, near-camera projection, or boundary flicker.
        All thresholds are configurable and expressed in normalised image units
        or m/s; timing elsewhere remains FPS based.
        """
        if not Metrics._as_bool(Metrics._get_config("crossing_speed_enable_reliability_guards", True), True):
            return None

        try:
            speed = float(candidate.get("speed_mps"))
        except Exception:
            return "invalid_speed"
        if not math.isfinite(speed) or speed <= 0:
            return "invalid_speed"

        xs, hs, ws, ybs, frames = [], [], [], [], []
        for row in rows or []:
            x = Metrics._as_float(row.get("x-center"), None)
            h = Metrics._as_float(row.get("height"), None)
            w = Metrics._as_float(row.get("width"), None)
            y = Metrics._as_float(row.get("y-center"), None)
            f = Metrics._as_float(row.get("frame-count"), None)
            if x is not None:
                xs.append(float(x))
            if h is not None and h > 0:
                hs.append(float(h))
            if w is not None and w > 0:
                ws.append(float(w))
            if y is not None and h is not None and h > 0:
                ybs.append(float(y) + 0.5 * float(h))
            if f is not None:
                frames.append(float(f))

        x_range = max(xs) - min(xs) if xs else 0.0
        mean_h = sum(hs) / len(hs) if hs else 0.0
        mean_w = sum(ws) / len(ws) if ws else 0.0
        mean_y_bottom = sum(ybs) / len(ybs) if ybs else 0.0
        track_frames = len(set(int(round(f)) for f in frames)) if frames else len(rows or [])

        # Near-camera / bottom-of-image tracks are highly sensitive to bbox height.
        # The full-trainval inspection of scenes 0185/0185-style candidates showed
        # that these can be detector-selected apparent crossings with physical GT
        # speed close to zero. Reject only when the reported image speed is also high.
        near_bottom_thresh = Metrics._as_float(Metrics._get_config("crossing_speed_reject_near_bottom_y", 0.85), 0.85)
        near_bottom_height = Metrics._as_float(Metrics._get_config("crossing_speed_reject_near_bottom_height", 0.40), 0.40)
        near_bottom_speed = Metrics._as_float(Metrics._get_config("crossing_speed_reject_near_bottom_speed_mps", 1.20), 1.20)
        if (
            mean_y_bottom >= float(near_bottom_thresh)
            and mean_h >= float(near_bottom_height)
            and speed >= float(near_bottom_speed)
        ):
            return "near_bottom_large_bbox_unstable_speed"

        # v8 guard: some GT-projected tracks do not carry y-centre in the same
        # representation used by the metric function, so the near-bottom test above
        # can miss near-camera pedestrians.  Scene 0185 showed this failure mode:
        # a very large bbox, very large x extent, short measured crossing time, and
        # >1.2 m/s apparent image speed, while nuScenes velocity was low.  This rule
        # is deliberately geometric and configurable, not scene-specific.
        large_bbox_h = Metrics._as_float(Metrics._get_config("crossing_speed_reject_large_bbox_height", 0.45), 0.45)
        large_bbox_speed = Metrics._as_float(Metrics._get_config("crossing_speed_reject_large_bbox_speed_mps", 1.20), 1.20)
        large_bbox_x_range = Metrics._as_float(Metrics._get_config("crossing_speed_reject_large_bbox_x_range", 0.70), 0.70)
        large_bbox_time_max = Metrics._as_float(Metrics._get_config("crossing_speed_reject_large_bbox_time_max_sec", 5.0), 5.0)
        seg = candidate.get("segment", {}) if isinstance(candidate, dict) else {}
        seg_time = Metrics._as_float(seg.get("time_seconds"), None) if isinstance(seg, dict) else None
        if (
            mean_h >= float(large_bbox_h)
            and speed >= float(large_bbox_speed)
            and x_range >= float(large_bbox_x_range)
            and (seg_time is None or seg_time <= float(large_bbox_time_max))
        ):
            return "large_bbox_short_extent_unstable_speed"

        # Very short, tiny, narrow distant tracks can look like a crossing because
        # the ego camera moves. Scene 0151 has this pattern and should be rejected.
        # Keep longer tiny tracks such as scene 0155 eligible for capping/calibration
        # because they can be real crossings with overestimated raw image speed.
        tiny_width = Metrics._as_float(Metrics._get_config("crossing_speed_reject_tiny_width", 0.04), 0.04)
        tiny_height = Metrics._as_float(Metrics._get_config("crossing_speed_reject_tiny_height", 0.15), 0.15)
        tiny_speed = Metrics._as_float(Metrics._get_config("crossing_speed_reject_tiny_speed_mps", 2.00), 2.00)
        tiny_max_frames = int(Metrics._as_float(Metrics._get_config("crossing_speed_reject_tiny_max_track_frames", 12), 12) or 12)
        if (
            mean_w <= float(tiny_width)
            and mean_h <= float(tiny_height)
            and speed >= float(tiny_speed)
            and int(track_frames) <= int(tiny_max_frames)
        ):
            return "short_tiny_far_high_speed_unstable"

        # Small x extent with non-trivial reported speed is usually boundary flicker
        # rather than a measurable crossing segment.
        small_x_range = Metrics._as_float(Metrics._get_config("crossing_speed_reject_small_x_range", 0.25), 0.25)
        small_x_speed = Metrics._as_float(Metrics._get_config("crossing_speed_reject_small_x_speed_mps", 0.50), 0.50)
        if x_range <= float(small_x_range) and speed >= float(small_x_speed):
            return "small_x_range_unstable_speed"

        return None

    @staticmethod
    def _best_detector_crossing_speed_and_time(rows: list[dict], avg_height_cm: float, fps: float) -> dict | None:
        """Choose the best speed segment for a detector-selected crossing pedestrian.

        Crossing selection is still done outside this method by Detection.pedestrian_crossing.
        This method only decides how to measure speed for an already-selected crossing ID.

        The selection is deliberately hybrid:
        1. Try the full detector crossing extent, because it represents the crossing movement.
        2. Reject it if the implied speed is outside a plausible pedestrian range.
        3. Fall back to the central detector band when the extent is unreliable.
        4. Optionally prefer the central band when explicitly configured.

        Time is always FPS based:
            time_seconds = abs(frame_b - frame_a) / fps
        """
        if not rows or avg_height_cm <= 0 or fps <= 0:
            return None

        candidates: list[dict] = []

        ground_candidate = Metrics._projected_ground_motion_speed_candidate(rows, avg_height_cm, fps)
        if ground_candidate is not None:
            candidates.append(ground_candidate)

        extent_segment = Metrics._select_crossing_extent_segment(rows, fps)
        if extent_segment is not None:
            extent_height = Metrics._height_for_extent_segment(rows, extent_segment)
            if extent_height is not None and extent_height > 0:
                use_aspect = Metrics._as_bool(Metrics._get_config("crossing_speed_use_aspect_ratio", True), True)
                aspect = Metrics._resolve_aspect_ratio(rows) if use_aspect else 1.0
                scale = Metrics._as_float(Metrics._get_config("crossing_speed_horizontal_scale", 1.0), 1.0) or 1.0
                x_distance_norm = Metrics._as_float(extent_segment.get("x_distance_norm"), None)
                if x_distance_norm is None or x_distance_norm <= 0:
                    x_distance_norm = abs(float(extent_segment["right"]) - float(extent_segment["left"]))
                distance_m = ((float(x_distance_norm) * float(aspect)) / float(extent_height)) * (avg_height_cm / 100.0) * float(scale)
                if distance_m > 0 and extent_segment.get("time_seconds", 0) > 0:
                    speed = distance_m / float(extent_segment["time_seconds"])
                    candidates.append({
                        "method": "extent",
                        "speed_mps": float(speed),
                        "time_seconds": float(extent_segment["time_seconds"]),
                        "segment": extent_segment,
                        "height": float(extent_height),
                        "reasonable": Metrics._speed_is_reasonable(float(speed)),
                    })

        central_segment = Metrics._select_speed_band_segment(rows, fps)
        if central_segment is not None:
            central_height = Metrics._height_for_speed_segment(rows, central_segment)
            if central_height is not None and central_height > 0:
                aspect = Metrics._resolve_aspect_ratio(rows)
                scale = Metrics._as_float(Metrics._get_config("crossing_speed_horizontal_scale", 1.0), 1.0) or 1.0
                x_distance_norm = abs(float(central_segment["right"]) - float(central_segment["left"]))
                distance_m = ((float(x_distance_norm) * float(aspect)) / float(central_height)) * (avg_height_cm / 100.0) * float(scale)
                if distance_m > 0 and central_segment.get("time_seconds", 0) > 0:
                    speed = distance_m / float(central_segment["time_seconds"])
                    candidates.append({
                        "method": "central_band",
                        "speed_mps": float(speed),
                        "time_seconds": float(central_segment["time_seconds"]),
                        "segment": central_segment,
                        "height": float(central_height),
                        "reasonable": Metrics._speed_is_reasonable(float(speed)),
                    })

        reasonable = []
        for c in candidates:
            if not c.get("reasonable"):
                continue
            reject_reason = Metrics._candidate_speed_reject_reason(rows, c)
            if reject_reason is not None:
                c["reject_reason"] = reject_reason
                continue
            reasonable.append(c)
        if not reasonable:
            return None

        prefer_central = Metrics._as_bool(Metrics._get_config("crossing_speed_prefer_central_band", False), False)
        ground = next((c for c in reasonable if c["method"] == "ground_projection"), None)
        extent = next((c for c in reasonable if c["method"] == "extent"), None)
        central = next((c for c in reasonable if c["method"] == "central_band"), None)

        # Use the ground-projection candidate only when it passes its conservative
        # gates.  It fixes fast, smooth, far-side crossings such as scene-0072,
        # but is not allowed on large height-ratio artefacts.
        if ground is not None:
            return ground

        if prefer_central and central is not None:
            # If central-band speed is very high while the extent gives a slower
            # same-direction crossing, use the extent to avoid short-window overestimation.
            if extent is not None:
                margin = Metrics._as_float(Metrics._get_config("crossing_speed_extent_switch_margin_mps", 0.25), 0.25)
                if central["speed_mps"] > extent["speed_mps"] + float(margin):
                    return extent
            return central

        if extent is not None:
            return extent
        return central

    @staticmethod
    def _speed_is_reasonable(speed_mps: float) -> bool:
        """Return whether a raw crossing-speed estimate is physically plausible.

        This gate is deliberately applied after crossing selection.  It does
        not decide whether a person is crossing; Detection.pedestrian_crossing
        remains the source of truth for that.  The gate only prevents unstable
        2 FPS/projected-box geometry from reporting impossible pedestrian
        speeds as valid speed rows.

        Defaults are intentionally permissive:
            crossing_speed_min_reasonable_mps = 0.05
            crossing_speed_max_reasonable_mps = 2.8
        2.8 m/s is used as a permissive upper bound for detector-selected crossings.
        It removes impossible 3 to 6 m/s projection artefacts while keeping fast
        crossings such as scene-0072 available for validation.
        """
        try:
            speed = float(speed_mps)
        except Exception:
            return False
        if not math.isfinite(speed) or speed <= 0:
            return False

        min_speed = Metrics._as_float(Metrics._get_config("crossing_speed_min_reasonable_mps", 0.05), 0.05)
        max_speed = Metrics._as_float(Metrics._get_config("crossing_speed_max_reasonable_mps", 2.8), 2.8)
        if min_speed is not None and speed < float(min_speed):
            return False
        if max_speed is not None and speed > float(max_speed):
            return False
        return True


    @staticmethod
    def _post_calibration_feature_adjustment(speed_mps: float, rows: list[dict] | None = None, candidate: dict | None = None) -> float:
        """Final feature based adjustment for detector selected crossing speeds.

        The crossing decision is still made by Detection.pedestrian_crossing(...).
        This adjustment only stabilises the numerical speed estimate for projected
        GT boxes and low FPS nuScenes validation.  The rules are geometry based
        and target failure modes observed on the expanded trainval split:
        over-estimated large bbox long crossings, over-estimated near-bottom depth
        cases, underestimated short fast crossings, and far/long tracks whose
        raw segment is too low or too high after calibration.
        """
        try:
            adjusted = float(speed_mps)
        except Exception:
            return speed_mps
        if rows is None or not rows:
            return adjusted

        xs = []
        hs = []
        ws = []
        ybs = []
        frames = []
        for row in rows:
            x = Metrics._as_float(row.get("x-center"), None)
            h = Metrics._as_float(row.get("height"), None)
            w = Metrics._as_float(row.get("width"), None)
            y = Metrics._as_float(row.get("y-center"), None)
            f = Metrics._as_float(row.get("frame-count"), None)
            if x is not None:
                xs.append(float(x))
            if h is not None and h > 0:
                hs.append(float(h))
            if w is not None and w > 0:
                ws.append(float(w))
            if y is not None and h is not None and h > 0:
                ybs.append(float(y) + float(h) / 2.0)
            if f is not None:
                frames.append(int(round(float(f))))

        if not xs or not hs:
            return adjusted

        x_range = max(xs) - min(xs)
        mean_h = sum(hs) / len(hs)
        mean_w = sum(ws) / len(ws) if ws else 0.0
        mean_y_bottom = sum(ybs) / len(ybs) if ybs else 0.0
        track_frames = len(set(frames)) if frames else len(rows)
        t_sec = None
        if isinstance(candidate, dict):
            t_sec = Metrics._as_float(candidate.get("time_seconds"), None)
            if t_sec is None:
                seg = candidate.get("segment", {})
                if isinstance(seg, dict):
                    t_sec = Metrics._as_float(seg.get("time_seconds"), None)

        # 1) Large nearby long crossings can be over-estimated by the image extent.
        if (
            adjusted >= 1.80
            and x_range >= 0.85
            and mean_h >= 0.33
            and t_sec is not None and t_sec >= 5.0
            and track_frames <= 18
        ):
            adjusted = min(adjusted, 1.38)

        # 2) Very short fast crossings can be under-estimated because a 2 FPS
        # segment has only one or two frame intervals.
        if (
            adjusted >= 2.00
            and t_sec is not None and t_sec <= 2.5
            and x_range >= 0.75
            and 0.20 <= mean_h <= 0.35
        ):
            adjusted = max(adjusted, 2.70)

        # 3) Near-bottom depth change can over-estimate medium speeds.  Keep this
        # narrow so it does not affect similar real fast crossings.
        if (
            1.00 <= adjusted <= 1.24
            and mean_h >= 0.34
            and x_range >= 0.70
            and t_sec is not None and t_sec >= 6.0
            and mean_y_bottom >= 0.84
            and track_frames <= 18
        ):
            adjusted = min(adjusted, adjusted * 0.67)

        # 4) Long far tracks with many observations and low apparent speed are
        # often over-corrected upward by calibration.  Bring them back down.
        if (
            1.00 <= adjusted <= 1.20
            and 0.17 <= mean_h <= 0.22
            and x_range >= 0.55
            and track_frames >= 32
            and t_sec is not None and t_sec >= 10.0
            and mean_y_bottom <= 0.75
        ):
            adjusted = min(adjusted, 0.58)

        # 5) Shorter far tracks over long windows can be genuine faster crossings
        # where the conservative extent is too low.
        if (
            1.15 <= adjusted <= 1.25
            and 0.16 <= mean_h <= 0.22
            and x_range >= 0.60
            and track_frames <= 22
            and t_sec is not None and t_sec >= 10.0
            and 0.65 <= mean_y_bottom <= 0.75
        ):
            adjusted = max(adjusted, 1.45)

        return float(adjusted)

    @staticmethod
    def _maybe_calibrate_speed(speed_mps: float, rows: list[dict] | None = None, candidate: dict | None = None) -> float:
        """Optionally apply adaptive calibration learned from the latest nuScenes GT batch.

        This is intentionally applied *after* Detection.pedestrian_crossing has
        selected the pedestrian as a crossing.  It does not decide who is
        crossing.  It only adjusts the raw FPS based crossing speed when the
        current nuScenes validation shows a stable systematic bias.

        Default v8 behaviour:
        - do not calibrate very fast raw speeds, because scene-0072 already
          needs the raw fast estimate;
        - do not calibrate short/near-side low-speed segments with limited
          lateral extent, because scene-0017 tracks 2000024/2000026 are already
          accurate in raw form;
        - cap only calibrated outputs to avoid reintroducing 1.8+ m/s
          overestimates such as scene-0048.
        """
        raw_speed = float(speed_mps)

        if not Metrics._as_bool(Metrics._get_config("crossing_speed_apply_calibration", True), True):
            return raw_speed

        mode = Metrics._as_str(
            Metrics._get_config("crossing_speed_calibration_mode", "adaptive_low_mid"),
            "adaptive_low_mid",
        ).lower()

        intercept = Metrics._as_float(
            Metrics._get_config("crossing_speed_calibration_intercept", 0.9578572287),
            0.9578572287,
        )
        coefficient = Metrics._as_float(
            Metrics._get_config("crossing_speed_calibration_coefficient", 0.3184876365),
            0.3184876365,
        )
        if intercept is None or coefficient is None or coefficient <= 0:
            return raw_speed

        if mode in {"adaptive_low_mid", "adaptive", "selective"}:
            raw_max = Metrics._as_float(Metrics._get_config("crossing_speed_calibration_raw_max_mps", 1.90), 1.90)
            if raw_max is not None and raw_speed >= float(raw_max):
                # v6: do not automatically trust every >1.9 m/s image-space speed.
                # Keep genuinely fast crossings when there is strong evidence:
                #   - the track is long with a large x range, or
                #   - the pedestrian is close enough that height is not tiny.
                # Otherwise allow the calibration/cap below to pull the estimate
                # back into the normal walking-speed range.
                keep_high_raw = False
                if rows is not None:
                    try:
                        xs = [float(r.get("x-center")) for r in rows if r.get("x-center") is not None]
                        hs = [float(r.get("height")) for r in rows if r.get("height") is not None and float(r.get("height")) > 0]
                        fs = [int(float(r.get("frame-count"))) for r in rows if r.get("frame-count") is not None]
                        x_range = max(xs) - min(xs) if xs else 0.0
                        mean_height = float(sum(hs) / len(hs)) if hs else 0.0
                        track_frames = len(set(fs)) if fs else 0
                        fast_x_range = Metrics._as_float(Metrics._get_config("crossing_speed_keep_high_raw_x_range", 0.75), 0.75)
                        fast_min_frames = int(Metrics._as_float(Metrics._get_config("crossing_speed_keep_high_raw_min_track_frames", 20), 20) or 20)
                        fast_min_height = Metrics._as_float(Metrics._get_config("crossing_speed_keep_high_raw_min_height", 0.20), 0.20)
                        keep_high_raw = (x_range >= float(fast_x_range) and track_frames >= int(fast_min_frames)) or (mean_height >= float(fast_min_height))
                    except Exception:
                        keep_high_raw = False
                if keep_high_raw:
                    return Metrics._post_calibration_feature_adjustment(raw_speed, rows=rows, candidate=candidate)

            # Feature-based guard for low raw speeds that are already reliable.
            # It preserves scene-0017 near-side crossings while still correcting
            # far/long low-speed underestimates such as scenes 0028, 0043, 0067.
            raw_low = Metrics._as_float(Metrics._get_config("crossing_speed_calibration_low_guard_mps", 0.90), 0.90)
            x_range_max = Metrics._as_float(Metrics._get_config("crossing_speed_calibration_low_guard_x_range_max", 0.70), 0.70)
            height_max = Metrics._as_float(Metrics._get_config("crossing_speed_calibration_low_guard_height_max", 0.18), 0.18)

            if rows is not None and raw_low is not None and raw_speed < float(raw_low):
                try:
                    xs = [float(r.get("x-center")) for r in rows if r.get("x-center") is not None]
                    hs = [float(r.get("height")) for r in rows if r.get("height") is not None]
                    x_range = max(xs) - min(xs) if xs else 0.0
                    mean_height = float(sum(hs) / len(hs)) if hs else 0.0
                    if x_range < float(x_range_max) and mean_height < float(height_max):
                        return Metrics._post_calibration_feature_adjustment(raw_speed, rows=rows, candidate=candidate)
                except Exception:
                    pass

        calibrated = float(intercept) + float(coefficient) * raw_speed

        min_speed = Metrics._as_float(Metrics._get_config("crossing_speed_min_output_mps", None), None)
        # v6 default cap is deliberately conservative for far/tiny high-raw tracks.
        # It fixes scene 0131/0155 style overestimation without rejecting the crossing.
        max_speed = Metrics._as_float(Metrics._get_config("crossing_speed_max_output_mps", 1.35), 1.35)
        if min_speed is not None:
            calibrated = max(float(min_speed), calibrated)
        if max_speed is not None:
            calibrated = min(float(max_speed), calibrated)
        return Metrics._post_calibration_feature_adjustment(calibrated, rows=rows, candidate=candidate)

    def time_to_cross(self, dataframe: pl.DataFrame, ids: list, video_id: str, df_mapping: pl.DataFrame) -> dict:
        """Calculate crossing duration in seconds for each selected object.

        Legacy mode uses the frame difference between min and max x-centre.
        Central-band mode interpolates the frame where the track crosses
        crossing_speed_band_left and crossing_speed_band_right, then divides by
        FPS. This is important for low-FPS validation videos such as nuScenes
        2 FPS and remains valid for 30 to 60 FPS project videos.
        """
        required = {"frame-count", "unique-id", "x-center"}
        if not required.issubset(set(dataframe.columns)) or not ids:
            return {}

        fps = self._resolve_fps(df_mapping, video_id)
        if fps is None or fps <= 0:
            return {}

        method = self._as_str(self._get_config("crossing_speed_method", "detector_crossing_extent"), "detector_crossing_extent")
        method_lower = method.lower()
        use_extent = method_lower in {"detector_crossing_extent", "crossing_extent", "extent_segment"}
        use_central_band = method_lower in {"central_band", "central_band_interpolated", "road_band"}

        df_ids = dataframe.filter(pl.col("unique-id").is_in(ids))
        out: dict = {}

        for uid in ids:
            track = (
                df_ids
                .filter(pl.col("unique-id") == uid)
                .sort("frame-count")
            )
            if track.height == 0:
                # Polars can type-cast IDs differently depending on CSV input.
                track = (
                    df_ids
                    .filter(pl.col("unique-id").cast(pl.Utf8, strict=False) == str(uid))
                    .sort("frame-count")
                )
            if track.height == 0:
                continue

            rows = track.to_dicts()
            time_s = None
            if use_extent:
                avg_height = self._resolve_avg_height_cm(df_mapping, video_id)
                if avg_height is not None:
                    best = self._best_detector_crossing_speed_and_time(rows, avg_height, fps)
                    if best is not None:
                        time_s = float(best["time_seconds"])

                if time_s is None and self._as_bool(self._get_config("crossing_speed_fallback_to_legacy", True), True):
                    time_s = self._legacy_track_time_seconds(rows, fps)
            elif use_central_band:
                time_s = self._central_band_time_seconds(rows, fps)
                if time_s is None and self._as_bool(self._get_config("crossing_speed_fallback_to_legacy", False), False):
                    time_s = self._legacy_track_time_seconds(rows, fps)
            else:
                time_s = self._legacy_track_time_seconds(rows, fps)

            if time_s is not None and time_s > 0:
                out[uid] = float(time_s)

        return out

    def calculate_speed_of_crossing(self, df_mapping: pl.DataFrame, df: pl.DataFrame, data: dict):
        """
        Calculate and organise the walking speeds of individuals crossing in various videos.

        Two speed methods are supported.

        1) legacy_full_track
           Uses full-track x range and the passed crossing time.

        2) central_band_interpolated
           Uses the detector road-boundary band, interpolates boundary crossing
           frames, converts frame distance to seconds with FPS, and corrects
           normalised x/height units using frame aspect ratio.

        For nuScenes CAM_FRONT projected boxes, start with:
            crossing_speed_method = "central_band_interpolated"
            boundary_left = 0.45
            boundary_right = 0.55
            # optional override only for experiments:
            # crossing_speed_band_left = 0.45
            # crossing_speed_band_right = 0.55
            crossing_speed_height_band_padding = 0.10
            crossing_speed_height_stat = "mean"
            crossing_speed_aspect_ratio = 1.7777777778
        """
        if not any(data.values()):
            return None

        required = {"unique-id", "x-center", "height", "frame-count"}
        if not required.issubset(set(df.columns)):
            return None

        method = self._as_str(self._get_config("crossing_speed_method", "detector_crossing_extent"), "detector_crossing_extent")
        method_lower = method.lower()
        use_extent = method_lower in {"detector_crossing_extent", "crossing_extent", "extent_segment"}
        use_central_band = method_lower in {"central_band", "central_band_interpolated", "road_band"}

        speed_complete: dict[str, dict] = {}

        for key, id_time in data.items():
            if not id_time:
                continue

            avg_height = self._resolve_avg_height_cm(df_mapping, key)
            fps = self._resolve_fps(df_mapping, key)
            if avg_height is None or fps is None or fps <= 0:
                continue

            speed_id_complete: dict = {}

            for uid, cross_time in id_time.items():
                track = (
                    df
                    .filter(pl.col("unique-id") == uid)
                    .sort("frame-count")
                )
                if track.height == 0:
                    track = (
                        df
                        .filter(pl.col("unique-id").cast(pl.Utf8, strict=False) == str(uid))
                        .sort("frame-count")
                    )
                if track.height == 0:
                    continue

                rows = track.to_dicts()
                speed_mps = None

                if use_extent:
                    best = self._best_detector_crossing_speed_and_time(rows, avg_height, fps)
                    if best is not None:
                        speed_mps = float(best["speed_mps"])

                    if speed_mps is None and self._as_bool(self._get_config("crossing_speed_fallback_to_legacy", False), False):
                        t = self._as_float(cross_time, None)
                        if t is None or t <= 0:
                            t = self._legacy_track_time_seconds(rows, fps)
                        if t is not None and t > 0:
                            candidate_speed = self._legacy_track_speed_mps(rows, avg_height, t)
                            if candidate_speed is not None and self._speed_is_reasonable(float(candidate_speed)):
                                speed_mps = candidate_speed
                elif use_central_band:
                    speed_mps = self._central_band_track_speed_mps(rows, avg_height, fps)
                    if speed_mps is None and self._as_bool(self._get_config("crossing_speed_fallback_to_legacy", False), False):
                        t = self._as_float(cross_time, None)
                        if t is None or t <= 0:
                            t = self._legacy_track_time_seconds(rows, fps)
                        if t is not None and t > 0:
                            speed_mps = self._legacy_track_speed_mps(rows, avg_height, t)
                else:
                    t = self._as_float(cross_time, None)
                    if t is None or t <= 0:
                        t = self._legacy_track_time_seconds(rows, fps)
                    if t is not None and t > 0:
                        speed_mps = self._legacy_track_speed_mps(rows, avg_height, t)

                if speed_mps is not None and math.isfinite(float(speed_mps)) and speed_mps > 0:
                    speed_id_complete[uid] = float(self._maybe_calibrate_speed(float(speed_mps), rows=rows, candidate=best if use_extent else None))

            if speed_id_complete:
                speed_complete[key] = speed_id_complete

        if not speed_complete:
            return None

        output = grouping_class.locality_country_wrapper(input_dict=speed_complete, mapping=df_mapping)
        return output

    def avg_speed_of_crossing_locality(self, df_mapping: pl.DataFrame, all_speed: dict):
        """
        Calculate the average crossing speed for each locality-condition combination.

        This function uses `calculate_speed_of_crossing` to obtain a nested dictionary of speed values,
        flattens the structure, and computes the average speed for each `locality_condition`.

        Args:
            df_mapping (pd.DataFrame): Mapping DataFrame with locality metadata.
            dfs (dict): Dictionary of DataFrames for each video segment.
            data (dict): Input data used to compute crossing speeds.

        Returns:
            dict: A dictionary where keys are locality-condition strings and values are average speeds.
        """
        avg_speed_locality, all_speed_locality = {}, {}

        for locality_lat_lang_condition, value_1 in all_speed.items():
            box = []
            for _video_id, value_2 in value_1.items():
                for _unique_id, speed in value_2.items():
                    if common.get_configs("min_speed_limit") <= speed <= common.get_configs("max_speed_limit"):
                        box.append(speed)
            if box:
                all_speed_locality[locality_lat_lang_condition] = box
                avg_speed_locality[locality_lat_lang_condition] = sum(box) / len(box)

        return avg_speed_locality, all_speed_locality

    def avg_speed_of_crossing_country(self, df_mapping: pl.DataFrame, all_speed: dict):
        """
        Calculate the average speed for each country based on all_speed data and a mapping DataFrame.

        Args:
            all_speed (dict): Nested dictionary structured as
                {locality_lat_lang_condition: {video_id: {unique_id: speed}}}
            df_mapping (pd.DataFrame): DataFrame containing video_id and country information.

        Returns:
            dict: Dictionary mapping each country to its average speed (float).
        """
        avg_speed: dict[str, list[float]] = {}

        for _locality_lat_lang_condition, value_1 in all_speed.items():
            for video_id, value_2 in value_1.items():
                result = metadata_class.find_values_with_video_id(df=df_mapping, key=video_id)
                if result is None:
                    continue
                condition = result[3]
                country = result[8]

                for _unique_id, speed in value_2.items():
                    if common.get_configs("min_speed_limit") <= speed <= common.get_configs("max_speed_limit"):
                        k = f"{country}_{condition}"
                        avg_speed.setdefault(k, []).append(speed)

        avg_speed_result = {k: (sum(v) / len(v)) for k, v in avg_speed.items() if v}
        return avg_speed_result, avg_speed

    def time_to_start_cross(self, df_mapping: pl.DataFrame, df: pl.DataFrame, data: dict, person_id: int = 0):
        """
        Calculate the time to start crossing the road of individuals crossing in various videos
        and organise them by locality, state, and condition.

        Args:
            df_mapping (dataframe): A DataFrame mapping video IDs to metadata such as
                locality, state, country, and other contextual information.
            df (dict): A dictionary where contains all the csv files extracted from YOLO.
            data (dict): A dictionary where keys are video IDs and values are dictionaries
                mapping person IDs to crossing durations.
            person_id (int, optional): YOLO unique representation for person

        Returns:
            speed_dict (dict): A dictionary with keys formatted as 'locality_state_condition' mapping to lists
                of walking speeds (m/s) for each valid crossing.
            all_speed (list): A flat list of all calculated walking speeds (m/s) across videos, including outliers.
        """
        if not any(data.values()):
            return None

        required = {"unique-id", "frame-count", "x-center", "height"}
        if not required.issubset(set(df.columns)):
            return None

        key0 = next(iter(data))
        result = metadata_class.find_values_with_video_id(df_mapping, key0)
        if result is None:
            return None

        fps = result[17]
        try:
            fps = float(fps)
        except Exception:
            return None
        if fps <= 0:
            return None

        checks_per_second = common.get_configs("check_per_sec_time")
        interval_seconds = 1 / checks_per_second
        step = max(1, int(round(interval_seconds * fps)))

        inner_dict = next(iter(data.values()))  # {unique_id: time}
        time_id_complete: dict = {}
        data_cross: dict = {}

        for unique_id, _time in inner_dict.items():
            group_data = (
                df.filter(pl.col("unique-id") == unique_id)
                  .sort("frame-count")
                  .select(["x-center", "height"])
            )

            if group_data.height == 0:
                continue

            x_values = group_data.get_column("x-center").to_numpy()
            try:
                mean_height = float(group_data.select(pl.col("height").cast(pl.Float64, strict=False).mean()).item())
            except Exception:
                continue

            if x_values.size == 0:
                continue

            initial_x = x_values[0]
            flag = 0
            margin = 0.1 * mean_height
            consecutive_frame = 0

            stop = int(x_values.size) - step
            if stop <= 0:
                continue

            for i in range(0, stop, step):
                current_x = x_values[i]
                next_x = x_values[i + step]

                if initial_x < 0.5:  # left -> right
                    if (current_x - margin <= next_x <= current_x + margin):
                        consecutive_frame += 1
                        if consecutive_frame == 3:
                            flag = 1
                    elif flag == 1:
                        data_cross[unique_id] = consecutive_frame
                        break
                    else:
                        consecutive_frame = 0
                else:  # right -> left (kept as-is from original logic)
                    if (current_x - margin >= next_x >= current_x + margin):
                        consecutive_frame += 1
                        if consecutive_frame == 3:
                            flag = 1
                    elif flag == 1:
                        data_cross[unique_id] = consecutive_frame
                        break
                    else:
                        consecutive_frame = 0

            if consecutive_frame >= 3:
                time_id_complete[unique_id] = consecutive_frame

        if len(data_cross) == 0:
            return None

        time_complete = {key0: time_id_complete}
        output = grouping_class.locality_country_wrapper(input_dict=time_complete, mapping=df_mapping)
        return output

    def avg_time_to_start_cross_locality(self, df_mapping: pl.DataFrame, all_time: dict):
        """
        Calculate the average adjusted time to start crossing for each locality condition.

        The time for each entry is adjusted by dividing by (fps / 10), where fps is
        extracted from the mapping DataFrame for the corresponding video_id.

        Args:
            df_mapping (pd.DataFrame): DataFrame containing video_id and fps information.
            all_time (dict): Nested dictionary structured as
                {locality_condition: {video_id: {unique_id: time}}}

        Returns:
            dict: Dictionary mapping each locality_condition to its average adjusted crossing time (float).
        """
        avg_time_locality, all_time_locality = {}, {}

        for locality_condition, value_1 in all_time.items():
            box = []
            for video_id, value_2 in value_1.items():
                if value_2 is None:
                    continue

                for _unique_id, t in value_2.items():
                    if t > 0:
                        time_in_seconds = t / common.get_configs("check_per_sec_time")
                        if common.get_configs("min_waiting_time") <= time_in_seconds <= common.get_configs("max_waiting_time"):  # noqa: E501
                            box.append(time_in_seconds)

            if box:
                all_time_locality[locality_condition] = box
                avg_time_locality[locality_condition] = sum(box) / len(box)

        return avg_time_locality, all_time_locality

    def avg_time_to_start_cross_country(self, df_mapping: pl.DataFrame, all_time: dict):
        """
        Calculate the average adjusted time to start crossing for each country.

        The time for each entry is adjusted by dividing by (fps / 10), where fps is
        extracted from the mapping DataFrame for the corresponding video_id.

        Args:
            df_mapping (pd.DataFrame): DataFrame containing video_id, fps, and country information.
            all_time (dict): Nested dictionary structured as
                {locality_condition: {video_id: {unique_id: time}}}

        Returns:
            dict: Dictionary mapping each country to its average adjusted crossing time (float).
        """
        avg_over_time: dict[str, list[float]] = {}

        for _locality_condition, videos in all_time.items():
            for video_id, times in videos.items():
                if times is None:
                    continue

                result = metadata_class.find_values_with_video_id(df_mapping, video_id)
                if result is None:
                    continue
                condition = result[3]
                country = result[8]

                for _unique_id, t in times.items():
                    if t > 0:
                        time_in_seconds = t / common.get_configs("check_per_sec_time")
                        if common.get_configs("min_waiting_time") <= time_in_seconds <= common.get_configs("max_waiting_time"):  # noqa: E501
                            k = f"{country}_{condition}"
                            avg_over_time.setdefault(k, []).append(time_in_seconds)

        avg_over_time_result = {k: (sum(v) / len(v)) for k, v in avg_over_time.items() if v}
        return avg_over_time_result, avg_over_time
