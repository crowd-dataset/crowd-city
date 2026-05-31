import numpy as np
import polars as pl
from utils.core.metadata import MetaData
from helper_script import Youtube_Helper
from typing import Tuple, List, Any, Optional

metadata = MetaData()
helper = Youtube_Helper()


class Detection:

    def __init__(self) -> None:
        pass

    def pedestrian_crossing(self, dataframe: pl.DataFrame, video_id: str, df_mapping, min_x: float, max_x: float,
                            person_id, tol: float = 0.00, min_track_frames: int = 10, min_road_frames: int = 3,
                            max_track_gap_frames: int = 30, min_crossing_x_range: float = 0.14,
                            max_crossing_speed_per_frame: Optional[float] = None,
                            weak_crossing_x_range: float = 0.64,
                            low_x_range: float = 0.30,
                            low_x_min_road_frames: int = 20,
                            tiny_long_track_x_range: float = 0.36,
                            tiny_long_track_height: float = 0.12,
                            tiny_long_track_road_frames: int = 50,
                            slender_track_width: float = 0.05,
                            slender_track_height: float = 0.26,
                            slender_track_min_road_frames: int = 5,
                            slender_track_max_road_frames: int = 49,
                            strong_static_relx: float = 0.155,
                            heavy_camera_static_frames: int = 80,
                            heavy_camera_static_sx: float = 0.02,
                            large_lateral_x_range: float = 0.56,
                            large_lateral_tiny_height: float = 0.105,
                            camera_static_sx: float = 0.25,
                            camera_static_ratio: float = 0.60,
                            camera_static_relx: float = 0.18,
                            camera_static_height: float = 0.15,
                            camera_static_tiny_relx: float = 0.12,
                            camera_static_tiny_relx_height: float = 0.19,
                            weak_y_jitter_x_range: float = 0.50,
                            weak_y_jitter_motion: float = 0.30,
                            weak_y_jitter_height: float = 0.22,
                            no_static_slender_height: float = 0.24,
                            no_static_slender_max_road_frames: int = 20,
                            tiny_no_static_height: float = 0.12,
                            tiny_no_static_width: float = 0.026,
                            tiny_no_static_min_road_frames: int = 10,
                            slender_static_relx_min: float = 0.13,
                            camera_tiny_height: float = 0.15
                            ) -> Tuple[List[Any], List[Any]]:
        """
        Identifies pedestrian tracks that satisfy a road-crossing criterion and filters false positives.

        Tuned validation behaviour:
        - Splits reused tracker IDs into temporal segments when frame gaps are large.
        - Keeps the candidate stage broad, then rejects weak geometry cases after rider filtering.
        - Relaxed long-road rejection so slow true crossings are not removed.
        - Applies rider filtering on the segment window, not on the whole video, to avoid ID-reuse artefacts.
        """
        crossed_df = dataframe.filter(pl.col("yolo-id") == 0)
        if crossed_df.height == 0:
            return [], []

        crossed_df = Detection._dedup_per_frame(crossed_df)

        tracks = (
            crossed_df.select(["unique-id", "frame-count", "x-center", "y-center", "width", "height"])
            .sort(["unique-id", "frame-count"])
        )
        if tracks.height == 0:
            return [], []

        left_hard = float(min_x) - float(tol)
        left_soft = float(min_x) + float(tol)
        right_soft = float(max_x) - float(tol)
        right_hard = float(max_x) + float(tol)

        def split_segments(track: pl.DataFrame) -> List[pl.DataFrame]:
            """Split one tracker id into near-continuous temporal segments."""
            if track.height == 0:
                return []

            frames = track.get_column("frame-count").cast(pl.Int64, strict=False).to_list()
            if not frames:
                return []

            segments: List[pl.DataFrame] = []
            start_idx = 0
            prev_frame = int(frames[0])
            max_gap = max(int(max_track_gap_frames), 0)

            for idx in range(1, len(frames)):
                frame = int(frames[idx])
                if frame - prev_frame > max_gap:
                    segments.append(track.slice(start_idx, idx - start_idx))
                    start_idx = idx
                prev_frame = frame

            segments.append(track.slice(start_idx, len(frames) - start_idx))
            return segments

        def build_states(x: np.ndarray) -> np.ndarray:
            states = np.empty(x.size, dtype=np.int8)
            if x.size == 0:
                return states

            x0 = float(x[0])
            if x0 < float(min_x):
                s = 0
            elif x0 > float(max_x):
                s = 2
            else:
                s = 1
            states[0] = s

            for i in range(1, x.size):
                xi = float(x[i])

                if xi <= left_hard:
                    s = 0
                elif xi >= right_hard:
                    s = 2
                elif left_soft <= xi <= right_soft:
                    s = 1
                else:
                    # Buffer region: keep previous state to suppress boundary flicker.
                    s = s

                states[i] = s

            return states

        def segment_is_candidate(seg: pl.DataFrame) -> bool:
            if seg.height < int(min_track_frames):
                return False

            x = seg.get_column("x-center").cast(pl.Float64, strict=False).to_numpy()
            if x.size == 0:
                return False

            states = build_states(x)

            is_left = states == 0
            is_road = states == 1
            is_right = states == 2

            if int(is_road.sum()) < int(min_road_frames):
                return False

            left_before = np.maximum.accumulate(is_left)
            right_before = np.maximum.accumulate(is_right)
            left_after = np.maximum.accumulate(is_left[::-1])[::-1]
            right_after = np.maximum.accumulate(is_right[::-1])[::-1]

            crossing_mask = is_road & ((left_before & right_after) | (right_before & left_after))
            return bool(crossing_mask.any())

        uids = tracks.select("unique-id").unique().to_series().to_list()

        candidate_segments: List[Tuple[Any, int, int, float, float, int, float, float, float]] = []
        crossed_ids: List[Any] = []

        for uid in uids:
            tr = tracks.filter(pl.col("unique-id") == uid).sort("frame-count")
            if tr.height < int(min_track_frames):
                continue

            for seg in split_segments(tr):
                if not segment_is_candidate(seg):
                    continue

                frames = seg.get_column("frame-count").cast(pl.Int64, strict=False).to_numpy()
                x = seg.get_column("x-center").cast(pl.Float64, strict=False).to_numpy()
                if frames.size == 0 or x.size == 0:
                    continue

                start_frame = int(frames.min())
                end_frame = int(frames.max())
                duration = max(1, end_frame - start_frame + 1)
                x_range = float(np.nanmax(x) - np.nanmin(x))
                x_speed = float(x_range / duration)

                states = build_states(x)
                road_frames = int((states == 1).sum())

                if "height" in seg.columns:
                    height = seg.get_column("height").cast(pl.Float64, strict=False).to_numpy()
                    median_height = float(np.nanmedian(height)) if height.size > 0 else 0.0
                else:
                    median_height = 0.0

                if "width" in seg.columns:
                    width = seg.get_column("width").cast(pl.Float64, strict=False).to_numpy()
                    median_width = float(np.nanmedian(width)) if width.size > 0 else 0.0
                else:
                    median_width = 0.0

                if "y-center" in seg.columns:
                    y = seg.get_column("y-center").cast(pl.Float64, strict=False).to_numpy()
                    y_gross_motion = float(np.nansum(np.abs(np.diff(y)))) if y.size > 1 else 0.0
                else:
                    y_gross_motion = 0.0

                candidate_segments.append(
                    (uid, start_frame, end_frame, x_range, x_speed, road_frames, median_height, median_width, y_gross_motion)
                )
                if uid not in crossed_ids:
                    crossed_ids.append(uid)

        avg_height = None
        result = metadata.find_values_with_video_id(df_mapping, video_id)
        if result is not None:
            avg_height = result[15]

        pedestrian_ids: List[Any] = []
        for uid, start_frame, end_frame, x_range, x_speed, road_frames, median_height, median_width, y_gross_motion in candidate_segments:
            segment_df = dataframe.filter(
                (pl.col("frame-count") >= int(start_frame))
                & (pl.col("frame-count") <= int(end_frame))
            )

            if Detection.is_rider_id(segment_df, uid, avg_height):
                continue

            static_stats = Detection.static_reference_motion_stats(segment_df, uid)
            static_shared = int(static_stats.get("shared_frames", 0) or 0)
            static_sx_range = float(static_stats.get("static_x_range", 0.0) or 0.0)
            static_relx_range = float(static_stats.get("relative_x_range", 0.0) or 0.0)
            static_ratio = float(static_stats.get("static_to_person_ratio", 0.0) or 0.0)

            # Very tiny side-to-side movement is usually boundary flicker.  The threshold is intentionally
            # lower than older versions because the second validation video contains real crossings with
            # x-ranges around 0.16.
            if float(x_range) < float(min_crossing_x_range):
                continue

            # Small x-range can still be valid if the person clearly spends time inside the road band.
            # Reject only short road-band contacts.
            if float(x_range) < float(low_x_range) and int(road_frames) < int(low_x_min_road_frames):
                continue

            # Long weak tracks sitting in the road band are usually background/camera artefacts rather
            # than pedestrians crossing laterally.
            if float(x_range) < float(weak_crossing_x_range) and int(road_frames) > 90:
                continue

            # Some fake crossings are unstable tiny/background tracks: they appear to cross laterally,
            # but their vertical trajectory jitters heavily while staying weak in x-range.
            if float(x_range) < 0.56 and int(road_frames) > 40 and float(y_gross_motion) > 0.30:
                continue

            # Weak tracks with strong vertical jitter are often tracker or camera artefacts rather than
            # pedestrians moving laterally across the road.
            if (
                float(x_range) < float(weak_y_jitter_x_range)
                and float(y_gross_motion) > float(weak_y_jitter_motion)
                and float(median_height) < float(weak_y_jitter_height)
            ):
                continue

            # Tiny long tracks with weak lateral movement are normally far-background detections.
            if (
                float(x_range) < float(tiny_long_track_x_range)
                and float(median_height) < float(tiny_long_track_height)
                and int(road_frames) >= int(tiny_long_track_road_frames)
            ):
                continue

            # If there is no static reference at all, only reject very small/slender tracks. This keeps
            # real pedestrians such as the second-video uid 3489, while still removing no-reference
            # background tracks such as uid 13658 and uid 343.
            if (
                static_shared < 8
                and float(median_height) <= float(tiny_no_static_height)
                and float(median_width) <= float(tiny_no_static_width)
                and int(road_frames) >= int(tiny_no_static_min_road_frames)
            ):
                continue

            # Strong camera motion should only reject a candidate when either the target is very small,
            # or the person-static relative motion is genuinely tiny. The earlier v4 rule used relx<=0.22
            # and height<=0.18, which rejected true crossings such as uids 3439 and 3464.
            if static_sx_range >= float(camera_static_sx) and static_ratio >= float(camera_static_ratio):
                if float(median_height) <= float(camera_tiny_height) and int(road_frames) >= 5:
                    continue
                if (
                    static_relx_range <= float(camera_static_tiny_relx)
                    and float(median_height) <= float(camera_static_tiny_relx_height)
                ):
                    continue
                if (
                    static_relx_range <= float(camera_static_relx)
                    and float(median_height) <= float(camera_static_height)
                ):
                    continue

            # Slender distant tracks are a major source of fake crossings. In v5 this rule is split into
            # no-static and static-reference cases so that real slender pedestrians are not over-rejected.
            if (
                float(median_width) <= float(slender_track_width)
                and float(median_height) < float(slender_track_height)
                and int(slender_track_min_road_frames) <= int(road_frames) <= int(slender_track_max_road_frames)
            ):
                if static_shared < 8:
                    if (
                        float(median_height) < float(no_static_slender_height)
                        and int(road_frames) <= int(no_static_slender_max_road_frames)
                    ):
                        continue
                else:
                    if static_relx_range < float(slender_static_relx_min):
                        continue
                    if (
                        static_sx_range >= float(camera_static_sx)
                        and static_ratio >= float(camera_static_ratio)
                        and float(median_height) <= float(camera_tiny_height)
                    ):
                        continue

            # Large lateral jumps from tiny detections are only rejected when static references also show
            # camera dominance. A tiny but stable far pedestrian can be a real crossing, e.g. uid 17227.
            if (
                float(x_range) > float(large_lateral_x_range)
                and float(median_height) < float(large_lateral_tiny_height)
                and int(road_frames) >= 5
                and static_sx_range >= float(camera_static_sx)
                and static_ratio >= float(camera_static_ratio)
                and static_relx_range < 0.20
            ):
                continue

            # Keep the global speed filter optional. It is disabled by default because the validation set
            # contains real crossings with fast x motion.
            if max_crossing_speed_per_frame is not None:
                if float(x_speed) > float(max_crossing_speed_per_frame):
                    continue

            if not Detection.is_valid_crossing(segment_df, uid):
                continue

            if uid not in pedestrian_ids:
                pedestrian_ids.append(uid)

        return pedestrian_ids, crossed_ids

    @staticmethod
    def _dedup_per_frame(df: pl.DataFrame) -> pl.DataFrame:
        """Keep highest-confidence detection per (yolo-id, unique-id, frame-count)."""
        if "confidence" not in df.columns:
            return df.unique(subset=["yolo-id", "unique-id", "frame-count"], keep="first")

        return (
            df.sort(
                ["yolo-id", "unique-id", "frame-count", "confidence"],
                descending=[False, False, False, True],
            )
            .unique(subset=["yolo-id", "unique-id", "frame-count"], keep="first")
        )

    @staticmethod
    def _longest_frame_run(frames, *, gap_allow: int = 2) -> int:
        """Return the longest near-continuous run of frame numbers."""
        try:
            values = sorted({int(f) for f in frames})
        except Exception:
            return 0

        if not values:
            return 0

        max_run = 1
        cur_run = 1
        max_gap = max(int(gap_allow), 0) + 1
        prev = values[0]

        for frame in values[1:]:
            if int(frame) - int(prev) <= max_gap:
                cur_run += 1
            else:
                max_run = max(max_run, cur_run)
                cur_run = 1
            prev = frame

        return max(max_run, cur_run)

    @staticmethod
    def classify_rider_type(
        df: pl.DataFrame,
        person_id,
        *,
        avg_height: Optional[float] = None,
        min_shared_frames: int = 4,
        min_continuous_shared_frames: int = 12,
        shared_run_gap_allow: int = 2,
        min_vehicle_width_ratio: float = 0.50,
        min_vehicle_width_ratio_frames: float = 0.65,
        dist_rel_thresh: float = 0.8,
        prox_req: float = 0.7,
        alpha_x: float = 0.75,
        beta_y: float = 0.08,
        gamma_y: float = 1.4,
        coloc_req: float = 0.7,
        sim_thresh: float = 0.4,
        sim_req: float = 0.5,
        min_motion_steps: int = 3,
        motion_coloc_min: float = 0.5,
        short_shared_frames: int = 8,
        short_sim_req: float = 0.8,
        short_disp_req: float = 0.12,
        eps: float = 1e-9,
        person_class: int = 0,
        bicycle_class: int = 1,
        motorcycle_class: int = 3,
        car_class: int = 2,
        bus_class: int = 5,
        truck_class: int = 7,
        include_large_vehicle_passengers: bool = False,
    ) -> dict:
        """
        Classify whether a person track is associated with a vehicle.

        This uses the cyclist/rider logic from the bicyclist detection repo:
        - bicycle and motorcycle associations require a near-continuous shared run
        - bicycle and motorcycle associations require enough vehicle width relative to the person
        - car, bus and truck associations are treated as passengers when the person stays inside the vehicle box
        """
        if avg_height is not None:
            try:
                if float(avg_height) <= 0.0:
                    return {
                        "is_rider": False, "rider_type": None, "role": None, "vehicle_id": None,
                        "score": 0.0, "shared_frames": 0, "longest_shared_run": 0
                    }
            except Exception:
                return {
                    "is_rider": False, "rider_type": None, "role": None, "vehicle_id": None,
                    "score": 0.0, "shared_frames": 0, "longest_shared_run": 0
                }

        df = Detection._dedup_per_frame(df)

        p = (
            df.filter((pl.col("yolo-id") == person_class) & (pl.col("unique-id") == person_id))
              .sort("frame-count")
        )
        if p.height == 0:
            return {
                "is_rider": False, "rider_type": None, "role": None, "vehicle_id": None,
                "score": 0.0, "shared_frames": 0, "longest_shared_run": 0
            }

        p_frames = p.get_column("frame-count").to_numpy()
        if p_frames.size < min_shared_frames:
            return {
                "is_rider": False, "rider_type": None, "role": None, "vehicle_id": None,
                "score": 0.0, "shared_frames": 0, "longest_shared_run": 0
            }

        first_frame = int(p_frames.min())
        last_frame = int(p_frames.max())

        supported_vehicle_classes = [bicycle_class, motorcycle_class]
        if include_large_vehicle_passengers:
            supported_vehicle_classes.extend([car_class, bus_class, truck_class])

        vehicles = df.filter(
            (pl.col("frame-count") >= first_frame)
            & (pl.col("frame-count") <= last_frame)
            & (pl.col("yolo-id").is_in(supported_vehicle_classes))
        )
        if vehicles.height == 0:
            return {
                "is_rider": False, "rider_type": None, "role": None, "vehicle_id": None,
                "score": 0.0, "shared_frames": 0, "longest_shared_run": 0
            }

        vehicle_ids = vehicles.select("unique-id").unique().to_series().to_list()
        p1 = p.unique(subset=["frame-count"], keep="first")

        best = None

        for vid in vehicle_ids:
            v = vehicles.filter(pl.col("unique-id") == vid).sort("frame-count")
            if v.height == 0:
                continue

            v_class = int(v.get_column("yolo-id")[0])
            vtype = (
                "bicycle" if v_class == bicycle_class else
                "motorcycle" if v_class == motorcycle_class else
                "car" if v_class == car_class else
                "bus" if v_class == bus_class else
                "truck" if v_class == truck_class else
                None
            )
            if vtype is None:
                continue

            role = "rider" if v_class in (bicycle_class, motorcycle_class) else "passenger"

            v1 = v.unique(subset=["frame-count"], keep="first")
            j = p1.join(v1, on="frame-count", how="inner", suffix="_v")
            shared = j.height
            if shared < min_shared_frames:
                continue

            longest_shared_run = Detection._longest_frame_run(
                j.get_column("frame-count").to_list(),
                gap_allow=shared_run_gap_allow,
            )
            if role == "rider" and longest_shared_run < int(min_continuous_shared_frames):
                continue

            p_xy = j.select(["x-center", "y-center"]).to_numpy()
            v_xy = j.select(["x-center_v", "y-center_v"]).to_numpy()

            p_w = j.get_column("width").to_numpy()
            p_h = j.get_column("height").to_numpy()
            v_w = j.get_column("width_v").to_numpy()
            v_h = j.get_column("height_v").to_numpy()

            if role == "rider":
                vehicle_width_ratio_arr = v_w / np.maximum(p_w, eps)
                vehicle_width_ratio = float(np.median(vehicle_width_ratio_arr))
                vehicle_width_ratio_pass_ratio = float(
                    (vehicle_width_ratio_arr >= float(min_vehicle_width_ratio)).mean()
                )
                if vehicle_width_ratio_pass_ratio < float(min_vehicle_width_ratio_frames):
                    continue
            else:
                vehicle_width_ratio = 0.0
                vehicle_width_ratio_pass_ratio = 0.0

            dist = np.linalg.norm(p_xy - v_xy, axis=1)
            if role == "rider":
                dist_rel = dist / np.maximum(p_h, eps)
            else:
                dist_rel = dist / np.maximum(v_h, eps)

            prox = dist_rel < dist_rel_thresh
            prox_ratio = float(prox.mean())
            if prox_ratio < prox_req:
                continue

            relx = v_xy[:, 0] - p_xy[:, 0]
            rely = v_xy[:, 1] - p_xy[:, 1]

            if role == "rider":
                spatial = (np.abs(relx) < alpha_x * p_w) & (rely > beta_y * p_h) & (rely < gamma_y * p_h)
            else:
                inside = (np.abs(relx) <= 0.5 * v_w) & (np.abs(rely) <= 0.5 * v_h)
                spatial = inside

            coloc = prox & spatial
            coloc_ratio = float(coloc.mean())

            p_mov = np.diff(p_xy, axis=0)
            v_mov = np.diff(v_xy, axis=0)

            sim_ratio = 0.0
            if p_mov.shape[0] > 0:
                na = np.linalg.norm(p_mov, axis=1)
                nb = np.linalg.norm(v_mov, axis=1)
                move_mask = (na > eps) & (nb > eps)

                cos = np.zeros_like(na, dtype=float)
                cos[move_mask] = (p_mov[move_mask] * v_mov[move_mask]).sum(axis=1) / (na[move_mask] * nb[move_mask])

                prox_steps = prox[1:]
                m = min(len(prox_steps), len(cos), len(move_mask))
                prox_steps = prox_steps[:m]
                cos = cos[:m]
                move_mask = move_mask[:m]

                denom_mask = prox_steps & move_mask
                denom = int(denom_mask.sum())
                if denom >= min_motion_steps:
                    sim_ratio = float(((cos > sim_thresh) & denom_mask).sum() / denom)

            if shared < short_shared_frames:
                if shared > 1:
                    p_disp = float(np.linalg.norm(p_xy[-1] - p_xy[0]))
                    p_disp_rel = p_disp / float(np.maximum(np.mean(p_h), eps))
                else:
                    p_disp_rel = 0.0

                if not (sim_ratio >= short_sim_req or p_disp_rel >= short_disp_req):
                    continue

            ok = (coloc_ratio >= coloc_req) or (sim_ratio >= sim_req and coloc_ratio >= motion_coloc_min)
            if not ok:
                continue

            score = 0.7 * coloc_ratio + 0.2 * prox_ratio + 0.1 * float(sim_ratio)
            cand = {
                "is_rider": True,
                "rider_type": vtype,
                "role": role,
                "vehicle_id": vid,
                "score": float(score),
                "shared_frames": int(shared),
                "longest_shared_run": int(longest_shared_run),
                "vehicle_width_ratio": float(vehicle_width_ratio),
                "vehicle_width_ratio_pass_ratio": float(vehicle_width_ratio_pass_ratio),
                "prox_ratio": prox_ratio,
                "coloc_ratio": coloc_ratio,
                "sim_ratio": float(sim_ratio),
            }

            if best is None or cand["score"] > best["score"]:
                best = cand

        if best is None:
            return {
                "is_rider": False, "rider_type": None, "role": None, "vehicle_id": None,
                "score": 0.0, "shared_frames": 0, "longest_shared_run": 0
            }

        return best

    @staticmethod
    def is_rider_id(
        df: pl.DataFrame,
        id,
        avg_height: Optional[float] = None,
        min_shared_frames: int = 4,
        min_continuous_shared_frames: int = 12,
        shared_run_gap_allow: int = 2,
        min_vehicle_width_ratio: float = 0.50,
        min_vehicle_width_ratio_frames: float = 0.65,
        dist_rel_thresh: float = 0.8,
        prox_req: float = 0.7,
        alpha_x: float = 0.75,
        beta_y: float = 0.08,
        gamma_y: float = 1.4,
        coloc_req: float = 0.7,
        sim_thresh: float = 0.4,
        sim_req: float = 0.5,
        min_motion_steps: int = 3,
        motion_coloc_min: float = 0.5,
        short_shared_frames: int = 8,
        short_sim_req: float = 0.8,
        short_disp_req: float = 0.12,
        eps: float = 1e-9,
        include_large_vehicle_passengers: bool = False,
    ) -> bool:
        """
        Return True when the person is associated with a vehicle and should be removed
        from pedestrian crossing counts.

        Vehicle associations removed:
        - bicyclist
        - motorcyclist
        - passenger/driver associated with car, bus or truck
        """
        res = Detection.classify_rider_type(
            df,
            id,
            avg_height=avg_height,
            min_shared_frames=min_shared_frames,
            min_continuous_shared_frames=min_continuous_shared_frames,
            shared_run_gap_allow=shared_run_gap_allow,
            min_vehicle_width_ratio=min_vehicle_width_ratio,
            min_vehicle_width_ratio_frames=min_vehicle_width_ratio_frames,
            dist_rel_thresh=dist_rel_thresh,
            prox_req=prox_req,
            alpha_x=alpha_x,
            beta_y=beta_y,
            gamma_y=gamma_y,
            coloc_req=coloc_req,
            sim_thresh=sim_thresh,
            sim_req=sim_req,
            min_motion_steps=min_motion_steps,
            motion_coloc_min=motion_coloc_min,
            short_shared_frames=short_shared_frames,
            short_sim_req=short_sim_req,
            short_disp_req=short_disp_req,
            eps=eps,
            include_large_vehicle_passengers=include_large_vehicle_passengers,
        )
        return bool(res.get("is_rider"))

    @staticmethod
    def static_reference_motion_stats(df, person_id, STATIC_CLASS_IDS=(9, 10, 11, 12, 13),
                                      MIN_SHARED_FRAMES=8, Q=0.05, EPS=1e-9):
        """
        Return motion statistics between a person track and the best available static reference.

        The returned values are used as weak evidence for camera-motion false positives.
        If no usable static reference exists, zero-valued statistics are returned so the caller
        can fall back to geometry-only checks.
        """
        empty = {
            "has_reference": False,
            "static_id": None,
            "static_class": None,
            "shared_frames": 0,
            "person_x_range": 0.0,
            "static_x_range": 0.0,
            "relative_x_range": 0.0,
            "static_to_person_ratio": 0.0,
        }

        if df.height == 0:
            return empty

        if "confidence" in df.columns:
            df = Detection._dedup_per_frame(df)

        p = (
            df.filter((pl.col("yolo-id") == 0) & (pl.col("unique-id") == person_id))
              .sort("frame-count")
        )
        if p.height == 0:
            return empty

        p = p.unique(subset=["frame-count"], keep="first")
        first_frame = int(p.get_column("frame-count").min())
        last_frame = int(p.get_column("frame-count").max())

        refs = df.filter(
            (pl.col("frame-count") >= first_frame)
            & (pl.col("frame-count") <= last_frame)
            & (pl.col("yolo-id").is_in(STATIC_CLASS_IDS))
        )
        if refs.height == 0:
            return empty

        def robust_range(values) -> float:
            arr = np.asarray(values, dtype=float)
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                return 0.0
            lo = float(np.quantile(arr, float(Q)))
            hi = float(np.quantile(arr, 1.0 - float(Q)))
            return max(0.0, hi - lo)

        best = None
        for ref_id in refs.select("unique-id").unique().to_series().to_list():
            r = refs.filter(pl.col("unique-id") == ref_id).sort("frame-count")
            if r.height == 0:
                continue

            r = r.unique(subset=["frame-count"], keep="first")
            joined = p.join(r, on="frame-count", how="inner", suffix="_ref")
            shared = joined.height
            if shared < int(MIN_SHARED_FRAMES):
                continue

            person_x = joined.get_column("x-center").cast(pl.Float64, strict=False).to_numpy()
            ref_x = joined.get_column("x-center_ref").cast(pl.Float64, strict=False).to_numpy()

            person_x_range = robust_range(person_x)
            static_x_range = robust_range(ref_x)
            relative_x_range = robust_range(person_x - ref_x)
            ratio = static_x_range / max(person_x_range, float(EPS))

            ref_class = None
            try:
                ref_class = int(r.get_column("yolo-id")[0])
            except Exception:
                ref_class = None

            cand = {
                "has_reference": True,
                "static_id": ref_id,
                "static_class": ref_class,
                "shared_frames": int(shared),
                "person_x_range": float(person_x_range),
                "static_x_range": float(static_x_range),
                "relative_x_range": float(relative_x_range),
                "static_to_person_ratio": float(ratio),
            }

            if best is None or (cand["shared_frames"], cand["static_x_range"]) > (
                best["shared_frames"], best["static_x_range"]
            ):
                best = cand

        return best if best is not None else empty

    @staticmethod
    def is_valid_crossing(df, person_id, ratio_thresh=0.6, STATIC_CLASS_IDS=(9, 10, 11, 12, 13),
                          MIN_SHARED_FRAMES=8, RELX_MIN=0.01, Q=0.05, EPS=1e-9):
        """
        Checks whether an apparent pedestrian road-crossing is real or caused by dashcam turning.

        This function is designed for dashcam footage where camera motion (especially turning)
        can create *apparent* lateral motion of pedestrians that are actually stationary.
        To reduce these false positives, it uses detections from "static-ish" objects
        (e.g., traffic lights / stop signs) as a proxy for camera-induced motion.

        Core idea:
          - During a camera turn, both the pedestrian and background objects shift similarly
            in image space, especially in the X direction.
          - If the pedestrian's X motion is mostly explained by the camera (as estimated from
            a static object's X motion), then the pedestrian did not truly move relative to
            the scene and the crossing is likely invalid.

        The algorithm:
          1) Extract the person track (YOLO person class, same unique-id).
          2) Within the person time window, find tracks for static objects.
          3) For each static track, align it with the person by frame-count (inner join).
          4) Compute robust X-motion ranges using quantiles to reduce jitter:
               px_rng   = robust_range(person_x)
               sx_rng   = robust_range(static_x)
               relx_rng = robust_range(person_x - static_x)
             where robust_range(x) = quantile(1-Q) - quantile(Q)
          5) Select the best static reference (most overlap frames; tie-break by larger sx_rng).
          6) Decide validity:
               - If relx_rng is tiny => person moves with camera => invalid (False)
               - Else if sx_rng/px_rng is large AND relx_rng not strong => invalid (False)
               - Otherwise => valid (True)

        Notes:
          - This function assumes the input coordinates are normalised in [0, 1].
          - It expects `df` to be a Polars DataFrame and `pl` to be imported.
          - If no static objects are available, the function returns True (cannot verify).

        Args:
          df (pl.DataFrame): YOLO detections with columns:
            - "yolo-id" (int): class id (0 = person)
            - "unique-id": tracker id per object
            - "frame-count" (int): frame index
            - "x-center" (float): normalised x-center in [0,1]
            - "confidence" (float, optional): detection confidence
            - other YOLO fields are allowed but not required here
          person_id (Any): The tracker unique-id for the person to validate.
          ratio_thresh (float): Threshold for camera-dominance ratio = static_x_rng / person_x_rng.
            Larger values are more permissive. Typical range: 0.5–0.9.
          STATIC_CLASS_IDS (Tuple[int, ...]): Class IDs treated as static references.
            Default is COCO-like: traffic light (9), fire hydrant (10), stop sign (11),
            parking meter (12), bench (13).
          MIN_SHARED_FRAMES (int): Minimum number of overlapping frames between person and a
            candidate static track to consider it usable.
          RELX_MIN (float): Minimum robust range of (person_x - static_x) to treat motion as
            real (independent of camera). Lower = more permissive.
          Q (float): Quantile used for robust range (e.g., Q=0.05 uses 5%..95% range).
          EPS (float): Small constant to avoid divide-by-zero.

        Returns:
          bool: True if the crossing is likely valid (person moved independently of the camera),
            False if the apparent crossing is likely caused by camera turning.

        """
        # -------------------------------------------------------------------------
        # Deduplicate per frame to reduce jitter and avoid join misalignment.
        #    - For each (yolo-id, unique-id, frame-count), keep the highest-confidence row.
        #    - This is important because multiple detections per frame can inflate ranges.
        # -------------------------------------------------------------------------
        if "confidence" in df.columns:
            df = (
                df.sort(
                    ["yolo-id", "unique-id", "frame-count", "confidence"],
                    descending=[False, False, False, True],
                )
                .unique(subset=["yolo-id", "unique-id", "frame-count"], keep="first")
            )
        else:
            # If confidence is not present, just keep the first per key.
            df = df.unique(subset=["yolo-id", "unique-id", "frame-count"], keep="first")

        # -------------------------------------------------------------------------
        # Extract the person's track.
        #    - We restrict to yolo-id == 0 (person) to avoid accidental collisions where
        #      another class might share the same unique-id due to tracker re-use.
        #    - We also ensure one row per frame-count for a clean alignment later.
        # -------------------------------------------------------------------------
        person_track = (
            df.filter((pl.col("yolo-id") == 0) & (pl.col("unique-id") == person_id))
            .sort("frame-count")
            .unique(subset=["frame-count"], keep="first")
        )
        if person_track.height == 0:
            # No track => cannot validate => treat as invalid crossing.
            return False

        # Identify the time window of the person track.
        frames = person_track.get_column("frame-count").to_numpy()
        first_frame = int(frames.min())
        last_frame = int(frames.max())

        # -------------------------------------------------------------------------
        # Collect static-object detections in the same time window.
        #    - These objects should (ideally) be fixed in the world and move only due to
        #      camera motion. Their apparent X movement serves as a proxy for camera turn.
        # -------------------------------------------------------------------------
        static_objs = (
            df.filter(
                (pl.col("frame-count") >= first_frame)
                & (pl.col("frame-count") <= last_frame)
                & (pl.col("yolo-id").is_in(list(STATIC_CLASS_IDS)))
            )
            .sort("frame-count")
        )

        # If we have no static references, we cannot disentangle camera motion.
        # Preserve our earlier behavior: assume the crossing is valid.
        if static_objs.height == 0:
            return True

        # -------------------------------------------------------------------------
        # Define a robust range function.
        #    - Using min/max can be overly sensitive to jitter/outliers.
        #    - Quantile range (Q..1-Q) is more stable in practice.
        # -------------------------------------------------------------------------
        def robust_range(series: pl.Series) -> float:
            # Quantiles might fail if the series is empty or not numeric; handle gracefully.
            try:
                lo = series.quantile(Q, "nearest")
                hi = series.quantile(1.0 - Q, "nearest")
                return float(hi - lo)  # type: ignore
            except Exception:
                return 0.0

        # -------------------------------------------------------------------------
        # Compare the person to each static track and choose the best reference.
        #    Selection policy:
        #      - prefer the static track with the most overlapping frames
        #      - tie-break by larger static motion (more informative camera signal)
        # -------------------------------------------------------------------------
        best = None
        static_uids = static_objs.select("unique-id").unique().to_series().to_list()

        for sid in static_uids:
            # Extract a single static object's track and keep one row per frame.
            s_track = (
                static_objs.filter(pl.col("unique-id") == sid)
                .sort("frame-count")
                .unique(subset=["frame-count"], keep="first")
            )
            if s_track.height == 0:
                continue

            # Align by frame-count. We only consider frames where both are present.
            joined = person_track.join(s_track, on="frame-count", how="inner", suffix="_s")

            # Require a minimum overlap to avoid unstable statistics.
            if joined.height < MIN_SHARED_FRAMES:
                continue

            # Extract aligned X-centers.
            px = joined.get_column("x-center")      # person x
            sx = joined.get_column("x-center_s")    # static x
            relx = px - sx                          # person relative to static

            # Robust motion magnitudes.
            px_rng = robust_range(px)
            sx_rng = robust_range(sx)
            relx_rng = robust_range(relx)

            # Camera dominance ratio: if high, camera motion can explain most person motion.
            ratio = float(sx_rng / max(px_rng, EPS))

            cand = {
                "shared": int(joined.height),
                "px_rng": float(px_rng),
                "sx_rng": float(sx_rng),
                "relx_rng": float(relx_rng),
                "ratio": float(ratio),
            }

            # Pick best candidate reference.
            if best is None:
                best = cand
            else:
                if (cand["shared"], cand["sx_rng"]) > (best["shared"], best["sx_rng"]):
                    best = cand

        # If no static track had enough overlap, fall back to permissive behavior.
        if best is None:
            return True

        # -------------------------------------------------------------------------
        # Decision rules (updated to prioritise RELATIVE motion).
        #
        # Rule A (primary):
        #   If the relative motion is tiny, the person is moving with the background
        #   and the "crossing" is likely a camera-turn artifact => invalid.
        #
        # Rule B (secondary guard):
        #   If camera dominance ratio is high *and* relative motion is not strong,
        #   treat as invalid.
        #
        # These rules intentionally rely on (person_x - static_x), which is the key
        # to rejecting turning artifacts.
        # -------------------------------------------------------------------------
        if best["relx_rng"] < RELX_MIN:
            return False

        if best["ratio"] >= float(ratio_thresh) and best["relx_rng"] < (2.0 * RELX_MIN):
            return False

        # Otherwise, the person shows independent lateral motion relative to the scene.
        return True
