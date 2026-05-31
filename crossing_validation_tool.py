"""
Frame-based crossing validation + tuning workflow.

What it does
------------
1. Uses the same crossing detector as the main analysis pipeline:
   Detection.pedestrian_crossing(...)
2. Saves all algorithm candidates, valid candidates, and rejected candidates.
3. Lets you label every algorithm candidate as actual crossing, fake candidate, or unsure.
4. Computes a filter-level confusion matrix.
5. Builds a conflict list containing only wrong cases:
   false positives and false negatives.
6. Exports short video snippets for wrong cases without showing them again.

Run from project root:

    python3 crossing_validation_tool.py

or override the video id:

    python3 crossing_validation_tool.py AdqE7mFQ7Y4
"""

import csv
import glob
import json
import math
import os
import pathlib
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse

import cv2
import numpy as np
import polars as pl
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

import common
from custom_logger import CustomLogger
from logmod import logs
from utils.core.metadata import MetaData
from utils.crossing.detection import Detection


logs(show_level=common.get_configs("logger_level"), show_color=True)
logger = CustomLogger(__name__)

metadata = MetaData()
detection = Detection()


# ============================================================
# EDIT ONLY THIS SECTION
# ============================================================
video_id = "AdqE7mFQ7Y4"

video_dir = "videos"
bbox_dir = common.get_configs("data")
output_root = "data/crossing_validation"

DF_MAPPING_CSV_PATH = common.get_configs("mapping")

FTP_BASE_URL = common.get_configs("ftp_base_url")
FTP_USERNAME = common.get_secrets("ftp_username")
FTP_PASSWORD = common.get_secrets("ftp_password")
FTP_TOKEN = common.get_secrets("ftp_token")

CROSS_MIN_X = common.get_configs("boundary_left")
CROSS_MAX_X = common.get_configs("boundary_right")

MIN_CONFIDENCE = common.get_configs("min_confidence")

DETECTION_TOL = 0.00
MIN_TRACK_FRAMES = 10
MIN_ROAD_FRAMES = 3

PERSON_CLASS = 0

PLAYBACK_SPEED = 1.0
SHOW_BOUNDARIES_DURING_REVIEW = True

SNIPPET_BEFORE_FRAMES = 60
SNIPPET_AFTER_FRAMES = 90

# Set True if you want snippets for every candidate, not only FP/FN.
EXPORT_ALL_CANDIDATE_SNIPPETS = False


# ============================================================
# Data models
# ============================================================
@dataclass
class CrossingEvent:
    source: str
    video_id: str
    unique_id: Optional[Any]
    frame_count: Optional[int]
    segment_frame_index: Optional[int]
    video_frame_index: Optional[int]
    filter_status: str = ""
    rejection_reason: str = ""
    direction: str = ""
    human_label: str = ""
    error_type: str = ""
    diagnosis: str = ""
    diagnosis_note: str = ""
    snippet_path: str = ""

    def as_row(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "video_id": self.video_id,
            "unique_id": self.unique_id,
            "frame_count": self.frame_count,
            "segment_frame_index": self.segment_frame_index,
            "video_frame_index": self.video_frame_index,
            "filter_status": self.filter_status,
            "rejection_reason": self.rejection_reason,
            "direction": self.direction,
            "human_label": self.human_label,
            "error_type": self.error_type,
            "diagnosis": self.diagnosis,
            "diagnosis_note": self.diagnosis_note,
            "snippet_path": self.snippet_path,
        }


# ============================================================
# CSV discovery + parsing
# ============================================================
def find_csv_for_video(video_id_value: str, data_dir="data/bbox") -> str:
    search_dirs: List[str] = []

    if isinstance(data_dir, (list, tuple, set)):
        for folder_path in data_dir:
            if folder_path is None:
                continue
            folder_path = str(folder_path)
            search_dirs.append(folder_path)

            for subfolder in common.get_configs("sub_domain"):
                search_dirs.append(os.path.join(folder_path, str(subfolder)))
    else:
        search_dirs.append(str(data_dir))

    # Keep the old direct location as a fallback.
    search_dirs.append("data/bbox")

    # De-duplicate while preserving order.
    clean_dirs: List[str] = []
    for folder_path in search_dirs:
        if folder_path and folder_path not in clean_dirs:
            clean_dirs.append(folder_path)

    matches: List[str] = []
    searched_patterns: List[str] = []

    for folder_path in clean_dirs:
        pattern = os.path.join(folder_path, f"{video_id_value}_*.csv")
        searched_patterns.append(pattern)
        matches.extend(glob.glob(pattern))

    if not matches:
        # Fallback for cases where CSVs are one level deeper than expected.
        for folder_path in clean_dirs:
            pattern = os.path.join(folder_path, "**", f"{video_id_value}_*.csv")
            searched_patterns.append(pattern)
            matches.extend(glob.glob(pattern, recursive=True))

    if not matches:
        raise FileNotFoundError(
            f"No CSV found for video_id='{video_id_value}'. Searched patterns: {searched_patterns}"
        )

    matches = list(set(matches))
    matches.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return matches[0]


def parse_csv_filename(csv_path: str) -> Tuple[str, float, str, str]:
    """
    Expected: {video_id}_{start_seconds}_{fps}.csv
    video_id may contain underscores; split from the end.
    """
    base = os.path.basename(csv_path)
    stem = os.path.splitext(base)[0]
    parts = stem.split("_")
    if len(parts) < 3:
        raise ValueError(f"CSV filename does not match expected pattern: {base}")

    fps_str = parts[-1]
    start_str = parts[-2]
    vid = "_".join(parts[:-2])

    return vid, float(start_str), fps_str, stem


def load_detection_csv(csv_path: str) -> pl.DataFrame:
    df = pl.read_csv(csv_path)

    required = {"yolo-id", "x-center", "y-center", "width", "height", "unique-id", "confidence", "frame-count"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {sorted(missing)}")

    if "confidence" in df.columns:
        df = df.filter(pl.col("confidence").cast(pl.Float64, strict=False) >= float(MIN_CONFIDENCE))

    return df


def dedup_per_frame(df: pl.DataFrame) -> pl.DataFrame:
    if "confidence" not in df.columns:
        return df.unique(subset=["yolo-id", "unique-id", "frame-count"], keep="first")

    return (
        df.sort(
            ["yolo-id", "unique-id", "frame-count", "confidence"],
            descending=[False, False, False, True],
        )
        .unique(subset=["yolo-id", "unique-id", "frame-count"], keep="first")
    )


# ============================================================
# Video helpers
# ============================================================
def get_video_fps(video_path: str) -> float:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 30.0
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if fps and fps > 0:
        return float(fps)
    return 30.0


def get_video_resolution_label(video_path: str) -> str:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return "unknown"
    height = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()
    if height <= 0:
        return "unknown"
    return f"{height}p"


def is_valid_video_file(video_path: str) -> bool:
    if not os.path.exists(video_path):
        return False
    if os.path.getsize(video_path) <= 0:
        return False

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        return False

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ok, _frame = cap.read()
    cap.release()

    return bool(ok) and frame_count > 0


def estimated_video_frame_index(start_seconds: float, segment_frame_index: int, video_fps: float) -> int:
    """
    Used only for seeking/display/exporting snippets. Matching remains frame-count based.
    """
    return int(round(float(start_seconds) * float(video_fps))) + int(segment_frame_index)


def xywh_to_xyxy_norm(xc: float, yc: float, w: float, h: float, W: int, H: int) -> Tuple[int, int, int, int]:
    xc_px, yc_px = xc * W, yc * H
    w_px, h_px = w * W, h * H

    x1 = int(round(xc_px - w_px / 2.0))
    y1 = int(round(yc_px - h_px / 2.0))
    x2 = int(round(xc_px + w_px / 2.0))
    y2 = int(round(yc_px + h_px / 2.0))

    x1 = max(0, min(W - 1, x1))
    y1 = max(0, min(H - 1, y1))
    x2 = max(0, min(W - 1, x2))
    y2 = max(0, min(H - 1, y2))

    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    return x1, y1, x2, y2


def draw_event_overlay(
    frame: np.ndarray,
    *,
    df: pl.DataFrame,
    csv_frame: int,
    event: CrossingEvent,
    boundary_left: float,
    boundary_right: float,
    title: str,
    line2: str,
    line3: str = "",
) -> np.ndarray:
    H, W = frame.shape[:2]

    if SHOW_BOUNDARIES_DURING_REVIEW:
        bx1 = int(round(float(boundary_left) * W))
        bx2 = int(round(float(boundary_right) * W))
        cv2.line(frame, (bx1, 0), (bx1, H - 1), (255, 255, 255), 2)
        cv2.line(frame, (bx2, 0), (bx2, H - 1), (255, 255, 255), 2)

    if event.unique_id is not None:
        rows = df.filter(
            (pl.col("yolo-id") == PERSON_CLASS)
            & (pl.col("unique-id") == event.unique_id)
            & (pl.col("frame-count") == int(csv_frame))
        )

        if rows.height > 0:
            row = rows.row(0, named=True)
            x1, y1, x2, y2 = xywh_to_xyxy_norm(
                float(row["x-center"]),
                float(row["y-center"]),
                float(row["width"]),
                float(row["height"]),
                W,
                H,
            )
            color = (0, 255, 255)
            if event.error_type == "FP":
                color = (0, 0, 255)
            elif event.error_type == "FN":
                color = (255, 0, 255)
            elif event.filter_status == "valid_after_filters":
                color = (0, 255, 0)
            elif event.filter_status == "rejected_by_filters":
                color = (0, 165, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            cv2.putText(
                frame,
                f"id={event.unique_id}",
                (x1, max(0, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
                cv2.LINE_AA,
            )

        # Draw short trajectory for the candidate.
        tr = (
            df.filter((pl.col("yolo-id") == PERSON_CLASS) & (pl.col("unique-id") == event.unique_id))
            .select(["frame-count", "x-center", "y-center"])
            .sort("frame-count")
        )
        if tr.height > 1:
            tr2 = tr.filter(
                (pl.col("frame-count") >= int(csv_frame) - SNIPPET_BEFORE_FRAMES)
                & (pl.col("frame-count") <= int(csv_frame) + SNIPPET_AFTER_FRAMES)
            )
            pts = []
            for r in tr2.iter_rows(named=True):
                x = int(round(float(r["x-center"]) * W))
                y = int(round(float(r["y-center"]) * H))
                pts.append((x, y))
            for p1, p2 in zip(pts[:-1], pts[1:]):
                cv2.line(frame, p1, p2, (255, 255, 0), 2)

    cv2.rectangle(frame, (10, 10), (W - 10, 115), (0, 0, 0), -1)
    cv2.putText(frame, title, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, line2, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    if line3:
        cv2.putText(frame, line3, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)

    return frame


# ============================================================
# FTP / HTTP file server download with resume
# ============================================================
def _content_range_total(content_range: str) -> Optional[int]:
    if not content_range or "/" not in content_range:
        return None
    tail = content_range.rsplit("/", 1)[-1].strip()
    if tail == "*":
        return None
    try:
        return int(tail)
    except Exception:
        return None


def _download_url_with_resume(
    session: requests.Session,
    url: str,
    local_path: str,
    *,
    params: Optional[dict],
    timeout: int = 20,
    max_retries: int = 8,
) -> bool:
    os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)

    for attempt in range(1, max_retries + 1):
        existing_size = os.path.getsize(local_path) if os.path.exists(local_path) else 0
        headers = {}
        mode = "wb"

        if existing_size > 0:
            headers["Range"] = f"bytes={existing_size}-"
            mode = "ab"

        try:
            response = session.get(url, timeout=timeout, params=params, headers=headers, stream=True)

            if response.status_code == 404:
                return False

            if response.status_code == 416 and existing_size > 0:
                return is_valid_video_file(local_path)

            response.raise_for_status()

            if existing_size > 0 and response.status_code == 200:
                logger.warning("Server ignored Range header. Restarting download from zero.")
                existing_size = 0
                mode = "wb"

            content_length = int(response.headers.get("content-length", 0) or 0)
            total = None

            if response.status_code == 206:
                total = _content_range_total(response.headers.get("content-range", ""))
            elif content_length:
                total = content_length

            progress_total = total if total else None
            initial = existing_size if response.status_code == 206 else 0

            with open(local_path, mode) as f, tqdm(
                total=progress_total,
                initial=initial,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=f"Downloading from ftp: {os.path.basename(local_path)}",
            ) as bar:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    f.write(chunk)
                    bar.update(len(chunk))

            if is_valid_video_file(local_path):
                return True

            logger.warning(f"Downloaded file is not readable yet, retrying. Attempt {attempt}/{max_retries}")

        except requests.RequestException as exc:
            logger.warning(f"Download interrupted on attempt {attempt}/{max_retries}: {exc}")

        time.sleep(min(10, attempt * 2))

    return False


def download_videos_from_ftp(
    filename: str,
    base_url: Optional[str] = None,
    out_dir: str = ".",
    username: Optional[str] = None,
    password: Optional[str] = None,
    token: Optional[str] = None,
    timeout: int = 20,
    max_pages: int = 500,
) -> Optional[Tuple[str, str, str, float]]:
    if not base_url:
        logger.error("Base URL is missing.")
        return None

    base = base_url if base_url.endswith("/") else base_url + "/"

    if username == "":
        username = None
    if password == "":
        password = None

    filename_with_ext = filename if filename.lower().endswith(".mp4") else f"{filename}.mp4"
    filename_lower = filename_with_ext.lower()
    aliases = ["tue1", "tue2", "tue3", "tue4"]
    req_params = {"token": token} if token else None

    logger.info(f"Starting download for '{filename_with_ext}'")

    with requests.Session() as session:
        if username and password:
            session.auth = (username, password)
        session.headers.update({"User-Agent": "multi-fileserver-downloader/1.0"})

        os.makedirs(out_dir, exist_ok=True)
        local_path = os.path.join(out_dir, filename_with_ext)

        for alias in aliases:
            direct_url = urljoin(base, f"v/{alias}/files/{filename_with_ext}")
            logger.info(f"Trying direct URL: {direct_url}")

            ok = _download_url_with_resume(
                session,
                direct_url,
                local_path,
                params=req_params,
                timeout=timeout,
            )
            if ok:
                fps = get_video_fps(local_path)
                resolution = get_video_resolution_label(local_path)
                logger.info(f"Saved '{filename_with_ext}' (res={resolution}, fps={fps})")
                return local_path, filename, resolution, fps

        visited: set[str] = set()

        def fetch(url: str) -> Optional[requests.Response]:
            try:
                r = session.get(url, timeout=timeout, params=req_params)
                if r.status_code == 404:
                    return None
                r.raise_for_status()
                return r
            except requests.RequestException as exc:
                logger.warning(f"Request failed [{url}]: {exc}")
                return None

        def is_dir_link(href: str) -> bool:
            return href.startswith("/v/") and "/browse" in href

        def is_file_link(href: str) -> bool:
            return "/files/" in href

        def crawl(start_url: str) -> Optional[str]:
            stack = [start_url]
            pages_seen = 0

            while stack:
                url = stack.pop()
                if url in visited:
                    continue

                visited.add(url)
                pages_seen += 1

                if pages_seen > max_pages:
                    logger.warning(f"Crawl aborted after {max_pages} pages.")
                    return None

                response = fetch(url)
                if response is None:
                    continue

                try:
                    soup = BeautifulSoup(response.text, "html.parser")
                except Exception as exc:
                    logger.warning(f"HTML parse failed at {url}: {exc}")
                    continue

                for a in soup.find_all("a"):
                    href = (a.get("href") or "").strip()
                    if not href:
                        continue

                    full = urljoin(url, href)

                    if is_file_link(href):
                        anchor_text = (a.text or "").strip().lower()
                        tail = pathlib.PurePosixPath(urlparse(full).path).name.lower()
                        if anchor_text == filename_lower or tail == filename_lower:
                            return full

                    if is_dir_link(href):
                        stack.append(full)

            return None

        for alias in aliases:
            start_url = urljoin(base, f"v/{alias}/browse")
            found_url = crawl(start_url)
            if not found_url:
                continue

            ok = _download_url_with_resume(
                session,
                found_url,
                local_path,
                params=req_params,
                timeout=timeout,
            )
            if ok:
                fps = get_video_fps(local_path)
                resolution = get_video_resolution_label(local_path)
                logger.info(f"Saved '{filename_with_ext}' (res={resolution}, fps={fps})")
                return local_path, filename, resolution, fps

    logger.warning(f"File '{filename_with_ext}' not found or could not be downloaded.")
    return None


def ensure_video_available(video_id_value: str, video_path: str) -> str:
    if is_valid_video_file(video_path):
        logger.info(f"Using existing local video: {video_path}")
        return video_path

    if os.path.exists(video_path):
        logger.warning(f"Existing local video is broken or incomplete. Will resume/re-download: {video_path}")

    result = download_videos_from_ftp(
        filename=video_id_value,
        base_url=FTP_BASE_URL,
        out_dir=os.path.dirname(video_path) or ".",
        username=FTP_USERNAME,
        password=FTP_PASSWORD,
        token=FTP_TOKEN,
    )

    if result is None:
        raise FileNotFoundError(f"Could not download video from FTP for video_id='{video_id_value}'")

    downloaded_path = result[0]
    if not is_valid_video_file(downloaded_path):
        raise RuntimeError(f"Downloaded video is not readable: {downloaded_path}")

    return downloaded_path


# ============================================================
# Crossing event extraction using project Detection filters
# ============================================================
def _track_state_arrays(
    track: pl.DataFrame,
    *,
    min_x: float,
    max_x: float,
    tol: float,
) -> Tuple[np.ndarray, np.ndarray]:
    track = track.sort("frame-count")
    frames = track.get_column("frame-count").cast(pl.Int64).to_numpy()
    x = track.get_column("x-center").cast(pl.Float64).to_numpy()

    left_hard = float(min_x) - float(tol)
    left_soft = float(min_x) + float(tol)
    right_soft = float(max_x) - float(tol)
    right_hard = float(max_x) + float(tol)

    states = np.empty(x.size, dtype=np.int8)

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
            s = s

        states[i] = s

    return frames, states


def crossing_event_frame_for_track(
    df: pl.DataFrame,
    uid: Any,
    *,
    min_x: float,
    max_x: float,
    tol: float,
    min_road_frames: int,
) -> Tuple[int, str]:
    track = (
        df.filter((pl.col("yolo-id") == PERSON_CLASS) & (pl.col("unique-id") == uid))
        .select(["unique-id", "frame-count", "x-center"])
        .sort("frame-count")
    )

    if track.height == 0:
        raise ValueError(f"No person track for uid={uid}")

    frames, states = _track_state_arrays(track, min_x=min_x, max_x=max_x, tol=tol)

    is_left = states == 0
    is_road = states == 1
    is_right = states == 2

    if int(is_road.sum()) < int(min_road_frames):
        mid_idx = int(len(frames) // 2)
        return int(frames[mid_idx]), "unknown"

    left_before = np.maximum.accumulate(is_left)
    right_before = np.maximum.accumulate(is_right)
    left_after = np.maximum.accumulate(is_left[::-1])[::-1]
    right_after = np.maximum.accumulate(is_right[::-1])[::-1]

    ltr_mask = is_road & left_before & right_after
    rtl_mask = is_road & right_before & left_after
    crossing_mask = ltr_mask | rtl_mask

    if bool(crossing_mask.any()):
        idx = int(np.where(crossing_mask)[0][0])
        if bool(ltr_mask[idx]):
            return int(frames[idx]), "left_to_right"
        if bool(rtl_mask[idx]):
            return int(frames[idx]), "right_to_left"
        return int(frames[idx]), "unknown"

    road_idx = np.where(is_road)[0]
    if len(road_idx) > 0:
        idx = int(road_idx[0])
        return int(frames[idx]), "inside_road_band"

    mid_idx = int(len(frames) // 2)
    return int(frames[mid_idx]), "unknown"


def rejection_reason_for_uid(
    df: pl.DataFrame,
    uid: Any,
    avg_height: Optional[float],
) -> str:
    try:
        if Detection.is_rider_id(df, uid, avg_height):
            return "vehicle_associated_rider_or_passenger"
    except Exception as exc:
        return f"rider_filter_error:{exc}"

    try:
        if not Detection.is_valid_crossing(df, uid):
            return "invalid_crossing_camera_motion_or_static_reference"
    except Exception as exc:
        return f"validity_filter_error:{exc}"

    return "not_in_filtered_output"


def compute_automatic_events(
    df: pl.DataFrame,
    *,
    video_id_full: str,
    df_mapping: pl.DataFrame,
    fc_min: int,
) -> Tuple[List[CrossingEvent], List[CrossingEvent], List[CrossingEvent]]:
    """
    Returns:
        valid_events: ids after Detection.pedestrian_crossing filters
        candidate_events: all state-machine crossing candidates
        rejected_events: candidates removed by filters
    """
    valid_ids, candidate_ids = detection.pedestrian_crossing(
        df,
        video_id_full,
        df_mapping,
        CROSS_MIN_X,
        CROSS_MAX_X,
        person_id=0,
        tol=DETECTION_TOL,
        min_track_frames=MIN_TRACK_FRAMES,
        min_road_frames=MIN_ROAD_FRAMES,
    )

    valid_id_set = set(valid_ids)
    candidate_id_set = set(candidate_ids)

    result = metadata.find_values_with_video_id(df_mapping, video_id_full)
    avg_height = None
    if result is not None:
        avg_height = result[15]

    valid_events: List[CrossingEvent] = []
    candidate_events: List[CrossingEvent] = []
    rejected_events: List[CrossingEvent] = []

    for uid in candidate_ids:
        try:
            frame_count, direction = crossing_event_frame_for_track(
                df,
                uid,
                min_x=CROSS_MIN_X,
                max_x=CROSS_MAX_X,
                tol=DETECTION_TOL,
                min_road_frames=MIN_ROAD_FRAMES,
            )
        except Exception as exc:
            logger.warning(f"Could not calculate crossing frame for uid={uid}: {exc}")
            continue

        segment_frame_index = int(frame_count) - int(fc_min)
        auto_filter_status = "valid_after_filters" if uid in valid_id_set else "rejected_by_filters"
        reject_reason = "" if uid in valid_id_set else rejection_reason_for_uid(df, uid, avg_height)

        candidate_event = CrossingEvent(
            source="algorithm_candidate",
            video_id=video_id_full,
            unique_id=uid,
            frame_count=int(frame_count),
            segment_frame_index=int(segment_frame_index),
            video_frame_index=None,
            filter_status=auto_filter_status,
            rejection_reason=reject_reason,
            direction=direction,
        )
        candidate_events.append(candidate_event)

        if uid in valid_id_set:
            valid_events.append(
                CrossingEvent(
                    source="algorithm_valid",
                    video_id=video_id_full,
                    unique_id=uid,
                    frame_count=int(frame_count),
                    segment_frame_index=int(segment_frame_index),
                    video_frame_index=None,
                    filter_status="valid_after_filters",
                    rejection_reason="",
                    direction=direction,
                )
            )
        else:
            rejected_events.append(
                CrossingEvent(
                    source="algorithm_rejected",
                    video_id=video_id_full,
                    unique_id=uid,
                    frame_count=int(frame_count),
                    segment_frame_index=int(segment_frame_index),
                    video_frame_index=None,
                    filter_status="rejected_by_filters",
                    rejection_reason=reject_reason,
                    direction=direction,
                )
            )

    valid_events.sort(key=lambda x: x.frame_count if x.frame_count is not None else 10**18)
    candidate_events.sort(key=lambda x: x.frame_count if x.frame_count is not None else 10**18)
    rejected_events.sort(key=lambda x: x.frame_count if x.frame_count is not None else 10**18)

    logger.info(f"Automatic candidate crossings: {len(candidate_id_set)}")
    logger.info(f"Automatic valid crossings after filters: {len(valid_id_set)}")
    logger.info(f"Automatic rejected crossings: {len(candidate_id_set - valid_id_set)}")

    return valid_events, candidate_events, rejected_events


# ============================================================
# Output helpers
# ============================================================
def write_events_csv(path: str, events: List[CrossingEvent]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    fieldnames = [
        "source",
        "video_id",
        "unique_id",
        "frame_count",
        "segment_frame_index",
        "video_frame_index",
        "filter_status",
        "rejection_reason",
        "direction",
        "human_label",
        "error_type",
        "diagnosis",
        "diagnosis_note",
        "snippet_path",
    ]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for event in events:
            writer.writerow(event.as_row())


def _clean_csv_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    text = str(value)
    if text.lower() in ("nan", "none", "null"):
        return None
    return value


def read_events_csv(path: str) -> List[CrossingEvent]:
    if not os.path.exists(path):
        return []

    df = pl.read_csv(path)
    events: List[CrossingEvent] = []

    for row in df.iter_rows(named=True):
        frame_count = _clean_csv_value(row.get("frame_count"))
        segment_frame_index = _clean_csv_value(row.get("segment_frame_index"))
        video_frame_index = _clean_csv_value(row.get("video_frame_index"))

        events.append(
            CrossingEvent(
                source=str(row.get("source", "")),
                video_id=str(row.get("video_id", "")),
                unique_id=_clean_csv_value(row.get("unique_id")),
                frame_count=None if frame_count is None else int(frame_count),
                segment_frame_index=None if segment_frame_index is None else int(segment_frame_index),
                video_frame_index=None if video_frame_index is None else int(video_frame_index),
                filter_status=str(row.get("filter_status", "")),
                rejection_reason=str(row.get("rejection_reason", "")),
                direction=str(row.get("direction", "")),
                human_label=str(row.get("human_label", "")),
                error_type=str(row.get("error_type", "")),
                diagnosis=str(row.get("diagnosis", "")),
                diagnosis_note=str(row.get("diagnosis_note", "")),
                snippet_path=str(row.get("snippet_path", "")),
            )
        )

    return events


def merge_existing_labels(events: List[CrossingEvent], label_csv: str) -> List[CrossingEvent]:
    existing = read_events_csv(label_csv)
    labels_by_key: Dict[Tuple[str, str], CrossingEvent] = {}

    for item in existing:
        key = (str(item.unique_id), str(item.frame_count))
        labels_by_key[key] = item

    for event in events:
        key = (str(event.unique_id), str(event.frame_count))
        old = labels_by_key.get(key)
        if old is None:
            continue
        event.human_label = old.human_label
        event.error_type = old.error_type
        event.diagnosis = old.diagnosis
        event.diagnosis_note = old.diagnosis_note
        event.snippet_path = old.snippet_path

    return events


def write_json(path: str, data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def write_confusion_matrix_csv(path: str, metrics: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    rows = [
        {"ground_truth": "actual_crossing", "auto_crossing": metrics["TP"], "auto_not_crossing": metrics["FN"]},
        {"ground_truth": "fake_candidate", "auto_crossing": metrics["FP"], "auto_not_crossing": metrics["TN"]},
    ]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["ground_truth", "auto_crossing", "auto_not_crossing"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def calculate_filter_metrics(events: List[CrossingEvent]) -> Dict[str, Any]:
    labelled = [e for e in events if e.human_label in ("actual_crossing", "fake_candidate")]

    tp = sum(1 for e in labelled if e.human_label == "actual_crossing" and e.filter_status == "valid_after_filters")
    fn = sum(1 for e in labelled if e.human_label == "actual_crossing" and e.filter_status == "rejected_by_filters")
    fp = sum(1 for e in labelled if e.human_label == "fake_candidate" and e.filter_status == "valid_after_filters")
    tn = sum(1 for e in labelled if e.human_label == "fake_candidate" and e.filter_status == "rejected_by_filters")

    precision = tp / (tp + fp) if (tp + fp) else math.nan
    recall = tp / (tp + fn) if (tp + fn) else math.nan
    specificity = tn / (tn + fp) if (tn + fp) else math.nan
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision == precision and recall == recall and (precision + recall)
        else math.nan
    )
    accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) else math.nan

    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "precision": None if math.isnan(precision) else round(precision, 4),
        "recall": None if math.isnan(recall) else round(recall, 4),
        "specificity": None if math.isnan(specificity) else round(specificity, 4),
        "f1": None if math.isnan(f1) else round(f1, 4),
        "accuracy": None if math.isnan(accuracy) else round(accuracy, 4),
        "labelled_candidates": len(labelled),
        "unlabelled_candidates": len([e for e in events if e.human_label == ""]),
        "unsure_candidates": len([e for e in events if e.human_label == "unsure"]),
        "note": "This is filter-level tuning: candidates are generated first, then kept/rejected by filters.",
    }


def update_error_types(events: List[CrossingEvent]) -> List[CrossingEvent]:
    for event in events:
        event.error_type = ""
        if event.human_label == "fake_candidate" and event.filter_status == "valid_after_filters":
            event.error_type = "FP"
        elif event.human_label == "actual_crossing" and event.filter_status == "rejected_by_filters":
            event.error_type = "FN"
    return events


def get_error_events(events: List[CrossingEvent]) -> List[CrossingEvent]:
    return [e for e in update_error_types(events) if e.error_type in ("FP", "FN")]


# ============================================================
# Candidate labelling UI
# ============================================================
class CandidateReviewer:
    def __init__(
        self,
        *,
        video_path: str,
        df: pl.DataFrame,
        events: List[CrossingEvent],
        output_csv: str,
        start_seconds: float,
    ):
        self.video_path = video_path
        self.df = df
        self.events = events
        self.output_csv = output_csv
        self.start_seconds = float(start_seconds)
        self.index = 0

        self.video_fps = get_video_fps(video_path)
        self.delay_ms = max(1, int(round(1000.0 / max(float(self.video_fps) * PLAYBACK_SPEED, 1e-9))))
        self.window_name = "Candidate tuning review"

    def save(self) -> None:
        write_events_csv(self.output_csv, self.events)

    def label_current(self, label: str) -> None:
        event = self.events[self.index]
        event.human_label = label
        update_error_types(self.events)
        self.save()
        logger.info(
            f"Candidate {self.index + 1}/{len(self.events)} uid={event.unique_id} "
            f"frame={event.frame_count} labelled as {label}"
        )

    def show_current_snippet(self) -> Optional[int]:
        event = self.events[self.index]
        if event.segment_frame_index is None:
            return None

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {self.video_path}")

        centre_video_frame = estimated_video_frame_index(
            self.start_seconds,
            int(event.segment_frame_index),
            self.video_fps,
        )
        start_video_frame = max(0, centre_video_frame - SNIPPET_BEFORE_FRAMES)
        end_video_frame = centre_video_frame + SNIPPET_AFTER_FRAMES

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_video_frame)

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            current_video_frame = int(round(cap.get(cv2.CAP_PROP_POS_FRAMES)))
            offset_from_centre = current_video_frame - centre_video_frame
            csv_frame = int(event.frame_count or 0) + int(offset_from_centre)

            title = (
                f"Candidate {self.index + 1}/{len(self.events)} | "
                f"uid={event.unique_id} | frame={event.frame_count} | label={event.human_label or 'UNLABELLED'}"
            )
            line2 = (
                f"auto={event.filter_status} | reject={event.rejection_reason or '-'} | "
                "A=actual F=fake U=unsure N=next B=back Q=quit"
            )
            line3 = f"direction={event.direction} | event_type={event.error_type or '-'}"

            frame = draw_event_overlay(
                frame,
                df=self.df,
                csv_frame=csv_frame,
                event=event,
                boundary_left=CROSS_MIN_X,
                boundary_right=CROSS_MAX_X,
                title=title,
                line2=line2,
                line3=line3,
            )

            cv2.imshow(self.window_name, frame)
            key = cv2.waitKey(self.delay_ms) & 0xFF

            if key in (ord("a"), ord("A")):
                self.label_current("actual_crossing")
                cap.release()
                return ord("n")
            if key in (ord("f"), ord("F")):
                self.label_current("fake_candidate")
                cap.release()
                return ord("n")
            if key in (ord("u"), ord("U")):
                self.label_current("unsure")
                cap.release()
                return ord("n")
            if key in (ord("n"), ord("N")):
                cap.release()
                return ord("n")
            if key in (ord("b"), ord("B")):
                cap.release()
                return ord("b")
            if key in (ord("r"), ord("R")):
                cap.release()
                return ord("r")
            if key in (ord("q"), ord("Q")):
                cap.release()
                return ord("q")

            if current_video_frame >= end_video_frame:
                break

        cap.release()
        return None

    def run(self) -> List[CrossingEvent]:
        if not self.events:
            logger.info("No candidate events to review.")
            return self.events

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

        while 0 <= self.index < len(self.events):
            action = self.show_current_snippet()

            if action == ord("q"):
                break
            if action == ord("b"):
                self.index = max(0, self.index - 1)
                continue
            if action == ord("r"):
                continue

            self.index += 1

        cv2.destroyWindow(self.window_name)
        self.save()
        return self.events


# ============================================================
# Error snippet export and diagnosis UI
# ============================================================
def export_snippet_video(
    *,
    video_path: str,
    df: pl.DataFrame,
    event: CrossingEvent,
    start_seconds: float,
    output_path: str,
) -> str:
    if event.segment_frame_index is None or event.frame_count is None:
        return ""

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    video_fps = get_video_fps(video_path)
    centre_video_frame = estimated_video_frame_index(start_seconds, int(event.segment_frame_index), video_fps)
    start_video_frame = max(0, centre_video_frame - SNIPPET_BEFORE_FRAMES)
    end_video_frame = centre_video_frame + SNIPPET_AFTER_FRAMES

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_video_frame)

    ok, frame = cap.read()
    if not ok:
        cap.release()
        return ""

    H, W = frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore
    out = cv2.VideoWriter(output_path, fourcc, float(video_fps), (W, H))
    if not out.isOpened():
        cap.release()
        return ""

    current_video_frame = int(round(cap.get(cv2.CAP_PROP_POS_FRAMES)))
    while ok and current_video_frame <= end_video_frame:
        offset_from_centre = current_video_frame - centre_video_frame
        csv_frame = int(event.frame_count) + int(offset_from_centre)

        title = (
            f"{event.error_type} | uid={event.unique_id} | csv_frame={event.frame_count} | "
            f"human={event.human_label}"
        )
        line2 = f"auto={event.filter_status} | reject={event.rejection_reason or '-'}"
        line3 = f"diagnosis={event.diagnosis or '-'}"

        frame = draw_event_overlay(
            frame,
            df=df,
            csv_frame=csv_frame,
            event=event,
            boundary_left=CROSS_MIN_X,
            boundary_right=CROSS_MAX_X,
            title=title,
            line2=line2,
            line3=line3,
        )
        out.write(frame)

        ok, frame = cap.read()
        current_video_frame = int(round(cap.get(cv2.CAP_PROP_POS_FRAMES)))

    cap.release()
    out.release()

    return output_path


def export_error_snippets(
    *,
    video_path: str,
    df: pl.DataFrame,
    error_events: List[CrossingEvent],
    start_seconds: float,
    snippets_dir: str,
) -> List[CrossingEvent]:
    os.makedirs(snippets_dir, exist_ok=True)

    for idx, event in enumerate(error_events, start=1):
        if event.snippet_path and os.path.exists(event.snippet_path):
            continue

        safe_uid = str(event.unique_id).replace("/", "_")
        frame_part = "unknown" if event.frame_count is None else str(event.frame_count)
        fname = f"{idx:03d}_{event.error_type}_uid_{safe_uid}_frame_{frame_part}.mp4"
        out_path = os.path.join(snippets_dir, fname)

        try:
            event.snippet_path = export_snippet_video(
                video_path=video_path,
                df=df,
                event=event,
                start_seconds=start_seconds,
                output_path=out_path,
            )
        except Exception as exc:
            logger.warning(f"Could not export snippet for uid={event.unique_id}: {exc}")

    return error_events


def fp_reason_map() -> Dict[str, str]:
    return {
        "1": "vehicle_filter_missed_rider_or_passenger",
        "2": "camera_motion_filter_too_weak",
        "3": "boundary_or_road_band_too_loose",
        "4": "min_road_frames_too_low",
        "5": "tracking_fragment_or_duplicate",
        "6": "human_label_mistake",
        "7": "other",
    }


def fn_reason_map() -> Dict[str, str]:
    return {
        "1": "vehicle_filter_too_strict",
        "2": "camera_motion_filter_too_strict",
        "3": "boundary_or_road_band_too_strict",
        "4": "min_track_or_road_frames_too_high",
        "5": "static_reference_problem",
        "6": "human_label_mistake",
        "7": "other",
    }


def diagnosis_menu_text(event: CrossingEvent) -> str:
    if event.error_type == "FP":
        return (
            "1 vehicle_missed | 2 camera_weak | 3 boundary_loose | "
            "4 road_frames_low | 5 tracking | 6 label_mistake | 7 other"
        )
    return (
        "1 vehicle_too_strict | 2 camera_too_strict | 3 boundary_strict | "
        "4 track/road_high | 5 static_ref | 6 label_mistake | 7 other"
    )


class ErrorReviewer:
    def __init__(
        self,
        *,
        video_path: str,
        df: pl.DataFrame,
        error_events: List[CrossingEvent],
        output_csv: str,
        start_seconds: float,
    ):
        self.video_path = video_path
        self.df = df
        self.error_events = error_events
        self.output_csv = output_csv
        self.start_seconds = float(start_seconds)
        self.index = 0

        self.video_fps = get_video_fps(video_path)
        self.delay_ms = max(1, int(round(1000.0 / max(float(self.video_fps) * PLAYBACK_SPEED, 1e-9))))
        self.window_name = "Wrong case diagnosis review"

    def save(self) -> None:
        write_events_csv(self.output_csv, self.error_events)

    def set_diagnosis(self, key: str) -> None:
        event = self.error_events[self.index]
        mapping = fp_reason_map() if event.error_type == "FP" else fn_reason_map()
        reason = mapping.get(key)
        if reason is None:
            return

        event.diagnosis = reason
        self.save()
        logger.info(
            f"{event.error_type} case {self.index + 1}/{len(self.error_events)} "
            f"uid={event.unique_id} diagnosis={reason}"
        )

    def show_current_snippet(self) -> Optional[int]:
        event = self.error_events[self.index]
        if event.segment_frame_index is None:
            return None

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {self.video_path}")

        centre_video_frame = estimated_video_frame_index(
            self.start_seconds,
            int(event.segment_frame_index),
            self.video_fps,
        )
        start_video_frame = max(0, centre_video_frame - SNIPPET_BEFORE_FRAMES)
        end_video_frame = centre_video_frame + SNIPPET_AFTER_FRAMES

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_video_frame)

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            current_video_frame = int(round(cap.get(cv2.CAP_PROP_POS_FRAMES)))
            offset_from_centre = current_video_frame - centre_video_frame
            csv_frame = int(event.frame_count or 0) + int(offset_from_centre)

            title = (
                f"{event.error_type} {self.index + 1}/{len(self.error_events)} | "
                f"uid={event.unique_id} | frame={event.frame_count} | diagnosis={event.diagnosis or 'UNSET'}"
            )
            line2 = (
                f"human={event.human_label} | auto={event.filter_status} | "
                f"reject={event.rejection_reason or '-'}"
            )
            line3 = diagnosis_menu_text(event) + " | N next | B back | R replay | Q quit"

            frame = draw_event_overlay(
                frame,
                df=self.df,
                csv_frame=csv_frame,
                event=event,
                boundary_left=CROSS_MIN_X,
                boundary_right=CROSS_MAX_X,
                title=title,
                line2=line2,
                line3=line3,
            )

            cv2.imshow(self.window_name, frame)
            key = cv2.waitKey(self.delay_ms) & 0xFF

            if key in (ord("1"), ord("2"), ord("3"), ord("4"), ord("5"), ord("6"), ord("7")):
                self.set_diagnosis(chr(key))
                cap.release()
                return ord("n")
            if key in (ord("n"), ord("N")):
                cap.release()
                return ord("n")
            if key in (ord("b"), ord("B")):
                cap.release()
                return ord("b")
            if key in (ord("r"), ord("R")):
                cap.release()
                return ord("r")
            if key in (ord("q"), ord("Q")):
                cap.release()
                return ord("q")

            if current_video_frame >= end_video_frame:
                break

        cap.release()
        return None

    def run(self) -> List[CrossingEvent]:
        if not self.error_events:
            logger.info("No false positive or false negative cases to diagnose.")
            return self.error_events

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

        while 0 <= self.index < len(self.error_events):
            action = self.show_current_snippet()

            if action == ord("q"):
                break
            if action == ord("b"):
                self.index = max(0, self.index - 1)
                continue
            if action == ord("r"):
                continue

            self.index += 1

        cv2.destroyWindow(self.window_name)
        self.save()
        return self.error_events


def build_tuning_recommendations(error_events: List[CrossingEvent]) -> Dict[str, Any]:
    counts: Dict[str, int] = {}
    by_error_type: Dict[str, Dict[str, int]] = {"FP": {}, "FN": {}}

    for event in error_events:
        diagnosis = event.diagnosis or "undiagnosed"
        counts[diagnosis] = counts.get(diagnosis, 0) + 1
        if event.error_type in by_error_type:
            by_error_type[event.error_type][diagnosis] = by_error_type[event.error_type].get(diagnosis, 0) + 1

    suggestions: List[str] = []

    if counts.get("vehicle_filter_missed_rider_or_passenger", 0) > 0:
        suggestions.append(
            "FP vehicle cases: make vehicle association more sensitive, but check if this creates FN. "
            "Possible directions: lower min_continuous_shared_frames, lower min_vehicle_width_ratio_frames, "
            "or lower prox_req in Detection.is_rider_id."
        )
    if counts.get("vehicle_filter_too_strict", 0) > 0:
        suggestions.append(
            "FN vehicle cases: vehicle association is too aggressive. Possible directions: increase "
            "min_continuous_shared_frames, increase min_vehicle_width_ratio_frames, or increase prox_req."
        )
    if counts.get("camera_motion_filter_too_weak", 0) > 0:
        suggestions.append(
            "FP camera-motion cases: make Detection.is_valid_crossing stricter for static-reference rejection."
        )
    if counts.get("camera_motion_filter_too_strict", 0) > 0:
        suggestions.append(
            "FN camera-motion cases: relax Detection.is_valid_crossing because real crossings are being rejected."
        )
    if counts.get("boundary_or_road_band_too_loose", 0) > 0 or counts.get("min_road_frames_too_low", 0) > 0:
        suggestions.append(
            "FP road-band cases: increase MIN_ROAD_FRAMES or add a small DETECTION_TOL to reduce boundary flicker."
        )
    if counts.get("boundary_or_road_band_too_strict", 0) > 0 or counts.get("min_track_or_road_frames_too_high", 0) > 0:
        suggestions.append(
            "FN road-band cases: reduce MIN_ROAD_FRAMES or MIN_TRACK_FRAMES, or check boundary placement."
        )
    if counts.get("tracking_fragment_or_duplicate", 0) > 0:
        suggestions.append(
            "Tracking cases: inspect duplicate or fragmented unique-id tracks. Parameter tuning may not fully fix this"
        )

    return {
        "diagnosis_counts": counts,
        "diagnosis_counts_by_error_type": by_error_type,
        "suggestions": suggestions,
    }


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    if len(sys.argv) > 1:
        video_id = sys.argv[1]

    video_path = os.path.join(video_dir, f"{video_id}.mp4")
    video_path = ensure_video_available(video_id, video_path)

    csv_path = find_csv_for_video(video_id, data_dir=bbox_dir)
    csv_video_id, start_seconds, fps_str, video_id_full = parse_csv_filename(csv_path)

    df_mapping = pl.read_csv(DF_MAPPING_CSV_PATH)

    df = load_detection_csv(csv_path)
    df = dedup_per_frame(df)

    fc_min = int(df.get_column("frame-count").min())
    fc_max = int(df.get_column("frame-count").max())

    out_dir = os.path.join(output_root, video_id_full)
    snippets_dir = os.path.join(out_dir, "wrong_case_snippets")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(snippets_dir, exist_ok=True)

    auto_valid_csv = os.path.join(out_dir, "automatic_valid_crossings.csv")
    auto_candidates_csv = os.path.join(out_dir, "automatic_all_candidates.csv")
    auto_rejected_csv = os.path.join(out_dir, "automatic_rejected_crossings.csv")
    candidate_labels_csv = os.path.join(out_dir, "candidate_human_labels.csv")
    metrics_json = os.path.join(out_dir, "filter_tuning_metrics.json")
    matrix_csv = os.path.join(out_dir, "filter_tuning_confusion_matrix.csv")
    error_cases_csv = os.path.join(out_dir, "wrong_cases_fp_fn.csv")
    error_diagnosis_csv = os.path.join(out_dir, "wrong_case_diagnosis.csv")
    recommendations_json = os.path.join(out_dir, "tuning_recommendations.json")
    filter_summary_json = os.path.join(out_dir, "filter_summary.json")

    logger.info(f"Video: {video_path}")
    logger.info(f"CSV: {csv_path}")
    logger.info(f"CSV key: {video_id_full}")
    logger.info(f"CSV frame-count range: {fc_min} .. {fc_max}")
    logger.info(f"Boundary left: {CROSS_MIN_X}")
    logger.info(f"Boundary right: {CROSS_MAX_X}")
    logger.info(f"Minimum confidence: {MIN_CONFIDENCE}")

    auto_events, candidate_events, rejected_events = compute_automatic_events(
        df,
        video_id_full=video_id_full,
        df_mapping=df_mapping,
        fc_min=fc_min,
    )

    write_events_csv(auto_valid_csv, auto_events)
    write_events_csv(auto_candidates_csv, candidate_events)
    write_events_csv(auto_rejected_csv, rejected_events)

    rejection_counts: Dict[str, int] = {}
    for event in rejected_events:
        rejection_counts[event.rejection_reason] = rejection_counts.get(event.rejection_reason, 0) + 1

    filter_summary = {
        "video_id": video_id,
        "video_id_full": video_id_full,
        "csv_path": csv_path,
        "frame_count_min": fc_min,
        "frame_count_max": fc_max,
        "candidate_count_before_filters": len(candidate_events),
        "valid_count_after_filters": len(auto_events),
        "rejected_count_after_filters": len(rejected_events),
        "rejection_counts": rejection_counts,
        "filters": {
            "min_confidence": MIN_CONFIDENCE,
            "boundary_left": CROSS_MIN_X,
            "boundary_right": CROSS_MAX_X,
            "detection_tol": DETECTION_TOL,
            "min_track_frames": MIN_TRACK_FRAMES,
            "min_road_frames": MIN_ROAD_FRAMES,
            "vehicle_association_filter": "Detection.is_rider_id",
            "camera_motion_filter": "Detection.is_valid_crossing",
        },
    }
    write_json(filter_summary_json, filter_summary)

    candidate_events = merge_existing_labels(candidate_events, candidate_labels_csv)

    logger.info("\nCandidate review controls:")
    logger.info("  A = actual pedestrian crossing")
    logger.info("  F = fake candidate")
    logger.info("  U = unsure / ignore")
    logger.info("  N = next")
    logger.info("  B = previous")
    logger.info("  R = replay candidate")
    logger.info("  Q = save and stop candidate review")

    candidate_reviewer = CandidateReviewer(
        video_path=video_path,
        df=df,
        events=candidate_events,
        output_csv=candidate_labels_csv,
        start_seconds=start_seconds,
    )
    candidate_events = candidate_reviewer.run()
    candidate_events = update_error_types(candidate_events)

    metrics = calculate_filter_metrics(candidate_events)
    write_events_csv(candidate_labels_csv, candidate_events)
    write_json(metrics_json, metrics)
    write_confusion_matrix_csv(matrix_csv, metrics)

    error_events = get_error_events(candidate_events)
    error_events = export_error_snippets(
        video_path=video_path,
        df=df,
        error_events=error_events,
        start_seconds=start_seconds,
        snippets_dir=snippets_dir,
    )

    # Save only the conflicts found after your first review.
    # These are the FP/FN cases; they are not shown again for diagnosis.
    write_events_csv(error_cases_csv, error_events)
    write_events_csv(error_diagnosis_csv, error_events)

    # Keep candidate_human_labels.csv in sync with snippet paths for conflict rows.
    conflict_by_key: Dict[Tuple[str, str], CrossingEvent] = {}
    for event in error_events:
        conflict_by_key[(str(event.unique_id), str(event.frame_count))] = event

    for event in candidate_events:
        found = conflict_by_key.get((str(event.unique_id), str(event.frame_count)))
        if found is not None:
            event.error_type = found.error_type
            event.snippet_path = found.snippet_path

    write_events_csv(candidate_labels_csv, candidate_events)

    recommendations = build_tuning_recommendations(error_events)
    recommendations["note"] = (
        "Conflict snippets were exported after candidate labelling. "
        "The second FP/FN diagnosis review was skipped, so diagnosis fields remain empty unless edited manually."
    )
    write_json(recommendations_json, recommendations)

    logger.info("\nConflict snippets exported. No second FP/FN review was opened.")

    logger.info("\nTuning validation complete.")
    logger.info(f"Automatic valid crossings: {auto_valid_csv}")
    logger.info(f"Automatic all candidates: {auto_candidates_csv}")
    logger.info(f"Automatic rejected crossings: {auto_rejected_csv}")
    logger.info(f"Candidate human labels: {candidate_labels_csv}")
    logger.info(f"Filter tuning metrics: {metrics_json}")
    logger.info(f"Filter tuning confusion matrix: {matrix_csv}")
    logger.info(f"Wrong FP/FN cases: {error_cases_csv}")
    logger.info(f"Wrong-case diagnosis: {error_diagnosis_csv}")
    logger.info(f"Wrong-case snippets folder: {snippets_dir}")
    logger.info(f"Tuning recommendations: {recommendations_json}")
    logger.info("Metrics summary:\n{}", json.dumps(metrics, indent=2))
