"""
Cyclist crossing validation tool.

Purpose
-------
Manually review bicyclist crossing detections and measure how many automatic
classifications are correct.

This is intentionally separate from analysis.py.  Use analysis.py for the full
batch pipeline and this file only for validation/tuning on one video segment.

Run from the project root:

    python3 cyclist_crossing_validation_tool.py

Edit only the USER SETTINGS section below.  No command line parser is used.

Outputs
-------
All outputs are written to:

    <repo_root>/data/cyclist_crossing_validation/<video_key>/

Main files:
    auto_confirmed.csv          automatic cyclist crossings after filters
    review_labels.csv           your labels for each automatic detection
    validation_metrics.json     counts and precision style metrics
    confusion_matrix.csv        TP/FP/unsure summary
    snippets/                   short clips for reviewed detections, if video is found

Review keys
-----------
    A = actual bicyclist crossing, correctly classified
    F = fake / wrong classification
    U = unsure / ignore
    N = next
    B = previous
    R = replay current clip
    Q = save and quit

Important limitation
--------------------
This tool measures correctness of automatic detections that the algorithm found.
It can count true positives and false positives.  To measure recall fully, you
must also add missed true crossings manually to missed_crossings.csv or review
the whole video independently.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Optional

import cv2
import polars as pl

import common
from custom_logger import CustomLogger
from logmod import logs
from utils.bicyclist_detect import Algorithm

try:
    from helper_script import Youtube_Helper
    HELPER_IMPORT_ERROR = None
except Exception as exc:
    Youtube_Helper = None  # type: ignore[assignment]
    HELPER_IMPORT_ERROR = exc

try:
    from utils.traffic_light_state import TrafficLightStateDetector
    TRAFFIC_LIGHT_IMPORT_ERROR = None
except Exception as exc:
    TrafficLightStateDetector = None  # type: ignore[assignment]
    TRAFFIC_LIGHT_IMPORT_ERROR = exc


logs(show_level=common.get_configs("logger_level"), show_color=True)
logger = CustomLogger(__name__)

algo = Algorithm()


# ============================================================
# USER SETTINGS
# ============================================================
# Option 1: set the exact detector CSV path.
CSV_PATH: str = ""

# Option 2: leave CSV_PATH empty and set VIDEO_ID / START_SECONDS.
# The tool will search config["data"] recursively for video_id_start_*.csv.
VIDEO_ID: str = "Y11w05u97Iw"
START_SECONDS: Optional[int] = None

# Optional exact source video path.  If empty, the tool searches config["videos"].
VIDEO_PATH: str = ""

# Download source video automatically when it is not already present locally.
# The helper uses the FastAPI HTTP file server paths such as:
#   https://files.mobility-squad.com/v/tue1/files/<video_id>.mp4
DOWNLOAD_VIDEO_IF_MISSING: bool = True
FTP_BASE_URL: str = "https://files.mobility-squad.com/"
FTP_USERNAME: str = ""
FTP_PASSWORD: str = ""
FTP_TOKEN: str = ""
FTP_TIMEOUT_SECONDS: int = 20
FTP_MAX_PAGES: int = 500

# Review all cyclist candidates, not only the algorithm-confirmed crossings.
# This lets the human reviewer decide which candidate tracks are real crossings.
# Confirmed events are marked as valid_after_filters; rejected candidates are
# marked as rejected_by_filters, so later metrics can compare human judgement
# against the algorithm decision.
REVIEW_CONFIRMED_ONLY: bool = False

# Include cyclist candidate IDs that were rejected by the final crossing filters.
# Candidate rows may have approximate start/end frames because the detector only
# returns full details for confirmed events.
REVIEW_CANDIDATES_TOO: bool = True

# Crossing boundaries.  These should match the main pipeline config.
BOUNDARY_LEFT: float = float(common.get_configs("boundary_left"))
BOUNDARY_RIGHT: float = float(common.get_configs("boundary_right"))
MIN_CONFIDENCE: float = float(common.get_configs("min_confidence"))

# Clip/review display settings.
SNIPPET_BEFORE_FRAMES: int = 60
SNIPPET_AFTER_FRAMES: int = 90
PLAYBACK_SPEED: float = 1.0
DRAW_BOUNDARIES: bool = True
DRAW_LABELS: bool = True
EXPORT_SNIPPETS: bool = True

# Traffic light state detection. This uses the pretrained KASTEL model stored
# inside <repo_root>/traffic-light-detection/model_weights/.
# The primary model is YOLOv8x because it is the model that reliably detects
# lights in your current standard Ultralytics environment. The split model loads,
# but in your test video it returned no detections, so it is kept only as a
# fallback.
ENABLE_TRAFFIC_LIGHT_STATE: bool = True
TRAFFIC_LIGHT_MODEL_PATH: str = os.path.join(
    "traffic-light-detection",
    "model_weights",
    "traffic_lights_yolov8x.pt",
)
TRAFFIC_LIGHT_FALLBACK_MODEL_PATH: str = os.path.join(
    "traffic-light-detection",
    "model_weights",
    "traffic_lights_split_yolov8.pt",
)
TRAFFIC_LIGHT_CONFIDENCE: float = 0.25

# This is multi-frame, not single-image. At 30 fps the default samples roughly
# 13 frames from a 2 second window centred on the crossing review frame.
TRAFFIC_LIGHT_WINDOW_BEFORE_FRAMES: int = 30
TRAFFIC_LIGHT_WINDOW_AFTER_FRAMES: int = 30
TRAFFIC_LIGHT_FRAME_STEP: int = 5
TRAFFIC_LIGHT_IMAGE_SIZE: int = 1280

TRAFFIC_LIGHT_UPPER_Y_RATIO: float = 0.70
TRAFFIC_LIGHT_MIN_BOX_AREA_RATIO: float = 0.0
TRAFFIC_LIGHT_MIN_STATE_CONFIDENCE: float = 0.70
TRAFFIC_LIGHT_ANCHOR_FRAME_TOLERANCE: int = 12
TRAFFIC_LIGHT_MAX_ANCHOR_CENTRE_DISTANCE: float = 0.18
TRAFFIC_LIGHT_MIN_ANCHOR_IOU: float = 0.01

# Automatic traffic light mode. The deep learning model runs automatically for
# every event before the review clip is shown and stores its output separately
# as the algorithm suggestion. The human reviewer remains the ground truth: the
# final/trusted traffic light state is unreviewed until you press 1-6.
TRAFFIC_LIGHT_PREANNOTATE_ALL_EVENTS: bool = False
TRAFFIC_LIGHT_AUTO_BEFORE_REVIEW: bool = True
TRAFFIC_LIGHT_AUTO_AFTER_REVIEW: bool = True
TRAFFIC_LIGHT_DETECT_ON_KEY: bool = True
TRAFFIC_LIGHT_RECOMPUTE_AUTOMATIC_STATE: bool = False

# Set False on a server/headless machine. It will ask labels in the terminal
# instead of opening an OpenCV review window.
USE_GUI_REVIEW: bool = True

# Main output folder name under <repo_root>/data.
OUTPUT_DIR_NAME: str = "cyclist_crossing_validation"


# ============================================================
# Data model
# ============================================================
@dataclass
class ValidationEvent:
    source: str
    video_key: str
    source_video_id: str
    csv_path: str
    video_path: str
    cyclist_id: int
    bicycle_id: int
    start_frame: int
    end_frame: int
    review_frame: int
    direction: str
    x_range: float = 0.0
    x_speed: float = 0.0
    road_frames: int = 0
    fps: float = 30.0
    segment_start_seconds: int = 0
    filter_status: str = "valid_after_filters"
    human_label: str = ""
    error_type: str = ""
    snippet_path: str = ""
    notes: str = ""
    traffic_light_auto_attempted: bool = False
    traffic_light_visible: bool = False
    traffic_light_state: str = ""
    traffic_light_raw_class: str = ""
    traffic_light_confidence: float = 0.0
    traffic_light_frame: int = -1
    traffic_light_bbox: str = ""
    traffic_light_detection_count: int = 0
    traffic_light_state_votes: str = ""
    traffic_light_relevance: str = ""
    traffic_light_notes: str = ""
    human_traffic_light_state: str = ""
    human_traffic_light_notes: str = ""
    final_traffic_light_state: str = ""
    traffic_light_review_status: str = ""


# ============================================================
# General helpers
# ============================================================
def repo_root() -> str:
    root = getattr(common, "root_dir", None)
    if root:
        return os.path.abspath(str(root))
    return os.getcwd()


def repo_data_dir() -> str:
    path = os.path.join(repo_root(), "data")
    os.makedirs(path, exist_ok=True)
    return path


def as_path_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(v) for v in value if v is not None and str(v).strip()]
    text = str(value).strip()
    return [text] if text else []


def parse_csv_name(csv_path: str) -> dict:
    stem = os.path.splitext(os.path.basename(csv_path))[0]
    try:
        video_id, start_text, fps_text = stem.rsplit("_", 2)
        start_seconds = int(float(start_text))
        fps = float(fps_text)
    except Exception as exc:
        raise ValueError(
            f"CSV filename must look like video_id_start_fps.csv, got: {os.path.basename(csv_path)}"
        ) from exc

    return {
        "video_key": stem,
        "source_video_id": video_id,
        "segment_start_seconds": start_seconds,
        "fps": fps,
    }


def find_csv_file() -> str:
    if CSV_PATH:
        if not os.path.exists(CSV_PATH):
            raise FileNotFoundError(f"CSV_PATH does not exist: {CSV_PATH}")
        return CSV_PATH

    if not VIDEO_ID:
        raise ValueError("Set either CSV_PATH or VIDEO_ID in the USER SETTINGS section.")

    data_roots = as_path_list(common.get_configs("data"))
    matches: list[str] = []

    for root in data_roots:
        if not os.path.exists(root):
            logger.warning(f"Data folder does not exist: {root}")
            continue

        for current_root, _, files in os.walk(root):
            for name in files:
                if not name.lower().endswith(".csv"):
                    continue
                stem = os.path.splitext(name)[0]
                try:
                    vid, start_text, _fps_text = stem.rsplit("_", 2)
                    start = int(float(start_text))
                except Exception:
                    continue
                if vid != VIDEO_ID:
                    continue
                if START_SECONDS is not None and start != int(START_SECONDS):
                    continue
                matches.append(os.path.join(current_root, name))

    if not matches:
        start_msg = "any start" if START_SECONDS is None else f"start={START_SECONDS}"
        raise FileNotFoundError(f"No detector CSV found for video_id={VIDEO_ID}, {start_msg}.")

    matches.sort()
    if len(matches) > 1:
        logger.warning("Multiple matching CSVs found. Using the first one:")
        for match in matches[:20]:
            logger.warning(f"  {match}")

    return matches[0]


def safe_get_config(key: str, default=""):
    """Read an optional config key without breaking validation for missing keys."""
    try:
        value = common.get_configs(key)
    except Exception:
        return default
    if value is None or value == "":
        return default
    return value


def safe_get_secret(key: str, default=""):
    """Read an optional secret key if the project exposes common.get_secrets."""
    getter = getattr(common, "get_secrets", None)
    if getter is None:
        return default
    try:
        value = getter(key)
    except Exception:
        return default
    if value is None or value == "":
        return default
    return value


def build_ftp_setting(*, env_name: str, config_key: str, default: str = "", secret: bool = False) -> str:
    """Resolve FTP settings from environment, config/secrets, then USER SETTINGS."""
    env_value = os.environ.get(env_name, "").strip()
    if env_value:
        return env_value

    if secret:
        value = safe_get_secret(config_key, "")
    else:
        value = safe_get_config(config_key, "")

    if value is None:
        return str(default or "")

    value_text = str(value).strip()
    if value_text:
        return value_text

    return str(default or "").strip()


def first_video_output_dir() -> str:
    """Return the first configured video directory and create it if needed."""
    roots = as_path_list(common.get_configs("videos"))
    if not roots:
        roots = ["videos"]

    out_dir = roots[0]
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def find_video_file(source_video_id: str) -> str:
    if VIDEO_PATH:
        if not os.path.exists(VIDEO_PATH):
            raise FileNotFoundError(f"VIDEO_PATH does not exist: {VIDEO_PATH}")
        return VIDEO_PATH

    video_root = common.get_configs("videos")
    roots = as_path_list(video_root)
    if not roots:
        roots = ["videos"]

    candidates = [
        f"{source_video_id}.mp4",
        f"{source_video_id}.mov",
        f"{source_video_id}.avi",
        f"{source_video_id}.mkv",
    ]
    candidate_set = {c.lower() for c in candidates}

    for root in roots:
        if not os.path.exists(root):
            continue
        for current_root, _, files in os.walk(root):
            for name in files:
                if name.lower() in candidate_set:
                    return os.path.join(current_root, name)

    return ""


def download_video_if_missing(source_video_id: str) -> str:
    """Find the source video locally or download it from the FTP/HTTP file server."""
    video_path = find_video_file(source_video_id)
    if video_path:
        return video_path

    if not DOWNLOAD_VIDEO_IF_MISSING:
        return ""

    if Youtube_Helper is None:
        logger.warning(f"Video download helper is unavailable: {HELPER_IMPORT_ERROR}")
        return ""

    base_url = build_ftp_setting(
        env_name="FTP_BASE_URL",
        config_key="ftp_base_url",
        default=FTP_BASE_URL,
    )
    username = build_ftp_setting(
        env_name="FTP_USERNAME",
        config_key="ftp_username",
        default=FTP_USERNAME,
        secret=True,
    )
    password = build_ftp_setting(
        env_name="FTP_PASSWORD",
        config_key="ftp_password",
        default=FTP_PASSWORD,
        secret=True,
    )
    token = build_ftp_setting(
        env_name="FTP_TOKEN",
        config_key="ftp_token",
        default=FTP_TOKEN,
    )

    if not base_url:
        logger.warning(
            "Source video was not found locally and no FTP base URL is configured. "
            "Set FTP_BASE_URL in USER SETTINGS, config, or environment."
        )
        return ""

    out_dir = first_video_output_dir()
    logger.info(f"Source video was not found locally. Trying FTP download for {source_video_id}.mp4")
    logger.info(f"Download target folder: {out_dir}")

    try:
        # Important: do not call Youtube_Helper(). Its __init__ loads many
        # unrelated project config keys such as tracking_model, bbox_tracker,
        # mapping, etc. The validation tool only needs download_videos_from_ftp,
        # which does not depend on those instance attributes.
        helper = Youtube_Helper.__new__(Youtube_Helper)
        result = helper.download_videos_from_ftp(
            filename=source_video_id,
            base_url=base_url,
            out_dir=out_dir,
            username=username or None,
            password=password or None,
            token=token or None,
            timeout=int(FTP_TIMEOUT_SECONDS),
            max_pages=int(FTP_MAX_PAGES),
        )
    except Exception as exc:
        logger.warning(f"FTP video download failed for {source_video_id}: {exc}")
        return ""

    if not result:
        logger.warning(f"FTP video download did not find {source_video_id}.mp4")
        return ""

    downloaded_path = str(result[0])
    if os.path.exists(downloaded_path):
        logger.info(f"Downloaded video: {downloaded_path}")
        return downloaded_path

    # Fallback in case helper returned a path that was moved or normalised.
    return find_video_file(source_video_id)

def load_detector_csv(csv_path: str) -> pl.DataFrame:
    required = {"yolo-id", "unique-id", "frame-count", "x-center", "y-center", "width", "height"}
    df = pl.read_csv(csv_path)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

    df = (
        df
        .filter(pl.col("unique-id") != -1)
        .with_columns(
            [
                pl.col("unique-id").cast(pl.Int64, strict=False),
                pl.col("yolo-id").cast(pl.Int64, strict=False),
                pl.col("frame-count").cast(pl.Int64, strict=False),
                pl.col("x-center").cast(pl.Float64, strict=False),
                pl.col("y-center").cast(pl.Float64, strict=False),
                pl.col("width").cast(pl.Float64, strict=False),
                pl.col("height").cast(pl.Float64, strict=False),
            ]
        )
        .filter(pl.col("unique-id").is_not_null())
        .filter(pl.col("yolo-id").is_not_null())
        .filter(pl.col("frame-count").is_not_null())
    )

    if "confidence" in df.columns:
        df = df.filter(pl.col("confidence").cast(pl.Float64, strict=False) >= float(MIN_CONFIDENCE))

    return df


def empty_mapping() -> pl.DataFrame:
    # The algorithm accepts df_mapping for FPS fallback only.  We pass an empty
    # frame because this validation script already parses FPS from the CSV name.
    return pl.DataFrame()


def frame_count_base(df: pl.DataFrame) -> int:
    try:
        min_frame = int(df.select(pl.min("frame-count")).item())
        return 0 if min_frame == 0 else 1
    except Exception:
        return 1



# ============================================================
# Traffic light state helpers
# ============================================================
def _candidate_paths(path_text: str) -> list[str]:
    """Return absolute and repo-relative candidates for one model path."""
    path_text = str(path_text or "").strip()
    if not path_text:
        return []
    if os.path.isabs(path_text):
        return [path_text]
    return [os.path.join(repo_root(), path_text), os.path.abspath(path_text)]


def resolve_traffic_light_model_paths() -> list[str]:
    """Return candidate traffic light model paths in load preference order.

    Preference order:
    1. TRAFFIC_LIGHT_MODEL_PATH environment variable, when set.
    2. TRAFFIC_LIGHT_MODEL_PATH setting, default split YOLOv8 state model.
    3. TRAFFIC_LIGHT_FALLBACK_MODEL_PATH setting, default YOLOv8 X.

    The loader tries every existing candidate. This is important because a .pt
    file can exist but still be corrupted or unreadable by PyTorch.
    """
    env_path = os.environ.get("TRAFFIC_LIGHT_MODEL_PATH", "").strip()
    configured_paths = [env_path, str(TRAFFIC_LIGHT_MODEL_PATH), str(TRAFFIC_LIGHT_FALLBACK_MODEL_PATH)]

    candidates: list[str] = []
    for path_text in configured_paths:
        for candidate in _candidate_paths(path_text):
            if candidate not in candidates:
                candidates.append(candidate)

    existing = [candidate for candidate in candidates if os.path.exists(candidate)]
    return existing if existing else candidates


def resolve_traffic_light_model_path() -> str:
    """Return the first candidate path for backwards compatibility."""
    candidates = resolve_traffic_light_model_paths()
    return candidates[0] if candidates else str(TRAFFIC_LIGHT_MODEL_PATH)


def create_traffic_light_detector():
    """Create the traffic light detector once, or return None if unavailable.

    This function now falls back on load failure, not only when the preferred
    model file is missing. For example, if traffic_lights_yolov8xl.pt exists but
    is corrupted, it will automatically try traffic_lights_yolov8x.pt.
    """
    if not ENABLE_TRAFFIC_LIGHT_STATE:
        return None

    if TrafficLightStateDetector is None:
        logger.warning(f"Traffic light helper is unavailable: {TRAFFIC_LIGHT_IMPORT_ERROR}")
        return None

    candidate_paths = resolve_traffic_light_model_paths()
    if not candidate_paths:
        logger.warning("No traffic light model path was configured.")
        return None

    tried_any_existing = False
    last_error = None

    for model_path in candidate_paths:
        if not os.path.exists(model_path):
            continue

        tried_any_existing = True
        try:
            detector = TrafficLightStateDetector(
                model_path=model_path,
                confidence=float(TRAFFIC_LIGHT_CONFIDENCE),
                upper_y_ratio=float(TRAFFIC_LIGHT_UPPER_Y_RATIO),
                min_box_area_ratio=float(TRAFFIC_LIGHT_MIN_BOX_AREA_RATIO),
                min_state_confidence=float(TRAFFIC_LIGHT_MIN_STATE_CONFIDENCE),
                anchor_frame_tolerance=int(TRAFFIC_LIGHT_ANCHOR_FRAME_TOLERANCE),
                max_anchor_centre_distance=float(TRAFFIC_LIGHT_MAX_ANCHOR_CENTRE_DISTANCE),
                min_anchor_iou=float(TRAFFIC_LIGHT_MIN_ANCHOR_IOU),
                imgsz=int(TRAFFIC_LIGHT_IMAGE_SIZE),
            )
        except Exception as exc:
            last_error = exc
            logger.warning(f"Traffic light model failed to load: {model_path} | {exc}")
            continue

        logger.info(f"Traffic light model: {model_path}")
        return detector

    if not tried_any_existing:
        logger.warning("Traffic light model was not found in any configured path.")
    elif last_error is not None:
        logger.warning(f"All available traffic light models failed to load. Last error: {last_error}")
    else:
        logger.warning("Traffic light model could not be loaded.")

    return None


def read_traffic_light_sample_frames(
    *,
    video_path: str,
    event: ValidationEvent,
    df: pl.DataFrame,
) -> list[tuple[int, object]]:
    """Read sparse frames around the crossing review frame for TL inference.

    This implementation seeks once to the start of the small review window and
    then reads sequentially. It is much faster than random seeking once per
    sampled frame on long dashcam videos.
    """
    if not video_path or not os.path.exists(video_path):
        return []

    base = frame_count_base(df)
    review_frame = int(event.review_frame or round((event.start_frame + event.end_frame) / 2))
    start_csv_frame = max(base, review_frame - int(TRAFFIC_LIGHT_WINDOW_BEFORE_FRAMES))
    end_csv_frame = max(start_csv_frame, review_frame + int(TRAFFIC_LIGHT_WINDOW_AFTER_FRAMES))
    step = max(1, int(TRAFFIC_LIGHT_FRAME_STEP))
    fps = max(1.0, float(event.fps or 30.0))

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.warning(f"Could not open video for traffic light detection: {video_path}")
        return []

    frames: list[tuple[int, object]] = []
    try:
        start_seconds = float(event.segment_start_seconds) + ((int(start_csv_frame) - int(base)) / fps)
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, start_seconds * 1000.0))

        for csv_frame in range(int(start_csv_frame), int(end_csv_frame) + 1):
            ok, frame = cap.read()
            if not ok:
                break
            if (int(csv_frame) - int(start_csv_frame)) % step == 0:
                frames.append((int(csv_frame), frame))
    finally:
        cap.release()

    return frames

def should_compute_traffic_light_state(event: ValidationEvent, *, force: bool = False) -> bool:
    """Return True when automatic TL state should be computed or recomputed."""
    if not ENABLE_TRAFFIC_LIGHT_STATE:
        return False
    if bool(force):
        return True
    if bool(TRAFFIC_LIGHT_RECOMPUTE_AUTOMATIC_STATE):
        return True
    if bool(getattr(event, "traffic_light_auto_attempted", False)):
        return False
    if event.traffic_light_detection_count:
        return False
    if event.traffic_light_state and event.traffic_light_state != "unknown":
        return False
    return True


def apply_traffic_light_state_to_event(
    event: ValidationEvent,
    *,
    detector,
    video_path: str,
    df: pl.DataFrame,
    event_index: int | None = None,
    total_events: int | None = None,
    force: bool = False,
) -> None:
    """Compute and attach automatic traffic light state for one event."""
    if detector is None or not should_compute_traffic_light_state(event, force=force):
        return

    if not video_path or not os.path.exists(video_path):
        return

    event.traffic_light_auto_attempted = True
    frames = read_traffic_light_sample_frames(video_path=video_path, event=event, df=df)
    if not frames:
        event.traffic_light_state = "unknown"
        event.traffic_light_notes = "automatic traffic light detection attempted, but no video frames were available"
        logger.warning(f"No frames available for traffic light detection on cyclist_id={event.cyclist_id}.")
        return

    try:
        summary = detector.summarise_frames(frames, target_frame=int(event.review_frame))
    except Exception as exc:
        event.traffic_light_state = "unknown"
        event.traffic_light_notes = f"automatic traffic light detection attempted, but failed: {exc}"
        logger.warning(f"Traffic light detection failed for cyclist_id={event.cyclist_id}: {exc}")
        return

    for key, value in summary.items():
        if hasattr(event, key):
            setattr(event, key, value)
    event.traffic_light_auto_attempted = True

    prefix = "Traffic light"
    if event_index is not None and total_events is not None:
        prefix = f"Traffic light {event_index}/{total_events}"

    logger.info(
        "{} cyclist_id={} state={} raw={} confidence={:.2f} detections={} votes={}",
        prefix,
        event.cyclist_id,
        event.traffic_light_state or "unknown",
        event.traffic_light_raw_class or "none",
        float(event.traffic_light_confidence or 0.0),
        int(event.traffic_light_detection_count or 0),
        event.traffic_light_state_votes or "{}",
    )


def annotate_traffic_light_states(events: list[ValidationEvent], *, video_path: str, df: pl.DataFrame, detector=None) -> None:
    """Attach traffic light state summaries to all validation events."""
    if not ENABLE_TRAFFIC_LIGHT_STATE:
        return

    if not video_path or not os.path.exists(video_path):
        logger.warning("Traffic light state detection skipped because source video is unavailable.")
        return

    if detector is None:
        detector = create_traffic_light_detector()
    if detector is None:
        return

    logger.info(f"Annotating traffic light state for {len(events)} crossing events.")
    for idx, event in enumerate(events, start=1):
        apply_traffic_light_state_to_event(
            event,
            detector=detector,
            video_path=video_path,
            df=df,
            event_index=idx,
            total_events=len(events),
        )


# ============================================================
# Detector wrappers
# ============================================================
def run_detector(
    *,
    df: pl.DataFrame,
    csv_info: dict,
    csv_path: str,
    video_path: str,
) -> tuple[list[ValidationEvent], list[int]]:
    cyclist_ids, candidate_ids, events_df = algo.cyclist_crossing(
        dataframe=df,
        video_id=csv_info["video_key"],
        df_mapping=empty_mapping(),
        min_x=BOUNDARY_LEFT,
        max_x=BOUNDARY_RIGHT,
        fps=float(csv_info["fps"]),
        return_details=True,
    )

    rows: list[ValidationEvent] = []
    if events_df.height > 0:
        for row in events_df.iter_rows(named=True):
            start_frame = int(row.get("start_frame", 0) or 0)
            end_frame = int(row.get("end_frame", start_frame) or start_frame)
            rows.append(
                ValidationEvent(
                    source="algorithm_confirmed",
                    video_key=str(csv_info["video_key"]),
                    source_video_id=str(csv_info["source_video_id"]),
                    csv_path=str(csv_path),
                    video_path=str(video_path),
                    cyclist_id=int(row.get("cyclist_id", -1) or -1),
                    bicycle_id=int(row.get("bicycle_id", -1) or -1),
                    start_frame=start_frame,
                    end_frame=end_frame,
                    review_frame=int(round((start_frame + end_frame) / 2)),
                    direction=str(row.get("direction", "")),
                    x_range=float(row.get("x_range", 0.0) or 0.0),
                    x_speed=float(row.get("x_speed", 0.0) or 0.0),
                    road_frames=int(row.get("road_frames", 0) or 0),
                    fps=float(csv_info["fps"]),
                    segment_start_seconds=int(csv_info["segment_start_seconds"]),
                    filter_status="valid_after_filters",
                )
            )

    rows.sort(key=lambda item: (item.start_frame, item.cyclist_id))
    return rows, [int(x) for x in candidate_ids]


def approximate_event_for_candidate(
    *,
    df: pl.DataFrame,
    cyclist_id: int,
    csv_info: dict,
    csv_path: str,
    video_path: str,
    confirmed_ids: set[int],
) -> Optional[ValidationEvent]:
    """Build an approximate review event for a candidate id.

    The detector returns full start/end details only for confirmed events, so for
    rejected candidate review we use the full person-track frame range.
    """
    track = (
        df.filter((pl.col("yolo-id") == 0) & (pl.col("unique-id") == int(cyclist_id)))
        .sort("frame-count")
    )
    if track.height == 0:
        return None

    frames = track.get_column("frame-count").to_list()
    start_frame = int(min(frames))
    end_frame = int(max(frames))

    xs = track.get_column("x-center").to_list()
    direction = ""
    if xs:
        direction = "left_to_right" if float(xs[-1]) > float(xs[0]) else "right_to_left"

    return ValidationEvent(
        source="algorithm_candidate",
        video_key=str(csv_info["video_key"]),
        source_video_id=str(csv_info["source_video_id"]),
        csv_path=str(csv_path),
        video_path=str(video_path),
        cyclist_id=int(cyclist_id),
        bicycle_id=-1,
        start_frame=start_frame,
        end_frame=end_frame,
        review_frame=int(round((start_frame + end_frame) / 2)),
        direction=direction,
        x_range=float(max(xs) - min(xs)) if xs else 0.0,
        x_speed=0.0,
        road_frames=0,
        fps=float(csv_info["fps"]),
        segment_start_seconds=int(csv_info["segment_start_seconds"]),
        filter_status="valid_after_filters" if int(cyclist_id) in confirmed_ids else "rejected_by_filters",
    )


# ============================================================
# CSV/JSON output helpers
# ============================================================
def event_key(event: ValidationEvent) -> tuple[str, str, str]:
    return (str(event.source), str(event.cyclist_id), str(event.start_frame))


def trusted_traffic_light_state(event: ValidationEvent) -> str:
    """Return only the human verified traffic light state.

    The automatic traffic light model is stored separately in traffic_light_state.
    It is never promoted to the final/trusted state inside the validation tool,
    because the human reviewer is the ground truth for validation.
    """
    human = str(event.human_traffic_light_state or "").strip()
    return human if human else "unreviewed"


def refresh_traffic_light_review_fields(events: list[ValidationEvent]) -> None:
    """Keep human decision columns consistent before every CSV write."""
    for event in events:
        human = str(event.human_traffic_light_state or "").strip()
        event.final_traffic_light_state = human if human else "unreviewed"
        if human:
            event.traffic_light_review_status = "human_reviewed"
        elif bool(getattr(event, "traffic_light_auto_attempted", False)):
            event.traffic_light_review_status = "pending_human_review_auto_available"
        else:
            event.traffic_light_review_status = "pending_human_review_auto_not_run"


def write_events_csv(path: str, events: list[ValidationEvent]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    refresh_traffic_light_review_fields(events)
    rows = [asdict(event) for event in events]
    if rows:
        pl.DataFrame(rows).write_csv(path)
    else:
        pl.DataFrame(
            schema={
                "source": pl.Utf8,
                "video_key": pl.Utf8,
                "source_video_id": pl.Utf8,
                "csv_path": pl.Utf8,
                "video_path": pl.Utf8,
                "cyclist_id": pl.Int64,
                "bicycle_id": pl.Int64,
                "start_frame": pl.Int64,
                "end_frame": pl.Int64,
                "review_frame": pl.Int64,
                "direction": pl.Utf8,
                "x_range": pl.Float64,
                "x_speed": pl.Float64,
                "road_frames": pl.Int64,
                "fps": pl.Float64,
                "segment_start_seconds": pl.Int64,
                "filter_status": pl.Utf8,
                "human_label": pl.Utf8,
                "error_type": pl.Utf8,
                "snippet_path": pl.Utf8,
                "notes": pl.Utf8,
                "traffic_light_auto_attempted": pl.Boolean,
                "traffic_light_visible": pl.Boolean,
                "traffic_light_state": pl.Utf8,
                "traffic_light_raw_class": pl.Utf8,
                "traffic_light_confidence": pl.Float64,
                "traffic_light_frame": pl.Int64,
                "traffic_light_bbox": pl.Utf8,
                "traffic_light_detection_count": pl.Int64,
                "traffic_light_state_votes": pl.Utf8,
                "traffic_light_relevance": pl.Utf8,
                "traffic_light_notes": pl.Utf8,
                "human_traffic_light_state": pl.Utf8,
                "human_traffic_light_notes": pl.Utf8,
                "final_traffic_light_state": pl.Utf8,
                "traffic_light_review_status": pl.Utf8,
            }
        ).write_csv(path)


def read_existing_labels(path: str) -> dict[tuple[str, str, str], dict]:
    if not os.path.exists(path):
        return {}
    try:
        df = pl.read_csv(path)
    except Exception:
        return {}

    out = {}
    for row in df.iter_rows(named=True):
        key = (str(row.get("source", "")), str(row.get("cyclist_id", "")), str(row.get("start_frame", "")))
        out[key] = row
    return out


def merge_existing_labels(events: list[ValidationEvent], labels_path: str) -> list[ValidationEvent]:
    existing = read_existing_labels(labels_path)
    for event in events:
        old = existing.get(event_key(event))
        if not old:
            continue
        event.human_label = str(old.get("human_label") or "")
        event.error_type = str(old.get("error_type") or "")
        event.snippet_path = str(old.get("snippet_path") or "")
        event.notes = str(old.get("notes") or "")
        event.human_traffic_light_state = str(old.get("human_traffic_light_state") or "")
        event.human_traffic_light_notes = str(old.get("human_traffic_light_notes") or "")
        event.final_traffic_light_state = str(old.get("final_traffic_light_state") or "")
        event.traffic_light_review_status = str(old.get("traffic_light_review_status") or "")
    return events


def write_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


# ============================================================
# Review / drawing helpers
# ============================================================
def normalised_bbox_to_pixels(row: dict, frame_width: int, frame_height: int) -> tuple[int, int, int, int]:
    xc = float(row["x-center"]) * frame_width
    yc = float(row["y-center"]) * frame_height
    w = float(row["width"]) * frame_width
    h = float(row["height"]) * frame_height
    x1 = max(0, min(frame_width - 1, int(round(xc - w / 2))))
    y1 = max(0, min(frame_height - 1, int(round(yc - h / 2))))
    x2 = max(0, min(frame_width - 1, int(round(xc + w / 2))))
    y2 = max(0, min(frame_height - 1, int(round(yc + h / 2))))
    return x1, y1, x2, y2


def frame_rows_by_count(df: pl.DataFrame) -> dict[int, list[dict]]:
    wanted = df.filter(pl.col("yolo-id").is_in([0, 1]))
    by_frame: dict[int, list[dict]] = {}
    for row in wanted.iter_rows(named=True):
        by_frame.setdefault(int(row["frame-count"]), []).append(row)
    return by_frame



def draw_traffic_light_zoom_panel(frame, event: ValidationEvent) -> None:
    """Draw an enlarged traffic light crop in the top right of the review frame.

    The crop uses the model selected bbox. It is a visual aid for manual review;
    the operator should still judge the actual visible traffic light state.
    """
    if not ENABLE_TRAFFIC_LIGHT_STATE or not str(event.traffic_light_bbox or "").strip():
        return

    try:
        bbox = json.loads(event.traffic_light_bbox or "{}")
    except Exception:
        return

    if not bbox:
        return

    h, w = frame.shape[:2]
    try:
        x1 = int(round(float(bbox.get("x1", 0))))
        y1 = int(round(float(bbox.get("y1", 0))))
        x2 = int(round(float(bbox.get("x2", 0))))
        y2 = int(round(float(bbox.get("y2", 0))))
    except Exception:
        return

    if x2 <= x1 or y2 <= y1:
        return

    # Add context around tiny traffic light boxes.
    pad_x = max(12, int(round((x2 - x1) * 2.0)))
    pad_y = max(12, int(round((y2 - y1) * 2.0)))
    cx1 = max(0, x1 - pad_x)
    cy1 = max(0, y1 - pad_y)
    cx2 = min(w, x2 + pad_x)
    cy2 = min(h, y2 + pad_y)
    if cx2 <= cx1 or cy2 <= cy1:
        return

    crop = frame[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        return

    panel_w = min(300, max(180, int(w * 0.24)))
    panel_h = min(220, max(140, int(h * 0.22)))
    try:
        zoom = cv2.resize(crop, (panel_w, panel_h), interpolation=cv2.INTER_CUBIC)
    except Exception:
        return

    margin = 12
    px2 = w - margin
    py1 = margin
    px1 = max(0, px2 - panel_w)
    py2 = min(h, py1 + panel_h)
    if py2 - py1 != panel_h:
        return

    frame[py1:py2, px1:px2] = zoom
    cv2.rectangle(frame, (px1, py1), (px2, py2), (255, 255, 255), 2)
    label = f"TL zoom | auto={event.traffic_light_state or 'unknown'}"
    if event.human_traffic_light_state:
        label += f" | human={event.human_traffic_light_state}"
    cv2.putText(frame, label, (px1, max(0, py1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2)

    # Draw the selected bbox inside the zoom panel. Transform coordinates from
    # full image crop coordinates into the resized panel coordinate space.
    sx = panel_w / max(float(cx2 - cx1), 1.0)
    sy = panel_h / max(float(cy2 - cy1), 1.0)
    zx1 = px1 + int(round((x1 - cx1) * sx))
    zy1 = py1 + int(round((y1 - cy1) * sy))
    zx2 = px1 + int(round((x2 - cx1) * sx))
    zy2 = py1 + int(round((y2 - cy1) * sy))
    cv2.rectangle(frame, (zx1, zy1), (zx2, zy2), (255, 255, 255), 2)


def draw_frame(
    *,
    frame,
    csv_frame: int,
    event: ValidationEvent,
    rows_by_frame: dict[int, list[dict]],
) -> None:
    h, w = frame.shape[:2]

    if DRAW_BOUNDARIES:
        left_x = int(round(float(BOUNDARY_LEFT) * w))
        right_x = int(round(float(BOUNDARY_RIGHT) * w))
        cv2.line(frame, (left_x, 0), (left_x, h - 1), (255, 255, 255), 2)
        cv2.line(frame, (right_x, 0), (right_x, h - 1), (255, 255, 255), 2)

    for row in rows_by_frame.get(int(csv_frame), []):
        yolo_id = int(row["yolo-id"])
        uid = int(row["unique-id"])
        draw = False
        colour = (150, 150, 150)
        label = ""

        if yolo_id == 0 and uid == int(event.cyclist_id):
            draw = True
            colour = (0, 255, 255)
            label = f"cyclist:{uid}"
        elif yolo_id == 1 and (uid == int(event.bicycle_id) or int(event.bicycle_id) < 0):
            draw = True
            colour = (255, 0, 0)
            label = f"bicycle:{uid}"

        if not draw:
            continue

        x1, y1, x2, y2 = normalised_bbox_to_pixels(row, w, h)
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
        if DRAW_LABELS:
            cv2.putText(frame, label, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2)

    status = f"{event.source} id={event.cyclist_id} {event.direction} frames={event.start_frame}-{event.end_frame}"
    cv2.putText(frame, status, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    if ENABLE_TRAFFIC_LIGHT_STATE:
        if bool(event.traffic_light_visible):
            tl_status = (
                f"TL auto: {event.traffic_light_state} "
                f"({float(event.traffic_light_confidence):.2f}) {event.traffic_light_raw_class}"
            )
        else:
            tl_status = "TL auto: not detected"

        if event.human_traffic_light_state:
            tl_status += f" | human: {event.human_traffic_light_state}"

        cv2.putText(frame, tl_status, (12, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2)

        # Draw the model-selected traffic light box when available.  The box is
        # from the best sampled frame, so treat it as a visual cue rather than
        # exact ground truth for every displayed frame.
        try:
            bbox = json.loads(event.traffic_light_bbox or "{}")
            if bbox:
                x1 = int(round(float(bbox.get("x1", 0))))
                y1 = int(round(float(bbox.get("y1", 0))))
                x2 = int(round(float(bbox.get("x2", 0))))
                y2 = int(round(float(bbox.get("y2", 0))))
                if x2 > x1 and y2 > y1:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
                    cv2.putText(frame, "TL model", (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 2)
        except Exception:
            pass

        draw_traffic_light_zoom_panel(frame, event)

    cv2.putText(frame, "A actual | F fake | U unsure | 1 red | 2 yellow | 3 green | 4 off | 5 unknown | 6 no TL | 0 clear TL | N next | B back | R replay | Q quit", (12, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 2)


def read_clip_frames(
    *,
    video_path: str,
    event: ValidationEvent,
    df: pl.DataFrame,
) -> list[tuple[int, object]]:
    if not video_path or not os.path.exists(video_path):
        return []

    base = frame_count_base(df)
    start_csv_frame = max(base, int(event.start_frame) - int(SNIPPET_BEFORE_FRAMES))
    end_csv_frame = int(event.end_frame) + int(SNIPPET_AFTER_FRAMES)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.warning(f"Could not open video: {video_path}")
        return []

    try:
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, float(event.segment_start_seconds) * 1000.0))
        frames_to_skip = max(0, int(start_csv_frame) - int(base))
        for _ in range(frames_to_skip):
            ok, _frame = cap.read()
            if not ok:
                return []

        frames = []
        for csv_frame in range(int(start_csv_frame), int(end_csv_frame) + 1):
            ok, frame = cap.read()
            if not ok:
                break
            frames.append((csv_frame, frame))
        return frames
    finally:
        cap.release()


def export_snippet(
    *,
    output_path: str,
    frames: list[tuple[int, object]],
    event: ValidationEvent,
    rows_by_frame: dict[int, list[dict]],
) -> str:
    if not frames:
        return ""

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    first_frame = frames[0][1]
    height, width = first_frame.shape[:2]
    writer_fps = max(1.0, float(event.fps))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
    out = cv2.VideoWriter(output_path, fourcc, writer_fps, (width, height))
    if not out.isOpened():
        return ""

    try:
        for csv_frame, frame in frames:
            draw_frame(frame=frame, csv_frame=csv_frame, event=event, rows_by_frame=rows_by_frame)
            out.write(frame)
    finally:
        out.release()

    return output_path


class EventReviewer:
    def __init__(
        self,
        *,
        video_path: str,
        df: pl.DataFrame,
        events: list[ValidationEvent],
        labels_csv: str,
        snippets_dir: str,
        traffic_light_detector=None,
    ):
        self.video_path = video_path
        self.df = df
        self.events = events
        self.labels_csv = labels_csv
        self.snippets_dir = snippets_dir
        self.rows_by_frame = frame_rows_by_count(df)
        self.traffic_light_detector = traffic_light_detector

    def play_event(self, event: ValidationEvent) -> str:
        frames = read_clip_frames(video_path=self.video_path, event=event, df=self.df)
        if not frames or not USE_GUI_REVIEW:
            if not frames:
                logger.warning("No video frames available. Label in terminal: A/F/U/N/B/Q")
            return self.terminal_label()

        delay_ms = max(1, int(round(1000.0 / max(float(event.fps) * float(PLAYBACK_SPEED), 1.0))))
        while True:
            for csv_frame, frame in frames:
                draw_frame(frame=frame, csv_frame=csv_frame, event=event, rows_by_frame=self.rows_by_frame)
                cv2.imshow("cyclist crossing validation", frame)
                key = cv2.waitKey(delay_ms) & 0xFF
                if key in [ord("a"), ord("A")]:
                    return "actual"
                if key in [ord("f"), ord("F")]:
                    return "fake"
                if key in [ord("u"), ord("U")]:
                    return "unsure"
                if key in [ord("n"), ord("N")]:
                    return "next"
                if key in [ord("b"), ord("B")]:
                    return "back"
                if key in [ord("q"), ord("Q")]:
                    return "quit"
                if key in [ord("r"), ord("R")]:
                    break
                if key in [ord("t"), ord("T")] and TRAFFIC_LIGHT_DETECT_ON_KEY:
                    self.compute_automatic_traffic_light_state(event)
                    continue
                if key == ord("1"):
                    self.set_human_traffic_light_state(event, "red")
                    continue
                if key == ord("2"):
                    self.set_human_traffic_light_state(event, "yellow")
                    continue
                if key == ord("3"):
                    self.set_human_traffic_light_state(event, "green")
                    continue
                if key == ord("4"):
                    self.set_human_traffic_light_state(event, "off")
                    continue
                if key == ord("5"):
                    self.set_human_traffic_light_state(event, "unknown")
                    continue
                if key == ord("6"):
                    self.set_human_traffic_light_state(event, "not_visible")
                    continue
                if key == ord("0"):
                    event.human_traffic_light_state = ""
                    event.human_traffic_light_notes = "manual traffic light label cleared during visual review"
                    refresh_traffic_light_review_fields(self.events)
                    self.save_progress()
                    logger.info("Cleared human traffic light state for cyclist_id={}", event.cyclist_id)
                    continue
            else:
                return "next"

    def compute_automatic_traffic_light_state(self, event: ValidationEvent) -> None:
        """Run automatic traffic light detection for the current event on demand."""
        if not ENABLE_TRAFFIC_LIGHT_STATE:
            logger.info("Automatic traffic light detection is disabled.")
            return
        if self.traffic_light_detector is None:
            self.traffic_light_detector = create_traffic_light_detector()
        if self.traffic_light_detector is None:
            logger.warning("Automatic traffic light detection is unavailable.")
            return

        apply_traffic_light_state_to_event(
            event,
            detector=self.traffic_light_detector,
            video_path=self.video_path,
            df=self.df,
            force=True,
        )
        refresh_traffic_light_review_fields(self.events)
        self.save_progress()
        logger.info(
            "Automatic traffic light state for cyclist_id={}: algorithm={}, human={}",
            event.cyclist_id,
            event.traffic_light_state or "unknown",
            event.human_traffic_light_state or "unreviewed",
        )

    def set_human_traffic_light_state(self, event: ValidationEvent, state: str) -> None:
        """Set and save the manually verified traffic light state immediately."""
        event.human_traffic_light_state = str(state)
        event.human_traffic_light_notes = "manual traffic light state set during visual review"
        refresh_traffic_light_review_fields(self.events)
        self.save_progress()
        logger.info(
            "Human traffic light state set for cyclist_id={}: {}",
            event.cyclist_id,
            event.human_traffic_light_state,
        )

    @staticmethod
    def terminal_label() -> str:
        value = input("Label [A actual / F fake / U unsure / N next / B back / Q quit]: ").strip().lower()
        if value == "a":
            return "actual"
        if value == "f":
            return "fake"
        if value == "u":
            return "unsure"
        if value == "b":
            return "back"
        if value == "q":
            return "quit"
        return "next"

    def save_progress(self) -> None:
        write_events_csv(self.labels_csv, self.events)

    def run(self) -> list[ValidationEvent]:
        if not self.events:
            return self.events

        index = 0
        while 0 <= index < len(self.events):
            event = self.events[index]

            if TRAFFIC_LIGHT_AUTO_BEFORE_REVIEW:
                apply_traffic_light_state_to_event(
                    event,
                    detector=self.traffic_light_detector,
                    video_path=self.video_path,
                    df=self.df,
                    event_index=index + 1,
                    total_events=len(self.events),
                )
                refresh_traffic_light_review_fields(self.events)
                self.save_progress()
            else:
                refresh_traffic_light_review_fields(self.events)

            logger.info(
                f"Review {index + 1}/{len(self.events)}: source={event.source}, "
                f"filter_status={event.filter_status}, cyclist_id={event.cyclist_id}, "
                f"frames={event.start_frame}-{event.end_frame}, current_label={event.human_label or 'none'}, "
                f"tl_algorithm={event.traffic_light_state or 'unknown'}, tl_human={event.human_traffic_light_state or 'unreviewed'}, "
                f"human_decision={trusted_traffic_light_state(event)}"
            )

            action = self.play_event(event)
            if action == "quit":
                self.save_progress()
                break
            if action == "back":
                index = max(0, index - 1)
                continue
            if action == "next":
                index += 1
                continue

            event.human_label = action
            if action == "actual":
                event.error_type = "TP" if event.filter_status == "valid_after_filters" else "FN_candidate_rejected"
            elif action == "fake":
                event.error_type = "FP" if event.filter_status == "valid_after_filters" else "TN_candidate_rejected"
            elif action == "unsure":
                event.error_type = "UNSURE"

            if EXPORT_SNIPPETS and self.video_path:
                frames = read_clip_frames(video_path=self.video_path, event=event, df=self.df)
                snippet_name = f"{event.video_key}_cyclist_{event.cyclist_id}_frames_{event.start_frame}_{event.end_frame}.mp4"
                snippet_path = os.path.join(self.snippets_dir, snippet_name)
                saved = export_snippet(output_path=snippet_path, frames=frames, event=event, rows_by_frame=self.rows_by_frame)
                if saved:
                    event.snippet_path = saved

            self.save_progress()
            index += 1

        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

        self.save_progress()
        return self.events


# ============================================================
# Metrics
# ============================================================
def calculate_metrics(events: list[ValidationEvent], missed_count: int = 0) -> dict:
    labelled = [e for e in events if e.human_label in {"actual", "fake", "unsure"}]
    tp = sum(1 for e in labelled if e.error_type == "TP")
    fp = sum(1 for e in labelled if e.error_type == "FP")
    unsure = sum(1 for e in labelled if e.error_type == "UNSURE")
    fn_rejected = sum(1 for e in labelled if e.error_type == "FN_candidate_rejected")
    tn_rejected = sum(1 for e in labelled if e.error_type == "TN_candidate_rejected")
    fn_total = fn_rejected + int(missed_count)

    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn_total) if (tp + fn_total) else None

    return {
        "reviewed_events": len(labelled),
        "total_events_in_review_table": len(events),
        "true_positives": tp,
        "false_positives": fp,
        "unsure": unsure,
        "false_negative_candidates_rejected": fn_rejected,
        "true_negative_candidates_rejected": tn_rejected,
        "missed_crossings_manually_added": int(missed_count),
        "false_negatives_total": fn_total,
        "precision_on_reviewed_confirmed_detections": precision,
        "recall_if_missed_crossings_are_complete": recall,
        "note": (
            "Precision is reliable for reviewed automatic positives. Recall is only reliable if "
            "you also add every missed true crossing to missed_crossings.csv or review the full video."
        ),
    }


def calculate_traffic_light_metrics(events: list[ValidationEvent]) -> dict:
    """Compare automatic traffic light state against manual review labels.

    Manual labels are the ground truth for this validation tool. The automatic
    detector is counted as correct only when it exactly matches the human state.
    Unknown and not_visible human labels are reported but excluded from the main
    visible-state accuracy because they are not colour states.
    """
    reviewed = [e for e in events if str(e.human_traffic_light_state or "").strip()]
    visible_states = {"red", "yellow", "green", "red_yellow", "off"}
    comparable = [e for e in reviewed if str(e.human_traffic_light_state or "").strip() in visible_states]

    correct = 0
    wrong = 0
    auto_unknown = 0
    no_auto_detection = 0
    rows = []

    for event in reviewed:
        human = str(event.human_traffic_light_state or "").strip() or "unknown"
        auto = str(event.traffic_light_state or "").strip() or "unknown"
        if auto == "":
            auto = "unknown"

        rows.append(
            {
                "cyclist_id": int(event.cyclist_id),
                "start_frame": int(event.start_frame),
                "end_frame": int(event.end_frame),
                "human_traffic_light_state": human,
                "automatic_traffic_light_state": auto,
                "traffic_light_confidence": float(event.traffic_light_confidence or 0.0),
                "traffic_light_detection_count": int(event.traffic_light_detection_count or 0),
                "is_colour_state_comparable": human in visible_states,
                "is_automatic_correct": bool(human in visible_states and auto == human),
            }
        )

        if human not in visible_states:
            continue

        if int(event.traffic_light_detection_count or 0) <= 0:
            no_auto_detection += 1
        if auto in {"", "unknown"}:
            auto_unknown += 1
        elif auto == human:
            correct += 1
        else:
            wrong += 1

    auto_known = correct + wrong
    comparable_count = len(comparable)

    return {
        "manual_traffic_light_reviewed_events": len(reviewed),
        "manual_visible_colour_events": comparable_count,
        "automatic_correct_visible_colour_events": correct,
        "automatic_wrong_visible_colour_events": wrong,
        "automatic_unknown_visible_colour_events": auto_unknown,
        "automatic_no_detection_visible_colour_events": no_auto_detection,
        "automatic_known_visible_colour_events": auto_known,
        "automatic_accuracy_when_known": correct / auto_known if auto_known else None,
        "automatic_coverage_on_manual_visible_colours": auto_known / comparable_count if comparable_count else None,
        "automatic_accuracy_counting_unknown_as_wrong": correct / comparable_count if comparable_count else None,
        "note": (
            "Manual traffic light labels are treated as ground truth. Main colour "
            "accuracy excludes human labels unknown and not_visible. Use "
            "automatic_accuracy_when_known together with coverage to judge quality."
        ),
        "per_event": rows,
    }


def write_traffic_light_confusion_matrix(path: str, events: list[ValidationEvent]) -> None:
    """Write automatic-vs-human traffic light confusion counts."""
    rows_by_pair: dict[tuple[str, str], int] = {}
    for event in events:
        human = str(event.human_traffic_light_state or "").strip()
        if not human:
            continue
        auto = str(event.traffic_light_state or "").strip() or "unknown"
        key = (auto, human)
        rows_by_pair[key] = rows_by_pair.get(key, 0) + 1

    rows = [
        {"automatic": auto, "human": human, "count": count}
        for (auto, human), count in sorted(rows_by_pair.items())
    ]
    if rows:
        pl.DataFrame(rows).write_csv(path)
    else:
        pl.DataFrame(schema={"automatic": pl.Utf8, "human": pl.Utf8, "count": pl.Int64}).write_csv(path)


def write_confusion_matrix(path: str, metrics: dict) -> None:
    rows = [
        {"predicted": "crossing", "human": "actual", "count": int(metrics["true_positives"]), "name": "TP"},
        {"predicted": "crossing", "human": "fake", "count": int(metrics["false_positives"]), "name": "FP"},
        {
            "predicted": "not_crossing_or_missing",
            "human": "actual",
            "count": int(metrics["false_negatives_total"]),
            "name": "FN",
        },
        {
            "predicted": "not_crossing_candidate_rejected",
            "human": "fake",
            "count": int(metrics["true_negative_candidates_rejected"]),
            "name": "TN_candidate",
        },
        {"predicted": "unsure", "human": "unsure", "count": int(metrics["unsure"]), "name": "UNSURE"},
    ]
    pl.DataFrame(rows).write_csv(path)


def read_missed_crossings_count(path: str) -> int:
    """Count manually entered missed crossings, if the file exists.

    Expected optional columns: cyclist_id,start_frame,end_frame,notes
    Any non-empty row counts as one missed crossing.
    """
    if not os.path.exists(path):
        pl.DataFrame(
            schema={
                "cyclist_id": pl.Utf8,
                "start_frame": pl.Utf8,
                "end_frame": pl.Utf8,
                "notes": pl.Utf8,
            }
        ).write_csv(path)
        return 0

    try:
        df = pl.read_csv(path)
    except Exception:
        return 0
    return int(df.height)


# ============================================================
# Main workflow
# ============================================================
def main() -> None:
    csv_path = find_csv_file()
    csv_info = parse_csv_name(csv_path)
    video_path = download_video_if_missing(csv_info["source_video_id"])

    logger.info(f"CSV: {csv_path}")
    if video_path:
        logger.info(f"Video: {video_path}")
    else:
        logger.warning("Source video was not found. You can still write/read labels, but interactive clip review is limited.")

    df = load_detector_csv(csv_path)

    events, candidate_ids = run_detector(df=df, csv_info=csv_info, csv_path=csv_path, video_path=video_path)
    confirmed_ids = {event.cyclist_id for event in events}

    if REVIEW_CANDIDATES_TOO:
        for candidate_id in candidate_ids:
            if REVIEW_CONFIRMED_ONLY and int(candidate_id) not in confirmed_ids:
                continue
            if int(candidate_id) in confirmed_ids:
                continue
            candidate_event = approximate_event_for_candidate(
                df=df,
                cyclist_id=int(candidate_id),
                csv_info=csv_info,
                csv_path=csv_path,
                video_path=video_path,
                confirmed_ids=confirmed_ids,
            )
            if candidate_event is not None:
                events.append(candidate_event)

    events.sort(key=lambda item: (item.start_frame, item.cyclist_id, item.source))
    for event in events:
        event.video_path = str(video_path)

    output_root = os.path.join(repo_data_dir(), OUTPUT_DIR_NAME, csv_info["video_key"])
    snippets_dir = os.path.join(output_root, "snippets")
    os.makedirs(output_root, exist_ok=True)
    os.makedirs(snippets_dir, exist_ok=True)

    traffic_light_detector = None
    if TRAFFIC_LIGHT_PREANNOTATE_ALL_EVENTS or TRAFFIC_LIGHT_AUTO_BEFORE_REVIEW:
        traffic_light_detector = create_traffic_light_detector()
    if TRAFFIC_LIGHT_PREANNOTATE_ALL_EVENTS:
        annotate_traffic_light_states(events, video_path=video_path, df=df, detector=traffic_light_detector)

    auto_csv = os.path.join(output_root, "auto_confirmed.csv")
    labels_csv = os.path.join(output_root, "review_labels.csv")
    metrics_json = os.path.join(output_root, "validation_metrics.json")
    matrix_csv = os.path.join(output_root, "confusion_matrix.csv")
    traffic_light_metrics_json = os.path.join(output_root, "traffic_light_validation_metrics.json")
    traffic_light_matrix_csv = os.path.join(output_root, "traffic_light_confusion_matrix.csv")
    missed_csv = os.path.join(output_root, "missed_crossings.csv")
    summary_json = os.path.join(output_root, "run_summary.json")

    write_events_csv(auto_csv, events)
    events = merge_existing_labels(events, labels_csv)

    rejected_candidate_count = sum(1 for event in events if event.filter_status == "rejected_by_filters")
    logger.info(f"Algorithm-confirmed cyclist crossings: {len(confirmed_ids)}")
    logger.info(f"Algorithm candidate cyclist tracks: {len(candidate_ids)}")
    logger.info(f"Rejected candidate tracks added for human review: {rejected_candidate_count}")
    logger.info(f"Total cyclist tracks/events shown to human reviewer: {len(events)}")
    logger.info("Review controls: A actual, F fake, U unsure, N next, B previous, R replay, Q quit")
    logger.info("Traffic light controls: 1 red, 2 yellow, 3 green, 4 off, 5 unknown, 6 no visible traffic light, 0 clear human TL label, T rerun algorithm TL for current event")

    reviewer = EventReviewer(
        video_path=video_path,
        df=df,
        events=events,
        labels_csv=labels_csv,
        snippets_dir=snippets_dir,
        traffic_light_detector=traffic_light_detector,
    )
    events = reviewer.run()

    if TRAFFIC_LIGHT_AUTO_AFTER_REVIEW:
        logger.info("Running automatic traffic light detection after manual review for metrics.")
        if traffic_light_detector is None:
            traffic_light_detector = create_traffic_light_detector()
        annotate_traffic_light_states(events, video_path=video_path, df=df, detector=traffic_light_detector)
        refresh_traffic_light_review_fields(events)
        write_events_csv(labels_csv, events)

    missed_count = read_missed_crossings_count(missed_csv)
    metrics = calculate_metrics(events, missed_count=missed_count)
    traffic_light_metrics = calculate_traffic_light_metrics(events)

    write_events_csv(auto_csv, events)
    write_events_csv(labels_csv, events)
    write_json(metrics_json, metrics)
    write_confusion_matrix(matrix_csv, metrics)
    write_json(traffic_light_metrics_json, traffic_light_metrics)
    write_traffic_light_confusion_matrix(traffic_light_matrix_csv, events)

    write_json(
        summary_json,
        {
            "csv_path": csv_path,
            "video_path": video_path,
            "video_key": csv_info["video_key"],
            "source_video_id": csv_info["source_video_id"],
            "segment_start_seconds": csv_info["segment_start_seconds"],
            "fps": csv_info["fps"],
            "boundary_left": BOUNDARY_LEFT,
            "boundary_right": BOUNDARY_RIGHT,
            "min_confidence": MIN_CONFIDENCE,
            "review_confirmed_only": REVIEW_CONFIRMED_ONLY,
            "review_candidates_too": REVIEW_CANDIDATES_TOO,
            "traffic_light_state_enabled": ENABLE_TRAFFIC_LIGHT_STATE,
            "traffic_light_model_path": resolve_traffic_light_model_path(),
            "traffic_light_model_candidates": resolve_traffic_light_model_paths(),
            "traffic_light_final_state_rule": "human_traffic_light_state only; automatic traffic_light_state is stored separately for comparison",
            "traffic_light_auto_attempted_count": sum(1 for e in events if bool(getattr(e, "traffic_light_auto_attempted", False))),
            "traffic_light_confidence": TRAFFIC_LIGHT_CONFIDENCE,
            "traffic_light_window_before_frames": TRAFFIC_LIGHT_WINDOW_BEFORE_FRAMES,
            "traffic_light_window_after_frames": TRAFFIC_LIGHT_WINDOW_AFTER_FRAMES,
            "traffic_light_frame_step": TRAFFIC_LIGHT_FRAME_STEP,
            "traffic_light_image_size": TRAFFIC_LIGHT_IMAGE_SIZE,
            "traffic_light_min_state_confidence": TRAFFIC_LIGHT_MIN_STATE_CONFIDENCE,
            "traffic_light_preannotate_all_events": TRAFFIC_LIGHT_PREANNOTATE_ALL_EVENTS,
            "traffic_light_auto_before_review": TRAFFIC_LIGHT_AUTO_BEFORE_REVIEW,
            "traffic_light_auto_after_review": TRAFFIC_LIGHT_AUTO_AFTER_REVIEW,
            "traffic_light_detect_on_key": TRAFFIC_LIGHT_DETECT_ON_KEY,
            "traffic_light_recompute_automatic_state": TRAFFIC_LIGHT_RECOMPUTE_AUTOMATIC_STATE,
            "traffic_light_anchor_frame_tolerance": TRAFFIC_LIGHT_ANCHOR_FRAME_TOLERANCE,
            "traffic_light_max_anchor_centre_distance": TRAFFIC_LIGHT_MAX_ANCHOR_CENTRE_DISTANCE,
            "traffic_light_min_anchor_iou": TRAFFIC_LIGHT_MIN_ANCHOR_IOU,
            "automatic_confirmed_count": len(confirmed_ids),
            "automatic_candidate_count": len(candidate_ids),
            "rejected_candidate_count_added_for_review": rejected_candidate_count,
            "total_review_events": len(events),
            "metrics": metrics,
            "traffic_light_metrics": traffic_light_metrics,
        },
    )

    logger.info("Validation complete.")
    logger.info(f"Algorithm detections proposed for review: {auto_csv}")
    logger.info(f"Human labels: {labels_csv}")
    logger.info(f"Metrics: {metrics_json}")
    logger.info(f"Confusion matrix: {matrix_csv}")
    logger.info(f"Traffic light metrics: {traffic_light_metrics_json}")
    logger.info(f"Traffic light confusion matrix: {traffic_light_matrix_csv}")
    logger.info(f"Optional missed-crossings file: {missed_csv}")
    logger.info(f"Snippets: {snippets_dir}")
    logger.info("Metrics summary:\n{}", json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
