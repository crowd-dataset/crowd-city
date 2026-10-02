"""Fetch only the crossing windows of a video, without downloading the file.

The corpus is far too large to copy in full, so ffmpeg input-seeks over HTTP
range requests and decodes just the seconds that contain a crossing. Frames are
scaled by ffmpeg to the segmentation input size and arrive as raw RGB, so no
intermediate image files are written.
"""

from __future__ import annotations

import base64
import json
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional
from urllib.parse import urljoin, urlsplit

import numpy as np
import requests

from custom_logger import CustomLogger


logger = CustomLogger(__name__)

# The video corpus lives in exactly these two shared folders on the file
# server: tue4 holds roughly 41000 videos and tue5 roughly 30000. The other
# aliases the server exposes (data, tuecoco) hold no crossing footage, and the
# tue1 to tue3 aliases used elsewhere in this project no longer exist and
# answer "Unknown folder alias".
SERVER_ALIASES = ("tue4", "tue5")
FFMPEG_BINARY = "ffmpeg"
# The file server returns transient 5XX errors and timeouts under load, so a
# failed read is retried after these pauses (seconds) before giving up.
READ_RETRY_PAUSES_SECONDS = (5.0, 15.0, 45.0)
# A healthy window reads in seconds; a read still hanging after this has
# stalled on the server, and a prompt retry recovers faster than waiting.
WINDOW_READ_TIMEOUT_SECONDS = 180


FFPROBE_BINARY = "ffprobe"

# When the file server itself is down (it answers 5XX or not at all for its
# front page), every read fails, and failing each segment in turn would burn
# through a whole run in hours and then stop it on the failure limit. Reads
# instead wait for the server to come back, polling it at this interval, for
# at most this long before the failure is accepted.
SERVER_POLL_SECONDS = 60.0
SERVER_OUTAGE_MAX_WAIT_SECONDS = 12 * 3600.0

_server_lock = threading.Lock()
_last_recovery_time = 0.0


def _server_responding(origin: str) -> bool:
    """Return whether the file server answers at all (any reply below 500)."""
    try:
        response = requests.get(origin, timeout=20, stream=True)
    except requests.RequestException:
        return False
    try:
        return response.status_code < 500
    finally:
        response.close()


def wait_through_outage(url: str, failed_since: float) -> bool:
    """Wait while the file server is down; return True once it is back.

    True means the caller's failures may have been caused by an outage that
    has now ended, so the read is worth repeating. False means the server is
    responding normally (the failure belongs to this file) or stayed down for
    SERVER_OUTAGE_MAX_WAIT_SECONDS. Only one thread polls; the others queue on
    the lock and then see the recovery it recorded.
    """
    global _last_recovery_time
    parts = urlsplit(str(url))
    if parts.scheme not in ("http", "https") or not parts.netloc:
        return False
    origin = f"{parts.scheme}://{parts.netloc}/"
    with _server_lock:
        if _last_recovery_time > failed_since:
            return True
        if _server_responding(origin):
            return False
        started = time.time()
        logger.warning(
            f"The file server {origin} is down; pausing reads until it responds again "
            f"(checking every {SERVER_POLL_SECONDS:.0f}s, for at most {SERVER_OUTAGE_MAX_WAIT_SECONDS / 3600:.0f} h)."
        )
        last_report = started
        while time.time() - started < SERVER_OUTAGE_MAX_WAIT_SECONDS:
            time.sleep(SERVER_POLL_SECONDS)
            if _server_responding(origin):
                _last_recovery_time = time.time()
                logger.warning(
                    f"The file server is back after {(_last_recovery_time - started) / 60:.0f} min; resuming reads."
                )
                return True
            if time.time() - last_report >= 1800:
                logger.warning(f"The file server is still down after {(time.time() - started) / 60:.0f} min.")
                last_report = time.time()
        logger.error(f"The file server stayed down for {SERVER_OUTAGE_MAX_WAIT_SECONDS / 3600:.0f} h; giving up.")
        return False


class WindowReadError(RuntimeError):
    """A video window could not be read even after retrying."""


@dataclass(frozen=True)
class FrameWindow:
    """One contiguous span of video time to decode."""

    start_seconds: float
    duration_seconds: float


@dataclass(frozen=True)
class FrameClock:
    """Map one detection segment's analysis frame numbers onto video time.

    The raw frame-count advances once per decoded frame, so it runs at the
    video's real rate, which the detection file name only carries rounded to
    an integer (29.97 is stored as 30). When processing_fps downsampled the
    detections, frames were also renumbered onto a coarser grid anchored at
    the first retained source frame; this undoes that as well.
    """

    start_seconds: float
    video_fps: float
    origin_frame: float = 0.0
    source_frames_per_frame: float = 1.0

    @classmethod
    def for_segment(
        cls,
        start_seconds: float,
        video_fps: float,
        source_fps: float,
        detection_fps: float,
        first_source_frame: float = 0.0,
    ) -> "FrameClock":
        # Mirrors _resample_detection_fps, which only renumbers when the
        # processing rate is below the (nominal) source rate.
        if 0 < detection_fps < source_fps - 1e-6:
            return cls(
                start_seconds=float(start_seconds),
                video_fps=float(video_fps),
                origin_frame=float(first_source_frame),
                source_frames_per_frame=float(source_fps) / float(detection_fps),
            )
        return cls(start_seconds=float(start_seconds), video_fps=float(video_fps))

    @property
    def frames_per_second(self) -> float:
        """Analysis frames per second of real video time."""
        return self.video_fps / self.source_frames_per_frame

    def seconds(self, frame: float) -> float:
        source_frame = self.origin_frame + float(frame) * self.source_frames_per_frame
        return self.start_seconds + source_frame / self.video_fps

    def frame(self, seconds: float) -> float:
        source_frame = (float(seconds) - self.start_seconds) * self.video_fps
        return (source_frame - self.origin_frame) / self.source_frames_per_frame


@dataclass
class RemoteCredentials:
    """Credentials for the CROWD file server."""

    base_url: str
    username: Optional[str] = None
    password: Optional[str] = None
    token: Optional[str] = None

    def request_params(self) -> Optional[Dict[str, str]]:
        return {"token": self.token} if self.token else None


# ---------------------------------------------------------------------
# URL resolution
# ---------------------------------------------------------------------

def _probe(session: requests.Session, url: str, params, timeout: int) -> bool:
    """Return whether ``url`` serves bytes, retrying transient server errors."""
    failed_since = time.time()
    for _round in range(2):
        for pause in (0.0,) + READ_RETRY_PAUSES_SECONDS:
            if pause:
                time.sleep(pause)
            result = _probe_once(session, url, params, timeout)
            if result is not None:
                return result
        if not wait_through_outage(url, failed_since):
            break
    logger.warning(f"The file server kept failing for {url}; treating it as unavailable.")
    return False


def _probe_once(session: requests.Session, url: str, params, timeout: int) -> Optional[bool]:
    """Return whether ``url`` serves bytes and honours range requests.

    None means the answer is unknown (a timeout, connection error or 5XX
    reply), which is worth retrying; a 404 is a definite no.
    """
    try:
        response = session.get(
            url,
            params=params,
            timeout=timeout,
            stream=True,
            headers={"Range": "bytes=0-1"},
        )
    except requests.RequestException as error:
        logger.debug(f"Probe failed [{url}]: {error}")
        return None

    try:
        if response.status_code >= 500:
            return None
        if response.status_code not in (200, 206):
            return False
        if response.status_code == 200:
            # The server ignored the range request. ffmpeg would then have to
            # stream the file from the beginning to reach a late crossing,
            # which defeats the point of seeking.
            logger.warning(
                f"File server ignored a range request for {url}; "
                "window extraction will be slow for late segments."
            )
        return True
    finally:
        response.close()


def resolve_video_url(
    video_id: str,
    credentials: RemoteCredentials,
    session: Optional[requests.Session] = None,
    timeout: int = 20,
) -> Optional[str]:
    """Return a directly fetchable URL for ``video_id``, or None.

    The ``v/<alias>/files/<name>.mp4`` pattern is authoritative on this server,
    so a video that 404s on both aliases is simply not hosted. No directory
    crawl is attempted: it cost seconds per miss and would dominate the
    runtime of a pass that resolves tens of thousands of videos.
    """
    base = credentials.base_url
    if not base:
        logger.error("Base URL is missing; cannot resolve remote videos.")
        return None
    base = base if base.endswith("/") else base + "/"

    filename = video_id if video_id.lower().endswith(".mp4") else f"{video_id}.mp4"
    params = credentials.request_params()

    owns_session = session is None
    session = session or requests.Session()
    if owns_session and credentials.username and credentials.password:
        session.auth = (credentials.username, credentials.password)
        session.headers.update({"User-Agent": "crowd-city-segmentation/1.0"})

    try:
        for alias in SERVER_ALIASES:
            direct = urljoin(base, f"v/{alias}/files/{filename}")
            if _probe(session, direct, params, timeout):
                logger.debug(f"Resolved {video_id} to {direct}.")
                return direct

        return None
    finally:
        if owns_session:
            session.close()


# ---------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------

def _authorisation_header(credentials: RemoteCredentials) -> Optional[str]:
    if not (credentials.username and credentials.password):
        return None
    raw = f"{credentials.username}:{credentials.password}".encode("utf-8")
    return "Authorization: Basic " + base64.b64encode(raw).decode("ascii") + "\r\n"


def _parse_frame_rate(text: object) -> Optional[float]:
    # ffprobe reports rates as fractions such as "30000/1001"; "0/0" means unknown.
    value = str(text or "").strip()
    if not value:
        return None
    try:
        if "/" in value:
            numerator, denominator = value.split("/", 1)
            rate = float(numerator) / float(denominator)
        else:
            rate = float(value)
    except (ValueError, ZeroDivisionError):
        return None
    return rate if np.isfinite(rate) and rate > 0 else None


def probe_video_fps(
    source: str,
    credentials: Optional[RemoteCredentials] = None,
    timeout: int = 60,
) -> Optional[float]:
    """Return the true frame rate of ``source`` as reported by ffprobe, or None.

    The detection filenames carry the frame rate rounded to an integer, so a
    29.97 fps video is labelled 30. Mapping detection frames to video time with
    the rounded value drifts by one frame every ~33 s, which is enough to put
    boxes visibly off their pedestrian late in a segment.
    """
    command: List[str] = [FFPROBE_BINARY, "-v", "error"]
    if str(source).startswith(("http://", "https://")) and credentials is not None:
        header = _authorisation_header(credentials)
        if header:
            command += ["-headers", header]
    command += [
        "-select_streams", "v:0",
        "-show_entries", "stream=avg_frame_rate,r_frame_rate",
        "-of", "json",
        str(source),
    ]

    try:
        completed = subprocess.run(
            command, capture_output=True, timeout=timeout, check=False,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError) as error:
        logger.warning(f"ffprobe could not read the frame rate of {source}: {error}")
        return None
    if completed.returncode != 0:
        message = completed.stderr.decode("utf-8", errors="replace").strip()
        logger.warning(f"ffprobe failed for {source}: {message}")
        return None

    try:
        streams = json.loads(completed.stdout.decode("utf-8")).get("streams") or []
    except (ValueError, UnicodeDecodeError):
        return None
    if not streams:
        return None
    # The average rate is what frame index to time conversion needs; the
    # nominal r_frame_rate is only a fallback when the container omits it.
    return _parse_frame_rate(streams[0].get("avg_frame_rate")) or _parse_frame_rate(
        streams[0].get("r_frame_rate")
    )


def extract_window_frames(
    source: str,
    window: FrameWindow,
    cadence_hz: float,
    width: int,
    height: int,
    credentials: Optional[RemoteCredentials] = None,
    timeout: int = WINDOW_READ_TIMEOUT_SECONDS,
) -> np.ndarray:
    """Decode one window and return ``(n, height, width, 3)`` uint8 RGB frames.

    Output frame ``i`` corresponds to ``window.start_seconds + i / cadence_hz``.
    """
    if window.duration_seconds <= 0 or cadence_hz <= 0:
        return np.zeros((0, height, width, 3), dtype=np.uint8)

    command: List[str] = [FFMPEG_BINARY, "-nostdin", "-loglevel", "error"]

    is_remote = str(source).startswith(("http://", "https://"))
    if is_remote:
        command += [
            "-reconnect", "1",
            "-reconnect_streamed", "1",
            "-reconnect_delay_max", "5",
        ]
        if credentials is not None:
            header = _authorisation_header(credentials)
            if header:
                command += ["-headers", header]

    command += [
        # Input seeking: ffmpeg jumps straight to the window with a range
        # request instead of decoding everything before it.
        "-ss", f"{float(window.start_seconds):.3f}",
        "-i", str(source),
        "-t", f"{float(window.duration_seconds):.3f}",
        "-an", "-sn",
        "-vf", f"fps={float(cadence_hz):g},scale={int(width)}:{int(height)}",
        "-pix_fmt", "rgb24",
        "-f", "rawvideo",
        "-",
    ]

    completed = None
    failed_since = time.time()
    pauses = (0.0,) + READ_RETRY_PAUSES_SECONDS
    # A second round of attempts is made only after an outage of the whole
    # server has ended (see wait_through_outage).
    schedule = list(enumerate(pauses))
    position = 0
    while position < len(schedule):
        attempt, pause = schedule[position]
        position += 1
        if pause:
            time.sleep(pause)
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            problem = f"timed out after {timeout}s"
        except FileNotFoundError as error:
            raise WindowReadError(f"{FFMPEG_BINARY} was not found on PATH.") from error
        else:
            if completed.returncode == 0:
                break
            problem = completed.stderr.decode("utf-8", errors="replace").strip()
        logger.warning(
            f"Reading {window.duration_seconds:.1f}s at {window.start_seconds:.1f}s of "
            f"{source} failed (attempt {attempt + 1} of {len(pauses)}): {problem}"
        )
        completed = None
        if position == len(schedule) and is_remote and len(schedule) == len(pauses):
            if wait_through_outage(str(source), failed_since):
                schedule += list(enumerate(pauses))
    if completed is None:
        # Raised rather than returning no frames, so the segment counts as
        # failed and is never stored as if the window had nothing in it.
        raise WindowReadError(
            f"Could not read {window.duration_seconds:.1f}s at {window.start_seconds:.1f}s of {source}."
        )

    frame_bytes = int(width) * int(height) * 3
    payload = completed.stdout
    usable = (len(payload) // frame_bytes) * frame_bytes
    if usable == 0:
        return np.zeros((0, height, width, 3), dtype=np.uint8)
    if usable != len(payload):
        logger.debug(
            f"Discarding {len(payload) - usable} trailing bytes from ffmpeg output."
        )

    return (
        np.frombuffer(payload[:usable], dtype=np.uint8)
        .reshape(-1, int(height), int(width), 3)
    )
