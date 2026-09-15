"""Fetch only the crossing windows of a video, without downloading the file.

The corpus is far too large to copy in full, so ffmpeg input-seeks over HTTP
range requests and decodes just the seconds that contain a crossing. Frames are
scaled by ffmpeg to the segmentation input size and arrive as raw RGB, so no
intermediate image files are written.
"""

from __future__ import annotations

import base64
import subprocess
from dataclasses import dataclass
from typing import Dict, List, Optional
from urllib.parse import urljoin

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


@dataclass(frozen=True)
class FrameWindow:
    """One contiguous span of video time to decode."""

    start_seconds: float
    duration_seconds: float


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
    """Return whether ``url`` serves bytes and honours range requests."""
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
        return False

    try:
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


def extract_window_frames(
    source: str,
    window: FrameWindow,
    cadence_hz: float,
    width: int,
    height: int,
    credentials: Optional[RemoteCredentials] = None,
    timeout: int = 600,
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

    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        logger.warning(
            f"ffmpeg timed out extracting {window.duration_seconds:.1f}s at "
            f"{window.start_seconds:.1f}s from {source}."
        )
        return np.zeros((0, height, width, 3), dtype=np.uint8)
    except FileNotFoundError:
        logger.error(f"{FFMPEG_BINARY} was not found on PATH.")
        return np.zeros((0, height, width, 3), dtype=np.uint8)

    if completed.returncode != 0:
        message = completed.stderr.decode("utf-8", errors="replace").strip()
        logger.warning(
            f"ffmpeg failed at {window.start_seconds:.1f}s of {source}: {message}"
        )
        return np.zeros((0, height, width, 3), dtype=np.uint8)

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
