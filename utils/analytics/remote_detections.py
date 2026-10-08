"""Detection files fetched from the file server only when the analysis needs them.

The analysis reads detection files from the local Parquet store. Copying every
city's detection CSVs first is wasteful when ``target_crossings_per_city`` stops
each city after a few segments, so with ``fetch_detections_on_demand`` the
analysis works from an index of the files on the server instead:

- ``build_index`` records, once per segment, which detection file the server has
  for it (the file name carries the frame rate, so several are tried). The index
  is kept beside the Parquet store and only segments not yet in it are probed.
- ``indexed_segments`` is what segment selection treats as available, so the
  footage budget and the random draw order cover every segment with detections
  on the server, exactly as if all of them were local.
- ``fetch`` downloads one CSV, converts it to Parquet in the store and removes the
  CSV; the analysis calls it for the segments of each round just before
  processing them.
"""

from __future__ import annotations

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import requests

import common
from custom_logger import CustomLogger

logger = CustomLogger(__name__)

INDEX_FILE = "remote_detection_index.json"
# Frame rates tried for a segment's file name, most common first.
FPS_GUESSES = (30, 60, 24, 25, 50, 29, 15, 20, 48, 59, 23, 10, 12)
PROBE_THREADS = 16
FETCH_THREADS = 8
REQUEST_TIMEOUT_SECONDS = 120
# Pauses before retrying a request the server failed (5xx or no answer).
RETRY_PAUSES_SECONDS = (5, 15, 45, 120)


def enabled() -> bool:
    """Return whether detection files are fetched from the server on demand."""
    value = common.get_configs("fetch_detections_on_demand")
    if not isinstance(value, bool):
        raise ValueError("fetch_detections_on_demand must be true or false")
    return value


def _store_root() -> Path:
    from utils.analytics.parquet_store import configured_parquet_roots

    return Path(configured_parquet_roots()[0])


def _index_path() -> Path:
    return _store_root() / INDEX_FILE


def _base_url() -> str:
    path = str(common.get_configs("detection_csv_url_path"))
    return str(common.get_configs("ftp_base_url")).rstrip("/") + "/" + path.strip("/") + "/"


def _credentials():
    from visualize_segmentation_samples import _load_credentials

    return _load_credentials()


_local = threading.local()


def _session(credentials) -> requests.Session:
    if not hasattr(_local, "session"):
        session = requests.Session()
        if credentials.username and credentials.password:
            session.auth = (credentials.username, credentials.password)
        _local.session = session
    return _local.session


def _read_index() -> Dict[str, Optional[str]]:
    path = _index_path()
    if not path.is_file():
        return {}
    try:
        return dict(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError) as error:
        logger.warning(f"Unreadable remote detection index {path}: {error}")
        return {}


def _write_index(index: Dict[str, Optional[str]]) -> None:
    path = _index_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(index, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def _key(video_id: str, start: int) -> str:
    return f"{video_id}_{int(start)}"


def _probe(video_id: str, start: int, credentials, base_url: str) -> Optional[str]:
    """Return the stem of the segment's detection file on the server, or None if it has none."""
    for fps in FPS_GUESSES:
        stem = f"{video_id}_{int(start)}_{fps}"
        status = _status(base_url + stem + ".csv", credentials)
        if status == 200:
            return stem
        if status != 404:
            raise RuntimeError(f"The file server answered {status} for {stem}.csv.")
    return None


def _status(url: str, credentials) -> int:
    """Return the server's answer for ``url``, retrying failures and waiting through outages."""
    from utils.segmentation.frames import wait_through_outage

    failed_since = time.time()
    status = 0
    while True:
        for pause in (0,) + RETRY_PAUSES_SECONDS:
            time.sleep(pause)
            try:
                with _session(credentials).get(
                    url, params=credentials.request_params(), stream=True, timeout=REQUEST_TIMEOUT_SECONDS,
                ) as response:
                    status = response.status_code
            except requests.RequestException:
                status = 0
            if status and status < 500:
                return status
        # Still failing: wait while the whole server is down, then try again;
        # give up only if it does not come back.
        if not wait_through_outage(url, failed_since):
            return status
        failed_since = time.time()


def build_index(segments: Iterable[Tuple[str, int]]) -> Dict[str, Optional[str]]:
    """Record which detection file the server has for each segment not yet indexed."""
    index = _read_index()
    looked_up = {(key.rsplit("_", 1)[0], int(key.rsplit("_", 1)[1])) for key in index}
    pending = sorted({(str(video), int(start)) for video, start in segments} - looked_up)
    if not pending:
        return index
    logger.info(f"Looking up {len(pending)} segment(s) on the file server for on-demand detection files.")
    credentials = _credentials()
    base_url = _base_url()
    lock = threading.Lock()
    done = [0]

    def probe(segment: Tuple[str, int]) -> None:
        stem = _probe(segment[0], segment[1], credentials, base_url)
        with lock:
            index[_key(*segment)] = stem
            done[0] += 1
            if done[0] % 500 == 0:
                _write_index(index)
                logger.info(f"Looked up {done[0]} of {len(pending)} segment(s).")

    try:
        with ThreadPoolExecutor(PROBE_THREADS) as executor:
            list(executor.map(probe, pending))
    finally:
        # Keep what was looked up even if the run stops, so it is not repeated.
        with lock:
            _write_index(index)
    found = sum(1 for segment in pending if index.get(_key(*segment)))
    logger.info(f"{found} of {len(pending)} looked-up segment(s) have a detection file on the server.")
    return index


def indexed_segments() -> Dict[Tuple[str, int], str]:
    """Return ``{(video, start): stem}`` for every indexed segment with a file on the server."""
    if not enabled():
        return {}
    segments: Dict[Tuple[str, int], str] = {}
    for key, stem in _read_index().items():
        if not stem:
            continue
        video, start = key.rsplit("_", 1)
        segments[(video, int(start))] = stem
    return segments


def mapping_segments(df_mapping) -> List[Tuple[str, int]]:
    """Return ``(video, start)`` for every segment in ``df_mapping`` filmed from an analysed vehicle type."""
    import ast

    vehicles = common.get_configs("vehicles_analyse")
    segments: List[Tuple[str, int]] = []
    for row in df_mapping.select(["videos", "start_time", "vehicle_type"]).iter_rows():
        videos = [video.strip() for video in str(row[0] or "").strip("[]").split(",") if video.strip()]
        try:
            starts = ast.literal_eval(str(row[1]))
            vehicle_types = ast.literal_eval(str(row[2]))
        except (ValueError, SyntaxError):
            continue
        for position, video in enumerate(videos):
            if vehicles and (position >= len(vehicle_types) or vehicle_types[position] not in vehicles):
                continue
            for start in (starts[position] if position < len(starts) else []):
                segments.append((video, int(start)))
    return segments


def parquet_path(stem: str) -> Path:
    from utils.analytics.parquet_store import DETECTION_FOLDER

    return _store_root() / DETECTION_FOLDER / f"{stem}.parquet"


def fetch(stem: str) -> Optional[Path]:
    """Download one detection CSV, convert it into the Parquet store and remove the CSV."""
    from utils.analytics.parquet_store import DEFAULT_COMPRESSION, DEFAULT_ROW_GROUP_SIZE, _convert_one

    target = parquet_path(stem)
    if target.is_file():
        return target
    credentials = _credentials()
    csv_folder = Path(common.get_configs("data")[0]) / "bbox"
    csv_folder.mkdir(parents=True, exist_ok=True)
    csv_path = csv_folder / f"{stem}.csv"
    temporary = csv_path.with_suffix(".csv.part")
    from utils.segmentation.frames import wait_through_outage

    failed_since = time.time()
    pauses = (0,) + RETRY_PAUSES_SECONDS
    attempt = 0
    while True:
        time.sleep(pauses[min(attempt, len(pauses) - 1)])
        try:
            with _session(credentials).get(
                _base_url() + stem + ".csv", params=credentials.request_params(), stream=True,
                timeout=REQUEST_TIMEOUT_SECONDS,
            ) as response:
                response.raise_for_status()
                with temporary.open("wb") as handle:
                    for chunk in response.iter_content(1 << 20):
                        handle.write(chunk)
            break
        except (requests.RequestException, OSError) as error:
            attempt += 1
            if attempt < len(pauses):
                continue
            # The server keeps failing: wait while it is down, then start over.
            if wait_through_outage(_base_url(), failed_since):
                attempt, failed_since = 0, time.time()
                continue
            logger.warning(f"Could not download {stem}.csv: {error}")
            temporary.unlink(missing_ok=True)
            return None
    os.replace(temporary, csv_path)
    try:
        _convert_one(str(csv_path), str(target), DEFAULT_COMPRESSION, DEFAULT_ROW_GROUP_SIZE)
    except Exception as error:
        logger.warning(f"Could not convert {stem}.csv to Parquet: {error}")
        return None
    finally:
        csv_path.unlink(missing_ok=True)
    return target


def fetch_tasks(tasks: List[dict]) -> List[dict]:
    """Make sure every task's Parquet file is local, fetching on demand; return the usable tasks."""
    missing = [task for task in tasks if task.get("remote") and not Path(task["file_path"]).is_file()]
    if missing:
        with ThreadPoolExecutor(FETCH_THREADS) as executor:
            list(executor.map(lambda task: fetch(str(task["filename_no_ext"])), missing))
    usable = [task for task in tasks if Path(task["file_path"]).is_file()]
    if len(usable) < len(tasks):
        logger.warning(f"{len(tasks) - len(usable)} detection file(s) could not be fetched and are skipped.")
    return usable
