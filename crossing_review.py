"""Local web tool for counting crossings by hand and comparing them with the algorithm.

Run ``python crossing_review.py`` and open the address it prints. Enter a
video id and press **Process video**. That does all the slow work once, before
any reviewing starts, so nothing lags afterwards:

* the whole video is downloaded from the file server into
  ``_output/crossing_review/videos``, so playback and seeking are local,
* every segment's YOLO person tracks are read from its detection file (fetched
  from the file server's ``data`` alias when it is not in the Parquet store),
* the crossing algorithm (detection worker, then ``crossing_rule``, with the
  current config) runs on every full segment, reading the local video.

While reviewing, the crossings the algorithm counted are drawn as yellow
boxes. The reviewer clicks one to confirm it is a real crossing (it turns
green), and clicks every other pedestrian seen crossing: if a YOLO person box
is there it is drawn in purple, otherwise an orange circle marks the spot. The
page keeps four counts as the reviewer goes:

1. crossing, not detected by YOLO (orange),
2. crossing detected by YOLO, missed by the algorithm (purple),
3. crossing detected by YOLO and counted by the algorithm (green),
4. fake crossing: a counted crossing never confirmed (still yellow).

Labels are saved after every change in
``_output/crossing_review/labels/<video_id>.json``. The file-server
credentials come from the secrets file and never reach the browser.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import re
import threading
import time
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, urlsplit

import polars as pl
import requests

import common
from custom_logger import CustomLogger
from logmod import logs

logger = CustomLogger(__name__)

ROOT = Path(__file__).resolve().parent
HTML_PATH = ROOT / "crossing_review.html"
REVIEW_DIR = Path(common.output_dir) / "crossing_review"
SEGMENT_DIR = REVIEW_DIR / "segments"
LABEL_DIR = REVIEW_DIR / "labels"
VIDEO_DIR = REVIEW_DIR / "videos"
# Person boxes are sent to the page at this rate; they are only needed to
# match clicks and to show the matched or counted box afterwards.
SAMPLE_HZ = 10.0
# Detection file names end in the frame rate, which the mapping does not record.
FPS_GUESSES = (30, 60, 24, 25, 50, 29, 15, 20, 48, 59, 23, 10, 12)
PERSON_CLASS = 0
DOWNLOAD_CHUNK_BYTES = 1024 * 1024
DOWNLOAD_PART_BYTES = 16 * 1024 * 1024
DOWNLOAD_CONNECTIONS = 8
SERVE_CHUNK_BYTES = 256 * 1024
VIDEO_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{6,20}$")
# Bumped when the cached segment files change shape or meaning.
SEGMENT_CACHE_VERSION = 3


# ---------------------------------------------------------------------------
# Mapping, detection files and the video
# ---------------------------------------------------------------------------

def _split_list(text: Optional[str]) -> List[str]:
    text = str(text or "").strip().strip("[]")
    return [part.strip().strip("'\"") for part in text.split(",") if part.strip()]


def _literal(text: Optional[str]) -> Any:
    import ast

    try:
        return ast.literal_eval(str(text))
    except (SyntaxError, ValueError):
        return []


class Mapping:
    """The mapping row that mentions each video, indexed once."""

    def __init__(self) -> None:
        self.df = pl.read_csv(common.get_configs("mapping"), infer_schema_length=0)
        self._by_video: Dict[str, Any] = {}
        for row in self.df.iter_rows(named=True):
            for position, video in enumerate(_split_list(row.get("videos"))):
                self._by_video.setdefault(video, (row, position))

    def segments(self, video_id: str) -> Optional[Dict[str, Any]]:
        found = self._by_video.get(video_id)
        if found is None:
            return None
        row, position = found
        starts = (_literal(row.get("start_time")) or [])[position]
        ends = (_literal(row.get("end_time")) or [])[position]
        times = (_literal(row.get("time_of_day")) or [])[position]
        city = ", ".join(str(v) for v in (row.get("locality"), row.get("state"), row.get("country")) if v)
        return {
            "video_id": video_id,
            "city": city,
            "row_id": row.get("id"),
            "segments": [
                {
                    "start": int(start),
                    "end": int(end),
                    "time_of_day": ("night" if int(times[index]) == 1 else "day") if index < len(times) else "",
                }
                for index, (start, end) in enumerate(zip(starts, ends))
            ],
        }


def _parquet_roots() -> List[str]:
    from utils.analytics.parquet_store import configured_parquet_roots

    return configured_parquet_roots()


def _local_parquet(video_id: str, start: int) -> Optional[Path]:
    for root in _parquet_roots():
        matches = sorted((Path(root) / "bbox").glob(f"{video_id}_{start}_*.parquet"))
        if matches:
            return matches[0]
    return None


def _credentials():
    from visualize_segmentation_samples import _load_credentials

    return _load_credentials()


def _session(credentials) -> requests.Session:
    session = requests.Session()
    if credentials.username and credentials.password:
        session.auth = (credentials.username, credentials.password)
    return session


def _download_detection_file(video_id: str, start: int, csv_url: str) -> Optional[Path]:
    """Fetch the detection CSV from the file server and convert it to Parquet."""
    from utils.analytics.parquet_store import DEFAULT_COMPRESSION, DEFAULT_ROW_GROUP_SIZE, _convert_one

    credentials = _credentials()
    session = _session(credentials)
    csv_root = Path(common.get_configs("data")[0]) / "bbox"
    parquet_root = Path(_parquet_roots()[0]) / "bbox"
    csv_root.mkdir(parents=True, exist_ok=True)
    for fps in FPS_GUESSES:
        stem = f"{video_id}_{start}_{fps}"
        response = session.get(csv_url + stem + ".csv", params=credentials.request_params(), timeout=600)
        if response.status_code == 404:
            continue
        response.raise_for_status()
        csv_path = csv_root / f"{stem}.csv"
        temporary = csv_path.with_suffix(".csv.part")
        temporary.write_bytes(response.content)
        os.replace(temporary, csv_path)
        parquet_path = parquet_root / f"{stem}.parquet"
        _convert_one(str(csv_path), str(parquet_path), DEFAULT_COMPRESSION, DEFAULT_ROW_GROUP_SIZE)
        return parquet_path
    return None


def video_path(video_id: str) -> Path:
    return VIDEO_DIR / f"{video_id}.mp4"


def _download_video(video_id: str, progress) -> Path:
    """Download the whole video once; later reviews and the algorithm read it locally.

    The file server limits each connection rather than the total, so the file
    is fetched as DOWNLOAD_PARTS-sized byte ranges over DOWNLOAD_CONNECTIONS
    connections at once, each written at its own offset.
    """
    from concurrent.futures import ThreadPoolExecutor
    from utils.segmentation.frames import resolve_video_url

    target = video_path(video_id)
    if target.is_file():
        return target
    credentials = _credentials()
    url = resolve_video_url(video_id, credentials)
    if not url:
        raise RuntimeError(f"{video_id} was not found on the file server.")
    session = _session(credentials)
    params = credentials.request_params()
    with session.get(url, params=params, headers={"Range": "bytes=0-0"}, stream=True, timeout=60) as probe:
        probe.raise_for_status()
        content_range = probe.headers.get("Content-Range", "")
    total = int(content_range.rsplit("/", 1)[1]) if "/" in content_range else 0
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".mp4.part")
    done = [0]
    lock = threading.Lock()
    started = time.time()

    def report() -> None:
        rate = done[0] / max(time.time() - started, 1e-6) / 1e6
        share = f"{100 * done[0] / total:.0f}% of {total / 1e6:.0f} MB" if total else f"{done[0] / 1e6:.0f} MB"
        progress(f"Downloading the video: {share} at {rate:.1f} MB/s.", done[0] / total if total else None)

    if not total:
        # No range support: one plain download.
        with session.get(url, params=params, stream=True, timeout=120) as response, temporary.open("wb") as handle:
            response.raise_for_status()
            for chunk in response.iter_content(DOWNLOAD_CHUNK_BYTES):
                handle.write(chunk)
                done[0] += len(chunk)
                report()
        os.replace(temporary, target)
        return target

    with temporary.open("wb") as handle:
        handle.truncate(total)

    def fetch(first: int) -> None:
        last = min(first + DOWNLOAD_PART_BYTES, total) - 1
        for attempt in range(5):
            written = 0
            try:
                own = _session(credentials)
                with own.get(url, params=params, headers={"Range": f"bytes={first}-{last}"},
                             stream=True, timeout=120) as response:
                    response.raise_for_status()
                    with temporary.open("r+b") as handle:
                        handle.seek(first)
                        for chunk in response.iter_content(DOWNLOAD_CHUNK_BYTES):
                            handle.write(chunk)
                            written += len(chunk)
                            with lock:
                                done[0] += len(chunk)
                                report()
                if written == last - first + 1:
                    return
            except requests.RequestException:
                pass
            with lock:
                done[0] -= written
            time.sleep(5 * (attempt + 1))
        raise RuntimeError(f"Could not download bytes {first}-{last} of {video_id}.")

    with ThreadPoolExecutor(DOWNLOAD_CONNECTIONS) as executor:
        list(executor.map(fetch, range(0, total, DOWNLOAD_PART_BYTES)))
    os.replace(temporary, target)
    return target


def _normalise_id(value: Any) -> str:
    text = str(value)
    return text[:-2] if text.endswith(".0") else text


def _person_tracks(detections: pl.DataFrame, clock, detection_fps: float) -> List[Dict[str, Any]]:
    """Every YOLO person track the analysis sees, sampled at SAMPLE_HZ, on video time."""
    people = detections.filter(
        (pl.col("yolo-id").cast(pl.Float64, strict=False) == PERSON_CLASS) & pl.col("unique-id").is_not_null()
    )
    step = max(1, int(round(detection_fps / SAMPLE_HZ)))
    tracks: List[Dict[str, Any]] = []
    for track in people.partition_by("unique-id", maintain_order=False):
        ordered = track.sort("frame-count")
        frames = ordered.get_column("frame-count").cast(pl.Int64, strict=False).to_list()
        boxes = ordered.select(["x-center", "y-center", "width", "height"]).cast(pl.Float64, strict=False).rows()
        times, sampled, last = [], [], None
        for frame, box in zip(frames, boxes):
            if last is not None and frame - last < step:
                continue
            last = frame
            times.append(round(clock.seconds(frame), 3))
            sampled.append([round(v, 4) for v in box])
        if len(times) >= 2:
            tracks.append({"id": _normalise_id(track.get_column("unique-id")[0]), "t": times, "b": sampled})
    tracks.sort(key=lambda item: item["t"][0])
    return tracks


# ---------------------------------------------------------------------------
# Processing a whole video
# ---------------------------------------------------------------------------

class Processor:
    """Downloads a video and runs the algorithm on all its segments, in the background."""

    def __init__(self, mapping: Mapping, csv_url: str) -> None:
        self.mapping = mapping
        self.csv_url = csv_url
        self.lock = threading.Lock()
        self.status: Dict[str, Dict[str, Any]] = {}
        self._ready = False

    @staticmethod
    def segment_path(video_id: str, start: int) -> Path:
        return SEGMENT_DIR / f"{video_id}_{start}.json"

    def segment_ready(self, video_id: str, start: int) -> bool:
        path = self.segment_path(video_id, start)
        if not path.is_file():
            return False
        try:
            return json.loads(path.read_text(encoding="utf-8")).get("version") == SEGMENT_CACHE_VERSION
        except (OSError, json.JSONDecodeError):
            return False

    def state(self, video_id: str) -> Dict[str, Any]:
        info = self.mapping.segments(video_id) or {"segments": []}
        ready = [s["start"] for s in info["segments"] if self.segment_ready(video_id, s["start"])]
        if video_path(video_id).is_file() and info["segments"] and len(ready) == len(info["segments"]):
            return {"status": "ready", "ready_segments": ready}
        current = dict(self.status.get(video_id, {"status": "not_processed"}))
        current["ready_segments"] = ready
        return current

    def start(self, video_id: str) -> Dict[str, Any]:
        current = self.state(video_id)
        if current["status"] in ("ready", "queued", "running"):
            return current
        self.status[video_id] = {"status": "queued", "message": "Waiting for another video to finish processing."}
        threading.Thread(target=self._run, args=(video_id,), daemon=True).start()
        return self.state(video_id)

    def _run(self, video_id: str) -> None:
        with self.lock:
            try:
                self._process(video_id)
                self.status[video_id] = {"status": "ready"}
            except Exception as error:
                logger.error(f"{video_id}: {error}\n{traceback.format_exc()}")
                self.status[video_id] = {"status": "error", "message": str(error)}

    def _progress(self, video_id: str, message: str, fraction: Optional[float] = None) -> None:
        previous = self.status.get(video_id, {}).get("message")
        self.status[video_id] = {"status": "running", "message": message, "fraction": fraction}
        if message != previous and "Downloading the video" not in message:
            logger.info(f"{video_id}: {message}")

    def _initialise(self) -> None:
        if self._ready:
            return
        import utils.analytics.csv_parallel as csv_parallel
        import utils.crossing.metrics as metrics

        # A review covers whole segments, so the analysis-wide footage cap and
        # crossing target must not trim or skip any of them. This changes only
        # this process's copy of the configuration.
        settings = common._load_config_once()
        settings["max_footage_hours_per_city"] = None
        settings["target_crossings_per_city"] = None
        metrics.ensure_waymo_processed(
            raw_dataset_path=common.get_configs("waymo_dataset_path"),
            repository_root=common.root_dir,
            output_root=common.output_dir,
            process_if_missing=bool(common.get_configs("process_waymo_if_missing")),
            log=lambda message: logger.debug(message),
        )
        if not metrics.metric_speed_is_qualified():
            raise RuntimeError("The frozen Waymo model is not available; check waymo_dataset_path in config.")
        csv_parallel.initialise_csv_worker(
            self.mapping.df,
            metrics.tuned_crossing_parameters(),
            float(common.get_configs("min_confidence")),
            float(common.get_configs("boundary_left")),
            float(common.get_configs("boundary_right")),
            dict(metrics._PIPELINE_MODEL),
            dict(metrics._SPEED_MODEL),
        )
        self._ready = True

    def _process(self, video_id: str) -> None:
        info = self.mapping.segments(video_id)
        if not info or not info["segments"]:
            raise RuntimeError(f"{video_id} has no segments in the mapping.")
        self._progress(video_id, "Starting the download.", 0.0)
        local_video = _download_video(video_id, lambda message, share: self._progress(video_id, message, share))
        self._progress(video_id, "Loading the crossing model.")
        self._initialise()
        segments = info["segments"]
        for index, segment in enumerate(segments, 1):
            if self.segment_ready(video_id, segment["start"]):
                continue
            label = f"Segment {index} of {len(segments)} ({segment['start']}–{segment['end']} s)"
            share = (index - 1) / len(segments)
            self._process_segment(
                video_id, info, segment, local_video,
                lambda message: self._progress(video_id, f"{label}: {message}", share),
            )

    def _process_segment(self, video_id: str, info: Dict[str, Any], segment: Dict[str, Any],
                         local_video: Path, progress) -> None:
        import utils.analytics.csv_parallel as csv_parallel
        import utils.crossing.metrics as metrics
        import utils.segmentation.crossing_pass as crossing_pass
        from utils.core.metadata import MetaData
        from utils.segmentation.frames import FrameClock, probe_video_fps

        start = int(segment["start"])
        progress("finding the detection file.")
        parquet = _local_parquet(video_id, start)
        if parquet is None:
            progress("downloading the detection file.")
            parquet = _download_detection_file(video_id, start, self.csv_url)
        if parquet is None:
            raise RuntimeError(f"No detection file exists for the segment at {start} s.")
        stem = parquet.stem
        fps = float(stem.rsplit("_", 1)[1])
        task = dict(
            file_path=str(parquet), file_name=parquet.name, filename_no_ext=stem, video_id=video_id,
            start_index=start, fps=fps, video_locality_id=int(info["row_id"]),
            time_video=float(segment["end"] - segment["start"]), is_bbox_stream=True,
        )
        progress("running the detection worker.")
        # The segment index remembers which detection files existed when it was
        # built; a file downloaded since then must be seen too.
        MetaData.clear_video_index_cache()
        result = csv_parallel.process_csv_task(task)
        if result.get("status") != "ok":
            raise RuntimeError(result.get("message") or "The detection worker failed.")

        rule = str(common.get_configs("crossing_rule"))
        if rule == "road_crossing":
            candidates = result.get("road_candidates") or {}
            candidate_ids = [str(v) for v in candidates.get("ids") or []]
            bounds = dict(candidates.get("id_bounds") or {})
            progress(f"checking the road under {len(candidate_ids)} candidate(s).")
            failures = [0, 0]
            original_request = crossing_pass._segment_request

            def local_request(*args, **kwargs):
                # Read the windows from the downloaded copy, not the file server.
                return dataclasses.replace(original_request(*args, **kwargs), video_path=str(local_video))

            crossing_pass._segment_request = local_request
            try:
                selected = crossing_pass.select_road_crossings(
                    self.mapping.df, [task], {stem: candidates} if candidate_ids else {},
                    metrics.tuned_crossing_parameters(), failure_totals=failures,
                )
            finally:
                crossing_pass._segment_request = original_request
            if failures[0]:
                raise RuntimeError(f"The road surface could not be read for the segment at {start} s.")
            counted = [_normalise_id(v) for v in selected.get(stem, [])]
        else:
            counted = [_normalise_id(v) for v in result.get("ids") or []]
            bounds = dict(result.get("id_bounds") or {})

        progress("reading the person boxes.")
        detections, effective_fps, first_source_frame = crossing_pass._prepared_detections(task)
        video_fps = probe_video_fps(str(local_video)) or fps
        clock = FrameClock.for_segment(
            start_seconds=float(start), video_fps=float(video_fps), source_fps=fps,
            detection_fps=float(effective_fps), first_source_frame=float(first_source_frame),
        )
        bounds = {_normalise_id(k): v for k, v in bounds.items()}
        windows = {
            track_id: [round(clock.seconds(bounds[track_id][0]), 3), round(clock.seconds(bounds[track_id][1]), 3)]
            for track_id in counted if track_id in bounds
        }
        payload = {
            "version": SEGMENT_CACHE_VERSION,
            "video_id": video_id,
            "stem": stem,
            "start": segment["start"],
            "end": segment["end"],
            "time_of_day": segment["time_of_day"],
            "city": info["city"],
            "video_fps": video_fps,
            "crossing_rule": rule,
            "counted": counted,
            "counted_windows": windows,
            "tracks": _person_tracks(detections, clock, float(effective_fps)),
            "processed": time.strftime("%Y-%m-%d %H:%M"),
        }
        SEGMENT_DIR.mkdir(parents=True, exist_ok=True)
        path = self.segment_path(video_id, start)
        temporary = path.with_suffix(".json.part")
        temporary.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
        os.replace(temporary, path)


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    mapping: Mapping
    processor: Processor

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        logger.debug(format % args)

    def _send(self, body: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, value: Any, status: int = 200) -> None:
        self._send(json.dumps(value).encode("utf-8"), "application/json", status)

    def _query(self) -> Dict[str, str]:
        return {key: values[0] for key, values in parse_qs(urlsplit(self.path).query).items()}

    @staticmethod
    def _video_id(value: Optional[str]) -> Optional[str]:
        value = (value or "").strip()
        return value if VIDEO_ID_PATTERN.match(value) else None

    @staticmethod
    def _labels(video_id: str) -> Dict[str, Any]:
        path = LABEL_DIR / f"{video_id}.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}

    def do_GET(self) -> None:  # noqa: N802
        path = urlsplit(self.path).path
        query = self._query()
        try:
            if path in ("/", "/index.html"):
                return self._send(HTML_PATH.read_bytes(), "text/html; charset=utf-8")
            if path.startswith("/video/"):
                return self._video(path.rsplit("/", 1)[1])
            video_id = self._video_id(query.get("video_id"))
            if not video_id:
                return self._json({"error": "invalid video id"}, 400)
            info = self.mapping.segments(video_id)
            if info is None:
                return self._json({"error": f"{video_id} is not in the mapping."}, 404)
            if path == "/api/video":
                info.pop("row_id", None)
                info["processing"] = self.processor.state(video_id)
                return self._json(info)
            if path == "/api/segment":
                start = int(query.get("start", "0"))
                if not self.processor.segment_ready(video_id, start):
                    return self._json({"error": "This segment has not been processed."}, 409)
                return self._send(Processor.segment_path(video_id, start).read_bytes(), "application/json")
            if path == "/api/labels":
                return self._json(self._labels(video_id))
            self._json({"error": "not found"}, 404)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as error:
            logger.error(f"{path}: {error}\n{traceback.format_exc()}")
            self._json({"error": str(error)}, 500)

    def do_POST(self) -> None:  # noqa: N802
        path = urlsplit(self.path).path
        length = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            return self._json({"error": "invalid JSON"}, 400)
        video_id = self._video_id(body.get("video_id"))
        if not video_id:
            return self._json({"error": "invalid video id"}, 400)
        if path == "/api/process":
            return self._json(self.processor.start(video_id))
        if path == "/api/labels":
            LABEL_DIR.mkdir(parents=True, exist_ok=True)
            label_path = LABEL_DIR / f"{video_id}.json"
            temporary = label_path.with_suffix(".json.part")
            temporary.write_text(json.dumps(body, indent=1), encoding="utf-8")
            os.replace(temporary, label_path)
            return self._json({"saved": time.strftime("%H:%M:%S")})
        self._json({"error": "not found"}, 404)

    def _video(self, video_id: str) -> None:
        """Serve the downloaded video with range support, so seeking is instant."""
        video_id = self._video_id(video_id)
        path = video_path(video_id) if video_id else None
        if path is None or not path.is_file():
            return self._json({"error": "the video has not been downloaded yet"}, 404)
        size = path.stat().st_size
        first, last = 0, size - 1
        requested = re.match(r"bytes=(\d*)-(\d*)", self.headers.get("Range") or "")
        if requested:
            if requested.group(1):
                first = int(requested.group(1))
                if requested.group(2):
                    last = min(int(requested.group(2)), size - 1)
            elif requested.group(2):
                first = max(0, size - int(requested.group(2)))
        if first > last:
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{size}")
            self.end_headers()
            return
        self.send_response(206 if requested else 200)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(last - first + 1))
        if requested:
            self.send_header("Content-Range", f"bytes {first}-{last}/{size}")
        self.end_headers()
        with path.open("rb") as handle:
            handle.seek(first)
            remaining = last - first + 1
            while remaining > 0:
                chunk = handle.read(min(SERVE_CHUNK_BYTES, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument(
        "--csv-url-path", default="v/data/files/pedestrians_in-youtube/data/bbox/",
        help="Path on the file server (after ftp_base_url) where the detection CSVs are served.",
    )
    parser.add_argument("--no-browser", action="store_true", help="Do not open the page automatically.")
    args = parser.parse_args()
    logs(show_level=common.get_configs("logger_level"), show_color=True)

    Handler.mapping = Mapping()
    csv_url = str(common.get_configs("ftp_base_url")).rstrip("/") + "/" + args.csv_url_path.lstrip("/")
    Handler.processor = Processor(Handler.mapping, csv_url)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    address = f"http://127.0.0.1:{args.port}/"
    logger.info(f"Crossing review tool running at {address} (Ctrl-C to stop).")
    if not args.no_browser:
        webbrowser.open(address)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
