"""Local web tool for counting crossings by hand and comparing them with a run.

Run ``python crossing_review.py`` and open the address it prints. Enter a
video id; the page lists the video's segments from the mapping and plays the
footage from the file server. Nothing is drawn on the video and no algorithm
runs while reviewing, so neither can bias the review: the reviewer clicks on
every pedestrian they see crossing the road.

**Compare with run** then sorts each click, using the segment's YOLO person
tracks and the crossings the last analysis.py run counted (its results.pickle):

1. crossing, not detected by YOLO: no YOLO person box at that spot and moment,
2. crossing detected by YOLO, missed by the algorithm: a box is there, but the
   run did not count that track as crossing,
3. crossing detected by YOLO and counted by the algorithm,
4. fake crossing: a crossing the run counted that the reviewer never clicked.

Labels are saved after every change in
``_output/crossing_review/labels/<video_id>.json``. The video is streamed
through this server, which adds the file-server credentials from the secrets
file, so they never reach the browser. Detection files missing from the local
Parquet store are downloaded from the file server's ``data`` alias (see
--csv-url-path) and converted, like a sync would.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import threading
import time
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
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
# Person boxes are sent to the page at this rate and interpolated in between,
# which keeps an hour of busy footage to a few megabytes.
SAMPLE_HZ = 10.0
# Detection file names end in the frame rate, which the mapping does not record.
FPS_GUESSES = (30, 60, 24, 25, 50, 29, 15, 20, 48, 59, 23, 10, 12)
PERSON_CLASS = 0
STREAM_CHUNK_BYTES = 256 * 1024
VIDEO_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{6,20}$")
# Bumped when the cached segment files change shape.
SEGMENT_CACHE_VERSION = 2


# ---------------------------------------------------------------------------
# Mapping and detection files
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
        df = pl.read_csv(common.get_configs("mapping"), infer_schema_length=0)
        self._by_video: Dict[str, Any] = {}
        for row in df.iter_rows(named=True):
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


def _normalise_id(value: Any) -> str:
    text = str(value)
    return text[:-2] if text.endswith(".0") else text


def _person_tracks(parquet: Path, start_seconds: float, video_fps: float, min_confidence: float) -> List[Dict]:
    """Every YOLO person track the analysis would see, sampled at SAMPLE_HZ.

    Boxes below min_confidence are dropped, as in the analysis. Frame numbers
    count from the segment start at the video's own frame rate, the same clock
    the analysis uses when processing_fps is null.
    """
    columns = ["unique-id", "frame-count", "yolo-id", "confidence", "x-center", "y-center", "width", "height"]
    df = pl.read_parquet(parquet, columns=columns)
    people = df.filter(
        (pl.col("yolo-id").cast(pl.Float64, strict=False) == PERSON_CLASS)
        & (pl.col("confidence").cast(pl.Float64, strict=False) >= min_confidence)
        & pl.col("unique-id").is_not_null()
    )
    step = max(1, int(round(video_fps / SAMPLE_HZ)))
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
            times.append(round(start_seconds + frame / video_fps, 3))
            sampled.append([round(v, 4) for v in box])
        if len(times) >= 2:
            tracks.append({"id": _normalise_id(track.get_column("unique-id")[0]), "t": times, "b": sampled})
    tracks.sort(key=lambda item: item["t"][0])
    return tracks


class Loader:
    """Prepares a segment's person boxes in the background, one at a time."""

    def __init__(self, mapping: Mapping, csv_url: str) -> None:
        self.mapping = mapping
        self.csv_url = csv_url
        self.lock = threading.Lock()
        self.status: Dict[str, Dict[str, Any]] = {}

    @staticmethod
    def cache_path(video_id: str, start: int) -> Path:
        return SEGMENT_DIR / f"{video_id}_{start}.json"

    def state(self, video_id: str, start: int) -> Dict[str, Any]:
        path = self.cache_path(video_id, start)
        if path.is_file():
            try:
                if json.loads(path.read_text(encoding="utf-8")).get("version") == SEGMENT_CACHE_VERSION:
                    return {"status": "ready"}
            except (OSError, json.JSONDecodeError):
                pass
        return self.status.get(f"{video_id}_{start}", {"status": "not_loaded"})

    def start(self, video_id: str, start: int) -> Dict[str, Any]:
        key = f"{video_id}_{start}"
        current = self.state(video_id, start)
        if current["status"] in ("ready", "queued", "running"):
            return current
        self.status[key] = {"status": "queued", "message": "Waiting for another segment to finish loading."}
        threading.Thread(target=self._run, args=(video_id, start), daemon=True).start()
        return self.status[key]

    def _run(self, video_id: str, start: int) -> None:
        key = f"{video_id}_{start}"
        with self.lock:
            try:
                self._load(video_id, start, key)
            except Exception as error:
                logger.error(f"{key}: {error}\n{traceback.format_exc()}")
                self.status[key] = {"status": "error", "message": str(error)}

    def _load(self, video_id: str, start: int, key: str) -> None:
        from utils.segmentation.frames import probe_video_fps, resolve_video_url

        info = self.mapping.segments(video_id)
        segment = next((s for s in (info or {}).get("segments", []) if s["start"] == start), None)
        if segment is None:
            raise RuntimeError(f"Segment starting at {start}s of {video_id} is not in the mapping.")
        self.status[key] = {"status": "running", "message": "Finding the detection file."}
        parquet = _local_parquet(video_id, start)
        if parquet is None:
            self.status[key] = {"status": "running", "message": "Downloading the detection file from the file server."}
            parquet = _download_detection_file(video_id, start, self.csv_url)
        if parquet is None:
            raise RuntimeError("No detection file exists for this segment, locally or on the file server.")
        self.status[key] = {"status": "running", "message": "Reading the person boxes."}
        credentials = _credentials()
        url = resolve_video_url(video_id, credentials)
        nominal_fps = float(parquet.stem.rsplit("_", 1)[1])
        video_fps = (probe_video_fps(url, credentials) if url else None) or nominal_fps
        tracks = _person_tracks(parquet, float(start), float(video_fps), float(common.get_configs("min_confidence")))
        payload = {
            "version": SEGMENT_CACHE_VERSION,
            "video_id": video_id,
            "stem": parquet.stem,
            "start": segment["start"],
            "end": segment["end"],
            "time_of_day": segment["time_of_day"],
            "city": info["city"],
            "video_fps": video_fps,
            "tracks": tracks,
        }
        SEGMENT_DIR.mkdir(parents=True, exist_ok=True)
        path = self.cache_path(video_id, start)
        temporary = path.with_suffix(".json.part")
        temporary.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
        os.replace(temporary, path)
        self.status[key] = {"status": "ready"}


# ---------------------------------------------------------------------------
# Comparison with an analysis run
# ---------------------------------------------------------------------------

# A click matches a YOLO person box shown within this many seconds of it, so a
# box missing for a frame or two does not turn a detected person into a miss.
MATCH_WINDOW_SECONDS = 0.5
# Boxes are enlarged by this fraction of their size when testing a click.
MATCH_MARGIN = 0.15
# A run's crossing is the track's frames between these bounds; the tracker can
# reuse an id later for someone else, so a click must fall near them.
BOUNDS_SLACK_SECONDS = 2.0
# Two clicks on the same track this close together are the same person.
DUPLICATE_SECONDS = 15.0


def _box_near(track: Dict[str, Any], time_s: float) -> Optional[List[float]]:
    """The track's box closest in time to ``time_s``, if within MATCH_WINDOW_SECONDS."""
    times = track["t"]
    if time_s < times[0] - MATCH_WINDOW_SECONDS or time_s > times[-1] + MATCH_WINDOW_SECONDS:
        return None
    import bisect

    index = bisect.bisect_left(times, time_s)
    candidates = [i for i in (index - 1, index) if 0 <= i < len(times)]
    best = min(candidates, key=lambda i: abs(times[i] - time_s))
    return track["b"][best] if abs(times[best] - time_s) <= MATCH_WINDOW_SECONDS else None


def match_click(tracks: List[Dict[str, Any]], mark: Dict[str, float]) -> Optional[Dict[str, Any]]:
    """Return the YOLO person track under a click, or None when there is none."""
    best, best_score = None, None
    for track in tracks:
        box = _box_near(track, float(mark["t"]))
        if box is None:
            continue
        cx, cy, width, height = box
        dx = abs(float(mark["x"]) - cx) / max(width * (0.5 + MATCH_MARGIN), 1e-6)
        dy = abs(float(mark["y"]) - cy) / max(height * (0.5 + MATCH_MARGIN), 1e-6)
        if dx <= 1 and dy <= 1:
            score = dx * dx + dy * dy
            if best_score is None or score < best_score:
                best, best_score = track, score
    return best


class RunResults:
    """The crossings an analysis.py run counted, per detection segment."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._mtime = None
        self._by_segment: Dict[str, Dict[str, Any]] = {}
        self.description = ""

    def _refresh(self) -> None:
        mtime = self.path.stat().st_mtime if self.path.is_file() else None
        if mtime == self._mtime:
            return
        self._mtime, self._by_segment = mtime, {}
        if mtime is None:
            self.description = f"{self.path} does not exist"
            return
        with self.path.open("rb") as handle:
            results = pickle.load(handle)
        counts = results[11] if isinstance(results, tuple) and len(results) > 11 else {}
        for stem, payload in (counts or {}).items():
            video, start, _fps = str(stem).rsplit("_", 2)
            payload = payload or {}
            bounds = {_normalise_id(k): v for k, v in (payload.get("id_bounds") or {}).items()}
            self._by_segment[f"{video}_{int(start)}"] = {
                "ids": [_normalise_id(value) for value in payload.get("ids") or []],
                "bounds": bounds,
            }
        settings = results[-1].get("config", {}) if isinstance(results[-1], dict) else {}
        written = time.strftime("%Y-%m-%d %H:%M", time.localtime(mtime))
        self.description = (
            f"run of {written}, rule {settings.get('crossing_rule', '?')}, "
            f"footage cap {settings.get('max_footage_hours_per_city', 'none')} h per city"
        )

    def compare(self, video_id: str, labels: Dict[str, Any], segments: List[Dict[str, Any]]) -> Dict[str, Any]:
        self._refresh()
        output: Dict[str, Any] = {"run": self.description, "segments": {}}
        for segment in segments:
            key = str(segment["start"])
            lab = (labels.get("segments") or {}).get(key) or {}
            marks = lab.get("crossings") or []
            run = self._by_segment.get(f"{video_id}_{segment['start']}")
            cache = Loader.cache_path(video_id, segment["start"])
            data = json.loads(cache.read_text(encoding="utf-8")) if cache.is_file() else None
            if not marks and run is None:
                continue
            entry: Dict[str, Any] = {"reviewed": bool(lab.get("reviewed")), "in_run": run is not None,
                                     "marks": [], "fake": []}
            tracks = (data or {}).get("tracks") or []
            by_id = {track["id"]: track for track in tracks}
            fps = float((data or {}).get("video_fps") or 30.0)
            start_s = float(segment["start"])

            def counted_window(track_id: str) -> Optional[Tuple[float, float]]:
                frames = (run or {}).get("bounds", {}).get(track_id)
                if frames is None:
                    return None
                return start_s + frames[0] / fps, start_s + frames[1] / fps

            claimed, matched_at = set(), {}
            for mark in sorted(marks, key=lambda item: float(item["t"])):
                where = {"t": mark["t"], "x": mark["x"], "y": mark["y"]}
                track = match_click(tracks, mark)
                if track is None:
                    entry["marks"].append({**where, "category": 1, "track": None})
                    continue
                previous = matched_at.get(track["id"])
                if previous is not None and float(mark["t"]) - previous < DUPLICATE_SECONDS:
                    # The same person clicked twice; category 0 is not counted.
                    # (Further apart, the tracker may have reused the id.)
                    entry["marks"].append({**where, "category": 0, "track": track["id"]})
                    continue
                matched_at[track["id"]] = float(mark["t"])
                counted = run is not None and track["id"] in run["ids"]
                window = counted_window(track["id"])
                if counted and window is not None:
                    counted = window[0] - BOUNDS_SLACK_SECONDS <= float(mark["t"]) <= window[1] + BOUNDS_SLACK_SECONDS
                if counted:
                    claimed.add(track["id"])
                entry["marks"].append({**where, "category": 3 if counted else 2, "track": track["id"]})
            if run is not None:
                for track_id in run["ids"]:
                    if track_id in claimed:
                        continue
                    window = counted_window(track_id)
                    track = by_id.get(track_id)
                    entry["fake"].append({
                        "track": track_id,
                        "t": window[0] if window else (track["t"][0] if track else start_s),
                        "t_end": window[1] if window else (track["t"][-1] if track else start_s),
                    })
            entry["counts"] = {
                "yolo": sum(1 for m in entry["marks"] if m["category"] == 1),
                "algo": sum(1 for m in entry["marks"] if m["category"] == 2),
                "real": sum(1 for m in entry["marks"] if m["category"] == 3),
                "fake": len(entry["fake"]) if run is not None else None,
            }
            output["segments"][key] = entry
        return output


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    mapping: Mapping
    loader: Loader
    run: RunResults
    video_urls: Dict[str, str] = {}

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
            if path == "/api/segments":
                for segment in info["segments"]:
                    segment.update(self.loader.state(video_id, segment["start"]))
                return self._json(info)
            if path == "/api/segment":
                start = int(query.get("start", "0"))
                state = self.loader.state(video_id, start)
                if state["status"] == "ready":
                    return self._send(Loader.cache_path(video_id, start).read_bytes(), "application/json")
                return self._json(state)
            if path == "/api/labels":
                return self._json(self._labels(video_id))
            if path == "/api/compare":
                return self._json(self.run.compare(video_id, self._labels(video_id), info["segments"]))
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
        if path == "/api/load":
            return self._json(self.loader.start(video_id, int(body.get("start"))))
        if path == "/api/labels":
            LABEL_DIR.mkdir(parents=True, exist_ok=True)
            label_path = LABEL_DIR / f"{video_id}.json"
            temporary = label_path.with_suffix(".json.part")
            temporary.write_text(json.dumps(body, indent=1), encoding="utf-8")
            os.replace(temporary, label_path)
            return self._json({"saved": time.strftime("%H:%M:%S")})
        self._json({"error": "not found"}, 404)

    def _video(self, video_id: str) -> None:
        from utils.segmentation.frames import resolve_video_url

        video_id = self._video_id(video_id)
        if not video_id:
            return self._json({"error": "invalid video id"}, 400)
        credentials = _credentials()
        url = self.video_urls.get(video_id) or resolve_video_url(video_id, credentials)
        if not url:
            return self._json({"error": "video not found on the file server"}, 404)
        self.video_urls[video_id] = url
        headers = {"Range": self.headers["Range"]} if self.headers.get("Range") else {}
        upstream = _session(credentials).get(
            url, params=credentials.request_params(), headers=headers, stream=True, timeout=60,
        )
        try:
            self.send_response(upstream.status_code)
            for name in ("Content-Type", "Content-Length", "Content-Range", "Accept-Ranges"):
                if upstream.headers.get(name):
                    self.send_header(name, upstream.headers[name])
            if not upstream.headers.get("Accept-Ranges"):
                self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            for chunk in upstream.iter_content(STREAM_CHUNK_BYTES):
                self.wfile.write(chunk)
        finally:
            upstream.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument(
        "--results", type=Path, default=ROOT / "results.pickle",
        help="results.pickle of the analysis.py run to compare the marks with.",
    )
    parser.add_argument(
        "--csv-url-path", default="v/data/files/pedestrians_in-youtube/data/bbox/",
        help="Path on the file server (after ftp_base_url) where the detection CSVs are served.",
    )
    parser.add_argument("--no-browser", action="store_true", help="Do not open the page automatically.")
    args = parser.parse_args()
    logs(show_level=common.get_configs("logger_level"), show_color=True)

    Handler.mapping = Mapping()
    csv_url = str(common.get_configs("ftp_base_url")).rstrip("/") + "/" + args.csv_url_path.lstrip("/")
    Handler.loader = Loader(Handler.mapping, csv_url)
    Handler.run = RunResults(args.results)
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
