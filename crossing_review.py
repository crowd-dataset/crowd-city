"""Local web tool for checking crossing detection by hand, one video at a time.

Run ``python crossing_review.py`` and open the address it prints. Enter a
video id; the page lists the video's detection segments from the mapping, plays
the footage from the file server, and draws every YOLO person box with what the
crossing algorithm decided about it. The reviewer sorts what they see into four
counts:

1. a pedestrian crossed but YOLO never detected them (marked by clicking the
   spot in the video, since there is no box to click),
2. YOLO detected a crossing pedestrian but the algorithm did not count it,
3. the algorithm counted a crossing and it is real,
4. the algorithm counted a crossing that is not real (a fake crossing).

A segment is analysed on demand with the same code and configuration as
analysis.py (detection worker, then the crossing rule in ``crossing_rule``), and
the result is cached in ``_output/crossing_review/segments``. Labels are saved
after every change in ``_output/crossing_review/labels/<video_id>.json``.

The video is streamed through this server, which adds the file-server
credentials from the secrets file, so they never reach the browser. Detection
files missing from the local Parquet store are downloaded from the file server's
``data`` alias (see --csv-url-path) and converted, like a sync would.
"""

from __future__ import annotations

import argparse
import json
import os
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
    """The mapping rows that mention each video, parsed on first use."""

    def __init__(self) -> None:
        self.df = pl.read_csv(common.get_configs("mapping"), infer_schema_length=0)
        self._by_video: Dict[str, Tuple[Dict[str, Any], int]] = {}
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
        folder = Path(root) / "bbox"
        for fps in FPS_GUESSES:
            candidate = folder / f"{video_id}_{start}_{fps}.parquet"
            if candidate.is_file():
                return candidate
        matches = sorted(folder.glob(f"{video_id}_{start}_*.parquet"))
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


# ---------------------------------------------------------------------------
# Running the algorithm on one segment
# ---------------------------------------------------------------------------

class Analyser:
    """Runs the detection worker and the crossing rule on one segment at a time."""

    def __init__(self, mapping: Mapping, csv_url: str) -> None:
        self.mapping = mapping
        self.csv_url = csv_url
        self.lock = threading.Lock()
        self.status: Dict[str, Dict[str, Any]] = {}
        self._ready = False

    def _initialise(self) -> None:
        if self._ready:
            return
        import utils.analytics.csv_parallel as csv_parallel
        import utils.crossing.metrics as metrics

        # The review covers whole segments, so the analysis-wide footage cap
        # and crossing target must not trim or skip any of them. This changes
        # only this process's copy of the configuration.
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

    @staticmethod
    def cache_path(video_id: str, start: int) -> Path:
        return SEGMENT_DIR / f"{video_id}_{start}.json"

    def state(self, video_id: str, start: int) -> Dict[str, Any]:
        key = f"{video_id}_{start}"
        if self.cache_path(video_id, start).is_file():
            return {"status": "ready"}
        return self.status.get(key, {"status": "not_analysed"})

    def start(self, video_id: str, start: int) -> Dict[str, Any]:
        key = f"{video_id}_{start}"
        current = self.state(video_id, start)
        if current["status"] in ("ready", "queued", "running"):
            return current
        self.status[key] = {"status": "queued", "message": "Waiting for the previous segment to finish."}
        threading.Thread(target=self._run, args=(video_id, start), daemon=True).start()
        return self.status[key]

    def _progress(self, key: str, message: str) -> None:
        self.status[key] = {"status": "running", "message": message}
        logger.info(f"{key}: {message}")

    def _run(self, video_id: str, start: int) -> None:
        key = f"{video_id}_{start}"
        with self.lock:
            try:
                self._analyse(video_id, start, key)
            except Exception as error:
                logger.error(f"{key}: {error}\n{traceback.format_exc()}")
                self.status[key] = {"status": "error", "message": str(error)}

    def _analyse(self, video_id: str, start: int, key: str) -> None:
        import utils.analytics.csv_parallel as csv_parallel
        import utils.crossing.metrics as metrics
        import utils.segmentation.crossing_pass as crossing_pass
        from utils.segmentation.frames import FrameClock, probe_video_fps, resolve_video_url

        self._progress(key, "Loading the crossing model.")
        self._initialise()
        info = self.mapping.segments(video_id)
        segment = next((s for s in (info or {}).get("segments", []) if s["start"] == start), None)
        if segment is None:
            raise RuntimeError(f"Segment starting at {start}s of {video_id} is not in the mapping.")

        self._progress(key, "Finding the detection file.")
        parquet = _local_parquet(video_id, start)
        if parquet is None:
            self._progress(key, "Downloading the detection file from the file server.")
            parquet = _download_detection_file(video_id, start, self.csv_url)
        if parquet is None:
            raise RuntimeError("No detection file exists for this segment, locally or on the file server.")
        stem = parquet.stem
        fps = float(stem.rsplit("_", 1)[1])
        task = dict(
            file_path=str(parquet), file_name=parquet.name, filename_no_ext=stem, video_id=video_id,
            start_index=int(start), fps=fps, video_locality_id=int(info["row_id"]),
            time_video=float(segment["end"] - segment["start"]), is_bbox_stream=True,
        )

        self._progress(key, "Running the detection worker.")
        # The segment index remembers which detection files existed when it
        # was built; a file downloaded since then must be seen too.
        from utils.core.metadata import MetaData

        MetaData.clear_video_index_cache()
        result = csv_parallel.process_csv_task(task)
        if result.get("status") != "ok":
            raise RuntimeError(result.get("message") or "The detection worker failed.")

        rule = str(common.get_configs("crossing_rule"))
        candidates = (result.get("road_candidates") or {}) if rule == "road_crossing" else {}
        candidate_ids = [str(v) for v in candidates.get("ids") or []]
        bounds = dict(candidates.get("id_bounds") or result.get("id_bounds") or {})
        if rule == "road_crossing":
            self._progress(
                key,
                f"Segmenting the road under {len(candidate_ids)} candidate track(s); "
                "this reads the video and can take several minutes.",
            )
            failures = [0, 0]
            selected = crossing_pass.select_road_crossings(
                self.mapping.df, [task], {stem: candidates} if candidate_ids else {},
                metrics.tuned_crossing_parameters(), failure_totals=failures,
            )
            if failures[0]:
                raise RuntimeError("The road surface could not be read for this segment (file server or GPU).")
            selected_ids = [str(v) for v in selected.get(stem, [])]
        else:
            selected_ids = [str(v) for v in result.get("ids") or []]

        self._progress(key, "Preparing the boxes for the page.")
        detections, effective_fps, first_source_frame = crossing_pass._prepared_detections(task)
        credentials = _credentials()
        url = resolve_video_url(video_id, credentials)
        video_fps = (probe_video_fps(url, credentials) if url else None) or fps
        clock = FrameClock.for_segment(
            start_seconds=float(start), video_fps=float(video_fps), source_fps=fps,
            detection_fps=float(effective_fps), first_source_frame=float(first_source_frame),
        )
        tracks = _person_tracks(detections, clock, float(effective_fps), set(selected_ids), set(candidate_ids), bounds)
        payload = {
            "video_id": video_id,
            "stem": stem,
            "start": segment["start"],
            "end": segment["end"],
            "time_of_day": segment["time_of_day"],
            "city": info["city"],
            "video_fps": video_fps,
            "crossing_rule": rule,
            "selected_count": len(selected_ids),
            "candidate_count": len(candidate_ids),
            "tracks": tracks,
            "created": time.strftime("%Y-%m-%d %H:%M"),
        }
        SEGMENT_DIR.mkdir(parents=True, exist_ok=True)
        path = self.cache_path(video_id, start)
        temporary = path.with_suffix(".json.part")
        temporary.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
        os.replace(temporary, path)
        self.status[key] = {"status": "ready"}


def _normalise_id(value: Any) -> str:
    text = str(value)
    return text[:-2] if text.endswith(".0") else text


def _person_tracks(
    detections: pl.DataFrame,
    clock,
    detection_fps: float,
    selected: set,
    candidates: set,
    bounds: Dict[Any, Tuple[int, int]],
) -> List[Dict[str, Any]]:
    """Every person track, sampled at SAMPLE_HZ, tagged with the algorithm's decision.

    A tracker id can be reused for an unrelated person later in the segment, so
    for selected and candidate tracks only the frames of the accepted crossing
    carry the decision; the rest of that id is shown as an ordinary person.
    """
    bounds = {_normalise_id(k): v for k, v in bounds.items()}
    people = detections.filter(pl.col("yolo-id").cast(pl.Float64, strict=False) == PERSON_CLASS)
    step = max(1, int(round(detection_fps / SAMPLE_HZ)))
    output: List[Dict[str, Any]] = []
    for track in people.partition_by("unique-id", maintain_order=False):
        track_id = _normalise_id(track.get_column("unique-id")[0])
        ordered = track.sort("frame-count")
        frames = ordered.get_column("frame-count").cast(pl.Int64, strict=False).to_list()
        boxes = ordered.select(["x-center", "y-center", "width", "height"]).cast(pl.Float64, strict=False).rows()
        decision = "selected" if track_id in selected else ("candidate" if track_id in candidates else "person")
        window = bounds.get(track_id) if decision != "person" else None
        parts: Dict[str, List[Tuple[float, Tuple[float, ...]]]] = {}
        last_frame: Dict[str, int] = {}
        for frame, box in zip(frames, boxes):
            inside = window is None or (window[0] <= frame <= window[1])
            part = decision if inside else "person"
            if part in last_frame and frame - last_frame[part] < step:
                continue
            last_frame[part] = frame
            parts.setdefault(part, []).append((round(clock.seconds(frame), 3), tuple(round(v, 4) for v in box)))
        for part, samples in parts.items():
            if len(samples) < 2 and part == "person":
                continue
            output.append({
                "id": track_id if part == decision else f"{track_id}~",
                "decision": part,
                "t": [s[0] for s in samples],
                "b": [list(s[1]) for s in samples],
            })
    output.sort(key=lambda track: track["t"][0])
    return output


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    mapping: Mapping
    analyser: Analyser
    video_urls: Dict[str, str] = {}

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        logger.debug(format % args)

    def _json(self, value: Any, status: int = 200) -> None:
        body = json.dumps(value).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _query(self) -> Dict[str, str]:
        return {key: values[0] for key, values in parse_qs(urlsplit(self.path).query).items()}

    def _video_id(self, value: Optional[str]) -> Optional[str]:
        value = (value or "").strip()
        return value if VIDEO_ID_PATTERN.match(value) else None

    def do_GET(self) -> None:  # noqa: N802
        path = urlsplit(self.path).path
        try:
            if path in ("/", "/index.html"):
                body = HTML_PATH.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif path == "/api/segments":
                self._segments()
            elif path == "/api/segment":
                self._segment()
            elif path == "/api/labels":
                video_id = self._video_id(self._query().get("video_id"))
                if not video_id:
                    return self._json({"error": "invalid video id"}, 400)
                label_path = LABEL_DIR / f"{video_id}.json"
                self._json(json.loads(label_path.read_text(encoding="utf-8")) if label_path.is_file() else {})
            elif path.startswith("/video/"):
                self._video(path.rsplit("/", 1)[1])
            else:
                self._json({"error": "not found"}, 404)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_POST(self) -> None:  # noqa: N802
        path = urlsplit(self.path).path
        length = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            return self._json({"error": "invalid JSON"}, 400)
        if path == "/api/analyse":
            video_id = self._video_id(body.get("video_id"))
            if not video_id:
                return self._json({"error": "invalid video id"}, 400)
            self._json(self.analyser.start(video_id, int(body.get("start"))))
        elif path == "/api/labels":
            video_id = self._video_id(body.get("video_id"))
            if not video_id:
                return self._json({"error": "invalid video id"}, 400)
            LABEL_DIR.mkdir(parents=True, exist_ok=True)
            label_path = LABEL_DIR / f"{video_id}.json"
            temporary = label_path.with_suffix(".json.part")
            temporary.write_text(json.dumps(body, indent=1), encoding="utf-8")
            os.replace(temporary, label_path)
            self._json({"saved": time.strftime("%H:%M:%S")})
        else:
            self._json({"error": "not found"}, 404)

    def _segments(self) -> None:
        video_id = self._video_id(self._query().get("video_id"))
        info = self.mapping.segments(video_id) if video_id else None
        if info is None:
            return self._json({"error": f"{video_id or 'That id'} is not in the mapping."}, 404)
        for segment in info["segments"]:
            segment.update(self.analyser.state(video_id, segment["start"]))
        self._json(info)

    def _segment(self) -> None:
        query = self._query()
        video_id = self._video_id(query.get("video_id"))
        if not video_id:
            return self._json({"error": "invalid video id"}, 400)
        start = int(query.get("start", "0"))
        state = self.analyser.state(video_id, start)
        if state["status"] == "ready":
            body = Analyser.cache_path(video_id, start).read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self._json(state)

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
        "--csv-url-path", default="v/data/files/pedestrians_in-youtube/data/bbox/",
        help="Path on the file server (after ftp_base_url) where the detection CSVs are served.",
    )
    parser.add_argument("--no-browser", action="store_true", help="Do not open the page automatically.")
    args = parser.parse_args()
    logs(show_level=common.get_configs("logger_level"), show_color=True)

    Handler.mapping = Mapping()
    csv_url = str(common.get_configs("ftp_base_url")).rstrip("/") + "/" + args.csv_url_path.lstrip("/")
    Handler.analyser = Analyser(Handler.mapping, csv_url)
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
