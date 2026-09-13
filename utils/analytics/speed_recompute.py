"""Recompute per-track crossing speed from the stored detection files.

The crossing speed cached in ``results.pickle`` carries a unit: metres per
second when a qualified Waymo metric speed model is installed, and a
dimensionless within-video relative index otherwise.  Installing or removing
that model changes the unit the pipeline would produce, but the cached values
keep whichever unit they were computed with.

This module recomputes those per-track values by replaying the existing
detection files through the same worker ``analysis.py`` uses, so the
preprocessing (confidence filter, duration cap, frame-rate resampling) stays
identical.  Crossing detection is not repeated: the caller supplies the
detection files already recorded in the cache, so the same tracks are analysed
and only the speed unit changes.
"""

import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List, Optional, Set, Tuple

import polars as pl
from tqdm import tqdm

import common
from custom_logger import CustomLogger
from utils.analytics.csv_parallel import initialise_csv_worker, process_csv_task
from utils.analytics.io import IO
from utils.analytics.parquet_store import configured_parquet_roots
from utils.core.metadata import MetaData
from utils.crossing import metrics as crossing_metrics_module

logger = CustomLogger(__name__)
analytics_IO = IO()
metadata = MetaData()

DETECTION_FOLDER = "bbox"
MISC_FILES = {"summary", "_combined", "combined"}


def metric_speed_unit() -> str:
    """Return the unit the installed Waymo model would currently produce."""
    return "m/s" if crossing_metrics_module.metric_speed_is_qualified() else "relative"


def build_detection_tasks(
    df_mapping: pl.DataFrame,
    wanted: Optional[Set[str]] = None,
) -> List[Dict[str, Any]]:
    """Rebuild the detection task list exactly as analysis.py builds it."""
    id_to_place = {
        int(row_id): (locality, state, country)
        for row_id, locality, state, country in df_mapping.select(
            ["id", "locality", "state", "country"]
        ).iter_rows()
    }
    segment_lookup = metadata.segment_lookup(df_mapping)

    tasks: List[Dict[str, Any]] = []
    for folder_path in configured_parquet_roots():
        bbox_path = os.path.join(folder_path, DETECTION_FOLDER)
        if not os.path.exists(bbox_path):
            logger.warning(f"Folder does not exist: {bbox_path}.")
            continue

        for file_name in tqdm(
            analytics_IO.parquet_detection_files(bbox_path),
            desc=f"Indexing detection files in {bbox_path}",
        ):
            filtered = analytics_IO.filter_detection_file(
                file=file_name,
                df_mapping=df_mapping,
            )
            if filtered is None:
                continue
            file_str = os.fspath(filtered)
            if file_str in MISC_FILES:
                continue
            filename_no_ext = os.path.splitext(file_str)[0]
            if wanted and filename_no_ext not in wanted:
                continue
            try:
                video_id, start_index_text, fps_text = filename_no_ext.rsplit("_", 2)
                start_index = int(start_index_text)
                fps = float(fps_text)
            except (TypeError, ValueError):
                logger.warning(f"Unexpected filename format: {filename_no_ext}")
                continue

            segment_meta = segment_lookup.get((video_id, start_index))
            video_locality_id = segment_meta[0] if segment_meta is not None else None
            time_video = float(segment_meta[1]) if segment_meta is not None else 0.0
            if video_locality_id is None:
                continue
            if id_to_place.get(int(video_locality_id)) is None:
                continue

            tasks.append(
                {
                    "file_path": os.path.join(bbox_path, file_str),
                    "file_name": file_str,
                    "filename_no_ext": filename_no_ext,
                    "video_id": video_id,
                    "start_index": start_index,
                    "fps": fps,
                    "video_locality_id": int(video_locality_id),
                    "time_video": time_video,
                    "is_bbox_stream": True,
                }
            )
    return tasks


def resolve_worker_count(requested: Optional[int], task_count: int) -> int:
    """Choose a worker count the same way analysis.py does."""
    if requested:
        workers = int(requested)
    else:
        worker_env = os.environ.get("CROWD_CSV_WORKERS", "").strip()
        if worker_env:
            try:
                workers = max(1, int(worker_env))
            except ValueError:
                workers = int(common.get_configs("cpu_worker") or 1)
        else:
            workers = int(common.get_configs("cpu_worker") or 1)
    workers = min(workers, max(1, os.cpu_count() or 1))
    return max(1, min(workers, task_count or 1))


def recompute_all_speed(
    df_mapping: pl.DataFrame,
    wanted: Optional[Set[str]] = None,
    workers: Optional[int] = None,
    tasks: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[Dict[str, Any], int, int]:
    """Recompute ``all_speed`` from the detection files.

    Returns the nested ``{locality: {source: {track: speed}}}`` mapping, the
    number of track values produced, and the number of failed detection files.
    """
    if tasks is None:
        tasks = build_detection_tasks(df_mapping, wanted)
    if not tasks:
        logger.error("No detection tasks were built; crossing speed not recomputed.")
        return {}, 0, 0

    worker_count = resolve_worker_count(workers, len(tasks))
    worker_init_args = (
        df_mapping,
        dict(crossing_metrics_module.tuned_crossing_parameters()),
        float(common.get_configs("min_confidence")),
        float(common.get_configs("boundary_left")),
        float(common.get_configs("boundary_right")),
        dict(getattr(crossing_metrics_module, "_PIPELINE_MODEL", {}) or {}),
        dict(getattr(crossing_metrics_module, "_SPEED_MODEL", {}) or {}),
    )

    all_speed: Dict[str, Any] = {}
    failures = 0
    tracks = 0

    def merge(result: Dict[str, Any]) -> None:
        nonlocal failures, tracks
        if result.get("status") != "ok":
            failures += 1
            logger.error(f"Worker failure: {result.get('message')}")
            return
        speed_value = result.get("speed_value")
        if not speed_value:
            return
        for outer_key, inner_dict in speed_value.items():
            all_speed.setdefault(outer_key, {}).update(inner_dict)
            for per_video in inner_dict.values():
                tracks += len(per_video)

    logger.info(
        "Recomputing crossing speed from {} detection files with {} workers.",
        len(tasks), worker_count,
    )
    if worker_count == 1:
        initialise_csv_worker(*worker_init_args)
        for task in tqdm(tasks, desc="Recomputing crossing speed"):
            merge(process_csv_task(task))
    else:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=context,
            initializer=initialise_csv_worker,
            initargs=worker_init_args,
        ) as executor:
            for result in tqdm(
                executor.map(process_csv_task, tasks, chunksize=1),
                total=len(tasks),
                desc="Recomputing crossing speed",
            ):
                merge(result)

    logger.info(
        "Recomputed {} crossing speed values across {} localities ({} failures).",
        tracks, len(all_speed), failures,
    )
    return all_speed, tracks, failures
