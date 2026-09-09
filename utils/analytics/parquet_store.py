"""Persistent Parquet store for CROWD detection CSV files.

Detection CSV files are treated as optional ingestion inputs. When
``sync_parquet_on_start`` is enabled, this module converts any new or updated
CSV from each ``data/bbox`` folder into the corresponding
``parquet_data/bbox`` folder.

Parquet files are persistent: a Parquet file is never deleted merely because
its source CSV is missing. This allows the CSV directories to be emptied after
successful conversion and lets future analyses run entirely from Parquet.

Empty CSV files are treated as valid no detection inputs. They are skipped and
do not make synchronisation fail. Existing Parquet files are never deleted or
overwritten because a source CSV becomes empty.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import polars as pl
from tqdm import tqdm

import common
from custom_logger import CustomLogger


logger = CustomLogger(__name__)

DEFAULT_COMPRESSION = "zstd"
DEFAULT_ROW_GROUP_SIZE = 100_000
DETECTION_FOLDER = "bbox"


def _normalise_roots(value, config_name: str) -> List[str]:
    """Normalise a string or sequence of paths to a non empty list of strings."""
    if isinstance(value, (str, os.PathLike)):
        roots = [os.fspath(value)]
    elif isinstance(value, Sequence):
        roots = [os.fspath(item) for item in value]
    else:
        raise TypeError(
            f"Config value {config_name!r} must be a path or a list of paths."
        )

    roots = [root for root in roots if str(root).strip()]
    if not roots:
        raise ValueError(f"Config value {config_name!r} must not be empty.")
    return roots


def configured_parquet_pairs() -> List[Tuple[str, str]]:
    """Return CSV ingestion roots paired with persistent Parquet roots."""
    source_roots = _normalise_roots(common.get_configs("data"), "data")
    parquet_roots = _normalise_roots(
        common.get_configs("parquet_data"),
        "parquet_data",
    )

    if len(source_roots) != len(parquet_roots):
        raise ValueError(
            "Config values 'data' and 'parquet_data' must contain the same "
            f"number of roots. Got {len(source_roots)} source root(s) and "
            f"{len(parquet_roots)} Parquet root(s)."
        )

    return list(zip(source_roots, parquet_roots))


def configured_parquet_roots() -> List[str]:
    """Return validated persistent Parquet roots in the same order as ``data``."""
    return [parquet_root for _, parquet_root in configured_parquet_pairs()]


def _discover_jobs() -> List[Tuple[str, str]]:
    """Discover CSV files that have a corresponding Parquet target path.

    This function never treats Parquet files without a source CSV as stale or
    orphaned. Such files remain part of the persistent Parquet store.
    """
    jobs: List[Tuple[str, str]] = []

    for source_root, parquet_root in configured_parquet_pairs():
        source_folder = Path(source_root) / DETECTION_FOLDER
        target_folder = Path(parquet_root) / DETECTION_FOLDER
        target_folder.mkdir(parents=True, exist_ok=True)

        if not source_folder.is_dir():
            logger.info(
                f"Detection CSV source folder is unavailable: {source_folder}. "
                "Existing Parquet files will be retained."
            )
            continue

        try:
            entries = list(os.scandir(source_folder))
        except OSError as exc:
            raise RuntimeError(
                f"Could not scan detection CSV source folder {source_folder}: {exc}"
            ) from exc

        for entry in entries:
            if not entry.is_file():
                continue
            if entry.name.startswith(".") or not entry.name.lower().endswith(".csv"):
                continue

            stem = os.path.splitext(entry.name)[0]
            target = target_folder / f"{stem}.parquet"
            jobs.append((entry.path, str(target)))

    jobs.sort(key=lambda item: item[0])
    return jobs


def _count_parquet_files() -> int:
    """Count usable Parquet files currently retained in configured stores."""
    count = 0
    for parquet_root in configured_parquet_roots():
        folder = Path(parquet_root) / DETECTION_FOLDER
        if not folder.is_dir():
            continue
        try:
            entries = os.scandir(folder)
        except OSError:
            continue
        with entries:
            for entry in entries:
                if (
                    entry.is_file()
                    and not entry.name.startswith(".")
                    and entry.name.lower().endswith(".parquet")
                ):
                    count += 1
    return count


def _conversion_state(csv_path: str, parquet_path: str) -> str:
    """Return ``empty``, ``convert``, ``reuse``, or ``missing`` for one CSV."""
    try:
        csv_stat = os.stat(csv_path)
    except OSError:
        return "missing"

    # A zero byte detection CSV contains no detections. It should not abort a
    # many hour store build, and an existing Parquet copy should be preserved.
    if csv_stat.st_size <= 0:
        return "empty"

    try:
        parquet_stat = os.stat(parquet_path)
    except OSError:
        return "convert"

    if parquet_stat.st_size <= 0:
        return "convert"

    if parquet_stat.st_mtime_ns < csv_stat.st_mtime_ns:
        return "convert"

    return "reuse"


def _is_empty_csv_error(exc: Exception) -> bool:
    """Return True when Polars reports that a CSV has no readable content."""
    message = str(exc).strip().lower()
    return "empty csv" in message


def _convert_one(
    csv_path: str,
    parquet_path: str,
    compression: str,
    row_group_size: int,
) -> None:
    """Convert one CSV atomically so analysis never sees a partial Parquet file."""
    target = Path(parquet_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")

    try:
        if temporary.exists():
            temporary.unlink()

        (
            pl.scan_csv(csv_path)
            .sink_parquet(
                str(temporary),
                compression=compression,
                statistics=True,
                row_group_size=row_group_size,
                maintain_order=True,
            )
        )

        if not temporary.exists() or temporary.stat().st_size <= 0:
            raise RuntimeError("Parquet conversion produced an empty output file.")

        os.replace(str(temporary), parquet_path)
    except Exception:
        try:
            if temporary.exists():
                temporary.unlink()
        except OSError:
            pass
        raise


def _sync_on_start_enabled() -> bool:
    """Return whether CSV to Parquet synchronisation should run at startup."""
    value = common.get_configs("sync_parquet_on_start")
    if not isinstance(value, bool):
        raise ValueError("sync_parquet_on_start must be true or false")
    return value


def sync_detection_parquet_store(
    force: bool = False,
    compression: str = DEFAULT_COMPRESSION,
    row_group_size: int = DEFAULT_ROW_GROUP_SIZE,
) -> Dict[str, int]:
    """Add new or updated CSV detections to the persistent Parquet store.

    When ``sync_parquet_on_start`` is false, return immediately without
    discovering CSV files, counting Parquet files, comparing timestamps, or
    converting anything. Existing Parquet files remain untouched and the
    normal analysis continues using the configured Parquet store.

    When enabled, existing up to date Parquet files are reused. A target is
    regenerated when it is missing, empty, older than its source CSV, or
    ``force=True``.

    Empty source CSVs are skipped and reported, not treated as fatal errors.
    Parquet files whose source CSV is absent are deliberately retained. This
    supports a Parquet only workflow after the initial conversion.
    """
    if not _sync_on_start_enabled():
        logger.info(
            "Parquet synchronisation skipped because "
            "sync_parquet_on_start is false."
        )
        return {
            "source_files": 0,
            "converted": 0,
            "reused": 0,
            "skipped_empty": 0,
            "stored_files": 0,
            "errors": 0,
        }

    jobs = _discover_jobs()
    stored_before = _count_parquet_files()

    if not jobs:
        if stored_before:
            logger.info(
                f"No detection CSV files require ingestion. Retaining "
                f"{stored_before} existing Parquet file(s)."
            )
        else:
            logger.warning(
                "No detection CSV files or existing Parquet detection files were found."
            )
        return {
            "source_files": 0,
            "converted": 0,
            "reused": 0,
            "skipped_empty": 0,
            "stored_files": stored_before,
            "errors": 0,
        }

    pending: List[Tuple[str, str]] = []
    reused = 0
    skipped_empty_paths: List[str] = []
    missing_during_scan = 0

    for csv_path, parquet_path in tqdm(
        jobs,
        desc="Checking CSV/Parquet status",
    ):
        state = _conversion_state(csv_path, parquet_path)

        if state == "empty":
            skipped_empty_paths.append(csv_path)
            continue

        if state == "missing":
            # The source may have been removed after directory discovery. Keep
            # any persistent Parquet copy and simply ignore this ingestion item.
            missing_during_scan += 1
            continue

        if force or state == "convert":
            pending.append((csv_path, parquet_path))
        else:
            reused += 1

    if skipped_empty_paths:
        logger.warning(
            f"Skipping {len(skipped_empty_paths)} empty detection CSV file(s). "
            "Existing Parquet copies, if any, are retained."
        )
        for csv_path in skipped_empty_paths[:20]:
            logger.warning(f"Empty detection CSV skipped: {csv_path}")
        if len(skipped_empty_paths) > 20:
            logger.warning(
                f"... and {len(skipped_empty_paths) - 20} more empty CSV file(s)."
            )

    if missing_during_scan:
        logger.info(
            f"{missing_during_scan} CSV file(s) disappeared during discovery and "
            "were ignored; existing Parquet files were retained."
        )

    if not pending:
        stored_files = _count_parquet_files()
        logger.info(
            f"Parquet store is up to date for the available non empty CSV files: "
            f"{reused} corresponding Parquet file(s) reused, "
            f"{len(skipped_empty_paths)} empty CSV file(s) skipped, "
            f"{stored_files} total Parquet file(s) retained."
        )
        return {
            "source_files": len(jobs),
            "converted": 0,
            "reused": reused,
            "skipped_empty": len(skipped_empty_paths),
            "stored_files": stored_files,
            "errors": 0,
        }

    logger.info(
        f"Updating Parquet store: {len(pending)} conversion(s) required, "
        f"{reused} available CSV file(s) already current, "
        f"{len(skipped_empty_paths)} empty CSV file(s) skipped."
    )

    converted = 0
    failures: List[str] = []

    for csv_path, parquet_path in tqdm(
        pending,
        desc="Converting CSV to Parquet",
    ):
        try:
            _convert_one(
                csv_path=csv_path,
                parquet_path=parquet_path,
                compression=compression,
                row_group_size=max(1, int(row_group_size)),
            )
            converted += 1
        except Exception as exc:
            # A nonzero file can still contain only whitespace or otherwise be
            # reported by Polars as an empty CSV. Treat that case exactly like
            # a zero byte source: skip it and preserve any existing Parquet.
            if _is_empty_csv_error(exc):
                skipped_empty_paths.append(csv_path)
                logger.warning(f"Empty detection CSV skipped: {csv_path}")
                continue
            failures.append(f"{csv_path}: {exc}")

    if failures:
        preview = "\n".join(failures[:10])
        extra = len(failures) - 10
        if extra > 0:
            preview += f"\n... and {extra} more conversion error(s)."
        raise RuntimeError(
            "Parquet ingestion failed for "
            f"{len(failures)} CSV file(s):\n{preview}"
        )

    stored_files = _count_parquet_files()
    logger.info(
        f"Parquet store updated: {converted} file(s) converted, "
        f"{reused} available CSV file(s) reused, "
        f"{len(skipped_empty_paths)} empty CSV file(s) skipped, "
        f"{stored_files} total Parquet file(s) retained."
    )

    return {
        "source_files": len(jobs),
        "converted": converted,
        "reused": reused,
        "skipped_empty": len(skipped_empty_paths),
        "stored_files": stored_files,
        "errors": 0,
    }
