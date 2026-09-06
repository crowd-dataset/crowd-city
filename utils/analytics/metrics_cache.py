"""
metrics_cache.py

Purpose
-------
Compute and cache per-video detection metrics derived from YOLO CSV outputs, then wrap/aggregate
those metrics via Grouping.locality_country_wrapper() using the mapping dataframe.

Key design points
-----------------
- Uses a class-level cache so multiple downstream calls do not re-scan the filesystem or re-read CSVs.
- Captures small per-file aggregate counts when analysis.py reads each detection CSV for the first time.
- Reuses those aggregates later so the normal analysis path does not read the same CSV files twice.
- Falls back to reading a CSV when it was not observed during the main analysis pass.
- Indexes CSV files once per compute run for fast lookup by filename/prefix.
- Computes rates per minute for most object classes.
- Computes a specialised cellphones-per-person normalised measure.

Compatibility / linting
-----------------------
- Avoids Python 3.10-only typing features such as:
    - typing.TypeAlias
    - PEP 604 unions: X | Y
  and avoids PEP 585 generics such as list[str], set[str], dict[str, str] which can trigger
  mypy/pylint issues depending on configured python_version.
"""

import ast
import os
import re
from typing import Any, ClassVar, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

import polars as pl
from tqdm import tqdm

import common
from custom_logger import CustomLogger
from utils.core.grouping import Grouping
from utils.core.metadata import MetaData

# ------------------------------------------------------------------------------
# Constants (YOLO class IDs) used by the project
# ------------------------------------------------------------------------------
YOLO_PERSON = 0
YOLO_BICYCLE = 1
YOLO_CAR = 2
YOLO_MOTORCYCLE = 3
YOLO_BUS = 5
YOLO_TRUCK = 7
YOLO_TRAFFIC_SIGN_IDS = (9, 11)
YOLO_CELLPHONE = 67

# ------------------------------------------------------------------------------
# Type aliases (kept compatible with older Python tooling)
# ------------------------------------------------------------------------------
UniqueValue = Union[str, Tuple[str, ...]]
UniqueValues = Set[UniqueValue]
FileMetricCounts = Dict[str, int]

# ------------------------------------------------------------------------------
# Shared helpers (external dependencies)
# ------------------------------------------------------------------------------
_METADATA = MetaData()
_GROUPING = Grouping()
_LOGGER = CustomLogger(__name__)


class MetricsCache:
    """
    Compute and cache metrics derived from YOLO detection CSVs.

    Public entrypoints
    ------------------
    - calculate_cellphones(df_mapping)
    - calculate_traffic_signs(df_mapping)
    - calculate_traffic(df_mapping, ...flags...)
    - get_unique_values(df, value, ...)
    - clear_cache()

    Cache format
    ------------
    _all_metrics_cache is a dictionary mapping metric name -> wrapped output from Grouping.locality_country_wrapper().
    Example keys:
      "cellphones", "traffic_signs", "vehicles", "bicycles", "cars", "motorcycles", "buses", "trucks", "persons"

    _file_metrics_cache stores only compact aggregate counts per CSV filename. It does not retain
    the full detection DataFrame, so using the single-pass path has a small memory footprint.
    """

    # Class-level cache: avoids recomputation across calls within the same process.
    _all_metrics_cache: ClassVar[Dict[str, Any]] = {}

    # Compact per-file metrics captured during the first pl.read_csv() call in analysis.py.
    _file_metrics_cache: ClassVar[Dict[str, FileMetricCounts]] = {}

    # Keep the original Polars reader so the fallback path can bypass the observer.
    _original_read_csv: ClassVar[Any] = pl.read_csv
    _read_observer_installed: ClassVar[bool] = False
    _capture_min_confidence: ClassVar[Optional[float]] = None
    _capture_roots: ClassVar[Tuple[str, ...]] = ()

    def __init__(self) -> None:
        self._install_read_observer()

    # --------------------------------------------------------------------------
    # Single-pass CSV observation
    # --------------------------------------------------------------------------
    @classmethod
    def _install_read_observer(cls) -> None:
        """Observe detection CSV reads and cache only the aggregate counts needed later.

        analysis.py imports Polars as a module and calls ``pl.read_csv``. Polars modules are shared
        within the Python process, so installing this light wrapper once lets MetricsCache see the
        already-loaded detection DataFrame during the main analysis pass. The returned DataFrame is
        unchanged.
        """
        if cls._read_observer_installed:
            return

        try:
            cls._capture_min_confidence = float(common.get_configs("min_confidence"))
        except Exception:
            cls._capture_min_confidence = None

        try:
            data_folders = common.get_configs("data") or []
            subfolders = common.get_configs("sub_domain") or []
            cls._capture_roots = tuple(
                os.path.realpath(os.path.join(os.fspath(folder), os.fspath(subfolder)))
                for folder in data_folders
                for subfolder in subfolders
            )
        except Exception:
            cls._capture_roots = ()

        original_read_csv = cls._original_read_csv

        def observed_read_csv(source, *args, **kwargs):
            dataframe = original_read_csv(source, *args, **kwargs)
            try:
                cls._record_loaded_dataframe(source, dataframe)
            except Exception as exc:
                # Metrics observation must never make an otherwise valid CSV read fail.
                _LOGGER.debug("Could not capture metrics from %s: %s", source, exc)
            return dataframe

        pl.read_csv = observed_read_csv
        cls._read_observer_installed = True

    @classmethod
    def _record_loaded_dataframe(cls, source: Any, dataframe: pl.DataFrame) -> None:
        """Cache aggregate detection counts for a CSV that has already been read.

        Non-detection CSV files, such as mapping.csv, are ignored automatically because they do not
        contain the required YOLO columns.
        """
        try:
            source_path = os.fspath(source)
        except TypeError:
            return

        if not isinstance(source_path, str) or not source_path.lower().endswith(".csv"):
            return

        source_realpath = os.path.realpath(source_path)
        if cls._capture_roots:
            inside_data_root = False
            for root in cls._capture_roots:
                try:
                    if os.path.commonpath([source_realpath, root]) == root:
                        inside_data_root = True
                        break
                except ValueError:
                    continue
            if not inside_data_root:
                return

        required_cols = {"confidence", "yolo-id", "unique-id"}
        if not required_cols.issubset(set(dataframe.columns)):
            return

        min_conf = cls._capture_min_confidence
        if min_conf is None:
            try:
                min_conf = float(common.get_configs("min_confidence"))
            except Exception:
                return
            cls._capture_min_confidence = min_conf

        filtered = dataframe.filter(
            pl.col("confidence").cast(pl.Float64, strict=False) >= float(min_conf)
        )
        cls._file_metrics_cache[os.path.basename(source_path)] = cls._metric_counts_from_dataframe(filtered)

    @staticmethod
    def _empty_metric_counts() -> FileMetricCounts:
        """Return zero counts for all metrics expected by the aggregation path."""
        return {
            "persons": 0,
            "cellphones": 0,
            "traffic_signs": 0,
            "vehicles": 0,
            "bicycles": 0,
            "cars": 0,
            "motorcycles": 0,
            "buses": 0,
            "trucks": 0,
        }

    @classmethod
    def _metric_counts_from_dataframe(cls, df: pl.DataFrame) -> FileMetricCounts:
        """Compute all required unique-object counts in one Polars aggregation.

        The aggregate definitions intentionally match the previous implementation. In particular,
        traffic-sign and vehicle counts use ``n_unique`` across the combined class sets rather than
        summing per-class counts, so tracker-ID overlap retains the existing behaviour.
        """
        required = {"yolo-id", "unique-id"}
        if df.height == 0 or not required.issubset(set(df.columns)):
            return cls._empty_metric_counts()

        working = df.with_columns(
            pl.col("yolo-id").cast(pl.Int64, strict=False).alias("_metric_yolo_id")
        )
        uid = pl.col("unique-id")
        yolo = pl.col("_metric_yolo_id")

        result = working.select([
            uid.filter(yolo == YOLO_PERSON).n_unique().alias("persons"),
            uid.filter(yolo == YOLO_CELLPHONE).n_unique().alias("cellphones"),
            uid.filter(yolo.is_in(list(YOLO_TRAFFIC_SIGN_IDS))).n_unique().alias("traffic_signs"),
            uid.filter(yolo.is_in([YOLO_CAR, YOLO_MOTORCYCLE, YOLO_BUS, YOLO_TRUCK])).n_unique().alias("vehicles"),
            uid.filter(yolo == YOLO_BICYCLE).n_unique().alias("bicycles"),
            uid.filter(yolo == YOLO_CAR).n_unique().alias("cars"),
            uid.filter(yolo == YOLO_MOTORCYCLE).n_unique().alias("motorcycles"),
            uid.filter(yolo == YOLO_BUS).n_unique().alias("buses"),
            uid.filter(yolo == YOLO_TRUCK).n_unique().alias("trucks"),
        ])

        return {
            name: int(result.item(0, name) or 0)
            for name in cls._empty_metric_counts()
        }

    # --------------------------------------------------------------------------
    # CSV indexing and parsing helpers
    # --------------------------------------------------------------------------
    @staticmethod
    def _parse_videos_cell(videos_cell: Optional[str]) -> List[str]:
        """
        Extract video IDs robustly from the mapping cell content.

        The mapping file may store values like:
          [abc]
          "[abc,def]"
          ["a","b"]
          []
        We treat any token matching [A-Za-z0-9_-]+ as an ID component.
        """
        if not isinstance(videos_cell, str):
            return []
        return re.findall(r"[\w-]+", videos_cell)

    @staticmethod
    def _index_csv_files(data_folders: Sequence[str], subfolders: Sequence[str]) -> Dict[str, str]:
        """
        Build a filename -> full_path index for detection CSVs.

        This allows O(1) lookup once we know the exact filename, and quick prefix checks when only
        vid/start_time is known.
        """
        csv_index: Dict[str, str] = {}
        for folder_path in data_folders:
            for sub in subfolders:
                sub_path = os.path.join(folder_path, sub)
                if not os.path.exists(sub_path):
                    continue

                for fname in os.listdir(sub_path):
                    if fname.endswith(".csv"):
                        csv_index[fname] = os.path.join(sub_path, fname)

        return csv_index

    @staticmethod
    def _extract_fps_from_filename(vid: str, start_time: str, filename: str) -> Optional[int]:
        """
        Extract FPS from filenames matching:
            {vid}_{start_time}_{fps}.csv

        Returns:
            int FPS if pattern matches, otherwise None.
        """
        pattern = r"^%s_%s_(\d+)\.csv$" % (re.escape(str(vid)), re.escape(str(start_time)))
        match = re.match(pattern, filename)
        if not match:
            return None
        return int(match.group(1))

    @staticmethod
    def _count_unique_objects(df: pl.DataFrame, yolo_ids: Iterable[int]) -> int:
        """Count unique object IDs for compatibility with callers outside the fast path."""
        required = {"yolo-id", "unique-id"}
        if not required.issubset(set(df.columns)):
            return 0

        filtered = df.filter(pl.col("yolo-id").cast(pl.Int64, strict=False).is_in(list(yolo_ids)))
        if filtered.height == 0:
            return 0

        return int(filtered.select(pl.col("unique-id").n_unique()).item())

    # --------------------------------------------------------------------------
    # Core compute path
    # --------------------------------------------------------------------------
    @classmethod
    def _compute_all_metrics(cls, df_mapping: pl.DataFrame) -> None:
        """
        Compute and cache all metrics for the provided mapping DataFrame.

        The normal path reuses aggregate counts captured when analysis.py first loaded each detection
        CSV. A disk read is performed only for files that were not observed during that first pass.
        """
        # 1) Build an index of available detection CSVs
        data_folders = common.get_configs("data")
        subfolders = common.get_configs("sub_domain")
        csv_files = cls._index_csv_files(data_folders, subfolders)

        # 2) Prepare metric layers as raw per-video dictionaries (video_key -> metric_value)
        cellphone_metric: Dict[str, float] = {}
        traffic_signs_metric: Dict[str, float] = {}
        vehicles_metric: Dict[str, float] = {}
        bicycles_metric: Dict[str, float] = {}
        cars_metric: Dict[str, float] = {}
        motorcycles_metric: Dict[str, float] = {}
        buses_metric: Dict[str, float] = {}
        trucks_metric: Dict[str, float] = {}
        persons_metric: Dict[str, float] = {}

        min_conf = float(common.get_configs("min_confidence"))
        reused_files = 0
        fallback_reads = 0

        # 3) Iterate mapping rows (Polars)
        mapping_iter = df_mapping.select(["videos", "start_time", "time_of_day"]).iter_rows(named=True)

        for row in tqdm(mapping_iter, total=df_mapping.height, desc="Analysing the csv files:"):
            videos_cell = row.get("videos")
            start_time_cell = row.get("start_time")
            time_of_day_cell = row.get("time_of_day")

            video_ids = cls._parse_videos_cell(videos_cell)

            try:
                start_times = ast.literal_eval(start_time_cell) if isinstance(start_time_cell, str) else None
                time_of_day = ast.literal_eval(time_of_day_cell) if isinstance(time_of_day_cell, str) else None
            except Exception:
                continue

            if not (isinstance(start_times, list) and isinstance(time_of_day, list)):
                continue

            for vid, start_times_list, time_of_day_list in zip(video_ids, start_times, time_of_day):
                if not isinstance(start_times_list, list) or not isinstance(time_of_day_list, list):
                    continue

                for start_time, _tod in zip(start_times_list, time_of_day_list):
                    prefix = "%s_%s_" % (vid, start_time)

                    matches = [
                        fname
                        for fname in csv_files.keys()
                        if fname.startswith(prefix) and fname.endswith(".csv")
                    ]
                    if not matches:
                        _LOGGER.warning("[WARNING] File not found for prefix: %s", prefix)
                        continue

                    if len(matches) > 1:
                        _LOGGER.warning(
                            "[WARNING] Multiple files found for prefix: %s, using first: %s",
                            prefix,
                            matches[0],
                        )

                    fps = cls._extract_fps_from_filename(str(vid), str(start_time), matches[0])
                    if fps is None:
                        _LOGGER.error("[ERROR] Could not extract fps from filename: %s", matches[0])
                        continue

                    filename = "%s_%s_%s.csv" % (vid, start_time, fps)
                    file_path = csv_files.get(filename)
                    if not file_path:
                        continue

                    # 4) Use mapping metadata to compute segment duration
                    key_for_meta = "%s_%s_%s" % (vid, start_time, fps)
                    meta = _METADATA.find_values_with_video_id(df_mapping, key_for_meta)
                    if meta is None:
                        continue

                    start_sec = meta[1]
                    end_sec = meta[2]
                    fps_from_meta = meta[17]

                    try:
                        fps_final = int(fps_from_meta)
                    except Exception:
                        fps_final = fps

                    try:
                        duration = float(end_sec) - float(start_sec)  # type: ignore
                    except Exception:
                        continue

                    if duration <= 0:
                        continue

                    video_key = "%s_%s_%s" % (vid, start_time, fps_final)

                    # 5) Reuse aggregate counts captured during analysis.py's first CSV pass.
                    counts = cls._file_metrics_cache.get(filename)
                    if counts is not None:
                        reused_files += 1
                    else:
                        # Preserve previous behaviour for files that were skipped or otherwise not
                        # observed in the main pass.
                        try:
                            df = cls._original_read_csv(file_path)
                        except Exception as exc:
                            _LOGGER.warning("[WARNING] Failed reading %s: %s", file_path, exc)
                            continue

                        required_cols = {"confidence", "yolo-id", "unique-id"}
                        if not required_cols.issubset(set(df.columns)):
                            continue

                        df = df.filter(
                            pl.col("confidence").cast(pl.Float64, strict=False) >= min_conf
                        )
                        counts = cls._metric_counts_from_dataframe(df)
                        cls._file_metrics_cache[filename] = counts
                        fallback_reads += 1

                    persons = counts["persons"]
                    cellphones = counts["cellphones"]
                    traffic_signs = counts["traffic_signs"]
                    vehicles = counts["vehicles"]
                    bicycles = counts["bicycles"]
                    cars = counts["cars"]
                    motorcycles = counts["motorcycles"]
                    buses = counts["buses"]
                    trucks = counts["trucks"]

                    # 6) Convert to per-minute rates (unique objects per minute)
                    per_min = 60.0 / duration

                    traffic_signs_metric[video_key] = float(traffic_signs) * per_min
                    vehicles_metric[video_key] = float(vehicles) * per_min
                    bicycles_metric[video_key] = float(bicycles) * per_min
                    cars_metric[video_key] = float(cars) * per_min
                    motorcycles_metric[video_key] = float(motorcycles) * per_min
                    buses_metric[video_key] = float(buses) * per_min
                    trucks_metric[video_key] = float(trucks) * per_min
                    persons_metric[video_key] = float(persons) * per_min

                    # 7) Cellphones metric: per-person normalised measure.
                    if persons > 0 and cellphones > 0:
                        cellphone_metric[video_key] = (
                            (float(cellphones) * 60.0) / duration / float(persons)
                        ) * 1000.0

        _LOGGER.info(
            "MetricsCache reused first-pass aggregates for %s CSV files; fallback disk reads: %s.",
            reused_files,
            fallback_reads,
        )

        # 8) Wrap metrics with grouping layer and store into the class cache
        metric_layers: List[Tuple[str, Dict[str, float]]] = [
            ("cellphones", cellphone_metric),
            ("traffic_signs", traffic_signs_metric),
            ("vehicles", vehicles_metric),
            ("bicycles", bicycles_metric),
            ("cars", cars_metric),
            ("motorcycles", motorcycles_metric),
            ("buses", buses_metric),
            ("trucks", trucks_metric),
            ("persons", persons_metric),
        ]

        cls._all_metrics_cache = {}

        for idx, (name, layer) in enumerate(metric_layers, start=1):
            _LOGGER.info("[%s/%s] Wrapping '%s' ...", idx, len(metric_layers), name)
            wrapped = _GROUPING.locality_country_wrapper(
                input_dict=layer,
                mapping=df_mapping,
                show_progress=True,
            )
            cls._all_metrics_cache[name] = wrapped

    @classmethod
    def _ensure_cache(cls, df_mapping: pl.DataFrame) -> None:
        """Compute metrics if cache is empty."""
        if not cls._all_metrics_cache:
            cls._compute_all_metrics(df_mapping)

    @classmethod
    def clear_cache(cls) -> None:
        """Clear wrapped metrics and captured per-file aggregates."""
        cls._all_metrics_cache.clear()
        cls._file_metrics_cache.clear()

    # --------------------------------------------------------------------------
    # Public API: metric getters
    # --------------------------------------------------------------------------
    @classmethod
    def calculate_cellphones(cls, df_mapping: pl.DataFrame) -> Any:
        """Return wrapped cellphone metric (computes cache on first call)."""
        cls._ensure_cache(df_mapping)
        return cls._all_metrics_cache["cellphones"]

    @classmethod
    def calculate_traffic_signs(cls, df_mapping: pl.DataFrame) -> Any:
        """Return wrapped traffic-sign metric (computes cache on first call)."""
        cls._ensure_cache(df_mapping)
        return cls._all_metrics_cache["traffic_signs"]

    @classmethod
    def calculate_traffic(
        cls,
        df_mapping: pl.DataFrame,
        person: bool = False,
        bicycle: bool = False,
        motorcycle: bool = False,
        car: bool = False,
        bus: bool = False,
        truck: bool = False,
    ) -> Any:
        """
        Return a requested traffic-related metric from the cache.

        Selection logic (kept consistent with your original intent):
        - If person=True: return persons
        - Else if bicycle=True: return bicycles
        - Else if motorcycle=car=bus=truck=True: return vehicles (aggregate)
        - Else return the first specific vehicle type requested
        - Else fallback: vehicles
        """
        cls._ensure_cache(df_mapping)

        if person:
            return cls._all_metrics_cache["persons"]
        if bicycle:
            return cls._all_metrics_cache["bicycles"]

        if motorcycle and car and bus and truck:
            return cls._all_metrics_cache["vehicles"]
        if car:
            return cls._all_metrics_cache["cars"]
        if motorcycle:
            return cls._all_metrics_cache["motorcycles"]
        if bus:
            return cls._all_metrics_cache["buses"]
        if truck:
            return cls._all_metrics_cache["trucks"]

        return cls._all_metrics_cache["vehicles"]

    # --------------------------------------------------------------------------
    # Utility: unique value extraction + duplicate reporting
    # --------------------------------------------------------------------------
    def get_unique_values(
        self,
        df: pl.DataFrame,
        value: Union[str, Sequence[str]],
        null_placeholder: str = "__NULL__",
        return_duplicates: bool = False,
    ) -> Tuple[UniqueValues, int, Optional[pl.DataFrame]]:
        """
        Returns (unique_values, count, dup_report).

        unique_values:
          - set[str] for single-column keys
          - set[tuple[str, ...]] for composite keys
        """
        cols = list(value) if isinstance(value, (list, tuple)) else [value]

        key_exprs = [
            pl.col(c).cast(pl.Utf8).fill_null(null_placeholder).alias(c)
            for c in cols
        ]
        keys_only = df.select(key_exprs)

        unique_values: UniqueValues = set()

        if len(cols) == 1:
            series = keys_only.get_column(cols[0])
            for value_item in series.unique().to_list():
                unique_values.add(str(value_item))
        else:
            for row in keys_only.unique().rows():
                unique_values.add(tuple(str(value_item) for value_item in row))

        dup_report: Optional[pl.DataFrame] = None
        if return_duplicates:
            keyed = df.with_row_index("row_index").with_columns(key_exprs)
            dup_report = (
                keyed.group_by(cols)
                .agg(
                    pl.len().alias("dup_count"),
                    pl.col("row_index").alias("row_indices"),
                )
                .filter(pl.col("dup_count") > 1)
                .sort("dup_count", descending=True)
            )

        return unique_values, len(unique_values), dup_report
