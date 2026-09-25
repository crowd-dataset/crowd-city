import ast
import math
import os
import random
from collections import OrderedDict
from typing import Any, ClassVar

import polars as pl

import common
from custom_logger import CustomLogger

logger = CustomLogger(__name__)  # use custom logger


class MetaData:
    """Metadata lookup helper with a shared per DataFrame video index."""

    # Shared by every MetaData instance used across IO, detection, metrics,
    # grouping, and analysis. Each entry keeps a strong DataFrame reference so
    # Python cannot reuse the object id while the cache entry is alive.
    _video_index_cache: ClassVar[OrderedDict] = OrderedDict()
    _max_cached_dataframes: ClassVar[int] = 8
    # (video, start) of every detection file in the Parquet store; see
    # _available_detection_segments.
    _available_segments_cache: ClassVar[set | None] = None

    def __init__(self) -> None:
        pass

    @staticmethod
    def _parse_videos_cell(v: str | None) -> list[str]:
        """Robustly parse a mapping ``videos`` cell into video IDs."""
        if not isinstance(v, str):
            return []

        s = v.strip()
        if (s.startswith('"') and s.endswith('"')) or (
            s.startswith("'") and s.endswith("'")
        ):
            s = s[1:-1].strip()

        if s.startswith("[") and s.endswith("]"):
            s = s[1:-1]

        parts = []
        for tok in s.split(","):
            t = tok.strip().strip('"').strip("'").strip()
            if t:
                parts.append(t)
        return parts

    @staticmethod
    def _safe_literal_eval(v: str | None):
        if not isinstance(v, str) or not v.strip():
            return None
        try:
            return ast.literal_eval(v)
        except Exception:
            return None

    @staticmethod
    def _state_or_unknown(state_val) -> str:
        if state_val is None:
            return "unknown"
        s = str(state_val).strip()
        if not s or s.lower() == "nan" or s == "NA":
            return "unknown"
        return s

    @staticmethod
    def _eq_expr(colname: str, value) -> pl.Expr:
        """Type aware equality expression to reduce mismatches."""
        if value is None:
            return pl.col(colname).is_null()
        if isinstance(value, float):
            if math.isnan(value):
                return pl.col(colname).is_null() | pl.col(colname).is_nan()
            return pl.col(colname).cast(pl.Float64, strict=False) == pl.lit(float(value))
        if isinstance(value, int):
            return pl.col(colname).cast(pl.Int64, strict=False) == pl.lit(int(value))
        return pl.col(colname).cast(pl.Utf8, strict=False) == pl.lit(str(value))

    @staticmethod
    def _normalise_max_footage_seconds(value: object) -> float | None:
        """Convert the optional per-city footage cap from hours to seconds."""
        if value is None:
            return None
        if isinstance(value, str) and not value.strip():
            return None
        if isinstance(value, bool):
            raise ValueError(
                "max_footage_hours_per_city must be a positive number or null"
            )

        try:
            hours = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "max_footage_hours_per_city must be a positive number or null"
            ) from error

        if not math.isfinite(hours) or hours < 0:
            raise ValueError(
                "max_footage_hours_per_city must be a positive finite number or null"
            )

        # null is the canonical no-cap value. Keep zero working for older local
        # configuration files that used the previous convention.
        if hours == 0:
            return None

        return hours * 3600.0

    @staticmethod
    def _city_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
        """Return the same city identity used by n_cities selection."""
        return tuple(
            "" if row.get(column) is None else str(row.get(column))
            for column in ("locality", "state", "iso3", "country")
        )

    @classmethod
    def clear_video_index_cache(cls) -> None:
        """Clear all cached mapping indexes."""
        cls._video_index_cache.clear()
        cls._available_segments_cache = None

    @staticmethod
    def _normalise_sampling_seed(value: object) -> int:
        """Return the seed that fixes the random per-city segment order."""
        if value is None or (isinstance(value, str) and not value.strip()):
            return 42
        if isinstance(value, bool):
            raise ValueError("footage_sampling_seed must be an integer or null")
        try:
            numeric = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError("footage_sampling_seed must be an integer or null") from error
        if not math.isfinite(numeric) or not numeric.is_integer():
            raise ValueError("footage_sampling_seed must be an integer or null")
        return int(numeric)

    @classmethod
    def _available_detection_segments(cls) -> set[tuple[str, int]]:
        """Return ``(video, start)`` for every detection file the analysis can read.

        The analysis only ever reads the Parquet store, so a segment whose file
        is missing there contributes nothing and must not consume a city's
        footage budget. The listing is cached per process: every worker and the
        parent must reach the same selection, and the store does not change
        during a run.
        """
        if cls._available_segments_cache is not None:
            return cls._available_segments_cache

        from utils.analytics.parquet_store import DETECTION_FOLDER, configured_parquet_roots

        available: set[tuple[str, int]] = set()
        for root in configured_parquet_roots():
            folder = os.path.join(root, DETECTION_FOLDER)
            try:
                names = os.listdir(folder)
            except OSError as error:
                logger.warning(f"Could not list detection files in {folder}: {error}")
                continue
            for name in names:
                stem, extension = os.path.splitext(name)
                if name.startswith(".") or extension.lower() != ".parquet":
                    continue
                try:
                    video, start_text, _fps = stem.rsplit("_", 2)
                    available.add((video, int(start_text)))
                except ValueError:
                    continue

        cls._available_segments_cache = available
        return available

    @classmethod
    def _build_indexes(
        cls,
        df: pl.DataFrame,
    ) -> tuple[
        dict[tuple[str, int], tuple[Any, ...]],
        dict[tuple[str, int], tuple[Any, float]],
    ]:
        """Parse the mapping once and build all hot-path segment indexes.

        ``analysis.py`` applies ``n_cities`` before this method is called.
        Therefore city ranking is always based on each city's full available
        footage. ``max_footage_hours_per_city`` is applied only afterwards to
        the already selected cities.

        Without a cap every segment is indexed in mapping order. With a cap,
        each city's segments are drawn in a random order fixed by
        ``footage_sampling_seed``, so the budget is not always spent on the
        first videos listed. Segments that cannot be analysed (no detection
        file in the Parquet store, or a vehicle type outside
        ``vehicles_analyse``) are skipped rather than counted, so the next
        segment is drawn in their place. If the final segment would cross the
        cap, its effective duration is shortened to the remaining budget. The
        worker then trims detection frames to the same duration.
        """
        metadata_index: dict[tuple[str, int], tuple[Any, ...]] = {}
        segment_index: dict[tuple[str, int], tuple[Any, float]] = {}

        max_footage_seconds = cls._normalise_max_footage_seconds(
            common.get_configs("max_footage_hours_per_city")
        )
        available_seconds_by_city: dict[
            tuple[str, str, str, str],
            float,
        ] = {}
        selected_seconds_by_city: dict[
            tuple[str, str, str, str],
            float,
        ] = {}
        # Every segment in mapping order, grouped by city:
        # (segment_key, start_value, end_value, duration, row id, metadata).
        segments_by_city: dict[tuple[str, str, str, str], list[tuple[Any, ...]]] = {}
        seen_keys: set[tuple[str, int]] = set()

        for row in df.iter_rows(named=True):
            video_ids = cls._parse_videos_cell(row.get("videos"))
            start_times = cls._safe_literal_eval(row.get("start_time"))
            end_times = cls._safe_literal_eval(row.get("end_time"))
            time_of_day = cls._safe_literal_eval(row.get("time_of_day"))
            vehicle_type = cls._safe_literal_eval(row.get("vehicle_type"))

            if not (
                isinstance(start_times, list)
                and isinstance(end_times, list)
                and isinstance(time_of_day, list)
                and isinstance(vehicle_type, list)
            ):
                continue

            locality = row.get("locality")
            state = cls._state_or_unknown(row.get("state"))
            latitude = row.get("lat")
            longitude = row.get("lon")
            country = row.get("country")
            gdp = row.get("gmp")
            population = row.get("population_locality")
            population_country = row.get("population_country")
            traffic_mortality = row.get("traffic_mortality")
            continent = row.get("continent")
            literacy_rate = row.get("literacy_rate")
            avg_height = row.get("avg_height")
            iso3 = row.get("iso3")
            city_key = cls._city_key(row)

            try:
                pop_i = int(population) if population is not None else 0
            except Exception:
                pop_i = 0

            try:
                gdp_i = int(gdp) if gdp is not None else 0
            except Exception:
                gdp_i = 0

            gpd_capita = (gdp_i / pop_i) if pop_i > 0 else 0

            for video, start_list, end_list, tod_list, vtype_list in zip(
                video_ids,
                start_times,
                end_times,
                time_of_day,
                vehicle_type,
            ):
                if not (
                    isinstance(start_list, list)
                    and isinstance(end_list, list)
                    and isinstance(tod_list, list)
                ):
                    continue

                for idx, start_value in enumerate(start_list):
                    try:
                        start_key = int(start_value)
                    except Exception:
                        continue

                    end_val = end_list[idx] if idx < len(end_list) else None
                    tod_val = tod_list[idx] if idx < len(tod_list) else None

                    try:
                        full_duration_seconds = float(end_val) - float(start_value)
                    except (TypeError, ValueError):
                        full_duration_seconds = 0.0

                    if (
                        not math.isfinite(full_duration_seconds)
                        or full_duration_seconds < 0
                    ):
                        full_duration_seconds = 0.0

                    available_seconds_by_city[city_key] = (
                        available_seconds_by_city.get(city_key, 0.0)
                        + full_duration_seconds
                    )

                    segment_key = (str(video), start_key)
                    # The old row scan returned the first match, so the first
                    # occurrence of a duplicated key is the one kept.
                    if segment_key in seen_keys:
                        continue
                    seen_keys.add(segment_key)

                    metadata_without_end = (
                        video,                 # 0
                        start_value,           # 1
                        None,                  # 2, effective end, set below
                        tod_val,               # 3
                        locality,              # 4
                        state,                 # 5
                        latitude,              # 6
                        longitude,             # 7
                        country,               # 8
                        gpd_capita,            # 9
                        population,            # 10
                        population_country,    # 11
                        traffic_mortality,     # 12
                        continent,             # 13
                        literacy_rate,         # 14
                        avg_height,            # 15
                        iso3,                  # 16
                        vtype_list,            # 17, returned at position 18
                    )
                    segments_by_city.setdefault(city_key, []).append(
                        (
                            segment_key,
                            start_value,
                            end_val,
                            full_duration_seconds,
                            row.get("id"),
                            metadata_without_end,
                        )
                    )

        skipped_missing = 0
        skipped_vehicle = 0
        if max_footage_seconds is not None:
            seed = cls._normalise_sampling_seed(common.get_configs("footage_sampling_seed"))
            available_files = cls._available_detection_segments()
            vehicle_list = common.get_configs("vehicles_analyse")

        for city_key, segments in segments_by_city.items():
            if max_footage_seconds is not None:
                # Seeded per city, so one city's draw never depends on which
                # other cities happen to be in the mapping.
                segments = list(segments)
                random.Random(f"{seed}|{'|'.join(city_key)}").shuffle(segments)

            for segment_key, start_value, end_val, full_duration, row_id, metadata in segments:
                duration_seconds = full_duration
                effective_end_val = end_val

                if max_footage_seconds is not None:
                    remaining_seconds = (
                        max_footage_seconds - selected_seconds_by_city.get(city_key, 0.0)
                    )
                    if remaining_seconds <= 0:
                        break
                    if segment_key not in available_files:
                        skipped_missing += 1
                        continue
                    if vehicle_list and metadata[17] not in vehicle_list:
                        skipped_vehicle += 1
                        continue
                    if duration_seconds > remaining_seconds:
                        duration_seconds = remaining_seconds
                        clipped_end = float(start_value) + duration_seconds
                        effective_end_val = (
                            int(clipped_end)
                            if clipped_end.is_integer()
                            else clipped_end
                        )

                selected_seconds_by_city[city_key] = (
                    selected_seconds_by_city.get(city_key, 0.0)
                    + duration_seconds
                )
                metadata_index[segment_key] = (
                    metadata[:2] + (effective_end_val,) + metadata[3:]
                )
                segment_index[segment_key] = (row_id, float(duration_seconds))

        if max_footage_seconds is not None:
            capped_city_count = sum(
                1
                for available_seconds in available_seconds_by_city.values()
                if available_seconds > max_footage_seconds + 1e-9
            )
            short_city_count = sum(
                1
                for city_key in segments_by_city
                if selected_seconds_by_city.get(city_key, 0.0) < max_footage_seconds - 1e-9
            )
            selected_total_seconds = sum(selected_seconds_by_city.values())

            logger.info(
                "Applied max_footage_hours_per_city={:.2f} with random segment "
                "order (footage_sampling_seed={}): {} of {} indexed cities had "
                "more footage than the cap; {:.2f} total footage hours were "
                "selected. Skipped {} segment(s) with no detection file and {} "
                "outside vehicles_analyse; {} cities have less analysable "
                "footage than the cap.",
                max_footage_seconds / 3600.0,
                seed,
                capped_city_count,
                len(available_seconds_by_city),
                selected_total_seconds / 3600.0,
                skipped_missing,
                skipped_vehicle,
                short_city_count,
            )

        return metadata_index, segment_index

    @classmethod
    def _indexes(
        cls,
        df: pl.DataFrame,
    ) -> tuple[
        dict[tuple[str, int], tuple[Any, ...]],
        dict[tuple[str, int], tuple[Any, float]],
    ]:
        """Return cached metadata and segment indexes for this DataFrame."""
        cache_key = id(df)
        cached = cls._video_index_cache.get(cache_key)

        if cached is not None:
            cached_df, metadata_index, segment_index = cached
            if cached_df is df:
                cls._video_index_cache.move_to_end(cache_key)
                return metadata_index, segment_index
            del cls._video_index_cache[cache_key]

        metadata_index, segment_index = cls._build_indexes(df)
        cls._video_index_cache[cache_key] = (
            df,
            metadata_index,
            segment_index,
        )
        cls._video_index_cache.move_to_end(cache_key)

        while len(cls._video_index_cache) > cls._max_cached_dataframes:
            cls._video_index_cache.popitem(last=False)

        logger.debug(
            f"Built metadata indexes for {df.height} mapping rows "
            f"with {len(metadata_index)} video segments."
        )
        return metadata_index, segment_index

    @classmethod
    def _video_index(
        cls,
        df: pl.DataFrame,
    ) -> dict[tuple[str, int], tuple[Any, ...]]:
        """Return the cached metadata index for this exact DataFrame object."""
        return cls._indexes(df)[0]

    @classmethod
    def segment_lookup(
        cls,
        df: pl.DataFrame,
    ) -> dict[tuple[str, int], tuple[Any, float]]:
        """Return ``(video, start) -> (locality id, duration seconds)``."""
        return cls._indexes(df)[1]

    def find_values_with_video_id(self, df: pl.DataFrame, key: str):
        """Return metadata for ``video_id_start_time_fps`` in constant time after indexing."""
        try:
            vid, start_str, fps_str = str(key).rsplit("_", 2)
            start_target = int(start_str)
            fps = int(fps_str)
        except (TypeError, ValueError):
            return None

        result = self._video_index(df).get((vid, start_target))
        if result is None:
            return None

        # Preserve the legacy 19 element tuple contract exactly.
        return result[:17] + (fps, result[17])

    def get_value(
        self,
        df: pl.DataFrame,
        column_name1: str,
        column_value1,
        column_name2: str | None,
        column_value2,
        target_column: str,
    ):
        """Retrieve a target value based on one or two column conditions."""
        if column_name2 is None or column_value2 is None:
            out = (
                df.filter(self._eq_expr(column_name1, column_value1))
                .select(target_column)
                .head(1)
            )
            return out.item(0, target_column) if out.height > 0 else None

        if isinstance(column_value2, str) and column_value2 == "unknown":
            column_value2 = None

        if isinstance(column_value2, float) and math.isnan(column_value2):
            column_value2 = None

        filt = self._eq_expr(column_name1, column_value1) & self._eq_expr(
            column_name2,
            column_value2,
        )

        out = df.filter(filt).select(target_column).head(1)
        return out.item(0, target_column) if out.height > 0 else None
