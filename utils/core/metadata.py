import ast
import math
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

        Segments are consumed in their existing mapping order. If the final
        segment would cross the configured cap, its effective duration is
        shortened to the remaining budget. The worker then trims detection
        frames to the same duration.
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

                    duration_seconds = full_duration_seconds
                    effective_end_val = end_val

                    if max_footage_seconds is not None:
                        already_selected = selected_seconds_by_city.get(
                            city_key,
                            0.0,
                        )
                        remaining_seconds = (
                            max_footage_seconds - already_selected
                        )

                        if remaining_seconds <= 0:
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

                    metadata_without_fps = (
                        video,                 # 0
                        start_value,           # 1
                        effective_end_val,     # 2
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

                    segment_key = (str(video), start_key)

                    # The old row scan returned the first match. setdefault
                    # preserves exactly that behaviour for duplicate keys.
                    metadata_index.setdefault(
                        segment_key,
                        metadata_without_fps,
                    )

                    segment_index.setdefault(
                        segment_key,
                        (row.get("id"), float(duration_seconds)),
                    )

        if max_footage_seconds is not None:
            capped_city_count = sum(
                1
                for available_seconds in available_seconds_by_city.values()
                if available_seconds > max_footage_seconds + 1e-9
            )
            selected_total_seconds = sum(selected_seconds_by_city.values())

            logger.info(
                "Applied max_footage_hours_per_city={:.2f}: "
                "{} of {} indexed cities had more footage than the cap; "
                "{:.2f} total footage hours were selected.",
                max_footage_seconds / 3600.0,
                capped_city_count,
                len(available_seconds_by_city),
                selected_total_seconds / 3600.0,
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
